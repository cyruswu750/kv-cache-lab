"""CPU checks of the strengthened M4-B GPU gate (tests/gpu/test_m4_replay.py). No weights, no CUDA.

The gate's helpers are device-agnostic, so this module imports the GPU test module and runs the
gate's own per-case driver (`run_case`) on a tiny random bf16 Qwen2 at C = 4. It shows:
- the new checks pass on correct code (positive controls);
- each check detects a targeted mismatch: a counter off by one, restored bytes that differ from
  the entry, chunks out of order, an alias of an entry, a wrong length or identity, an
  incomplete diagnostics file;
- the gate's real scenarios (C = 256, key-level models only) make the reconciled counters
  nonzero, so the reconciliation is not only zero-equals-zero;
- the diagnostics are written outside pytest's temporary directories and are required.

Correctness of the checks only; nothing here is GPU or performance evidence.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import trace_models as tm
from kvlab import traces as tr
from kvlab.boundaries import max_restore_len
from kvlab.gpu_prefix_cache import GPUPrefixCache
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.prefix_aware_cache import PrefixAwareGPUPrefixCache
from kvlab.tiered_cache import TieredKVCache
from kvlab.trace_replay import GPUOnly

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("m4_gate_replay_under_test", ROOT / "tests/gpu/test_m4_replay.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
sys.path.insert(0, str(ROOT / "scripts"))
import m4_gate as gate  # noqa: E402

C, STEPS = 4, 2


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized(), "these CPU tests must not initialise CUDA"


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=10000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)


def probe_budgets(trace, disk_files=None):
    w = tr.trace_stats(trace, C)["unique_chunks"]
    return {"C1PA_W": 4, "C1PA_HALF": 3, "host_slots": 2, "disk_files": disk_files or w + 2, "W": w, "budget_W": w}


def run(model, ns, tmp_path, tag, disk_files=None):
    trace = G.probe_trace(C)
    (tmp_path / tag).mkdir()
    path = tmp_path / f"{tag}.json"
    doc = G.run_case(model, ns, tag, trace, probe_budgets(trace, disk_files), tmp_path / tag, steps=STEPS,
                     mem=G.NoMemory(), diag_path=path)
    return trace, doc, path


def final(doc, config):
    return [r for r in doc["records"] if r["config"] == config][-1]["counters"]["observed"]


# ---- the gate's scenarios and identity
def test_gate_probe_is_the_attempt_2_probe():
    """The probe was refactored into probe_trace(C); the gate's probe (C = 256) is unchanged."""
    C256 = G.SPEC["chunk_size"]
    A = tuple(range(1000, 1000 + 5 * C256 + 1))
    S = tuple(range(3000, 3000 + 2 * C256))
    X, Y, Z = S + tuple(range(5000, 5000 + 2 * C256 + 1)), S + tuple(range(7000, 7000 + 3 * C256 + 1)), S + (9, 9, 9)
    old = tr.Trace(tr._header("probe", 0, {}), tuple(tr.TraceRequest(f"p{i}", "g", p)
                                                     for i, p in enumerate((A, A, A, X, Z, Y, Z, Z))))
    assert G.gate_traces()["probe"].digest == old.digest
    assert G.OUT_ENV == gate.GATE_OUT_ENV


def test_gate_scenarios_make_the_reconciled_counters_nonzero():
    """Key-level models over the gate's own cases (C = 256): the counters the gate now reconciles
    are nonzero in some case, for every configuration where they can occur."""
    traces, B, C256 = G.gate_traces(), G.budgets_for(G.gate_traces()), G.SPEC["chunk_size"]
    tot = {}
    for case in G.CASES:
        b, P = B[case], [list(r.tokens) for r in traces[case].requests]
        recs = {"C1LRU_HALF": tm.lru_model(P, C256, b["C1PA_HALF"]), "C1PA_W": tm.pa_model(P, C256, b["C1PA_W"]),
                "C1PA_HALF": tm.pa_model(P, C256, b["C1PA_HALF"])}
        for name, rs in recs.items():
            e = G.c1_expected(P, C256, rs, len(P) - 1, "after")
            for k, v in e.items():
                tot.setdefault(name, {}).setdefault(k, 0)
                tot[name][k] += v
        e = G.c2_expected(P, C256, tm.tiered_model(P, C256, b["host_slots"], b["disk_files"]), len(P) - 1, 1, 1)
        s = G.c2_store_expected(tm.tiered_model(P, C256, b["host_slots"], b["disk_files"]), len(P) - 1, 1, 1)
        for k in ("host_hits", "disk_hits", "host_spills_to_disk", "chunk_misses", "restored_requests"):
            tot.setdefault("C2", {}).setdefault(k, 0)
            tot["C2"][k] += e[k]
        tot["C2"].setdefault("disk_evictions", 0)
        tot["C2"]["disk_evictions"] += s["disk_evictions"]
    for name in ("C1LRU_HALF", "C1PA_W", "C1PA_HALF"):
        for k in ("lookups", "chunk_hits", "chunk_misses", "restored_requests", "chunks_stored", "evictions"):
            assert tot[name].get(k, 0) > 0, (name, k, tot[name])
    for name in ("C1PA_W", "C1PA_HALF"):
        assert tot[name].get("admission_stops_prefix_capacity", 0) > 0 and \
            tot[name].get("chunks_refused_prefix_capacity", 0) > 0, (name, tot[name])
    assert all(v > 0 for v in tot["C2"].values()), tot["C2"]


# ---- positive controls: the gate's own driver on CPU
def test_run_case_passes_and_writes_complete_diagnostics(model, ns, tmp_path):
    trace, doc, path = run(model, ns, tmp_path, "probe")
    on_disk = json.loads(path.read_text())
    assert on_disk["complete"] is True and on_disk["failure"] is None
    assert G.diagnostics_problems(on_disk, "probe", trace, steps=STEPS, require_allocator=False) == []
    # CPU has no allocator values: the gate's completeness check (require_allocator=True) refuses them
    assert any("allocator values" in p for p in G.diagnostics_problems(on_disk, "probe", trace, steps=STEPS,
                                                                        require_allocator=True))
    pa_w, pa_h, lru = final(on_disk, "C1PA_W"), final(on_disk, "C1PA_HALF"), final(on_disk, "C1LRU_HALF")
    assert pa_w["admission_stops_prefix_capacity"] == 4 and pa_w["chunks_refused_prefix_capacity"] == 4
    assert pa_h["admission_stops_prefix_capacity"] == 5 and pa_h["chunks_refused_prefix_capacity"] == 9
    assert pa_w["evictions"] == 6 and pa_h["evictions"] == 4 and lru["evictions"] == 23
    for cfg in ("C2_base", "C2_batch"):
        c2 = final(on_disk, cfg)
        assert c2["host_hits"] == 4 and c2["disk_hits"] == 14 and c2["host_spills_to_disk"] == 10
        assert c2["disk_file_bytes_read"] == 14 * cf.file_size(ns) and c2["bytes_h2d"] == 18 * cf.payload_len(ns)
    seg = [r["allocator"]["segments"][0]["segment"] for r in on_disk["records"]]
    assert seg == ["whole_prefill"] * 8 + ["restore"] * 24 + ["whole_prefill"] * 16


def test_run_case_with_a_tight_disk_reconciles_disk_evictions(model, ns, tmp_path):
    trace, doc, _ = run(model, ns, tmp_path, "tight", disk_files=3)
    assert final(doc, "C2_base")["disk_evictions"] > 0 and final(doc, "C2_batch")["disk_evictions"] > 0


def test_run_case_frees_its_c1_caches_without_cyclic_gc(model, ns, tmp_path, monkeypatch):
    """Review P3: the restore observers must not keep a case's C1 entries alive into the next case
    (which would inflate its allocator baseline) until a cyclic collection happens."""
    import gc
    import weakref
    seen, real = [], G.configs

    def configs(*a):
        out = real(*a)
        seen.extend(weakref.ref(c.cache) for n, c in out.items() if n.startswith("C1"))
        return out
    monkeypatch.setattr(G, "configs", configs)
    gc.collect()
    gc.disable()
    try:
        run(model, ns, tmp_path, "lifetime")
        assert len(seen) == 3 and all(r() is None for r in seen), [r() for r in seen]
    finally:
        gc.enable()


# ---- targeted counter mismatches are detected
def _fails(model, ns, tmp_path, tag, needle):
    with pytest.raises(AssertionError) as ei:
        run(model, ns, tmp_path, tag)
    doc = json.loads((tmp_path / f"{tag}.json").read_text())
    assert doc["complete"] is False and doc["failure"], doc["failure"]  # partial evidence is kept
    assert needle in str(ei.value), str(ei.value)[:800]
    return doc


def test_an_extra_refused_chunk_is_detected(model, ns, tmp_path, monkeypatch):
    real = PrefixAwareGPUPrefixCache.store

    def store(self, cache, token_ids, keys):
        stops = self.counters["admission_stops_prefix_capacity"]
        out = real(self, cache, token_ids, keys)
        if self.counters["admission_stops_prefix_capacity"] > stops:
            self.counters["chunks_refused_prefix_capacity"] += 1
        return out
    monkeypatch.setattr(PrefixAwareGPUPrefixCache, "store", store)
    _fails(model, ns, tmp_path, "refused", "C1PA_W")


def test_a_missing_admission_stop_is_detected(model, ns, tmp_path, monkeypatch):
    real = PrefixAwareGPUPrefixCache.store

    def store(self, cache, token_ids, keys):
        stops = self.counters["admission_stops_prefix_capacity"]
        out = real(self, cache, token_ids, keys)
        if self.counters["admission_stops_prefix_capacity"] > stops:
            self.counters["admission_stops_prefix_capacity"] -= 1
        return out
    monkeypatch.setattr(PrefixAwareGPUPrefixCache, "store", store)
    _fails(model, ns, tmp_path, "stops", "C1PA_W")


def test_an_extra_lru_eviction_is_detected(model, ns, tmp_path, monkeypatch):
    real = GPUPrefixCache.store  # the plain-LRU class; C1-PA overrides store()

    def store(self, cache, token_ids, keys):
        ev = self.counters["evictions"]
        out = real(self, cache, token_ids, keys)
        if self.counters["evictions"] > ev:
            self.counters["evictions"] += 1
        return out
    monkeypatch.setattr(GPUPrefixCache, "store", store)
    _fails(model, ns, tmp_path, "lru_ev", "C1LRU_HALF")


def test_an_extra_c2_disk_hit_is_detected(model, ns, tmp_path, monkeypatch):
    """The attempt-2 gate did not compare disk_hits; the strengthened gate must."""
    real = TieredKVCache.restore

    def restore(self, token_ids, keys, device, config):
        out = real(self, token_ids, keys, device, config)
        if "disk" in out[2]:
            self.counters.disk_hits += 1
        return out
    monkeypatch.setattr(TieredKVCache, "restore", restore)
    _fails(model, ns, tmp_path, "c2_disk", "C2_base")


def test_restored_bytes_differing_from_the_entry_are_detected_at_the_restore_boundary(model, ns, tmp_path,
                                                                                       monkeypatch):
    real = GPUPrefixCache.restore

    def restore(self, token_ids, keys, config):
        kv, L, sources = real(self, token_ids, keys, config)
        if kv is not None:
            kv.layers[1].values[0, 0, L - 1, 0] += 1  # the last restored chunk, after the D4 copy
        return kv, L, sources
    monkeypatch.setattr(GPUPrefixCache, "restore", restore)
    _fails(model, ns, tmp_path, "bytes", "restored V bytes != entry V bytes")


# ---- the direct restored-vs-entry check, unit level
@pytest.fixture
def restored(model, ns):
    """A C1-PA cache holding P's 3 complete chunks, and a fresh restore of them (L = 12)."""
    cache = PrefixAwareGPUPrefixCache(ns, 6 * cf.payload_len(ns), "cpu")
    P = list(range(100, 113))  # N = 13: 3 complete chunks, all eligible
    GPUOnly(model, cache).prefill(P)
    keys = chain_keys(ns.digest(), P, C)[:max_restore_len(len(P), C) // C]
    kv, L, sources = cache.restore(P, keys, model.config)
    assert L == 12 and G.restored_vs_entries(cache, P, keys, kv, L, sources) == []
    return cache, P, keys, kv, L, sources


def _layers(kv):
    return [(layer.keys.clone(), layer.values.clone()) for layer in kv.layers]


def test_direct_check_detects_one_flipped_byte(restored):
    cache, P, keys, kv, L, src = restored
    kv.layers[0].keys[0, 1, 5, 3] += 1  # chunk 1
    probs = G.restored_vs_entries(cache, P, keys, kv, L, src)
    assert probs == ["chunk 1 layer 0: restored K bytes != entry K bytes"], probs


def test_direct_check_detects_chunks_out_of_order(model, restored):
    cache, P, keys, kv, L, src = restored
    swap = [(torch.cat([k[:, :, 4:8], k[:, :, 0:4], k[:, :, 8:]], dim=-2),
             torch.cat([v[:, :, 4:8], v[:, :, 0:4], v[:, :, 8:]], dim=-2)) for k, v in _layers(kv)]
    probs = G.restored_vs_entries(cache, P, keys, DynamicCache(ddp_cache_data=swap, config=model.config), L, src)
    assert any(p.startswith("chunk 0 ") for p in probs) and any(p.startswith("chunk 1 ") for p in probs), probs


def test_direct_check_detects_an_alias_of_an_entry(model, restored):
    cache = restored[0]
    P1 = list(range(200, 205))  # N = 5: one eligible chunk, so one entry can stand for the whole restore
    GPUOnly(model, cache).prefill(P1)
    k1 = chain_keys(cache.ns_digest, P1, C)[:1]
    kv1, L1, src1 = cache.restore(P1, k1, model.config)
    assert L1 == C and G.restored_vs_entries(cache, P1, k1, kv1, L1, src1) == []
    _, t = cache.entries[k1[0]]
    for li, layer in enumerate(kv1.layers):  # the same bytes, but the entry's own storage
        layer.keys, layer.values = t[li, 0], t[li, 1]
    probs = G.restored_vs_entries(cache, P1, k1, kv1, L1, src1)
    assert probs == ["restored K/V share storage with a cached entry (an alias, not a copy)"], probs


def test_direct_check_detects_wrong_lengths_identity_order_and_stops(model, restored):
    cache, P, keys, kv, L, src = restored
    assert any("expected exactly" in p for p in G.restored_vs_entries(cache, P, keys, kv, L - C, src[:-1]))
    assert any("not a whole number" in p for p in G.restored_vs_entries(cache, P, keys, kv, L - 1, src))
    assert any("own eligible chain keys" in p for p in G.restored_vs_entries(cache, P, keys[::-1], kv, L, src))
    assert any("sources" in p for p in G.restored_vs_entries(cache, P, keys, kv, L, ["host"] * 3))
    ident, t = cache.entries[keys[2]]
    cache.entries[keys[2]] = (ChunkIdentity(ident.ns_digest, ident.key, ident.parent, 1, ident.tokens), t)
    assert G.restored_vs_entries(cache, P, keys, kv, L, src) == [
        "chunk 2: the entry's identity is not the request's chunk 2"]
    cache.entries[keys[2]] = (ident, t)
    cache.entries.move_to_end(keys[0])  # the hits are no longer the most recent, root first
    assert G.restored_vs_entries(cache, P, keys, kv, L, src) == [
        "the hit entries are not the most recently used, root first"]
    kv2, L2, src2 = cache.restore(P, keys[:2], model.config)  # a restore that stops before cached chunk 2
    assert "the restore stopped at chunk 2, which is cached" in G.restored_vs_entries(cache, P, keys, kv2, L2, src2)


# ---- diagnostics: validator, no overwrite, required under the gate
def test_the_diagnostics_validator_detects_incomplete_documents(model, ns, tmp_path):
    trace, doc, path = run(model, ns, tmp_path, "probe")
    good = json.loads(path.read_text())

    def probs(mut):
        d = json.loads(json.dumps(good))
        mut(d)
        return G.diagnostics_problems(d, "probe", trace, steps=STEPS, require_allocator=False)
    assert probs(lambda d: None) == []
    assert probs(lambda d: d["records"].pop(17))
    assert probs(lambda d: d.update(complete=False))
    assert probs(lambda d: d.update(failure="AssertionError()"))
    assert probs(lambda d: d["records"][20]["allocator"]["segments"].pop())
    assert probs(lambda d: d["records"][20]["counters"]["observed"].update(evictions=999))
    assert probs(lambda d: d["records"][40]["logical"].pop("disk_used_B"))
    assert probs(lambda d: d["records"][3].update(prompt_sha256=hashlib.sha256(b"x").hexdigest()))
    assert probs(lambda d: d["wall_time_s"].pop("C0"))
    assert probs(lambda d: d["trace"].update(digest="0" * 64))
    with pytest.raises(FileExistsError):  # never overwrites earlier evidence
        G.write_diagnostics(path, good)


def _child(tmp_path, env_value):
    env = {k: v for k, v in os.environ.items() if k != G.OUT_ENV}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    if env_value is not None:
        env[G.OUT_ENV] = env_value
    return subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                           f"--basetemp={tmp_path / 'bt'}", "-m", "gpu and m4", "tests/gpu/test_m4_replay.py",
                           "-k", "test_gate_diagnostics_are_complete"], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=300)


def test_the_gate_completeness_test_fails_without_the_output_directory(tmp_path):
    """A real pytest child of the GPU module, selecting only its final completeness test (no model
    is loaded): without the gate's variable it fails; with an empty output directory it fails
    because no case left diagnostics."""
    r = _child(tmp_path, None)
    assert r.returncode == 1 and f"{G.OUT_ENV} is not set" in r.stdout, r.stdout[-1500:]
    out = tmp_path / "out"
    out.mkdir()
    r = _child(tmp_path, str(out))
    assert r.returncode == 1 and "1 failed" in r.stdout and "T0.json" in r.stdout, r.stdout[-1500:]
    assert (out / G.DIAG_SUBDIR).is_dir() and not any((out / G.DIAG_SUBDIR).iterdir())
