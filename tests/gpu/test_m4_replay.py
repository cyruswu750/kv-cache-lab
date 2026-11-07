"""M4-B bounded GPU correctness gate (docs/M4B_PROTOCOL.md §9).

Run exclusively on the GPU, once per reviewed version, before any timed session, and only through
the bounded fail-fast wrapper (one 900 s deadline over the M1, M3
and M4 GPU suites):

  HF_HOME=<weights cache> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m4_gate.py --out-dir <new dir>

The wrapper puts that fresh directory into KVLAB_M4_GATE_OUT_DIR. The diagnostics are written
under it, as `m4_diagnostics/<case>.json`, outside pytest's temporary directories. Without the
variable the test fails: §9 requires the diagnostics to be retained.

**Inputs:**
- Calibration seeds only (kvlab.m4_spec role "calibration"): the first 12 requests of each trace.
- The CPU probe's oversized and shared-root sequences, scaled to C = 256.
- No timing is analysed.

**Checks.** Per configuration (C0, C1-LRU(W/2), C1-PA(W), C1-PA(W/2), C2-base, C2-batch), with
steps = 8:
- **Strict:** every step logit and token is bitwise equal to ResidentReference replaying the
  same configuration's own restore lengths and stored chunks (D12). There are exactly `steps` of
  each.
- **C1 restore boundary.** This is the instant restore() returns: before the suffix prefill, and
  before store(), whose admission may evict entries. At that instant:
  - the restored K/V [0, L) equal the cached entries' bytes for the request's own chain keys,
    chunk by chunk and in order;
  - the entries' identities, recomputed here from the tokens, match;
  - the lengths are exact;
  - no restored tensor shares storage with any entry;
  - the hits are the most recently used entries, root first;
  - the restore stopped at the first uncached eligible chunk.
- **Reference bytes:** the restored KV bytes [0, L) equal the reference's stored chunks on every
  hit (D12).
- **Models:** the decisions equal the independent key-level models (kvlab.trace_models), and so
  do all the counters:
  - C1 at the restore boundary and after admission;
  - C2 after every request: every Counters field, host and disk evictions, and occupancy in
    entries and bytes;
  - every fault counter is zero.
- **C1:** prefix closure, retained bytes = entries x payload <= G, and no storage overlap
  between the request KV and the entries.
- **C2:** the budgets.

**Where expected values come from.** Only from the models, the trace and the chunk format,
never from the counters under test.

**Diagnostics only** (§9, using the §7 conventions). Per request:
- the process-level CUDA allocator observations around prefill(): A0, P and P - A0. The
  allocations of the C1 boundary check itself are excluded;
- the logical storage terms.

Each value is labelled as logical storage or as a process-level allocator measurement. None is a
standalone per-configuration peak.

**Failure.** A failure stops the gate (FAIL), and no performance session may follow. The existing
GPU M1 and M3 suites are run separately as part of the gate (`-m "gpu and m1"`,
`-m "gpu and m3"`).

The helpers below are device-agnostic, and are also exercised on CPU with a tiny random model by
tests/test_m4_gate_checks_cpu.py.
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest
import torch

from kvlab import chunk_format as cf
from kvlab import m4_spec as ms
from kvlab import trace_models as tm
from kvlab import traces as tr
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.boundaries import max_restore_len
from kvlab.disk_store import DiskStore
from kvlab.gpu_prefix_cache import GPUPrefixCache
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.kv_snapshot import bytes_equal
from kvlab.model import greedy_continue
from kvlab.prefix_aware_cache import PrefixAwareGPUPrefixCache
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import TieredKVCache
from kvlab.trace_replay import GPUOnly, NoCache, Tiered, cache_storage_bytes

pytestmark = [pytest.mark.gpu, pytest.mark.m4]
SPEC, STEPS, N_REQ = ms.M4_EVAL_V1, 8, 12
CASES = (*ms.TRACE_NAMES, "probe")
CONFIGS = ("C0", "C1LRU_HALF", "C1PA_W", "C1PA_HALF", "C2_base", "C2_batch")
OUT_ENV = "KVLAB_M4_GATE_OUT_DIR"  # scripts/m4_gate.py GATE_OUT_ENV: the gate's fresh output directory
DIAG_SUBDIR, DIAG_SCHEMA = "m4_diagnostics", "m4_gate_diagnostics/1"
C1_SEGMENTS, OTHER_SEGMENTS = ["restore", "suffix_prefill_and_admission"], ["whole_prefill"]

FIELDS = {
    "units": "every *_B value is in bytes; every *_s value is in monotonic seconds",
    "allocator.*": ("process_allocator: torch.cuda caching-allocator statistics of the WHOLE gate process, "
                    "on the model's device. They include the model weights, the current configuration's "
                    "ResidentReference chunks, the retained entries of every C1 configuration constructed "
                    "for this case (earlier configurations' caches are still alive), the request KV and the "
                    "activations. They are NOT a standalone per-configuration peak. On CPU they are null."),
    "allocator.A0_B": ("memory_allocated() after torch.cuda.synchronize() and reset_peak_memory_stats(), "
                       "immediately before the configuration's prefill() call"),
    "allocator.segments": ("C1: 'restore' runs from A0 until restore() returns (lookup, gather, D4 constructor "
                           "copy). The gate's boundary check then runs, outside any segment. "
                           "'suffix_prefill_and_admission' runs from a fresh reset after that check until "
                           "prefill() returns. C0 and C2: one 'whole_prefill' segment (C2: restore, suffix "
                           "prefill and admission). Each segment records allocated_at_start_B, "
                           "max_memory_allocated_B (the peak within the segment) and allocated_at_end_B."),
    "allocator.P_B": "the max of the segments' max_memory_allocated_B; the C1 boundary check is excluded",
    "allocator.P_minus_A0_B": "P - A0; includes the persistent growth inside prefill() (C1 entries admitted)",
    "logical.*": "logical_storage: bytes held by the named structures, counted from their storages or counters",
    "logical.request_kv_storage_B": "the distinct storages behind the request's K/V after prefill()",
    "logical.restore_scratch_B": ("C1: the gather scratch (n_r x P). C2: the device destination plus the batched "
                                  "staging (TieredKVCache.last_restore). 0 on a miss"),
    "logical.device_persistent_before_B / _after_B": ("per configuration, before and after this prefill(): "
                                                      "C1 retained_bytes(); 0 for C0 and C2 (no persistent "
                                                      "device cache)"),
    "logical.host_resident_B / disk_used_B": ("C2 only: logical host occupancy inside the pinned slab; disk "
                                              "used bytes (committed + temp + undeletable)"),
    "counters": ("observed = the production counters; expected = derived from the key-level model, the trace and "
                 "the chunk format only. C1 has both 'at_restore' (the restore boundary) and 'after' (after "
                 "admission); zero counts are omitted, as in a Counter"),
    "wall_time_s": ("per configuration, over this case, INCLUDING the reference replays and every check: a "
                    "feasibility diagnostic only, never performance evidence"),
}


@pytest.fixture(scope="module")
def diag_dir():
    out = os.environ.get(OUT_ENV)
    if not out:
        pytest.fail(f"{OUT_ENV} is not set: run the gate through scripts/m4_gate.py (§9 retains diagnostics)")
    out = Path(out)
    if not out.is_absolute() or not out.is_dir():
        pytest.fail(f"{OUT_ENV}={out} is not an existing absolute directory")
    d = out / DIAG_SUBDIR
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="module")
def model():
    m, _ = km.load_model()
    return m


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, SPEC["chunk_size"])


def probe_trace(C):
    A = tuple(range(1000, 1000 + 5 * C + 1))
    S = tuple(range(3000, 3000 + 2 * C))
    X, Y, Z = S + tuple(range(5000, 5000 + 2 * C + 1)), S + tuple(range(7000, 7000 + 3 * C + 1)), S + (9, 9, 9)
    hand = (A, A, A, X, Z, Y, Z, Z)
    return tr.Trace(tr._header("probe", 0, {}), tuple(tr.TraceRequest(f"p{i}", "g", p) for i, p in enumerate(hand)))


def gate_traces():
    out = {}
    for t in ms.TRACE_NAMES:
        full = ms.trace(SPEC, t, "calibration")
        out[t] = tr.Trace(full.header, full.requests[:N_REQ])
    out["probe"] = probe_trace(SPEC["chunk_size"])
    return out


def budgets_for(traces):
    b = ms.budgets(SPEC, "calibration", {t: traces[t] for t in ms.TRACE_NAMES})
    w = tr.trace_stats(traces["probe"], SPEC["chunk_size"])["unique_chunks"]
    b["probe"] = {"C1PA_W": 4, "C1PA_HALF": 3, "host_slots": 2, "disk_files": w + 2, "W": w, "budget_W": w}
    return b


def configs(model, ns, b, tmp):
    pl, fs, dev = cf.payload_len(ns), cf.file_size(ns), next(model.parameters()).device
    out = {"C0": NoCache(model),
           "C1LRU_HALF": GPUOnly(model, GPUPrefixCache(ns, b["C1PA_HALF"] * pl, dev)),
           "C1PA_W": GPUOnly(model, PrefixAwareGPUPrefixCache(ns, b["C1PA_W"] * pl, dev)),
           "C1PA_HALF": GPUOnly(model, PrefixAwareGPUPrefixCache(ns, b["C1PA_HALF"] * pl, dev))}
    for name, backend in (("C2_base", "baseline"), ("C2_batch", "batched")):
        host = HostStore((b["host_slots"] + 1) * pl, pl, max_entries=b["host_slots"], pin=dev.type == "cuda")
        disk = DiskStore(tmp / name, b["disk_files"] * fs, max_entries=b["disk_files"])
        out[name] = Tiered(AlwaysRestoreEngine(model, TieredKVCache(ns, host, disk, restore_backend=backend,
                                                                    restore_staging_chunks=SPEC["staging_chunks"])))
    return out


def _ranges(ts):
    return [(t.untyped_storage().data_ptr(), t.untyped_storage().data_ptr() + t.untyped_storage().nbytes()) for t in ts]


def _overlap(xs, ys) -> bool:
    return any(a < d and c < b for a, b in xs for c, d in ys)


# ---- allocator observation (diagnostic only)
class CudaMemory:
    """Process-level CUDA caching-allocator observations around one prefill() call (see FIELDS)."""

    def begin(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.a0 = self._start = torch.cuda.memory_allocated()
        self.segments = []

    def cut(self, label):
        torch.cuda.synchronize()
        self.segments.append({"segment": label, "allocated_at_start_B": self._start,
                              "max_memory_allocated_B": torch.cuda.max_memory_allocated(),
                              "allocated_at_end_B": torch.cuda.memory_allocated()})

    def resume(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._start = torch.cuda.memory_allocated()

    def end(self, label) -> dict:
        self.cut(label)
        P = max(s["max_memory_allocated_B"] for s in self.segments)
        return {"A0_B": self.a0, "P_B": P, "P_minus_A0_B": P - self.a0, "segments": self.segments}


class NoMemory:
    """CPU: the same segment boundaries, no allocator values."""

    def begin(self):
        self.segments = []

    def cut(self, label):
        self.segments.append({"segment": label, "allocated_at_start_B": None, "max_memory_allocated_B": None,
                              "allocated_at_end_B": None})

    def resume(self):
        pass

    def end(self, label) -> dict:
        self.cut(label)
        return {"A0_B": None, "P_B": None, "P_minus_A0_B": None, "segments": self.segments}


# ---- direct C1 restore check at the restore boundary
def restored_vs_entries(cache, token_ids, keys, kv, L, sources) -> list:
    """Problems (empty = pass) of one C1 restore, compared DIRECTLY with the cached entries it must
    come from. The chain keys and identities are recomputed from the tokens here; nothing is taken
    from the restore's own bookkeeping."""
    token_ids, C, ns = list(token_ids), cache.C, cache.ns
    want = chain_keys(cache.ns_digest, token_ids, C)
    eligible = max_restore_len(len(token_ids), C) // C
    probs = []
    if list(keys) != want[:eligible]:
        probs.append("restore() was not given the request's own eligible chain keys")
    if L % C or not 0 <= L // C <= eligible:
        return probs + [f"restore length {L} is not a whole number of eligible chunks"]
    n = L // C
    if n < eligible and want[n] in cache.entries:
        probs.append(f"the restore stopped at chunk {n}, which is cached")
    if n == 0:
        if kv is not None or sources:
            probs.append("a miss returned a cache or sources")
        return probs
    if list(sources) != ["gpu"] * n:
        probs.append(f"sources {sources!r}, expected {n} x 'gpu'")
    if kv is None or len(kv.layers) != ns.num_layers:
        return probs + ["the restored cache is missing or has the wrong layer count"]
    shape = (1, ns.num_kv_heads, L, ns.head_dim)
    for li, layer in enumerate(kv.layers):
        for what, x in (("K", layer.keys), ("V", layer.values)):
            if tuple(x.shape) != shape or x.dtype != cache.dtype:
                probs.append(f"layer {li} {what}: {tuple(x.shape)} {x.dtype}, expected exactly {shape} {cache.dtype}")
    if probs:
        return probs
    for i in range(n):
        if want[i] not in cache.entries:
            probs.append(f"chunk {i} was restored but is not cached")
            continue
        ident, t = cache.entries[want[i]]
        if ident != ChunkIdentity.for_chunk(cache.ns_digest, token_ids, C, i, want):
            probs.append(f"chunk {i}: the entry's identity is not the request's chunk {i}")
        if tuple(t.shape) != cache.shape or t.dtype != cache.dtype:
            probs.append(f"chunk {i}: entry {tuple(t.shape)} {t.dtype}")
            continue
        s, e = i * C, (i + 1) * C
        for li, layer in enumerate(kv.layers):
            if not bytes_equal(layer.keys[:, :, s:e, :], t[li, 0]):
                probs.append(f"chunk {i} layer {li}: restored K bytes != entry K bytes")
            if not bytes_equal(layer.values[:, :, s:e, :], t[li, 1]):
                probs.append(f"chunk {i} layer {li}: restored V bytes != entry V bytes")
    if _overlap(_ranges(x for layer in kv.layers for x in (layer.keys, layer.values)),
                _ranges(t for _, t in cache.entries.values())):
        probs.append("restored K/V share storage with a cached entry (an alias, not a copy)")
    if list(cache.entries)[-n:] != want[:n]:
        probs.append("the hit entries are not the most recently used, root first")
    return probs


class RestoreBoundary:
    """Test-side observer of ONE C1 cache instance's restore(), at the instant it returns. The
    cache's own restore() runs unchanged and its result is returned untouched; only this
    instance's attribute is replaced, never the class."""

    def __init__(self, cache, mem):
        self.cache, self.mem, self.calls, self._restore = cache, mem, [], cache.restore
        cache.restore = self

    def detach(self) -> None:
        """Remove the instance override, so lookup returns to the class's restore(). This also breaks
        the cache -> observer -> cache cycle, so the entries are freed by reference counting when
        the case ends, not at a later cyclic GC."""
        if self.cache.__dict__.get("restore") is self:
            del self.cache.restore

    def __call__(self, token_ids, keys, config):
        kv, L, sources = self._restore(token_ids, keys, config)
        self.mem.cut("restore")  # lookup, gather and D4 copy; the check below is outside every segment
        self.calls.append({"restore_len": L, "counters": _nonzero(self.cache.counters),
                           "problems": restored_vs_entries(self.cache, token_ids, keys, kv, L, sources)})
        self.mem.resume()
        return kv, L, sources


# ---- counters expected from the key-level models
def _nonzero(counter) -> dict:
    return {k: int(v) for k, v in counter.items() if v}


def c1_expected(prompts, C, recs, idx, phase) -> dict:
    """Cumulative C1 counters implied by the key-level model after request `idx`'s restore
    (phase "restore") or after its admission ("after"). Zero counts are omitted."""
    e = Counter()
    for j, (toks, r) in enumerate(zip(prompts[:idx + 1], recs, strict=False)):
        n = r["restore_len"] // C
        e["lookups"] += 1
        e["chunk_misses"] += n < max_restore_len(len(toks), C) // C
        e["chunk_hits"] += n
        e["restored_requests"] += n > 0
        e["recovered_prefix_tokens"] += r["restore_len"]
        if j < idx or phase == "after":
            e["chunks_stored"] += len(r["stored"])
            e["evictions"] += r["evictions"]
            e["write_skips_no_capacity"] += r.get("no_capacity_skips", 0)
            e["admission_stops_prefix_capacity"] += r.get("stopped", False)
            e["chunks_refused_prefix_capacity"] += r.get("refused", 0)
    return _nonzero(e)


def c2_expected(prompts, C, recs, idx, pl, fs) -> dict:
    """Every TieredKVCache.Counters field implied by the tiered model after request `idx`. The model
    assumes that every admission, spill and read succeeds, so the fault Counters are empty. Its
    tiers are disjoint (a host victim is never already on disk), so every spill writes one file."""
    e = dict.fromkeys(("requests", "restored_requests", "full_recompute_requests", "host_hits", "disk_hits",
                       "chunk_misses", "recovered_prefix_tokens", "chunks_stored", "host_spills_to_disk"), 0)
    for toks, r in zip(prompts[:idx + 1], recs, strict=False):
        n = r["restore_len"] // C
        e["requests"] += 1
        e["restored_requests" if n else "full_recompute_requests"] += 1
        e["host_hits"] += r["sources"].count("host")
        e["disk_hits"] += r["sources"].count("disk")
        e["chunk_misses"] += int(n < max_restore_len(len(toks), C) // C)
        e["recovered_prefix_tokens"] += r["restore_len"]
        e["chunks_stored"] += len(r["stored"])
        e["host_spills_to_disk"] += r["spills"]
    e.update(miss_reasons={"absent": e["chunk_misses"]} if e["chunk_misses"] else {}, write_skips={},
             spill_failures={}, bytes_d2h=e["chunks_stored"] * pl, bytes_h2d=(e["host_hits"] + e["disk_hits"]) * pl,
             disk_file_bytes_read=e["disk_hits"] * fs, disk_file_bytes_written=e["host_spills_to_disk"] * fs)
    return e


def c2_store_expected(recs, idx, pl, fs) -> dict:
    cum = recs[:idx + 1]
    h, d = cum[-1]["host_entries"], cum[-1]["disk_entries"]
    return {"host_evictions": sum(x["spills"] for x in cum), "disk_evictions": sum(x["disk_evictions"] for x in cum),
            "host_entries": h, "disk_entries": d, "host_resident_B": h * pl, "disk_committed_B": d * fs,
            "disk_used_B": d * fs, "disk_tmp_B": 0, "disk_undeletable_B": 0, "disk_stale_removed": 0,
            "disk_unlink_failures": {}}


def c2_store_observed(t) -> dict:
    return {"host_evictions": t.host.evictions, "disk_evictions": t.disk.evictions, "host_entries": len(t.host),
            "disk_entries": len(t.disk), "host_resident_B": t.host.resident_bytes,
            "disk_committed_B": t.disk.committed_bytes, "disk_used_B": t.disk.used_bytes,
            "disk_tmp_B": t.disk.tmp_bytes, "disk_undeletable_B": t.disk.undeletable_bytes,
            "disk_stale_removed": t.disk.stale_removed, "disk_unlink_failures": dict(t.disk.unlink_failures)}


def device_persistent(cfgs) -> dict:
    return {n: (c.cache.retained_bytes() if n.startswith("C1") else 0) for n, c in cfgs.items()}


# ---- diagnostics file
def write_diagnostics(path, doc) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "x") as f:  # exclusive create: never overwrites earlier evidence
        json.dump(doc, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())


def diagnostics_problems(doc, case, trace, *, steps: int, require_allocator: bool) -> list:
    """Problems (empty = complete) of one case's diagnostics document."""
    probs = []
    for k, v in (("schema", DIAG_SCHEMA), ("case", case), ("complete", True), ("failure", None), ("fields", FIELDS),
                 ("steps", steps)):
        if doc.get(k) != v:
            probs.append(f"{k} = {doc.get(k)!r}")
    if (doc.get("trace") or {}).get("digest") != trace.digest:
        probs.append("trace digest differs")
    if set(doc.get("wall_time_s") or {}) != set(CONFIGS) or set(doc.get("configs") or {}) != set(CONFIGS):
        probs.append("configurations missing from wall_time_s or configs")
    recs, n = doc.get("records") or [], len(trace.requests)
    if [(r.get("config"), r.get("request_index")) for r in recs] != [(c, i) for c in CONFIGS for i in range(n)]:
        return probs + ["records are not exactly every configuration x request, in order"]
    for r in recs:
        where = (r["config"], r["request_index"])
        req = trace.requests[r["request_index"]]
        if r.get("request_id") != req.id or r.get("prompt_len") != len(req.tokens) or \
                r.get("prompt_sha256") != hashlib.sha256(repr(req.tokens).encode()).hexdigest():
            probs.append(f"{where}: request identity")
        for k in ("eligible_chunks", "restore_len", "sources", "stored_chunks"):
            if k not in r:
                probs.append(f"{where}: {k} missing")
        a, lg, ct = r.get("allocator") or {}, r.get("logical") or {}, r.get("counters") or {}
        want_seg = C1_SEGMENTS if r["config"].startswith("C1") else OTHER_SEGMENTS
        if [s.get("segment") for s in a.get("segments") or []] != want_seg:
            probs.append(f"{where}: allocator segments")
        if require_allocator:
            vals = [a.get("A0_B"), a.get("P_B"), a.get("P_minus_A0_B")] + [
                s.get(k) for s in a.get("segments") or [] for k in ("allocated_at_start_B", "max_memory_allocated_B",
                                                                    "allocated_at_end_B")]
            if not all(isinstance(v, int) and v >= 0 for v in vals) or a["P_minus_A0_B"] != a["P_B"] - a["A0_B"] or \
                    a["P_B"] != max(s["max_memory_allocated_B"] for s in a["segments"]):
                probs.append(f"{where}: allocator values incomplete or inconsistent")
        lkeys = ["request_kv_storage_B", "restore_scratch_B", "device_persistent_before_B", "device_persistent_after_B"]
        lkeys += ["host_resident_B", "disk_used_B"] if r["config"].startswith("C2") else []
        if not all(isinstance(lg.get(k), (int, dict)) for k in lkeys):
            probs.append(f"{where}: logical storage terms missing")
        if r["config"] != "C0" and not ("observed" in ct and "expected" in ct and ct["observed"] == ct["expected"]):
            probs.append(f"{where}: counters missing or unequal")
        if r["config"].startswith("C1") and not (ct.get("at_restore") or {}).keys() >= {"observed", "expected"}:
            probs.append(f"{where}: restore-boundary counters missing")
    return probs


# ---- one case: every configuration, every request, every check
def run_case(model, ns, case, trace, b, tmp, *, steps, mem, diag_path) -> dict:
    C, pl, fs = ns.chunk_size, cf.payload_len(ns), cf.file_size(ns)
    dev = next(model.parameters()).device
    prompts = [list(r.tokens) for r in trace.requests]
    models = {"C1LRU_HALF": tm.lru_model(prompts, C, b["C1PA_HALF"]), "C1PA_W": tm.pa_model(prompts, C, b["C1PA_W"]),
              "C1PA_HALF": tm.pa_model(prompts, C, b["C1PA_HALF"]),
              "C2": tm.tiered_model(prompts, C, b["host_slots"], b["disk_files"])}
    cfgs, boundaries = configs(model, ns, b, tmp), []
    assert tuple(cfgs) == CONFIGS
    doc = {"schema": DIAG_SCHEMA, "case": case, "complete": False, "failure": None,
           "trace": {"header": trace.header, "digest": trace.digest, "requests": len(prompts),
                     "ids": [r.id for r in trace.requests]},
           "chunk_size": C, "steps": steps, "payload_B": pl, "chunk_file_B": fs, "budgets": b,
           "configs": {n: c.describe() for n, c in cfgs.items()},
           "device": {"device": str(dev), "name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                      "torch": torch.__version__, "cuda": torch.version.cuda},
           "fields": FIELDS, "wall_time_s": {}, "records": []}
    try:
        for name, cfg in cfgs.items():
            t_start = time.monotonic()
            is_c1, is_c2 = name.startswith("C1"), name.startswith("C2")
            want = models.get("C2" if is_c2 else name)
            ref = ResidentReference(model, ns.digest(), C)
            boundary = RestoreBoundary(cfg.cache, mem) if is_c1 else None
            boundaries += [boundary] if boundary else []
            if is_c1:  # the C1 models omit no-capacity skips, so the gate's capacities must be > 0
                assert cfg.cache.capacity_chunks == b["C1PA_W" if name == "C1PA_W" else "C1PA_HALF"] > 0
            for idx, toks in enumerate(prompts):
                before = device_persistent(cfgs)
                mem.begin()
                p = cfg.prefill(toks)
                alloc = mem.end(C1_SEGMENTS[1] if is_c1 else OTHER_SEGMENTS[0])
                after = device_persistent(cfgs)
                rec = {"config": name, "request_index": idx, "request_id": trace.requests[idx].id,
                       "prompt_len": len(toks),
                       "prompt_sha256": hashlib.sha256(repr(trace.requests[idx].tokens).encode()).hexdigest(),
                       "eligible_chunks": max_restore_len(len(toks), C) // C, "restore_len": p.restore_len,
                       "sources": list(p.sources), "stored_chunks": list(p.stored_chunks), "allocator": alloc,
                       "logical": {"request_kv_storage_B": cache_storage_bytes(p.cache),
                                   "restore_scratch_B": p.restore_scratch_bytes,
                                   "device_persistent_before_B": before, "device_persistent_after_B": after},
                       "counters": {}}
                doc["records"].append(rec)  # recorded before any assertion, so a failure keeps its evidence
                if want is None:
                    assert p.restore_len == 0 and not p.stored_chunks and not p.sources, (name, idx)
                else:
                    w = want[idx]
                    assert (p.restore_len, list(p.stored_chunks)) == (w["restore_len"], w["stored"]), (name, idx)
                if is_c1:
                    c = cfg.cache
                    call = boundary.calls[-1] if len(boundary.calls) == idx + 1 else None
                    assert call is not None, (name, idx, "restore() was not observed exactly once per request")
                    rec["counters"] = {"at_restore": {"observed": call["counters"],
                                                      "expected": c1_expected(prompts, C, want, idx, "restore")},
                                       "observed": _nonzero(c.counters),
                                       "expected": c1_expected(prompts, C, want, idx, "after")}
                    assert call["restore_len"] == p.restore_len and not call["problems"], (name, idx, call["problems"])
                    assert rec["counters"]["at_restore"]["observed"] == rec["counters"]["at_restore"]["expected"], \
                        (name, idx, rec["counters"]["at_restore"])
                    assert rec["counters"]["observed"] == rec["counters"]["expected"], (name, idx, rec["counters"])
                if is_c2:
                    t = cfg.engine.tiered
                    assert list(p.sources) == w["sources"], (name, idx)
                    rec["logical"].update(host_resident_B=t.host.resident_bytes, disk_used_B=t.disk.used_bytes)
                    rec["counters"] = {"observed": {**t.counters.to_dict(), **c2_store_observed(t)},
                                       "expected": {**c2_expected(prompts, C, want, idx, pl, fs),
                                                    **c2_store_expected(want, idx, pl, fs)}}
                    assert rec["counters"]["observed"] == rec["counters"]["expected"], (name, idx, rec["counters"])
                    assert t.host.resident_bytes <= t.host.slots * pl and t.disk.used_bytes <= t.disk.budget_bytes
                if p.restore_len:
                    chk = ref.check_restored(toks, p.cache, p.restore_len)
                    assert chk["all_bytes_equal"], (name, idx, chk)
                if is_c1:
                    assert c.retained_bytes() == len(c.entries) * pl <= c.budget_bytes
                    if isinstance(c, PrefixAwareGPUPrefixCache):
                        c.check_invariants()
                    ent = _ranges(t for _, t in c.entries.values())
                    req = _ranges(x for layer in p.cache.layers for x in (layer.keys, layer.values))
                    assert not _overlap(ent, req), (name, idx)
                tokens, step_logits = greedy_continue(model, p.cache, p.logits, len(toks), steps)
                ref_tokens, ref_steps = ref.run(toks, p.restore_len, p.stored_chunks, steps)
                assert len(tokens) == len(ref_tokens) == len(step_logits) == len(ref_steps) == steps, (name, idx)
                assert tokens == ref_tokens, (name, idx)
                assert all(torch.equal(x, y) for x, y in zip(step_logits, ref_steps, strict=True)), (name, idx)
            if is_c2:
                t = cfg.engine.tiered
                assert not t.counters.spill_failures and not t.counters.write_skips
                assert set(t.counters.miss_reasons) <= {"absent"}
                assert t.disk.undeletable_bytes == 0 and t.disk.tmp_bytes == 0 and not t.disk.unlink_failures
                assert t.disk.used_bytes <= t.disk.budget_bytes
                t.disk.clear()
                t.host.close()
            del ref
            doc["wall_time_s"][name] = time.monotonic() - t_start
        doc["complete"] = True
    except BaseException as e:  # recorded in the diagnostics, then re-raised
        doc["failure"] = repr(e)[:4000]
        raise
    finally:
        for bd in boundaries:
            bd.detach()
        write_diagnostics(diag_path, doc)
    return doc


@pytest.mark.parametrize("trace_name", CASES)
def test_gate_strict_decisions_ownership_and_faults(diag_dir, model, ns, tmp_path, trace_name):
    traces = gate_traces()
    run_case(model, ns, trace_name, traces[trace_name], budgets_for(traces)[trace_name], tmp_path, steps=STEPS,
             mem=CudaMemory(), diag_path=diag_dir / f"{trace_name}.json")


def test_gate_diagnostics_are_complete(diag_dir):
    """Runs after the five cases (file order). Each must have left exactly one complete diagnostics
    file, so a skipped, deselected or missing case fails the gate here."""
    traces = gate_traces()
    assert sorted(p.name for p in diag_dir.iterdir()) == sorted(f"{c}.json" for c in CASES)
    for c in CASES:
        probs = diagnostics_problems(json.loads((diag_dir / f"{c}.json").read_text()), c, traces[c], steps=STEPS,
                                     require_allocator=True)
        assert not probs, (c, probs[:20])
