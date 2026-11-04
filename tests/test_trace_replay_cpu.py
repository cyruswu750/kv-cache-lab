"""M4-A trace replay of C0 (no cache), C1 (bounded GPU-only prefix cache) and C2 (existing tiered
always-restore) on CPU, with a tiny randomly initialised Qwen2 model (no weights, no GPU).

These are correctness tests, not performance evidence. Every configuration is checked against
an independent resident replay of its OWN decisions (D12); different configurations are never
compared bitwise with each other where their splits differ (D10).
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import trace_replay as rp
from kvlab import traces as tr
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.gpu_prefix_cache import GPUPrefixCache
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.model import greedy_continue
from kvlab.tiered_cache import TieredKVCache

C, STEPS = 4, 3
V = (10, 500)
TRACES = {
    "no_reuse": tr.no_reuse(1, n_requests=6, min_len=3, max_len=14, vocab=V),
    "shared_prefix": tr.shared_prefix(2, n_requests=6, prefix_len=9, suffix_min=1, suffix_max=6, vocab=V),
    "multi_turn": tr.multi_turn(3, n_conversations=2, n_turns=3, opening_len=5, user_min=1, user_max=4,
                                answer_min=1, answer_max=4, vocab=V),
    "doc_reuse": tr.doc_reuse(4, n_requests=8, n_docs=3, instruction_len=3, doc_min=4, doc_max=9, question_min=1,
                              question_max=3, zipf_s=1.1, vocab=V),
}


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized(), "M4-A tests must not initialise CUDA"


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)


def per_token_kv_bytes(ns):
    return ns.num_layers * 2 * ns.num_kv_heads * ns.head_dim * 2  # bf16


def c1(model, ns, chunks=None, budget=None):
    cb = cf.payload_len(ns)
    return rp.GPUOnly(model, GPUPrefixCache(ns, budget if budget is not None else chunks * cb, "cpu"))


def c2(model, ns, root, host_slots=2, disk_files=3, backend="baseline"):
    cb = cf.payload_len(ns)
    host = HostStore((host_slots + 1) * cb, cb, max_entries=64, pin=False)
    disk = DiskStore(root, disk_files * cf.file_size(ns), max_entries=64)
    return rp.Tiered(AlwaysRestoreEngine(model, TieredKVCache(ns, host, disk, restore_backend=backend)))


CONFIGS = {
    "C0": lambda m, ns, root: rp.NoCache(m),
    "C1_ample": lambda m, ns, root: c1(m, ns, chunks=100),
    "C1_two_chunks": lambda m, ns, root: c1(m, ns, chunks=2),
    "C2_baseline_spilling": lambda m, ns, root: c2(m, ns, root),            # host 2 slots + disk 3 files
    "C2_batched_spilling": lambda m, ns, root: c2(m, ns, root, backend="batched"),  # opt-in, explicitly
}


@pytest.mark.parametrize("kind", list(TRACES))
@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_configuration_matches_a_resident_replay_of_its_own_decisions(model, ns, tmp_path, kind, name):
    trace = TRACES[kind]
    cfg = CONFIGS[name](model, ns, tmp_path / "disk")
    result = rp.replay(cfg, trace, STEPS)
    assert result.trace_sha256 == trace.digest and len(result.records) == len(trace.requests)
    assert all(r["forward_calls"] == STEPS for r in result.records)  # 1 prefill + STEPS-1 decode, never repeated
    assert rp.strict_mismatches(model, ns.digest(), C, trace, result) == []
    restored = [r["restore_len"] for r in result.records]
    assert all(L % C == 0 and L <= max(len(q.tokens) - 1, 0) for L, q in zip(restored, trace.requests))
    if name == "C0" or kind == "no_reuse":
        assert not any(restored)
    elif name in ("C1_ample", "C2_baseline_spilling", "C2_batched_spilling"):
        assert any(restored), "the reuse trace must exercise restores"


def test_the_strict_check_detects_a_wrong_output(model, ns):
    """Negative controls for the oracle. (A different declared split is NOT used as a control:
    under D10 another split may, but need not, round differently; on this tiny model a split one
    chunk shorter happens to give bitwise-equal outputs. The strict check verifies outputs given
    the declared decisions; the decisions are checked by the LRU and restore-length tests.)"""
    trace = TRACES["multi_turn"]
    result = rp.replay(c1(model, ns, chunks=100), trace, STEPS)
    i = next(i for i, r in enumerate(result.records) if r["restore_len"])
    result.records[i]["tokens"][-1] ^= 1              # one greedy token off
    assert rp.strict_mismatches(model, ns.digest(), C, trace, result) == [trace.requests[i].id]
    result = rp.replay(c1(model, ns, chunks=100), trace, STEPS)
    result.records[1]["step_logits"][0] = result.records[1]["step_logits"][0].clone()
    result.records[1]["step_logits"][0][0] += 1.0     # one logit off
    assert rp.strict_mismatches(model, ns.digest(), C, trace, result) == [trace.requests[1].id]


def test_compared_configurations_replay_identical_requests(model, ns, tmp_path):
    trace = TRACES["doc_reuse"]
    results = [rp.replay(c, trace, STEPS) for c in (rp.NoCache(model), c1(model, ns, chunks=3), c2(model, ns, tmp_path / "d"))]
    assert len({r.trace_sha256 for r in results}) == 1
    columns = [[(x["id"], x["prompt_len"], x["prompt_sha256"]) for x in r.records] for r in results]
    assert columns[0] == columns[1] == columns[2]
    # the request's own prompt KV has the same size in every configuration: N tokens, owned storage
    for recs in zip(*(r.records for r in results)):
        assert {x["request_kv_bytes"] for x in recs} == {recs[0]["prompt_len"] * per_token_kv_bytes(ns)}


@pytest.mark.parametrize("kind", ["shared_prefix", "multi_turn", "doc_reuse"])
@pytest.mark.parametrize("cap", [0, 1, 2, 3, 100])
def test_gpu_only_restores_follow_the_independent_key_level_lru_model(model, ns, kind, cap):
    trace = TRACES[kind]
    got = [r["restore_len"] for r in rp.replay(c1(model, ns, chunks=cap), trace, STEPS).records]
    assert got == tr.lru_restore_lengths(trace, C, cap)
    if cap == 100:  # larger than every working set here: the unbounded oracle
        assert got == tr.oracle_reuse(trace, C)


@pytest.mark.parametrize("budget_chunks, extra, cap", [(0, 0, 0), (1, -1, 0), (1, 0, 1), (3, -1, 2), (3, 0, 3)])
def test_capacity_is_a_hard_bound_on_owned_retained_storage(model, ns, budget_chunks, extra, cap):
    cb = cf.payload_len(ns)
    cfg = c1(model, ns, budget=budget_chunks * cb + extra)
    cache = cfg.cache
    assert cache.capacity_chunks == cap
    trace = TRACES["multi_turn"]
    for r in rp.replay(cfg, trace, STEPS).records:
        assert r["persistent_device_cache_bytes"] <= cache.budget_bytes
    tensors = [t for _, t in cache.entries.values()]
    assert len(tensors) == min(cap, tr.trace_stats(trace, C)["unique_chunks"])
    assert cache.retained_bytes() == len(tensors) * cb <= cache.budget_bytes
    assert cache.peak_retained_bytes <= cache.budget_bytes
    ptrs = [t.untyped_storage().data_ptr() for t in tensors]
    assert len(set(ptrs)) == len(ptrs)  # one storage per entry, nothing shared
    for t in tensors:  # an owned allocation of exactly one chunk, not a view into a request's KV
        assert t.is_contiguous() and t.untyped_storage().nbytes() == cb and t.storage_offset() == 0
    if 0 < cap < tr.trace_stats(trace, C)["unique_chunks"]:
        assert cache.counters["evictions"] > 0
    if cap == 0:
        assert cache.counters["write_skips_no_capacity"] > 0 and not cache.entries


def test_zero_capacity_gpu_only_is_bitwise_the_no_cache_path(model, ns):
    trace = TRACES["multi_turn"]
    a = rp.replay(rp.NoCache(model), trace, STEPS).records
    b = rp.replay(c1(model, ns, chunks=0), trace, STEPS).records
    for x, y in zip(a, b, strict=True):  # same split (L = 0) on both sides
        assert y["restore_len"] == 0 and y["stored_chunks"] == [] and x["tokens"] == y["tokens"]
        assert all(torch.equal(p, q) for p, q in zip(x["step_logits"], y["step_logits"]))


def _storage_ranges(tensors):
    return [(t.untyped_storage().data_ptr(), t.untyped_storage().data_ptr() + t.untyped_storage().nbytes())
            for t in tensors]


def test_cached_prefixes_are_never_aliased_or_mutated_by_continuation(model, ns):
    cfg = c1(model, ns, chunks=100)
    cache = cfg.cache
    snapshots = {}
    for r in TRACES["multi_turn"].requests:
        p = cfg.prefill(r.tokens)
        for key, (_, t) in cache.entries.items():
            snapshots.setdefault(key, t.clone())
        stored = _storage_ranges(t for _, t in cache.entries.values())
        request = _storage_ranges(x for layer in p.cache.layers for x in (layer.keys, layer.values))
        assert not any(a < d and c < b for a, b in stored for c, d in request), "request KV aliases the cache"
        greedy_continue(model, p.cache, p.logits, len(r.tokens), STEPS)   # suffix decode appends to the request KV
        with torch.inference_mode():                                       # and a hostile in-place write
            for layer in p.cache.layers:
                layer.keys.zero_()
                layer.values.fill_(1.0)
        for key, (_, t) in cache.entries.items():
            assert torch.equal(t, snapshots[key]), "a cached prefix changed after the request continued"
    assert cache.counters["chunk_hits"] > 0
    # the restored cache itself, before any suffix prefill: owned copies, not views of the entries
    last = list(TRACES["multi_turn"].requests[-1].tokens)
    keys = chain_keys(ns.digest(), last, C)
    kv, L, _ = cache.restore(last, keys[:(len(last) - 1) // C], model.config)
    assert L > 0
    restored = _storage_ranges(x for layer in kv.layers for x in (layer.keys, layer.values))
    stored = _storage_ranges(t for _, t in cache.entries.values())
    assert not any(a < d and c < b for a, b in stored for c, d in restored)
    for layer in kv.layers:
        layer.keys.zero_()
    assert all(torch.equal(t, snapshots[k]) for k, (_, t) in cache.entries.items())


def test_partial_prefix_exact_multiple_and_full_eligible_restores(model, ns, tmp_path):
    P = tuple(range(20, 31))                    # N = 11: chunks [0,4) [4,8); eligible L = 8
    prompts = [P,
               P[:6] + (99, 98),                # shares chunk 0 only: partial restore, L = 4
               P[:8],                           # N = 8 is a multiple of C: L <= 7, so 4 although both chunks are cached
               P[:9]]                           # N = 9: L = 8
    trace = tr.Trace(tr._header("hand", 0, {}), tuple(tr.TraceRequest(f"r{i}", "g", p) for i, p in enumerate(prompts)))
    for cfg in (c1(model, ns, chunks=100), c2(model, ns, tmp_path / "d", host_slots=8)):
        result = rp.replay(cfg, trace, STEPS)
        assert [r["restore_len"] for r in result.records] == [0, 4, 4, 8]
        assert [r["stored_chunks"] for r in result.records] == [[0, 1], [1], [], []]
        assert rp.strict_mismatches(model, ns.digest(), C, trace, result) == []


def test_storing_does_not_refresh_present_chunks(model, ns):
    """As in the host tier, storing skips a present chunk without touching it. Hand-worked, C = 4,
    capacity 2: X stores x0 x1; Y = X[:8] is an exact multiple, so only x0 is eligible (touched)
    while x1 is present but not refreshed; Z evicts the least recent chunk, x1; X then restores x0
    only. If storing refreshed x1, Z would evict x0 instead and X would restore nothing."""
    X = tuple(range(20, 29))
    prompts = [X, X[:8], (40, 41, 42, 43, 44), X]
    trace = tr.Trace(tr._header("hand", 0, {}), tuple(tr.TraceRequest(f"r{i}", "g", p) for i, p in enumerate(prompts)))
    got = [r["restore_len"] for r in rp.replay(c1(model, ns, chunks=2), trace, STEPS).records]
    assert got == tr.lru_restore_lengths(trace, C, 2) == [0, 4, 0, 4]


def test_identity_and_model_mismatches_raise_instead_of_missing(model, ns):
    cfg = c1(model, ns, chunks=4)
    P = tuple(range(20, 31))
    cfg.prefill(P)
    keys = chain_keys(ns.digest(), list(P), C)
    ident, t = cfg.cache.entries[keys[0]]
    cfg.cache.entries[keys[0]] = (ChunkIdentity(ident.ns_digest, ident.key, ident.parent, ident.index, (0, 0, 0, 0)), t)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        cfg.prefill(P)
    other = namespace_for(model, "toy/tiny-qwen2", "0" * 40, 8)   # another chunk size: different payload shape
    p = rp.NoCache(model).prefill(P)
    with pytest.raises(ValueError, match="does not match"):
        GPUPrefixCache(other, 10 ** 6, "cpu").store(
            type(p.cache)(ddp_cache_data=[(layer.keys.float(), layer.values.float()) for layer in p.cache.layers],
                          config=model.config), list(P), chain_keys(other.digest(), list(P), 8))


def test_tiered_configuration_is_the_unchanged_engine(model, ns, tmp_path):
    trace = TRACES["doc_reuse"]
    a = rp.replay(c2(model, ns, tmp_path / "a"), trace, STEPS).records
    engine = c2(model, ns, tmp_path / "b").engine
    for x, r in zip(a, trace.requests, strict=True):
        y = engine.run(r.tokens, STEPS)
        assert (x["restore_len"], x["sources"], x["stored_chunks"], x["tokens"]) == \
               (y.restore_len, y.sources, y.stored_chunks, y.tokens)
        assert all(torch.equal(p, q) for p, q in zip(x["step_logits"], y.step_logits))


@pytest.mark.parametrize("backend", ["baseline", "batched"])
def test_resource_terms_are_reported_separately(model, ns, tmp_path, backend):
    cb, trace = cf.payload_len(ns), TRACES["shared_prefix"]
    r0 = rp.replay(rp.NoCache(model), trace, STEPS)
    r1 = rp.replay(c1(model, ns, chunks=5), trace, STEPS)
    r2 = rp.replay(c2(model, ns, tmp_path / "d", host_slots=1, disk_files=4, backend=backend), trace, STEPS)
    for x in r0.records:
        assert (x["persistent_device_cache_bytes"], x["restore_scratch_bytes"], x["host_slab_bytes"], x["disk_used_bytes"]) == (0, 0, 0, 0)
    for x in r1.records:
        assert 0 < x["persistent_device_cache_bytes"] <= 5 * cb and x["host_slab_bytes"] == x["disk_used_bytes"] == 0
        assert x["restore_scratch_bytes"] == x["restore_len"] * per_token_kv_bytes(ns)  # the gather of L tokens
    for x in r2.records:
        n = x["restore_len"] // C
        staging = min(n, 4) * cb if backend == "batched" else 0
        assert x["persistent_device_cache_bytes"] == 0 and x["host_slab_bytes"] == 2 * cb  # 1 slot + 1 staging
        assert x["restore_scratch_bytes"] == n * cb + staging
        assert x["disk_used_bytes"] <= 4 * cf.file_size(ns)
    assert any(x["disk_used_bytes"] > 0 for x in r2.records) and any("disk" in x["sources"] for x in r2.records)
    s = rp.summarize(r2)
    assert s["restored_tokens"] + s["recomputed_tokens"] == tr.trace_stats(trace, C)["prompt_tokens"]
    assert "not summed" in s["note"] and s["restore_backend"] == backend
