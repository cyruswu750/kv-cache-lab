"""C1-PA (kvlab/prefix_aware_cache.py) on CPU with a tiny random bf16 Qwen2 (no weights, no GPU).

Hand-worked sequences use C = 4. Correctness, not performance: every replay is also checked bitwise
against ResidentReference replaying C1-PA's own decisions (D12)."""

import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import trace_models as tm
from kvlab import trace_replay as rp
from kvlab import traces as tr
from kvlab.gpu_prefix_cache import GPUPrefixCache
from kvlab.identity import ROOT_PARENT, ChunkIdentity, chain_keys, namespace_for
from kvlab.model import greedy_continue
from kvlab.prefix_aware_cache import PrefixAwareGPUPrefixCache

C, STEPS = 4, 2


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized(), "M4-B CPU tests must not initialise CUDA"


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)


def pa(model, ns, chunks):
    return rp.GPUOnly(model, PrefixAwareGPUPrefixCache(ns, chunks * cf.payload_len(ns), "cpu"))


def hand(prompts):
    return tr.Trace(tr._header("hand", 0, {}), tuple(tr.TraceRequest(f"r{i}", "g", tuple(p)) for i, p in enumerate(prompts)))


def labels(ns, cache, named):
    lab = {}
    for tag, p in named.items():
        for i, k in enumerate(chain_keys(ns.digest(), list(p), C)):
            lab.setdefault(k, f"{tag}{i}")
    return [lab[k] for k in cache.entries]


A = tuple(range(20, 41))                 # N = 21: 5 complete chunks, all eligible
S = tuple(range(100, 108))               # a 2-chunk shared root
X = S + tuple(range(200, 209))           # s0 s1 x2 x3 (N = 17)
Y = S + tuple(range(300, 313))           # s0 s1 y2 y3 y4 (N = 21)
Z = S + tuple(range(400, 403))           # s0 s1 (N = 11)


def test_oversized_prompt_keeps_its_leading_chunks_and_stops(model, ns):
    """Capacity 3, a 5-chunk prompt three times. r0 admits A0 A1 A2; at A3 the cache is full and
    every leaf (A2) is protected (the prompt's own chunk), so admission stops (refused 2). r1, r2
    restore 3 chunks (L = 12) and stop again at A3. Plain LRU keeps only A2-A4 and restores 0."""
    cfg = pa(model, ns, 3)
    trace = hand([A, A, A])
    res = rp.replay(cfg, trace, STEPS)
    assert [r["restore_len"] for r in res.records] == [0, 12, 12]
    assert [r["stored_chunks"] for r in res.records] == [[0, 1, 2], [], []]
    assert labels(ns, cfg.cache, {"A": A}) == ["A0", "A1", "A2"]
    c = cfg.cache.counters
    assert (c["admission_stops_prefix_capacity"], c["chunks_refused_prefix_capacity"], c["evictions"]) == (3, 6, 0)
    assert tr.lru_restore_lengths(trace, C, 3) == [0, 0, 0]          # the reviewed plain LRU, for contrast
    assert rp.strict_mismatches(model, ns.digest(), C, trace, res) == []
    m = tm.pa_model([A, A, A], C, 3)
    assert [x["restore_len"] for x in m] == [0, 12, 12] and [x["refused"] for x in m] == [2, 2, 2]


def test_shared_root_survives_an_overflowing_request(model, ns):
    """Capacity 4: X stores s0 s1 x2 x3; Z hits s0 s1. Y hits s0 s1 (protected), then:
    y2 evicts the LRU leaf x3; y3 evicts x2 (now a leaf); at y4 the only leaf is y3 (protected),
    so admission stops. The root survives, so both later Z requests restore 8. Plain LRU evicts s0
    while admitting y4, and the next Z restores 0."""
    cfg = pa(model, ns, 4)
    trace = hand([X, Z, Y, Z, Z])
    res = rp.replay(cfg, trace, STEPS)
    assert [r["restore_len"] for r in res.records] == [0, 8, 8, 8, 8]
    assert [r["stored_chunks"] for r in res.records] == [[0, 1, 2, 3], [], [2, 3], [], []]
    # after Y the order (LRU first) is s0 s1 y2 y3; the two Z hits then touch s0 s1 again
    assert labels(ns, cfg.cache, {"X": X, "Y": Y}) == ["Y2", "Y3", "X0", "X1"]
    assert (cfg.cache.counters["evictions"], cfg.cache.counters["chunks_refused_prefix_capacity"]) == (2, 1)
    assert tr.lru_restore_lengths(trace, C, 4) == [0, 8, 8, 0, 8]
    assert rp.strict_mismatches(model, ns.digest(), C, trace, res) == []


def test_random_traces_keep_closure_follow_the_model_and_match_the_reference(model, ns):
    rng = random.Random(7)
    pool = [tuple(rng.randrange(10, 500) for _ in range(rng.randint(5, 13))) for _ in range(4)]
    for case in range(12):
        prompts = []
        for _ in range(10):  # extend shared stems so prefixes overlap at chunk boundaries
            base = rng.choice(pool)
            prompts.append(base[:rng.randint(1, len(base))] + tuple(rng.randrange(10, 500) for _ in range(rng.randint(0, 6))))
        trace = hand(prompts)
        for cap in (0, 1, 2, 3, 5):
            cfg = pa(model, ns, cap)
            res = rp.replay(cfg, trace, 1)
            model_recs = tm.pa_model(prompts, C, cap)
            assert [r["restore_len"] for r in res.records] == [m["restore_len"] for m in model_recs], (case, cap)
            assert [r["stored_chunks"] for r in res.records] == [m["stored"] for m in model_recs], (case, cap)
            cache = cfg.cache
            cache.check_invariants()
            assert all(p == ROOT_PARENT or p in cache.entries for p in cache.parent.values())
            assert cache.retained_bytes() == len(cache.entries) * cache.chunk_bytes <= cache.budget_bytes
            assert cache.counters["evictions"] == sum(m["evictions"] for m in model_recs)
            assert rp.strict_mismatches(model, ns.digest(), C, trace, res) == []
            if cap == 0:
                assert not cache.entries and cache.counters["write_skips_no_capacity"] > 0


def test_continuation_never_aliases_or_mutates_the_cache(model, ns):
    cfg = pa(model, ns, 3)
    cache = cfg.cache
    snaps = {}
    for p in (X, Z, Y, Z):
        out = cfg.prefill(p)
        for k, (_, t) in cache.entries.items():
            snaps.setdefault(k, t.clone())
        ranges = [(t.untyped_storage().data_ptr(), t.untyped_storage().data_ptr() + t.untyped_storage().nbytes())
                  for _, t in cache.entries.values()]
        req = [(x.untyped_storage().data_ptr(), x.untyped_storage().data_ptr() + x.untyped_storage().nbytes())
               for layer in out.cache.layers for x in (layer.keys, layer.values)]
        assert not any(a < d and c < b for a, b in ranges for c, d in req)
        greedy_continue(model, out.cache, out.logits, len(p), 3)
        with torch.inference_mode():
            for layer in out.cache.layers:
                layer.keys.fill_(7.0)
        assert all(torch.equal(t, snaps[k]) for k, (_, t) in cache.entries.items())
    assert cache.counters["chunk_hits"] > 0


def test_identity_mismatch_raises_and_broken_bookkeeping_is_detected(model, ns):
    cfg = pa(model, ns, 4)
    cfg.prefill(X)
    cache = cfg.cache
    k0 = next(iter(cache.entries))
    ident, t = cache.entries[k0]
    cache.entries[k0] = (ChunkIdentity(**{**ident.__dict__, "index": 5}), t)
    keys = chain_keys(ns.digest(), list(Z), C)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        cache.restore(list(Z), keys[:2], model.config)
    cache.entries[k0] = (ident, t)
    cache.children[k0] += 1
    with pytest.raises(RuntimeError, match="child counts"):
        cache.check_invariants()


def test_plain_lru_is_preserved_for_the_offline_diagnostic(model, ns):
    """The reviewed GPUPrefixCache is unchanged; the independent lru_model agrees with it and with
    kvlab.traces.lru_restore_lengths, and counts the unreachable tail of the oversized case."""
    trace = hand([A, A, X, Z, Y, Z])
    for cap in (2, 3, 4):
        res = rp.replay(rp.GPUOnly(model, GPUPrefixCache(ns, cap * cf.payload_len(ns), "cpu")), trace, 1)
        m = tm.lru_model([r.tokens for r in trace.requests], C, cap)
        got = [r["restore_len"] for r in res.records]
        assert got == [x["restore_len"] for x in m] == tr.lru_restore_lengths(trace, C, cap)
    assert tm.lru_model([A], C, 3)[0]["dead_after"] == 3  # A2 A3 A4 cached, A0 missing


def test_storing_does_not_refresh_present_chunks(model, ns):
    """Capacity 3, two competing leaves. A = a0 a1 (+1); B = b0 (+1); A8 = A[:8] is an exact
    multiple, so only a0 is eligible and touched, while a1 is present but not refreshed. The LRU
    order is then a1, b0, a0, and Z's admission evicts the least recent leaf, a1. So B later
    restores b0 (4) and A restores a0 only (4). If storing refreshed present chunks, Z would evict
    b0 instead, and B would restore 0 and A 8."""
    Ap, B, Zp = tuple(range(20, 29)), tuple(range(50, 55)), tuple(range(70, 75))
    prompts = [Ap, B, Ap[:8], Zp, B, Ap]
    got = [r["restore_len"] for r in rp.replay(pa(model, ns, 3), hand(prompts), 1).records]
    assert got == [x["restore_len"] for x in tm.pa_model(prompts, C, 3)] == [0, 0, 4, 0, 4, 4]
