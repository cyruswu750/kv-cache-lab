"""M3: the batched restore backend must behave exactly like the baseline (CPU, synthetic caches).

Every scenario builds two stores with identical contents and history, one per backend
(`twins`), applies the same operations to both, and compares everything observable:
restore result (L, sources, miss reason), restored K/V bytes, counters, tier of every key,
host LRU order and pins, staging state and budgets. The restored bytes are also compared with
the source cache. G (restore_staging_chunks) is set small so that groups, partial groups and
group boundaries are exercised with few chunks. device='cpu', pin=False.
"""

import math

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import chain_keys, namespace_for
from kvlab.kv_snapshot import bytes_equal
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import RESTORE_BACKENDS, TieredKVCache
from m1_toy import tokens, toy_cache, toy_ns

NS = toy_ns()
C, CB, FILE = NS.chunk_size, cf.payload_len(NS), cf.file_size(NS)
LAYERS = NS.num_layers


def make(root, backend, host_slots, disk_files, group):
    host = HostStore(budget_bytes=(host_slots + 1) * CB, chunk_bytes=CB, max_entries=100, pin=False)
    disk = DiskStore(root, budget_bytes=disk_files * FILE, max_entries=100)
    return TieredKVCache(NS, host, disk, restore_backend=backend, restore_staging_chunks=group)


def twins(tmp_path, host_slots, disk_files, group=2):
    return {b: make(tmp_path / b, b, host_slots, disk_files, group) for b in RESTORE_BACKENDS}


def state(t, keys):
    b = t.budgets()
    b["disk"].pop("dir_bytes_scanned")  # same value; scanned from a different directory
    return {"counters": t.counters.to_dict(), "tiers": [t.tier_of(k) for k in keys],
            "host_lru": list(t.host.entries), "pins": [e.pins for e in t.host.entries.values()],
            "staging_busy": t.host._staging_busy, "disk_entries": sorted(t.disk.entries), "budgets": b}


def kv_bytes(cache):
    return [(layer.keys.contiguous().view(torch.uint8).clone(), layer.values.contiguous().view(torch.uint8).clone())
            for layer in cache.layers]


def assert_prefix_equal(restored, source, L):
    for a, b in zip(restored.layers, source.layers):
        assert bytes_equal(a.keys, b.keys[:, :, :L, :]) and bytes_equal(a.values, b.values[:, :, :L, :])


def restore_both(ts, toks, keys, source=None):
    """Restore on both twins; check identical results, state and bytes. Returns the baseline result."""
    out = {b: t.restore(toks, keys, "cpu", None) for b, t in ts.items()}
    (cb, Lb, sb, wb), (cx, Lx, sx, wx) = out["baseline"], out["batched"]
    assert (Lb, sb, wb) == (Lx, sx, wx)
    assert state(ts["baseline"], keys) == state(ts["batched"], keys)
    if Lb:
        assert all(torch.equal(p, q) for x, y in zip(kv_bytes(cb), kv_bytes(cx)) for p, q in zip(x, y))
        if source is not None:
            assert_prefix_equal(cx, source, Lb)
    else:
        assert cb is None and cx is None
    return out


def store_both(ts, n_chunks, seed=0, start=100):
    toks = tokens(n_chunks * C + 1, start=start)
    cache = toy_cache(NS, len(toks), seed)
    keys = chain_keys(NS.digest(), toks, C)
    stored = {b: t.store(cache, toks, keys) for b, t in ts.items()}
    assert stored["baseline"] == stored["batched"]
    return toks, cache, keys


def test_one_chunk_host_and_disk(tmp_path):
    ts = twins(tmp_path, host_slots=4, disk_files=4)
    toks, cache, keys = store_both(ts, 1)
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][1:] == (C, ["host"], None)
    ts = twins(tmp_path / "d", host_slots=1, disk_files=4)
    toks, cache, keys = store_both(ts, 2)          # chunk 0 spilled to disk, chunk 1 in host
    out = restore_both(ts, toks, keys[:1], cache)  # plan only chunk 0
    assert out["batched"][1:] == (C, ["disk"], None)
    assert ts["batched"].last_restore["placement_ops"] == 1


@pytest.mark.parametrize("group", [1, 2, 3, 5, 8])
def test_multi_chunk_host_only_for_every_group_size(tmp_path, group):
    ts = twins(tmp_path, host_slots=8, disk_files=1, group=group)
    toks, cache, keys = store_both(ts, 5)
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][1:] == (5 * C, ["host"] * 5, None)
    lr = ts["batched"].last_restore
    assert (lr["h2d_copy_calls"], lr["placement_ops"]) == (5, math.ceil(5 / group))
    assert lr["device_scratch_bytes"] == min(5, group) * CB and lr["h2d_bytes"] == 5 * CB
    assert ts["baseline"].last_restore["h2d_copy_calls"] == 5 * 2 * LAYERS


def test_disk_only_reads_reuse_one_staging_slot_with_distinct_payloads(tmp_path):
    ts = twins(tmp_path, host_slots=1, disk_files=8)
    toks, cache, keys = store_both(ts, 6, seed=7)  # chunks 0-4 on disk, 5 in host
    assert [ts["batched"].tier_of(k) for k in keys] == ["disk"] * 5 + ["host"]
    chunk = [cache.layers[0].keys[:, :, i * C:(i + 1) * C, :] for i in range(5)]
    assert all(not torch.equal(chunk[i], chunk[j]) for i in range(5) for j in range(i))  # distinct payloads
    out = restore_both(ts, toks, keys[:5], cache)
    assert out["batched"][1:] == (5 * C, ["disk"] * 5, None)
    assert ts["batched"].host.staging_slots == 1 and ts["batched"].counters.disk_file_bytes_read == 5 * FILE
    assert [ts["batched"].tier_of(k) for k in keys] == ["disk"] * 5 + ["host"]  # no promotion


def test_mixed_tiers_follow_the_stored_order(tmp_path):
    ts = twins(tmp_path, host_slots=2, disk_files=8, group=2)
    toks, cache, keys = store_both(ts, 5)  # disk, disk, disk, host, host
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][2] == ["disk"] * 3 + ["host"] * 2


@pytest.mark.parametrize("bad", [0, 2, 4])
def test_corrupt_chunk_stops_at_the_verified_prefix(tmp_path, bad):
    ts = twins(tmp_path, host_slots=1, disk_files=8, group=2)
    toks, cache, keys = store_both(ts, 6)  # chunks 0-4 on disk
    for t in ts.values():
        path = t.disk.root / f"{keys[bad].hex()}.kvc"
        raw = bytearray(path.read_bytes())
        raw[-6] ^= 0x40
        path.write_bytes(raw)
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][1:] == (bad * C, ["disk"] * bad, "disk_payload_crc")
    if bad == 0:
        assert out["batched"][0] is None  # zero-length valid prefix: no cache at all
    lr = ts["batched"].last_restore
    assert lr["chunks_loaded"] == bad and lr["placement_ops"] == math.ceil(bad / 2)
    assert keys[bad] not in ts["batched"].disk and all(keys[i] in ts["batched"].disk for i in range(bad + 1, 5))


def test_missing_middle_file_and_absent_middle_chunk(tmp_path):
    ts = twins(tmp_path, host_slots=1, disk_files=8)
    toks, cache, keys = store_both(ts, 5)  # chunks 0-3 on disk, 4 in host
    for t in ts.values():
        (t.disk.root / f"{keys[1].hex()}.kvc").unlink()
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][1:] == (C, ["disk"], "disk_file_missing")
    ts = twins(tmp_path / "absent", host_slots=8, disk_files=1)
    toks, cache, keys = store_both(ts, 4)
    for t in ts.values():
        t.host.discard(keys[2])
    out = restore_both(ts, toks, keys, cache)
    assert out["batched"][1:] == (2 * C, ["host"] * 2, "absent")


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_partial_plans_and_chunk_boundaries(tmp_path, k):
    ts = twins(tmp_path, host_slots=2, disk_files=8, group=3)
    toks, cache, keys = store_both(ts, 4)
    out = restore_both(ts, toks, keys[:k], cache)
    assert out["batched"][1] == k * C


def test_repeated_requests_with_eviction_never_change_an_earlier_cache(tmp_path):
    ts = twins(tmp_path, host_slots=2, disk_files=3, group=2)
    toks_a, cache_a, keys_a = store_both(ts, 3, seed=1, start=100)
    first = {b: t.restore(toks_a, keys_a, "cpu", None)[0] for b, t in ts.items()}
    saved = {b: kv_bytes(c) for b, c in first.items()}
    toks_b, cache_b, keys_b = store_both(ts, 3, seed=2, start=500)  # evicts A's chunks (host -> disk -> deleted)
    restore_both(ts, toks_b, keys_b, cache_b)
    restore_both(ts, toks_a, keys_a, cache_a)
    toks_c, cache_c, keys_c = store_both(ts, 2, seed=3, start=900)
    restore_both(ts, toks_c, keys_c, cache_c)
    for t in ts.values():  # overwrite every host slot and the staging slot
        t.host.slab.fill_(0xA5)
    for b, c in first.items():
        assert all(torch.equal(p, q) for x, y in zip(saved[b], kv_bytes(c)) for p, q in zip(x, y))
        assert_prefix_equal(c, cache_a, 3 * C)


def test_restored_tensors_own_exactly_their_bytes(tmp_path):
    ts = twins(tmp_path, host_slots=8, disk_files=1, group=2)
    toks, cache, keys = store_both(ts, 5)
    for b, t in ts.items():
        restored, L, _, _ = t.restore(toks, keys[:3], "cpu", None)  # destination planned for 3 of 5 chunks
        ptrs = set()
        for layer in restored.layers:
            for x in (layer.keys, layer.values):
                assert x.is_contiguous() and x.shape[2] == L
                assert x.untyped_storage().nbytes() == x.numel() * x.element_size()  # no retained capacity
                ptrs.add(x.untyped_storage().data_ptr())
        assert len(ptrs) == 2 * LAYERS  # no two layers share (or alias) a buffer


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
@pytest.mark.parametrize("fail_at", [1, 3])  # chunk 1 = disk (staging), chunk 3 = host (lease)
def test_copy_failure_releases_leases_and_staging_and_propagates(tmp_path, monkeypatch, backend, fail_at):
    t = make(tmp_path / "s", backend, host_slots=2, disk_files=8, group=2)
    toks = tokens(4 * C + 1)
    cache = toy_cache(NS, len(toks))
    keys = chain_keys(NS.digest(), toks, C)
    t.store(cache, toks, keys)  # disk, disk, host, host
    real, calls = t._typed, []

    def failing_typed(flat):  # called once per chunk copy by both backends
        calls.append(1)
        if len(calls) == fail_at + 1:
            raise RuntimeError("simulated device copy failure")
        return real(flat)
    monkeypatch.setattr(t, "_typed", failing_typed)
    before = t.counters.to_dict()
    with pytest.raises(RuntimeError, match="simulated"):
        t.restore(toks, keys, "cpu", None)
    assert not t.host._staging_busy and all(e.pins == 0 for e in t.host.entries.values())
    assert t.counters.recovered_prefix_tokens == before["recovered_prefix_tokens"]
    assert t.counters.bytes_h2d == before["bytes_h2d"]
    # hits count completed copies only: chunks before fail_at were copied, the failing one was not
    hits = (t.counters.host_hits - before["host_hits"], t.counters.disk_hits - before["disk_hits"])
    assert hits == {1: (0, 1), 3: (1, 2)}[fail_at]
    monkeypatch.setattr(t, "_typed", real)
    restored, L, sources, why = t.restore(toks, keys, "cpu", None)  # the store is still usable
    assert (L, sources, why) == (4 * C, ["disk", "disk", "host", "host"], None)
    assert_prefix_equal(restored, cache, L)


def test_backend_and_group_size_are_validated(tmp_path):
    with pytest.raises(ValueError, match="unknown restore backend"):
        make(tmp_path / "a", "fast", 1, 1, 2)
    with pytest.raises(ValueError, match="restore_staging_chunks"):
        make(tmp_path / "b", "batched", 1, 1, 0)
    t = make(tmp_path / "c", "batched", 2, 1, 2)
    toks, cache, keys = tokens(C + 1), toy_cache(NS, C + 1), chain_keys(NS.digest(), tokens(C + 1), C)
    t.store(cache, toks, keys)
    t.restore_backend = "nope"
    with pytest.raises(ValueError, match="unknown restore backend"):
        t.restore(toks, keys, "cpu", None)
    t.restore_backend = "baseline"  # switching between valid backends is allowed (A/B on one store)
    assert t.restore(toks, keys, "cpu", None)[1] == C
    cfg = t.restore_config()
    assert cfg["batched_max_scratch_bytes"] == 2 * CB and cfg["new_host_memory_bytes"] == 0


# ---- engine level: tiny random Qwen2 on CPU, both backends, strict resident replay
@pytest.fixture(scope="module")
def tiny_model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


def test_engine_sequences_are_bitwise_identical_across_backends(tiny_model, tmp_path):
    Ce, steps = 4, 4
    ns = namespace_for(tiny_model, "toy/tiny-qwen2", "0" * 40, Ce)
    cb, fsize = cf.payload_len(ns), cf.file_size(ns)
    prompts = [list(range(10, 23)), list(range(10, 27)), list(range(10, 19)) + [400, 401, 402],
               list(range(300, 313)), list(range(10, 31)), list(range(10, 23))]
    results = {}
    for b in RESTORE_BACKENDS:
        host = HostStore(3 * cb, cb, max_entries=16, pin=False)  # 2 slots -> eviction to disk
        disk = DiskStore(tmp_path / b, 4 * fsize, max_entries=16)
        t = TieredKVCache(ns, host, disk, restore_backend=b, restore_staging_chunks=2)
        eng, ref = AlwaysRestoreEngine(tiny_model, t), ResidentReference(tiny_model, ns.digest(), Ce)
        runs = []
        for p in prompts:
            calls = []
            hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))
            try:
                r = eng.run(p, steps)
            finally:
                hook.remove()
            assert len(calls) == steps
            ref_tokens, ref_steps = ref.run(p, r.restore_len, r.stored_chunks, steps)
            assert r.tokens == ref_tokens and all(torch.equal(x, y) for x, y in zip(r.step_logits, ref_steps))
            runs.append(r)
        results[b] = (runs, t.counters.to_dict())
    (rb, cb_), (rx, cx) = results["baseline"], results["batched"]
    assert cb_ == cx and cx["disk_hits"] > 0 and cx["host_hits"] > 0
    for a, b in zip(rb, rx):
        assert (a.restore_len, a.sources, a.miss_reason, a.stored_chunks, a.tokens) == \
            (b.restore_len, b.sources, b.miss_reason, b.stored_chunks, b.tokens)
        assert all(torch.equal(x, y) for x, y in zip(a.step_logits, b.step_logits))
