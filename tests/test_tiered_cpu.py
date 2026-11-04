"""TieredKVCache on CPU with synthetic caches (no model). device='cpu', pin=False."""

import errno

import pytest
import torch

from kvlab import chunk_format as cf
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import chain_keys
from kvlab.kv_snapshot import bytes_equal
from kvlab.tiered_cache import TieredKVCache
from m1_toy import deny_unlink, tokens, toy_cache, toy_ns

NS = toy_ns()
C, CB, FILE = NS.chunk_size, cf.payload_len(NS), cf.file_size(NS)


def make(tmp_path, host_slots, disk_files, write_fn=cf.default_write):
    host = HostStore(budget_bytes=(host_slots + 1) * CB, chunk_bytes=CB, max_entries=100, pin=False)
    disk = DiskStore(tmp_path / "disk", budget_bytes=disk_files * FILE, max_entries=100, write_fn=write_fn)
    return TieredKVCache(NS, host, disk)


def stored_and_keys(t, n_chunks, seed=0):
    toks = tokens(n_chunks * C + 1)
    cache = toy_cache(NS, len(toks), seed)
    keys = chain_keys(NS.digest(), toks, C)
    return toks, cache, keys, t.store(cache, toks, keys)


def assert_prefix_equal(restored, source, L):
    for a, b in zip(restored.layers, source.layers):
        assert bytes_equal(a.keys, b.keys[:, :, :L, :]) and bytes_equal(a.values, b.values[:, :, :L, :])


def test_host_only_round_trip(tmp_path):
    t = make(tmp_path, host_slots=4, disk_files=4)
    toks, cache, keys, stored = stored_and_keys(t, 3)
    assert stored == [0, 1, 2] and t.counters.bytes_d2h == 3 * CB
    restored, L, sources, why = t.restore(toks, keys, "cpu", None)
    assert (L, sources, why) == (3 * C, ["host"] * 3, None)
    assert_prefix_equal(restored, cache, L)
    assert t.counters.host_hits == 3 and t.counters.recovered_prefix_tokens == 3 * C


def test_eviction_spills_to_disk_and_tiers_are_exclusive(tmp_path):
    t = make(tmp_path, host_slots=1, disk_files=4)
    toks, cache, keys, stored = stored_and_keys(t, 3)
    assert stored == [0, 1, 2]
    tiers = [t.tier_of(k) for k in keys]
    assert tiers == ["disk", "disk", "host"]
    assert all((k in t.host) != (k in t.disk) for k in keys)  # exactly one tier each
    assert t.counters.host_spills_to_disk == 2 and t.host.evictions == 2
    restored, L, sources, _ = t.restore(toks, keys, "cpu", None)
    assert L == 3 * C and sources == ["disk", "disk", "host"]
    assert_prefix_equal(restored, cache, L)
    assert t.counters.disk_hits == 2 and t.counters.disk_file_bytes_read == 2 * FILE
    assert [t.tier_of(k) for k in keys] == tiers  # disk hits are not promoted


def test_invalid_middle_chunk_stops_contiguous_prefix(tmp_path):
    t = make(tmp_path, host_slots=1, disk_files=4)
    toks, cache, keys, _ = stored_and_keys(t, 4)  # chunks 0-2 on disk, 3 in host
    path = t.disk.root / f"{keys[1].hex()}.kvc"
    raw = bytearray(path.read_bytes())
    raw[-6] ^= 0x40
    path.write_bytes(raw)
    restored, L, sources, why = t.restore(toks, keys, "cpu", None)
    assert (L, sources, why) == (C, ["disk"], "disk_payload_crc")
    assert_prefix_equal(restored, cache, L)
    assert keys[1] not in t.disk and keys[2] in t.disk and keys[3] in t.host  # later chunks unused
    assert t.counters.miss_reasons == {"disk_payload_crc": 1}
    assert not t.host._staging_busy and all(e.pins == 0 for e in t.host.entries.values())


def test_absent_first_chunk_means_full_recompute(tmp_path):
    t = make(tmp_path, host_slots=2, disk_files=2)
    toks = tokens(9)
    assert t.restore(toks, chain_keys(NS.digest(), toks, C), "cpu", None) == (None, 0, [], "absent")


def test_spill_failure_drops_chunk_without_affecting_store(tmp_path):
    def enospc(f, mv):
        raise OSError(errno.ENOSPC, "full")
    t = make(tmp_path, host_slots=1, disk_files=4, write_fn=enospc)
    toks, cache, keys, stored = stored_and_keys(t, 2)
    assert stored == [0, 1]
    assert t.tier_of(keys[0]) is None and t.tier_of(keys[1]) == "host"
    assert t.counters.spill_failures == {"write_oserror:ENOSPC": 1}
    assert t.disk.scan_dir_bytes() == (0, [])  # temp file cleaned up
    restored, L, sources, why = t.restore(toks, keys, "cpu", None)
    assert restored is None and L == 0 and why == "absent"


def test_all_pinned_skips_write_without_exceeding_budget(tmp_path):
    t = make(tmp_path, host_slots=1, disk_files=4)
    toks, cache, keys, _ = stored_and_keys(t, 1)
    toks2, cache2, keys2 = tokens(2 * C, start=900), toy_cache(NS, 2 * C, 5), None
    keys2 = chain_keys(NS.digest(), toks2, C)
    with t.host.lease(keys[0]):
        before = (t.host.resident_bytes, t.host.slab_bytes)
        assert t.store(cache2, toks2, keys2) == []
        assert (t.host.resident_bytes, t.host.slab_bytes) == before and t.host.slab_bytes <= t.host.budget_bytes
    assert t.counters.write_skips == {"host_all_pinned": 2}


def test_pins_released_and_error_propagates_when_restore_copy_fails(tmp_path, monkeypatch):
    t = make(tmp_path, host_slots=1, disk_files=4)
    toks, cache, keys, _ = stored_and_keys(t, 3)  # disk, disk, host
    real = t._typed
    calls = []

    def failing_typed(flat):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated device copy failure")
        return real(flat)
    monkeypatch.setattr(t, "_typed", failing_typed)
    with pytest.raises(RuntimeError, match="simulated"):
        t.restore(toks, keys, "cpu", None)
    assert not t.host._staging_busy and all(e.pins == 0 for e in t.host.entries.values())


def test_host_identity_mismatch_raises(tmp_path):
    t = make(tmp_path, host_slots=4, disk_files=1)
    toks, cache, keys, _ = stored_and_keys(t, 2)
    e = t.host.entries[keys[1]]
    e.ident = e.ident.__class__(e.ident.ns_digest, e.ident.key, e.ident.parent, e.ident.index, (0,) * C)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        t.restore(toks, keys, "cpu", None)
    assert all(x.pins == 0 for x in t.host.entries.values())


def test_wrong_cache_layout_raises_instead_of_storing(tmp_path):
    t = make(tmp_path, host_slots=4, disk_files=1)
    toks = tokens(2 * C)
    keys = chain_keys(NS.digest(), toks, C)
    bad = toy_cache(toy_ns(kv_dtype="float16"), 2 * C)
    with pytest.raises(ValueError, match="does not match namespace"):
        t.store(bad, toks, keys)
    assert len(t.host) == 0


def test_store_skips_chunks_already_cached_in_any_tier(tmp_path):
    t = make(tmp_path, host_slots=1, disk_files=4)
    toks, cache, keys, _ = stored_and_keys(t, 2)
    assert t.store(cache, toks, keys) == []
    assert t.counters.chunks_stored == 2


def test_budget_accounting_reported(tmp_path):
    t = make(tmp_path, host_slots=1, disk_files=2)
    stored_and_keys(t, 4)
    b = t.budgets()
    h, d = b["host"], b["disk"]
    assert h["slab_allocated_bytes"] == 2 * CB <= h["budget_bytes"]  # 1 slot + 1 staging, allocated at init
    assert (h["staging_reserved_bytes"], h["resident_logical_bytes"], h["resident_logical_peak_bytes"]) == (CB, CB, CB)
    assert h["resident_logical_bytes"] + h["staging_reserved_bytes"] <= h["slab_allocated_bytes"]
    assert h["cuda_host_registered_bytes"] == 0 and "not measured" in h["metadata_note"]
    assert d["peak_used_bytes"] <= d["budget_bytes"]
    assert d["dir_bytes_scanned"] == d["committed_bytes"] == d["used_bytes"] == 2 * FILE
    assert (d["tmp_bytes"], d["undeletable_bytes"], d["unlink_failures"]) == (0, 0, {})
    assert d["evictions"] == 1  # 3 spills into a 2-file disk


def test_spill_whose_disk_eviction_fails_is_a_reported_write_skip(tmp_path, monkeypatch):
    """M1.1: an undeletable disk victim must not abort store() (the request already has its result)."""
    t = make(tmp_path, host_slots=1, disk_files=1)
    toks = tokens(3 * C + 1)
    cache = toy_cache(NS, len(toks))
    keys = chain_keys(NS.digest(), toks, C)
    victim = f"{keys[0].hex()}.kvc"
    attempts = deny_unlink(monkeypatch, lambda n: n == victim)
    # chunk 0 -> host; chunk 1 spills chunk 0 to disk; chunk 2 spills chunk 1, which needs chunk 0's file evicted.
    assert t.store(cache, toks, keys) == [0, 1, 2]
    assert attempts == [victim] and t.counters.spill_failures == {"evict_oserror:EACCES": 1}
    assert [t.tier_of(k) for k in keys] == [None, None, "host"]  # chunk 1 dropped; chunk 0 no longer readable
    assert t.disk.undeletable == {victim: FILE} and t.disk.used_bytes == t.disk.scan_dir_bytes()[0] == FILE
    assert t.restore(toks, keys, "cpu", None) == (None, 0, [], "absent")
    b = t.budgets()["disk"]
    assert (b["undeletable_bytes"], b["undeletable_files"], b["unlink_failures"]) == (FILE, [victim], {"evict:EACCES": 1})
