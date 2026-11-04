import pytest

from kvlab.host_store import HostStore
from m1_toy import ident, tokens, toy_ns

CB = 256  # toy chunk bytes


def _ident(i):
    ns = toy_ns()
    return ident(ns, tokens(4 * (i + 1)), i)


def _fill(h, n, spill=lambda *a: None):
    ids = []
    for i in range(n):
        slot, why = h.allocate(spill)
        assert why == "ok"
        idt = _ident(i)
        h.insert(idt.key, idt, slot)
        ids.append(idt)
    return ids


def test_capacity_accounting_includes_staging():
    """Slab (allocated once), staging (reserved) and resident (logical occupancy) are separate figures."""
    h = HostStore(budget_bytes=3 * CB + 100, chunk_bytes=CB, max_entries=100, pin=False)
    ptr = h.slab.data_ptr()
    assert (h.slots, h.staging_slots, h.slab_bytes, h.slab.numel()) == (2, 1, 3 * CB, 3 * CB)
    assert h.slab_bytes <= h.budget_bytes
    assert (h.staging_bytes, h.resident_bytes, h.peak_resident_bytes) == (CB, 0, 0)
    assert h.registered_bytes == 0 and not h.pinned  # pin=False: nothing registered with CUDA
    _fill(h, 2)
    assert (h.resident_bytes, h.peak_resident_bytes) == (2 * CB, 2 * CB)
    assert h.resident_bytes + h.staging_bytes == h.slab_bytes  # occupancy lives inside the slab
    assert h.metadata_bytes_estimate() == 2 * (32 * 3 + 8 + 4 * 4)
    slot, why = h.allocate(lambda *a: None)  # evicts one entry
    assert why == "ok" and h.resident_bytes == CB and h.peak_resident_bytes == 2 * CB
    h.release_slot(slot)
    assert (h.slab_bytes, h.slab.numel(), h.slab.data_ptr()) == (3 * CB, 3 * CB, ptr)  # never resized


def test_max_entries_bounds_slots():
    h = HostStore(budget_bytes=100 * CB, chunk_bytes=CB, max_entries=3, pin=False)
    assert h.slots == 3 and h.slab_bytes == 4 * CB


def test_lru_eviction_order_follows_leases():
    spilled = []
    h = HostStore(budget_bytes=3 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    a, b = _fill(h, 2)
    with h.lease(a.key):  # touch a -> b becomes least recently used
        pass
    slot, why = h.allocate(lambda k, i, v: spilled.append(k))
    assert why == "ok" and spilled == [b.key] and b.key not in h and a.key in h
    assert h.evictions == 1


def test_pinned_entries_are_never_evicted_and_all_pinned_skips():
    spilled = []
    h = HostStore(budget_bytes=3 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    a, b = _fill(h, 2)
    with h.lease(a.key), h.lease(b.key):
        before = (h.resident_bytes, h.slab_bytes, h.slab.data_ptr())
        assert h.allocate(lambda *x: spilled.append(x)) == (None, "host_all_pinned")
        assert spilled == [] and (h.resident_bytes, h.slab_bytes, h.slab.data_ptr()) == before and len(h) == 2
    with h.lease(a.key):
        slot, why = h.allocate(lambda k, i, v: spilled.append(k))
        assert why == "ok" and spilled == [b.key]  # only the unpinned one


def test_lease_released_on_exception():
    h = HostStore(budget_bytes=2 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    (a,) = _fill(h, 1)
    with pytest.raises(RuntimeError, match="boom"):
        with h.lease(a.key):
            raise RuntimeError("boom")
    assert h.entries[a.key].pins == 0
    with pytest.raises(RuntimeError, match="boom"):
        with h.staging():
            raise RuntimeError("boom")
    with h.staging():  # not left busy
        pass


def test_spill_exception_propagates_and_entry_is_kept():
    h = HostStore(budget_bytes=2 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    (a,) = _fill(h, 1)

    def bad_spill(*_):
        raise TypeError("bug in spill")
    with pytest.raises(TypeError):
        h.allocate(bad_spill)
    assert a.key in h and len(h) == 1


def test_oversized_chunk_bypassed_without_allocation():
    h = HostStore(budget_bytes=CB - 1, chunk_bytes=CB, max_entries=10, pin=False)
    assert (h.slots, h.staging_slots, h.slab_bytes, h.slab.numel()) == (0, 0, 0, 0)
    assert h.allocate(lambda *a: None) == (None, "oversized_for_host")
    with pytest.raises(RuntimeError):
        with h.staging():
            pass


def test_budget_for_staging_only_has_no_slots():
    h = HostStore(budget_bytes=CB + 10, chunk_bytes=CB, max_entries=10, pin=False)
    assert (h.slots, h.staging_slots) == (0, 1)
    assert h.allocate(lambda *a: None) == (None, "host_no_slots")


def test_release_slot_and_double_insert_are_checked():
    h = HostStore(budget_bytes=3 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    slot, _ = h.allocate(lambda *a: None)
    h.release_slot(slot)
    with pytest.raises(ValueError):
        h.release_slot(slot)
    (a,) = _fill(h, 1)
    with pytest.raises(KeyError):
        h.insert(a.key, a, 1)
    with pytest.raises(ValueError):
        h.insert(b"\0" * 32, a, 1)


def test_discard_frees_the_slot_and_refuses_pinned_entries():
    h = HostStore(budget_bytes=3 * CB, chunk_bytes=CB, max_entries=10, pin=False)
    a, b = _fill(h, 2)
    with h.lease(a.key):
        with pytest.raises(RuntimeError, match="pinned"):
            h.discard(a.key)
    h.discard(a.key)
    assert a.key not in h and len(h) == 1 and h.resident_bytes == CB
    slot, why = h.allocate(lambda *x: pytest.fail("must not spill: a slot is free"))
    assert why == "ok"
    with pytest.raises(KeyError):
        h.discard(a.key)
