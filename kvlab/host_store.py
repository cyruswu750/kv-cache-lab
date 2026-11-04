"""Bounded pinned-host chunk store (docs/design.md §6).

One slab of exactly (slots + staging) * chunk_bytes is allocated once at
construction and, with pin=True, registered with cudaHostRegister (bypassing
PyTorch's caching pinned allocator, which rounds sizes up and keeps freed
blocks; decisions D16). The slab is never resized, so slab_bytes <= budget is
the host memory this store holds from start to close().

Accounting, reported separately (none of these is a measurement of RSS):
- slab_bytes        requested size of the one slab, allocated at init (constant)
- registered_bytes  bytes passed to a successful cudaHostRegister (0 if pin=False)
- staging_bytes     the part of the slab reserved as the staging buffer (constant)
- resident_bytes    logical occupancy: entries * chunk_bytes, inside the slab;
                    it is not additional memory (peak: peak_resident_bytes)
- metadata_bytes_estimate()  estimate of identity field bytes, not measured heap
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from kvlab.identity import ChunkIdentity


@dataclass
class HostEntry:
    slot: int
    ident: ChunkIdentity
    pins: int = 0


class HostStore:
    def __init__(self, budget_bytes: int, chunk_bytes: int, max_entries: int, pin: bool = True):
        if chunk_bytes <= 0 or budget_bytes < 0 or max_entries < 0:
            raise ValueError("chunk_bytes must be > 0; budget_bytes and max_entries >= 0")
        self.budget_bytes, self.chunk_bytes, self.max_entries = budget_bytes, chunk_bytes, max_entries
        self.staging_slots = 1 if budget_bytes >= chunk_bytes else 0
        self.slots = min(max_entries, (budget_bytes - chunk_bytes) // chunk_bytes) if self.staging_slots else 0
        self.slab_bytes = (self.slots + self.staging_slots) * chunk_bytes
        assert self.slab_bytes <= budget_bytes
        self.slab = torch.empty(self.slab_bytes, dtype=torch.uint8)  # allocated once, never resized
        self.pinned = False
        if pin and self.slab_bytes > 0:
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.slab.data_ptr(), self.slab_bytes, 0)
            if err != cudart.cudaError.success:
                raise RuntimeError(f"cudaHostRegister({self.slab_bytes} B) failed: {err}")
            self.pinned = True
        self._free = list(range(self.slots - 1, -1, -1))  # pop() yields slot 0 first
        self.entries: OrderedDict[bytes, HostEntry] = OrderedDict()  # least recently used first
        self._staging_busy = False
        self.evictions = 0
        self.peak_resident_bytes = 0

    # ---- accounting (see module docstring)
    @property
    def registered_bytes(self) -> int:
        return self.slab_bytes if self.pinned else 0

    @property
    def staging_bytes(self) -> int:
        return self.staging_slots * self.chunk_bytes

    @property
    def resident_bytes(self) -> int:
        return len(self.entries) * self.chunk_bytes

    def metadata_bytes_estimate(self) -> int:
        """ESTIMATE, not a measurement: raw identity field sizes per entry
        (3 x 32-byte keys, 8 for the index, 4 per token id). Python object
        overhead (dicts, tuples, ints) is larger and is not counted."""
        return sum(32 * 3 + 8 + 4 * len(e.ident.tokens) for e in self.entries.values())

    # ---- access
    def __contains__(self, key: bytes) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def slot_view(self, slot: int) -> torch.Tensor:
        if not 0 <= slot < self.slots:
            raise IndexError(f"slot {slot} out of range")
        return self.slab[slot * self.chunk_bytes:(slot + 1) * self.chunk_bytes]

    @contextmanager
    def lease(self, key: bytes):
        """Pin an entry for the duration of a restore; always unpinned on exit."""
        entry = self.entries[key]
        entry.pins += 1
        self.entries.move_to_end(key)  # LRU touch
        try:
            yield self.slot_view(entry.slot), entry.ident
        finally:
            entry.pins -= 1

    @contextmanager
    def staging(self):
        """The single project-managed staging buffer (for disk reads)."""
        if not self.staging_slots:
            raise RuntimeError("host budget has no room for a staging buffer")
        if self._staging_busy:
            raise RuntimeError("staging buffer already in use")
        self._staging_busy = True
        try:
            yield self.slab[self.slots * self.chunk_bytes:(self.slots + 1) * self.chunk_bytes]
        finally:
            self._staging_busy = False

    # ---- allocation / eviction
    def allocate(self, spill):
        """Return (slot, "ok") or (None, reason). Evicts the least recently used
        unpinned entry if needed, calling spill(key, ident, view) synchronously
        before its slot is reused. Never exceeds the slot count."""
        if self.slots == 0:
            return None, "oversized_for_host" if self.staging_slots == 0 else "host_no_slots"
        if not self._free:
            victim = next((k for k, e in self.entries.items() if e.pins == 0), None)
            if victim is None:
                return None, "host_all_pinned"
            entry = self.entries[victim]
            spill(victim, entry.ident, self.slot_view(entry.slot))
            del self.entries[victim]
            self._free.append(entry.slot)
            self.evictions += 1
        return self._free.pop(), "ok"

    def insert(self, key: bytes, ident: ChunkIdentity, slot: int) -> None:
        if key in self.entries:
            raise KeyError("chunk already resident")
        if key != ident.key:
            raise ValueError("key does not match identity")
        self.entries[key] = HostEntry(slot, ident)
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def discard(self, key: bytes) -> None:
        """Remove an unpinned entry and free its slot without spilling it
        (used to reset benchmark state between measurements)."""
        entry = self.entries[key]
        if entry.pins:
            raise RuntimeError("cannot discard a pinned entry")
        del self.entries[key]
        self._free.append(entry.slot)

    def release_slot(self, slot: int) -> None:
        """Return an allocated-but-not-inserted slot."""
        if slot in self._free or any(e.slot == slot for e in self.entries.values()):
            raise ValueError(f"slot {slot} is not allocated")
        self._free.append(slot)

    def close(self) -> None:
        if self.pinned:
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostUnregister(self.slab.data_ptr())
            if err != cudart.cudaError.success:
                raise RuntimeError(f"cudaHostUnregister failed: {err}")
            self.pinned = False
        self.entries.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
