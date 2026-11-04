"""Tiered (pinned host + disk) prefix KV store: synchronous store and restore
(docs/design.md §6–8). Tiers are exclusive: a chunk is in host XOR disk.

Restore backends (docs/M3_DESIGN.md, decisions D30). Both run the same lookup, lease,
staging, validation, counter and miss logic; they differ only in how a verified chunk's
bytes reach the device:
  baseline - M1 copies: 56 per-layer blocking H2D copies per chunk into 56 per-layer
             destination tensors (the default).
  batched  - one blocking contiguous H2D per chunk into a chunk-major device staging buffer
             of at most `restore_staging_chunks` chunks, then one permuted device copy per
             group into a single layer-major destination tensor.
Both build the returned cache with the public DynamicCache constructor (D4), which copies
exactly [0, L) into fresh tensors. No copy is non_blocking, so every transfer out of a host
slot or the staging slot has finished before its lease or the staging slot is released.
"""

from collections import Counter
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kvlab import chunk_format as cf
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, Namespace

RESTORE_BACKENDS = ("baseline", "batched")


@dataclass
class Counters:
    requests: int = 0
    restored_requests: int = 0
    full_recompute_requests: int = 0
    host_hits: int = 0
    disk_hits: int = 0
    chunk_misses: int = 0
    miss_reasons: Counter = field(default_factory=Counter)
    recovered_prefix_tokens: int = 0
    chunks_stored: int = 0
    write_skips: Counter = field(default_factory=Counter)
    host_spills_to_disk: int = 0
    spill_failures: Counter = field(default_factory=Counter)
    bytes_d2h: int = 0
    bytes_h2d: int = 0
    disk_file_bytes_read: int = 0      # file reads; may be served by the OS page cache
    disk_file_bytes_written: int = 0

    def to_dict(self) -> dict:
        return {k: dict(v) if isinstance(v, Counter) else v for k, v in self.__dict__.items()}


class _BaselineDest:
    """M1 restore destination: per-layer K and V tensors [1, kv_heads, n*C, head_dim];
    each chunk arrives as 2 * num_layers blocking H2D copies (unchanged M1 code)."""

    def __init__(self, tiered, n: int, device):
        ns, C = tiered.ns, tiered.C
        self.tiered, self.C = tiered, C
        self.dest = [(torch.empty((1, ns.num_kv_heads, n * C, ns.head_dim), dtype=tiered.dtype, device=device),
                      torch.empty((1, ns.num_kv_heads, n * C, ns.head_dim), dtype=tiered.dtype, device=device))
                     for _ in range(ns.num_layers)]
        self.copy_calls = 0
        self.scratch_bytes = 0      # no explicit scratch (PyTorch makes a per-copy temporary when n > 1)
        self.placement_ops = 0

    def put(self, flat, i: int) -> None:
        C = self.C
        src = self.tiered._typed(flat)
        for li, (k_dev, v_dev) in enumerate(self.dest):
            k_dev[:, :, i * C:(i + 1) * C, :].copy_(src[li, 0])  # H2D from pinned memory, synchronous
            v_dev[:, :, i * C:(i + 1) * C, :].copy_(src[li, 1])
        self.copy_calls += 2 * len(self.dest)

    def finish(self, loaded: int) -> None:
        pass

    def layer_views(self, L: int) -> list:
        return [(k[:, :, :L, :], v[:, :, :L, :]) for k, v in self.dest]


class _BatchedDest:
    """M3 restore destination (docs/M3_DESIGN.md §3 option b).
    X: chunk-major device staging [g, layer, K/V, 1, kv_heads, C, head_dim], g <= G chunks.
    Y: layer-major destination [layer, K/V, 1, kv_heads, n*C, head_dim].
    put(): one contiguous blocking H2D of the whole chunk into the next free X row. When X is
    full (or at finish), one copy_ of X permuted to [layer, K/V, 1, kv_heads, g, C, head_dim]
    into Y[..., s:s+g*C, :] places the group. All work is on the current stream; X is only
    rewritten by later memcpys, which are stream-ordered after the placement that reads it."""

    def __init__(self, tiered, n: int, device, group: int):
        ns, C = tiered.ns, tiered.C
        self.tiered, self.C = tiered, C
        shape = (ns.num_layers, 2, 1, ns.num_kv_heads)
        self.Y = torch.empty((*shape, n * C, ns.head_dim), dtype=tiered.dtype, device=device)
        g = min(n, group)
        self.X = torch.empty((g, *shape, C, ns.head_dim), dtype=tiered.dtype, device=device)
        self.next_chunk = 0   # first chunk index of the group currently in X
        self.filled = 0       # chunks of that group already in X
        self.copy_calls = 0
        self.scratch_bytes = self.X.numel() * self.X.element_size()
        self.placement_ops = 0

    def put(self, flat, i: int) -> None:
        if i != self.next_chunk + self.filled:
            raise RuntimeError(f"chunk {i} out of order (expected {self.next_chunk + self.filled})")
        self.X[self.filled].copy_(self.tiered._typed(flat))  # contiguous H2D, blocking (memcpy + stream sync)
        self.copy_calls += 1
        self.filled += 1
        if self.filled == self.X.shape[0]:
            self._place()

    def _place(self) -> None:
        g = self.filled
        if g == 0:
            return
        s, C = self.next_chunk * self.C, self.C
        self.Y[..., s:s + g * C, :].unflatten(4, (g, C)).copy_(self.X[:g].permute(1, 2, 3, 4, 0, 5, 6))
        self.placement_ops += 1
        self.next_chunk += g
        self.filled = 0

    def finish(self, loaded: int) -> None:
        self._place()
        if self.next_chunk != loaded:
            raise RuntimeError(f"placed {self.next_chunk} chunks, loaded {loaded}")

    def layer_views(self, L: int) -> list:
        return [(self.Y[li, 0, :, :, :L, :], self.Y[li, 1, :, :, :L, :]) for li in range(self.Y.shape[0])]


class TieredKVCache:
    def __init__(self, ns: Namespace, host: HostStore, disk: DiskStore | None,
                 restore_backend: str = "baseline", restore_staging_chunks: int = 4):
        self.ns, self.host, self.disk = ns, host, disk
        self.ns_digest = ns.digest()
        self.C = ns.chunk_size
        self.dtype = getattr(torch, ns.kv_dtype)
        self.chunk_bytes = cf.payload_len(ns)
        if host.chunk_bytes != self.chunk_bytes:
            raise ValueError(f"host chunk_bytes {host.chunk_bytes} != payload {self.chunk_bytes}")
        self.restore_backend = restore_backend
        self.restore_staging_chunks = restore_staging_chunks
        self._check_backend()
        self.counters = Counters()
        self.last_restore = None  # diagnostic copy/op/byte counts of the latest restore (not a counter)

    def _check_backend(self) -> None:
        if self.restore_backend not in RESTORE_BACKENDS:
            raise ValueError(f"unknown restore backend {self.restore_backend!r}; expected one of {RESTORE_BACKENDS}")
        if not isinstance(self.restore_staging_chunks, int) or self.restore_staging_chunks < 1:
            raise ValueError(f"restore_staging_chunks must be an int >= 1, got {self.restore_staging_chunks!r}")

    # ---- helpers
    def _typed(self, flat: torch.Tensor) -> torch.Tensor:
        """uint8 slot -> [layer, K/V, 1, kv_heads, C, head_dim] view (no copy)."""
        ns = self.ns
        return flat.view(self.dtype).view(ns.num_layers, 2, 1, ns.num_kv_heads, self.C, ns.head_dim)

    def tier_of(self, key: bytes) -> str | None:
        if key in self.host:
            return "host"
        if self.disk is not None and key in self.disk:
            return "disk"
        return None

    def _check_model_cache(self, cache: DynamicCache, num_tokens: int) -> None:
        ns = self.ns
        if len(cache.layers) != ns.num_layers:
            raise ValueError(f"cache has {len(cache.layers)} layers, namespace says {ns.num_layers}")
        want = (1, ns.num_kv_heads, ns.head_dim)
        for li, layer in enumerate(cache.layers):
            for t in (layer.keys, layer.values):
                got = (t.shape[0], t.shape[1], t.shape[3])
                if t.dtype != self.dtype or got != want or t.shape[2] < num_tokens:
                    raise ValueError(f"layer {li}: KV {tuple(t.shape)} {t.dtype} does not match namespace "
                                     f"{want} {self.dtype} with >= {num_tokens} tokens")

    def _spill(self, key: bytes, ident: ChunkIdentity, view: torch.Tensor) -> None:
        """Called by the host store before it reuses a slot. Failure drops the chunk."""
        if self.disk is None:
            self.counters.spill_failures["no_disk_tier"] += 1
            return
        ok, reason = self.disk.put(key, cf.encode_header(self.ns, ident), view.numpy())
        if ok:
            self.counters.host_spills_to_disk += 1
            if reason == "written":
                self.counters.disk_file_bytes_written += self.disk.entries[key]
        else:
            self.counters.spill_failures[reason] += 1

    # ---- store
    def store(self, cache: DynamicCache, token_ids, keys: list[bytes]) -> list[int]:
        """Copy every complete prompt chunk not already cached into host slots.
        Returns the chunk indices stored. Failures skip caching, never recompute."""
        self._check_model_cache(cache, len(keys) * self.C)
        stored = []
        for i, key in enumerate(keys):
            if self.tier_of(key) is not None:
                continue
            ident = ChunkIdentity.for_chunk(self.ns_digest, token_ids, self.C, i, keys)
            slot, reason = self.host.allocate(self._spill)
            if slot is None:
                self.counters.write_skips[reason] += 1
                continue
            try:
                view = self._typed(self.host.slot_view(slot))
                s, e = i * self.C, (i + 1) * self.C
                for li, layer in enumerate(cache.layers):
                    view[li, 0].copy_(layer.keys[:, :, s:e, :])  # D2H into pinned slot, synchronous
                    view[li, 1].copy_(layer.values[:, :, s:e, :])
            except BaseException:
                self.host.release_slot(slot)  # do not leak the slot; the error still propagates
                raise
            self.host.insert(key, ident, slot)
            self.counters.chunks_stored += 1
            self.counters.bytes_d2h += self.chunk_bytes
            stored.append(i)
        return stored

    # ---- restore
    def restore(self, token_ids, keys: list[bytes], device, config):
        """Restore the longest contiguous valid prefix of `keys` into a fresh
        DynamicCache. Returns (cache | None, L, sources, miss_reason)."""
        self.last_restore = None
        n = 0
        while n < len(keys) and self.tier_of(keys[n]) is not None:
            n += 1
        miss_reason = None
        if n < len(keys):
            miss_reason = "absent"
            self.counters.chunk_misses += 1
            self.counters.miss_reasons["absent"] += 1
        if n == 0:
            return None, 0, [], miss_reason

        ns, C = self.ns, self.C
        self._check_backend()
        if self.restore_backend == "batched":
            dest = _BatchedDest(self, n, device, self.restore_staging_chunks)
        else:
            dest = _BaselineDest(self, n, device)
        copy_into_dest = dest.put

        loaded, sources = 0, []
        for i in range(n):
            key = keys[i]
            ident = ChunkIdentity.for_chunk(self.ns_digest, token_ids, C, i, keys)
            if key in self.host:
                with self.host.lease(key) as (flat, stored_ident):
                    if stored_ident != ident:
                        raise RuntimeError(f"host entry {key.hex()[:12]} identity mismatch (in-process invariant)")
                    copy_into_dest(flat, i)
                sources.append("host")
                self.counters.host_hits += 1
            else:
                if self.host.staging_slots == 0:
                    ok, reason, nbytes = False, "no_staging_buffer", 0
                else:
                    with self.host.staging() as staging:
                        ok, reason, nbytes = self.disk.read_into(key, ns, ident, staging.numpy())
                        if ok:
                            copy_into_dest(staging, i)
                if not ok:
                    # Stop at the first invalid chunk: only the verified prefix [0, i*C) is used.
                    miss_reason = reason
                    self.counters.chunk_misses += 1
                    self.counters.miss_reasons[reason] += 1
                    break
                sources.append("disk")
                self.counters.disk_hits += 1
                self.counters.disk_file_bytes_read += nbytes
            loaded += 1
        dest.finish(loaded)  # batched: place the last (possibly partial) group of verified chunks
        self.last_restore = {"backend": self.restore_backend, "chunks_planned": n, "chunks_loaded": loaded,
                             "h2d_copy_calls": dest.copy_calls, "h2d_bytes": loaded * self.chunk_bytes,
                             "placement_ops": dest.placement_ops, "device_destination_bytes": n * self.chunk_bytes,
                             "device_scratch_bytes": dest.scratch_bytes}
        if loaded == 0:
            return None, 0, sources, miss_reason
        L = loaded * C
        cache = DynamicCache(ddp_cache_data=dest.layer_views(L), config=config)  # copies [0, L) (D4)
        del dest
        if len(cache.layers) != ns.num_layers or any(layer.get_seq_length() != L for layer in cache.layers):
            raise RuntimeError("restored cache has wrong layer count or length")
        self.counters.recovered_prefix_tokens += L
        self.counters.bytes_h2d += loaded * self.chunk_bytes
        return cache, L, sources, miss_reason

    def restore_config(self) -> dict:
        """Restore backend and its explicit transient device-memory bounds (per restore of n chunks:
        destination n * chunk_bytes for both backends; batched scratch min(n, G) * chunk_bytes)."""
        return {"restore_backend": self.restore_backend, "restore_staging_chunks": self.restore_staging_chunks,
                "chunk_bytes": self.chunk_bytes,
                "batched_max_scratch_bytes": self.restore_staging_chunks * self.chunk_bytes,
                "new_host_memory_bytes": 0}

    def budgets(self) -> dict:
        """Host: the slab is allocated (and registered) once at init; resident bytes
        are logical occupancy inside it, not extra memory. Disk: used = readable
        entries + temp reservations + undeletable files. See design §6."""
        h, d = self.host, self.disk
        out = {"host": {"budget_bytes": h.budget_bytes,
                        "slab_allocated_bytes": h.slab_bytes,
                        "cuda_host_registered_bytes": h.registered_bytes,
                        "staging_reserved_bytes": h.staging_bytes,
                        "resident_slots": h.slots,
                        "resident_logical_bytes": h.resident_bytes,
                        "resident_logical_peak_bytes": h.peak_resident_bytes,
                        "entries": len(h), "max_entries": h.max_entries, "evictions": h.evictions,
                        "metadata_bytes_estimate": h.metadata_bytes_estimate(),
                        "metadata_note": "estimate of raw identity field bytes; not measured Python heap usage",
                        "pinned": h.pinned}}
        if d is not None:
            out["disk"] = {"budget_bytes": d.budget_bytes, "used_bytes": d.used_bytes,
                           "peak_used_bytes": d.peak_bytes, "committed_bytes": d.committed_bytes,
                           "tmp_bytes": d.tmp_bytes, "undeletable_bytes": d.undeletable_bytes,
                           "undeletable_files": sorted(d.undeletable), "unlink_failures": dict(d.unlink_failures),
                           "entries": len(d), "max_entries": d.max_entries, "evictions": d.evictions,
                           "dir_bytes_scanned": d.scan_dir_bytes()[0]}
        return out
