"""Tiered (pinned host + disk) prefix KV store: synchronous store and restore
(docs/design.md §6–8). Tiers are exclusive: a chunk is in host XOR disk."""

from collections import Counter
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kvlab import chunk_format as cf
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, Namespace


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


class TieredKVCache:
    def __init__(self, ns: Namespace, host: HostStore, disk: DiskStore | None):
        self.ns, self.host, self.disk = ns, host, disk
        self.ns_digest = ns.digest()
        self.C = ns.chunk_size
        self.dtype = getattr(torch, ns.kv_dtype)
        self.chunk_bytes = cf.payload_len(ns)
        if host.chunk_bytes != self.chunk_bytes:
            raise ValueError(f"host chunk_bytes {host.chunk_bytes} != payload {self.chunk_bytes}")
        self.counters = Counters()

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
        dest = [(torch.empty((1, ns.num_kv_heads, n * C, ns.head_dim), dtype=self.dtype, device=device),
                 torch.empty((1, ns.num_kv_heads, n * C, ns.head_dim), dtype=self.dtype, device=device))
                for _ in range(ns.num_layers)]

        def copy_into_dest(flat, i):
            src = self._typed(flat)
            for li, (k_dev, v_dev) in enumerate(dest):
                k_dev[:, :, i * C:(i + 1) * C, :].copy_(src[li, 0])  # H2D from pinned memory, synchronous
                v_dev[:, :, i * C:(i + 1) * C, :].copy_(src[li, 1])

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
        if loaded == 0:
            return None, 0, sources, miss_reason
        L = loaded * C
        cache = DynamicCache(ddp_cache_data=[(k[:, :, :L, :], v[:, :, :L, :]) for k, v in dest], config=config)
        del dest
        if len(cache.layers) != ns.num_layers or any(layer.get_seq_length() != L for layer in cache.layers):
            raise RuntimeError("restored cache has wrong layer count or length")
        self.counters.recovered_prefix_tokens += L
        self.counters.bytes_h2d += loaded * self.chunk_bytes
        return cache, L, sources, miss_reason

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
