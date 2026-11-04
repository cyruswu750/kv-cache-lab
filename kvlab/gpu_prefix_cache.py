"""Bounded device-resident prefix KV cache: the GPU-only baseline of M4 (configuration C1).

This is a performance *baseline*, not correctness machinery. `kvlab.resident_reference`
(an unbounded oracle that replays another engine's decisions) stays separate and is used to
check this class. See docs/design.md §12 and decisions D32.

Policy (the same rules as the tiered engine, so that only the storage location differs):
- **Identity:** chunk keys from `kvlab.identity.chain_keys` under this cache's namespace digest.
  On every hit, the stored `ChunkIdentity` is compared with the expected one; a mismatch is an
  in-process invariant violation and raises (as for the host tier, design §3). It is never a
  miss.
- **Lookup:** the longest run of leading eligible chunks (first-token rule, `kvlab.boundaries`;
  the caller passes only the eligible keys) that are present. The hit chunks are touched in
  order (LRU).
- **Store:** after prefill, every complete prompt chunk not already present is copied into an
  owned tensor, in chunk order, evicting the least recently used entry when full. A present
  chunk is not touched by storing. Capacity = budget // chunk_bytes; with capacity 0 nothing is
  stored (`write_skips["no_capacity"]`).

Ownership:
- Each entry is one contiguous tensor of shape [layers, 2 (K, V), 1, kv_heads, C, head_dim]
  (the chunk payload layout, design §4) and dtype, allocated by this class and filled by
  `copy_`. It is never a view of a request's cache, so it retains exactly `chunk_bytes` and
  keeps no request storage alive. `retained_bytes()` measures the actual storages.
- A hit gathers the chunks with `torch.cat` into new tensors (temporary scratch), and the public
  `DynamicCache` constructor copies them again (D4). The request therefore never receives a
  stored tensor, and continuing the request (suffix prefill, decode) cannot modify the cache.

Device-agnostic: the device is the model's (CUDA in GPU experiments, CPU in the M4-A tests). The
M4-A CPU tests check correctness only; they are not GPU performance evidence.
"""

from collections import Counter, OrderedDict

import torch
from transformers import DynamicCache

from kvlab import chunk_format as cf
from kvlab.identity import ChunkIdentity, Namespace


def same_device(a, b) -> bool:
    """Equal device type, and equal index when both name one ("cuda" means the current device).
    Never queries CUDA."""
    a, b = torch.device(a), torch.device(b)
    return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)


class GPUPrefixCache:
    def __init__(self, ns: Namespace, budget_bytes: int, device):
        if not isinstance(budget_bytes, int) or isinstance(budget_bytes, bool) or budget_bytes < 0:
            raise ValueError(f"budget_bytes must be a non-negative int, got {budget_bytes!r}")
        self.ns, self.ns_digest, self.C = ns, ns.digest(), ns.chunk_size
        self.device = torch.device(device)
        self.dtype = getattr(torch, ns.kv_dtype)
        self.chunk_bytes = cf.payload_len(ns)
        self.budget_bytes = budget_bytes
        self.capacity_chunks = budget_bytes // self.chunk_bytes
        self.shape = (ns.num_layers, 2, 1, ns.num_kv_heads, self.C, ns.head_dim)
        self.entries: OrderedDict[bytes, tuple] = OrderedDict()  # key -> (identity, tensor); LRU first
        self.counters = Counter()
        self.peak_retained_bytes = 0
        self.last_restore = None  # {"chunks", "scratch_bytes"} of the latest lookup (diagnostic)

    def __contains__(self, key: bytes) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def retained_bytes(self) -> int:
        """Bytes of the storages the entries actually hold (each storage counted once)."""
        storages = {}
        for _, t in self.entries.values():
            s = t.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
        return sum(storages.values())

    def _check_model_cache(self, cache: DynamicCache, num_tokens: int) -> None:
        want = (1, self.ns.num_kv_heads, self.ns.head_dim)
        if len(cache.layers) != self.ns.num_layers:
            raise ValueError(f"cache has {len(cache.layers)} layers, namespace says {self.ns.num_layers}")
        for li, layer in enumerate(cache.layers):
            for t in (layer.keys, layer.values):
                if t.dtype != self.dtype or (t.shape[0], t.shape[1], t.shape[3]) != want or t.shape[2] < num_tokens \
                        or not same_device(t.device, self.device):
                    raise ValueError(f"layer {li}: KV {tuple(t.shape)} {t.dtype} {t.device} does not match the namespace "
                                     f"{want} {self.dtype} {self.device} with >= {num_tokens} tokens")

    # ---- lookup
    def restore(self, token_ids, keys: list, config):
        """Restore the longest present run of `keys` (the eligible chunk keys, in order) into a
        new DynamicCache. Returns (cache | None, L, sources)."""
        n = 0
        while n < len(keys) and keys[n] in self.entries:
            n += 1
        self.counters["lookups"] += 1
        if n < len(keys):
            self.counters["chunk_misses"] += 1
        if n == 0:
            self.last_restore = {"chunks": 0, "scratch_bytes": 0}
            return None, 0, []
        hit = []
        for i in range(n):
            ident, t = self.entries[keys[i]]
            if ident != ChunkIdentity.for_chunk(self.ns_digest, token_ids, self.C, i, keys):
                raise RuntimeError(f"GPU prefix-cache entry {keys[i].hex()[:12]} identity mismatch (in-process invariant)")
            self.entries.move_to_end(keys[i])
            hit.append(t)
        layers = [(torch.cat([t[li, 0] for t in hit], dim=-2), torch.cat([t[li, 1] for t in hit], dim=-2))
                  for li in range(self.ns.num_layers)]  # new tensors: temporary scratch
        scratch = sum(k.untyped_storage().nbytes() + v.untyped_storage().nbytes() for k, v in layers)
        cache = DynamicCache(ddp_cache_data=layers, config=config)  # copies again (D4)
        del layers
        L = n * self.C
        if len(cache.layers) != self.ns.num_layers or any(layer.get_seq_length() != L for layer in cache.layers):
            raise RuntimeError("restored cache has wrong layer count or length")
        self.counters["chunk_hits"] += n
        self.counters["restored_requests"] += 1
        self.counters["recovered_prefix_tokens"] += L
        self.last_restore = {"chunks": n, "scratch_bytes": scratch}
        return cache, L, ["gpu"] * n

    # ---- store
    def store(self, cache: DynamicCache, token_ids, keys: list) -> list:
        """Copy every complete prompt chunk not already present into an owned tensor. Returns
        the chunk indices stored by this call."""
        self._check_model_cache(cache, len(keys) * self.C)
        stored = []
        for i, key in enumerate(keys):
            if key in self.entries:
                continue
            if self.capacity_chunks == 0:
                self.counters["write_skips_no_capacity"] += 1
                continue
            while len(self.entries) >= self.capacity_chunks:
                self.entries.popitem(last=False)
                self.counters["evictions"] += 1
            ident = ChunkIdentity.for_chunk(self.ns_digest, token_ids, self.C, i, keys)
            t = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            s, e = i * self.C, (i + 1) * self.C
            for li, layer in enumerate(cache.layers):
                t[li, 0].copy_(layer.keys[:, :, s:e, :])
                t[li, 1].copy_(layer.values[:, :, s:e, :])
            self.entries[key] = (ident, t)
            self.counters["chunks_stored"] += 1
            stored.append(i)
        retained = self.retained_bytes()
        if retained > self.budget_bytes:  # cannot happen by construction; checked, never assumed
            raise RuntimeError(f"GPU prefix cache retains {retained} B, budget {self.budget_bytes} B")
        self.peak_retained_bytes = max(self.peak_retained_bytes, retained)
        return stored

    def resources(self) -> dict:
        return {"budget_bytes": self.budget_bytes, "capacity_chunks": self.capacity_chunks,
                "chunk_bytes": self.chunk_bytes, "entries": len(self.entries),
                "retained_bytes": self.retained_bytes(), "peak_retained_bytes": self.peak_retained_bytes}
