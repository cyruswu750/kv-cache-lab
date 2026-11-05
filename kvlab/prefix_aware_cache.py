"""C1-PA: the prefix-aware bounded GPU-only prefix cache (M4-B; decision D33; docs/M4B_PROTOCOL.md §5).

A subclass of the reviewed plain-LRU `kvlab.gpu_prefix_cache.GPUPrefixCache`, which stays unchanged
and is kept for offline diagnostics. Everything except admission and eviction is inherited:
- identity (chain keys, field-by-field `ChunkIdentity` check on every hit; a mismatch raises);
- lookup (the longest leading run of eligible present chunks; hits are touched root-first);
- ownership (one owned contiguous tensor per entry; a hit gathers into new tensors and the
  public `DynamicCache` constructor copies again, so continuing a request never touches the
  cache);
- the byte budget (capacity = budget // chunk_bytes; retained storage measured, never assumed).

Prefix-aware admission and eviction (store()):
- **Prefix closure (invariant):** every cached chunk's parent (the chunk before it in its own
  prompt, or the root for chunk 0) is cached. So every cached chunk is reachable by a lookup.
- **Order:** chunks are considered in prompt order. A present chunk is skipped without being
  refreshed (as the plain LRU and the host tier do).
- **Eviction:** when full, the victim is the least recently used *leaf* (an entry with no
  cached child) that is not protected. Removing a leaf preserves prefix closure.
- **Protection lifecycle:** the protected set starts empty when store() is entered. A chunk of
  the current prompt joins it when the loop reaches it, whether it was already present or has
  just been admitted. The set is dropped when store() returns. So a request never evicts its
  own prefix to admit a later chunk of itself. restore() never evicts and needs no
  protection; nothing runs between a request's restore() and its store() (sequential, batch 1).
- **Admission stop:** if no unprotected leaf exists, admission of this prompt stops at that
  chunk. The remaining chunks are not admitted (`admission_stops_prefix_capacity` +1,
  `chunks_refused_prefix_capacity` + the number left). An oversized prompt therefore keeps its
  first `capacity` chunks, instead of the unreachable tail that plain LRU keeps.
- **Capacity 0:** nothing is stored (`write_skips_no_capacity`, per absent chunk, as the base).

The invariants (closure, child counts, bound) are rechecked after every store() and raise on
violation: a violation is a programming error, never a cache miss.
"""

from kvlab.gpu_prefix_cache import GPUPrefixCache
from kvlab.identity import ROOT_PARENT, ChunkIdentity

import torch


class PrefixAwareGPUPrefixCache(GPUPrefixCache):
    policy = "prefix_aware_lru"

    def __init__(self, ns, budget_bytes: int, device):
        super().__init__(ns, budget_bytes, device)
        self.parent: dict[bytes, bytes] = {}    # key -> parent key (ROOT_PARENT for chunk 0)
        self.children: dict[bytes, int] = {}    # key -> number of cached children

    def _evict(self, key: bytes) -> None:
        del self.entries[key]
        p = self.parent.pop(key)
        if self.children.pop(key) != 0:
            raise RuntimeError("evicted a chunk that still has cached children (prefix closure)")
        if p != ROOT_PARENT:
            self.children[p] -= 1
        self.counters["evictions"] += 1

    def check_invariants(self) -> None:
        keys = set(self.entries)
        if set(self.parent) != keys or set(self.children) != keys:
            raise RuntimeError("prefix-aware bookkeeping out of sync with the entries")
        counts = dict.fromkeys(keys, 0)
        for k, p in self.parent.items():
            if p != ROOT_PARENT:
                if p not in keys:
                    raise RuntimeError(f"entry {k.hex()[:12]} has an uncached parent (prefix closure violated)")
                counts[p] += 1
        if counts != self.children:
            raise RuntimeError("cached-child counts are wrong")
        if len(self.entries) > self.capacity_chunks:
            raise RuntimeError(f"{len(self.entries)} entries exceed the capacity {self.capacity_chunks}")

    def store(self, cache, token_ids, keys: list) -> list:
        """Admit the prompt's complete chunks in order under the prefix-aware rules above. Returns
        the chunk indices stored by this call."""
        self._check_model_cache(cache, len(keys) * self.C)
        stored, protected = [], set()
        for i, key in enumerate(keys):
            if key in self.entries:
                protected.add(key)
                continue
            if self.capacity_chunks == 0:
                self.counters["write_skips_no_capacity"] += 1
                continue
            parent = keys[i - 1] if i else ROOT_PARENT
            if i and parent not in self.entries:  # the loop stops at the first chunk it cannot admit
                raise RuntimeError("admission would break prefix closure (programming error)")
            if len(self.entries) >= self.capacity_chunks:
                victim = next((k for k in self.entries if self.children[k] == 0 and k not in protected), None)
                if victim is None:
                    self.counters["admission_stops_prefix_capacity"] += 1
                    self.counters["chunks_refused_prefix_capacity"] += len(keys) - i
                    break
                self._evict(victim)
            ident = ChunkIdentity.for_chunk(self.ns_digest, token_ids, self.C, i, keys)
            t = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            s, e = i * self.C, (i + 1) * self.C
            for li, layer in enumerate(cache.layers):
                t[li, 0].copy_(layer.keys[:, :, s:e, :])
                t[li, 1].copy_(layer.values[:, :, s:e, :])
            self.entries[key] = (ident, t)
            self.parent[key], self.children[key] = parent, 0
            if parent != ROOT_PARENT:
                self.children[parent] += 1
            protected.add(key)
            self.counters["chunks_stored"] += 1
            stored.append(i)
        self.check_invariants()
        retained = self.retained_bytes()
        if retained > self.budget_bytes:  # cannot happen by construction; checked, never assumed
            raise RuntimeError(f"GPU prefix cache retains {retained} B, budget {self.budget_bytes} B")
        self.peak_retained_bytes = max(self.peak_retained_bytes, retained)
        return stored

    def resources(self) -> dict:
        return {**super().resources(), "policy": self.policy}
