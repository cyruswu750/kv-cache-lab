"""Independent key-level models of the M4-B cache decisions (CPU only; no tensors, no model).

The M4-B validator compares every recorded decision with these models (docs/M4B_PROTOCOL.md §8).
They are written independently of the cache classes: integer recency stamps instead of an
OrderedDict, and parent links recomputed by scanning instead of child counters. Agreement is
therefore evidence, not a tautology.

All models follow the engine rules: chain keys (`kvlab.identity.chain_keys`; any fixed namespace
digest gives the same equality structure within a trace), the first-token rule
(`kvlab.boundaries.max_restore_len`), the longest leading run of present eligible chunks,
storage of every complete prompt chunk, and present chunks skipped without refresh.

- `pa_model`: C1-PA (kvlab/prefix_aware_cache.py).
- `lru_model`: C1-LRU (kvlab/gpu_prefix_cache.py), with unreachable ("dead") entries, for the
  offline diagnostic only. Its restore lengths must equal `kvlab.traces.lru_restore_lengths`.
- `tiered_model`: C2 (kvlab/tiered_cache.py with HostStore and DiskStore) assuming every
  admission, spill and read succeeds: host LRU of `host_slots`, spill of the host victim to a disk
  LRU of `disk_files` files (no promotion of disk hits, D15), and restore from either tier.
"""

from kvlab.boundaries import max_restore_len
from kvlab.identity import ROOT_PARENT, chain_keys

NS = b"\x00" * 32  # the namespace digest only renames keys; equality structure is what matters


def _keys(tokens, C):
    return chain_keys(NS, list(tokens), C)


def _lookup(keys, n_tokens, C, present):
    n = 0
    while n < max_restore_len(n_tokens, C) // C and present(keys[n]):
        n += 1
    return n


def pa_model(prompts, C: int, cap: int) -> list:
    """Per prompt: restore_len, stored (chunk indices), evictions, refused (chunks not admitted
    because no unprotected leaf existed), stopped (bool), no_capacity_skips."""
    stamp, parent, clock, out = {}, {}, 0, []
    for tokens in prompts:
        keys = _keys(tokens, C)
        n = _lookup(keys, len(tokens), C, lambda k: k in stamp)
        for k in keys[:n]:
            clock += 1
            stamp[k] = clock
        rec = {"restore_len": n * C, "stored": [], "evictions": 0, "refused": 0, "stopped": False,
               "no_capacity_skips": 0}
        mine = set()
        for i, k in enumerate(keys):
            if k in stamp:
                mine.add(k)
                continue
            if cap <= 0:
                rec["no_capacity_skips"] += 1
                continue
            if len(stamp) >= cap:
                cached_parents = {p for x, p in parent.items() if x in stamp}
                leaves = [x for x in stamp if x not in cached_parents and x not in mine]
                if not leaves:
                    rec["stopped"], rec["refused"] = True, len(keys) - i
                    break
                victim = min(leaves, key=lambda x: stamp[x])
                del stamp[victim], parent[victim]
                rec["evictions"] += 1
            clock += 1
            stamp[k], parent[k] = clock, (keys[i - 1] if i else ROOT_PARENT)
            mine.add(k)
            rec["stored"].append(i)
        assert len(stamp) <= max(cap, 0)
        assert all(p == ROOT_PARENT or p in stamp for x, p in parent.items()), "model lost prefix closure"
        out.append(rec)
    return out


def lru_model(prompts, C: int, cap: int) -> list:
    """Per prompt: restore_len, stored, evictions, dead_after (cached entries with an uncached
    ancestor after this prompt's admission)."""
    stamp, parent, clock, out = {}, {}, 0, []
    for tokens in prompts:
        keys = _keys(tokens, C)
        for i, k in enumerate(keys):
            parent.setdefault(k, keys[i - 1] if i else ROOT_PARENT)
        n = _lookup(keys, len(tokens), C, lambda k: k in stamp)
        for k in keys[:n]:
            clock += 1
            stamp[k] = clock
        rec = {"restore_len": n * C, "stored": [], "evictions": 0}
        for i, k in enumerate(keys):
            if k in stamp or cap <= 0:
                continue
            if len(stamp) >= cap:
                del stamp[min(stamp, key=stamp.get)]
                rec["evictions"] += 1
            clock += 1
            stamp[k] = clock
            rec["stored"].append(i)

        def reachable(x):
            while x != ROOT_PARENT:
                if x not in stamp:
                    return False
                x = parent[x]
            return True
        rec["dead_after"] = sum(1 for x in stamp if not reachable(x))
        out.append(rec)
    return out


def tiered_model(prompts, C: int, host_slots: int, disk_files: int) -> list:
    """Per prompt: restore_len, sources ("host"/"disk" per restored chunk), stored, spills (host
    victims written to disk), disk_evictions, and the occupancy after the prompt."""
    host, disk, clock, out = {}, {}, 0, []  # key -> recency stamp, per tier
    for tokens in prompts:
        keys = _keys(tokens, C)
        n = _lookup(keys, len(tokens), C, lambda k: k in host or k in disk)
        rec = {"restore_len": n * C, "sources": [], "stored": [], "spills": 0, "disk_evictions": 0}
        for k in keys[:n]:
            clock += 1
            if k in host:
                host[k] = clock
                rec["sources"].append("host")
            else:
                disk[k] = clock
                rec["sources"].append("disk")
        for i, k in enumerate(keys):
            if k in host or k in disk or host_slots <= 0:
                continue
            if len(host) >= host_slots:
                victim = min(host, key=host.get)
                del host[victim]
                if disk_files <= 0:
                    raise ValueError("the model assumes a disk tier that accepts spills")
                if len(disk) >= disk_files:
                    del disk[min(disk, key=disk.get)]
                    rec["disk_evictions"] += 1
                clock += 1
                disk[victim] = clock
                rec["spills"] += 1
            clock += 1
            host[k] = clock
            rec["stored"].append(i)
        rec["host_entries"], rec["disk_entries"] = len(host), len(disk)
        out.append(rec)
    return out


def oracle_lengths(prompts, C: int) -> list:
    seen, out = set(), []
    for tokens in prompts:
        keys = _keys(tokens, C)
        out.append(_lookup(keys, len(tokens), C, lambda k: k in seen) * C)
        seen.update(keys)
    return out
