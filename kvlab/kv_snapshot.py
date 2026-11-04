"""Export a prefix of a HF DynamicCache to independent CPU storage, and restore
it into a brand-new DynamicCache on a target device.

Layout (transformers 5.22.1, verified by inspection): DynamicCache.layers is a
list of DynamicLayer; each layer holds `.keys` / `.values` shaped
[batch, num_kv_heads, seq_len, head_dim].

Snapshot layout: snapshot.chunks[c][layer] = (k, v), each a contiguous CPU
tensor shaped [batch, num_kv_heads, chunk_size, head_dim]. Chunk c covers token
positions [c*C, (c+1)*C).

Ownership:
- Export allocates fresh CPU tensors and copies into them, so the snapshot
  never aliases GPU memory or the source cache.
- Restore allocates fresh device tensors and copies the snapshot into them;
  DynamicCache then copies once more (DynamicLayer.update does torch.cat), so
  the restored cache owns its storage and later appends never touch the
  snapshot.

Invariant violations (wrong layer type, short cache, misaligned lengths) raise.
They are programming errors, not cache misses.
"""

import hashlib
from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


@dataclass
class KVSnapshot:
    chunk_size: int
    num_layers: int
    chunks: list  # chunks[c][layer] = (k_cpu, v_cpu)

    @property
    def num_tokens(self) -> int:
        return len(self.chunks) * self.chunk_size

    def tensors(self):
        """Yield (chunk_idx, layer_idx, 'k'|'v', tensor)."""
        for c, layers in enumerate(self.chunks):
            for li, (k, v) in enumerate(layers):
                yield c, li, "k", k
                yield c, li, "v", v

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for *_, t in self.tensors())


def raw_bytes(t: torch.Tensor) -> torch.Tensor:
    """Flat uint8 view of a tensor's bytes (copied to contiguous if needed)."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Strict raw-byte equality. Shapes and dtypes must also match."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.device != b.device:
        raise ValueError(f"compare on one device, got {a.device} and {b.device}")
    return torch.equal(raw_bytes(a), raw_bytes(b))


def sha256_of(t: torch.Tensor) -> str:
    if t.device.type != "cpu":
        raise ValueError("sha256_of expects a CPU tensor")
    return hashlib.sha256(raw_bytes(t).numpy().tobytes()).hexdigest()


def _check_layers(cache: DynamicCache, num_tokens: int) -> None:
    if not cache.layers:
        raise ValueError("cache has no layers")
    for li, layer in enumerate(cache.layers):
        if type(layer) is not DynamicLayer:
            raise TypeError(f"layer {li}: expected DynamicLayer, got {type(layer).__name__}")
        if layer.keys.shape != layer.values.shape:
            raise ValueError(f"layer {li}: K shape {layer.keys.shape} != V shape {layer.values.shape}")
        if layer.get_seq_length() < num_tokens:
            raise ValueError(f"layer {li}: has {layer.get_seq_length()} tokens, need {num_tokens}")


def export_prefix_to_cpu(cache: DynamicCache, num_tokens: int, chunk_size: int) -> KVSnapshot:
    """Copy tokens [0, num_tokens) of every layer into fresh CPU tensors, per chunk."""
    if num_tokens <= 0 or num_tokens % chunk_size != 0:
        raise ValueError(f"num_tokens={num_tokens} must be a positive multiple of chunk_size={chunk_size}")
    _check_layers(cache, num_tokens)

    chunks = []
    for c in range(num_tokens // chunk_size):
        start, end = c * chunk_size, (c + 1) * chunk_size
        layer_pairs = []
        for layer in cache.layers:
            pair = []
            for src in (layer.keys[:, :, start:end, :], layer.values[:, :, start:end, :]):
                dst = torch.empty(src.shape, dtype=src.dtype, device="cpu")
                dst.copy_(src)  # D2H into pageable memory: synchronous w.r.t. the host
                pair.append(dst)
            layer_pairs.append(tuple(pair))
        chunks.append(layer_pairs)
    return KVSnapshot(chunk_size=chunk_size, num_layers=len(cache.layers), chunks=chunks)


def restore_to_device(snapshot: KVSnapshot, num_tokens: int, config, device) -> DynamicCache:
    """Build a new DynamicCache holding tokens [0, num_tokens) from the CPU snapshot."""
    if num_tokens <= 0 or num_tokens % snapshot.chunk_size != 0:
        raise ValueError(f"num_tokens={num_tokens} must be a positive multiple of {snapshot.chunk_size}")
    if num_tokens > snapshot.num_tokens:
        raise ValueError(f"snapshot holds {snapshot.num_tokens} tokens, asked for {num_tokens}")
    n_chunks = num_tokens // snapshot.chunk_size
    C = snapshot.chunk_size

    k00, _ = snapshot.chunks[0][0]
    b, h, _, d = k00.shape
    expected_shape, expected_dtype = (b, h, C, d), k00.dtype
    per_layer = []
    for li in range(snapshot.num_layers):
        k_dev = torch.empty((b, h, num_tokens, d), dtype=expected_dtype, device=device)
        v_dev = torch.empty((b, h, num_tokens, d), dtype=expected_dtype, device=device)
        for c in range(n_chunks):
            k_cpu, v_cpu = snapshot.chunks[c][li]
            for name, t in (("k", k_cpu), ("v", v_cpu)):
                # copy_ would silently broadcast a wrong shape or cast a wrong dtype; refuse instead.
                if t.device.type != "cpu":
                    raise ValueError(f"chunk {c} layer {li} {name}: snapshot tensor is on {t.device}, not CPU")
                if tuple(t.shape) != expected_shape or t.dtype != expected_dtype:
                    raise ValueError(f"chunk {c} layer {li} {name}: got {tuple(t.shape)} {t.dtype}, "
                                     f"expected {expected_shape} {expected_dtype}")
            k_dev[:, :, c * C:(c + 1) * C, :].copy_(k_cpu)
            v_dev[:, :, c * C:(c + 1) * C, :].copy_(v_cpu)
        per_layer.append((k_dev, v_dev))

    cache = DynamicCache(ddp_cache_data=per_layer, config=config)
    _check_layers(cache, num_tokens)
    if len(cache.layers) != snapshot.num_layers:
        raise ValueError(f"restored {len(cache.layers)} layers, snapshot has {snapshot.num_layers}")
    for li, layer in enumerate(cache.layers):
        if layer.get_seq_length() != num_tokens:
            raise ValueError(f"layer {li}: restored length {layer.get_seq_length()} != {num_tokens}")
    return cache


def compare_snapshot_to_cache(snapshot: KVSnapshot, cache: DynamicCache, num_tokens: int) -> dict:
    """Byte-compare snapshot chunks [0, num_tokens) against the cache's tensors.

    The cache slice is copied to the CPU only for this comparison.
    Returns a summary with per-tensor shape/dtype/nbytes and any mismatches.
    """
    C = snapshot.chunk_size
    n_chunks = num_tokens // C
    if len(cache.layers) != snapshot.num_layers:
        raise ValueError(f"cache has {len(cache.layers)} layers, snapshot has {snapshot.num_layers}")
    if num_tokens % C != 0 or n_chunks > len(snapshot.chunks):
        raise ValueError(f"cannot compare {num_tokens} tokens against a {snapshot.num_tokens}-token snapshot")
    checked, mismatches, total_bytes = 0, [], 0
    kinds = {}
    for c in range(n_chunks):
        for li, layer in enumerate(cache.layers):
            k_snap, v_snap = snapshot.chunks[c][li]
            for name, snap, full in (("k", k_snap, layer.keys), ("v", v_snap, layer.values)):
                live = full[:, :, c * C:(c + 1) * C, :].to("cpu")
                nbytes = snap.numel() * snap.element_size()
                key = (tuple(snap.shape), str(snap.dtype), nbytes)
                kinds[key] = kinds.get(key, 0) + 1
                checked += 1
                total_bytes += nbytes
                if not bytes_equal(snap, live):
                    mismatches.append({"chunk": c, "layer": li, "kind": name,
                                       "snap_shape": list(snap.shape), "live_shape": list(live.shape),
                                       "snap_dtype": str(snap.dtype), "live_dtype": str(live.dtype)})
    return {
        "tensors_checked": checked,
        "all_bytes_equal": not mismatches and checked > 0,
        "total_bytes": total_bytes,
        "tensor_kinds": [{"shape": list(s), "dtype": d, "nbytes": n, "count": cnt}
                         for (s, d, n), cnt in kinds.items()],
        "mismatches": mismatches,
    }


def fingerprint(snapshot: KVSnapshot) -> list[str]:
    return [sha256_of(t) for *_, t in snapshot.tensors()]


def zeroed_copy(snapshot: KVSnapshot) -> KVSnapshot:
    """Negative-control snapshot: same shapes, all-zero K/V."""
    chunks = [[(torch.zeros_like(k), torch.zeros_like(v)) for k, v in layers] for layers in snapshot.chunks]
    return KVSnapshot(chunk_size=snapshot.chunk_size, num_layers=snapshot.num_layers, chunks=chunks)
