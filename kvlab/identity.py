"""Cache identity: namespace digest and chained chunk keys (docs/design.md §3).

Everything here is deterministic across processes: canonical JSON + SHA-256.
Python's built-in hash() (randomized per process) is never used.
"""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass

LAYOUT_SCHEMA = "kvlab-kv-layout-v1"  # [layer][K,V] x [1, kv_heads, C, head_dim], C-order, little-endian
ROOT_PARENT = bytes(32)
_NS_TAG = b"kvlab-namespace-v1\0"
_CHUNK_TAG = b"kvlab-chunk-v1\0"
_MAX_TOKEN_ID = 2**32 - 1


def _check_canonical(obj, path="ns"):
    if obj is None or isinstance(obj, (bool, int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{path}: non-finite float {obj!r}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_canonical(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"{path}: non-string key {k!r}")
            _check_canonical(v, f"{path}.{k}")
        return
    raise TypeError(f"{path}: unsupported type {type(obj).__name__}")


def canonical_json(obj) -> bytes:
    _check_canonical(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True)
class Namespace:
    model_id: str
    revision: str
    kv_dtype: str            # e.g. "bfloat16" -- the actual dtype of the KV tensors
    num_layers: int
    num_kv_heads: int
    head_dim: int
    chunk_size: int
    rope_parameters: dict
    layer_types: tuple
    use_sliding_window: bool
    sliding_window: int | None
    attn_implementation: str
    transformers_version: str
    torch_version: str
    layout_schema: str = LAYOUT_SCHEMA

    def fields(self) -> dict:
        d = asdict(self)
        d["layer_types"] = list(self.layer_types)
        return d

    def canonical(self) -> bytes:
        return canonical_json(self.fields())

    def digest(self) -> bytes:
        return namespace_digest_of_canonical(self.canonical())


def namespace_digest_of_canonical(canonical: bytes) -> bytes:
    return hashlib.sha256(_NS_TAG + canonical).digest()


def namespace_for(model, model_id: str, revision: str, chunk_size: int) -> Namespace:
    """Namespace of the loaded model. kv_dtype is the dtype the K/V projections
    produce (the parameter dtype); storing re-checks every cached tensor against it."""
    import torch
    import transformers

    cfg = model.config
    dtype = next(model.parameters()).dtype
    return Namespace(
        model_id=model_id,
        revision=revision,
        kv_dtype=str(dtype).removeprefix("torch."),
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        chunk_size=chunk_size,
        rope_parameters=dict(cfg.rope_parameters),
        layer_types=tuple(cfg.layer_types),
        use_sliding_window=bool(getattr(cfg, "use_sliding_window", False)),
        sliding_window=getattr(cfg, "sliding_window", None) if getattr(cfg, "use_sliding_window", False) else None,
        attn_implementation=cfg._attn_implementation,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
    )


def chunk_key(ns_digest: bytes, parent_key: bytes, chunk_index: int, tokens) -> bytes:
    if len(ns_digest) != 32 or len(parent_key) != 32:
        raise ValueError("ns_digest and parent_key must be 32 bytes")
    tokens = list(tokens)
    for t in tokens:
        if not (isinstance(t, int) and 0 <= t <= _MAX_TOKEN_ID):
            raise ValueError(f"invalid token id {t!r}")
    h = hashlib.sha256(_CHUNK_TAG)
    h.update(ns_digest)
    h.update(parent_key)
    h.update(struct.pack("<II", chunk_index, len(tokens)))
    h.update(struct.pack(f"<{len(tokens)}I", *tokens))
    return h.digest()


def chain_keys(ns_digest: bytes, token_ids, chunk_size: int) -> list[bytes]:
    """Keys of every complete chunk of token_ids, in order."""
    token_ids = list(token_ids)
    keys, parent = [], ROOT_PARENT
    for i in range(len(token_ids) // chunk_size):
        parent = chunk_key(ns_digest, parent, i, token_ids[i * chunk_size:(i + 1) * chunk_size])
        keys.append(parent)
    return keys


@dataclass(frozen=True)
class ChunkIdentity:
    """Everything that identifies one stored chunk; compared field by field on read."""
    ns_digest: bytes
    key: bytes
    parent: bytes
    index: int
    tokens: tuple

    @staticmethod
    def for_chunk(ns_digest: bytes, token_ids, chunk_size: int, index: int, keys: list[bytes]):
        tokens = tuple(token_ids[index * chunk_size:(index + 1) * chunk_size])
        if len(tokens) != chunk_size:
            raise ValueError(f"chunk {index} is incomplete ({len(tokens)} tokens)")
        parent = keys[index - 1] if index > 0 else ROOT_PARENT
        return ChunkIdentity(ns_digest, keys[index], parent, index, tokens)
