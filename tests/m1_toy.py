"""Tiny CPU-only fixtures for M1 tests (imported by tests/test_*.py)."""

import errno
import os

import torch
from transformers import DynamicCache

from kvlab import chunk_format as cf
from kvlab.identity import ChunkIdentity, Namespace, chain_keys


def toy_ns(**over) -> Namespace:
    fields = dict(model_id="toy/model", revision="0" * 40, kv_dtype="bfloat16", num_layers=2, num_kv_heads=1,
                  head_dim=8, chunk_size=4, rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
                  layer_types=("full_attention", "full_attention"), use_sliding_window=False, sliding_window=None,
                  attn_implementation="sdpa", transformers_version="5.22.1", torch_version="2.14.1+cu130")
    fields.update(over)
    return Namespace(**fields)


def tokens(n, start=100):
    return list(range(start, start + n))


def ident(ns, token_ids, index):
    keys = chain_keys(ns.digest(), token_ids, ns.chunk_size)
    return ChunkIdentity.for_chunk(ns.digest(), token_ids, ns.chunk_size, index, keys)


def payload(ns, seed=0) -> bytearray:
    g = torch.Generator().manual_seed(seed)
    return bytearray(torch.randint(0, 256, (cf.payload_len(ns),), generator=g, dtype=torch.uint8).numpy().tobytes())


def chunk_file_bytes(ns, idt, data) -> bytes:
    import io
    buf = io.BytesIO()
    cf.write_chunk(buf, cf.encode_header(ns, idt), data)
    return buf.getvalue()


def toy_cache(ns, seq_len, seed=0) -> DynamicCache:
    g = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, ns.kv_dtype)
    data = [(torch.randn(1, ns.num_kv_heads, seq_len, ns.head_dim, generator=g).to(dtype),
             torch.randn(1, ns.num_kv_heads, seq_len, ns.head_dim, generator=g).to(dtype))
            for _ in range(ns.num_layers)]
    return DynamicCache(ddp_cache_data=data)


def deny_unlink(monkeypatch, match, err=errno.EACCES) -> list:
    """Controlled fault boundary: os.unlink (also used by Path.unlink) raises
    OSError(err) for file names accepted by `match`; other paths are unlinked
    normally. Returns the list of refused names, in order (one per attempt).
    No real permissions are changed."""
    real = os.unlink
    attempts = []

    def fake_unlink(path, *args, **kwargs):
        name = os.path.basename(os.fspath(path))
        if match(name):
            attempts.append(name)
            raise OSError(err, os.strerror(err), os.fspath(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", fake_unlink)
    return attempts
