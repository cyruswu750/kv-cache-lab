"""Disk chunk file format v1 (docs/design.md §5). No pickle, no object deserialization.

Layout (little-endian): 140-byte fixed header | u32 header_crc | token ids (u32 each)
| namespace meta (canonical JSON, <= 4096 B) | payload | u32 payload_crc.
CRC32 detects accidental corruption; it is not authentication.
"""

import errno
import json
import struct
import sys
import zlib

from kvlab.identity import ChunkIdentity, Namespace, chunk_key, namespace_digest_of_canonical

if sys.byteorder != "little":
    raise ImportError("kvlab chunk format v1 stores native little-endian tensor bytes; big-endian hosts unsupported")

MAGIC = b"KVLCHNK1"
VERSION = 1
DTYPE_CODES = {"bfloat16": 1, "float16": 2, "float32": 3}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}
ELEM_SIZE = {"bfloat16": 2, "float16": 2, "float32": 4}
FIXED = struct.Struct("<8sHH32s32s32sIIIIIIQ")  # 140 bytes
U32 = struct.Struct("<I")
HEADER_PREFIX = FIXED.size + U32.size  # 144
MAX_META = 4096


class ChunkReadError(Exception):
    """A known, expected cache-entry problem. Callers turn it into a miss."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def payload_len(ns: Namespace) -> int:
    return ns.num_layers * 2 * ns.num_kv_heads * ns.chunk_size * ns.head_dim * ELEM_SIZE[ns.kv_dtype]


def file_size(ns: Namespace) -> int:
    return HEADER_PREFIX + U32.size * ns.chunk_size + len(ns.canonical()) + payload_len(ns) + U32.size


def encode_header(ns: Namespace, ident: ChunkIdentity) -> bytes:
    meta = ns.canonical()
    if len(meta) > MAX_META:
        raise ValueError(f"namespace metadata is {len(meta)} B, limit {MAX_META}")
    if len(ident.tokens) != ns.chunk_size or ident.ns_digest != ns.digest():
        raise ValueError("identity does not belong to this namespace / chunk size")
    tokens = struct.pack(f"<{len(ident.tokens)}I", *ident.tokens)
    fixed = FIXED.pack(MAGIC, VERSION, DTYPE_CODES[ns.kv_dtype], ident.ns_digest, ident.key, ident.parent,
                       ident.index, len(ident.tokens), ns.num_layers, ns.num_kv_heads, ns.head_dim, len(meta),
                       payload_len(ns))
    crc = zlib.crc32(meta, zlib.crc32(tokens, zlib.crc32(fixed)))
    return fixed + U32.pack(crc) + tokens + meta


def default_write(f, mv):
    return f.write(mv)


def default_readinto(f, mv):
    return f.readinto(mv)


def _write_all(f, data, write_fn):
    mv = memoryview(data).cast("B")
    off = 0
    while off < len(mv):
        n = write_fn(f, mv[off:])
        if not n:  # 0 or None: no progress on a blocking file
            raise OSError(errno.EIO, "write made no progress")
        off += n


def write_chunk(f, header: bytes, payload, write_fn=default_write) -> None:
    payload = memoryview(payload).cast("B")
    crc = zlib.crc32(payload)
    for part in (header, payload, U32.pack(crc)):
        _write_all(f, part, write_fn)


def _readinto_exact(f, mv, readinto_fn):
    off = 0
    while off < len(mv):
        n = readinto_fn(f, mv[off:])
        if not n:
            raise ChunkReadError("truncated", f"EOF after {off} of {len(mv)} bytes")
        off += n


def _read_exact(f, n, readinto_fn) -> bytes:
    buf = bytearray(n)  # n is bounded by the caller before this call
    _readinto_exact(f, memoryview(buf), readinto_fn)
    return bytes(buf)


def _namespace_mismatch_field(meta: bytes, expected: Namespace) -> str:
    try:
        stored = json.loads(meta.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "unparseable_meta"
    if not isinstance(stored, dict):
        return "unparseable_meta"
    for name, value in sorted(expected.fields().items()):
        if stored.get(name) != value:
            return name
    return "digest"


def read_chunk_into(f, size: int, ns: Namespace, ident: ChunkIdentity, dest, readinto_fn=default_readinto) -> None:
    """Validate a chunk file against the expected namespace and identity and read
    its payload into `dest` (a writable buffer of exactly payload_len(ns) bytes).
    Raises ChunkReadError for every known entry problem; nothing is sized by an
    unvalidated header field."""
    dest = memoryview(dest).cast("B")
    expected_plen = payload_len(ns)
    if len(dest) != expected_plen:
        raise ValueError(f"destination is {len(dest)} B, expected {expected_plen}")  # caller bug
    if size < HEADER_PREFIX:
        raise ChunkReadError("truncated", f"file is {size} B")
    head = _read_exact(f, HEADER_PREFIX, readinto_fn)
    (magic, version, dtype_code, ns_d, key, parent, index, tcount, n_layers, n_heads, h_dim, meta_len,
     plen) = FIXED.unpack(head[:FIXED.size])
    (header_crc,) = U32.unpack(head[FIXED.size:])
    if magic != MAGIC:
        raise ChunkReadError("bad_magic")
    if version != VERSION:
        raise ChunkReadError("bad_version", str(version))
    if dtype_code not in CODE_DTYPES:
        raise ChunkReadError("bad_dtype_code", str(dtype_code))
    for name, got, want in (("kv_dtype", CODE_DTYPES[dtype_code], ns.kv_dtype), ("num_layers", n_layers, ns.num_layers),
                            ("num_kv_heads", n_heads, ns.num_kv_heads), ("head_dim", h_dim, ns.head_dim),
                            ("token_count", tcount, ns.chunk_size), ("payload_len", plen, expected_plen)):
        if got != want:
            raise ChunkReadError(f"layout_mismatch:{name}", f"{got} != {want}")
    if meta_len > MAX_META:
        raise ChunkReadError("bad_meta_len", str(meta_len))
    expected_size = HEADER_PREFIX + U32.size * tcount + meta_len + plen + U32.size
    if size != expected_size:
        raise ChunkReadError("size_mismatch", f"{size} != {expected_size}")

    rest = _read_exact(f, U32.size * tcount + meta_len, readinto_fn)  # <= 4*C + 4096 bytes
    if zlib.crc32(rest, zlib.crc32(head[:FIXED.size])) != header_crc:
        raise ChunkReadError("header_crc")
    tokens = struct.unpack(f"<{tcount}I", rest[:U32.size * tcount])
    meta = rest[U32.size * tcount:]
    if ns_d != ident.ns_digest:
        raise ChunkReadError(f"namespace_mismatch:{_namespace_mismatch_field(meta, ns)}")
    if namespace_digest_of_canonical(meta) != ns_d:
        raise ChunkReadError("namespace_mismatch:meta_inconsistent")
    for name, got, want in (("index", index, ident.index), ("parent", parent, ident.parent),
                            ("tokens", tokens, tuple(ident.tokens)), ("key", key, ident.key)):
        if got != want:
            raise ChunkReadError(f"identity_mismatch:{name}")
    if chunk_key(ns_d, parent, index, tokens) != key:
        raise ChunkReadError("identity_mismatch:key_inconsistent")

    _readinto_exact(f, dest, readinto_fn)
    (payload_crc,) = U32.unpack(_read_exact(f, U32.size, readinto_fn))
    if zlib.crc32(dest) != payload_crc:
        raise ChunkReadError("payload_crc")
