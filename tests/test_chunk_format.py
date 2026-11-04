import dataclasses
import errno
import io
import struct
import tracemalloc
import zlib

import pytest
import torch

from kvlab import chunk_format as cf
from kvlab.chunk_format import ChunkReadError
from kvlab.identity import ChunkIdentity
from m1_toy import chunk_file_bytes, ident, payload, tokens, toy_ns


def read(data: bytes, ns, idt, readinto_fn=cf.default_readinto):
    dest = bytearray(cf.payload_len(ns))
    cf.read_chunk_into(io.BytesIO(data), len(data), ns, idt, dest, readinto_fn)
    return bytes(dest)


def reason(data, ns, idt):
    with pytest.raises(ChunkReadError) as e:
        read(data, ns, idt)
    return e.value.reason


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_lossless_round_trip(dtype):
    ns = toy_ns(kv_dtype=dtype)
    idt = ident(ns, tokens(8), 1)
    data = payload(ns, seed=3)
    blob = chunk_file_bytes(ns, idt, data)
    assert len(blob) == cf.file_size(ns)
    assert read(blob, ns, idt) == bytes(data)


def test_bf16_special_values_preserved_bitwise():
    ns = toy_ns()
    n = cf.payload_len(ns) // 2
    t = torch.zeros(n, dtype=torch.bfloat16)
    t[:4] = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.0])
    raw = t.view(torch.int16)
    raw[4] = 0x7FC1  # a non-default NaN payload
    raw[5] = 0x0001  # smallest subnormal
    data = bytearray(t.view(torch.uint8).numpy().tobytes())
    idt = ident(ns, tokens(4), 0)
    out = read(chunk_file_bytes(ns, idt, data), ns, idt)
    assert out == bytes(data)
    assert torch.equal(torch.frombuffer(bytearray(out), dtype=torch.int16)[:6], raw[:6])


def test_truncation_at_every_offset_is_a_miss():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = chunk_file_bytes(ns, idt, payload(ns))
    for cut in range(len(blob)):
        assert reason(blob[:cut], ns, idt) in {"truncated", "size_mismatch"}, cut
    assert reason(blob + b"\0", ns, idt) == "size_mismatch"


def test_every_header_bit_flip_is_a_miss():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = bytearray(chunk_file_bytes(ns, idt, payload(ns)))
    header_len = cf.file_size(ns) - cf.payload_len(ns) - 4
    for byte in range(header_len):
        for bit in range(8):
            bad = bytearray(blob)
            bad[byte] ^= 1 << bit
            with pytest.raises(ChunkReadError):
                read(bytes(bad), ns, idt)


def test_payload_and_trailer_bit_flips_are_payload_crc_misses():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = bytearray(chunk_file_bytes(ns, idt, payload(ns)))
    start = cf.file_size(ns) - cf.payload_len(ns) - 4
    for pos in list(range(start, len(blob)))[::7] + [len(blob) - 1]:
        bad = bytearray(blob)
        bad[pos] ^= 0x10
        assert reason(bytes(bad), ns, idt) == "payload_crc", pos


@pytest.mark.parametrize("change, expected", [
    (dict(index=2), "identity_mismatch:index"),
    (dict(parent=b"\x01" * 32), "identity_mismatch:parent"),
    (dict(tokens=(100, 101, 102, 999)), "identity_mismatch:tokens"),
    (dict(key=b"\x02" * 32), "identity_mismatch:key"),
])
def test_wrong_identity_is_a_miss(change, expected):
    ns = toy_ns()
    written = ident(ns, tokens(4), 0)
    blob = chunk_file_bytes(ns, written, payload(ns))
    assert reason(blob, ns, dataclasses.replace(written, **change)) == expected


def test_inconsistent_expected_identity_detected():
    ns = toy_ns()
    written = ident(ns, tokens(4), 0)
    other_key = ident(ns, tokens(4, start=500), 0).key
    blob = bytearray(chunk_file_bytes(ns, written, payload(ns)))
    # Stored key replaced by a (validly hashed) key of other tokens; header CRC recomputed.
    blob[44:76] = other_key
    _fix_header_crc(blob, ns)
    assert reason(bytes(blob), ns, dataclasses.replace(written, key=other_key)) == "identity_mismatch:key_inconsistent"


@pytest.mark.parametrize("field, value", [("revision", "f" * 40), ("model_id", "x/y"), ("attn_implementation", "eager"),
                                          ("torch_version", "9.9"), ("rope_parameters", {"rope_theta": 1.0, "rope_type": "default"})])
def test_wrong_namespace_is_a_diagnosed_miss(field, value):
    written_ns = toy_ns(**{field: value})
    expected_ns = toy_ns()
    blob = chunk_file_bytes(written_ns, ident(written_ns, tokens(4), 0), payload(written_ns))
    assert reason(blob, expected_ns, ident(expected_ns, tokens(4), 0)) == f"namespace_mismatch:{field}"


@pytest.mark.parametrize("field, value", [("num_layers", 3), ("num_kv_heads", 2), ("head_dim", 16),
                                          ("kv_dtype", "float16"), ("chunk_size", 8)])
def test_wrong_layout_is_a_miss(field, value):
    written_ns = toy_ns(**{field: value})
    expected_ns = toy_ns()
    toks = tokens(written_ns.chunk_size)
    blob = chunk_file_bytes(written_ns, ident(written_ns, toks, 0), payload(written_ns))
    expected = ident(expected_ns, tokens(4), 0)
    got = reason(blob, expected_ns, expected)
    assert got.startswith("layout_mismatch:"), got


def _fix_header_crc(blob: bytearray, ns):
    tcount, meta_len = struct.unpack_from("<I", blob, 112)[0], struct.unpack_from("<I", blob, 128)[0]
    crc = zlib.crc32(bytes(blob[144:144 + 4 * tcount + meta_len]), zlib.crc32(bytes(blob[:140])))
    struct.pack_into("<I", blob, 140, crc)


def test_meta_inconsistent_with_digest_is_a_miss():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = bytearray(chunk_file_bytes(ns, idt, payload(ns)))
    meta_start = 144 + 4 * ns.chunk_size
    i = blob.index(b"toy/model", meta_start)
    blob[i:i + 3] = b"TOY"
    _fix_header_crc(blob, ns)
    assert reason(bytes(blob), ns, idt) == "namespace_mismatch:meta_inconsistent"


@pytest.mark.parametrize("offset, value, expected", [
    (132, 2**60, "layout_mismatch:payload_len"),   # payload_len (u64)
    (128, 2**31, "bad_meta_len"),                  # meta_len
    (112, 2**31, "layout_mismatch:token_count"),   # token_count
])
def test_huge_lengths_rejected_before_any_large_allocation(offset, value, expected):
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = bytearray(chunk_file_bytes(ns, idt, payload(ns)))
    struct.pack_into("<Q" if offset == 132 else "<I", blob, offset, value)
    data = bytes(blob)
    dest = bytearray(cf.payload_len(ns))
    tracemalloc.start()
    try:
        with pytest.raises(ChunkReadError) as e:
            cf.read_chunk_into(io.BytesIO(data), len(data), ns, idt, dest)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert e.value.reason == expected
    assert peak < 64 * 1024, peak  # only the bounded 144-byte header was read


def test_wrong_destination_size_is_a_programming_error():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    blob = chunk_file_bytes(ns, idt, payload(ns))
    with pytest.raises(ValueError):
        cf.read_chunk_into(io.BytesIO(blob), len(blob), ns, idt, bytearray(10))


def test_short_writes_are_completed_and_zero_progress_raises():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    data = payload(ns)
    buf = io.BytesIO()
    cf.write_chunk(buf, cf.encode_header(ns, idt), data, write_fn=lambda f, mv: f.write(mv[:7]))
    assert read(buf.getvalue(), ns, idt) == bytes(data)
    with pytest.raises(OSError) as e:
        cf.write_chunk(io.BytesIO(), cf.encode_header(ns, idt), data, write_fn=lambda f, mv: 0)
    assert e.value.errno == errno.EIO


def test_short_reads_are_completed():
    ns = toy_ns()
    idt = ident(ns, tokens(4), 0)
    data = payload(ns)
    blob = chunk_file_bytes(ns, idt, data)
    assert read(blob, ns, idt, readinto_fn=lambda f, mv: f.readinto(mv[:5])) == bytes(data)


def test_encode_rejects_foreign_identity():
    ns = toy_ns()
    other = toy_ns(revision="9" * 40)
    with pytest.raises(ValueError):
        cf.encode_header(ns, ident(other, tokens(4), 0))
    with pytest.raises(ValueError):
        cf.encode_header(ns, ChunkIdentity(ns.digest(), b"\0" * 32, b"\0" * 32, 0, (1, 2)))
