"""CPU tests for snapshot export/restore using synthetic DynamicCache contents."""

import pytest
import torch
from transformers import DynamicCache

from kvlab.kv_snapshot import (bytes_equal, compare_snapshot_to_cache, export_prefix_to_cpu, fingerprint,
                               restore_to_device, zeroed_copy)
from kvlab.metrics import compare_logits, greedy_agreement

LAYERS, HEADS, DIM, C = 3, 2, 8, 4


def make_cache(seq_len, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = [(torch.randn(1, HEADS, seq_len, DIM, generator=g).to(dtype),
             torch.randn(1, HEADS, seq_len, DIM, generator=g).to(dtype)) for _ in range(LAYERS)]
    return DynamicCache(ddp_cache_data=data)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_round_trip_bytes_equal(dtype):
    cache = make_cache(10, dtype)
    snap = export_prefix_to_cpu(cache, 8, C)
    assert snap.num_tokens == 8 and len(snap.chunks) == 2
    check = compare_snapshot_to_cache(snap, cache, 8)
    assert check["all_bytes_equal"] and check["tensors_checked"] == 2 * LAYERS * 2
    restored = restore_to_device(snap, 8, None, "cpu")
    assert restored.get_seq_length() == 8
    for src, dst in zip(cache.layers, restored.layers):
        assert bytes_equal(src.keys[:, :, :8, :], dst.keys)
        assert bytes_equal(src.values[:, :, :8, :], dst.values)
        assert dst.keys.dtype == dtype


def test_partial_restore_uses_leading_chunks():
    cache = make_cache(12)
    snap = export_prefix_to_cpu(cache, 12, C)
    restored = restore_to_device(snap, 4, None, "cpu")
    assert restored.get_seq_length() == 4
    for src, dst in zip(cache.layers, restored.layers):
        assert bytes_equal(src.keys[:, :, :4, :], dst.keys)


def test_snapshot_is_independent_of_source():
    cache = make_cache(8)
    snap = export_prefix_to_cpu(cache, 8, C)
    fp = fingerprint(snap)
    src_ptrs = {layer.keys.untyped_storage().data_ptr() for layer in cache.layers}
    snap_ptrs = {t.untyped_storage().data_ptr() for *_, t in snap.tensors()}
    assert not (src_ptrs & snap_ptrs)
    assert len(snap_ptrs) == 2 * LAYERS * 2  # every snapshot tensor owns its own storage
    for layer in cache.layers:  # mutate the source after export
        layer.keys.zero_()
        layer.values.zero_()
    assert fingerprint(snap) == fp
    assert not compare_snapshot_to_cache(snap, cache, 8)["all_bytes_equal"]


def test_restored_cache_appends_do_not_touch_snapshot():
    cache = make_cache(8)
    snap = export_prefix_to_cpu(cache, 8, C)
    fp = fingerprint(snap)
    restored = restore_to_device(snap, 8, None, "cpu")
    for li in range(LAYERS):
        restored.update(torch.ones(1, HEADS, 3, DIM, dtype=torch.bfloat16),
                        torch.ones(1, HEADS, 3, DIM, dtype=torch.bfloat16), li)
        restored.layers[li].keys[..., 0, :].fill_(7)  # write into the restored prefix
    assert restored.get_seq_length() == 11
    assert fingerprint(snap) == fp


def test_restore_twice_gives_independent_caches():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    a = restore_to_device(snap, 8, None, "cpu")
    b = restore_to_device(snap, 8, None, "cpu")
    a.layers[0].keys.fill_(3)
    assert not bytes_equal(a.layers[0].keys, b.layers[0].keys)


def test_detects_byte_mismatch():
    cache = make_cache(8)
    snap = export_prefix_to_cpu(cache, 8, C)
    snap.chunks[1][2][1].view(torch.int16)[0, 0, 0, 0] ^= 1  # flip one bit in a V tensor
    check = compare_snapshot_to_cache(snap, cache, 8)
    assert not check["all_bytes_equal"]
    assert check["mismatches"] == [{"chunk": 1, "layer": 2, "kind": "v", "snap_shape": [1, HEADS, C, DIM],
                                    "live_shape": [1, HEADS, C, DIM], "snap_dtype": "torch.bfloat16",
                                    "live_dtype": "torch.bfloat16"}]


def test_bytes_equal_distinguishes_dtype_and_negative_zero():
    a = torch.zeros(4, dtype=torch.float32)
    assert not bytes_equal(a, -a)          # +0.0 vs -0.0: numerically equal, bytes differ
    assert not bytes_equal(a, a.half())
    nan = torch.full((2,), float("nan"))
    assert bytes_equal(nan, nan.clone())   # identical NaN payloads compare equal as bytes


@pytest.mark.parametrize("n_tokens", [0, 3, 5])
def test_export_rejects_non_chunk_lengths(n_tokens):
    with pytest.raises(ValueError):
        export_prefix_to_cpu(make_cache(8), n_tokens, C)


def test_export_rejects_short_cache():
    with pytest.raises(ValueError):
        export_prefix_to_cpu(make_cache(6), 8, C)


def test_restore_rejects_too_long_or_misaligned():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    with pytest.raises(ValueError):
        restore_to_device(snap, 12, None, "cpu")
    with pytest.raises(ValueError):
        restore_to_device(snap, 6, None, "cpu")


def test_zeroed_copy_shapes():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    z = zeroed_copy(snap)
    assert z.num_tokens == snap.num_tokens and z.nbytes() == snap.nbytes()
    assert all(int(t.abs().sum()) == 0 for *_, t in z.tensors())


def test_compare_logits_metrics():
    ref = torch.tensor([1.0, 3.0, 2.0])
    out = compare_logits(ref, ref + torch.tensor([0.0, 0.1, -0.2]), atol=0.25, rtol=0.0)
    assert out["allclose"] and out["argmax_equal"]
    assert out["max_abs_err"] == pytest.approx(0.2)
    assert out["ref_top2_margin"] == pytest.approx(1.0)
    assert not compare_logits(ref, ref + 1.0, atol=0.25, rtol=0.0)["allclose"]


def test_compare_logits_rejects_nonfinite():
    ref = torch.tensor([1.0, 2.0])
    out = compare_logits(ref, torch.tensor([1.0, float("inf")]), atol=10.0, rtol=1.0)
    assert out["test_finite"] is False and out["allclose"] is False


def test_greedy_agreement():
    assert greedy_agreement([1, 2, 3], [1, 2, 3])["identical"]
    g = greedy_agreement([1, 2, 3], [1, 5, 3])
    assert g["first_divergence"] == 1 and g["matching_positions"] == 2


# --- M0.1 regression coverage: restore must refuse, not broadcast or cast ---

def test_restore_rejects_broadcastable_wrong_shape():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    k, v = snap.chunks[1][0]
    snap.chunks[1][0] = (k[:, :1].contiguous(), v)  # 1 head instead of 2: copy_ would broadcast it
    with pytest.raises(ValueError, match="expected"):
        restore_to_device(snap, 8, None, "cpu")


def test_restore_rejects_mixed_dtype():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    k, v = snap.chunks[0][2]
    snap.chunks[0][2] = (k, v.float())  # copy_ would silently cast it back to bf16
    with pytest.raises(ValueError, match="expected"):
        restore_to_device(snap, 8, None, "cpu")


def test_compare_rejects_layer_count_mismatch():
    snap = export_prefix_to_cpu(make_cache(8), 8, C)
    fewer = DynamicCache(ddp_cache_data=[(layer.keys, layer.values) for layer in make_cache(8).layers[:2]])
    with pytest.raises(ValueError, match="layers"):
        compare_snapshot_to_cache(snap, fewer, 8)
