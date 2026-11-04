"""M0.1 GPU regressions: resident split (B) vs restored split (C), and negative controls.

Run: .venv/bin/python -m pytest -m "gpu and m0_1" -v
These checks are strict (bitwise / byte-exact); none of them uses the M0
full-prefill tolerance (atol=1.45).
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import pytest
import torch

from kvlab.experiment import gpu_resident_cache
from kvlab.fixtures import BOUNDARY_TEXT, NATURAL_PROMPTS, chat_ids, exact_length_ids
from kvlab.kv_snapshot import KVSnapshot, compare_snapshot_to_cache, export_prefix_to_cpu, restore_to_device

pytestmark = [pytest.mark.gpu, pytest.mark.m0_1]
C, STEPS = 256, 8


@pytest.fixture(scope="module")
def model_tok():
    if not torch.cuda.is_available():
        pytest.skip("BLOCKED: CUDA not available")
    return km.load_model()


def _decode(model, ids, cache, start):
    logits, cache = km.prefill(model, ids, cache, start)
    return km.greedy_continue(model, cache, logits, start + ids.shape[1], STEPS)


def _snapshot(model, ids, P):
    _, cache = km.prefill(model, ids[:, :P], None, 0)
    snap = export_prefix_to_cpu(cache, P, C)
    assert compare_snapshot_to_cache(snap, cache, P)["all_bytes_equal"]
    del cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return snap


def _ids(tok, name):
    if name in NATURAL_PROMPTS:
        return torch.tensor([chat_ids(tok, NATURAL_PROMPTS[name])], device="cuda")
    return torch.tensor([exact_length_ids(tok, BOUNDARY_TEXT, int(name.split("_")[1]))], device="cuda")


@pytest.mark.parametrize("name, L, P", [
    ("nat_long", 256, 256),       # the original M0 failing case
    ("boundary_513", 256, 512),   # partial restore
    ("boundary_513", 512, 512),
])
def test_restored_split_bitwise_equals_resident_split(model_tok, name, L, P):
    model, tok = model_tok
    ids = _ids(tok, name)
    cache_b = gpu_resident_cache(model, ids, P, L) if P > L else km.prefill(model, ids[:, :L], None, 0)[1]
    tok_b, steps_b = _decode(model, ids[:, L:], cache_b, L)
    del cache_b
    for _ in range(2):  # repeated, each with a fresh export and a fresh restored cache
        snap = _snapshot(model, ids, P)
        cache_c = restore_to_device(snap, L, model.config, "cuda")
        assert compare_snapshot_to_cache(snap, cache_c, L)["all_bytes_equal"]
        tok_c, steps_c = _decode(model, ids[:, L:], cache_c, L)
        assert tok_c == tok_b
        assert all(torch.equal(x, y) for x, y in zip(steps_b, steps_c)), "restored != resident (bitwise)"
        del cache_c, snap


def _clone(snap):
    return KVSnapshot(snap.chunk_size, snap.num_layers,
                      [[(k.clone(), v.clone()) for k, v in layers] for layers in snap.chunks])


def test_corrupted_snapshot_fails_byte_check(model_tok):
    """A one-bit flip may not change any logit; the byte check must still catch it."""
    model, tok = model_tok
    ids = _ids(tok, "boundary_513")
    snap = _snapshot(model, ids, 512)
    _, written = km.prefill(model, ids[:, :512], None, 0)  # what the cache writer produced
    bad = _clone(snap)
    bad.chunks[0][13][0].view(torch.int16).view(-1)[1000] ^= 1
    check = compare_snapshot_to_cache(bad, written, 512)
    assert not check["all_bytes_equal"]
    assert [(m["chunk"], m["layer"], m["kind"]) for m in check["mismatches"]] == [(0, 13, "k")]


def test_wrong_prefix_detected_by_byte_check(model_tok):
    """A compatible-shape snapshot of a prefix differing in ONE token. In the
    M0.1 audit the full-prefill tolerance does not catch this (max_abs 0.94 <
    atol 1.45), so detection must come from byte identity against what was
    written (and, in M1, from the token-chain identity key)."""
    model, tok = model_tok
    ids = _ids(tok, "nat_long")
    L = 256
    other = ids[:, :L].clone()
    other[0, L // 2] = (other[0, L // 2] + 1) % model.config.vocab_size
    wrong = _snapshot(model, other, L)
    _, written = km.prefill(model, ids[:, :L], None, 0)
    check = compare_snapshot_to_cache(wrong, written, L)
    assert not check["all_bytes_equal"] and len(check["mismatches"]) == 56  # every K/V tensor of the chunk
    del written
    cache_b = km.prefill(model, ids[:, :L], None, 0)[1]
    logits_b, _ = km.prefill(model, ids[:, L:], cache_b, L)
    cache_w = restore_to_device(wrong, L, model.config, "cuda")
    logits_w, _ = km.prefill(model, ids[:, L:], cache_w, L)
    assert not torch.equal(logits_b, logits_w)


def test_position_guard_rejects_offset(model_tok):
    """A +1 query-position shift is within the M0 tolerance in bf16; the explicit
    cache-length/start_pos invariant in prefill() must reject it."""
    model, tok = model_tok
    ids = _ids(tok, "boundary_513")
    snap = _snapshot(model, ids, 512)
    cache = restore_to_device(snap, 512, model.config, "cuda")
    with pytest.raises(ValueError, match="cache length 512 != start_pos 513"):
        km.prefill(model, ids[:, 512:], cache, 513)
