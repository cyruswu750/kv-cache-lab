"""M0 experiment: full recomputation (path A) vs CPU-snapshot restore (path B).

For one prompt of N tokens and chunk size C:
  A. prefill all N tokens into a fresh cache; greedy-decode.
  B. prefill only the prefix [0, Lmax) into a separate fresh cache, export it to
     CPU, delete the GPU cache, then for every eligible L <= Lmax:
       restore [0, L) from the CPU snapshot into a new cache, prefill [L, N),
       greedy-decode.
Every path builds its own cache; no cache object is shared between paths.
"""

import gc

import torch
from transformers import DynamicCache

from kvlab.boundaries import eligible_boundaries
from kvlab.kv_snapshot import (compare_snapshot_to_cache, export_prefix_to_cpu, fingerprint,
                               restore_to_device, zeroed_copy)
from kvlab.metrics import compare_logits, finite, greedy_agreement
from kvlab.model import greedy_continue, prefill


def _free_gpu():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _decode(model, ids, cache, start_pos, steps):
    """Prefill ids at start_pos onto cache, then greedy-decode `steps` tokens."""
    logits, cache = prefill(model, ids, cache, start_pos)
    tokens, step_logits = greedy_continue(model, cache, logits, start_pos + ids.shape[1], steps)
    return logits, tokens, step_logits


def gpu_resident_cache(model, ids, prefix_len, keep_len):
    """Reference without any CPU round trip: prefill [0, prefix_len) on GPU, then
    copy the first keep_len tokens GPU->GPU into a new cache."""
    _, src = prefill(model, ids[:, :prefix_len], None, 0)
    data = [(layer.keys[:, :, :keep_len, :].clone(), layer.values[:, :, :keep_len, :].clone())
            for layer in src.layers]
    del src
    return DynamicCache(ddp_cache_data=data, config=model.config)


def run_case(model, name, token_ids, chunk_size, atol, rtol, greedy_steps=8,
             negative_control=True, gpu_resident_check=True):
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], device=device)
    n = ids.shape[1]
    bounds = eligible_boundaries(n, chunk_size)
    record = {"name": name, "prompt_len": n, "chunk_size": chunk_size,
              "eligible_restore_lens": bounds, "restores": []}

    # Path A: full recomputation in its own cache.
    logits_a, tokens_a, steps_a = _decode(model, ids, None, 0, greedy_steps)
    record["full"] = {"first_logits_finite": finite(logits_a),
                      "all_step_logits_finite": all(finite(t) for t in steps_a),
                      "greedy_tokens": tokens_a}
    _free_gpu()
    if not bounds:
        record["restore_len_max"] = 0
        record["status_note"] = "no eligible chunk boundary (L must satisfy L % C == 0 and 0 < L <= N-1)"
        return record

    # Path B, step 1: compute prefix in a separate cache and export to CPU.
    l_max = bounds[-1]
    record["restore_len_max"] = l_max
    _, prefix_cache = prefill(model, ids[:, :l_max], None, 0)
    snapshot = export_prefix_to_cpu(prefix_cache, l_max, chunk_size)
    record["export_check"] = compare_snapshot_to_cache(snapshot, prefix_cache, l_max)
    record["snapshot_bytes"] = snapshot.nbytes()
    fp_before = fingerprint(snapshot)
    # Drop the source GPU cache so restoration can only come from the CPU snapshot.
    del prefix_cache
    _free_gpu()

    for restore_len in bounds:
        cache_b = restore_to_device(snapshot, restore_len, model.config, device)
        restore_check = compare_snapshot_to_cache(snapshot, cache_b, restore_len)
        suffix = ids[:, restore_len:]
        logits_b, tokens_b, steps_b = _decode(model, suffix, cache_b, restore_len, greedy_steps)
        del cache_b
        entry = {
            "restore_len": restore_len,
            "suffix_len": suffix.shape[1],
            "partial": restore_len < l_max,
            "restore_check": restore_check,
            "first_logits": compare_logits(logits_a, logits_b, atol, rtol),
            "all_step_logits_finite": all(finite(t) for t in steps_b),
            "greedy_tokens": tokens_b,
            "greedy": greedy_agreement(tokens_a, tokens_b),
        }
        if gpu_resident_check:
            # Informational: same chunked computation with the cache kept on GPU.
            cache_g = gpu_resident_cache(model, ids, l_max, restore_len)
            logits_g, _ = prefill(model, suffix, cache_g, restore_len)
            entry["bitwise_equal_to_gpu_resident_chunked"] = bool(torch.equal(logits_g, logits_b))
            del cache_g
        if negative_control and restore_len == l_max:
            # Proof that the suffix forward actually reads the restored K/V:
            # an all-zero snapshot of the same shape must change the logits.
            cache_z = restore_to_device(zeroed_copy(snapshot), restore_len, model.config, device)
            logits_z, _ = prefill(model, suffix, cache_z, restore_len)
            neg = compare_logits(logits_a, logits_z, atol, rtol)
            entry["negative_control_zero_kv"] = {"max_abs_err": neg.get("max_abs_err"),
                                                 "allclose": neg["allclose"],
                                                 "argmax_equal": neg.get("argmax_equal")}
            del cache_z
        record["restores"].append(entry)
        _free_gpu()

    record["snapshot_unchanged"] = fingerprint(snapshot) == fp_before
    del snapshot
    _free_gpu()
    return record


def case_passed(record) -> bool:
    """PASS criteria, fixed before acceptance evaluation (docs/decisions.md)."""
    full = record["full"]
    if not (full["first_logits_finite"] and full["all_step_logits_finite"]):
        return False
    if not record["restores"]:
        return record["restore_len_max"] == 0  # nothing eligible to restore
    if not (record["export_check"]["all_bytes_equal"] and record["snapshot_unchanged"]):
        return False
    for r in record["restores"]:
        fl = r["first_logits"]
        if not (r["restore_check"]["all_bytes_equal"] and fl["ref_finite"] and fl["test_finite"]
                and r["all_step_logits_finite"] and fl["allclose"] and fl["argmax_equal"]
                and r["greedy"]["identical"]):
            return False
        neg = r.get("negative_control_zero_kv")
        if neg is not None and neg["allclose"]:
            return False  # restored data had no observable effect
    return True
