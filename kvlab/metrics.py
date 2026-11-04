"""Numerical comparison of a reference logit vector against a test vector.

Definitions (both inputs cast to float32):
- max_abs_err  = max_i |test_i - ref_i|
- rel_l2_err   = ||test - ref||_2 / ||ref||_2        (documented relative measure)
- allclose     = all_i |test_i - ref_i| <= atol + rtol * |ref_i|
- top2_margin  = ref[top1] - ref[top2]  (how close the reference was to a tie)
"""

import torch


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def compare_logits(ref: torch.Tensor, test: torch.Tensor, atol: float, rtol: float) -> dict:
    ref, test = ref.float(), test.float()
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(test.shape)}")
    ref_ok, test_ok = finite(ref), finite(test)
    result = {"ref_finite": ref_ok, "test_finite": test_ok}
    if not (ref_ok and test_ok):
        result["allclose"] = False
        return result
    diff = (test - ref).abs()
    top2 = torch.topk(ref, 2)
    result.update({
        "max_abs_err": float(diff.max()),
        "rel_l2_err": float(torch.linalg.vector_norm(test - ref) / torch.linalg.vector_norm(ref)),
        "allclose": bool((diff <= atol + rtol * ref.abs()).all()),
        "atol": atol,
        "rtol": rtol,
        "ref_argmax": int(top2.indices[0]),
        "test_argmax": int(torch.argmax(test)),
        "ref_top2_margin": float(top2.values[0] - top2.values[1]),
    })
    result["argmax_equal"] = result["ref_argmax"] == result["test_argmax"]
    return result


def greedy_agreement(ref_tokens: list[int], test_tokens: list[int]) -> dict:
    if len(ref_tokens) != len(test_tokens):
        raise ValueError("greedy sequences must have equal length")
    first_div = next((i for i, (a, b) in enumerate(zip(ref_tokens, test_tokens)) if a != b), None)
    return {
        "steps": len(ref_tokens),
        "matching_positions": sum(a == b for a, b in zip(ref_tokens, test_tokens)),
        "identical": first_div is None,
        "first_divergence": first_div,
    }
