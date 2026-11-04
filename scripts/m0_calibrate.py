"""Tolerance calibration for M0 (run BEFORE scripts/m0_experiment.py).

Measures, on CALIBRATION_TEXT only (disjoint from acceptance fixtures):
  - full vs full repeat (determinism of the same computation)
  - full vs GPU-resident chunked prefill (prefix [0, L) then suffix [L, N);
    no CPU round trip) -- the inherent chunked-vs-unchunked numerical gap.
Writes results/m0/calibration.json and prints the atol derived by the rule
in docs/decisions.md (D7). It does not edit kvlab/tolerances.py.

Usage: .venv/bin/python scripts/m0_calibrate.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402  (sets HF_HOME before transformers import)
import torch  # noqa: E402
from kvlab.boundaries import eligible_boundaries  # noqa: E402
from kvlab.experiment import gpu_resident_cache  # noqa: E402
from kvlab.fixtures import CALIBRATION_TEXT, exact_length_ids  # noqa: E402
from kvlab.metrics import compare_logits  # noqa: E402

CHUNK = 256
LENGTHS = [300, 400, 600, 700]
RTOL = 0.02


def main():
    out_path = km.PROJECT_ROOT / "results" / "m0" / "calibration.json"
    model, tok = km.load_model()
    rows = []
    for n in LENGTHS:
        ids = torch.tensor([exact_length_ids(tok, CALIBRATION_TEXT, n)], device="cuda")
        ref, _ = km.prefill(model, ids, None, 0)
        rep, _ = km.prefill(model, ids, None, 0)
        rows.append({"N": n, "L": 0, "kind": "full_vs_full_repeat",
                     **compare_logits(ref, rep, atol=0.0, rtol=0.0)})
        bounds = eligible_boundaries(n, CHUNK)
        for L in bounds:
            cache = gpu_resident_cache(model, ids, bounds[-1], L)
            test, _ = km.prefill(model, ids[:, L:], cache, L)
            rows.append({"N": n, "L": L, "kind": "full_vs_gpu_chunked",
                         **compare_logits(ref, test, atol=0.0, rtol=0.0)})
            del cache
        torch.cuda.empty_cache()

    chunked = [r for r in rows if r["kind"] == "full_vs_gpu_chunked"]
    m = max(r["max_abs_err"] for r in chunked)
    atol = max(0.25, math.ceil(4 * m / 0.05) * 0.05)
    summary = {
        "rule": "atol = max(0.25, ceil_to_0.05(4 * M)); rtol = 0.02 (docs/decisions.md D7)",
        "M_max_abs_err_full_vs_gpu_chunked": m,
        "max_rel_l2_full_vs_gpu_chunked": max(r["rel_l2_err"] for r in chunked),
        "full_vs_full_max_abs_err": max(r["max_abs_err"] for r in rows if r["kind"] == "full_vs_full_repeat"),
        "derived_atol": round(atol, 4),
        "rtol": RTOL,
        "argmax_all_equal": all(r["argmax_equal"] for r in rows),
    }
    out_path.write_text(json.dumps({"model": km.model_info(model), "chunk_size": CHUNK,
                                    "rows": rows, "summary": summary}, indent=2))
    for r in rows:
        print(f"{r['kind']:22s} N={r['N']:4d} L={r['L']:4d} max_abs={r['max_abs_err']:.4f} "
              f"rel_l2={r['rel_l2_err']:.2e} argmax_eq={r['argmax_equal']} margin={r['ref_top2_margin']:.3f}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
