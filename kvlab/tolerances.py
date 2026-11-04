"""Frozen M0 numerical tolerances for first-next-token logits (bf16, SDPA).

Derived by the rule in docs/decisions.md D7, which was fixed before calibration:
    atol = max(0.25, ceil_to_0.05(4 * M)), rtol = 0.02
with M = 0.353515625, the largest max_abs_err of full prefill vs GPU-resident
chunked prefill measured by scripts/m0_calibrate.py
(results/m0/calibration.json, 2026-11-02). Frozen before scripts/m0_experiment.py
was run for the first time. Do not edit these values to make a failing case pass.
"""

ATOL = 1.45
RTOL = 0.02
CALIBRATION_M = 0.353515625
