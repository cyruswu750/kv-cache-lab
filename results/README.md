# Evidence

Raw evidence and analysis outputs of every experiment, kept byte for byte as produced: freeze
manifests, protocol hashes and analyzers hash these files, so they are never edited. What each
experiment shows, and its scope, is in [docs/results.md](../docs/results.md). Smoke and probe
directories are pipeline checks and are not used for any claim.

| Directory | Experiment | Status | Main outputs |
|---|---|---|---|
| `m0/` | KV round trip GPU → CPU → GPU; tolerance calibration | `nat_long` full-prefill divergence recorded (D10) | `m0_results.json`, `divergence_nat_long.json`, `environment.json`, `pip_freeze.txt` |
| `m0_1/` | Audit: restored vs resident split, negative controls | gate passed | `audit_bf16_*.json`, `audit_fp32_*.json` |
| `m0_reruns/` | M0 smoke reruns | pipeline check | |
| `m1/`, `m1_1/`, `m1_2/` | Tiered-cache evidence runs and mutation checks | M1.2 is the corrected run (VAL-004); earlier runs kept | `m1_2/m1_run_20261103T121800.json` |
| `m2/` | Latency of full/host/disk paths; offline cost model | qualified by VAL-005 | `run_20261103T131543/` |
| `m2_1/` | Dependence-aware reanalysis; slow-mode diagnosis | diagnostic | `reanalysis/`, `diag_*/` |
| `m3/` | Baseline vs batched restore (exploratory A/B); VAL-006 reanalysis | valid; adoption rule not met (R3) | `eval_20261103T173632/`, `reanalysis_val006_*/`, `protocol_freeze_*.sha256` |
| `m3_confirm/` | M3_CONFIRM_V1 preregistered confirmation | valid; confirmed | `M3_CONFIRM_V1_analysis/`, `M3_CONFIRM_V1.freeze.json` |
| `m4/` | Trace evaluation M4_EVAL_V1 and M4_EVAL_V2 | V1 invalid (no verdicts); V2 valid | `M4_EVAL_V2_analysis/`, `M4_EVAL_V1_analysis/m4_validation.json` |
| `m5/` | Restore-length policy: calibration, fit, held-out evaluation | valid; no incremental benefit demonstrated | `M5_EVAL_V1_analysis/`, `M5_FIT_V1/`, `SHA256SUMS` |

Some files record absolute paths of the checkout they were produced in; those paths are part of the
recorded evidence. The `pre_*`, `post_*`, `change_inventory_*`, `review_archive*` and `CHECKPOINT.txt`
files in the milestone directories are snapshot manifests and checkpoints of the working tree, kept as
recorded; they are bookkeeping, and no script, test or analysis reads them.
