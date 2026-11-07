# Documentation

## Start here
| Document | Contents |
|---|---|
| [design.md](design.md) | The implemented system: identity, chunk format, tiers, failure handling, restore backends, policies, evaluation tooling |
| [results.md](results.md) | Every experiment with its scope, verdicts and evidence; how to reproduce the analyses |
| [decisions.md](decisions.md) | Indexed design decisions (D1–D44) with alternatives and trade-offs |
| [BUGLOG.md](BUGLOG.md) | Defects found and fixed, with root cause and regression tests |
| [correctness-matrix.md](correctness-matrix.md) | Which test establishes which property |
| [environment.md](environment.md) | Machine, packages, model, and the observed `transformers` cache API |

## Experiment protocols
Each formal experiment was specified before it was measured. These documents are kept as records
of what was fixed in advance; status notes inside them describe the state when they were written.
Files marked **bound** are hashed by a freeze manifest and must stay byte-identical, so they are never
edited.

| Experiment | Protocol | Scripts | Evidence |
|---|---|---|---|
| M0, M0.1: KV round trip and audit | (in [decisions.md](decisions.md) D7–D12) | `m0_calibrate`, `m0_experiment`, `m0_diagnose_divergence`, `m0_1_audit`, `env_report` | `results/m0/`, `results/m0_1/` |
| M1: tiered cache evidence runs | (in [design.md](design.md)) | `m1_run`, `m1_mutation_check`, `m1_1_repro_review_issues` | `results/m1*/` |
| M2: latency and offline cost model | [M2_PROTOCOL.md](M2_PROTOCOL.md) | `m2_measure`, `m2_fit_predict`, `m2_score`, `m2_spike_probe` | `results/m2/` |
| M2.1: measurement audit, slow mode | [M2_1_DIAGNOSTIC_PLAN.md](M2_1_DIAGNOSTIC_PLAN.md) | `m2_1_reanalyze`, `m2_1_diagnose`, `m2_1_analyze_diag`, `m2_1_regimes` | `results/m2_1/` |
| M3: baseline vs batched restore | [M3_DESIGN.md](M3_DESIGN.md), [M3_PROTOCOL.md](M3_PROTOCOL.md) (bound) | `m3_measure`, `m3_analyze`, `m3_mutation_check` | `results/m3/` |
| M3_CONFIRM_V1: preregistered confirmation | [M3_CONFIRM_PROTOCOL.md](M3_CONFIRM_PROTOCOL.md) (bound), [M3_CONFIRM_CLARIFICATIONS.md](M3_CONFIRM_CLARIFICATIONS.md) | `m3_confirm`, `m3_analyze --spec M3_CONFIRM_V1`, `m3_confirm_reconstruct` | `results/m3_confirm/` |
| M4: trace evaluation (V1 invalid, V2 valid) | [M4B_PROTOCOL.md](M4B_PROTOCOL.md), [M4_EVAL_V2_PROTOCOL.md](M4_EVAL_V2_PROTOCOL.md) (bound) | `m4_gate`, `m4_eval`, `m4_measure`, `m4_analyze`, `m4_reconstruct` | `results/m4/` |
| M5: restore-length policy and cost model | [M5_DESIGN.md](M5_DESIGN.md), [M5_PROTOCOL_DRAFT.md](M5_PROTOCOL_DRAFT.md) | `m5_gate`, `m5_run`, `m5_fit`, `m5_analyze` | `results/m5/` |

Scripts live in `scripts/`. GPU scripts run one process at a time; analysis scripts are CPU-only and
refuse to overwrite existing outputs.

## Code map
- **Cache library:** `kvlab/identity.py`, `chunk_format.py`, `host_store.py`, `disk_store.py`,
  `tiered_cache.py`, `always_restore.py`, `restore_policy.py`, `restore_cost.py`, `boundaries.py`,
  `model.py`, `kv_snapshot.py`.
- **Verification:** `kvlab/resident_reference.py` (the strict oracle), `metrics.py`, `tolerances.py`,
  `experiment.py`, `fixtures.py`.
- **Evaluation tooling:** everything else in `kvlab/` (`perf*`, `diag_harness`, `m3_*`, `traces`,
  `gpu_prefix_cache`, `prefix_aware_cache`, `trace_*`, `warm_disk`, `m4_*`, `m5_*`). The experiment
  modules keep their names and locations because the freeze manifests and reconstruction tools refer to
  them by path.
- **Tests:** `tests/` (CPU, the default selection) and `tests/gpu/` (real model; `-m gpu`).
