# M5 evaluation analysis (kvlab-m5-analysis-v1)

Scope: One machine, model and software stack; batch 1; synchronous first-token latency from a controlled, warm-page-cache state; the declared case grid only (fixed, not sampled). The holdout tests new request/state configurations and token content inside the calibrated box, not other workloads, machines, cold storage, concurrency, trace-level cache dynamics, tail latency or service-level TTFT.

## Delivered policy latency (paired per repetition; equal case weights)

| Difference | Mean ms | Interval ms | Direction |
|---|---:|---|---|
| cost_aware_minus_always_restore | 1.037 | [-0.07146347484374997, 2.1462962539062485] | inconclusive |
| cost_aware_minus_recompute | -46.479 | [-48.010156823593746, -44.972222064921866] | below 0 |
| always_restore_minus_recompute | -47.516 | [-48.98454288296875, -45.960384399843754] | below 0 |

## Candidate selection quality (FixedChunks latencies of the selected candidates)

| Policy | Agreement (noisy point estimate) | Naive mean regret ms | Cross-evaluated mean ms |
|---|---:|---:|---:|
| always_restore | 0.750 | 0.706 | -0.011 |
| recompute | 0.125 | 48.271 | 47.554 |
| cost_aware | 0.750 | 0.706 | -0.011 |

Prediction error: {'n_rows': 47, 'n_unavailable': 0, 'row_weight': 'equal per candidate', 'mean_signed_error_ms': -0.6490024667558945, 'mean_abs_error_ms': 4.444747188100627, 'mean_abs_relative_error': 0.0500917644690017, 'max_abs_relative_error': 0.3115196765448545}

Ambiguous cases: ['ev-01', 'ev-08']
Out-of-domain fallbacks: []
