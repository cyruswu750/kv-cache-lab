# M4_EVAL_V2 analysis (evaluation evidence)

Assumptions: Sessions are treated as independent, approximately normal replicates. Fresh processes do not prove independence: machine, driver or thermal regimes can persist across processes run back to back. Every interval and verdict is conditional on this assumption.

Primary family: 15 comparisons; verdicts need unanimity across 10 sessions and a Bonferroni-level t-interval excluding 0.

| trace | A | B | mean D (ms) | 95% t-interval | Bonferroni interval | unanimous | verdict |
|---|---|---|---|---|---|---|---|
| T1 | C2_base | C0 | -5.759 | [-14.382, 2.865] | [-20.833, 9.315] | no | inconclusive |
| T1 | C2_base | C1PA_HALF | 23.386 | [15.418, 31.355] | [9.458, 37.315] | no | inconclusive |
| T1 | C2_batch | C0 | -37.887 | [-42.885, -32.890] | [-46.623, -29.152] | neg | A_faster |
| T1 | C2_batch | C1PA_HALF | -8.742 | [-13.406, -4.078] | [-16.894, -0.590] | neg | A_faster |
| T2 | C2_base | C0 | -0.373 | [-6.208, 5.461] | [-10.572, 9.825] | no | inconclusive |
| T2 | C2_base | C1PA_HALF | 0.975 | [-4.327, 6.277] | [-8.292, 10.242] | no | inconclusive |
| T2 | C2_batch | C0 | -16.412 | [-20.182, -12.643] | [-23.001, -9.824] | neg | A_faster |
| T2 | C2_batch | C1PA_HALF | -15.064 | [-17.764, -12.365] | [-19.783, -10.345] | neg | A_faster |
| T3 | C2_base | C0 | 1.955 | [-5.714, 9.625] | [-11.451, 15.361] | no | inconclusive |
| T3 | C2_base | C1PA_HALF | 30.870 | [25.796, 35.944] | [22.001, 39.739] | pos | A_slower |
| T3 | C2_batch | C0 | -20.097 | [-24.840, -15.354] | [-28.388, -11.806] | neg | A_faster |
| T3 | C2_batch | C1PA_HALF | 8.818 | [6.792, 10.844] | [5.277, 12.359] | pos | A_slower |
| T0 | C2_base | C0 | 62.172 | [48.616, 75.728] | [38.476, 85.868] | pos | A_slower |
| T0 | C2_batch | C0 | 60.541 | [48.186, 72.896] | [38.945, 82.137] | pos | A_slower |
| T0 | C1PA_HALF | C0 | 5.905 | [2.826, 8.984] | [0.523, 11.286] | pos | A_slower |

Claim limits:
- synthetic traces only; batch 1, sequential, synchronous first-token latency (D22), not service TTFT
- configurations use different resources: C1 holds persistent GPU KV; C2 holds none on the GPU but a pinned host slab, a disk tier and an uncounted page cache; equal GPU budgets do not mean equal total resources
- every G below the device's free memory is an artificial GPU cap
- conditional on an interleaved, co-resident experiment: both C1 caches stay allocated while every configuration runs, and all share one allocator, page cache and thermal state; not standalone deployment
- allocator peaks are whole-process figures, never per-configuration GPU peaks
- warm disk (M4_EVAL_V2): each C2 request's candidate files (the disk files its restore reads on the normal path) were observed fully resident by mincore in one verification pass immediately before the request, after at most one prewarm read per file; these are per-file point observations, not a simultaneous snapshot, and not a guarantee at timer start or during the timed read (pages are not pinned); warm page cache only, no cold-NVMe, storage-device or production claim (D26); not procedurally comparable with M4_EVAL_V1's whole-tier preparation, and not a correction of V1
- C1 is this project's GPU-only baseline (prefix-aware LRU), not vLLM or SGLang; C1-LRU is offline only
- no p99/tail, throughput, concurrency or production claims; this machine, model and stack only
