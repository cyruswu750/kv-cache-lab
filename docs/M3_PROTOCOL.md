# M3 performance protocol: baseline vs batched restore (frozen 2026-11-03, before any evaluation sample)

Diagnostic comparison of two restore backends on the same machine, software and state. It is
not a TTFT benchmark, and it is not a cost model. Implementation: `scripts/m3_measure.py`,
`kvlab/m3_bench.py` (on top of the unchanged M2 `kvlab/perf_harness.py`). Analysis:
`scripts/m3_analyze.py`.

## 1. Question and hypotheses
- **H1:** the batched backend lowers end-to-end first-token latency on the host and warm-disk
  restore paths for restored prefixes of ≥ 4 chunks.
- **H2:** it lowers the restore-phase time on both tiers at every restored length.
- **H3:** it does not make any case slower, including the 1-chunk case. The effect there is
  expected to be small: 56 vs 2 transfers.
- Context only (no hypothesis): the prepared-state recompute baseline (`full`, D28).

## 2. What is fixed
- Model, dtype, SDPA and C = 256 as in M2. A single process holds the GPU at a time. No power,
  driver, clock or system setting is changed.
- **Timed region:** unchanged M2 (`Bench.measure`): `torch.cuda.synchronize(); t0;
  engine.prefill_request(prompt); int(argmax); t1`.
  - Inside it: ingest, restore (including the batched staging, its placement copies and the
    D4 constructor copy), suffix prefill, admission D2H and first-token selection.
  - Outside it: preparation, page-cache handling, resets, memory statistics and every check.
- **State:** store A holds prefix chunks [0, L) in pinned host slots; store B holds the same
  chunks as warm files (100% page-cache resident, verified per sample), with no prefix chunk
  in host. Both backends use the same stores; only `restore_backend` is switched before a
  sample. Admitted chunks are discarded after every sample, so every sample starts from the
  same state.
- `restore_staging_chunks` G = 4 (fixed during development; not tuned on evaluation data).

## 3. Cases
- **Tuning / development cases (excluded from evaluation):** (512, 1) and (2048, 64). Used by
  the smoke run `results/m3/smoke_*`, and (512, 1)-like lengths by the tests.
- **Evaluation cases (L, S):**

  | case | chunks | role |
  |---|---|---|
  | (256, 32) | 1 | short; fresh combination |
  | (1024, 1) | 4 | medium; an M2 grid case |
  | (1536, 128) | 6 | medium; fresh, crosses a staging-group boundary (4 + 2) |
  | (3072, 256) | 12 | long; fresh; **admits one new chunk** (N = 3328 → 13 complete chunks) |
  | (4096, 64) | 16 | long; the M2.1 diagnostic case |

## 4. Sampling
- **5 sessions** (s1–s5), each a fresh process, run one after another. Seeds 31001–31005.
- Per session: the 5 cases in a random order, repeated twice (blocks = 2), so each case is
  measured in two separated stretches per session.
- Per block: 2 warmup repetitions (kept, flagged, never used) + 12 measured repetitions.
  `gc.collect()` runs before each repetition.
- Each repetition runs these 9 samples in a fresh random order:
  - `full` (plain);
  - host and disk with the baseline backend (plain);
  - host and disk with the batched backend (plain);
  - the same 4 tier × backend samples again in the synchronized-phase mode (restore-phase
    timing).
- A process warmup at the start runs every combination twice at the smallest and largest
  case (flagged).
- **No sample is filtered.** Slow samples stay. Exclusions happen only for the M2 validity
  rules plus "the requested backend ran with the expected copy count".

## 5. Untimed checks (each session, first block of each case)
- **M2 strict correctness for each backend:** restored K/V bytes equal the source, and host
  == disk == resident split logits bitwise. The full-vs-restored comparison is report-only (D10).
- **A/B check:** one request per backend and tier under identical state. Every K/V byte
  after the suffix prefill, the first-token logits, 8 greedy steps' logits and tokens, and the
  admissions must be equal.
- **Restore-only device memory per backend:** peak above start, and bytes kept after return.

## 6. Statistics (fixed now)
- **Units:** milliseconds.
- **Pairs:** within a repetition, d = batched − baseline for the same tier and mode.
  - Primary: plain e2e.
  - Secondary: the synced restore phase.
  - Context: d = restore path − `full` per backend.
- **Uncertainty:** a stratified circular moving-block bootstrap.
  - Strata are the contiguous blocks (session × block, 12 repetitions each, chronological).
    Blocks are resampled within each stratum, with block lengths b ∈ {1, 2, 4, 8}.
  - 4000 resamples, 95% percentile intervals, for the median and the mean of d.
- **Classification per (case, tier, statistic):** "batched faster" / "batched slower" only
  if all of the following hold; otherwise **inconclusive**:
  - **R1:** the interval excludes 0 in the same direction for every b;
  - **R2:** the point estimate has that sign in every one of the 5 sessions;
  - **R3:** the measured stretch is long relative to the slow regimes. The summed span of the
    case's blocks is ≥ 10× the longest slow regime observed in them, and the AR(1)
    effective sample size of d (per stratum, summed) is ≥ 10.

  A slow sample is e2e > 1.3× the minimum of its (case, tier, backend, mode); a regime is a
  maximal chronological run of slow samples, as in M2.1.
- **Speedups** (each with block-bootstrap intervals, b = 4; sensitivity reported):
  - `marginal_median_ratio` = median(baseline) / median(batched);
  - `marginal_mean_ratio` = mean(baseline) / mean(batched);
  - `median_paired_ratio` = median over repetitions of baseline_r / batched_r.

  These are different quantities and are reported separately.
- **Also reported:**
  - per-session medians and means of d, and their between-session range;
  - slow shares per backend;
  - the chronological sample order (CSV);
  - H2D bytes and copy calls per restore (from the restore record);
  - restore-only and per-request peak device memory, and the explicit scratch size.
- **Not reported:** p99, or any tail claim beyond slow shares.

## 7. Call counts (separate run)
`scripts/m3_measure.py --profile` runs in its own process after the timing sessions, on
(256, 32), (1536, 128) and (4096, 64), with 5 calls per unit. It counts CUDA runtime API
calls and device ops per `restore()` call and per whole request, for each backend and tier.
These are measured counts; the counts in `docs/M3_DESIGN.md` §4 are estimates. Profiler
timings are not used.

## 8. Decision rule (fixed now)
The evidence **supports adopting** the batched backend only if all of the following hold:
1. every correctness and A/B check passes in every session;
2. no (case, tier) has an established regression in the median or the mean;
3. the mean and the median of d (plain e2e) are established negative (R1–R3) for every case
   with ≥ 4 chunks, on both tiers.

The 1-chunk case may be inconclusive. Even if the rule is met, the default stays `baseline`
in M3, pending review. If it is not met, the result is reported as such, and the default
stays `baseline`.

## 9. Budget and limits
- At most 45 GPU-minutes for M3 in total, one GPU process at a time, and no full-grid rerun.
- Warm-disk only. Cold-NVMe latency stays BLOCKED/unclaimed (WSL; D26).
- Results are conditional on this sample, machine and software. The M2.1 slow mode is an
  observed association with higher host-side CUDA-call cost, not an established cause.
