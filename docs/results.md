# Results and limitations

This document summarizes what each milestone measured or verified, the exact scope of each
result, and where the raw evidence is. Design: `docs/design.md`. Decisions: `docs/decisions.md`.
Defects: `docs/BUGLOG.md`. Test-by-scenario table: `docs/correctness-matrix.md`.

**Scope of every number below:**
- **Machine and software:** one RTX 4080 Laptop GPU (12 GiB, Windows driver 596.49) under
  WSL2, with torch 2.14.1+cu130 and transformers 5.22.1.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct@989aa79…` in bf16 with SDPA, chunk size C = 256
  tokens.
- **Workload:** batch 1, a single process, sequential requests, tokenized input.

Latencies are synchronous first-token latencies, not service-level TTFT. Disk-tier results are
application-level hits with a warm Linux page cache. **Cold-NVMe latency and p99/tail latency
are not established.**

## Conclusions at a glance
- **Storage correctness is proven strictly.** Restored K/V bytes equal what was written. A
  restored request's step logits and tokens are bitwise equal to a resident-GPU replay with the
  same restore length and split (D12), and this holds for both restore backends.
- **Restored output is not claimed to equal full-prefill output** (D10). Full-recompute
  comparisons are report-only. In bf16 the two execution shapes round differently, and exact
  ties or 1-ulp margins can flip greedy tokens.
- **The original M0 strict suite still FAILs on `nat_long`** (greedy 6/8, an exact bf16 tie).
  That result is kept as recorded. It is not a defect and not an xfail.
- **M2:**
  - No restore-vs-recompute break-even was established for L = 256…4096.
  - The M2 cost model is offline only, and it describes the **baseline** restore backend. It
    was not refitted for the batched backend.
  - Its interval claims are qualified by VAL-005 (below).
- **M2.1:** restore-path latency on this platform alternates between regimes of different
  host-side CUDA-call cost. This is an association; the root cause is not established.
- **M3:** the `batched` restore backend is **opt-in**, and the default backend remains
  **`baseline`**. Under the frozen adoption rule, `supports_adoption` is **false**: the
  (1024, 1) case is inconclusive under the duration rule R3. Its speedups hold for the tested
  hardware, model, workloads and warm-disk condition only.
- **M3_CONFIRM_V1** is a separate, preregistered confirmation, frozen before measurement:
  - **Result:** all 8 primary verdicts are "batched faster", and
    `m3_confirm_v1_confirmed = true`.
  - **Scope:** two restoration microbenchmark cases, (1024, 1) and (1280, 96), on host and
    warm-disk tiers, on this machine and stack.
  - **What it does not change:** it is not pooled with M3, and M3's `supports_adoption`
    stays **false**. The default backend remains **`baseline`**: the confirmation does not
    authorize a default change.
- **M4** (trace evaluation of C0, C1-PA and C2; details and limits below):
  - **M4_EVAL_V1** ran once and is **invalid** under its frozen whole-tier warm-disk rule. It has
    no performance conclusion, and the cause of its residency failures is unresolved.
  - **M4_EVAL_V2**, with a request-scoped warm-disk preparation, ran once and is **valid**. Of
    its 15 primary comparisons, 5 are "A faster", 5 "A slower" and 5 inconclusive under the
    predeclared joint rule. Every verdict is conditional on the stated statistical assumptions
    and scope.
  - **The default backend remains `baseline`.** M4 does not change it.

## Test status (only runs that actually occurred)
| Run | Result | Log |
|---|---|---|
| Full CPU suite, after the first VAL-006 fix | **436 passed**, 57 deselected | `results/m3/reanalysis_val006_20261103T200427/logs/pytest_cpu_20261103T201405.log` |
| Focused M3 validation + statistics, after the VAL-006 warmup follow-up | **56 passed** | `results/m3/reanalysis_val006_warmups_20261103T202825/logs/pytest_m3_analyze_validation_20261103T203111.log` |
| GPU M1 suite, both restore backends (M3) | 23 passed | `results/m3/logs/pytest_gpu_m1_20261103T175241.log` |
| GPU M3 checks | 17 passed | `results/m3/logs/pytest_gpu_m3_20261103T175241.log` |
| GPU M0.1 regressions | 6 passed | `results/m3/logs/pytest_gpu_m0_1_20261103T175241.log` |
| GPU M0 strict (original criteria) | 10 passed, **1 FAILED (`nat_long`, known)** | `results/m3/logs/pytest_gpu_m0_strict_20261103T175241.log` |
| M1 evidence runner, baseline and batched | 19/19 strict records each, identical records and counters | `results/m3/m1_run_{baseline,batched}_20261103T175241.json` |
| Restore-backend mutation check | 10/10 killed (an earlier run: 9/10; a test was strengthened) | `results/m3/mutation_check_20261103T173143.txt` |

The full CPU suite was not rerun after the warmup follow-up. The GPU suites were last run in M3,
before VAL-006.

## M0 and M0.1: KV round trip, numerical contract, audit
**M0** (`results/m0/`, read-only) exported a real KV prefix GPU → CPU → GPU and restored it.
- **Byte checks and boundaries:** every export/restore byte check passed. The boundary cases
  N = 255, 256, 257, 511, 512 and 513 passed, and so did `nat_short`, `nat_medium` and
  `nat_xlong`.
- **`nat_long` (N = 426, L = 256) failed** greedy agreement, 6/8:
  - at step 6 the full-prefill reference has an exact bf16 tie, `' re'` = `' described'` =
    23.75;
  - the chunked path moves one logit by 1 ulp (0.125).

  The restored path is bitwise equal to a GPU-resident chunked prefill, so the gap comes from
  chunked vs unchunked bf16 compute, not from storage.
- **Tolerances** were fixed before any comparison: atol 1.45, rtol 0.02 (D7, D8).
- **M0 timings** are diagnostic only and are not used for any claim.

**M0.1** audited M0 (`results/m0_1/`). Gate: **PASS**. The original M0 strict result is
unchanged.
- **Restored vs resident split (C vs B):** bitwise equal on all 8 step logits and tokens, in 9
  bf16 splits × 3 repeats and in fp32. That is 4,032 bf16 tensor byte comparisons with 0
  mismatches.
- **`nat_long` mechanism, an execution-shape rounding effect:**
  - before output rounding, every path ranks `' described'` higher;
  - path A's 23.7957 rounds to 23.75 and ties;
  - B and C's 23.8144 rounds to 23.875;
  - an fp32 instrument with the same weights removes the gap (A−B max_abs 2.2e-5 … 1.1e-4) and
    the divergence.
- **The tolerance is a weak gross-error check only.** It misses these faults:
  - a one-token prefix change (max_abs 0.94);
  - query positions shifted by +1 (0.33–0.99);
  - a chunk swap (0.25);
  - a one-bit LSB flip (0.25).

  Storage correctness therefore rests on byte identity, identity keys and bitwise
  restored-vs-resident checks (D12).
- **Negative controls:** the byte check catches every storage fault, including some that are
  invisible in the logits. A position shift is caught by the `prefill` guard.
- **Fixes:** three restore-hardening gaps and one audit-script indexing bug were fixed; see
  the BUGLOG preamble.

## M1–M1.2: tiers, identity, synchronous always-restore
**M1** (`results/m1/m1_run_20261103T023956.json`):
- **Sequence scenario:** 9 requests over a 2-slot pinned host tier and a 2-file disk tier.
  Every request's strict check passed. The counters were:
  - host hits 2, disk hits 7, chunk misses 5;
  - 2,304 recovered prefix tokens;
  - 7 host spills and 5 disk evictions.
- **Fault scenarios** each produced the intended miss and never repeated inference:
  - a corrupt middle chunk → `disk_payload_crc`, restoring the valid prefix L = 256;
  - a missing file → `disk_file_missing`;
  - an incompatible namespace or layout;
  - ENOSPC on every spill → 8 forward calls, 2 spill failures, and an empty directory
    afterwards.
- **Report only (D10):** greedy output matched full recomputation in 3 of 7 restored requests.
  The other 4 come from 2 prompts and diverge at a full-path top-2 margin of 0.000 or 0.125.
  Every one of them is bitwise equal to the resident split replay.

**M1.1** fixed BUG-001, BUG-002 and RPT-003 and recorded D10. Its rerun
(`results/m1_1/m1_run_20261103T112930.json`) has records and counters identical to M1.

**M1.2** fixed VAL-004: the runner had reported 19 requests but strictly validated 18. The
corrected run (`results/m1_2/m1_run_20261103T121800.json`) has 19 requests and 19 strict
records, and ENOSPC makes 8/8 forward calls.

**Limitations** (unchanged since M1):
- single process, sequential, batch 1;
- no fsync;
- no promotion of disk hits (D15);
- CRC32 is not authentication; the cache directory must be trusted;
- no write-through when the host has no free slot;
- GPU restore peak about 2× the prefix (D4);
- an undeletable file reduces disk capacity until `clear()`;
- host figures are allocation/registration sizes, not RSS.

## M2: synchronous latency and an offline cost model
Protocol: `docs/M2_PROTOCOL.md`. Run: `results/m2/run_20261103T131543/`.

**Grid and correctness:** the full 15-case grid was measured, with 0 of 2,580 measured samples
excluded and strict correctness 15/15. Full vs restored first tokens were equal in 15/15
cases (report only, max |Δlogit| 0.39).

**Cost model:**
- It was fitted on calibration medians only, frozen, and scored on 9 held-out cases × 2 tiers.
- **Latency error:** full median abs error 3.5%. Restore cases are poorly predicted in places
  (host up to 67%), because a platform slow mode is not modeled.
- **Model terms (`model.json`):**
  - restore ≈ 3.62 ms per chunk (host) and 5.75 ms per chunk (warm disk);
  - suffix prefill floor ≈ 17 ms, launch-bound even for S = 1.
- **Per-chunk restore cost is dominated by per-copy overhead, CRC and assembly**, not PCIe
  bandwidth, judging by these micro-benchmarks:

  | Operation | Time |
  |---|---|
  | one contiguous 7 MB H2D | 0.61 ms |
  | the 56 per-layer copies the baseline makes | 2.31 ms |
  | validated read | 1.60 ms |
  | — of which CRC32 | 1.31 ms |
  | `DynamicCache` assembly | 0.75 ms |

**Break-even:**
- No crossing is predicted or reliably measured within L = 256…4096.
- Restore was the M2 winner wherever the paired interval excluded 0.
- **No break-even length is claimed.**

**Disk:** `disk` means a warm page cache, verified by `mincore`. The diagnostic `disk_evicted`
path (`fadvise(DONTNEED)`) is not a cold-NVMe measurement, because WSL2 may serve it from the
Windows host cache. Cold device reads are BLOCKED (D26).

**Qualified by M2.1 (VAL-005).** M2 reported "11 of 11 scorable held-out decisions correct,
0 ms penalty". That rested on i.i.d. intervals over time-clustered series (lag-1 ρ up to
0.93). With block-bootstrap intervals:

| Statistic | Held-out decisions (18) |
|---|---|
| median | 9 restore-faster, 9 inconclusive (the L = 1024, S = 64 wins are no longer established) |
| mean | 8 restore-faster, 6 recompute-faster (all at L = 512), 4 inconclusive |

- The "0 ms" penalty covered scorable decisions only. Over all 18, the mean-based point penalty
  is 110 ms, of which 92 ms is established at L = 512.
- **The model's choices equal always-restore**, so it shows no decision benefit over that
  baseline in this dataset.
- "Tie" means **inconclusive**, never equivalence.
- `full` means recompute from the prepared cache state (D28).

The M2 data, model and predictions are unchanged (`results/m2_1/reanalysis/`).

## M2.1: measurement audit and slow-mode diagnosis
Plan: `docs/M2_1_DIAGNOSTIC_PLAN.md`, written before any M2.1 GPU sample. Sessions:
`results/m2_1/diag_*`. Reanalysis: `results/m2_1/reanalysis/`.

- **Design:** interleaved paths × modes per repetition across 9 sessions, with 0 exclusions.
  The strict and no-cache bitwise checks PASS.
- **Finding:** in time-clustered regimes (median 1.4 s, up to ~10 s), the host-side CPU cost of
  CUDA driver/runtime calls rises 2–5×:
  - kernel launches go from ≈10–12 to ≈48 µs each;
  - small synchronous H2D copies go from ≈40 to ≈90 µs each.
- **Unchanged in those regimes:** GPU kernel time, ordinary CPU work, guest system calls and the
  SM clock. The thread stays on-CPU.
- **Who is exposed:** the regime is shared by all paths. Recompute hides it behind GPU time;
  restore paths, made of thousands of small CUDA calls, expose it.
- **Root cause:** not established. It cannot be separated from inside the guest without
  host-side tracing.
- **Separate effect:** GPU clock regimes, which affect GPU-bound work.
- **Statistical directions are conditional** on that sample, analysis and the baseline backend.
  For example, "the mean favors recompute at L = 512" is not universal and does not transfer to
  the batched backend.

## M3: batched restore backend (restore mechanics only)
Design note: `docs/M3_DESIGN.md`. Protocol (frozen before evaluation):
`docs/M3_PROTOCOL.md`, hash in `results/m3/protocol_freeze_20261103T173632.sha256`.

**Evaluation:**
- 5 fresh-process sessions (`results/m3/eval_20261103T173632/`) and 5 cases;
- 2 blocks per case, each 2 warmup + 12 measured repetitions, in interleaved random order;
- 5,400 measured samples, none excluded or filtered.

Copy and sync counts come from a separate profiler session (`session_prof.json`).

**Backend:**
- One blocking contiguous H2D per chunk into a bounded chunk-major device staging buffer
  (≤ 4 chunks, ≤ 29.4 MB).
- One device placement copy per group.
- Chunk format, identity, tiers, longest-valid-prefix and admission are unchanged.

**Correctness:** every strict check passed for both backends. Under identical state, the two
backends are bitwise equal in:
- every K/V byte;
- the first-token logits;
- 8 greedy steps and their tokens.

**Measured CUDA calls per restore:**

| Chunks | Baseline memcpy / sync / launches | Batched memcpy / sync / launches |
|---|---|---|
| 1 | 56 / 56 / 56 | 2 / 1 / 56 |
| 6 | 336 / 336 / 392 | 6 / 6 / 58 |
| 16 | 896 / 896 / 952 | 16 / 16 / 60 |

At 16 chunks, kernel launches per whole request fall from 2,192 to 1,300. H2D bytes are
identical for both backends.

**End-to-end first-token latency, median (ms):**
- d = batched − baseline, paired by repetition, with stratified block-bootstrap 95% CIs (b = 4);
- verdicts follow rules R1–R3.

| Case (L, S), chunks | Host baseline → batched | Host median d | Disk baseline → batched | Verdict |
|---|---|---|---|---|
| (256, 32), 1 | 23.9 → 22.5 | −1.99 [−2.29, −1.71] | 26.5 → 24.6 | faster |
| (1024, 1), 4 | 33.3 → 21.3 | −12.3 [−12.9, −11.9] | 41.9 → 29.1 | **inconclusive (R3)** |
| (1536, 128), 6 | 43.7 → 25.9 | −17.9 [−18.8, −17.3] | 55.3 → 37.0 | faster |
| (3072, 256), 12, admits 1 chunk | 78.7 → 41.7 | −37.0 [−38.3, −36.2] | 101.4 → 62.7 | faster |
| (4096, 64), 16 | 86.2 → 37.6 | −48.5 [−80.4, −47.8] | 117.7 → 65.0 | faster |

- **Speedup** at (4096, 64) host, under three different definitions: 2.29 (ratio of
  medians), 2.48 (ratio of means), 2.32 (median of paired ratios).
- **Restore phase alone** at 16 chunks: 62.6 → 12.1 ms (host) and 90.8 → 39.4 ms (disk). The
  disk remainder is mostly the validated read + CRC, ≈ 1.7 ms per chunk, the same for both
  backends.
- **No case is slower** with batched.

**Memory:**
- Batched adds min(n, 4) × 7,340,032 B of transient device memory, and no host memory.
- Both backends keep exactly the restored cache after return. At L = 3072 the allocator's block
  rounding shows extra bytes for both backends; a probe confirms that requested bytes equal the
  payload.

**Adoption rule (frozen, `docs/M3_PROTOCOL.md` §8).** It requires every case with ≥ 4 chunks to
be established faster in median and mean on both tiers.
- (1024, 1) satisfies R1 (all block lengths agree) and R2 (all 5 sessions agree).
- It fails R3: its summed measured span is 73 s, less than 10× its longest slow regime
  (8.2 s).
- So **`supports_adoption` is false**, and the default stays `baseline`. Switching is one
  constructor argument.

**Qualifications:**
- The slow mode remains for both backends (33–42% slow samples), now mostly through the suffix
  forward's ≈1,150–1,350 kernel launches per request.
- Unaffected by M3: admission's 56 D2H copies per admitted chunk, the D4 constructor copy, and
  the validated disk read.
- At (256, 32) the paired **mean** still favors prepared-state recompute for both backends
  (report only).
- Not claimed: p99 or tail latency, cold NVMe, and any other machine or stack.
- The M2 cost model was not refitted for batched restore. Any restore/recompute policy would
  need a fresh evaluation with the batched backend.

## M3 analysis validation (VAL-006)
`scripts/m3_analyze.py` now validates a run against the explicit frozen-evaluation spec in
`kvlab/m3_validate.py` before any statistic is computed.

- **Measured-evidence completeness:**
  - all planned sessions, cases and blocks are present;
  - every (block, rep, path, backend, mode) sample identity occurs exactly once;
  - pairs are complete;
  - the M2/M3 sample validity rules pass.
- **Retained-warmup completeness:**
  - 900 block-warmup identities;
  - 180 process-warmup records, matched by (case, combination) multiplicity against the frozen
    runner;
  - correct warmup and stage flags.

  Warmups are evidence of the procedure and are never used in a statistic.
- **Correctness acceptance** is recomputed from the detailed strict, A/B and restore-memory
  records against spec-derived counts: 50 strict and 50 A/B checks. Stored summary flags are
  only compared, never trusted.
- **Exit codes:** invalid or incomplete evidence exits 3, a failed correctness check exits 4,
  and neither ever reports adoption.
- **Reanalysis of the preserved run** (`results/m3/reanalysis_val006_*`): it is valid, all 5,938
  historical values are reproduced exactly, and the verdict is unchanged (`supports_adoption`
  false).

## M3_CONFIRM_V1: preregistered confirmation of the batched backend (restoration microbenchmark)
Protocol: `docs/M3_CONFIRM_PROTOCOL.md`, frozen 2026-11-04T01:22-08:00 before any
measurement; decision D31. Post-measurement clarification: `docs/M3_CONFIRM_CLARIFICATIONS.md`.
Evidence: `results/m3_confirm/`.

**Design:**
- A new experiment with its own validator spec (`M3_CONFIRM_V1`), analysed on its own and
  never pooled with the M3 samples.
- The runner, timed region, statistics and rules R1–R3 are M3's, unchanged.
- **Freeze:** `results/m3_confirm/M3_CONFIRM_V1.freeze.json` binds:
  - the base HEAD and the diff;
  - every executable and analysis file;
  - the protocol documents, the commands and the stack.

  Its SHA-256, `dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216`, was also
  recorded outside the repository before measurement.

**Cases:**
- (1024, 1), 4 chunks: the M3 case that was inconclusive under R3. This is a targeted
  confirmation, not a random draw.
- (1280, 96), 5 chunks: a fresh case from the same range.
- Tiers: host, and disk with a **warm** page cache.

**Sampling:**
- 5 fresh-process sessions (seeds 32001–32005);
- per case and session, 4 blocks of 2 warmup + 24 measured repetitions, in interleaved random
  order;
- exactly one batch, which ran 798 s of its 1,800 s budget. All 5 sessions exited 0.

**Evidence and correctness:** the analysis exited 0, and the evidence is valid with 0 problems
and 0 exclusions. The validator recomputed every count from the detailed records:

| Item | Present / expected |
|---|---|
| Measured samples | 8,640 / 8,640 |
| Block-warmup records | 720 / 720 |
| Process-warmup records | 180 / 180 |
| Strict correctness checks (passed) | 20 / 20 |
| Baseline-vs-batched A/B checks (passed) | 20 / 20 |
| Restore-memory records | 10 / 10 |

**Primary verdicts** (plain end-to-end first-token latency):
- d = batched − baseline, in ms, paired by repetition;
- 480 pairs per row, in 20 strata (session × block);
- stratified block-bootstrap 95% intervals at b = 4.

| Case | Tier | Median d [95% CI] | Mean d [95% CI] | Median verdict | Mean verdict |
|---|---|---|---|---|---|
| (1024, 1) | host | −13.99 [−14.49, −13.61] | −21.54 [−23.44, −19.63] | batched faster | batched faster |
| (1024, 1) | disk (warm) | −14.27 [−15.14, −13.88] | −22.24 [−24.04, −20.51] | batched faster | batched faster |
| (1280, 96) | host | −16.42 [−16.66, −16.12] | −23.35 [−25.37, −21.38] | batched faster | batched faster |
| (1280, 96) | disk (warm) | −16.84 [−17.18, −16.54] | −24.52 [−26.51, −22.56] | batched faster | batched faster |

No primary verdict is inconclusive, and there is no established regression.

Marginal latency, baseline → batched (ms):

| Case | Tier | Median | Mean |
|---|---|---|---|
| (1024, 1) | host | 36.0 → 22.5 | 63.5 → 42.0 |
| (1024, 1) | disk (warm) | 44.1 → 30.3 | 71.4 → 49.2 |
| (1280, 96) | host | 42.1 → 25.9 | 65.1 → 41.8 |
| (1280, 96) | disk (warm) | 52.1 → 35.5 | 76.1 → 51.5 |

The means lie far above the medians because of the slow mode (below).

**Rules:**
- **R1:** for b ∈ {1, 2, 4, 8}, every interval excludes 0 in the batched-faster direction, in
  all 8 rows.
- **R2:** all 5 per-session point estimates are negative, in all 8 rows.
- **R3, span:**
  - (1024, 1): the summed span is 319.0 s ≥ 10 × 11.76 s (its longest slow regime);
  - (1280, 96): it is 333.4 s ≥ 10 × 8.46 s.

  In M3, (1024, 1) failed this condition with 73 s < 10 × 8.2 s.
- **R3, effective sample size:** the AR(1) effective sample size is computed within each
  stratum and summed, and the rule compares the sum with 10. The sums are:
  - 352.2 / 367.8 for (1024, 1), host / disk;
  - 368.7 / 360.5 for (1280, 96).

  Six individual strata are below 10. The frozen rule has no per-stratum minimum: see
  clarification C1 in `docs/M3_CONFIRM_CLARIFICATIONS.md`, which cites the premeasurement
  sources and does not change the rule.

**Paired differences and speedups are different quantities.** The verdicts above use the paired
difference d in milliseconds. The speedups below are reported only as descriptions. Each one is
a ratio of baseline to batched under a different definition, with b = 4 intervals. They must
not be mixed with each other or with d.

| Case | Tier | Ratio of medians | Ratio of means | Median of paired ratios |
|---|---|---|---|---|
| (1024, 1) | host | 1.600 [1.577, 1.620] | 1.513 [1.470, 1.558] | 1.543 [1.533, 1.554] |
| (1024, 1) | disk (warm) | 1.458 [1.445, 1.471] | 1.452 [1.421, 1.484] | 1.439 [1.431, 1.444] |
| (1280, 96) | host | 1.627 [1.613, 1.635] | 1.559 [1.514, 1.604] | 1.588 [1.580, 1.599] |
| (1280, 96) | disk (warm) | 1.469 [1.459, 1.481] | 1.476 [1.441, 1.510] | 1.468 [1.462, 1.474] |

**Outcome:**
- **The confirmation flag:** `m3_confirm_v1_confirmed = true`. All three conditions of the
  protocol's §7 hold: valid evidence, all checks passing, and all 8 primary verdicts
  "batched faster".
- **M3 is unchanged:** this flag is separate from M3's adoption rule. M3's own result is
  unchanged: (1024, 1) is inconclusive there, and `supports_adoption = false`.
- **The default backend remains `baseline`.** The analysis records
  `authorizes_default_change: false`. Switching would need a separate, reviewed decision in
  `docs/decisions.md`.

**Qualifications:**
- **Scope:**
  - one machine: RTX 4080 Laptop GPU, WSL2, driver 596.49;
  - one stack and model: Qwen2.5-1.5B bf16 SDPA, C = 256, G = 4;
  - two cases, and a warm page cache only.

  No cold-NVMe, p99 or tail-latency claim follows.
- **Not a workload result.** These are batch-1, sequential, synchronous first-token
  restoration microbenchmarks (D22). They do not establish request-level throughput, serving
  latency or TTFT, or any production benefit.
- **The slow mode persists** for both backends: slow-sample shares are 31–42% per row. Its root
  cause is open (D29).
- **Unaffected by the batched backend:** the suffix forward, admission's D2H copies, the D4
  constructor copy and the validated disk read.
- **Comparison with full recompute (`full`)** is context only (D10) and conditional on this
  sample (VAL-005). With the baseline backend at (1024, 1), disk, the paired mean of
  restore − full is +3.69 [0.10, 7.23] ms: baseline restore was slower than recompute in the
  mean. This is not a policy result.
- **The M2 cost model** still describes the baseline backend only, and was not refitted.
- **GPU exclusivity is not proven:**
  - before launch, no competing GPU process or GPU memory use was observable;
  - after the batch, no measurement process remained;
  - however, `nvidia-smi` under WSL2 does not list Windows host processes, so host-side GPU
    use cannot be excluded.
- **Self-attested freeze:** the freeze lives inside the repository, and the out-of-band digest
  record narrows that.
- **Unchanged items:** D4, D10, the M0 `nat_long` strict FAIL, and the VAL-005 and VAL-006
  qualifications above all still apply.

**Re-analysis (CPU only).** Two different checks recompute every number above from the
preserved raw evidence. Neither relaxes a frozen check.

1. **Current-tree check.** The frozen analysis runs in the current tree only while its 7
   analysis files and 4 protocol documents keep their frozen bytes:

   ```bash
   .venv/bin/python scripts/m3_analyze.py --spec M3_CONFIRM_V1 \
     --run-dir results/m3_confirm/M3_CONFIRM_V1_run --out-dir <new dir> \
     --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216
   ```

   - Once any of those files changes, it refuses the evidence (exit 3). That is intended.
   - `scripts/m3_confirm.py verify` is a pre-launch check of the whole frozen tree, including
     HEAD and the full diff. It fails after any later commit, and it is not a post-measurement
     check.
2. **Historical reconstruction**, for use after development has changed any of those files.
   `scripts/m3_confirm_reconstruct.py`:
   - rebuilds the frozen source tree in a new directory, from either:
     - base commit `858f952` + the committed diff sidecar + the four files that were
       untracked at freeze time, taken from commit `13afde2` (`--untracked-rev`); or
     - the tree of commit `13afde2` (`--tree-rev`), which equals the frozen tree;
   - checks every hash in the manifest: 65 source files with no extras, 4 untracked files and
     4 documents;
   - copies in the manifest, the sidecar and the run directory, compared byte for byte;
   - exits 3 on any mismatch.

   The reconstructed tree's own analyzer then runs from inside it:

   ```bash
   .venv/bin/python scripts/m3_confirm_reconstruct.py --out <new dir> --tree-rev 13afde2 \
     --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216
   cd <new dir> && <repo>/.venv/bin/python scripts/m3_analyze.py --spec M3_CONFIRM_V1 \
     --run-dir results/m3_confirm/M3_CONFIRM_V1_run --out-dir <another new dir> \
     --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216
   ```

   - **Executed once on 2026-11-04,** from a reconstruction built from the freeze-time
     snapshot: exit 0, valid, confirmed.
     - `samples_chronological.csv` is byte-identical.
     - The validation JSON, the analysis Markdown and the analysis JSON differ only in the
       absolute path of the freeze manifest, which names the tree the analysis ran in, and in
       `out_dir`.
   - **Requirement:** the Python environment must still hold the frozen package versions
     (`requirements.txt`, recorded in the manifest).

## M4: trace evaluation of C0, C1-PA and C2 (M4_EVAL_V1 invalid; M4_EVAL_V2 valid)
Protocols:
- `docs/M4B_PROTOCOL.md` for M4_EVAL_V1 (decisions D33–D35);
- `docs/M4_EVAL_V2_PROTOCOL.md` for M4_EVAL_V2 (D36–D38). It amends only the warm-disk preparation
  and adds a page-cache interference rule.

Both protocol documents are frozen, so their status notes predate the batches. Archive and
reproduction: D39. Evidence: `results/m4/`.

**Design common to both evaluations:**
- **Traces:** four synthetic traces (evaluation seeds 42001–42004):
  - T0, a no-reuse control;
  - T1, a shared system prompt;
  - T2, multi-turn conversations;
  - T3, document reuse.
- **Configurations:** every session replays every trace through all of them. Requests are
  interleaved in a seeded order, with fresh state per trace.
  - **C0:** no cache, the full prefill.
  - **C1-PA(W) and C1-PA(W/2):** this project's GPU-only, prefix-aware LRU cache, with the
    trace's working set of W chunks or with ⌈W/2⌉ chunks (an artificial cap).
  - **C2-base and C2-batch:** tiered always-restore, with a pinned host slab of ⌈W/2⌉ slots and
    a disk tier. C2-base uses the default `baseline` restore backend; C2-batch uses the opt-in
    `batched` backend.
- **Metric:** for each session, trace and configuration, the trace-mean synchronous first-token
  latency (D22). For a comparison A − B, D_s = mean_A,s − mean_B,s.
- **Sessions:** one batch of R = 10 fresh-process sessions under a 2,700 s deadline. It was
  launched once from a freeze manifest whose SHA-256 had already been recorded outside the
  repository.
- **Primary family:** 15 comparisons, fixed in advance:
  - on T1, T2 and T3: each C2 backend against C0, and against C1-PA(W/2);
  - on T0: C2-base − C0, C2-batch − C0 and C1-PA(W/2) − C0.
- **Joint decision rule:** a verdict needs both of the following; otherwise the result is
  inconclusive:
  - all 10 D_s share one sign;
  - the Bonferroni-level (1 − 0.05/15) t-interval excludes 0.

### M4_EVAL_V1: executed once; invalid; no performance conclusion
- **Freeze:** `results/m4/M4_EVAL_V1.freeze.json`, from commit `b858d9e`. Its SHA-256 is
  `137220aa1f40c9604fda983380f03d2e4f897cb2a8b6ad64ac97eabd3a3625b2`.
- **Execution (2026-11-04):**
  - 10/10 sessions exited 0, in 1,833.1 s of the 2,700 s deadline;
  - the strict comparison of the 11,200 evaluation records found 0 mismatches. The 1,000
    warmup records are not strict-checked.
- **Invalid under its frozen warm-disk rule.**
  - **The rule:** before every C2 request, every chunk file of that instance's whole disk tier
    had to be fully resident in the page cache (`mincore`), after a prewarm of any file below
    full residency.
  - **The failure:** for 12 of the 4,480 C2 evaluation requests (sessions e04, e05 and e09),
    the resident fraction stayed below 1.0 after the prewarm.
  - **The consequence:** the frozen analyzer exited 3 and wrote only
    `results/m4/M4_EVAL_V1_analysis/m4_validation.json`, which lists the 12 problems.
- **No performance conclusion.**
  - None of the 15 primary comparisons is evaluated, in either direction.
  - The batch is not re-analysed in part, by session, trace or request.
- **The cause is unresolved.** The batch kept no synchronized memory or reclaim history, so
  page-cache reclaim, a WSL cache drop and contention cannot be told apart.
- **Provenance:**
  - These files are byte-for-byte copies of the untracked evidence in the `m4a-trace-replay`
    worktree at `b858d9e`. They sit at the same relative paths, which the manifest and the
    frozen analyzer use.
  - The absolute `run_dir` recorded in `m4_validation.json` names that worktree.

### M4_EVAL_V2: one valid batch
**What V2 changes** (`docs/M4_EVAL_V2_PROTOCOL.md`; `kvlab/warm_disk.py`): only the warm-disk
preparation, plus one validity rule.
- **Preparation:** before each C2 request, untimed, the runner:
  1. plans the request's candidate files, the disk files its restore reads on the normal path;
  2. observes their residency once;
  3. reads in full, at most once and with no retry, each file that is not fully resident;
  4. observes all of them once more.

  The timer starts after that final pass.
- **Validity:**
  - every candidate must be fully resident in the final pass;
  - the batch is invalid if the WSL `drop_pagecache` counter changes during it. Ordinary
    reclaim counters are recorded as context only.
- **Relation to V1:** V2 is a different procedure, not a correction of V1. The two evaluations are
  not compared.

**Execution:**
- **Freeze:** once, at `d2122d7`: `results/m4/M4_EVAL_V2.freeze.json`, with an empty diff
  sidecar. Its SHA-256 is `150f232ef11040ef1bcaf095a3f9beb341e708c59fc91de244f78e848d4cc83a`.
- **The batch:** one, on 2026-11-05. `execution_status` is `succeeded`, and 10/10 sessions
  exited 0 in 2,005.68 s of the 2,700 s deadline. No session was stopped, retried or replaced.

**Evidence: valid.** The frozen analyzer exited 0, with 0 problems and 0 correctness failures
(`results/m4/M4_EVAL_V2_analysis/m4_validation.json`).
- **Records:**
  - 11,200 evaluation records, each strictly compared with a resident-GPU reference replay of
    its own configuration's decisions: 0 mismatches;
  - 1,000 warmup records, which are not strict-checked.
- **Warm disk:** all 4,880 C2 records passed the request-scoped rule: 2,120 `verified`, and
  2,760 with an empty candidate set.
  - **Prewarms:** 22 needed-prewarm operations, each one whole-file read, completed
    successfully. Each was followed by a fully resident final pass. They occurred in sessions
    e05, e08, e09 and e10, on traces T2 and T3.
  - **What this does not show:** it shows only that these 22 prewarms completed as specified in
    this batch. It does not show that page-cache eviction is fixed, and it does not identify
    V1's cause.
- **Interference:** `drop_pagecache` was unchanged in all 9,782 snapshots (batch, session and
  request). Ordinary reclaim did occur; it is context only.
- **Other checks:** all passed:
  - decisions and counters against the independent key-level models;
  - faults, storage and copy traffic;
  - coverage;
  - source and stack binding.

**Primary results.**
- **The table below** restates the frozen results, rounded to 0.01 ms directly from the
  full-precision values in `m4_analysis.json`. The frozen Markdown table,
  `results/m4/M4_EVAL_V2_analysis/m4_analysis.md`, shows them to 0.001 ms.
- **The sign counts** come from the per-session D_s.
- **The complete record,** including every D_s, is `.primary` in `m4_analysis.json` (same
  directory).

d = mean_A − mean_B, in ms; d < 0 means A is faster.

> All intervals and verdicts are conditional on treating the ten session-level differences as
> independent and approximately normally distributed. Fresh processes do not establish
> independence, persistent machine/driver/thermal regimes are possible, and ten sessions cannot
> establish normality.

| Trace | A − B | Mean d (ms) | 95% t-interval | Bonferroni interval (99.67%) | Sessions D_s < 0 / > 0 | Verdict |
|---|---|---|---|---|---|---|
| T1 | C2-base − C0 | −5.76 | [−14.38, 2.86] | [−20.83, 9.31] | 5 / 5 | inconclusive |
| T1 | C2-base − C1-PA(W/2) | +23.39 | [15.42, 31.35] | [9.46, 37.32] | 1 / 9 | inconclusive (not unanimous) |
| T1 | C2-batch − C0 | −37.89 | [−42.88, −32.89] | [−46.62, −29.15] | 10 / 0 | **A faster** |
| T1 | C2-batch − C1-PA(W/2) | −8.74 | [−13.41, −4.08] | [−16.89, −0.59] | 10 / 0 | **A faster** |
| T2 | C2-base − C0 | −0.37 | [−6.21, 5.46] | [−10.57, 9.83] | 6 / 4 | inconclusive |
| T2 | C2-base − C1-PA(W/2) | +0.97 | [−4.33, 6.28] | [−8.29, 10.24] | 6 / 4 | inconclusive |
| T2 | C2-batch − C0 | −16.41 | [−20.18, −12.64] | [−23.00, −9.82] | 10 / 0 | **A faster** |
| T2 | C2-batch − C1-PA(W/2) | −15.06 | [−17.76, −12.36] | [−19.78, −10.35] | 10 / 0 | **A faster** |
| T3 | C2-base − C0 | +1.96 | [−5.71, 9.62] | [−11.45, 15.36] | 6 / 4 | inconclusive |
| T3 | C2-base − C1-PA(W/2) | +30.87 | [25.80, 35.94] | [22.00, 39.74] | 0 / 10 | **A slower** |
| T3 | C2-batch − C0 | −20.10 | [−24.84, −15.35] | [−28.39, −11.81] | 10 / 0 | **A faster** |
| T3 | C2-batch − C1-PA(W/2) | +8.82 | [6.79, 10.84] | [5.28, 12.36] | 0 / 10 | **A slower** |
| T0 | C2-base − C0 | +62.17 | [48.62, 75.73] | [38.48, 85.87] | 0 / 10 | **A slower** |
| T0 | C2-batch − C0 | +60.54 | [48.19, 72.90] | [38.95, 82.14] | 0 / 10 | **A slower** |
| T0 | C1-PA(W/2) − C0 | +5.90 | [2.83, 8.98] | [0.52, 11.29] | 0 / 10 | **A slower** |

**Interpretation:**
- **Counts:** under the predeclared joint rule, 5 are "A faster", 5 "A slower" and 5
  inconclusive.
- **Inconclusive does not mean equivalent.** Take T1 C2-base − C1-PA(W/2): its Bonferroni
  interval lies above 0, but one of the ten sessions has D_s < 0. It therefore fails unanimity
  and stays inconclusive.
- **No other verdicts.** The following are reported in `m4_analysis.json` with no verdict, and
  they acquire none here:
  - the 13 secondary comparisons, including C2-batch − C2-base and the comparisons against
    C1-PA(W);
  - the steady-state versions;
  - the bootstrap sensitivity analysis;
  - the descriptive statistics.
- **No batched-versus-baseline verdict.** C2-batch − C2-base is a secondary comparison. No
  verdict between the two C2 restore backends is inferred from their separate comparisons with
  C0 or C1-PA(W/2).
- **Adoption:** M4 does not change the default (`docs/M4B_PROTOCOL.md` §4). `baseline` remains
  the default restore backend. M3's `supports_adoption = false` and the M3_CONFIRM_V1
  conclusions are unchanged.

**Scope** (the analysis's `claim_limits`, in full):
- **Workload and platform:** synthetic traces, on the recorded machine, model and software stack
  only.
- **Metric:** sequential, batch-1, synchronous first-token latency; not service TTFT.
- **An interleaved, co-resident experiment, not a standalone deployment:**
  - both C1 caches stay allocated while every configuration runs;
  - all configurations share one allocator, page cache and thermal state.
- **Unequal total resources:**
  - C1 holds persistent GPU KV;
  - C2 holds none on the GPU, but a pinned host slab, a disk tier and an uncounted page cache;
  - every GPU budget below the device's free memory is an artificial cap.
- **Allocator peaks** are whole-process figures, never per-configuration GPU peaks.
- **Residency is a request-scoped observation:**
  - per-file point observations in one pass before each request, not a simultaneous snapshot;
  - not guaranteed residency at the timer start or during the timed reads, since pages are not
    pinned;
  - a warm page cache only.
- **The baseline:** C1 is this project's own baseline. This is not a comparison with vLLM or
  SGLang.
- **Not claimed:** service TTFT, p99/tail latency, throughput, concurrency, cold-NVMe or
  production behaviour.

### Reproducing and verifying the M4 evidence (CPU only)
Three kinds of check must be kept apart.
1. **Offline evidence and source verification, and analysis reproduction** (below):
   - CPU only, from committed files;
   - no model, no GPU query and no package check.
2. **Live environment verification:** `scripts/m4_eval.py verify --spec <name>` compares the
   checkout and the live stack with the manifest. The live stack includes an `nvidia-smi` query.
   - **When it passed:** before each batch, on the recording machine. Each batch record keeps the
     result, `pre_launch_verification: {"problems": []}`.
   - **Why it now fails:** it is a pre-launch check that binds HEAD and the source set, so it
     fails in this checkout after the archive commits. That is expected. It was not weakened, and
     no manifest was regenerated.
   - **What the offline steps do not do:** they do not re-establish the live stack.
3. **A fresh GPU experiment:** none was run for the archive. A new batch would need a new spec,
   a new freeze and the owner's approval.

**Current-tree check (V2 only).**
- **Why it still works:** the V2 analyzer compares only its 13 analysis files and 3 documents
  with the manifest. The archive commits change none of them, so it runs in this checkout:

  ```bash
  .venv/bin/python scripts/m4_analyze.py --spec M4_EVAL_V2 --run-dir results/m4/M4_EVAL_V2_run \
    --out-dir <new dir> \
    --expect-freeze-sha256 150f232ef11040ef1bcaf095a3f9beb341e708c59fc91de244f78e848d4cc83a
  ```

- **Why V1 is different:** this branch changed 4 of V1's 13 bound analysis files
  (`kvlab/m4_validate.py`, `m4_analysis.py`, `m4_spec.py` and `m4_freeze.py`).
  - In this checkout, V1's analysis reports those 4 files as differing from their frozen bytes,
    in addition to the 12 residency problems, and exits 3.
  - That is intended, and it is not a reproduction. V1 is reproduced only from its frozen
    revision (below).

**Frozen-revision reconstruction (V1 and V2).** `scripts/m4_reconstruct.py` runs in three
steps.
1. **`build`:**
   - extracts the manifest's base commit (`b858d9e` for V1, `d2122d7` for V2) into a new
     directory;
   - checks every manifest hash: the source set, with no extra files, plus the documents and
     analysis files;
   - copies in the manifest, the sidecar and the run directory, compared byte for byte.
2. **Analysis:** that tree's own `scripts/m4_analyze.py` runs from inside it, into another new
   directory.
3. **`compare`:** compares the regenerated outputs with the archived ones, JSON value by value and
   Markdown line by line. Both directories must hold `m4_validation.json`.
   - **The one allowed path-only difference:** the top-level `run_dir` of `m4_validation.json`.
     The archived value must lie below the archived root, and the regenerated value below the
     reconstruction, with the same relative part. The file's bytes must otherwise be identical.
   - Every other difference, a missing file or an extra file is substantive.

```bash
.venv/bin/python scripts/m4_reconstruct.py build --spec M4_EVAL_V2 --out <new dir> \
  --expect-freeze-sha256 150f232ef11040ef1bcaf095a3f9beb341e708c59fc91de244f78e848d4cc83a
cd <new dir> && <repo>/.venv/bin/python scripts/m4_analyze.py --spec M4_EVAL_V2 \
  --run-dir results/m4/M4_EVAL_V2_run --out-dir <new out dir> \
  --expect-freeze-sha256 150f232ef11040ef1bcaf095a3f9beb341e708c59fc91de244f78e848d4cc83a
cd <repo> && .venv/bin/python scripts/m4_reconstruct.py compare \
  --archived results/m4/M4_EVAL_V2_analysis --regenerated <new out dir> \
  --archived-root /home/cyrus/projects/kv-cache-lab-m4v2 --regenerated-root <new dir>
```

For V1, use:
- `--spec M4_EVAL_V1` and the digest
  `137220aa1f40c9604fda983380f03d2e4f897cb2a8b6ad64ac97eabd3a3625b2`;
- `results/m4/M4_EVAL_V1_run` and `results/m4/M4_EVAL_V1_analysis`;
- the archived root `/home/cyrus/projects/kv-cache-lab-m4a`.

Its analyzer is expected to exit 3.

**Executed on 2026-11-05,** with this tool on the archived bytes:

| Step | V2 (`d2122d7`) | V1 (`b858d9e`) |
|---|---|---|
| `build` | VERIFIED: 95 source files, 3 documents, 13 analysis files; 34 evidence files copied | VERIFIED: 92 source files, 2 documents, 13 analysis files; 33 evidence files copied |
| Frozen analyzer | exit 0: valid, 0 problems, 0 correctness failures | exit 3: invalid, 12 problems, 0 correctness failures; validation output only |
| `compare` | `m4_analysis.json` and `m4_analysis.md` byte-identical; `m4_validation.json` differs only in `run_dir` (path-only) | `m4_validation.json` differs only in `run_dir` (path-only) |
| Current-tree check | exit 0; all three outputs byte-identical | exit 3 with the 4 changed analysis files reported (not a reproduction, as intended) |

- **The path-only difference is expected:** the validator records the absolute run directory of
  the tree it ran in.
- **Everything else agrees exactly:** every statistical value, verdict, count and problem
  string.
- **Requirement:** the Python environment must hold the frozen package versions
  (`requirements.txt`; recorded in each manifest's `stack`).
- **Tests:** `tests/test_m4_reconstruct.py` covers the tool on fixture repositories.

## Reproducing
Commands are in the README. Each evidence directory keeps its raw session JSON, logs and
command lines. GPU experiments must run one process at a time. The analysis scripts are
CPU-only and refuse to overwrite existing outputs.

## Provenance of references to unpublished working documents
Until the documentation cleanup, the repository also tracked local working documents: agent
instructions (`CLAUDE.md`), the original working plan (`kv-cache-plan.md`), per-milestone
review reports (`docs/*_REVIEW.md`) and walkthroughs (`docs/*_WALKTHROUGH.md`). They remain in
the Git history but are no longer part of the public tree. This document carries their
technical content.

Some preserved evidence names them, and is kept byte-for-byte:
- manifests and change inventories under `results/`;
- the `git_status_porcelain` field of M3 session records;
- preservation logs and archive file lists;
- the docstring of `scripts/m2_1_reanalyze.py`.

These are historical provenance references, not live dependencies. No script or test reads
those documents.

`kvlab/boundaries.py` changed after the M3 evidence was recorded: one docstring line, which
cited the plan's first-token rule, now cites `docs/design.md` §2 in English. Behavior is
unchanged. The hashes recorded for this file in earlier manifests and session records refer to
the previous text.
