# Results

What was measured or verified, the exact scope of each result, and where its evidence is. Design:
[design.md](design.md). Decisions: [decisions.md](decisions.md). Defects: [BUGLOG.md](BUGLOG.md). Tests
by property: [correctness-matrix.md](correctness-matrix.md).

Raw evidence, logs and analysis outputs are under `results/` ([index](../results/README.md)). The
earlier long form of this document, with every statistic and reproduction log in full, is
[`docs/results.md` at commit `eed48ad`](https://github.com/cyruswu750/kv-cache-lab/blob/eed48ad77ee86fe72eae572a031c94af72ba31ad/docs/results.md). See [Reproducing](#reproducing).

**Scope of every number below:**
- **Machine and software:** one RTX 4080 Laptop GPU (12 GiB, Windows driver 596.49) under WSL2, torch
  2.14.1+cu130, transformers 5.22.1.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA, chunk size C = 256 tokens (one chunk is
  7,340,032 B of K/V).
- **Workload:** batch 1, one process, sequential requests, tokenized input.
- **Metric:** synchronous first-token latency (D22): from a synchronize before the request to the
  first-token argmax, including restoration, suffix prefill and cache admission. It is **not**
  service-level TTFT, decode throughput or serving throughput.
- **Disk tier:** application-level hits with a warm Linux page cache. Residency was observed with
  `mincore` before measured requests (point observations per file, not a guarantee during the timed
  read). **Cold-NVMe latency and p99/tail latency are not established.**

## Summary

| Question | Experiment | Result |
|---|---|---|
| Is restored state correct? | strict checks in every GPU suite and experiment | Restored K/V bytes equal what was written; outputs are bitwise equal to a GPU-resident replay of the same split. 11,200 strictly compared trace requests in M4_EVAL_V2: 0 mismatches. Not claimed: equality with full-prefill output (bf16, below). |
| Does one contiguous copy per chunk beat per-layer copies? | M3_CONFIRM_V1, preregistered | Yes on both tested cases and tiers: median first-token latency 1.46–1.63× lower (ratio of medians), all 8 primary verdicts "batched faster". At 16 chunks, H2D copies per restore fall from 896 to 16 for the same bytes (M3 profiler). |
| Does the tiered cache help on request traces? | M4_EVAL_V2, synthetic traces | With batched restore it beat no caching on all three reuse traces and a half-budget GPU-only prefix cache on two of them; it lost to that cache on document reuse and to no caching on the no-reuse trace. With the baseline backend most comparisons were inconclusive. |
| Does a calibrated cost model choose better restore lengths? | M5, held-out evaluation | Not demonstrated: on all 8 held-out cases it chose exactly what always-restore chose. Prediction error 5.0% mean absolute relative error over 47 candidates, up to 31%. |

## Correctness

**Contract (D10, D12):**
- **Storage** is verified strictly: restored K/V bytes must equal the bytes written, and every step's
  logits must be bitwise equal to a GPU-resident replay of the same request with the same restore length
  and split (`kvlab/resident_reference.py`). This holds for both restore backends, which are also
  bitwise equal to each other.
- **Comparison with full recomputation is report-only.** In bf16 the chunked and unchunked execution
  shapes round differently, so exact ties or 1-ulp margins can flip greedy tokens.
- **Forward calls:** every request makes exactly one prefill and `steps − 1` decode calls. Cache write,
  spill, eviction and cleanup failures never add or remove a forward call.

**The documented full-prefill divergence (M0, M0.1).** M0 restored a real KV prefix GPU → CPU → GPU. All
byte checks and boundary cases (N = 255…513) passed, and `nat_long` (N = 426, L = 256) failed greedy
agreement with full prefill, 6/8:
- at step 6 the full-prefill reference has an exact bf16 tie, `' re'` = `' described'` = 23.75, and the
  chunked path moves one logit by 1 ulp (0.125);
- before output rounding every path ranks `' described'` higher; an fp32 instrument with the same weights
  removes both the gap and the divergence;
- the restored path is bitwise equal to a GPU-resident chunked prefill, so storage is not the cause.

The M0.1 audit confirmed restored = resident bitwise in 9 bf16 splits × 3 repeats and in fp32 (4,032 bf16
tensor comparisons, 0 mismatches). It also showed that the M0 tolerance (atol 1.45) is only a gross-error
check: it misses a one-token prefix change, a +1 position shift, a chunk swap and a one-bit LSB flip.
Storage correctness therefore rests on byte identity, identity keys and bitwise checks, never on atol.
The M0 strict suite therefore still FAILs on `nat_long`; that FAIL is the recorded result, kept with
its original criteria (no tie rule and no xfail, D10).

**Tiers, identity and faults (M1–M1.2).** A 9-request sequence over a 2-slot pinned host tier and a 2-file
disk tier passed every strict check (host hits 2, disk hits 7, chunk misses 5; 2,304 recovered tokens;
7 spills, 5 disk evictions). Each injected fault produced the intended miss without repeating inference:
- a corrupt middle chunk → `disk_payload_crc`, restoring only the valid prefix L = 256;
- a missing file → `disk_file_missing`; an incompatible namespace or layout → a miss;
- ENOSPC on every spill → the cache write is skipped, 8 forward calls as expected, 2 spill failures,
  and an empty directory afterwards.

The corrected evidence runner (VAL-004) validates all 19 strict records
(`results/m1_2/m1_run_20261103T121800.json`).

Limitations of the tiered cache itself: single process, sequential, batch 1; no fsync; no promotion
of disk hits (D15); CRC32 is not authentication, so the cache directory must be trusted; no
write-through when the host has no free slot; a GPU restore peaks at about 2× the prefix (D4); an
undeletable file reduces disk capacity until `clear()`; host figures are allocation/registration
sizes, not RSS. Report only: greedy output matched full recomputation in 3
of 7 restored requests; the other 4 diverge at a full-path top-2 margin of 0.000 or 0.125 and are
bitwise equal to the resident replay.

**Mutation checks:** deliberately broken copies of the code must fail the tests. CPU mutants of the
tiered store: 21/21 killed (`scripts/m1_mutation_check.py`). Restore-backend mutants on CPU and GPU:
10/10 killed (`scripts/m3_mutation_check.py`; an earlier run killed 9/10, and a test was strengthened).

## Restoration microbenchmarks

### M2: latency of full, host and disk paths, and an offline cost model (baseline backend)
Protocol `docs/M2_PROTOCOL.md`; evidence `results/m2/run_20261103T131543/`. 15 cases, 2,580 measured
samples, none excluded; strict correctness 15/15.
- **Cost model:** fitted on calibration medians only, frozen, and scored on 9 held-out cases × 2 tiers.
  Full-path median absolute error 3.5%; restore cases poorly predicted in places (host up to 67%) because
  a platform slow mode is not modelled. Restore costs ≈ 3.62 ms per chunk from host and 5.75 ms from warm
  disk; suffix prefill has a floor of ≈ 17 ms.
- **Where restore time goes** (micro-benchmarks): one contiguous 7 MB H2D takes 0.61 ms, but the 56
  per-layer copies of the baseline take 2.31 ms; a validated disk read 1.60 ms (CRC32 1.31 ms);
  `DynamicCache` assembly 0.75 ms. Per-copy overhead, not PCIe bandwidth, dominates. This motivated M3.
- **No restore-vs-recompute break-even** was established for L = 256…4096.
- **Disk:** `disk` means a warm page cache verified by `mincore`. The diagnostic `disk_evicted` path
  (`fadvise(DONTNEED)`) is not a cold-NVMe measurement, because WSL2 may serve it from the Windows
  host cache; cold device reads were not measured (D26).
- **Qualified by VAL-005 (M2.1):** M2's "11 of 11 scorable decisions correct" used i.i.d. intervals over
  time-clustered samples (lag-1 ρ up to 0.93). With block-bootstrap intervals, of 18 held-out decisions
  the median gives 9 restore-faster and 9 inconclusive; the mean gives 8 restore-faster, 6
  recompute-faster (all at L = 512) and 4 inconclusive. The model's choices equal always-restore, so it
  showed no decision benefit over that baseline.

### M2.1: the restore slow mode
Plan `docs/M2_1_DIAGNOSTIC_PLAN.md`, written before any M2.1 sample; 9 interleaved sessions. In
time-clustered regimes (median 1.4 s, up to ~10 s), the host-side cost of CUDA driver/runtime calls rises
2–5× (kernel launches ≈ 10–12 → ≈ 48 µs; small synchronous H2D copies ≈ 40 → ≈ 90 µs), while GPU kernel
time, ordinary CPU work, system calls and SM clock are unchanged. Recompute hides it behind GPU time;
restore paths made of thousands of small CUDA calls expose it. **The root cause is not established**
(it cannot be separated from inside the WSL2 guest).

### M3: baseline vs batched restore (exploratory A/B)
Design `docs/M3_DESIGN.md`; protocol `docs/M3_PROTOCOL.md`, hashed before evaluation
(`results/m3/protocol_freeze_20261103T173632.sha256`); evidence `results/m3/eval_20261103T173632/`.
5 fresh-process sessions, 5 cases, 5,400 measured samples, none excluded. Both backends passed every
strict check and were bitwise equal under identical state.

Measured CUDA calls per restore (profiler session):

| Chunks | Baseline memcpy / sync / launches | Batched memcpy / sync / launches |
|---|---|---|
| 1 | 56 / 56 / 56 | 2 / 1 / 56 |
| 6 | 336 / 336 / 392 | 6 / 6 / 58 |
| 16 | 896 / 896 / 952 | 16 / 16 / 60 |

H2D bytes are identical for both backends. Median first-token latency (d = batched − baseline, paired,
stratified block-bootstrap 95% CI):

| Case (L, S), chunks | Host baseline → batched (ms) | Host median d | Disk baseline → batched (ms) | Verdict |
|---|---|---|---|---|
| (256, 32), 1 | 23.9 → 22.5 | −1.99 [−2.29, −1.71] | 26.5 → 24.6 | faster |
| (1024, 1), 4 | 33.3 → 21.3 | −12.3 [−12.9, −11.9] | 41.9 → 29.1 | inconclusive (rule R3) |
| (1536, 128), 6 | 43.7 → 25.9 | −17.9 [−18.8, −17.3] | 55.3 → 37.0 | faster |
| (3072, 256), 12 | 78.7 → 41.7 | −37.0 [−38.3, −36.2] | 101.4 → 62.7 | faster |
| (4096, 64), 16 | 86.2 → 37.6 | −48.5 [−80.4, −47.8] | 117.7 → 65.0 | faster |

- No case was slower with batched. (1024, 1) failed only the duration rule R3 (73 s of measured span <
  10× its longest slow regime of 8.2 s), so the frozen adoption rule (`supports_adoption`) was false.
- These are exploratory figures for one sample. At (4096, 64) host the speedup is 2.29 (ratio of medians),
  2.48 (ratio of means) or 2.32 (median of paired ratios), depending on the definition.
- The slow mode remains for both backends (33–42% of samples).
- At (256, 32) the paired **mean** still favors prepared-state recompute for both backends (report
  only).
- Unaffected by the batched backend: admission's 56 D2H copies per admitted chunk, the D4 constructor
  copy and the validated disk read.
- **VAL-006:** the M3 analysis originally inferred the experiment from whichever files were present, so
  missing evidence could still report adoption. It now validates against an explicit frozen spec first
  (exit 3 on invalid evidence, exit 4 on a correctness failure, never a verdict). The reanalysis
  reproduced all 5,938 historical values exactly ([BUGLOG.md](BUGLOG.md)).

### M3_CONFIRM_V1: preregistered confirmation of batched restore
Protocol `docs/M3_CONFIRM_PROTOCOL.md`, frozen before any measurement; post-measurement clarification
`docs/M3_CONFIRM_CLARIFICATIONS.md` (it does not change a rule); evidence `results/m3_confirm/`. A
premeasurement manifest (`M3_CONFIRM_V1.freeze.json`, SHA-256 `dd58b894…8216`, also recorded outside the
repository before measurement) binds the source tree, analysis files, protocol documents, commands and
package stack.

- **Cases:** (1024, 1), 4 chunks, the case M3 left inconclusive (a targeted confirmation, not a random
  draw); (1280, 96), 5 chunks, a fresh case. Host tier and warm-disk tier.
- **Sampling:** 5 fresh-process sessions; per case and session 4 blocks of 2 warmup + 24 measured
  repetitions in interleaved random order; one batch, 798 s of its 1,800 s budget.
- **Validity:** valid, 0 problems, 0 exclusions. Measured samples 8,640/8,640; strict correctness checks
  20/20; baseline-vs-batched A/B checks 20/20.

| Case | Tier | Median d, ms [95% CI] | Median latency baseline → batched (ms) | Ratio of medians | Verdict |
|---|---|---|---|---|---|
| (1024, 1) | host | −13.99 [−14.49, −13.61] | 36.0 → 22.5 | 1.600 [1.577, 1.620] | batched faster |
| (1024, 1) | disk (warm) | −14.27 [−15.14, −13.88] | 44.1 → 30.3 | 1.458 [1.445, 1.471] | batched faster |
| (1280, 96) | host | −16.42 [−16.66, −16.12] | 42.1 → 25.9 | 1.627 [1.613, 1.635] | batched faster |
| (1280, 96) | disk (warm) | −16.84 [−17.18, −16.54] | 52.1 → 35.5 | 1.469 [1.459, 1.481] | batched faster |

- All 8 primary verdicts (median and mean, per row) are "batched faster", so the preregistered
  confirmation holds. Duration rule R3 passed (e.g. (1024, 1): 319.0 s ≥ 10 × 11.76 s).
- Verdicts use the paired difference d. The speedup ratios are descriptive only; the ratio of means
  gives 1.45–1.56 and the median of paired ratios 1.44–1.59.
- Means lie far above medians because of the slow mode (31–42% slow samples per row).
- The duration rule's effective sample size is summed over strata (352–369 per row); six individual
  strata are below 10, and the frozen rule has no per-stratum minimum (clarification C1 in
  `docs/M3_CONFIRM_CLARIFICATIONS.md`, which does not change the rule).
- Context only: with the baseline backend at (1024, 1), disk, the paired mean of restore − full
  recompute is +3.69 [0.10, 7.23] ms, i.e. baseline restore was slower than recompute in the mean.
- The freeze is self-attested (the manifest lives in the repository); the out-of-band digest record
  narrows that.
- **Not established:** a request-level, serving or throughput benefit; other machines, models or stacks;
  cold storage. GPU exclusivity is not proven (WSL2's `nvidia-smi` does not list Windows host processes).
- **The default backend remains `baseline`.** The analysis records `authorizes_default_change: false`;
  switching would need a separate decision (D31).

## Trace evaluation (M4)
Protocols `docs/M4B_PROTOCOL.md` (V1) and `docs/M4_EVAL_V2_PROTOCOL.md` (V2), maintained copies of the
frozen texts (see [Evidence notes](#evidence-notes)); evidence `results/m4/`.

**Design:**
- **Traces:** four synthetic traces, evaluation seeds 42001–42004: T0 no reuse (a control), T1 a shared
  system prompt, T2 multi-turn conversations, T3 document reuse.
- **Configurations,** replayed in one process with requests interleaved in a seeded order and fresh state
  per trace:
  - **C0:** no cache, full prefill.
  - **C1-PA(W/2):** this project's GPU-only prefix-aware LRU cache, capped at ⌈W/2⌉ chunks of GPU KV, where
    W is the trace's working set (T1: 3 of 6 chunks, 22 MB; T2: 28 of 56, 206 MB; T3: 29 of 58, 213 MB).
    C1-PA(W) was also run; comparisons against it are secondary.
  - **C2-base, C2-batch:** the tiered always-restore cache with a pinned host slab of ⌈W/2⌉ slots plus a
    disk tier of W + 2 files and no persistent GPU KV, using the baseline or the batched backend.
  - Total resources are therefore unequal: C2 trades GPU memory for pinned host memory, disk and the
    (uncounted) page cache.
- **Metric:** per session, trace and configuration, the mean synchronous first-token latency. For A − B,
  D_s = mean_A,s − mean_B,s.
- **Primary family:** 15 comparisons fixed in advance. A verdict needs all 10 D_s to share one sign **and**
  the Bonferroni-level (1 − 0.05/15) t-interval to exclude 0; otherwise the result is inconclusive.

**M4_EVAL_V1: executed once, invalid, no performance conclusion.** Its frozen rule required every chunk
file of the whole disk tier to be fully resident before each C2 request. For 12 of 4,480 C2 requests the
resident fraction stayed below 1.0 after the prewarm, so the frozen analyzer exited 3. All 10 sessions
completed and the 11,200 strict comparisons found 0 mismatches, but none of the 15 comparisons is
evaluated, and the batch is not re-analysed in part. The cause is unresolved (page-cache reclaim, a WSL
cache drop and contention cannot be told apart from the recorded data).

**M4_EVAL_V2: one valid batch.** It changes only the warm-disk preparation (request-scoped: the files a
request will read are observed, read once if not fully resident, and observed again before the timer
starts) and adds a rule invalidating the batch if WSL's `drop_pagecache` counter changes. Freeze SHA-256
`150f232e…c83a`; 10/10 sessions in 2,005.7 s of 2,700 s; no session stopped, retried or replaced.
Evidence valid: 11,200 evaluation records strictly compared with a resident replay of their own
configuration's decisions, **0 mismatches**; 22 prewarms, all followed by a fully resident pass;
`drop_pagecache` unchanged in all 9,782 snapshots; decisions and counters match the key-level models.
The residency evidence is a per-file observation in one pass immediately before each request: not a
simultaneous snapshot, and not a guarantee at timer start or during the timed reads (pages are not
pinned). It shows that the 22 prewarms completed as specified; it does not show that page-cache
eviction is fixed or identify V1's cause.

d = mean_A − mean_B in ms (d < 0: A faster). Trace means are averaged over the 10 sessions.

| Trace | A − B | Mean A → B (ms) | Mean d (ms) | Bonferroni interval | D_s < 0 / > 0 | Verdict |
|---|---|---|---|---|---|---|
| T1 shared prompt | C2-batch − C0 | 64.0 vs 101.8 | −37.89 | [−46.62, −29.15] | 10 / 0 | **A faster** |
| T1 | C2-batch − C1-PA(W/2) | 64.0 vs 72.7 | −8.74 | [−16.89, −0.59] | 10 / 0 | **A faster** |
| T1 | C2-base − C0 | 96.1 vs 101.8 | −5.76 | [−20.83, 9.31] | 5 / 5 | inconclusive |
| T1 | C2-base − C1-PA(W/2) | 96.1 vs 72.7 | +23.39 | [9.46, 37.32] | 1 / 9 | inconclusive (not unanimous) |
| T2 multi-turn | C2-batch − C0 | 66.9 vs 83.3 | −16.41 | [−23.00, −9.82] | 10 / 0 | **A faster** |
| T2 | C2-batch − C1-PA(W/2) | 66.9 vs 81.9 | −15.06 | [−19.78, −10.35] | 10 / 0 | **A faster** |
| T2 | C2-base − C0 | 82.9 vs 83.3 | −0.37 | [−10.57, 9.83] | 6 / 4 | inconclusive |
| T2 | C2-base − C1-PA(W/2) | 82.9 vs 81.9 | +0.97 | [−8.29, 10.24] | 6 / 4 | inconclusive |
| T3 document reuse | C2-batch − C0 | 70.8 vs 90.9 | −20.10 | [−28.39, −11.81] | 10 / 0 | **A faster** |
| T3 | C2-batch − C1-PA(W/2) | 70.8 vs 62.0 | +8.82 | [5.28, 12.36] | 0 / 10 | **A slower** |
| T3 | C2-base − C0 | 92.9 vs 90.9 | +1.96 | [−11.45, 15.36] | 6 / 4 | inconclusive |
| T3 | C2-base − C1-PA(W/2) | 92.9 vs 62.0 | +30.87 | [22.00, 39.74] | 0 / 10 | **A slower** |
| T0 no reuse | C2-batch − C0 | 153.8 vs 93.3 | +60.54 | [38.95, 82.14] | 0 / 10 | **A slower** |
| T0 | C2-base − C0 | 155.4 vs 93.3 | +62.17 | [38.48, 85.87] | 0 / 10 | **A slower** |
| T0 | C1-PA(W/2) − C0 | 99.2 vs 93.3 | +5.90 | [0.52, 11.29] | 0 / 10 | **A slower** |

**Reading:**
- Counts under the predeclared rule: 5 "A faster", 5 "A slower", 5 inconclusive. Inconclusive never
  means equivalent.
- On T0, where nothing is ever reused, the tiered cache restores nothing but still admits every prompt
  chunk (device-to-host copies and synchronous spills to disk): about +60 ms per request on average. The
  cost was not decomposed further.
- On T3, the half-budget GPU cache serves its hits with device-to-device copies, while C2 copies every
  restored chunk from host or disk; with batched restore C2 is 8.8 ms slower. The cause of the gap was
  not isolated.
- The direct C2-batch − C2-base comparison is secondary and carries no verdict (its point estimates are
  −32.1, −16.0 and −22.1 ms on T1–T3, all 10 sessions negative on each).
- **Assumptions:** every interval and verdict treats the ten session differences as independent and
  approximately normal. Fresh processes do not establish independence, and persistent
  machine/driver/thermal regimes are possible.
- **Scope:** synthetic traces; an interleaved, co-resident experiment (both GPU caches stay allocated
  while every configuration runs), not standalone deployments; C1 is this project's own baseline, not
  vLLM or SGLang; no TTFT, tail-latency, throughput, concurrency or cold-storage claim.

## Restore-length policy and cost model (M5)
Design `docs/M5_DESIGN.md`; protocol `docs/M5_PROTOCOL_DRAFT.md` (executed once, stages A–F); evidence
`results/m5/` with `SHA256SUMS`. All M5 measurements used the batched backend.

**Setup:**
- **Policies compared:** `AlwaysRestore` (longest feasible prefix), `Recompute` (L = 0, admissions still
  stored), `FixedChunks(k)` for every feasible candidate, and `CostAware`.
- **Cost model:** 9 non-negative coefficients fitted by NNLS on a calibration grid of 12 cases (81
  case-candidate rows), then frozen with its predictions before the evaluation launch.
- **Held-out evaluation:** 8 cases with new prompt lengths, tier layouts and admission counts inside the
  calibrated domain (47 candidates in total); 4 sessions.
- **Strict checks** on every sample: restored bytes and bitwise first-token logits against a reference at
  the sample's actual split.

**Execution:** a GPU correctness gate (47 tests) passed; calibration (3,402 samples) and evaluation (3,408
samples) each ran once and passed full validation, with every policy sample executing its frozen choice.
A one-session pilot was diagnostic only and excluded from every result.

**Results** (2,000 block-bootstrap replicates; 95% intervals; equal case weights):
- **Choices:** CostAware chose the same restore length as AlwaysRestore, the maximal feasible prefix, on
  **all 8** cases. On this grid the cost model changed no decision, so an incremental benefit over
  AlwaysRestore is **not demonstrated**.
- **Measured best:** the longest prefix was measured best on 6 of 8 cases. Recompute was measured best
  only on ev-01 (N = 400, one disk chunk); ev-01 and ev-08 are ambiguous (more than one candidate in the
  possibly-best set).

| Delivered difference | Mean (ms) | 95% interval (ms) | Direction |
|---|---:|---|---|
| CostAware − AlwaysRestore | +1.04 | [−0.07, 2.15] | inconclusive |
| CostAware − Recompute | −46.48 | [−48.01, −44.97] | below 0 |
| AlwaysRestore − Recompute | −47.52 | [−48.98, −45.96] | below 0 |

Selection quality (the measured latency of each policy's chosen candidate): CostAware and AlwaysRestore
agree with the measured best on 6/8 cases, Recompute on 1/8. Their signed cross-evaluated mean regret
is −0.01 ms [−1.23, 1.45]; this near-zero aggregate includes cancellation (+3.61 ms on ev-01, −5.74 ms
on ev-02, +2.04 ms on ev-08, 0 elsewhere), and the naive regret (0.71 ms) is biased high. The ev-01
AlwaysRestore − Recompute point estimate (+4.59 ms) is one case, not a general restore overhead.

The CostAware − AlwaysRestore difference is neither a selection effect (the choices are identical) nor an
isolated measurement of planning overhead (the policy runs are separate samples in the seeded order).

**Prediction error** of the frozen model. Relative error = (predicted − measured candidate mean) /
measured candidate mean; the aggregate weights the 47 candidate rows equally:

| Rows | Mean signed error | Mean absolute error | Mean absolute relative error | Max absolute relative error |
|---:|---:|---:|---:|---:|
| 47 | −0.65 ms | 4.44 ms | 5.01% | 31.15% |

The three largest errors are under-predictions of the longest candidate of the three shortest prompts
(ev-01 L=256 −31.2%, ev-08 L=768 −30.0%, ev-02 L=512 −21.2%); overall 27 rows are over-predicted and 20
under-predicted. A 5% mean error is not "95% accuracy" and hides errors up to 31% on single candidates.

**Scope:** one machine, warm page cache, batch 1; one fixed 8-case grid and one fitted model; the intervals
are conditional on the fixed model, grid and observed sessions and do not cover all thermal drift
(software thermal slowdown was flagged on up to 39% of monitor rows late in the evaluation). The
evaluation data have now been observed and cannot serve as an untouched holdout for later model work.

## Test runs
Only runs that actually occurred are listed; "local" means the log was kept outside the repository.

| Run | Result |
|---|---|
| Full CPU suite at the end of the M5 preparation | 1,160 passed, 71 deselected (local) |
| GPU M1 (both backends), M3 and M5 suites in the M5 gate (`9785c33`) | 23 + 17 + 7 = 47 passed (local) |
| GPU M0.1 regressions, M0 strict suite (M3, `results/m3/logs/`) | 6 passed; 10 passed, 1 failed (`nat_long`, documented) |
| M4 GPU correctness gate (M4-B preparation) | 46 passed on the third attempt; attempt 1 failed on a test-harness basetemp defect (local) |

## Reproducing
Analyses are CPU-only, need no model weights, and refuse to overwrite existing outputs. GPU experiments
must run one process at a time. Every formal experiment is reproducible offline from its archived
evidence with the analysis code of the revision that produced it:

| Experiment | Recorded source | Offline check |
|---|---|---|
| M3_CONFIRM_V1 | tree of `13afde2` (equals the frozen tree) | its analyzer in the current tree, or `scripts/m3_confirm_reconstruct.py --tree-rev 13afde2` and that tree's `scripts/m3_analyze.py` |
| M4_EVAL_V2 (and V1) | `d2122d7` (V1: `b858d9e`) | `scripts/m4_reconstruct.py build`, that tree's `scripts/m4_analyze.py`, then `scripts/m4_reconstruct.py compare` |
| M5 | `a3c6341` | `git archive a3c6341`, then that revision's `scripts/m5_fit.py` and `scripts/m5_analyze.py` |

**What the freeze manifests record.** Each M3_CONFIRM_V1 and M4 manifest records three kinds of hashes:
- the whole source tree at freeze time (`files_sha256`), which the reconstruction tools check against the
  rebuilt historical tree;
- the analysis files (7 for M3_CONFIRM_V1, 13 for M4), which a frozen analyzer compares with the tree it
  runs in;
- the protocol documents (`documents_sha256`), compared the same way.

So a frozen analyzer runs in the current tree only while its analysis files and documents keep their
frozen bytes; otherwise it refuses the evidence (exit 3) by design, and the reconstruction is the way to
reproduce. M3_CONFIRM_V1's files and documents are unchanged, so its analyzer runs in this tree. For M4,
`kvlab/m4_spec.py` (docstring only) and the two M4 protocol documents were edited after the evaluations
(see Evidence notes), so M4_EVAL_V1 and M4_EVAL_V2 are reproduced from their frozen trees.

The README has the commands for M3_CONFIRM_V1, M4_EVAL_V2 and the M5 analysis. With the same variables
(`REPO`, `PY`, `WORK`), the remaining two are:

```bash
# M4_EVAL_V1 (invalid evidence): its frozen analyzer is expected to exit 3 with the 12 residency problems
$PY scripts/m4_reconstruct.py build --spec M4_EVAL_V1 --out $WORK/m4v1-tree \
  --expect-freeze-sha256 137220aa1f40c9604fda983380f03d2e4f897cb2a8b6ad64ac97eabd3a3625b2
(cd $WORK/m4v1-tree && $PY scripts/m4_analyze.py --spec M4_EVAL_V1 --run-dir results/m4/M4_EVAL_V1_run \
  --out-dir $WORK/m4v1-out --expect-freeze-sha256 137220aa1f40c9604fda983380f03d2e4f897cb2a8b6ad64ac97eabd3a3625b2)
ARCHIVED_ROOT=$($PY -c "import json; print(json.load(open('results/m4/M4_EVAL_V1_analysis/m4_validation.json'))['run_dir'].rsplit('/results/m4/', 1)[0])")
$PY scripts/m4_reconstruct.py compare --archived results/m4/M4_EVAL_V1_analysis \
  --regenerated $WORK/m4v1-out --archived-root "$ARCHIVED_ROOT" --regenerated-root $WORK/m4v1-tree

# M5 fit, from the recorded revision extracted in the README's M5 step ($WORK/m5-src)
(cd $WORK/m5-src && $PY scripts/m5_fit.py --cal-run-dir $REPO/results/m5/M5_CAL_V1_run --out-dir $WORK/m5-fit)
for f in validation.json calibration_means.json cost_model.json; do cmp $WORK/m5-fit/$f results/m5/M5_FIT_V1/$f; done
```

A reconstruction of each experiment has reproduced every archived value. The only differences were
recorded absolute paths (`out_dir` in the M3 analysis JSON; `run_dir` in `m4_validation.json`, the one
path-only difference `compare` accepts) and, in a regenerated `predictions.json`, its creation time and
its own digest; the archived predictions were frozen before the evaluation and must not be replaced.
These reproductions ran on the recording machine. On a CPU with a different BLAS kernel, values computed
through BLAS can differ in their last bits: for the M3 analysis's `ar1_effective_n`, the observed
difference is 1 ULP, and the tests accept up to 4 ULPs ([D45](decisions.md)).

**What offline reproduction does not verify:** latencies (a new measurement would be a new experiment),
the GPU tensors behind the strict checks (the validators check their recorded results), and the
pre-launch environment checks, which ran on the recording machine.

## Evidence notes
- Archived evidence keeps its original bytes, including absolute paths of the checkouts it was recorded
  in (for example a validator's `run_dir`); manifests and analyzers hash these files, so they are never
  edited.
- Historical reproduction uses the recorded source revisions listed above, not the current tree.
- Files edited after their hashes were recorded, so that recorded and current hashes differ:
  - `kvlab/boundaries.py`: one docstring line changed after the M3 evidence was recorded;
  - `kvlab/m4_spec.py`: its module docstring was revised after both M4 evaluations; the spec dictionaries,
    and therefore the recorded `spec_sha256` values, are unchanged;
  - `docs/M4B_PROTOCOL.md` and `docs/M4_EVAL_V2_PROTOCOL.md`: maintained copies whose status notes were
    edited after the evaluations, with the rules unchanged. The frozen texts are at
    [`b858d9e`](https://github.com/cyruswu750/kv-cache-lab/blob/b858d9e9f1ff78bd157b2af80720b54ae85cda84/docs/M4B_PROTOCOL.md) and
    [`d2122d7`](https://github.com/cyruswu750/kv-cache-lab/blob/d2122d7672c327b9fb48ad595dff9d665ec614b8/docs/M4_EVAL_V2_PROTOCOL.md).
