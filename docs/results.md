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
