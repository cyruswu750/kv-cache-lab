# M2.1 Review — measurement audit and restore-path slow-mode diagnosis

Date: 2026-11-03.

> **M3 qualification note (2026-11-03).** The next task recommended in §11 (fewer CUDA
> calls on the restore path) was implemented in M3 as an opt-in backend. The statistical
> directions below are conditional on the M2.1 sample, analysis and the baseline backend.
> M3 measured the batched backend separately (`docs/M3_REVIEW.md`). With it, the slow mode
> remains (similar slow shares), now mostly through the suffix forward's launches. The
> cause is still not established.

**Status: M2.1 done.** The M2 timing code is sound. The M2 *scoring*
overstated its certainty (BUGLOG VAL-005), and several M2 claims are
qualified: see the note at the top of `docs/M2_REVIEW.md`.

**The slow mode is narrowed, not explained.** In time-clustered regimes
(median 1.4 s, up to ~10 s), the host-side CPU cost of CUDA driver/runtime
calls rises 2–5×. Kernel launches go from ~10 to ~48 µs each, and small
synchronous H2D copies from ~40 to ~90 µs each. In those regimes:
- GPU kernel execution time is unchanged;
- ordinary CPU work and guest system calls are unchanged;
- the GPU clock is unchanged;
- the thread stays on-CPU.

The regime is shared by every path. Recompute hides it because its GPU work
outlasts the slowed submission. Restore paths expose it, because they
consist of thousands of small CUDA calls. **Root cause: not established.**

No runtime policy, kernel, dependency, power or driver change was made.
Nothing was committed or staged.

## 1. Baseline preservation
- Git before editing: branch `main` with **no commits (HEAD does not
  exist)**; 0 index entries; all project files untracked
  (`results/m2_1/pre_m2_1_git_state_20261103T145039.txt`).
- Pre-M2.1 snapshot: `backups/pre-m2_1-20261103T145039.tar.gz`, sha256
  `7ea560848ae842a882d854396921990c37dc5546c2a57b73059a248988a3b658`.
  - 180 files: source, tests, scripts, configs, docs and small evidence,
    including all of `results/m2/`.
  - It excludes `.git`, `.venv`, `.hf_cache`, `.kvcache`, `backups/`,
    bytecode and `.pytest_cache`.
  - It was extracted to a scratch directory and verified file by file
    against `results/m2_1/pre_m2_1_manifest_20261103T145039.sha256`.
  - It was never restored over the working tree.
- The pre-M2 snapshot was re-verified unchanged (`sha256sum -c`: OK).
- **Historical evidence untouched.** The M2 samples, `model.json`,
  `predictions.json`, `results.*` and `spike_probe.*` are byte-identical to
  the snapshot (change inventory, §9). `kvlab/perf.py`, which holds the
  frozen model code, is byte-identical too.

## 2. Code audit (actual code, not the review document)
| # | Item | Finding | Verdict |
|---|---|---|---|
| A1 | First-token timing boundary, CUDA completion | `torch.cuda.synchronize(); t0; prefill_request; int(argmax); t1` (`perf_harness.Bench.measure`). All engine copies are blocking `copy_` on the default stream (`tiered_cache.restore`/`store`). No other stream exists in `kvlab/`. `int()` waits for the stream. Admission D2H is blocking, so it is included. | correct |
| A2 | Reference caches or temporary GPU tensors surviving into timing | `correctness()` runs before each L group's loops, deletes its caches and keeps logits on the CPU. `Bench` holds no GPU tensors. Measured in M2.1: `num_device_alloc`, `num_device_free` and `num_alloc_retries` deltas are **0 in all 3,456 baseline samples** (and in every later session). | correct |
| A3 | Reset and cache-state equivalence | `check_state()` after every sample asserts host A = prefix keys, host B empty, disk B = prefix keys, and no undeletable files. Not controlled: which free host slot receives an admitted chunk, allocator block reuse across random path orders, and growing counters. None of these changes restored data or the work done. | adequate (residual noted) |
| A4 | Tier, restore-length and admission checks | `perf.sample_problems` checks `restore_len`, per-chunk `sources`, exact `stored_chunks` and page-cache residency on every sample. The values are the engine's own returns, and once per case they are backed by byte and bitwise logit checks. | correct |
| A5 | Calibration/evaluation separation, freezing | `fit_cost_model` refuses non-calibration cases. `m2_fit_predict` refuses if evaluation samples exist. The evaluation records the predictions digest. `m2_score` verifies the digest chain, the time order and that the cases equal the split. Re-verified by `m2_1_reanalyze` (all checks true). | correct |
| A6 | Paired differences, CIs, ties, penalties | Pairing by repetition is correct. The CI is an **i.i.d.** percentile bootstrap of the median, but the series are autocorrelated (lag-1 up to +0.93). Penalties are `None` for ties and are summed over scorable decisions only, so "0 ms penalty" excluded every ambiguous decision. Penalties use separate per-path medians, not paired data. | **defect → VAL-005** (§3) |
| A7 | Source identity | `source_manifest` hashes `kvlab/*.py` and `scripts/m2_*.py`, so it also covers the `m2_1_*` scripts. The environment records torch, transformers, CUDA, Python, GPU and driver. **Missing:** numpy and `nvidia-*` wheel versions, model blob hashes (only the revision string), SDPA backend flags, TF32 and thread settings. M2.1 sessions record them (`packages`, `weights`, `process`); for example the weights blob is sha256 `dd924a11…c6d3ee`. | gap (recorded from M2.1 on) |
| A8 | Spike probe (`m2_spike_probe.py`) | **No cache growth:** a new `DynamicCache` is built from the cloned K/V in every repetition, `prefill` asserts that the length equals `start_pos`, and the K/V are not mutated (`DynamicLayer.update` concatenates into new tensors). The M2.1 audit re-ran the probe function 20× in each of 10 sessions (smoke included): every repetition went 256 → 257, and the source K/V pointers, shapes and bytes were unchanged. **But** the probe timed its components in *sequential* blocks of 210. Its own data shows slow samples concentrated in particular deciles of each block, so per-component slow shares are confounded with time. | design flaw; M2 localization claim withdrawn |
| A9 | Instrumentation offset | Instrumented loops run after the uninstrumented ones, not interleaved, so the offset (e2e − instrumented total) also absorbs drift. M2.1 shows the synced marks do not change the slow share (§4). | minor; documented |
| A10 | The `full` baseline's meaning | Prefix chunks are cached in store A and restore is not attempted. Chain keys, counters and admission run as in the engine: **recompute from the prepared cache state** (D28). It is kept unchanged as the baseline of every M2 score. | renamed in reports |

**New diagnostic baseline `full_nocache`** (D28):
- **What it does:** token ids → device → `kvlab.model.prefill` with an empty
  cache → `int(argmax)`. The timing boundary is the same as for every other
  path. There are no chain keys, lookups or admissions.
- **Checks (untimed, 26 first-visit blocks across 10 sessions, smoke
  included):** its logits are bitwise equal to `full`'s in every block.
- **Timing:** within 0.4 ms of `full` in the measured cases. Those cases
  have S < 256, so `full` admits nothing either.
- **Scope:** it is not mixed into any M2 score.

## 3. Reanalysis of the saved M2 data (`results/m2_1/reanalysis/`)
**Method** (fixed in `scripts/m2_1_reanalyze.py` before it first ran):
- **Series:** per-repetition paired d = restore − `full`, in chronological
  order (checked against `t_wall`).
- **Intervals:** circular moving-block bootstrap with block lengths
  b = 1 (= M2's i.i.d. method), 2, 4, 8, for the median and the mean;
  4,000 resamples.
- **Directions:** a direction is established only if all four block lengths
  agree; otherwise "inconclusive".
- **Reproduction check:** every original M2 interval was first reproduced
  exactly (`all_original_cis_reproduced: true`).
- **Refitting:** none.

**Held-out decisions (18 = 9 cases × 2 tiers):**

| Statistic | M2 (i.i.d.) | M2.1 (all block lengths agree) |
|---|---|---|
| Median of d | 11 restore, 7 "tie" | **9 restore faster, 9 inconclusive** |
| Mean of d | not reported | **8 restore faster, 6 recompute faster, 4 inconclusive** |

**Cases that changed:**
- **L=1024, S=64 (host and disk):** M2 called restore the winner. With
  b = 8 the result is inconclusive: host median CI [−27.0, +30.2] ms
  against M2's [−27.0, −26.0]. These series have lag-1 ρ ≈ 0.85–0.88 and
  AR(1) n_eff ≈ 3 of 40.
- **L=512, all S, both tiers (6 decisions):** M2 called these ties or
  restore wins. By the **mean**, recompute is established faster at every
  block length: +11.7 to +19.1 ms per decision. By the **median**, five are
  inconclusive and one (disk S=64) is restore-faster by 4 ms.
  - The two statistics disagree because restore is bimodal: its fast mode
    beats recompute by a few ms, and its slow mode loses by ~50 ms.
  - Which one matters is a choice of metric. **Expected latency (the mean)
    favors recompute at L = 512 in this dataset.**
- **Stable at every block length, for both statistics:** restore faster at
  L = 2048 and 4096 (both tiers).

**Uncertainty is not reliably estimable for some series:**
- **Highly autocorrelated cases:** the L=256 S=64 and L=1024 S=64 series
  have AR(1) effective sizes of 1–3, and their 40 repetitions span only
  12–14 s.
- **Regime length vs run length:** slow regimes last about 1.4 s (median)
  and up to about 10 s (§4). These runs are therefore too short to estimate
  uncertainty reliably, at any block length.
- **Calibration:** with 20 repetitions, b = 8 leaves only 2.5 blocks, which
  is flagged in the data.

**Selection penalty of the frozen model's choices** (metric:
mean(chosen) − mean(other) over paired repetitions, clipped at 0):

| Decisions | Point penalty | Status |
|---|---|---|
| all 18 | 110 ms in total | 6 **established** (all at L = 512; 92 ms) · 4 inconclusive (L = 256 and 1024 at S = 64; 18 ms) · 8 established absent (L ≥ 2048) |

By the median, the point penalty is 39 ms in total, none of it statistically
established. **The earlier "0 ms" figure covered the scorable decisions
only.**

**Policies on the held-out decisions** (sum of per-decision means):

| Policy | Sum (ms) | Regret vs per-case oracle (ms) |
|---|---|---|
| frozen model | 1395 | 110 |
| always-restore | 1395 | 110 |
| always-recompute | 1742 | 457 |

- **The model's choices are identical to always-restore** (it predicted
  restore everywhere). It therefore gives **no measured decision benefit
  over always-restore in this dataset**.
- Always-restore beats always-recompute by 347 ms in total. Joint block
  bootstrap 95% CI: [−440, −251] ms at b = 8, and similar at every b. That
  benefit belongs to the always-restore baseline, not to the model.
- The model was not refitted. Any revised model needs a fresh evaluation
  set (§6).

## 4. Diagnostic experiment (`docs/M2_1_DIAGNOSTIC_PLAN.md`, written before any M2.1 GPU sample, with two amendments recorded before their sessions)
**Setup:**
- Cases (L, S): (512, 1), (1024, 64), (4096, 64).
- Paths: `full`, `host`, `disk` (warm), `full_nocache`.
- Modes: `plain` (the unchanged M2 region), `events` (CUDA events + wall +
  thread CPU time at phase boundaries, no sync), `synced` (the unchanged M2
  instrumented region).
- In every repetition, `gc.collect()` runs, then all 12 (path, mode) pairs
  run in a fresh random order. Each session visits each case in two
  separate blocks (3 warmup + 12 measured repetitions).
- Every raw sample is kept, with session, block, repetition, order
  position, `t_wall`, thread rusage and `schedstat`, and allocator deltas.

| Run directory | Sessions | Config (one factor changed at a time) | Measured samples | Excluded | Correctness |
|---|---|---|---|---|---|
| `diag_20261103T150340` | s1–s4 | baseline (M2 config, monitor on) | 3,456 | 0 | strict + nocache bitwise PASS; spike audit PASS |
| `diag_probes_20261103T151540` | p1, p2 | baseline + probes (Amendment 1) | 2,304 | 0 | PASS |
| `diag_nomonitor_20261103T151540` | f1 | baseline + probes, **GPU monitor off** (F1) | 1,152 | 0 | PASS |
| `diag_profiler_20261103T151540` | prof | torch.profiler, host vs full_nocache, L=512 S=1 | 120 | 0 | — |
| `diag_syscall_20261103T152505` | q1 | baseline + probes + syscall probes (Amendment 2) | 1,296 | 0 | PASS |

The analysis outputs for each directory are `diag_analysis.md` and
`diag_analysis.json`; regime durations are in
`diag_20261103T150340/logs/regime_durations.txt`.

The slow label is the M2 definition: e2e > 1.3× the minimum of the same
(config, path, mode, case).

**Findings, by hypothesis:**

| Hypothesis | Contrast | Result | Verdict |
|---|---|---|---|
| H1 GPU clock | slow vs fast at the monitor-reported SM clock | median SM clock 2370 MHz for slow *and* fast samples; 1,658 of 1,728 restore samples at ≥ 2200 MHz, with a slow share of 32% there | **excluded** as the explanation of the slow mode (clock regimes are a separate effect, below) |
| H2 GPU execution | profiler GPU kernel time, slow vs fast host samples | 9.93 vs 9.92 ms kernels and 1.26 vs 1.25 ms GPU memcpy; corr(e2e, kernel time) ≈ 0.0 | **excluded** |
| H3 host-side submission/sync | `events`: forward submission wall time and thread CPU | host L=512 S=1: submission 16.3 → 61.6 ms wall, 16.3 → 61.3 ms thread CPU; the GPU finishes ~1 ms after submission ends in both cases; restore loop 8.3 → 21.2 ms; similar at L=1024 and 4096 | **supported** |
| H3a vs H3b on-CPU vs off-CPU | thread CPU vs wall; `schedstat` runqueue wait; context switches | thread CPU = wall; runqueue wait ≈ 0.00–0.01 ms; 0–1 involuntary switches | **H3a**: the thread runs, but the calls take longer. Hypervisor time is invisible to the guest; WSL reports no steal |
| H4 instrumentation | slow share plain / events / synced (same repetitions); F1 without monitor | host L=4096: 40% / 42% / 45%, disk 44% / 44% / 44%, and similar elsewhere; event-mode medians within ~1.5 ms of plain; **without the monitor**, 29% slow, the same clustering, and the launch probe still 4.65× slower | **excluded** as the cause (marks and the monitor are not needed for the slow mode; one F1 session cannot exclude a small effect on its frequency) |
| H5 restore-specific vs shared | `full`/`full_nocache` forward submission in repetitions where restore is slow vs fast | L=512: 17.3 → 38.2 ms; L=1024: 17.4 → 34.8 ms; recompute e2e unchanged (38.3 vs 40.5 ms; 64.5 vs 65.4 ms) and slow share 0%, because the GPU span (38/64 ms) covers the submission. At L=4096, submission ~38 ms in both states (launch queue full, GPU-bound) | **shared state**, hidden in recompute by GPU time |
| H6 allocator | device alloc/free/retry deltas | 0 in every sample (543 slow, 2,913 fast) | **excluded** |
| H7 session | slow share per session and block | sessions 22–45%; blocks 6–65% within every session; P(slow \| previous slow) 0.70–0.78 | **excluded** as session-level: a time-varying state within sessions |
| H8a all CPU work slower vs H8b only CUDA calls | probes in restore-slow vs fast repetitions (p1, p2; f1; q1) | SHA-256 8 MiB ×0.99–1.00; pure-Python loop ×1.00–1.01; `getppid` ×1.01; `read /dev/zero` ×1.00; **500 launches ×4.0–5.0** (≈10–12 → 48 µs per launch); **56 sync H2D copies ×2.2–2.35** (≈38–41 → 90 µs per copy); thread CPU = wall | **H8b**: the slowdown is confined to CUDA driver/runtime calls |

**Profiler** (perturbed; never used as a latency estimate):
- **Slow samples:** in the profiled session, 4 of 40 host samples were
  slow. Their CPU-side `cudaLaunchKernel` time rose from 7.7 to 23.3 µs per
  call, with 1,343 launches per sample. The correlation of e2e with
  per-launch API time was 0.99.
- **Perturbation:** profiled host medians were about 38 ms, against about
  26.5 ms for the baseline-session `plain` medians, i.e. about +40%.
  `full_nocache` rose about 6%.
- **The in-session unprofiled reference is unusable.** It ran after the
  profiler stopped, at an SM clock of about 1300 MHz (event reason 0x1,
  GPU idle). That doubled the GPU-bound `full_nocache` (38 → 64 ms) and left
  the launch-bound host path at ~31 ms.
- So GPU clock regimes are a real effect, but a *different* one. They hit
  GPU-bound work, while the slow mode hits CUDA-call-bound work.

**Regime durations** (baseline sessions, restore samples): slow regimes
lasted a median of 1.4 s (about 6 samples), p75 2.7 s, max 10.7 s; fast
regimes lasted a median of 1.5 s, max 17.5 s.

**Spike-probe audit:** PASS in all 10 non-profiler sessions (8 + 2 smoke) (start 256, end 257; source K/V unchanged).

**Host syncs inside the forward** (torch sync-debug, untimed): 2
synchronizing operations in both the cached and the full forward. In both,
submission is far shorter than GPU time for GPU-bound cases, so the syncs
occur early. They do not turn "submission" into a long wait.

**Diagnostic replication (not a score; same 3 cases, baseline `plain`, 96
samples per path):**

| Case | full med / mean | host med / mean | disk med / mean |
|---|---|---|---|
| L=512, S=1 | 38.4 / 39.0 | 26.5 / 41.9 | 30.4 / 44.0 |
| L=1024, S=64 | 64.8 / 64.9 | 37.2 / 52.7 | 45.1 / 57.9 |
| L=4096, S=64 | 251.1 / 254.2 | 88.6 / 131.5 | 119.0 / 168.8 |

These are consistent with §3: at L=512, restore wins on the median but not
on the mean.

**Narrowed explanation and what remains open:**
- **What is established:** in this environment (WSL2, RTX 4080 Laptop GPU,
  driver 596.49), host-side CUDA driver/runtime calls alternate between
  regimes whose per-call CPU cost differs by 2–5×. Ordinary CPU work, guest
  system calls, GPU execution and the GPU clock do not change between them.
- **What remains open:** which layer below the CUDA runtime causes this.
  The candidates (user-mode driver internals, the WSL GPU paravirtualization
  path, host-side GPU scheduling, host power management) cannot be told
  apart from inside the guest without host-side tracing or privileges. None
  is claimed.
- **Stopping rule:** the plan's rule (stop when contrasts no longer separate
  hypotheses) was applied after the syscall session.

## 5. Implications (no runtime change made)
- **Which work is exposed:**
  - **Restore latency on this platform is roughly a two-state mixture.**
  - Its exposure scales with the *number of CUDA calls*: 56 copies + 56
    syncs per restored chunk, plus about 1,340 launches in the suffix
    forward. It does not scale with bytes moved.
  - Recompute is exposed only where it is launch-bound (small N).
- **What the M2 model gets wrong:**
  - It predicts fast-mode medians, and its per-chunk restore slope mixes
    copy-call overhead with bandwidth.
  - Its choices equal always-restore. They lost on the mean at L = 512 and
    were indistinguishable at L ≤ 1024 for S = 64.

## 6. Proposal only — a revised predictive model (not fitted)
**Proposed form:**
- Predict the **mean**, as `fast_components + p_slow × Σ_calls Δc`.
- The call counts are known per path: launches of the suffix or full
  forward, and H2D copies and syncs of the restore.
- Δc would be the per-call excess, measurable with the M2.1 probes
  (~38 µs/launch, ~50 µs/copy).
- p_slow is a platform parameter, estimated in calibration with block-aware
  uncertainty.
- A GPU-clock term should apply only to GPU-bound components.

**Evaluation requirement:** a fitted version would need a **fresh, frozen
held-out evaluation set**. The original M2 held-out set has now been
inspected and can no longer serve as unseen data.

## 7. Verification
| Check | Result |
|---|---|
| Focused CPU tests: `tests/test_perf_stats.py` (10), `tests/test_diag_bookkeeping.py` (16), `tests/test_perf.py` (18), `tests/test_engine_cpu.py`, `tests/test_host_store.py` | **PASS** (final run below) |
| M2 smoke after the `perf_harness` refactor (`page_cache_before` extracted; behavior unchanged) | **PASS**: `results/m2_1/m2_smoke_after_refactor_20261103T152815/` (140 valid, 0 excluded, strict correctness, exit 0) |
| Correctness of the new diagnostic paths, untimed, every block: M2 strict check + `full_nocache` bitwise = `full` | **PASS** in all 26 first-visit blocks (10 sessions incl. 2 smoke) |
| Every diagnostic sample validated (tier, L, admissions, residency, event marks, probe work checks) | **PASS**: 0 excluded in all sessions |
| Freeze chain of M2 re-verified; original intervals reproduced | **PASS** |
| Full CPU suite, GPU M1/M0.1 suites, original M0 strict suite | **NOT RUN** (unrelated; not rerun by instruction). D10 and the known M0 strict `nat_long` FAIL are unchanged |

GPU time: 9 diagnostic sessions (about 17 min of measurement, plus about
0.3 min of model load each), 3 smoke sessions and 1 M2 smoke, so about
24 min of GPU time. One GPU process at a time. No OOM.

## 8. PASS / FAIL / BLOCKED / NOT RUN
| Item | Status |
|---|---|
| Pre-M2.1 snapshot, manifest, verification; pre-M2 snapshot preserved | PASS |
| Code audit (A1–A10) | done: 1 scoring defect (VAL-005); 1 diagnostic-design flaw (A8, qualification); 1 identity gap (A7) |
| Timing boundary / CUDA completion / no surviving tensors / state reset / tier checks / freeze chain | PASS |
| Dependence-aware reanalysis of M2 | done; M2 decision claims **qualified** (§3) |
| Frozen model vs always-restore / always-recompute | model = always-restore; **no measured decision benefit** over it |
| Diagnostic experiment | done; slow mode **narrowed** to CPU-side CUDA call cost; root cause **open** |
| Profiler capture | PASS (available without installation; perturbation recorded; in-session reference invalidated by a clock drop) |
| F1 (monitor off) | run: slow mode persists |
| F2 (spin schedule) | NOT RUN: conditional on H3b, which the data does not support |
| Revised model fit | NOT RUN (proposal only, §6; needs a fresh evaluation set) |
| Full M2 grid rerun | NOT RUN (not needed for the contrasts) |
| Cold-NVMe latency | still BLOCKED; unclaimed |
| p99 / tail latency | NOT RUN |

## 9. Change inventory against `backups/pre-m2_1-20261103T145039.tar.gz`
Machine-readable: `results/m2_1/change_inventory_vs_pre_m2_1.txt` (computed
last).
- **Modified:**
  - `kvlab/perf_harness.py`: `page_cache_before()` extracted from
    `measure()`, with no behavior change;
  - `README.md`; `docs/M2_REVIEW.md` (qualification note at the top);
    `docs/BUGLOG.md` (VAL-005); `docs/decisions.md` (D27–D29);
    `docs/design.md` (§10 note); `docs/correctness-matrix.md` (M2.1
    sections).
- **Added:**
  - code: `kvlab/perf_stats.py`, `kvlab/diag_harness.py`;
  - scripts: `scripts/m2_1_reanalyze.py`, `scripts/m2_1_diagnose.py`,
    `scripts/m2_1_analyze_diag.py`, `scripts/m2_1_regimes.py`;
  - tests: `tests/test_perf_stats.py`, `tests/test_diag_bookkeeping.py`;
  - docs: `docs/M2_1_DIAGNOSTIC_PLAN.md`, `docs/M2_1_REVIEW.md`,
    `docs/M2_1_WALKTHROUGH.md`;
  - evidence: everything under `results/m2_1/`.
- **Deleted:** none. `kvlab/perf.py`, all M2 scripts and all of
  `results/m0*`, `results/m1*` and `results/m2/` are unchanged.

## 10. Limitations
- **Platform:** one laptop GPU under WSL2. The regime behavior may not exist
  on other systems, and its cause is unknown.
- **Run length:** the runs are short compared with regime durations, so
  confidence intervals, even block-based ones, are approximate. Some M2
  series cannot support a reliable uncertainty estimate at all.
- **Scope of the "slow" label:** it is a relative-threshold label (1.3× the
  minimum), not a fitted mixture model.
- **Profiler:** it inflates per-op cost by about 40% on the launch-bound
  path. It was used only for within-session slow-vs-fast contrasts.
- **Factor tests:** single-factor tests had one session each (F1).
- **Workload scope:** S = 256 (admission) and the true-miss recompute (all
  chunks admitted) were not part of M2.1.

## 11. Recommended next engineering task (not started)
**Reduce the number of CUDA calls on the restore path:** replace the 56
per-layer synchronous H2D copies per chunk with one contiguous H2D transfer
per chunk into a device staging buffer, followed by device-side placement
into the per-layer cache tensors with a single synchronization per restore.
Then re-measure host and disk restore with the M2.1 protocol (interleaved
paths, block-bootstrap intervals, mean and median), on a fresh evaluation
set.

**Supporting measurements:**
- **Restore dominates the restore-path latency.** At L=4096 the restore
  phase is 61 of 87 ms (host) and 89 of 116 ms (disk) in the fast regime,
  and 152 / 181 ms in the slow regime.
- **The slowdown is per call.** Synchronous small copies cost 2.2–2.35× as
  much in the slow regime, while bytes-bound work is unaffected.
- **Fewer calls cost less.** A single 7 MB H2D copy took 0.61 ms, against
  2.31 ms for the 56 copies the restore makes (M2 micro-benchmark).
- **Recompute shows the same effect.** It is insensitive to the slow regime
  exactly where its CUDA-call submission is short compared with its GPU work.

**What this does not address:** the ~1,340-launch suffix forward remains
exposed. The task changes restore mechanics only, not policy, chunk format
or numerics. Its correctness gate is the existing strict bitwise checks.
