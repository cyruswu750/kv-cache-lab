# M2 Review — synchronous performance calibration and an offline restore-vs-recompute cost model

Date: 2026-11-03.

> **M3 qualification note (2026-11-03).** The frozen M2 cost model, its predictions
> and every M2 timing describe the **baseline** restore backend (56 blocking H2D copies per
> chunk). M3 added an opt-in `batched` backend. Measured, it lowers the restore-phase time
> 1.5–5.2× and median e2e 1.07–2.29× in its own sample (`docs/M3_REVIEW.md`). The M2 model was
> not refitted, and it does not describe the batched backend. Restore-vs-recompute
> conclusions here hold for the baseline backend only.

> **M2.1 qualification note (2026-11-03).** The audit in
> `docs/M2_1_REVIEW.md` qualifies the following claims below. The data,
> model and predictions are unchanged.
> 1. **"11 of 11 scorable decisions correct, 0 ms penalty" (§6, §9).** The
>    intervals assumed independent repetitions, but the series are clustered
>    in time (lag-1 ρ up to 0.93). BUGLOG VAL-005.
>    - **Medians, block bootstrap:** 9 of 18 held-out decisions are
>      established restore-faster and 9 are inconclusive. The two L=1024 S=64
>      wins are no longer established.
>    - **Means:** recompute is established faster in all six L = 512
>      decisions. The model chose restore there, with a point penalty of
>      92 ms in total.
>    - **"0 ms" penalty:** it covered scorable decisions only.
>    - **The model's decisions equal always-restore,** so the model gives no
>      decision benefit over that baseline in this dataset.
> 2. **"Tie"** should read **"inconclusive"**. It never meant equivalence.
> 3. **"`full`" baseline:** it means *recompute from the prepared cache
>    state* (D28).
> 4. **Spike-probe localization (§6, "Localization").** The probe ran its
>    components in sequential blocks, and the slow mode is clustered in time.
>    So "a single 7 MB copy and the validated disk read never spike" is not
>    supported as a property of those operations. The cache-length concern is
>    cleared: every repetition started at 256 tokens. The interleaved M2.1
>    experiment replaces this localization (M2_1_REVIEW §4).
> 5. **Source identity (§1)** omitted wheel versions, model blob hashes and
>    backend flags. M2.1 sessions record them.
> 6. **Slow-mode candidates (§6, "Root cause not established").** M2.1 narrows
>    them. In the slow regime, the CPU-side cost of each CUDA launch or
>    synchronous copy rises 2–5×, while GPU kernel time, pure-CPU work, guest
>    system calls and the SM clock are unchanged. The root cause is still
>    open (M2_1_REVIEW §4).

**Status: M2 done.** Measurements were made on the full 15-case grid, with
no NOT RUN cases and no exclusions. A cost model was fitted on calibration
data, frozen, and scored on 9 held-out cases.
- **Decision quality:** 11 of 18 held-out decisions were scorable, and all 11
  were correct with a 0 ms penalty. The other 7 were statistical ties.
- **Latency prediction:** good for full recomputation (median error 3.5%),
  but **poor for some restore cases** (up to −67%). The cause is a
  platform-dependent slow mode, documented in §6.
- **Break-even:** no crossing is predicted or reliably measured within
  L = 256…4096. Restore wins wherever the difference is statistically clear;
  small L (≤ 512) is a tie.

No runtime policy was changed, and nothing was committed. The protocol is in
`docs/M2_PROTOCOL.md`; the Chinese walkthrough is `docs/M2_WALKTHROUGH.md`.

## 1. Baseline preservation and identity
- Git: branch `main` has **no commits (HEAD does not exist)**, the index is
  empty, and every project file is untracked. The index was never touched
  (`results/m2/pre_m2_git_state_20261103T125219.txt`).
- Pre-M2 snapshot: `backups/pre-m2-20261103T125219.tar.gz`, sha256
  `3193e1f2cb9c3828139151f81192c8e0802d5ade2788ff5b77f4af017772f6de`.
  - It holds 128 files: source, tests, scripts, configs, docs and small
    evidence.
  - Excluded: `.git`, `.venv`, `.hf_cache`, `.kvcache`, `backups/`, bytecode,
    `.pytest_cache`.
  - It was extracted to a scratch directory, and every entry was verified
    against `results/m2/pre_m2_manifest_20261103T125219.sha256`.
  - It was never restored over the working tree.
- Change inventory against it: §10, and
  `results/m2/change_inventory_vs_pre_m2.txt`.
- Environment (per-run record in `environment` inside `*_samples.json`):
  - GPU and software: RTX 4080 Laptop GPU, 12 GiB, driver 596.49 (max SM
    3105 MHz, power limit not reported); torch 2.14.1+cu130 (CUDA 13.0);
    transformers 5.22.1; Python 3.14.4.
  - Host: AMD Ryzen 9 7945HX on WSL2 kernel 6.18.54.1, AC power (battery
    full).
  - Storage: the disk tier is on the WSL2 ext4 virtual disk `/dev/sdd`
    (`rw,relatime,discard`).
  - Model: `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA, C = 256, chunk
    payload 7,340,032 B (file 7,342,121 B).
- The source manifest (sha256 of `kvlab/*.py`, `scripts/m2_*.py`) is
  recorded in each run.

## 2. What was implemented
| Item | Where |
|---|---|
| `AlwaysRestoreEngine.prefill_request()`: everything before first-token selection, in the existing order (ingest, restore, prefill, store). `run()` calls it unchanged; `allow_restore=False` exists only for offline measurement; an optional `mark` callback gives instrumented samples | `kvlab/always_restore.py` |
| `HostStore.discard(key)`: free an unpinned entry without spilling, to reset benchmark state | `kvlab/host_store.py` |
| Grid/split, expected tier/L/admissions, sample validation, statistics (median, IQR, bootstrap CI, paired differences), exact small NNLS, cost model, frozen predictions, scoring and crossings | `kvlab/perf.py` |
| GPU harness: tier states from one source prefill through the store APIs, the timed path, resets, `mincore` / `posix_fadvise` page-cache handling on own files, correctness check, micro-benchmarks, environment and GPU monitor | `kvlab/perf_harness.py` |
| Scripts: `m2_measure.py` (smoke, calibration, evaluation); `m2_fit_predict.py` (freeze); `m2_score.py`; `m2_spike_probe.py` (diagnostic) | `scripts/` |
| Tests: `tests/test_perf.py` (18); engine `prefill_request` test; `discard` test | `tests/` |

Ordering facts from the existing engine, applied to the metric:
- `run()` stores (D2H admission) **before** the first token is selected, so
  admission is inside the timed region. It costs 4.5 ms per admitted chunk
  (S = 256 cases).
- In the prepared state no spill occurs. The cost of spill-on-admission was
  not measured (§9).

## 3. Commands and exit codes (in order)
| # | Command | Exit | Result |
|---|---|---|---|
| 1 | snapshot + manifest + verification | 0 | 128 files verified |
| 2 | `pytest -q tests/test_engine_cpu.py tests/test_host_store.py tests/test_tiered_cpu.py` (after the engine split) | 0 | 24 passed |
| 3 | `m2_measure --split smoke --run-dir results/m2/smoke_20261103T125831 --warmup 2 --reps 10 --instr-warmup 1 --instr-reps 10` | **1** | harness bug: the byte check compared a CUDA slice with a CPU tensor (`ValueError`); fixed with `.cpu()`. No samples file was written; the log is kept |
| 4 | same, `smoke_20261103T125910` | 0 | 189 samples, 0 excluded, strict correctness; full path ~82 ms (investigated, §6) |
| 5 | pre-protocol probes (`results/m2/pre_protocol_probes/`) | 0 | identified the GPU clock regimes; exploratory, not evidence |
| 6 | same smoke, `smoke_20261103T130506` | 0 | the 82 ms full-path regime reproduced |
| 7 | `pytest -q` (CPU) | 0 | **345 passed** (`results/m2/pytest_cpu_20261103T131401.log`) |
| 8 | `pytest -m "gpu and m1" -v` (the engine refactor touches the M1 path) | 0 | **12 passed** (`results/m2/pytest_gpu_m1_after_engine_refactor_20261103T131436.log`) |
| 9 | smoke with the GPU monitor, `smoke_20261103T131505` | 0 | monitor CSV written; 0 excluded |
| 10 | `m2_measure --split calibration --run-dir results/m2/run_20261103T131543` | 0 | 1015 samples (840 valid, 175 warmup, **0 excluded**), 1.9 min, strict correctness |
| 11 | `m2_fit_predict --run-dir …` | 0 | model `0774b1a2…`, predictions `1dfb47fa…` written 13:18, **before any evaluation sample** |
| 12 | protocol Amendment A, then `pytest -q tests/test_perf.py` | 0 | 18 passed |
| 13 | `m2_measure --split evaluation --run-dir … --reps 40 --budget-min 35` | 0 | started 13:19:05; 1983 samples (1740 valid, **0 excluded**), 3.7 min, strict correctness, no failures or NOT RUN |
| 14 | `m2_score --run-dir …` | 0 | all freeze-chain checks true; `results.json`, `results.md` |
| 15 | `m2_spike_probe --out …/spike_probe.json` | 0 | slow mode localized (§6) |
| 16 | CPU analyses of saved samples (`logs/slow_mode_time_clustering.txt`, `logs/disk_evicted_paired.txt`) | 0 | §6, §7 |
| 17 | final `pytest -q` | 0 | **345 passed** (`results/m2/pytest_cpu_final_20261103T132853.log`) |

Total GPU time was about 30 minutes, probes included. Each run used one GPU
process. Not rerun, because this change does not touch them: the M0 strict
suite (still FAIL on `nat_long`, D10) and the M0.1 suite.

## 4. Correctness of the measured paths (separate from timing)
- **Every timed sample** was checked for its tier, restore length and
  admissions (`perf.sample_problems`); 0 of 2,580 measured samples were
  excluded. Disk-path samples were 100% page-cache resident before t0;
  `disk_evicted` samples were 0% resident.
- **Once per case** (15 cases): restored K/V bytes equal the source bytes
  (host and disk). First-token logits are **bitwise equal** for
  host == disk == the resident split reference. **Strict pass 15/15.**
- **Report-only (D10):** full vs restored first token was equal in 15 of 15
  cases, with a maximum absolute logit difference of 0.39. Nothing is gated
  on this, and D10 is unchanged.

## 5. Cost model (fitted on calibration only; `run_20261103T131543/model.json`)
| Component | Fitted (ms; kt = 1000 tokens) | Notes |
|---|---|---|
| ingest | 0.075 + 0.133·N_kt | ids to device + SHA-256 chain keys |
| restore_host | 3.62 per chunk (const 0) | ≈ 2.0 GB/s effective per 7 MB chunk |
| restore_disk | 5.75 per chunk (const 0) | ≈ 1.3 GB/s effective (warm page cache) |
| prefill_suffix | 17.19 + 15.53·S_kt + 8.45·S_kt(L_kt + S_kt/2) | ~17 ms floor: launch-bound even for S = 1 |
| prefill_full | 13.39 + 40.27·N_kt + 8.24·N_kt²/2 | |
| store | 0.042 + 4.51 per admitted chunk | D2H into pinned host |
| select | 0.105 | argmax + token to host |
| offset | full −0.54, host −0.42, disk +0.03 | e2e − instrumented total (sync overhead) |

The micro-benchmarks (calibration run; one chunk; warm) are diagnostic only:

| Measurement | Time | Bandwidth |
|---|---|---|
| single contiguous 7 MB H2D from the registered slab | 0.61 ms | ≈ 12 GB/s, raw reference |
| the 56 per-layer H2D copies the restore actually makes | 2.31 ms | 3.2 GB/s |
| validated file read into staging | 1.60 ms | |
| of which raw page-cache read | 0.22 ms | |
| of which zlib CRC32 | 1.31 ms | |
| `DynamicCache` assembly (D2D copy, D4) | 0.75 ms | |

So the per-chunk restore cost is dominated by per-copy overhead, CRC and
assembly, not by PCIe bandwidth.

## 6. Results on held-out cases (`run_20261103T131543/results.md`, full tables)
**Latency prediction error** (predicted − measured median; 9 cases per path):

| Path | Median abs error | Max abs error | In-sample median (calibration) |
|---|---|---|---|
| full | 3.5% | 32.8% (L=256, S=64, low-clock regime) | 5.4% |
| host | 14.3% | 67.0% (L=512, S=1) | 1.1% |
| disk | 5.6% | 45.1% (L=4096, S=64) | 0.8% |

Representative held-out medians (ms, n = 40, IQR in brackets):

| Case | full | host | disk |
|---|---|---|---|
| L=2048, S=1 | 123.45 [123.09–123.87] | 46.43 [45.53–76.77] | 61.54 [60.20–146.99] |
| L=2048, S=256 | 134.63 [134.23–138.76] | 59.88 [58.32–99.53] | 72.85 [72.22–103.19] |
| L=1024, S=64 | 64.15 [63.82–65.46] | 37.36 [36.73–54.84] | 44.23 [43.26–46.06] |
| L=512, S=1 | 39.24 [38.35–40.47] | 73.81 [25.95–84.99] | 45.36 [29.51–89.32] |
| L=4096, S=64 | 249.87 [249.46–250.25] | 201.77 [85.28–209.24] | 206.34 [112.28–236.66] |

**Decisions** (restore from the tier holding the prefix vs full recompute):

| Set | Decisions | Scorable | Correct | Wrong | Ties | Penalty |
|---|---|---|---|---|---|---|
| held-out | 18 | 11 | 11 | 0 | 7 (all at L ≤ 512) | 0 ms |
| in-sample | 12 | 11 | 11 | 0 | 1 | 0 ms |

- The model predicts restore everywhere, so these results show that the
  model never chose a clearly slower path in this grid.
- They do **not** show that it discriminates near a crossing: no measured
  case had full recompute as the reliable winner.

**Break-even:**
- **Predicted:** none in L = 256…4096 for any S or tier. The smallest margin
  is the disk tier at L = 256, S = 1: restore is predicted 0.5 ms faster,
  which is inside the model error. L < 256 is impossible (chunk
  granularity).
- **Measured:** no reliable crossing. Restore wins at every L where the
  paired CI excludes 0 (all of L ≥ 1024, and some L = 256/512 cases).
  Several L ≤ 512 cases are ties. **No break-even length is claimed.**

**The slow mode, and why restore predictions fail in some cases:**
- **Shape and size.** Restore-path latencies are bimodal: a fast mode near
  the model, and a slow mode about +50 ms per sample (up to +120 ms at
  L = 4096).
- **Time clustering.** Slow samples come in regimes lasting tens of seconds.
  P(slow | previous slow) was 84% against a 40% base rate in evaluation, and
  63% against 20% in calibration (`logs/slow_mode_time_clustering.txt`).
- **Calibration vs evaluation.** The evaluation run had twice the slow share
  of the calibration run. The model, fitted on fast-mode medians, therefore
  underpredicts wherever the evaluation median fell in the slow mode (host
  L=512 S=1; both tiers at L=4096 S=64).
- **Which work is affected.** The full path at L ≥ 512 is barely affected
  (IQR ≤ 6 ms). The affected work is CPU- or launch-bound (896 small
  synchronized copies for 16 chunks; forward passes with small kernels).
- **Localization** (`spike_probe.json`): a single 7 MB copy and the validated
  disk read never spike. The 56 small copies and `DynamicCache` assembly
  jitter by a few ms. The ~50 ms events appear in the model forward on a
  pre-filled cache (min 16.4 ms, max 66.8 ms) and in end-to-end restore
  samples.
- **Root cause not established.** Candidates are WSL2 GPU paravirtualization
  or Windows GPU scheduling, and CPU/GPU power sharing on the laptop.
  Separately, the SM clock moved between about 1335 and 2385 MHz across the
  runs (`*_gpu_monitor.csv`; event reasons include SW power cap), which
  slowed the launch-bound full path at L = 256 in evaluation.
- **Means.** Restore-path means are 10–50 ms above their medians. This model
  predicts typical (median, fast-mode) latency, not mean or tail latency.

## 7. Disk measurements
- **`disk` (modeled):** application-level disk-cache hits with **warm Linux
  page cache**, verified with `mincore` at 100% resident before every sample.
  These are file reads, not device reads.
- **`disk_evicted` (diagnostic, S = 1)** (`logs/disk_evicted_paired.txt`):
  - Method: the experiment's own files were fsynced at preparation, then
    `posix_fadvise(DONTNEED)` was applied, giving 0% resident before t0
    (verified).
  - It is slower than the warm disk path by +3.4 ms at L=256, +13.8 ms at
    L=1024, and +23 ms at L=2048 (paired medians).
  - Against full recompute: a **tie at L = 256 and L = 512**; restore wins at
    L = 1024 (−8.5 ms), 2048 (−29.9 ms) and 4096 (−88.6 ms).
  - The WSL2 virtual disk may still be served from the Windows host cache, so
    **this is not a cold-NVMe measurement**.
- **Cold device reads: BLOCKED.** Establishing them would need host-side
  cache control that requires privileges or settings outside this project.
  No sudo was used, and no global cache drop was performed.

## 8. Resource accounting (calibration run; evaluation identical)
- **Store A host:**
  - `slab_allocated_bytes` = `cuda_host_registered_bytes` = 132,120,576
    (17 slots + staging);
  - `resident_logical_peak_bytes` = 124,780,544 (16 prefix chunks + 1
    admitted).
- **Store B:**
  - host slab 14,680,064 (1 slot + staging);
  - disk budget 124,816,057, peak used 117,473,936 (16 files), no unlink
    failures.
- **Source chunk store:** pageable slab 124,780,544 (`pin=False`).
- These are allocation/registration and logical-occupancy figures, **not
  RSS or measured locked memory**.
- All experiment disk files lived under `.kvcache/m2/<run>/` and were
  removed at the end.

## 9. PASS / FAIL / BLOCKED / NOT RUN
| Item | Status |
|---|---|
| Baseline snapshot + manifest + verification | PASS |
| Protocol written before calibration and evaluation; Amendment A before evaluation | done |
| Harness smoke (3 good runs; 1 failed run, its bug fixed) | PASS |
| Correctness of measured paths (tier/L/admissions per sample; strict bytes + bitwise logits per case) | PASS (0 exclusions; 15/15 strict) |
| Calibration measurement (6 cases) | PASS |
| Held-out evaluation measurement (9 cases) | PASS |
| Freeze chain (model digest → predictions digest → evaluation started after them, recorded their digest) | PASS |
| Latency prediction | full: good (3.5% median). host/disk: **FAIL in slow-mode cases** (max 67%); medians 14.3% / 5.6% |
| Winner prediction | 11/11 scorable correct, 7 ties; no wrong choice; no crossing to discriminate |
| Break-even length | **not established** (none predicted; none reliably measured) |
| Cold-NVMe latency | **BLOCKED** (no privileged or host cache control); Linux-page-cache-evicted diagnostic reported instead |
| Spill-on-admission cost; cost of a host-full state | NOT RUN (the prepared states always have a free slot) |
| Partial-restore length choice (L* over candidate boundaries) | NOT RUN (offline restore-vs-full only, by instruction) |
| p99 / tail latency | NOT RUN (n = 20/40; not claimed) |
| Runtime policy change | not in scope (no change) |
| Original M0 strict suite | not rerun; known FAIL on `nat_long` unchanged (D10) |

## 10. Change inventory against `backups/pre-m2-20261103T125219.tar.gz`
Machine-readable: `results/m2/change_inventory_vs_pre_m2.txt` (computed last).
- **Modified:**
  - `kvlab/always_restore.py`: `prefill_request` split; `run()` behavior
    unchanged.
  - `kvlab/host_store.py`: `discard`.
  - `tests/test_engine_cpu.py`, `tests/test_host_store.py`: +1 test each.
  - docs: `README.md`, `docs/design.md`, `docs/decisions.md`,
    `docs/correctness-matrix.md`.
- **Added:**
  - code: `kvlab/perf.py`, `kvlab/perf_harness.py`,
    `scripts/m2_measure.py`, `scripts/m2_fit_predict.py`,
    `scripts/m2_score.py`, `scripts/m2_spike_probe.py`, `tests/test_perf.py`;
  - docs: `docs/M2_PROTOCOL.md`, `docs/M2_REVIEW.md`,
    `docs/M2_WALKTHROUGH.md`;
  - everything under `results/m2/`.
- **Deleted:** none.
- Earlier evidence (`results/m0*`, `results/m1*`) is unchanged.

## 11. Limitations
- **Platform.** This is a single laptop GPU under WSL2. Its clock and power
  state and the restore-path slow mode change latencies by 25% to 3× between
  runs. Absolute numbers will not transfer to other machines, and should be
  re-measured even on this one.
- **What the model predicts.** It predicts **fast-mode median** latency and
  has no term for the slow mode or the clock regime. Its winner decisions
  held here because the margins in favor of restore were large (L ≥ 1024) or
  tied (L ≤ 512).
- **Scope.** One model, batch 1, synchronous only, S ≤ 256 and L ≤ 4096.
  There is no queueing or tokenization, so this is not service-level TTFT.
  Admission into a free slot is included; spill is not.
- **Restore implementation.** The restore makes 56 small synchronized copies
  per chunk, and assembly is an extra D2D copy (D4). These are the measured
  costs of the current code, not of an optimized pipeline.
- **Repetitions.** Calibration used 20 repetitions and evaluation 40
  (Amendment A). Medians are reported with bootstrap CIs; no tail claims are
  made.
- **Exploratory probes.** The pre-protocol probes are exploratory. Their
  console output was summarized by hand in `pre_protocol_probes/NOTES.md`.

## 12. Next acceptance commands
```bash
.venv/bin/python -m pytest -q                                                    # expect 345 passed
.venv/bin/python scripts/m2_score.py --run-dir <copy of results/m2/run_20261103T131543 without results.*>   # re-score the saved data
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m2_measure.py --split smoke --run-dir results/m2/smoke_<ts> --warmup 2 --reps 10 --instr-warmup 1 --instr-reps 10
```
To run a full new run, use a new run directory:
1. `m2_measure --split calibration`
2. `m2_fit_predict`
3. `m2_measure --split evaluation --reps 40`
4. `m2_score`
