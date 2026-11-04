# M3 Review — batched KV restoration (restore mechanics only)

Date: 2026-11-03. Baseline: `main` at `362ee40aeaaae204b2b1dc88398b46bc5826f410` (the
reorganized 51-commit M0–M2.1 history; tree `b6c07930…`). Nothing was staged, committed,
pushed or tagged during M3.

**Status: M3 implemented and measured.** A new, opt-in restore backend (`batched`) replaces
the M1 per-layer copies. Each chunk now makes one blocking contiguous H2D into a bounded
chunk-major device staging buffer, and each group of ≤ 4 chunks gets one permuted device
copy. Everything else is unchanged, and the default backend is still `baseline`.

- **Correctness:** every strict check passed for both backends. KV bytes, every step's
  logits and the tokens are bitwise equal to the same-provenance resident replay, and
  baseline and batched are bitwise equal to each other under identical state.
- **Calls (measured):** a 16-chunk restore went from 896 memcpy / 896 syncs / 952 kernel
  launches to 16 / 16 / 60.
- **Latency:** in this sample, median end-to-end first-token latency fell by 1.4–2.3× for 4–16
  chunk prefixes, and by 2 ms for a 1-chunk prefix, with no regression anywhere.
- **Adoption:** the pre-registered rule is **not fully met**. One required case, (1024, 1), is
  inconclusive under the duration rule R3, although its direction is consistent at every block
  length and in every session. The default therefore stays `baseline` (§8).

> **Correction note (VAL-006, added 2026-11-03 after M3; analysis/acceptance only).**
> - **The defect:** `scripts/m3_analyze.py` inferred the experiment from the files it found.
>   An empty run directory exited 0 with `supports_adoption: true`. Removing every
>   (1024, 1) sample from copies of the five sessions also turned the verdict to true,
>   because the missing case dropped out of the decision rule. See `docs/BUGLOG.md` VAL-006.
> - **The fix:** the analysis now first validates the run against an explicit spec of this
>   frozen evaluation (`kvlab/m3_validate.py`), covering sessions, cases, blocks,
>   repetitions, every sample identity, and the correctness, A/B and memory records. It
>   recomputes acceptance from the detailed checks, and decides over the spec's case list.
>   Invalid evidence exits 3 and never reports adoption. The statistics, R1–R3 and the
>   measurement code are unchanged.
> - **Follow-up (same day): warmup evidence.** The first fix did not check the retained
>   warmups. Removing all 1,080 warmup rows, or substituting one, still validated and exited
>   0. Block warmups (900) and process warmups (180) are now checked against the frozen plan:
>   - block warmups by identity;
>   - process warmups by multiplicity per (case, combination), on their recorded fields.
>
>   Missing, duplicate, substituted or misflagged warmups are now invalid. Warmup timings
>   remain excluded from every statistic.
> - **This review's results stand.** The preserved evidence validates: 5,400/5,400 samples,
>   900/900 block and 180/180 process warmups, and 50/50 strict and 50/50 A/B checks.
>   Reanalysed into `results/m3/reanalysis_val006_20261103T200427/analysis/` (and, after the
>   follow-up, `results/m3/reanalysis_val006_warmups_20261103T202825/analysis/`), it reproduces every
>   historical value exactly, and the verdict is still **supports adoption: false**, from the
>   same four (1024, 1) entries. The historical outputs in
>   `results/m3/eval_20261103T173632/` are byte-identical.
> - **Superseded statements:**
>   - §4's "BUGLOG has no new entry" holds for product code only.
>   - §1's "the executed code matches the final tree" holds for the measurement path.
>     `scripts/m3_analyze.py` has changed since the sessions recorded its hash, and
>     `kvlab/m3_validate.py` was added afterwards.
>   - The CPU suite was 436 passed at the first fix (§4: 408). After the warmup follow-up
>     only the focused M3 validation and statistics tests were run: 56 passed.

## 1. Baseline preservation and identity
- **Recorded before any edit:** `results/m3/pre_m3_git_state_20261103T171930.txt`.
  - HEAD `362ee40a…`; 51 commits; clean tree, 0 staged.
  - Backup branch `backup/one-file-commits-20261103T170624` → `8657d137…`.
  - History bundle sha256 `c1135858…`. Both are preserved.
- **Pre-M3 snapshot:** `backups/pre-m3-20261103T171930.tar.gz`, sha256
  `5c46ad464a2176a193d814c532e892ef50bab4c9ef88e6b11edb1a63d68fb78e`.
  - Contents: the 271 tracked files. Environments, weights, `.kvcache`, backups and archives
    are excluded.
  - Verified by extraction against `results/m3/pre_m3_manifest_20261103T171930.sha256`
    (271/271).
- The pre-M2, pre-M2.1 snapshots and the M2.1 review archive were re-verified (`sha256sum -c`
  OK).
- **Historical evidence untouched:** all 197 files under `results/m0*`–`results/m2_1` match
  the pre-M3 manifest; see the change inventory.
- **Source identity per experiment:** every session file records the git HEAD, `git status`,
  the sha256 of `git diff HEAD`, and the sha256 of every `kvlab/`, `scripts/` and `tests/`
  file, plus wheel versions, weight blobs and backend flags.
  - All 6 evaluation/profile sessions executed byte-identical sources (57 files compared).
  - Their `git diff` hash differs only between s2 and s3, because `docs/design.md` and
    `docs/decisions.md` were edited between those sessions.
  - The executed code matches the final tree.

## 2. What changed (code)
| File | Change |
|---|---|
| `kvlab/tiered_cache.py` | Restore backend option (`restore_backend`, `restore_staging_chunks`, validated on every restore). The M1 copy loop moved verbatim into `_BaselineDest`; new `_BatchedDest`. The lookup, lease, staging, validation, counter and miss code around them is unchanged and shared. Adds `last_restore` (diagnostic copy/byte counts) and `restore_config()`. |
| `kvlab/m3_bench.py` (new) | Benchmark harness on top of the unchanged M2 `Bench`: per-sample backend switch on the same stores, memory and copy records, untimed strict, A/B and memory checks, source identity. |
| `kvlab/m3_stats.py` (new) | Stratified circular block bootstrap, R1–R3 classification, regime durations (reuses `perf_stats` unchanged). |
| `scripts/m3_measure.py`, `scripts/m3_analyze.py`, `scripts/m3_mutation_check.py` (new) | Session runner (timing and `--profile`), analysis, mutation check. |
| `scripts/m1_run.py` | `--restore-backend` (default `baseline`), recorded in the report. |
| `tests/gpu/test_m1_tiered.py` | The `harness` fixture is parametrized over both backends: every M1 GPU scenario also runs batched. |
| `tests/test_restore_backends_cpu.py`, `tests/gpu/test_m3_restore.py`, `tests/test_m3_stats.py` (new) | See §4. |
| `pyproject.toml` | `m3` marker. |

`kvlab/perf.py`, `perf_harness.py`, `perf_stats.py`, `diag_harness.py`, the chunk format,
identity, host/disk stores, engine, model and tolerance code are byte-identical to pre-M3.

## 3. Design (details: `docs/M3_DESIGN.md`, decision D30)
- **Layouts:**
  - host slot: `[layer][K,V][1, 2, 256, 128]` bf16, 7,340,032 B contiguous;
  - baseline destination: 56 per-layer tensors `[1, 2, n·256, 128]`, so a chunk lands as
    strided slices;
  - batched: chunk-major staging `X[min(n,4), layer, K/V, 1, head, t, d]` and one layer-major
    destination `Y[layer, K/V, 1, head, n·256, d]`.

  One `copy_` of `X[:g]` permuted to `[layer, K/V, 1, head, g, t, d]`, into `Y`'s slice
  unflattened to `(g, 256)`, places a whole group.
- **Unchanged for both backends:** the cache is built with the public `DynamicCache`
  constructor (D4). It copies exactly [0, L) into fresh tensors (56 kernels).
- **Synchronization:** no copy is `non_blocking`. PyTorch implements a blocking H2D as
  `cudaMemcpyAsync` + `cudaStreamSynchronize` (measured). So:
  - a host lease is released only after its chunk's transfer finished;
  - the single staging slot is reused only after the previous disk chunk's transfer finished.

  All device work runs on the current stream. `X` is rewritten only by memcpys ordered after
  the placement that reads it. `X` and `Y` are freed at return under the caching allocator's
  same-stream rule. `non_blocking=True` alone would not be safe here: PyTorch records events
  only for its own pinned allocator, not for our `cudaHostRegister`-ed slab.
- **Partial prefixes:** at the first invalid chunk, the verified part of the current group
  is placed and the cache is built from [0, L) only. The unwritten part of `Y` is never
  exposed.
- **Options evaluated:**
  - (a) one strided `copy_` per chunk;
  - (b) chosen;
  - (c) `cudaMemcpy2D` via ctypes;
  - (d) `non_blocking` with held leases;
  - (e) skipping the constructor copy.

  See the design note and D30 for why a, c, d and e were not taken.

## 4. Correctness gates (all actually run; logs in `results/m3/logs/`)
| Gate | Command | Result |
|---|---|---|
| Full CPU suite | `.venv/bin/python -m pytest -q` | **PASS** 408 passed, 57 deselected (`pytest_cpu_20261103T175215.log`) |
| GPU M1 suite, both backends | `pytest -m "gpu and m1" -v` | **PASS** 23/23: 11 scenarios × {baseline, batched} + namespace (`pytest_gpu_m1_20261103T175241.log`) |
| GPU M3 checks | `pytest -m "gpu and m3" -v` | **PASS** 17/17 (`pytest_gpu_m3_20261103T175241.log`) |
| GPU M0.1 | `pytest -m "gpu and m0_1" -v` | **PASS** 6/6 (`pytest_gpu_m0_1_20261103T175241.log`) |
| GPU M0 strict (original criteria) | `pytest -m "gpu and m0_strict" -v` | **FAIL 1/11 (`nat_long`), as recorded since M0**. Unchanged known result (D10); not a regression (`pytest_gpu_m0_strict_20261103T175241.log`) |
| M1 evidence runner, baseline | `scripts/m1_run.py --out results/m3/m1_run_baseline_20261103T175241.json` | **PASS** exit 0; 19/19 strict; `--validate` exit 0 |
| M1 evidence runner, batched | `… --restore-backend batched` | **PASS** exit 0; 19/19 strict; `--validate` exit 0. All 19 records (L, sources, miss reasons, stored chunks, tiers after, forward calls, strict fields, report-only fields) and all counters are identical to the baseline run (`logs/m1_run_backend_comparison_*.txt`) |
| Mutation check | `scripts/m3_mutation_check.py <scratch> --gpu` | **10/10 KILLED** (`results/m3/mutation_check_20261103T173143.txt`; the earlier run `…173035.txt` had 9/10, see below) |
| In-session strict + A/B checks | per session, per case, untimed | **PASS** 25/25 strict (both backends) and 25/25 A/B |

**What the new tests cover:**
- `tests/test_restore_backends_cpu.py`, 24 CPU tests. Each runs twin stores, one per backend,
  with identical history, and compares the result, KV bytes, counters, tiers, LRU order, pins,
  staging and budgets. Scenarios:
  - one chunk on host and on disk; 5 host chunks for G ∈ {1, 2, 3, 5, 8};
  - disk-only with 5 distinct payloads through the one staging slot;
  - mixed tiers;
  - a corrupt chunk at index 0 (zero-length valid prefix), 2 and 4;
  - a missing middle file and an absent middle chunk;
  - partial plans of 1–4 chunks;
  - repeated requests with eviction, where the earlier cache is unchanged after later
    requests and after the whole host slab is overwritten;
  - returned tensors own exactly their bytes and share no storage;
  - copy failure at a disk chunk and at a host chunk: leases and staging are released, the
    error propagates, only completed copies are counted as hits, and the store stays usable;
  - backend and group-size validation;
  - engine-level sequences on a tiny Qwen2: bitwise identical across backends and equal to
    the resident replay.
- `tests/gpu/test_m3_restore.py`, 17 GPU tests:
  - A/B on one store for n ∈ {1, 3, 5, 6} × {host, disk, mixed}: KV bytes, logits, 8 greedy
    steps and tokens bitwise equal, then the resident replay;
  - staging reuse with distinct payloads;
  - a sync-count guard: 6 chunks → 336 synchronizing copies (baseline) vs 6 (batched);
  - ownership under later requests and slab overwrite;
  - a bounded peak device memory;
  - a copy failure on GPU.
- The mutation check kills: wrong group offset, reversed chunk order in a group, an unplaced
  last group, an exposed unverified destination, unbounded scratch, deferred placement, a
  silently ignored backend, a hit counted before the copy, a `non_blocking` chunk copy (GPU),
  and a placement error (GPU). The hit-counter mutant first **survived**: no existing or new
  test checked hit counts after a failed copy. The failure test was strengthened, and the
  mutant was then killed.
- **Test-development issues fixed before the final runs:** these were not product defects,
  so there is no BUGLOG entry.
  - The sync-count guard initially also counted torch's one-time "sync debug mode is a
    prototype" notice.
  - One GPU A/B case (1-chunk disk) was skipped by a test setup error.
  - One synthetic autocorrelation fixture in `test_m3_stats.py` was not autocorrelated
    enough.

**No product defect was found in M3, so BUGLOG has no new entry.**
(*Correction, VAL-006:* a later review found an analysis-script validation defect, now BUGLOG
VAL-006; see the correction note at the top.)

## 5. Experiment (protocol `docs/M3_PROTOCOL.md`, frozen before any evaluation sample: `results/m3/protocol_freeze_20261103T173632.sha256`)
- **Sessions:** 5 fresh-process sessions s1–s5 (seeds 31001–31005), run sequentially with
  one GPU process at a time, 2.2–2.4 min each.
- **Cases:** (256, 32), (1024, 1), (1536, 128), (3072, 256) and (4096, 64). Three of them
  are fresh combinations. The tuning cases (512, 1) and (2048, 64) were used only for the
  smoke run.
- **Blocks:** 2 blocks per case per session, each 2 warmups + 12 repetitions. Each repetition
  ran 9 samples in random order:
  - `full`;
  - host and disk × {baseline, batched}, plain;
  - the same four in the synchronized-phase mode.
- **Admission:** (3072, 256) admits one chunk, with identical admission state for both
  backends (reset after every sample).
- **Totals:** 5,400 measured samples, 0 excluded, none filtered. The SM clock median is
  2370 MHz in every session (monitor CSVs).
- **Raw data:**
  - `results/m3/eval_20261103T173632/session_s*.json` (every sample, including warmups,
    with session/block/rep/order and `t_wall`);
  - `samples_chronological.csv`;
  - `m3_analysis.{json,md}`.
- **Call counts:** a separate profiler session (`session_prof.json`, 5 calls per unit).
- **GPU time:** about 20 GPU-minutes for all of M3. That covers the design probes, dev tests,
  mutation runs, smoke, 5 sessions, profile, final suites and the M1 runner ×2, all within
  the 45-minute bound.

## 6. Results
d = batched − baseline, paired within a repetition (ms). Intervals: stratified block
bootstrap, b = 4; the other block lengths are in `m3_analysis.md`. Verdicts apply rules
R1–R3. These are statements about this sample on this machine, not general guarantees.

### 6.1 End-to-end first-token latency (the M2 timed region; plain samples)
| case (chunks) | tier | baseline median / mean | batched median / mean | median d [95% CI] | mean d [95% CI] | verdict (median, mean) |
|---|---|---|---|---|---|---|
| 256:32 (1) | host | 23.9 / 45.6 | 22.5 / 39.1 | −1.99 [−2.29, −1.71] | −6.4 [−8.9, −4.1] | faster, faster |
| 256:32 (1) | disk | 26.5 / 46.8 | 24.6 / 42.8 | −2.08 [−2.39, −1.83] | −4.0 [−6.3, −1.8] | faster, faster |
| 1024:1 (4) | host | 33.3 / 55.8 | 21.3 / 37.7 | −12.3 [−12.9, −11.9] | −18.1 [−20.9, −15.4] | **inconclusive (R3)** |
| 1024:1 (4) | disk | 41.9 / 64.9 | 29.1 / 46.3 | −13.8 [−14.5, −13.2] | −18.6 [−21.4, −16.0] | **inconclusive (R3)** |
| 1536:128 (6) | host | 43.7 / 71.0 | 25.9 / 42.3 | −17.9 [−18.8, −17.3] | −28.7 [−32.8, −24.3] | faster, faster |
| 1536:128 (6) | disk | 55.3 / 80.5 | 37.0 / 52.1 | −19.1 [−19.8, −18.6] | −28.4 [−31.9, −25.1] | faster, faster |
| 3072:256 (12, admits 1) | host | 78.7 / 114.9 | 41.7 / 55.1 | −37.0 [−38.3, −36.2] | −59.8 [−65.4, −54.1] | faster, faster |
| 3072:256 (12, admits 1) | disk | 101.4 / 136.0 | 62.7 / 75.7 | −39.5 [−41.6, −38.1] | −60.3 [−65.3, −55.1] | faster, faster |
| 4096:64 (16) | host | 86.2 / 135.4 | 37.6 / 54.6 | −48.5 [−80.4, −47.8] | −80.8 [−90.0, −71.7] | faster, faster |
| 4096:64 (16) | disk | 117.7 / 164.1 | 65.0 / 81.6 | −52.8 [−59.9, −50.9] | −82.5 [−90.4, −74.6] | faster, faster |

- **Why (1024, 1) is inconclusive:** R1 (all block lengths) and R2 (all 5 sessions negative)
  hold. R3 fails: the case's summed measured span is 73 s, less than 10× its longest observed
  slow regime of 8.2 s. The R3 threshold was fixed in advance, so this verdict is reported as
  is, not overridden.
- **Session variability is large** (`m3_analysis.md`, per-session table). For example, the
  per-session median d at (4096, 64) host is −124, −111, −49, −48 and −46 ms.
  - s1 and s2 had higher slow shares there for both backends: baseline 62% / 54%, batched
    58% / 46%, vs 25–38% in s3–s5.
  - In every case and tier, |mean d| exceeds |median d|. The batched advantage is larger
    in the slow tail, as expected if per-call cost rises there.
  - All sessions agree in sign in every case.
- **Slow shares stay similar per backend:** 33–42% for both (slow = e2e > 1.3× its own
  minimum). Means remain well above medians for both backends. The slow mode did not
  disappear (§7).

### 6.2 Speedups (b = 4 CIs). Three different definitions, not interchangeable
| case | tier | median(baseline) / median(batched) | mean(baseline) / mean(batched) | median over reps of baseline/batched |
|---|---|---|---|---|
| 256:32 | host | 1.065 [1.050, 1.578] | 1.164 [1.103, 1.232] | 1.070 [1.063, 1.076] |
| 256:32 | disk | 1.078 [1.057, 1.166] | 1.092 [1.042, 1.147] | 1.065 [1.060, 1.075] |
| 1024:1 | host | 1.563 [1.537, 1.593] | 1.479 [1.414, 1.547] | 1.522 [1.509, 1.539] |
| 1024:1 | disk | 1.441 [1.419, 1.470] | 1.402 [1.348, 1.454] | 1.443 [1.437, 1.452] |
| 1536:128 | host | 1.691 [1.651, 1.728] | 1.678 [1.569, 1.791] | 1.662 [1.647, 1.675] |
| 1536:128 | disk | 1.494 [1.485, 1.530] | 1.544 [1.474, 1.620] | 1.519 [1.508, 1.532] |
| 3072:256 | host | 1.886 [1.858, 1.912] | 2.086 [1.997, 2.174] | 1.898 [1.873, 1.918] |
| 3072:256 | disk | 1.618 [1.603, 1.644] | 1.795 [1.731, 1.856] | 1.640 [1.610, 1.666] |
| 4096:64 | host | 2.294 [1.991, 3.469] | 2.480 [2.345, 2.613] | 2.316 [2.284, 2.446] |
| 4096:64 | disk | 1.811 [1.781, 1.986] | 2.012 [1.935, 2.091] | 1.828 [1.791, 1.889] |

The wide marginal-median intervals at (256, 32) host and (4096, 64) host come from the
bimodal (fast/slow-regime) baseline distribution.

### 6.3 Restore phase (separate synchronized samples; diagnostic)
| case | host baseline → batched median (ms) | ratio of medians | disk baseline → batched median (ms) | ratio of medians |
|---|---|---|---|---|
| 256:32 | 3.48 → 1.80 | 1.93 | 5.67 → 3.87 | 1.46 |
| 1024:1 | 15.90 → 3.61 | 4.41 | 24.28 → 11.13 | 2.18 |
| 1536:128 | 22.84 → 5.05 | 4.52 | 35.23 → 15.67 | 2.25 |
| 3072:256 | 45.03 → 9.20 | 4.90 | 67.52 → 29.76 | 2.27 |
| 4096:64 | 62.63 → 12.10 | 5.18 | 90.82 → 39.38 | 2.31 |

- The paired median and mean differences are established negative for every case and tier
  except (1024, 1), which fails R3 as above.
- **The disk-tier remainder is mostly the validated file read**, not the transfers:
  - batched disk minus batched host is ≈ 27 ms at 16 chunks, i.e. ≈ 1.7 ms per chunk for
    `readinto` + CRC32 from the page cache;
  - it is identical for both backends.

### 6.4 Calls and transfers
**Measured** with torch.profiler, per `restore()` call. The profiler session averages 5 calls
per unit and matches the §4 estimates of `docs/M3_DESIGN.md` exactly.

| restored chunks | backend | cudaMemcpyAsync | cudaStreamSynchronize | cudaLaunchKernel | device HtoD / DtoD copies |
|---|---|---|---|---|---|
| 1 | baseline | 56 | 56 | 56 | 56 / 0 |
| 1 | batched | 2 | 1 | 56 | 1 / 1 |
| 6 | baseline | 336 | 336 | 392 | 336 / 0 |
| 6 | batched | 6 | 6 | 58 | 6 / 0 |
| 16 | baseline | 896 | 896 | 952 | 896 / 0 |
| 16 | batched | 16 | 16 | 60 | 16 / 0 |

- **Whole request (the timed region), cudaLaunchKernel:**
  - 1 chunk: 1,212 → 1,212;
  - 6 chunks: 1,744 → 1,410;
  - 16 chunks: 2,192 → 1,300.

  The memcpy calls per request go from 58 / 338 / 898 to 4 / 8 / 18.
- **H2D bytes are identical** for both backends (n × 7,340,032 B). Python-level copy calls
  per restore are 56·n vs n, with ⌈n/4⌉ placement copies.
- The 56 launches still left in a batched restore are the D4 constructor copy.

### 6.5 Device and host memory
- **Restore-only peak above the starting allocation** (untimed `restore()`, every session
  identical):
  - baseline: 2·n chunks (destination + cache);
  - batched: 2·n + min(n, 4) chunks, i.e. +7.3 MB at n = 1 and +29.4 MB for n ≥ 4 (the
    explicit staging `X`);
  - at 16 chunks: 234.9 MB vs 264.2 MB.

  The per-request peak increment (plain samples) shows the same difference. Peak reserved
  memory reached 4.34 GB for both backends.
- **After return**, both keep exactly the restored cache (n chunks), with one exception. At
  (3072, 256), `memory_allocated` showed 14 chunks kept instead of 12, for both backends. A
  synthetic probe (`results/m3/design_probe/probe_allocator_rounding_*.json`) shows the
  cause: for the 1.5 MiB per-layer tensors of L = 3072, the caching allocator's block bytes
  exceed the requested bytes, and requested bytes equal the payload exactly. This is
  allocator block rounding of the constructor's output, the same for both backends. It is not
  a view that keeps scratch alive. Tensor storages are exactly L tokens (CPU and GPU tests).
- **Host memory:** no new allocation. The existing slots and the one staging slot are the
  only H2D sources; `restore_config()["new_host_memory_bytes"] = 0`.

### 6.6 Context: restore vs prepared-state recompute (report only, b = 4; not part of the decision rule)
- With batched, the restore path's paired median is below `full` in every case and tier.
  - At (1024, 1) disk, the baseline mean d vs `full` was inconclusive: −0.7 [−5.7, 4.8] ms.
    With batched it is −19.3 [−22.8, −15.5].
- At (256, 32) the paired **mean** still favors recompute for both backends:
  - host: baseline +9.7 [5.8, 13.6], batched +3.3 [0.1, 6.6];
  - disk: +11.0 vs +7.0.

  This is the M2.1 pattern: slow-mode restore samples dominate the mean at 1 chunk.
- This changes nothing in the runtime. The M2 cost model was fitted to the baseline backend
  and is not refitted. Any restore/recompute policy would need a fresh evaluation with the
  batched backend.

## 7. Qualifications
- **M2.1 directions are conditional.** M2.1's directions (e.g. "recompute faster in mean at
  L = 512") are conditional on that recorded sample and analysis. They do not transfer to the
  batched backend, and they are not universal.
- **Slow mode: an association, not a cause.** The M2.1 slow mode is described as associated
  with increased host-side CUDA-call cost in this environment. M3 does not establish a driver,
  WSL, scheduler or power root cause.
  - Batched lowers the number of calls the restore path makes.
  - It does not remove the slow mode: slow shares are similar, and means still exceed medians.
  - The remaining calls are mostly the suffix forward.
- **Measured vs estimated.** Measured: the profiler counts in §6.4 and the copy and byte
  records. Estimates: the design note's table, now confirmed.
- **Unaffected by M3, still costly:**
  - the suffix forward: ≈ 1,150–1,350 kernel launches per request once the restore part is
    subtracted (measured request minus restore counts); it sets the floor for host restores;
  - admission (store): 56 blocking D2H copies per admitted chunk;
  - the constructor copy: 56 kernels and one prefix-sized D2D copy (option e, deferred);
  - ingest and chain keys;
  - the validated disk read and CRC: ≈ 1.7 ms per chunk from the page cache.
- **Not claimed:**
  - no p99 or tail claim;
  - warm disk only: cold-NVMe latency stays BLOCKED/unclaimed (WSL; D26);
  - no claim beyond this GPU, driver and software stack.

## 8. Decision (protocol §8, fixed before the data)
- (1) every correctness and A/B check passes: **yes**;
- (2) no established regression: **yes, none**;
- (3) median and mean established faster for every case with ≥ 4 chunks on both tiers:
  **no**. (1024, 1) host and disk are inconclusive under R3; (1536, 128), (3072, 256) and
  (4096, 64) are established.

**The evidence therefore does not yet meet the pre-registered adoption rule.**

- It does show large, direction-consistent reductions wherever the rule could be applied,
  and no regression.
- The default backend stays `baseline`. Switching is a one-argument change
  (`restore_backend="batched"`), left for your review.

## 9. Remaining limitations
- One machine (RTX 4080 Laptop, WSL2, driver 596.49), one model and one chunk size. Sessions
  were minutes apart, not days.
- R3 fails at (1024, 1) because the measured stretch is short relative to the regimes. The
  other cases pass R3 by margins of 1.4–3.8× (span vs 10× the longest regime).
- G = 4 was fixed during development and not swept. Its effect is bounded by ≤ n − ⌈n/4⌉
  placement launches and 29 MB of device memory.
- The batched backend's correctness is shown for this layout (bf16, 28 layers, 2 KV heads,
  head_dim 128) and for the toy CPU layout. Other namespaces would rerun the same tests.
- Full-prefill vs restored greedy divergence remains a report-only limitation (D10). In the
  in-session checks every first token matched; the max |Δlogit| was 0.20–0.39 and identical
  for both backends.

## 10. Recommended next step (not started)
Run **one pre-registered confirmation of the medium-prefix case** before switching the
default:
- (1024, 1) plus one fresh 4–6 chunk case;
- host and warm disk, baseline vs batched;
- the same harness, rules R1–R3 and decision rule, with only the measured stretch per case
  lengthened (for example 4 blocks × 12 reps per session, so that the span exceeds 10× the
  longest regime).

If it establishes the direction there, the batched backend meets the adoption rule, and the
default can be switched after review.

Grounds: this is the only unmet clause of the adoption rule. The failure is about duration
(R3), not direction: all block lengths and all 5 sessions agree.

## 11. Files and archive
- **Change inventory:** `results/m3/change_inventory_vs_pre_m3.txt`.
- **Post-M3 manifest:** `results/m3/post_m3_manifest.sha256`.
- **Review archive:** `backups/kvlab-review-m3.tar.gz`. Its sha256 is in
  `results/m3/review_archive_m3.sha256`, and its file list in
  `results/m3/review_archive_m3.filelist.txt`. Both are in the final summary.
