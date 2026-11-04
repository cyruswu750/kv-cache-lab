# M2 performance protocol (fixed 2026-11-03, before any calibration or evaluation run)

Scope: synchronous, single-request measurement of the existing M1/M1.2 code path
(model `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA, chunk C = 256, chunk file v1).
There is no new runtime policy: the cost model is an offline predictor.
Notation: L = restored prefix length (chunk-aligned), S = suffix length (S ≥ 1), N = L + S.

## 1. Primary metric: tokenized-input first-token latency
Wall clock, `time.perf_counter_ns()`:
```
torch.cuda.synchronize(); t0
engine.prefill_request(prompt, allow_restore=...)   # = AlwaysRestoreEngine.run order up to the first token
int(torch.argmax(logits))                           # first token id on the host (implies device sync)
t1
```
- **Included:**
  - the token-id list → device tensor;
  - chain-key computation (SHA-256);
  - lookup, restore and per-chunk validation (CRC, identity), with H2D copies
    and `DynamicCache` assembly;
  - suffix prefill (or full prefill);
  - **cache admission** of new complete prompt chunks (`store`, D2H into
    pinned host). `run()` performs it before decoding, so it precedes the
    first token;
  - argmax and the transfer of the token id.
- **Excluded:** model loading; tokenization; queueing (this is **not**
  service-level TTFT); benchmark state preparation and resets; page-cache
  handling; correctness checks; reference replay; full-logit copies; GPU
  monitoring.
- **No hooks in timed samples.** Diagnostic phase timings come from separate
  *instrumented* samples: `mark()` synchronizes the device at the phase
  boundaries ingest | restore | prefill | store | select. Instrumented and
  uninstrumented samples are never mixed in one statistic.
- **CUDA events are not used for any reported latency**, which is wall clock
  throughout.

## 2. Paths and cache states (`kvlab/perf_harness.py`)
Before each L group, the prefix chunks [0, L) are prepared from one untimed
source prefill, through the M1 store APIs:

| Path | State | Must report (checked on every sample; otherwise excluded) |
|---|---|---|
| `full` | store A (prefix in pinned host); restore not attempted | restore_len 0, no sources |
| `host` | store A: prefix chunks in pinned host slots | restore_len L, sources all `host` |
| `disk` | store B: prefix chunks as files; host B holds no prefix chunk | restore_len L, sources all `disk`; every file 100% resident in the Linux page cache before t0 (`mincore`) |
| `disk_evicted` (diagnostic, S = 1 only) | store B, after `posix_fadvise(DONTNEED)` on the experiment's own fsynced files | as `disk`; ≤ 1% resident before t0 |

For every path, the stored chunks must equal the complete prompt chunks
beyond L (one chunk when S = 256).
- Each store keeps a free host slot, so admission never spills. Admitted
  chunks are discarded after each sample, outside the timed region.
- Disk hits are not promoted (D15), so repetitions cannot turn disk hits into
  host hits.
- The state invariant is asserted after every sample.

**Disk honesty.** Files live on the WSL2 ext4 virtual disk (`/dev/sdd`, see
the environment record).
- `disk` is a **warm-page-cache** application-level disk-cache hit, not a
  device read.
- `disk_evicted` removes the pages from the *Linux* page cache only; the
  Windows host may still cache the VHD. **No cold-NVMe claim is made.**
- No sudo is used, and caches are never dropped globally.

## 3. Workload grid and split (`kvlab/perf.py`)
- L ∈ {256, 512, 1024, 2048, 4096}, S ∈ {1, 64, 256}: 15 cases, all within
  the model context; L = 4096 + 256 is 4352 tokens.
- **Calibration:** L ∈ {256, 1024, 4096} × S ∈ {1, 256} (6 cases).
- **Held-out evaluation:** the other 9 cases. These are L ∈ {512, 2048} × all
  S (interpolation in L), plus S = 64 at L ∈ {256, 1024, 4096}
  (interpolation in S).
- **Smoke:** (256, 1) and (1024, 64), with reduced repetitions, for harness
  checks only. It is never used for fitting.
- Prompts are prefixes of one fixed token sequence (the fixture texts,
  cycled), so all cases share the same prefix chunks.

## 4. Sampling
- One GPU process at a time.
- Process-level warmup: 2 samples per path at the smallest and largest L of
  the split, discarded.
- Per L group (group order randomized with a recorded seed): the
  correctness check, then the uninstrumented loop, then the instrumented
  loop.
  - **Uninstrumented:** 5 warmup + 20 measured repetitions.
  - **Instrumented:** 2 warmup + 20 measured.
  - In each repetition, every (S, path) of the group runs once in a fresh
    random order.
- `gc.collect()` runs once per repetition; GC is disabled inside every timed
  region.
- All samples, warmups included, are saved raw (ns) with their wall-clock
  timestamps. Exclusions are recorded with reasons, and nothing is dropped
  silently.
- Statistics: median, IQR, min/max, and a bootstrap 95% CI of the median.
  No p99 is reported (n = 20).

**Platform drift, found in pre-protocol probes and recorded here.** The
laptop GPU's SM clock moves between about 2280 and 1750 MHz
(`clocks_event_reasons` includes SW power cap) on a time scale of seconds,
changing GPU-bound latency by about 25%. Power settings must not be changed,
so:
- paths are interleaved within each repetition, and restore-vs-full winners
  use **paired per-repetition differences**;
- a read-only `nvidia-smi -lms 200` monitor runs concurrently, so every
  sample can be joined to a clock reading;
- results are reported with this caveat.

## 5. Correctness (separate from timing)
Once per (L, S), and never timed:
- restored K/V bytes equal the source chunk bytes for the host and disk
  paths;
- **strict:** first-token logits bitwise equal for host == disk == the
  resident split reference (GPU tensors built from the same bytes, never
  through a tier);
- **report-only:** full vs restored first token and the maximum absolute
  logit difference. D10: there is no tie rule and no tolerance gate.

## 6. Cost model (fit on calibration medians only; `kvlab/perf.py`)
`predicted(path) = ingest + restore_path + prefill + store + select + offset_path` (ms):

| Component | Features | Fit on |
|---|---|---|
| ingest | 1, N_kt | instrumented ingest phase, all paths |
| restore_host / restore_disk | 1, chunks | instrumented restore phase per tier |
| prefill_suffix | 1, S_kt, S_kt·(L_kt + S_kt/2) | instrumented prefill phase, host + disk paths |
| prefill_full | 1, N_kt, N_kt²/2 | instrumented prefill phase, full path |
| store | 1, new_chunks | instrumented store phase |
| select, restore_skipped | constant (median) | instrumented phases |
| offset_path | constant | median over calibration cases of (uninstrumented e2e − instrumented total) |

- Fitting uses non-negative least squares with relative-error weights.
- The phases are disjoint by construction (device sync at each boundary), so
  they add up. The offset corrects the instrumentation's own synchronization
  cost instead of summing incompatible numbers.
- Micro-benchmarks (H2D of one chunk, validated file read, raw read, CRC32,
  `DynamicCache` assembly) are diagnostic, used to interpret the per-chunk
  slopes, and are not model inputs.

## 7. Freezing and scoring
1. `m2_measure --split calibration`.
2. `m2_fit_predict` writes `model.json` and `predictions.json`, both
   SHA-256-digested. It refuses to run if any evaluation samples exist.
3. `m2_measure --split evaluation` requires the frozen `predictions.json` and
   records its digest and modification time.
4. `m2_score` verifies the digests and that the predictions predate the
   evaluation start, then joins.

Reported:
- per-path latency prediction error (ms and %) on held-out cases. In-sample
  calibration error is reported separately and labeled as in-sample;
- predicted vs measured winner per (case, tier). The measured winner is
  `restore` or `full` only if the 95% CI of the paired median difference
  excludes 0; otherwise it is `tie`, which is not scorable;
- the penalty (ms, %) of each wrong choice;
- measured crossings between adjacent measured L, and predicted crossings on
  the chunk grid L = 256…4096 (no extrapolation). If there is no crossing,
  or the winner is a tie, that is stated.

Time budget: about 60 minutes of GPU measurement in total. Cases that do not
fit are marked NOT RUN, and an OOM is recorded per L group.

## Amendment A (2026-11-03 13:19, after calibration and model freeze, before any evaluation sample)
Calibration (`results/m2/run_20261103T131543/calibration_samples.json`) showed that the
**restore paths are bimodal**:
- 15–23% of samples, at every position in the repetition, are about 50 ms slower
  (one chunk 20 → 71 ms; 16 disk chunks 107 → 245 ms);
- the full path is not affected (IQR ≤ 4.7 ms);
- the same spikes appeared in pre-protocol probes without the GPU monitor, and all
  calibration samples were in the ≥ 2200 MHz clock regime.

This widens the CIs of paired differences, so the **evaluation uses 40 measured
repetitions** (still 5 warmups; instrumented 2 + 20). Reports add the mean and the
share of samples > 1.3× the case minimum next to the median.

The model and predictions were already frozen and are unaffected. Their digests:
- model `0774b1a2…`
- predictions `1dfb47fa…`
