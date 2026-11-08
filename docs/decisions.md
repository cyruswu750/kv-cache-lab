# Decisions

Why the system is built the way it is. [design.md](design.md) describes what is implemented;
this file records the alternatives and trade-offs behind it, in the order the decisions were made.
Defects are in [BUGLOG.md](BUGLOG.md) and measured results in [results.md](results.md).

Decision IDs are stable and are cited by code, tests and the archived protocols, so entries are
never renumbered. An entry that a later decision replaced keeps its text, and the index marks it.
Entries D31–D39 and D42–D44 govern the formal experiments (their protocols, execution and evidence
handling) rather than the library.

| ID | Decision | Status |
|---|---|---|
| D1 | HF Transformers + PyTorch as the reference harness | Active |
| D2 | bf16 KV, not fp16 | Active |
| D3 | SDPA attention, pinned | Active |
| D4 | Restore through the public `DynamicCache` constructor | Active |
| D5 | Explicit positions and masks | Active |
| D6 | Chunk-granular CPU snapshot | Active |
| D7 | Numerical tolerance fixed before any comparison | Active |
| D8 | Calibration outcome | Record |
| D9 | Fixture `nat_xlong` added | Record |
| D10 | No tie rule; separate storage and full-prefill contracts | Active |
| D11 | M0.1 gate; M0 strict suite kept | Superseded by D10 |
| D12 | Storage correctness is bitwise, never atol | Active |
| D13 | Identity: SHA-256 chain over a canonical namespace | Active |
| D14 | Chunk file: LE header + raw payload + CRC32, no pickle | Active |
| D15 | Exclusive tiers, no promotion | Active |
| D16 | Pinned host slab allocated once | Active |
| D17 | Disk directory ownership | Active |
| D18 | Store after prefill, complete chunks only | Active |
| D19 | Release disk bytes only after unlink succeeds | Active |
| D20 | Validate the directory before touching it | Active |
| D21 | Host accounting terms | Active |
| D22 | Metric: synchronous first-token latency | Active |
| D23 | Engine split instead of a benchmark copy | Active |
| D24 | Interpretable offline cost model (M2) | Active |
| D25 | Platform drift handling | Active |
| D26 | Disk results labelled by page-cache state | Active |
| D27 | Dependence-aware intervals; inconclusive, not tie | Active |
| D28 | Named recompute baselines | Active |
| D29 | Slow mode reported as unresolved | Active |
| D30 | Batched restore backend (opt-in) | Active |
| D31 | Preregistered confirmation with a premeasurement freeze | Experiment record |
| D32 | Trace replay with a bounded GPU-only baseline | Experiment record |
| D33 | Prefix-aware LRU GPU-only baseline (C1-PA) | Experiment record |
| D34 | One timing-only feasibility pilot | Experiment record |
| D35 | Proceed to the M4_EVAL_V1 freeze | Experiment record |
| D36 | Request-scoped warm disk; page-cache interference rule | Experiment record |
| D37 | One V2 calibration session | Experiment record |
| D38 | Freeze V2 and run one batch | Experiment record |
| D39 | Archive M4 evaluations; reproduce from frozen revisions | Experiment record |
| D40 | Opt-in restore-length policy | Active |
| D41 | Separately versioned cost model | Active |
| D42 | Candidate-measurement harness with fail-closed evidence | Experiment record |
| D43 | Multi-candidate statistics | Experiment record |
| D44 | Publish the M5 evidence byte for byte | Experiment record |
| D45 | Cross-CPU tolerance for one derived statistic in historical-analysis tests | Active |

## D1. HF Transformers + PyTorch as the M0 correctness reference harness
- **Decision:** M0 uses `transformers` `DynamicCache` with plain PyTorch
  forward calls. vLLM was not installed or evaluated.
- **Reason:** HF lets us read and write per-layer K/V
  tensors directly and control positions and masks explicitly. That makes it a
  transparent reference for byte-exact and numerical checks.
- **Alternative considered:** vLLM's KV-connector interface was the initial integration
  candidate. It was not installed or evaluated; integration with a serving engine remains
  future work.

## D2. dtype = bfloat16, not fp16
- **Decision:** load the model with an explicit `dtype=torch.bfloat16`.
- **Reason:** the pinned checkpoint's `config.json` declares
  `"torch_dtype": "bfloat16"`, and the GPU supports bf16 (`torch.cuda.is_bf16_supported()` = True).
  bf16 avoids fp16 overflow risk on a bf16-trained model. KV size per token
  is the same (2 bytes/element, 28 KiB/token).
- **Cost:** the dtype is one constant (`kvlab/model.py: DTYPE`); changing it requires
  recalibrating the tolerances.

## D3. Attention backend = SDPA (pinned explicitly)
- `attn_implementation="sdpa"` is passed explicitly and checked after load.
  It was also the default observed in this environment.

## D4. Cache restore through the public `DynamicCache(ddp_cache_data=..., config=...)` constructor
- **Reason:** this avoids writing into private layer fields.
  `DynamicLayer.update` concatenates onto an empty tensor, so the cache ends
  up owning a fresh copy.
- **Cost:** restore does one H2D copy into a staging device tensor, then one
  D2D copy inside the constructor. The diagnostic restore timing includes
  both. A later milestone can remove the extra D2D copy if it matters.

## D5. Explicit positions and masks
- Every forward passes `position_ids = arange(L, L+n)` and a 2D all-ones
  `attention_mask` of length `L+n`. It also asserts that the cache length is
  `L` before the forward and `L+n` after. In transformers 5.22.1, Qwen2 has
  no `cache_position` argument; positions default to `cache.get_seq_length()`.
  Passing them explicitly makes any disagreement impossible to miss.
- `logits_to_keep=1` is used on both paths. The LM head is per-position, so
  this does not change the last position's logits.

## D6. Snapshot is chunk-granular, in pageable CPU memory
- `chunks[c][layer] = (K, V)`, each `[1, kv_heads, C, head_dim]`, contiguous,
  freshly allocated. There are no pinned buffers, files, pools, or async copies
  (all excluded from M0).

## D7. Numerical acceptance rule (fixed 2026-11-02 03:07, before any comparison ran)
Byte equality of every K/V tensor in the round trip is **strict** and is
checked separately from the numerical rule below.

Per case and per restore length L, PASS requires all of the following:
1. No NaN/Inf in first-token logits or any greedy-step logits on either path.
2. First-next-token logits: elementwise `|B - A| <= atol + rtol * |A|` (float32).
3. First-next-token argmax is equal.
4. The 8-token greedy continuation is identical.
5. Negative control (zeroed snapshot at L = Lmax) is *not* allclose to A.

Reported but not gated: `max_abs_err`, `rel_l2_err = ||B-A||_2 / ||A||_2`,
the reference top-2 margin, and bitwise equality against a GPU-resident
chunked prefill (which has no CPU round trip).

**How atol/rtol are set (rule fixed before calibration):**
- `rtol = 0.02`, fixed a priori (about 5 bf16 ulps of relative precision, 2^-8 ≈ 0.0039).
- `atol = max(0.25, 4 × M)`, rounded up to the next 0.05. M is the largest
  `max_abs_err` observed in the calibration run `scripts/m0_calibrate.py`.
  That run compares full prefill against GPU-resident chunked prefill
  (no CPU round trip) on `CALIBRATION_TEXT`, which is disjoint from the
  acceptance fixtures. It measures the partial-vs-full prefill
  shape difference, and the script also measures full-vs-full repeat
  determinism. The 0.25 floor is 2 bf16 ulps at logit magnitude 16–32.
- The resulting values are frozen in `kvlab/tolerances.py` before
  `scripts/m0_experiment.py` is run. If an acceptance case fails, the result
  is reported as FAIL and investigated. The tolerances are not loosened.

## D8. Calibration outcome (2026-11-02, before the first acceptance run)
- `scripts/m0_calibrate.py` → `results/m0/calibration.json`:
  - full vs full repeat: max_abs_err **0.0** on all 4 lengths (deterministic).
  - full vs GPU-resident chunked: M = **0.3535** max_abs_err, max rel_l2 = 3.31e-2,
    argmax equal everywhere.
- Rule D7 gives **atol = 1.45, rtol = 0.02**, frozen in `kvlab/tolerances.py`.
- Note: atol = 1.45 is wide in absolute terms because chunked vs unchunked bf16
  prefill legitimately differs by ~0.2–0.6. The checks that separate a correct
  restore from a broken one are byte equality, bitwise equality with the
  GPU-resident chunked path, and the zeroed-K/V negative control (max_abs
  16–24, far above atol).

## D9. Fixture added after run1: `nat_xlong`
- Run1 had only one natural prompt that could be restored (`nat_long`, N=426).
  No natural prompt exercised partial restore. `nat_xlong` (N=611, L ∈ {256, 512})
  was added for run2. Tolerances and PASS criteria were **not** changed.
  Run1's raw output is kept in `results/m0/run1/`.

## D10. No tie rule; standard argmax; separate storage and full-prefill contracts
**Decision:**
- Preserve standard argmax and the original M0 strict test and its result.
- No approximate-tie rule, no rounding, no tolerance inflation, no hidden xfail.
- Storage correctness is established by byte identity and by strict
  comparison with the resident-GPU reference under matched computation
  provenance (same restore length, same split, same stored chunks; D12).
- Full-prefill vs split-prefill numerical differences, and the greedy
  divergences they cause, remain separately reported limitations.

**Consequences:**
- The M0 strict suite (`-m "gpu and m0_strict"`) keeps its D7 criteria and is
  expected to keep failing on `nat_long`. That FAIL is the recorded result,
  not a test to be fixed.
- `kvlab/metrics.py`, `kvlab/tolerances.py` and the greedy decode are unchanged.
  Inference behavior did not change in M1.1.
- What is claimed: a restored request computes bit-for-bit the same as the
  resident-GPU replay with the same split. What is **not** claimed: that
  restored output equals full-prefill output, or that every full-vs-split
  difference is harmless. The observed differences are bf16 execution-shape
  effects (M0.1); they flipped greedy tokens at exact ties or 1-ulp margins in
  `nat_long` and in 4 of 7 restored `m1_run` requests. Other prompts, models or
  kernels may show other differences, and none of that is gated.

**Original record (2026-11-02, kept for history):**
- `nat_long` fails criterion 4 (greedy 6/8). At step 6, the reference logits
  have an exact tie (`' re'` = `' described'` = 23.75, margin 0.0). The chunked
  computation moves one of them by 1 bf16 ulp (0.125). See
  `results/m0/divergence_nat_long.json`.
- **Not applied** in M0. A possible pre-registered rule for M1: a greedy
  divergence at step s is acceptable only if (a) the reference top-2 margin at
  s is ≤ atol, (b) path B teacher-forced on A's tokens has allclose logits at
  every step, and (c) all byte checks pass. Alternatives: compare in fp32 for
  the reference harness, or gate only on teacher-forced logits. Choose one
  before M1 acceptance runs.

## D11. M0.1 gate passed; M0 strict suite kept as-is while D10 was open
- The M0.1 audit (`docs/results.md`, M0.1) isolated the nat_long mismatch to a bf16
  rounding tie caused by execution shape. Restored and resident splits are
  bitwise equal.
- The M0 strict suite is unchanged and still FAILs. No tie rule was adopted,
  because D10 was still open.
- New suites get their own markers (`m0_1`, `m1`); `-m gpu` still runs all of them.

## D12. Storage correctness is proven strictly, not with atol
- M1 GPU acceptance gates on byte identity and on bitwise equality of all step
  logits against a resident-GPU replay of the same request sequence
  (`kvlab/resident_reference.py`).
- Reason: M0.1 showed that atol=1.45 cannot detect a one-token-wrong prefix,
  a +1 position shift, a chunk swap, or an LSB flip.
- Full-recompute metrics are reported, not gated (known bf16 limitation).

## D13. Identity: SHA-256 chain over a canonical-JSON namespace
- Exact token ids, the parent key, the chunk index, and the namespace digest
  are all hashed. Header fields are stored and compared field by field.
- `attn_implementation`, `torch_version` and `transformers_version` are in the
  namespace even though the bytes layout does not depend on them, because
  they change K/V numerics. Trade-off: any upgrade turns every entry into a
  miss, which is the safe direction.

## D14. File format: little-endian header + raw payload + CRC32, no pickle
- CRC32 (zlib) protects the header and payload against accidental corruption.
  It is not authentication.
- Validation is ordered so that no read or allocation is sized by an
  unchecked header field. meta_len ≤ 4096; payload_len must equal the
  expected value.

## D15. Exclusive tiers, no promotion on disk hit
- The simplest coherent ownership model: one copy per chunk. Host eviction
  spills to disk; disk hits stream through the single staging buffer.
- Trade-off: a hot chunk that was spilled stays on disk until it is evicted
  there. Promotion would need eviction during restore. Deferred.

## D16. Host slab pinned with `cudaHostRegister`, allocated once
- Observed in the installed torch (`CachingHostAllocator.h`): `pin_memory=True`
  allocations round up to a power of two below a threshold, and freed blocks
  stay cached, so 7 MiB chunks would really reserve 8 MiB each and never be
  released.
- One exact-size registered slab makes the accounted bytes equal to the
  pinned bytes. Unregistered in `close()`.
- CPU tests use `pin=False` (same code path, pageable slab).

## D17. Disk directory ownership
- The disk tier uses a marker file. It refuses a directory with unknown
  entries and deletes only files matching its own patterns, as direct
  children of the root. Stale files are deleted at startup, because
  restart recovery is out of scope.

## D18. Store after prefill, before decode; store only complete prompt chunks
- The cache then holds exactly N prompt tokens.
- Generated tokens are not cached.
- A write or spill failure never re-runs inference, because the result
  already exists.

## D19. (M1.1) Disk bytes are released only after the file is gone; undeletable files stay accounted
- **Prompted by:** BUGLOG BUG-001 (trigger, root cause, impact, tests, evidence).
- **Decision:** unlink first; release bytes only if the file is gone (an
  already missing file counts as gone). If unlink raises `OSError`, the entry
  stops being readable and its bytes move to `undeletable` (still in
  `used_bytes`, counted against `max_entries`, never read, and its key is
  never rewritten: `undeletable_file_for_key`).
- **Eviction:** a failed victim makes `put()` return
  `(False, "evict_oserror:<errno>")`, which is a reported write skip. The
  victim is not retried automatically. A later `put()` evicts the next LRU
  entry, or returns `disk_budget_unavailable` without evicting anything when
  temp reservations + undeletable bytes leave no room.
- **Invalid entry:** if a bad entry's file cannot be deleted, the read is
  still a miss (safe fallback), and the bytes stay accounted.
- **Temp file:** if cleanup fails, its full reservation stays accounted as
  undeletable. The actual partial file is at most that size.
- **Only `OSError` is handled** (`_unlink`); `ValueError` from path checks,
  `TypeError`, etc. propagate. `clear()` makes one explicit attempt per
  undeletable file. There is no background cleanup and no retry loop.
- **Startup:** a stale file that cannot be deleted makes construction fail
  with that `OSError`, before any request and with nothing accounted.
- Trade-off: an undeletable file permanently reduces disk capacity for the
  process lifetime. This is the safe direction; it is reported in `budgets()`.

## D20. (M1.1) Validate the whole directory before touching it; never write through the marker
- **Prompted by:** BUGLOG BUG-002.
- **Decision:** validate first, using `os.lstat` / `os.scandir` without
  following symlinks. The root must be a real directory, and the marker must
  be a regular file. Every other entry must have our name pattern and be a
  regular file or symlink. Only then are stale files unlinked (unlink does not
  follow symlinks).
- A missing marker is created with `O_CREAT|O_EXCL|O_NOFOLLOW`. An existing
  marker is never rewritten. Chunk reads open with `O_NOFOLLOW`, so a
  symlinked chunk file is a miss (`ELOOP`) and the link is removed.
- **Limit:** this is for a trusted, single-process directory. It does not
  defend against concurrent or adversarial changes between the check and the
  use (TOCTOU).

## D21. (M1.1) Host accounting terms
- **Prompted by:** BUGLOG RPT-003 (a reporting definition, not a functional defect).
- The slab is allocated once at construction and never resized, so
  `slab_bytes` (≤ budget) is what the host tier holds from start to `close()`.
- These are reported separately: `slab_allocated_bytes`,
  `cuda_host_registered_bytes` (0 when `pin=False`), `staging_reserved_bytes`,
  `resident_logical_bytes` and `resident_logical_peak_bytes`.
- Resident bytes are logical occupancy inside the slab, not additional
  memory. None of these figures is an RSS or page-lock measurement.
- `metadata_bytes_estimate` is an estimate of raw identity field bytes, not
  measured Python heap usage.
- The ambiguous `HostStore.used_bytes` / `peak_bytes` were removed. M1 evidence
  files keep the old keys; how to read them is in RPT-003.

## D22. (M2) Metric = tokenized-input first-token latency, including cache admission
- Wall clock runs from a synchronized start to the first token id on the
  host, around `engine.prefill_request()` + argmax.
- The existing engine stores (D2H) before selecting the first token, so
  admission of new complete chunks is **inside** the metric (~4.5 ms per
  chunk). Leaving it out would flatter every path equally, but would be wrong
  in absolute terms.
- Excluded: tokenization, queueing, model load, setup and checks. It is
  therefore not called TTFT.
- Phase timings come only from separate synchronized (instrumented) samples.

## D23. (M2) Engine split instead of a benchmark copy of the request path
- `prefill_request()` was factored out of `run()` so that the measured path
  *is* the engine path. `allow_restore=False` makes full recomputation
  measurable with identical ingest/store work.
- `HostStore.discard()` was added for resets.
- Both are covered by CPU tests, and the GPU M1 suite was rerun (12 passed).
- This is not a runtime policy: `run()` still always restores.

## D24. (M2) Interpretable cost model, fitted only on calibration medians
- Components: ingest(N), restore per tier (per chunk), suffix prefill
  (1, S, S·(L + S/2)), full prefill (1, N, N²/2), store (per admitted chunk),
  select, and a per-path instrumentation offset.
- Fitting: non-negative least squares, weighted by relative error. Medians
  are used for robustness.
- The S² term was dropped because calibration has only two S values
  (identifiability).
- Held-out cases interpolate in both L and S.
- The model and predictions are SHA-256-frozen before evaluation, and the
  score verifies the chain.

## D25. (M2) Platform drift: paired winners, more repetitions, honest reporting
- Observed: SM clock regimes between 1335 and 2385 MHz (SW power cap), and a
  time-clustered ~50 ms slow mode on the restore paths. Power and driver
  settings may not be changed.
- Response:
  - paths are interleaved within each repetition, and winners come from
    paired differences with a bootstrap CI (a tie when the CI contains 0);
  - a concurrent read-only clock monitor runs;
  - evaluation used 40 repetitions (Amendment A, recorded before evaluation);
  - medians, means and slow shares are reported, and no p99.
- Not done: filtering slow samples (they are real latency), or locking
  clocks.

## D26. (M2) Disk results are labeled by page-cache state; no cold-device claim
- `disk` is warm (100% resident, verified with `mincore`).
- `disk_evicted` uses `posix_fadvise(DONTNEED)` on the experiment's own
  fsynced files (0% resident), but the WSL2 host may still cache the VHD.
- Cold NVMe latency is BLOCKED without host or privileged cache control.

## D27. (M2.1) Dependence-aware intervals; "inconclusive", not "tie"; penalties over every decision
- **Problem:** M2 bootstrapped paired differences as if the repetitions were
  independent. The data has lag-1 autocorrelation up to 0.93. Penalties were
  summed over scorable decisions only. See BUGLOG VAL-005.
- **Change:**
  - `kvlab/perf_stats.py` adds a circular moving-block bootstrap over
    chronological repetitions (block lengths 1, 2, 4, 8) for the median and
    the mean of paired differences.
  - A direction is called established only if every block length agrees.
  - An interval that contains 0 is "inconclusive", which is never read as
    equivalence.
  - The selection penalty is reported for every decision, as a point estimate
    of mean(chosen) − mean(other), clipped at 0. Its status is established,
    established_absent or inconclusive.
- **Unchanged:** `kvlab/perf.py` stays byte-identical. The M2 scores, model
  and predictions are not rewritten; `scripts/m2_1_reanalyze.py` reads them.
  The original model is not refitted.
- **Limitation:** a run of 40 repetitions spanning 14–50 s cannot reliably
  estimate uncertainty when the slow regimes last about as long as the run
  itself. AR(1) effective sample sizes of 1–3 are reported as such.

## D28. (M2.1) Name the M2 `full` baseline; a separate no-cache diagnostic baseline
- **M2 `full`** means *recompute from the prepared cache state*. The prefix
  chunks are cached in store A and restore is not attempted. Ingest (chain
  keys), lookup-free counters and admission of new complete chunks run as in
  the engine. It remains the baseline of every M2 score.
- **`full_nocache`** (M2.1 diagnostic only, `kvlab/diag_harness.py`):
  - token ids go to the device, then `kvlab.model.prefill` with an empty
    cache, then argmax;
  - there are no chain keys, lookups or admission;
  - the timing boundary is the same as for every other path;
  - its logits must be bitwise equal to `full`'s (checked untimed in every
    session).
- **How they differ:** for S < 256 no new complete chunk exists, so the two
  differ only by chain keys and bookkeeping. For S = 256 (not measured in
  M2.1), `full` also pays one admission.
- **Not measured:** a true miss, where a recompute admits *every* prompt
  chunk.
- `full_nocache` is never mixed into the historical scores.

## D29. (M2.1) The restore slow mode is reported as a narrowed, unresolved platform effect; no model or policy change
- **Evidence** (`docs/results.md`, M2.1): in time-clustered regimes, the
  host-side CPU cost per CUDA launch or synchronous copy rises 2–5×. GPU
  kernel time, pure-CPU work, guest system calls and the SM clock are
  unchanged, and the thread stays on-CPU. Every path is affected; recompute
  hides it when GPU-bound.
- **Decision:**
  - no layer below the CUDA runtime (WSL paravirtualization, host
    scheduling, power management, the driver) is named as the cause;
  - slow samples are never filtered;
  - the frozen M2 model is not refitted;
  - a mean-predicting two-state model with per-call costs is a documented
    proposal only. It needs a fresh evaluation set.
- **Consequence for later work:** restore-path changes should be judged by
  the number of CUDA calls they remove, as well as by bytes. They should be
  measured with the M2.1 protocol (interleaved, block-bootstrap, mean and
  median).

## D30. (M3) Batched restore backend: chunk-major device staging, one H2D per chunk, one placement per group; opt-in
- **Problem:** the M1 restore issues 56 blocking H2D copies per chunk (28 layers × K/V). For
  n > 1 chunks, PyTorch adds a device temporary and a copy kernel to each copy, because the
  per-layer destination slice is strided. A 16-chunk restore makes about 900 synchronizing
  copies and 950 kernel launches. M2.1 associated the restore slow mode with a higher
  host-side cost per CUDA call (D29), so the call count is the target.
- **Choice** (docs/M3_DESIGN.md §3, option b):
  - one blocking contiguous H2D of the whole 7 MB slot per chunk, into a chunk-major device
    staging tensor of min(n, G) chunks (G = 4);
  - one permuted `copy_` per group into a single layer-major destination;
  - the unchanged D4 constructor, which copies exactly [0, L) into fresh tensors.
- **Rejected or deferred:**
  - one strided `copy_` per chunk (option a): same transfers, n placement kernels, and an
    implicit temporary;
  - `cudaMemcpy2D` through ctypes (option c): not a standard PyTorch operation;
  - `non_blocking` copies with leases held to a group sync (option d): PyTorch records events
    only for its own pinned allocator, not for our registered slab, and the staging slot
    would still need a sync per disk chunk;
  - skipping the constructor copy (option e): would need private `DynamicLayer` field writes
    (conflicts with D4). It remains a measured cost.
- **Kept identical:** chunk format, identity, tiers, longest-valid-prefix, stop at the first
  invalid chunk, sources, counters, miss reasons, budgets, admission and LRU. The default
  backend stays `baseline`; changing it needs a separate decision. Switching is one
  constructor argument or attribute.
- **Cost:** transient device memory of min(n, G) × 7,340,032 B (≤ 28 MiB) per restore. No new
  host memory.
- **Evidence (`docs/results.md`, M3):**
  - All strict, A/B and M1 checks pass for both backends, and the mutation check kills 10/10.
  - Measured per 16-chunk restore: 896 → 16 memcpy, 896 → 16 syncs, 952 → 60 launches.
  - Median e2e falls 1.07× (1 chunk) to 2.29× (16 chunks, host).
  - The pre-registered adoption rule is not fully met. (1024, 1) is inconclusive under the
    duration rule R3, although every block length and session agrees on the direction.
    Hence the default stays `baseline`.
  - *Correction note (VAL-006, added after M3):* the M3 analysis script failed to validate
    missing or incomplete evidence, measured samples and retained warmups alike. It now validates against the frozen spec first, and the
    reanalysis reproduces every number. `supports_adoption` remains false for the same
    (1024, 1) reason, and the default stays `baseline`. See BUGLOG VAL-006.

## D31. (M3_CONFIRM_V1; frozen and measured 2026-11-04) A separate, preregistered confirmation, bound to a premeasurement freeze
- **Problem:** the M3 adoption rule failed only on (1024, 1), under R3. Its measured span was
  too short relative to the slow regimes. Re-analysing the M3 data, or extending it, would
  make the rule depend on the outcome.
- **Choice** (`docs/M3_CONFIRM_PROTOCOL.md`):
  - a new experiment with its own validator spec, `M3_CONFIRM_V1`, analysed on its own and
    never pooled with M3. Cases: (1024, 1) and the fresh (1280, 96). Five fresh-process
    sessions, seeds 32001–32005; 4 blocks × (2 warmup + 24 measured) repetitions per case;
  - the unchanged runner, statistics and R1–R3;
  - confirmed only if every check passes and all 8 primary verdicts (case × tier × mean and
    median of the plain e2e difference) are "batched faster";
  - the flag is `m3_confirm_v1_confirmed`. It does not change the M3 result and does not by
    itself authorize a default change.
- **Freeze:** before measurement, `scripts/m3_confirm.py freeze` writes a
  manifest without a commit. It records the base HEAD, the diff, the actual bytes of every
  executable file (untracked included), the documents, the commands and the stack, which must
  equal the M3 stack. The launch, every session record, the batch record and the analysis are
  checked against that manifest, not against each other. While an analysis file differs from
  its frozen bytes, the analysis rejects the evidence under this spec. Restoring the frozen
  bytes reproduces the frozen analysis of the preserved raw measurements, which a later
  source edit does not invalidate. Adopting a revised analysis needs a new spec.
- **Execution:** one 1,800 s monotonic deadline for the whole batch, rechecked before every
  launch and before accepting a session's exit. No early stopping, retries, replacements or
  extensions.
  - **Stopping:** any nonzero session exit stops the batch: nothing further is launched,
    whatever the session file says. This was chosen over a correctness-only exception
    (2026-11-03).
  - **Execution status:** it is recorded separately from evidence completeness. The wrapper
    returns nonzero for every execution failure. The analysis judges the actual evidence:
    exit 3 for missing, inconsistent, timed-out, OOM-affected or excluded-sample evidence;
    exit 4 for complete, consistent evidence with a failed check, which the stopping rule
    allows only in the last session.
  - **Visibility of failures:** recorded correctness failures are listed in every case.
  - **Process groups:** each session runs in its own process group. Stop signals are received
    synchronously, so cleanup always terminates and reaps that group and signals nothing else.
    SIGKILL of the wrapper is covered only by the parent-death signal.
- **Exact identities:** the runner draws all randomness from the session seed, so the
  validator replays the whole sample sequence, not only counts. The replay reproduces all
  6,480 M3 sample identities.
- **Times as numbers** (2026-11-04): every compared time must be a JSON
  number, not a boolean, that converts to a finite float (`kvlab/m3_confirm.py` `finite()`).
  - **Problem:** a JSON integer beyond the float range made the check itself raise
    `OverflowError`, so the analysis crashed (exit 1, no report) instead of rejecting the
    evidence. Accepting every integer would only move the crash into the analysis, which
    subtracts and formats sample wall times as floats.
  - **Shared with the frozen M3 spec:** the sample wall-time check (`t_wall`) now uses the same
    helper. For `FROZEN_M3_EVAL` this changes only malformed input: JSON `true` was accepted as
    a time and an out-of-range integer crashed the check; both are now invalid evidence
    (exit 3). The spec itself and its decision on the preserved M3 evaluation are unchanged.
- **Rejected:**
  - pooling with M3, or adding sessions to M3: outcome-dependent;
  - a generic orchestration framework: not needed for five sequential commands;
  - writing samples incrementally: it would change the runner and the timed loop.
- **Outcome (2026-11-04):**
  - **Freeze:** frozen before any measurement, manifest SHA-256
    `dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216`.
  - **Batch:** exactly one, 798 s of the 1,800 s budget; all sessions exited 0.
  - **Analysis:** exit 0, valid evidence; all 8 primary verdicts are "batched faster", so
    `m3_confirm_v1_confirmed = true` (`docs/results.md`, M3_CONFIRM_V1).
  - **R3:** the effective-sample-size condition is applied to the sum over strata, as frozen;
    the dated clarification C1 (`docs/M3_CONFIRM_CLARIFICATIONS.md`) records this without
    changing the protocol.
  - **Unchanged:** M3's `supports_adoption = false`; the default backend stays `baseline`.
    No default change has been decided.

## D32. (M4-A, CPU only) Request-trace replay with a separate, bounded GPU-only baseline; unequal resources are accounted, not equalized
Numbering note: D31 is M3_CONFIRM_V1. When D32 was written, this branch (`m4a-trace-replay`,
from `858f952`) did not yet contain it; `main` was merged in afterwards.
- **Problem:** every timing so far is a single-request microbenchmark. A workload-level
  comparison needs identical request traces, a GPU-only baseline that makes its own decisions,
  and honest resource accounting.
- **Choice:**
  - **Traces:** synthetic, seeded traces (T0–T3, design §12), identified by digest and labeled
    synthetic.
  - **C1:** a new `GPUPrefixCache` with the tiered engine's identity, lookup, first-token and
    storage rules, and plain LRU. Only the storage location differs.
  - **Ownership:** entries are owned contiguous copies, measured by actual storage.
  - **Strictness:** each configuration is checked bitwise against a resident replay of its own
    decisions.
  - **Resources:** reported as separate terms. C1's persistent device KV and C2's pinned host,
    disk and (uncounted) page cache are different resources, never summed or called equal.
- **Rejected:**
  - extending `ResidentReference` into the baseline (the oracle would check itself);
  - storing slices of the request's KV (hidden retention of the whole request KV);
  - a single "matched total budget" (GPU memory and pinned host memory are not
    interchangeable);
  - a store that refreshes present chunks (the host tier does not; the policies should differ
    only in storage location).
- **Scope:**
  - CPU correctness only;
  - no GPU execution, timing, calibration, runtime policy or framework integration;
  - GPU validation is deferred to M4-B.

## D33. (M4-B, CPU preparation) Prefix-aware LRU as the primary GPU-only baseline; plain LRU kept as an offline diagnostic
Context: M4-B preparation, 2026-11-04; protocol `docs/M4B_PROTOCOL.md`.
- **Problem:** the M4-A C1 uses plain chunk LRU, the same rule as the host tier (D32). Under a
  tight GPU cap, CPU probes on the unchanged class showed three failure modes:
  - a prompt with more new chunks than the capacity evicts its own leading chunks, and keeps an
    unreachable tail;
  - root-first hit touches let an overflowing request evict a shared root while its
    descendants stay cached;
  - such dead entries still count against the budget.

  On the draft T1 trace at ⌈W/2⌉ = 3 chunks, plain LRU restores 0 tokens, against an oracle of
  96,768. Comparing the tiered cache with that baseline would compare it with a degenerate
  GPU-only cache.
- **Choice:** C1-PA (`kvlab/prefix_aware_cache.py`), a separate subclass. The existing
  `GPUPrefixCache` is unchanged.
  - **Kept:** identity checks, lookup, owned storage and the byte budget.
  - **Admission:** keeps the cache prefix-closed. When full, it evicts the least recently used
    leaf that is not one of the current request's chunks; if none exists, it stops admitting
    that request.
  - **Protection lifecycle:** a request's chunks join the protected set as the store loop
    reaches them, and the set is dropped when store() returns.
  - **Offline effect:** on the draft traces C1-PA restores 48,384 / 8,448 / 65,536 tokens at
    ⌈W/2⌉ (T1 / T2 / T3), against plain LRU's 0 / 8,192 / 63,744. It is never below plain LRU
    on any request there.
- **Basis:** the choice rests on CPU mechanics and deterministic trace statistics only. No GPU
  timing informed it.
- **Scope:** prefix-aware eviction is used for C1 only, although it had originally been left
  out of M4. C2's host and disk tiers keep plain chunk LRU, unchanged.
  - The asymmetry is disclosed.
  - No direction of latency bias is claimed: prefix maintenance and different
    restore/recompute splits have their own costs.
- **Also decided:**
  - plain C1-LRU is an offline diagnostic (`kvlab.trace_models.lru_model`), not a timed arm;
  - both C2 backends run, explicitly selected: `baseline`, which remains the project default,
    and `batched`, opt-in;
  - draft parameters: 10 sessions, a 2,700 s deadline and a 15-minute GPU correctness gate.
- **Rejected:**
  - timing plain LRU as another arm: more GPU time for a baseline already known to be
    degenerate under these caps;
  - changing the existing M4-A class in place: it would silently change tested behavior;
  - prefix-aware eviction in C2: it would change the measured engine.

## D34. (M4-B) One timing-only feasibility pilot on calibration traces
Context: 2026-11-04, after the §9 GPU correctness gate passed (attempt 3).
- **Problem:** protocol §9.5 requires that the evaluation batch be projected to finish within
  80% of its 2,700 s deadline. The existing evidence did not support such an estimate either
  way:
  - the M2 cost model's projection was about 1,400 s;
  - scaled by the gate's ratio of measured to predicted time, it rose to about 2,300 s.
- **Decision:** one fresh-process session, with a 600 s cap, using:
  - the full calibration traces (41001–41004), the protocol's warmup, the five configurations,
    steps = 1, the strict checks, mincore/prewarm and the full records;
  - a private local adapter, not committed, that runs the unchanged `kvlab.m4_bench.run_session`
    with an isolated pilot spec;
  - no evaluation seed.

  This is a limited exception to the rule that no calibration timing is analysed. It is
  allowed solely for the feasibility decision.
- **Pre-declared before the run:** the inputs, the validity rules, the projection formulas, the
  workload mapping and the readings.
- **Result:**
  - the session passed its validation (0 strict mismatches) and took 189 s end to end;
  - the ten-session projections are 1,891 s (10 × the session), and 1,920 s and 1,935 s when
    adjusted for the evaluation workload by the M2 model and by the token count;
  - all are below 2,160 s.
- **Limits:**
  - one session measures no run-to-run variability, and does not guarantee that ten sessions
    finish in time (M3 recorded a persistent platform slow mode);
  - the batch's own deadline and stopping rule remain the protection.
- **Not decided by it:** no policy, configuration, evaluation seed, budget, metric, threshold,
  statistical rule, session count or deadline was selected or changed. It is not performance
  evidence, and it is not by itself a basis for the freeze.

## D35. (M4-B) Proceed to the M4_EVAL_V1 freeze on the basis of the single pilot
Context: 2026-11-04, after D34's pilot, which served only the feasibility question.
- **Basis:** the single D34 pilot is accepted as the planning basis, with its documented
  uncertainty:
  - **Gate attempt 3:** passed, with 46/46 tests.
  - **Pilot session:** one complete calibration session of 189.1 s, which passed validation.
  - **Projections:** the pre-declared ten-session projections are 1,891 s, 1,920 s and 1,935 s,
    all below the 2,160 s (80%) planning threshold.
  - **Pilot assessment:** passed, with qualifications about extrapolation.
- **Accepted uncertainty:**
  - one pilot does not establish session-to-session variability;
  - the calibration-to-evaluation workload mapping is approximate;
  - the batch's overhead between sessions may differ.
- **Unchanged:**
  - 2,700 s remains the hard batch deadline. A timeout, or incomplete or inconsistent evidence,
    remains invalid (exit 3), with no performance verdict;
  - no additional sessions, retries, extensions or parameter changes;
  - the workload, seeds, configurations, budgets, R = 10, statistics, the 15 primary comparisons,
    the validity checks and the stopping rules;
  - `baseline` stays the default.
- **Scope:** removing the protocol's DRAFT marker, then creating and verifying the
  premeasurement freeze, in the manner of D31. Running the evaluation batch was a separate step.

## D36. (M4_EVAL_V2, DRAFT; CPU only) A request-scoped warm-disk preparation and a batch page-cache interference rule
Proposed on 2026-11-04 after M4_EVAL_V1 was executed once and its evidence was invalid under the frozen
whole-tier warm-disk rule (12 of 4,480 C2 requests below full residency after one prewarm; no performance
verdict). A post-outcome diagnostic replay did not reproduce the losses, and their cause is unresolved. This is
a design change for a new spec. It does not correct, reinterpret or re-analyse V1.
- **Change:** before each C2 request, only the request's candidate disk files are handled: the files its restore
  reads on the normal path, computed without side effects. The procedure is one initial `mincore` pass, at most
  one whole-file read per candidate below full residency (no retry) and one final pass. Then the unchanged timed
  region starts. Details: `docs/M4_EVAL_V2_PROTOCOL.md` §3.
- **Claim, narrowed on purpose:** every candidate was observed fully resident in the one verification pass
  immediately before the request. It is not a simultaneous snapshot, and not a guarantee at timer start or
  during the read. Pages are not pinned.
- **Interference rule:** any change of `drop_pagecache` between the batch's start and end reads, or any
  missing, unreadable or differing reading in the sessions, invalidates the batch. Ordinary reclaim counters are
  context only.
- **Why:**
  - V1's whole-tier condition covered files the request never reads. Its sequential, unpinned check could not be
    a simultaneous guarantee.
  - The request-scoped condition matches the direct disk reads more closely.
  - Pinning, retries, system tuning and threshold relaxation are not justified by the evidence.
- **Unchanged:** V1's spec dict and digest, its protocol (byte-identical), its evidence and its invalid status;
  every V2 parameter other than the name, run directory, protocol document and `warm_disk`; the restore path;
  `baseline` as the default.
- **Known risk:** WSL's `mini_init` dropped caches 18 times in about 6.7 h of uptime on this host, at gaps of
  42–4,291 s. A V2 batch that overlaps such an event is invalid by this rule.

## D37. (M4_EVAL_V2, DRAFT) One bounded calibration session of the committed V2 implementation
Context: 2026-11-04, after the V2 implementation was committed, while the V2 protocol was still a draft.
- **Scope:**
  - exactly one session of a private calibration spec, not registered in `kvlab.m4_spec.SPECS`;
  - every experimental field of `M4_EVAL_V2`, including the request-scoped warm-disk preparation;
  - the full calibration traces (41001–41004) in the timed slot, the unchanged warmup (43001–43004) and session
    seed 45001, the D34 pilot's;
  - no evaluation seed.
- **Execution:**
  - the D34 pilot's bounded launcher, with one absolute 600 s deadline that includes termination and reaping;
  - the committed source, bound by HEAD and a clean tree, and the verified interpreter, weights cache and stack;
  - batch-style `drop_pagecache` reads around the single launch.
- **Validity:** the V2 rules apply without exception. These include the warm-disk rule on every C2 record and the
  interference rule. A failure, timeout or invalid result is kept as evidence and is not retried or replaced.
- **Purpose:** execution correctness, V2 evidence validity, and a runtime-feasibility projection with D34's
  pre-declared formulas and mapping, compared with 2,160 s and 2,700 s.
- **Not a basis for:** any performance, latency or backend conclusion.
- **Records:** the calibration adapter and its evidence are kept locally, not in the repository.

## D38. (M4_EVAL_V2) Freeze the V2 protocol and run one formal batch
Context: 2026-11-05, after D37's single calibration session.
- **Basis:** the D37 calibration, a private spec with the calibration traces and session seed 45001.
  - Its launcher passed: 197.6 s of the 600 s cap, with the group reaped.
  - Its evidence was valid under the V2 rules:
    - 1,220 records (100 warmup and 1,120 timed), with 0 strict mismatches in the 1,120 timed records;
    - all 488 C2 records `verified` or `empty`, with 0 prewarms;
    - `drop_pagecache` unchanged.
  - Its pre-declared ten-session projections were 1,976 s, 2,005 s and 2,021 s, all below the 2,160 s planning
    threshold.
  - It is not performance evidence and not a guarantee.
- **Accepted uncertainty:**
  - the calibration-to-evaluation workload mapping is approximate;
  - the overhead between sessions and the variability from session to session are unmeasured;
  - the branch where a candidate needs a prewarm never executed on the GPU;
  - ordinary page-cache reclaim and WSL `drop_pagecache` events can occur during the batch. One WSL
    `drop_pagecache` event during the batch makes it invalid; ordinary reclaim counters remain context only;
  - a formal batch may be invalid despite the successful calibration.
- **Unchanged:**
  - the environment, the workload, seeds, configurations, budgets, R = 10, the statistics and the 15 primary
    comparisons;
  - the request-scoped warm-disk rule, the batch interference rule and every other validity rule;
  - the 2,700 s hard deadline and the stopping rules;
  - M4_EVAL_V1's protocol, its invalid result, its freeze and its evidence;
  - `baseline` as the default.
- **Scope:**
  - removing the DRAFT marker from `docs/M4_EVAL_V2_PROTOCOL.md`;
  - freezing and verifying `M4_EVAL_V2`;
  - at most ONE formal ten-session batch;
  - the frozen validation and analysis of its evidence.
- **Excluded:**
  - retries, replacement sessions, a second batch, deadline extensions or parameter changes;
  - system or WSL setting changes, page pinning, cache drops, or relaxed checks;
  - default changes.

  A failed, timed-out or invalid batch is kept and reported, with no performance statement.

## D39. (M4) Archive the executed M4 evaluations and reproduce them from their frozen revisions
Context: 2026-11-05, after the one M4_EVAL_V2 batch.
- **What is committed, byte for byte, under `results/m4/`:**
  - **M4_EVAL_V2:** the freeze manifest and its empty diff sidecar, the run directory and the frozen validation and
    analysis outputs. Until now they were untracked in this worktree.
  - **M4_EVAL_V1:** the freeze manifest and its empty sidecar, the run directory and the frozen validation output.
    The analyzer exited 3, so there are no statistics.
    - These are copies of the untracked files in the `m4a-trace-replay` worktree at `b858d9e`, at the same
      relative paths. Each manifest's `run_dir` and the frozen analyzer's manifest path are relative to the
      repository root, and the paths are distinct from V2's.
    - The V1 worktree is not modified. V1 stays invalid, with no performance conclusion.
- **Unchanged:**
  - no manifest is regenerated;
  - no evidence file, recorded path or analysis output is edited. The absolute `run_dir` in each
    `m4_validation.json` names the worktree that analysis ran in;
  - the frozen protocols and every rule, M3, and `baseline` as the default.
- **Verification after these commits:**
  - `scripts/m4_eval.py verify` binds HEAD and the source set, so it now fails for this checkout. That is expected,
    and it is not weakened.
  - The analyses are reproduced offline instead, by each frozen revision's own analyzer, using
    `scripts/m4_reconstruct.py` (`docs/results.md`, M4).
  - This does not re-establish the live stack. The batch records keep the pre-launch verification that ran on the
    recording machine.

## D40. (M5a; CPU only) An opt-in restore-length policy with side-effect-free planning
Implemented 2026-11-05 (`docs/M5_DESIGN.md`), before any GPU run, calibration or measurement.
- **Hook:** `AlwaysRestoreEngine(model, tiered, *, policy=None)`.
  - `policy=None` keeps the M1 operational flow unchanged; the only additions are a `decision = None` default and
    a policy check. A CPU differential test compares it with an adapted copy of the pre-M5a `prefill_request`
    body, and planning is patched to fail during that test.
  - `allow_restore=False` never consults a policy.
- **Planning** (`kvlab/restore_policy.py`):
  - **Inputs:** index membership and configuration only. A planned chunk is indexed, not validated.
  - **Candidates:** `0, C, …, k·C` under the first-token rule, stopping at the first absent chunk and, without a
    staging buffer, at the first disk chunk.
- **Selection** is validated before any restore. Booleans, non-integers, negative, misaligned and out-of-set values
  raise `PolicyError`. Nothing is clamped, and `FixedChunks` never substitutes another length.
- **Execution** restores exactly `keys[:selected // C]` through the unchanged `TieredKVCache.restore()`.
  - A validation failure inside the selected prefix keeps the verified prefix (`actual_len < selected_len`), with
    no re-plan and exactly one prefill.
  - L = 0 skips `restore()`, clears `last_restore` and is not a miss.
- **Records and counters:**
  - **Record:** `RestoreDecision`, on `Prefilled.decision` and `RequestResult.decision` (default `None`).
  - **Counters:** no field was added. Under a policy, miss counters cover only failures inside the selected
    prefix; limits found by planning are in the plan.
- **Rejected alternatives:**
  - a per-call policy argument: the constructor hook is smaller, and `engine.policy` can be swapped;
  - re-planning after truncation: a second decision path with no measured benefit;
  - silently clamping invalid outputs: it would hide policy bugs and corrupt candidate measurements;
  - counting plan limits as misses: it would fabricate lookup events the opt-in path never performs.

## D41. (M5a; CPU only) A separately versioned cost model; no artifact ships
- **Why a new model** (`kvlab/restore_cost.py`, schema `kvlab-restore-cost-v1`): the M2 model describes the
  baseline backend, S ≤ C, medians and all-host or all-disk prefixes. It is frozen and is not reused, and neither
  are its coefficients.
- **Basis (9 non-negative coefficients):**
  - the full-prefill path: const, N, N²/2;
  - the suffix path: const, host chunks, disk chunks, S, S·(L + S/2);
  - one admission term (admitted chunks), shared by both paths.

  The pair features are query-key pair counts; omitting a free S² term is a chosen model constraint, not an
  identifiability limit. Full column rank of the column-scaled design is required, not a row count.
- **Target:** case means per candidate, because the future metric is mean regret.
- **Fit:** NNLS by exact support enumeration, with an explicit `uniform` or `inverse_mean` weighting.
  `perf.nnls_small` caps at 8 columns and is frozen, so it is not reused.
- **Objective:** the policy omits the admission term. It is identical for every candidate when a candidate does
  not change what is admitted, which the draft protocol requires (no admission spill). Admission stays in
  predictions and in every measurement.
- **Artifact:**
  - **Contents:** a canonical-JSON digest over the schema, units, compatibility (namespace, chunk size, backend,
    staging group), coefficients, fit diagnostics, a box domain, the calibration and evaluation ids and the
    provenance.
  - **Validation:** once, at load.
  - **Errors:** an incompatible artifact raises. A plan with no positive candidate is decided (L = 0) before any
    domain check. Otherwise decisions check the box projected onto the objective's inputs (N, and S and the chunk
    counts of each positive candidate; S bounds apply to L > 0 only): inside it the policy uses the model; outside
    it the policy falls back to always-restore with a recorded reason. The admitted-chunk range is prediction-only
    (`predict_ms`), because m does not enter the objective. Arbitrary errors are never caught as a fallback.
- **Fit isolation:** fitting refuses declared evaluation cases, undeclared, duplicate or missing cases, and
  incomplete candidate sets. Nothing is filtered silently.
- **Not done:** calibration, evaluation and GPU correctness. Tests use synthetic coefficients only.

## D42. (M5b preparation; CPU only) A bounded candidate-measurement harness with a declared grid and fail-closed evidence
Implemented 2026-11-05, before the GPU gate, pilot, calibration and evaluation. Protocol: `docs/M5_PROTOCOL_DRAFT.md`.
- **Specification** (`kvlab/m5_spec.py`):
  - **The grid:** 12 calibration and 8 evaluation cases are declared before any measurement (no timing chose
    them). Every case lists the state of each complete chunk; the candidates, admissions, capacities and
    provenance are derived from that.
  - **The backend:** `batched` with a staging group of 4, selected for performance. The default stays `baseline`.
  - **Checks before measurement:** the calibration design is checked on CPU (rank 9 of 9, condition number 28.4).
    The declared domain covers every evaluation case; one outside would be a build error, never dropped.
  - **The holdout:** separated by configuration signature (the id and the prompt seed excluded), and prompts are
    disjoint.
- **State** (`kvlab/m5_state.py`):
  - **Preparation:** one bank per case (one full prefill of `prompt[:P]` through the M1 store path) fills the
    stores and seeds the `ResidentReference`.
  - **Reset:** it uses only public store operations (discard in reverse allocation order, lease or `put` in
    declared order).
  - **Verification:** a structural fingerprint before every sample, and content hashes at the case's start and
    end.
- **Samples** (`kvlab/m5_bench.py`):
  - **Runs:** every candidate through FixedChunks; in evaluation, real CostAware, AlwaysRestore and Recompute
    runs. The orders are seeded.
  - **Page cache:** the common `warm_disk.prepare` over the maximal prefix's disk files.
  - **The timed region:** exactly synchronize, then `prefill_request`, then the first-token argmax.
  - **The strict check:** every sample's actual split, untimed. The collector runs once per repetition.
- **Validation** (`kvlab/m5_validate.py`):
  - **What is recomputed:** every expectation, from the spec and the recorded namespace.
  - **What is reused:** `m3_confirm.run_batch` and `batch_problems` for execution (one deadline, stop on a nonzero
    exit, no retries).
  - **Exit codes:** 3 for any invalidity, 4 for a strict mismatch alone.
  - **Hardening:** non-object files are rejected, a fully successful batch is required, the frozen
    predictions are recomputed from the bound cost model (before launch and in validation), page-cache records
    are checked for schema, bounds and chronology, and identity fields are required.
- **Rejected alternatives:**
  - a per-sample forward hook in the timed region (instrumentation);
  - a per-sample content hash of every payload (redundant with the restored-byte check and the setup and end
    hashes, and it would double the I/O);
  - reopening the disk store between samples (it deletes files);
  - resetting production counters (falsifies accounting);
  - an independent recomputation of the reference (different provenance);
  - a new orchestration framework or freeze manifest.

## D43. (M5b preparation) Multi-candidate statistics: three separate quantities, a cross-evaluated difference and per-case simultaneous ambiguity
- **The three quantities** (`kvlab/m5_analysis.py`):
  - candidate selection quality (FixedChunks latency of the selected candidate);
  - delivered policy latency (the policy runs themselves);
  - prediction error.

  They are never substituted for one another.
- **The best measured candidate:** the argmin of the candidate means (ties go to the larger L). Naive regret
  against it is labelled a biased, descriptive estimate. The cross-evaluated difference (select on one session
  half, evaluate on the other, swap, weights ½ and ½) keeps its sign, and it is not called unbiased regret.
- **Ambiguity:** studentized-max simultaneous bootstrap intervals over all pairwise candidate differences of one
  case give the possibly-best set. A winner is never selected first and then tested against the runner-up.
- **Uncertainty:**
  - whole-repetition resampling, with circular blocks within each (case, session) stream;
  - sessions are fixed strata; the case grid is fixed (no claim beyond it);
  - equal case weights;
  - "inadequate" instead of an interval when the data are too few.
- **Primary practical report:** the paired delivered differences against AlwaysRestore and Recompute, in either
  direction. There are no tail, TTFT, matched-memory or cold-storage claims.

## D44. (M5) Publish the executed M5 evidence byte for byte
Context: 2026-11-05, after stages A–F of `docs/M5_PROTOCOL_DRAFT.md`.

**Completion note for D40–D43:** their "not run" status lines describe the state when they were written. The GPU
gate (at `9785c33`), the diagnostic pilot, the calibration, the fit, the held-out evaluation and the analysis
(at `a3c6341`) later each ran once (`docs/results.md`, M5). D40–D43 themselves are unchanged.

- **What is committed, byte for byte, under `results/m5/`:**
  - the four output directories written by stages C–F: `M5_CAL_V1_run`, `M5_FIT_V1`, `M5_EVAL_V1_run` and
    `M5_EVAL_V1_analysis`, 27 files in all;
  - `SHA256SUMS`, which lists their original digests.

  The files were force-added by explicit path. The `.gitignore` rules for these directories stay, so a new file
  there is still ignored.
- **Unchanged:**
  - no evidence file, recorded path or digest is edited;
  - no experiment is refrozen, and no validator is relaxed.

  The recorded interpreter path in the batch plans and records names the recording checkout, as in the M3 and
  M4 evidence.
- **Not published:**
  - the private records of the execution conditions;
  - the external GPU telemetry and the pilot;
  - the gate logs and the review material.

  The results cite them only as local records.
- **Verification:**
  - **Source binding:** the session files bind source `a3c6341` (recorded head, clean tree and per-file hashes).
    The later publication commits change no executable file.
  - **Offline reproduction:** the fit and the analysis are reproduced offline from the public package with that
    revision's own scripts (`docs/results.md`, M5).
  - **What it does not do:** it does not re-measure anything or re-establish the recording machine.
- **Integration:** the M5 commits were fast-forwarded into `main`, with no squash, rebase or rewrite.

## D45. (CI) Compare the archived M3 `ar1_effective_n` across CPUs within 4 ULPs
Context: 2026-11-07, a CPU CI run (Actions run 42458240422) whose values match a non-AVX-512 OpenBLAS kernel.

- **Problem:** two tests in `tests/test_m3_analyze_validation.py` rerun the M3 analysis on the preserved
  evidence and compared its JSON with the archived `m3_analysis.json` exactly. R3's `ar1_effective_n` comes
  from `np.dot` (`lag1_autocorr` in `kvlab/perf_stats.py`), and numpy's bundled OpenBLAS picks its dot
  kernel by CPU. The archive was written with the AVX-512 (SkylakeX) kernel; other kernels sum in a
  different order.
- **Observed** on the recording machine, with the kernel set by `OPENBLAS_CORETYPE` before each process
  (Haswell also with numpy's AVX-512 dispatch disabled):
  - with the Haswell, Sandybridge, Nehalem and Katmai kernels, 6 of the 40 `ar1_effective_n` values
    differ, each by exactly 1 ULP (absolute 1.42e-14, relative 1.5e-16 to 2.0e-16). Which 6 depends
    on the kernel;
  - every other archived value, and `samples_chronological.csv`, is unchanged. The comparison leaves
    out only the recorded paths and the fields the analysis has added since VAL-006, as the tests do.
    The CI runner's values equal the Haswell kernel's.
- **Decision:** those two comparisons accept each of the 40 `ar1_effective_n` values within 4 ULPs of
  the archived value. Each must still be present and a finite float. Everything else is compared
  exactly, as before: structure, keys, list lengths, all other numbers, verdicts, R3 pass flags, the
  decision, counts, the CSV bytes and the Markdown report.
  - 4 ULPs is the observed 1 ULP plus headroom for kernels that cannot be reproduced here.
  - At these magnitudes (71 to 113) that is under 6e-14. This is far from R3's `n_eff >= 10` threshold
    and far below the one decimal the report prints.
- **Unchanged:**
  - the analysis code, the frozen M3 files, the archived evidence and outputs, and every GPU check;
  - D12: storage correctness stays bitwise, and restored K/V and logits are still compared bitwise.

  The tolerance concerns cross-machine comparison of a derived statistic, not the cache's correctness
  contract.
- **Consequence:** on the recording machine, the regenerated analysis reproduces every archived value
  exactly, apart from the recorded paths and the fields added since VAL-006. That does not necessarily
  hold on every CPU: a value computed through BLAS can differ in its last bits elsewhere.
