# Decisions

Each entry: decision, reason, and whether it deviates from `kv-cache-plan.md`.
`docs/design.md` (created in M1) is the authoritative design; this file records why.
Defects and their evidence are in `docs/BUGLOG.md`; entries here link to them
instead of repeating them.

## D1. HF Transformers + PyTorch as the M0 correctness reference harness
- **Decision:** M0 uses `transformers` `DynamicCache` with plain PyTorch
  forward calls. vLLM was not installed or evaluated.
- **Reason:** Your M0 instruction. HF lets us read and write per-layer K/V
  tensors directly and control positions and masks explicitly. That makes it a
  transparent reference for byte-exact and numerical checks.
- **Not decided:** the production integration. The plan's vLLM KV-connector
  option (time-boxed to 6–8 h) is still open for later.
- **Plan deviation:** the plan lists vLLM as the first choice for Week 1. You
  overrode that for M0.

## D2. dtype = bfloat16 (plan says fp16)
- **Decision:** load the model with an explicit `dtype=torch.bfloat16`.
- **Reason:** the pinned checkpoint's `config.json` declares
  `"torch_dtype": "bfloat16"`, and the GPU supports bf16 (`torch.cuda.is_bf16_supported()` = True).
  bf16 avoids fp16 overflow risk on a bf16-trained model. KV size per token
  is the same (2 bytes/element, 28 KiB/token), so the plan's memory figures
  still hold.
- **Plan deviation:** yes (fp16 → bf16). You can change it back with one
  constant (`kvlab/model.py: DTYPE`). If you change it, the tolerances must be
  recalibrated.

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
  acceptance fixtures. It is the plan's "partial prefill vs full prefill
  shape difference", and the script also measures full-vs-full repeat
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

## D10. DECIDED (2026-11-03, your decision, recorded in M1.1): no tie rule; standard argmax; separate contracts
**Decision (your words, recorded as given):**
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

## D11. M0.1 gate passed; D10 left open (decided later, see D10); M0 strict suite kept as-is
- The M0.1 audit (docs/M0_1_REVIEW.md) isolated the nat_long mismatch to a bf16
  rounding tie caused by execution shape. Restored and resident splits are
  bitwise equal.
- The M0 strict suite is unchanged and still FAILs. No tie rule was adopted,
  because D10 is still your decision.
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
- **Evidence** (`docs/M2_1_REVIEW.md` §4): in time-clustered regimes, the
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
