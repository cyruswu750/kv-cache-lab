# M4_EVAL_V2 protocol: M4-B trace evaluation with a request-scoped warm-disk preparation (approved for freeze 2026-11-05, decision D38)

> **Status: approved for freeze (2026-11-05, decision D38).**
> - Until that approval this document was a DRAFT (decision D36). The approval changes no experimental rule.
> - **Implementation:** reviewed, and committed in `2ec8151`..`29db781`.
> - **Calibration (D37):** one private session on the calibration traces (session seed 45001, no evaluation seed),
>   run on 2026-11-05.
>   - Its launcher passed: 197.6 s of a 600 s cap.
>   - Its evidence was valid under the V2 rules:
>     - 0 strict mismatches;
>     - all 488 C2 records `verified` or `empty`, with 0 prewarms;
>     - `drop_pagecache` unchanged in every snapshot.
>   - Its pre-declared ten-session projections were 1,976 s, 2,005 s and 2,021 s, below the 2,160 s planning
>     threshold.
>   - It is not performance evidence and not a guarantee. It never needed a prewarm, so that branch has run only in
>     CPU tests. A WSL `drop_caches` event occurred about 120 s after it ended.
>   - Its evidence is kept locally, not committed.
> - **Accepted uncertainty (D38):** the approximate workload mapping; the unmeasured overhead and variability between
>   sessions; possible ordinary reclaim and WSL `drop_pagecache` interference; and a formal batch that may be
>   invalid despite the calibration.
> - **Authorized (D38):** the freeze and its verification, at most ONE formal ten-session batch, and its frozen
>   validation and analysis. The freeze manifest is created from the commit that contains this revision.
> - **Not done when this revision was approved:** the freeze and the batch. No formal V2 session has run, and no V2
>   performance result exists.

This document amends `docs/M4B_PROTOCOL.md`, the frozen M4_EVAL_V1 protocol, which this branch keeps
byte-identical (SHA-256 `adae50484edd35cb758b48080fba180a30f7bf1be9375fe19c79b37c951aec6f`, commit `b858d9e`).
Every rule of that protocol applies to M4_EVAL_V2 unless a section below replaces it. A V2 freeze binds both
documents (`kvlab.m4_freeze.documents`).

## 1. Why there is a V2, and what it is not
**M4_EVAL_V1** was frozen at `b858d9e` (manifest SHA-256 `137220aa…25b2`) and executed exactly once, on
2026-11-04.
- **Execution:** 10/10 sessions exited 0 in 1,833 s of the 2,700 s deadline, with 0 strict-correctness
  mismatches.
- **Evidence: INVALID** under its §6 warm-disk rule. 12 of 4,480 C2 requests had a page-residency minimum below
  1.0 after the runner's one prewarm, in sessions e04, e05 and e09 (T0 10, T2 2).
- **Consequences:** the frozen analysis exited 3, and there is **no performance verdict**: no comparison is
  faster, slower, equivalent or inconclusive. The evidence stays as recorded; it is never re-analysed, in whole
  or by subsets of sessions, traces, configurations or requests.

**The later diagnosis** was one post-outcome replay of session e05. It did **not reproduce** the losses.
- WSL's `drop_caches` events are excluded for the V1 batch window.
- The cause remains **unresolved**. Ordinary page-cache reclaim racing V1's sequential, unpinned whole-tier check
  is a plausible hypothesis. It is not an established cause.

**What V2 changes.** V2 changes an experimental *preparation condition*: which files are made warm and
checked before each C2 request, and how. It is **not** a retroactive correction of V1, and it does not
reinterpret V1's evidence: V1 remains executed and invalid. V2 needs its own spec name (§10 of the V1 protocol),
its own documented reason (this section), owner approval and its own freeze.

**What was not used:** no V1 measurement selects anything in V2. V1's timings, its failure locations and the
diagnostic's latencies informed no parameter, configuration, seed, trace, comparison or threshold. V2 keeps all
of them as V1 had them.

## 2. What is unchanged
The spec `M4_EVAL_V2` (`kvlab/m4_spec.py`) equals `M4_EVAL_V1` except in four fields: `name`, `run_dir`
(`results/m4/M4_EVAL_V2_run`), `protocol_doc` (this file) and the new `warm_disk = "request_candidates"`
(tested: `tests/test_m4_v2_evidence_cpu.py`). In particular these are unchanged:
- the traces and every seed, including session seeds 44001–44010;
- the configurations, budgets, interleaving and per-request configuration order;
- R = 10, the 15 primary comparisons, the statistics and the claim forms;
- the timed region (D22), including key generation and lookup inside it;
- the strict ResidentReference check;
- every fault, placement, counter, storage, schema, coverage and exit rule;
- the 2,700 s deadline, the stopping rule and the single batch;
- `baseline` as the default backend.

`M4_EVAL_V1`'s spec dict, and so its `spec_sha256` (`442e13ec…b541`), are unchanged on this branch. V1's
evidence can be re-verified only from its frozen tree, commit `b858d9e`. This branch's HEAD and tree do not pass
V1's identity checks, and no such claim is made.

## 3. Warm-disk preparation (replaces V1 §6 "Warm disk only" and §8 item 4)
**Before every C2 request** (warmup and evaluation), untimed, the runner (`kvlab/m4_bench.py`, with
`kvlab/warm_disk.py`) does exactly this:
1. **Plan the candidate set** (`plan_disk_reads`): the disk files this request's restore will read on the normal
   path. The plan takes:
   - the prompt's chain keys;
   - the first-token rule: complete chunks only, with at least the final prompt token recomputed;
   - the leading run of eligible keys present in either tier; it stops at the first absent key;
   - host precedence for each key: a host entry is never a candidate;
   - the staging rule: with no staging buffer, the restore stops at the first disk chunk without reading it.
     The M4 budgets always provide one.

   The plan uses only dictionary membership and the key-to-file-name mapping. It does not:
   - lease, stage, read, validate or repair anything;
   - change an LRU order, a counter or a restore decision.
2. **Initial pass:** one `mincore` observation of every candidate, recording resident pages, pages and bytes.
3. **Prewarm:** every candidate observed below full residency is read once, whole, sequentially and unbuffered
   (V1's read loop), and the bytes read are recorded. **There is no retry.**
4. **Final pass:** one `mincore` observation of every candidate, whether or not anything was read.
5. Then the unchanged timed region starts.

**What sits between the final pass and the timer.** Only:
- the preparation's closing clock read, stored in its record, and the return;
- the forward-call count, the wall-clock read and (CUDA) the synchronize.

The record is assembled before the passes and filled in by them. Its outcome, phase labels, prewarm count,
measured interval and second counter snapshot are computed after the timed region ends
(`kvlab.warm_disk.complete`). In V2 the preparation runs after the C1 occupancy and allocator readouts; V1 ran
its whole-tier check before them. The preparation makes no device allocation, so A0 and P keep their meaning.

**The only claim this supports:**
> every candidate file of the request was observed fully resident in the one verification pass immediately
> before the request.

**What it does not claim:**
- that the candidate files were all resident at one instant: the observations are sequential, one per file;
- that they were resident at timer start or during the timed read: pages are not pinned, and reclaim can act
  after each observation;
- anything about cold-disk, storage-device or production-service performance (D26 unchanged);
- that a timed read was served from the page cache. A post-request page observation is deliberately not taken:
  it could not distinguish a cache hit from pages brought in by the read itself.

**Candidate set versus actual reads:**
- **Valid normal path:** the candidates equal, in order, the files the restore reads.
- **Invalid file or failed read:** the restore attempts that file (reading part or all of it) and stops. The
  candidates are then a conservative superset: they include the attempted file, and possibly later files that
  were never read. They never miss an attempted file.

  Such a stop is already a fault under V1 §8 (a miss reason other than `absent`), so the evidence is invalid
  anyway. Corruption is never repaired or excluded during preparation.

**An empty candidate set** (nothing to restore from disk: an empty tier, a first absent key, or a host-only
prefix) is recorded explicitly:
- outcome `empty`, with phases `not_applicable` and no file observed;
- its counter snapshots are still recorded;
- every other check applies as usual.

**Invalidation rule, declared in advance** (`kvlab/m4_validate.py`). The evidence is invalid (exit 3, no
statistics) if any C2 record:
- lacks a well-formed warm-disk record;
- has candidates that differ from the disk chunks of the restore predicted by the independent key-level model
  (`kvlab.trace_models.tiered_model`), named by the real chain keys of the session's namespace, with their chunk
  indices and plan counts;
- has a candidate that was:
  - not observed in either pass (an observation error);
  - below full residency initially but not read exactly once in full;
  - read although it was fully resident;
  - of the wrong size or page count;
  - below full residency in the final pass;
- has phase timestamps that are not in the order plan, initial pass, prewarms, final pass;
- records an outcome or phase labels that differ from those recomputed from its per-file entries.

The required outcome is `verified`, or `empty` with no candidates. **Preparation errors** (an `OSError` while
observing or reading) are recorded with their text, and the outcome becomes `error`, so the evidence is invalid.
Any other exception propagates and stops the session (V1 semantics).

**Records kept** in every C2 record's `residency` field:
- the plan counts and stop reason;
- the candidate file names;
- for each candidate: chunk index; initial and final observations (status, resident pages, pages, bytes,
  monotonic time); and the prewarm (status `not_needed`, `read` with bytes and times, `error`, or `skipped` after
  a failed initial observation);
- phase statuses, the prewarm count and the outcome;
- the tier's entry count, as context;
- `timer_start_minus_verified_ns`: from the end of the final pass to the timer start, measured on the same clock;
- kernel counter snapshots (`counters_before`, read before the plan; `counters_after_request`, read after the
  timed region ends).

## 4. Effects of the preparation on caches (disclosed, not corrected)
**Fewer checks than V1.** The `mincore` calls read no file data, but they cost CPU time and touch the page tables.
V2 checks only the candidates, usually fewer files than V1's whole tier.

**Prewarm reads change the page cache.** A prewarm brings the file's pages into the page cache and refreshes
their position in the page-cache LRU. It can push out other pages: other tier files, the model weights or
anything else. Each read passes about 7 MB through the CPU caches just before the request.

**Non-candidate files are left alone.** Files of the tier that are not candidates are neither checked nor
warmed; V1 warmed them. The page-cache state at timer start therefore differs from V1's procedure, and V2
latencies are **not procedurally comparable** with V1. That comparison is moot anyway: V1 has no valid result.

**The plan warms CPU caches.** The plan computes the prompt's chain keys (SHA-256) and looks them up in the two
tier dictionaries. Those objects may be warm in the CPU caches when the timed region recomputes and looks them
up. The timed work itself is unchanged and is not moved out of the timer.

## 5. Global page-cache interference (new)
**The rule, declared in advance and conservative.**
- **Batch counter reads:** `scripts/m4_eval.py run` reads `/proc/vmstat` `drop_pagecache` twice:
  - before the first launch, into `<run_dir>/batch_interference.json` (exclusive create);
  - after the batch wrapper returns or raises, rewriting the record atomically.
- **Session and request reads:** every session records snapshots at its start and end, and every C2 request
  records one before its plan and one after its timed region.
- **Invalid (exit 3)** if any of these holds:
  - the batch record is missing or malformed, or does not name this spec and `drop_pagecache`;
  - its two read times (wall clock) are missing, not finite or reversed, or do not bracket the batch: the start
    read must come at or before the first launch and every session start, and the end read at or after every
    session end. As with V1's wall-time checks, a backward clock step during the batch fails this check closed;
  - either batch read did not obtain the count (unreadable file, or no `drop_pagecache` line);
  - the end count is above the start count (a global page-cache drop during the batch);
  - the end count is below the start count (a counter reset or reboot);
  - any session or request snapshot lacks the count, or differs from the batch start value.

**What an unchanged count does not show:** that any timed read was warm. **What an increase does not show:** that
any particular candidate was evicted. Ordinary reclaim counters (kswapd and direct steals, refaults, allocation
stalls) and `MemAvailable` are recorded **as context only**. They do not decide validity, and they do not by
themselves explain a failure.

**Feasibility risk, known before any V2 run.** On this WSL2 host, WSL's init (`mini_init`) issued 18
`drop_caches: 3` events between boot and 2026-11-04 22:40 (about 6.7 h), at gaps of 42 to 4,291 s. None fell in
V1's 1,833 s batch window, and the nearest came about 120 s after it ended. Whether the events depend on idle
time is not established. Under this rule, one such event during a V2 batch invalidates it. Whether to keep this
rule unchanged, or to change the host configuration before a freeze, is an owner decision; this branch changes no
system setting.

## 6. Validity summary (amends V1 §8)
V1 §8's item 4 ("`resident_fraction_before` = 1.0 for every C2 request") is replaced by §3's warm-disk rule.
§5's interference rule is added. Every other V1 validity rule, correctness rule and exit code is unchanged.

## 7. Claims (amends V1 §12)
V1 §12's allowed form and excluded claims apply. The phrase "warm page cache" means §3's operational
definition. The analysis output states it with these limits (`kvlab.m4_analysis.claim_limits`):
- per-file point observations, not a simultaneous snapshot;
- no guarantee at timer start or during the timed read;
- no cold-NVMe, storage or production claim;
- not comparable with V1's whole-tier preparation, and not a correction of V1.

## 8. Implementation status (CPU only)
**Implemented:**
- `kvlab/warm_disk.py`: the planner, the measurement primitives, the preparation and the counters (new);
- `kvlab/m4_bench.py`: the spec-selected preparation and the V2 records;
- `kvlab/m4_spec.py`: `M4_EVAL_V2`, and `warm_disk_protocol`;
- `kvlab/m4_validate.py`: the V2 rules;
- `kvlab/m4_freeze.py`: `documents(spec)`;
- `kvlab/m4_analysis.py`: `claim_limits(spec)`;
- `scripts/m4_eval.py`: the batch interference record.

**Unchanged:** the restore path (`kvlab/tiered_cache.py`, `kvlab/always_restore.py`, `kvlab/disk_store.py`,
`kvlab/host_store.py`) and the V1 rule's code path.

**Tests:**
- `tests/test_warm_disk.py`: the planner against the reads the unchanged engine actually attempts; side-effect
  freedom; the preparation with scripted observations;
- `tests/test_m4_v2_evidence_cpu.py`: V2 sessions, validator tampers, the interference rule, and the spec and
  freeze binding.

**Done:** the one D37 calibration session (see the status note).

**Not done when this revision was approved (D38):** the freeze and the formal batch.

## 9. Before any V2 batch (each step needs separate owner approval)
1. Review and approve this protocol, including §5's feasibility risk. Then remove DRAFT from the first line.
   **Done:** D38.
2. Recommended: one bounded calibration-input session under the V2 code (no evaluation seed), to check the new
   records, the interference record and the runtime, before any freeze. It is no performance evidence.
   **Done:** D37, one session.
3. Freeze `M4_EVAL_V2` (D31 pattern) and verify it. **Authorized by D38.**
4. The single V2 batch, then the frozen analysis. **Authorized by D38:** at most one batch.
