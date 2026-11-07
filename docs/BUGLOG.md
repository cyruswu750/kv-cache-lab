# Bug log

Real defects found in project code: what went wrong, why, how it was fixed, and the tests that now
catch it. The design decisions taken in response are in [decisions.md](decisions.md). Evidence paths
refer to `results/`, where the full logs are kept.

Prefixes: **BUG** functional defect in the cache; **RPT** reporting definition; **VAL** defect in an
evidence runner or analysis, which can make a result look stronger than its evidence. All entries are
FIXED.

Not recorded here: faults injected on purpose by tests and mutation checks, and the documented bf16
full-prefill divergence of `nat_long` (D10), which is a numerical limitation, not a defect.

| ID | Summary | Component |
|---|---|---|
| [VAL-006](#val-006-the-m3-analysis-inferred-the-experiment-from-the-files-present) | Analysis failed open: missing evidence could report `supports_adoption=true` | `scripts/m3_analyze.py` |
| [BUG-001](#bug-001-a-failed-disk-deletion-released-its-budget-and-escaped-the-cache-layer) | Failed unlink released disk budget and aborted a finished request | `kvlab/disk_store.py` |
| [BUG-002](#bug-002-a-symlinked-marker-let-the-disk-store-overwrite-a-file-outside-its-directory) | Symlinked marker let the store overwrite an outside file | `kvlab/disk_store.py` |
| [VAL-005](#val-005-m2-scoring-treated-clustered-samples-as-independent) | i.i.d. intervals over clustered samples overstated restore wins | `kvlab/perf.py`, `scripts/m2_score.py` |
| [VAL-004](#val-004-the-evidence-runner-reported-19-strict-requests-but-validated-18) | Runner reported 19 strict requests, validated 18 | `scripts/m1_run.py` |
| [RPT-003](#rpt-003-host-memory-figures-mixed-occupancy-with-allocated-memory) | Host memory figures mixed occupancy with allocation | `kvlab/host_store.py` |

**Earlier hardening (M0.1 audit, before the tiered cache existed):** three gaps in
`kvlab/kv_snapshot.py` (a restore could cast V to the first chunk's K dtype; `copy_` could broadcast a
`[1, 1, C, D]` chunk into `[1, 2, C, D]`; the snapshot comparison did not check the layer count) and one
audit-script indexing bug (fp32 dot products for one path read from the wrong `lm_head` call; fixed in
audit run 2, run 1 kept). None affected an M0 result. Each has a regression test.

---

## VAL-006: The M3 analysis inferred the experiment from the files present
- **Symptom:** an empty run directory exited 0 with `supports_adoption: true`. Removing every (1024, 1)
  sample from copies of the five sessions turned `supports_adoption` from false to true.
- **Impact:** a fail-open analysis: incomplete evidence could produce the strongest verdict. The
  historical M3 result was not affected: the preserved evidence passes the new validation, its
  reanalysis reproduces all 5,938 historical values exactly, and the verdict stays false.
- **Root cause:**
  - sessions were whatever `session_s*.json` files existed, and cases were the (L, S) values of the loaded
    samples, so a missing required case simply disappeared from the decision rule;
  - the correctness gate was `all(...)` over loaded sessions using stored summary flags (`all([])` is
    true), never recomputed from the detailed checks;
  - incomplete pairs were dropped silently.
  - **Follow-up, same defect:** the first fix validated measured samples but not retained warmups.
    Removing all 1,080 warmup rows, or substituting one warmup with a copy of another, still gave
    `valid`, exit 0.
- **Found:** in review of the M3 analysis; both reproductions confirmed locally with the pre-fix script.
- **Fix:** `kvlab/m3_validate.py` validates the run against an explicit spec derived from the frozen
  protocol and the recorded commands before any statistic is computed:
  - every planned session, case, block, repetition and (path, backend, mode) identity occurs exactly
    once; 900 block-warmup and 180 process-warmup records are matched against the frozen runner's plan;
  - acceptance is recomputed from the detailed strict, A/B and restore-memory records against
    spec-derived counts (5,400 samples, 50 strict and 50 A/B checks); stored flags are compared, never
    trusted;
  - `decide()` iterates over the spec's case list, so a missing case blocks adoption;
  - three statuses with distinct exits: `valid` 0, `invalid` 3, `correctness_failed` 4; the latter two
    write no analysis and never report adoption.

  Statistics, rules R1–R3 and all measurement code are unchanged.
- **Regression tests:** `tests/test_m3_analyze_validation.py` (42 cases at the fix, through the real
  CLI and as a CUDA-hidden subprocess on scratch copies of the preserved run): empty directory, the
  missing-(1024, 1) reproduction, missing/duplicate/unpaired samples, stale passing summaries over
  failed details, warmup removal and substitution, and a check that warmup timings never enter a
  statistic.
- **Evidence:** `results/m3/reanalysis_val006_20261103T200427/` and
  `results/m3/reanalysis_val006_warmups_20261103T202825/` (pre- and post-fix reproductions, the
  reanalysis, a 0-difference comparison with the historical outputs, and preservation checks).
- **Design consequence:** every later analysis (M3_CONFIRM_V1, M4, M5) validates against an explicit
  frozen spec first and fails closed.

## BUG-001: A failed disk deletion released its budget and escaped the cache layer
- **Symptom:** with `budget_bytes=8`, after an 8-byte entry whose `unlink` fails with `EACCES`, the next
  `put()` raised `PermissionError`, and `committed_bytes` read 0 while the file still existed.
- **Impact:** accounting understated disk use, so later writes could exceed the budget on disk; inside a
  host spill the exception escaped `TieredKVCache.store()` and aborted a request whose inference had
  already completed; a failed invalid-entry cleanup turned a miss into an exception. Not observed in any
  M1 run (none produced an unlink failure).
- **Root cause:** `_remove()` popped the entry and subtracted its size *before* `unlink`, and nothing
  handled the error. The temp-file cleanup in `put()`'s `finally` had the same order, so an unlink
  error replaced the return value with an exception.
- **Found:** in review of the M1 code; reproduced locally.
- **Fix (D19):** unlink first and release bytes only once the file is gone (missing counts as gone). An
  `OSError` makes the file *undeletable*: still accounted, never read, its key never rewritten, not
  retried automatically. Eviction failure is a reported write skip (`evict_oserror:<errno>`);
  invalid-entry cleanup failure stays a miss; a failed temp cleanup keeps its reservation. Only
  `OSError` is handled.
- **Regression tests:** `tests/test_disk_store.py` (eviction, `max_entries`, invalid-entry and
  temp-cleanup failures), `tests/test_tiered_cpu.py`, `tests/test_engine_cpu.py` (exactly STEPS
  forward calls per request), and on GPU
  `tests/gpu/test_m1_tiered.py::test_disk_cleanup_failure_neither_aborts_nor_repeats_inference`.
  The CPU tests fail on the pre-fix code.
- **Evidence:** `results/m1_1/logs/` (pre-fix reproduction, before/after runs of
  `scripts/m1_1_repro_review_issues.py`, the new tests failing on the pre-fix source).

## BUG-002: A symlinked marker let the disk store overwrite a file outside its directory
- **Symptom:** constructing a `DiskStore` on a directory whose `.kvlab-disk-cache-v1` marker was a
  symlink overwrote the symlink's target with the marker text, after deleting existing cache files.
- **Impact:** writes outside the owned directory. A hard-linked marker was rewritten the same way; a
  directory as the marker caused stale files to be deleted before construction failed. Chunk reads also
  followed a symlink at a chunk path (read only, still a miss). Not observed in M1 runs, which used
  fresh directories.
- **Root cause:** `_open_root` validated entry *names* only, deleted stale files, then rewrote the
  marker on every start with `Path.write_text`, which follows links. Entry types were never checked,
  and validation did not finish before the first deletion.
- **Found:** in review of the M1 code; reproduced locally.
- **Fix (D20):** validate the whole existing directory first without following symlinks (real
  directory, regular-file marker, every other entry ours by name and type), raising `ValueError` before
  anything is deleted or written; create a missing marker with `O_CREAT|O_EXCL|O_NOFOLLOW` and never
  rewrite it; open chunk files with `O_NOFOLLOW`. Trust boundary: one process, trusted directory, no
  TOCTOU defence.
- **Regression tests:** `tests/test_disk_store.py` (symlinked and dangling marker, non-regular entries,
  marker never rewritten, symlinked chunk not followed); they fail on the pre-fix code.
- **Evidence:** as BUG-001.

## VAL-005: M2 scoring treated clustered samples as independent
- **Symptom:** M2 reported "11 of 11 scorable held-out decisions correct, 0 ms penalty".
- **Impact:** overconfident wins and hidden penalties. The held-out restore − full series have lag-1
  autocorrelation up to +0.93 (effective sample sizes as low as 1–3 of 40). With block-bootstrap
  intervals two of the 11 wins (L = 1024, S = 64) become inconclusive; by the mean, recompute is
  established faster in all 6 held-out L = 512 decisions, where the frozen model chose restore (point
  penalty 92 ms; 110 ms over all 18 decisions). No measured sample was wrong; the medians, the
  correctness results and the restore direction at L = 2048 and 4096 are unaffected.
- **Root cause:** `perf.paired_difference` resampled per-repetition differences i.i.d.; `decisions`
  gave non-scorable ("tie") decisions no penalty, and `m2_score.dec_stats` summed penalties over
  scorable decisions only.
- **Found:** in the M2.1 measurement audit.
- **Fix (D27, reporting only):** circular moving-block bootstrap at block lengths 1/2/4/8 (a direction
  needs all to agree), "inconclusive" instead of "tie", and a mean-based point penalty with a status for
  every decision (`kvlab/perf_stats.py`, `scripts/m2_1_reanalyze.py`). `kvlab/perf.py` and the saved M2
  scores are unchanged, so the historical files keep what M2 reported.
- **Regression tests:** `tests/test_perf_stats.py`, including two tests on the preserved M2 evidence
  showing the original call `restore` becoming `inconclusive`, and the 0 ms total excluding a +33 ms case.
- **Evidence:** `results/m2_1/reanalysis/` (first reproduces every original M2 interval exactly).

## VAL-004: The evidence runner reported 19 strict requests but validated 18
- **Symptom:** M1 evidence said "19 requests, `strict_pass_all` true", but the ENOSPC request had no
  strict record; editing its recorded forward-call count from 8 to 800 left acceptance true.
- **Impact:** one request's tokens and logits were never compared with its resident replay. Inference
  was not repeated (the runs recorded 8 forward calls, as expected). The other 18 strict records remain
  valid evidence, and the historical files are kept unchanged.
- **Root cause:** the ENOSPC request called `engine.run()` directly and produced no record; the summary
  computed `requests = len(records) + 1` and acceptance over the records that existed; step logits were
  compared through `zip()`, which truncates silently; missing scenarios and faults that did not occur
  were not detected.
- **Found:** in review of the M1.1 changes.
- **Fix:** every request goes through one checked path with a real ENOSPC injection counter;
  `strict_checks()` first requires token and logit counts equal to STEPS; `validate_report()` re-derives
  acceptance from recorded fields (every scenario and record, the intended fault observed); exit 0 only
  if acceptance holds; `--validate <json>` re-checks a saved report on CPU.
- **Regression tests:** `tests/test_m1_run_validation.py` (19 cases, production functions and the real
  CLI exit code), including one on the preserved M1.1 JSON where the old rule accepts the 800 edit and the
  new one rejects it. Five mutants reintroducing the old behaviors are killed; the `zip()` mutant first
  survived, and the truncation test was tightened.
- **Evidence:** `results/m1_2/` (corrected run: 19 requests, 19 strict records, ENOSPC 8/8 forward
  calls; CLI and mutation logs).

## RPT-003: Host memory figures mixed occupancy with allocated memory
- **Symptom:** a reported host "observed peak" of 22,020,096 B was the slab size with both slots
  occupied, not a measurement.
- **Impact:** misleading evidence, no wrong behavior: the slab is allocated and registered once at
  construction, and never exceeded its budget. Before slots filled, `used_bytes` understated the memory
  actually held.
- **Root cause:** `used_bytes` meant occupied slots + staging and `peak_bytes` its maximum, so one number
  mixed logical occupancy with reserved memory; the metadata figure was not labelled an estimate.
- **Found:** in review of the M1 code.
- **Fix (D21):** separate `slab_allocated_bytes`, `cuda_host_registered_bytes`,
  `staging_reserved_bytes`, `resident_logical_bytes` and its peak; metadata reported as an estimate. The
  ambiguous properties were removed; historical evidence files keep their old keys.
- **Regression tests:** `tests/test_host_store.py::test_capacity_accounting_includes_staging`,
  `tests/test_tiered_cpu.py::test_budget_accounting_reported`, and per-request slab/budget checks in the
  GPU M1 suite.
