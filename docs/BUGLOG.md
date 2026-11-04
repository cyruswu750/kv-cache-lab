# Bug log

Real defects found in project code, with root cause, fix, regression tests and
evidence. Design decisions taken in response are in `docs/decisions.md`; this
log covers what went wrong, not the rationale.

Not recorded here:
- Faults deliberately injected for testing. These include the mutants in
  `scripts/m1_mutation_check.py` and the I/O faults injected by tests. They
  are test instruments, not discovered bugs.
- Failures of the original M0 strict suite on `nat_long`. That is a bf16
  numerical limitation, kept as recorded under decisions D10, not a defect.

Earlier findings, from the M0.1 audit (fixed before M1, with regression tests):
- three hardening gaps in `kvlab/kv_snapshot.py`:
  - restore silently cast V to the first chunk's K dtype (now a per-chunk
    dtype/shape check);
  - `copy_` could silently broadcast a `[1, 1, C, D]` chunk into `[1, 2, C, D]`;
  - `compare_snapshot_to_cache` did not check the layer count;
- one audit-script indexing bug. In audit run 1, the fp32 dot products for path
  B were read from the wrong `lm_head` call. This was fixed in run 2, and the
  run-1 files are kept.

None of them affected an M0 result (`docs/results.md`, M0.1).

Status values: FIXED (fix + regression tests passing), OPEN.

---

## BUG-001: Failed disk deletion released its budget and escaped the cache layer
- **Status:** FIXED in M1.1 (2026-11-03). Found by external review of M1 and
  reproduced locally.
- **Component:** `kvlab/disk_store.py` (`DiskStore._remove`, the callers in
  `put` / `read_into` / `clear`, and the temp-file cleanup in `put`'s
  `finally`).
- **Trigger:** `os.unlink` raises an `OSError` other than
  `FileNotFoundError` (for example `EACCES`) for one of the store's own files
  during any of:
  - LRU eviction in `put()`, including a host spill;
  - removal of an invalid entry after a failed read;
  - `clear()`;
  - removal of a temp file after a failed write.

  Minimal repro: `budget_bytes=8`; `put(a, b"", b"1234")` writes an 8-byte
  file; unlink of a's file fails with `EACCES`; then `put(b, b"", b"5678")`.
- **Root cause:**
  - `_remove()` popped the entry and subtracted its size from
    `committed_bytes` *before* calling `unlink`, and nothing handled the
    unlink error.
  - The temp-file path had the same order: it subtracted `tmp_bytes`, then
    unlinked, inside `finally`, so an unlink error replaced the return value
    with an exception.
- **Impact (pre-fix):**
  - Accounting understated disk usage. After the repro, `committed_bytes` and
    `used_bytes` were 0 while the 8-byte file still existed. Later writes
    could therefore exceed the configured budget on disk.
  - The `PermissionError` propagated out of `put()`. Inside a host spill it
    escaped `TieredKVCache.store()`, aborting a request whose inference had
    already completed.
  - A failed invalid-entry cleanup turned a cache miss into an exception.
  - Not observed in any M1 run: none of them produced an unlink failure. It
    appears only under the injected fault.
- **Fix:** see decisions D19.
  - Unlink first; release bytes only once the file is gone (missing counts as
    gone).
  - An `OSError` moves the file to `undeletable`: still accounted, never read,
    key not rewritten, not retried automatically.
  - Eviction failure is a reported write skip (`evict_oserror:<errno>`).
    Invalid-entry cleanup failure is still a miss. A failed temp cleanup keeps
    its reservation.
  - Only `OSError` is handled; all other exceptions propagate.
- **Regression tests** (all PASS after the fix; the starred ones fail on the
  pre-fix code):
  - `tests/test_disk_store.py`:
    - `test_eviction_unlink_failure_is_a_bounded_write_skip`\* (the repro above)
    - `test_eviction_failure_keeps_other_entries_and_moves_on_to_the_next_victim`\*
    - `test_undeletable_files_count_against_max_entries`\*
    - `test_invalid_entry_cleanup_failure_is_a_safe_miss[flip_payload|read_eio]`\*
    - `test_temp_cleanup_failure_keeps_the_reservation[enospc|replace_exdev|non_oserror]`\*
    - `test_stale_file_that_cannot_be_deleted_aborts_construction`
      (documents the startup rule; passes on both versions)
  - `tests/test_tiered_cpu.py::test_spill_whose_disk_eviction_fails_is_a_reported_write_skip`\*
  - `tests/test_engine_cpu.py` (exactly STEPS forward calls per request):
    - `test_disk_eviction_unlink_failure_neither_aborts_nor_repeats_inference`\*
    - `test_invalid_entry_cleanup_failure_falls_back_without_repeating_inference`\*
    - `test_temp_cleanup_failure_after_enospc_neither_aborts_nor_repeats_inference`\*
  - `tests/gpu/test_m1_tiered.py::test_disk_cleanup_failure_neither_aborts_nor_repeats_inference[evict_unlink|invalid_cleanup]`
    (real model; strict resident replay)
- **Evidence:**
  - `results/m1_1/logs/repro_pre_fix_20261103T111458.log`: first local repro.
  - `results/m1_1/logs/repro_review_issues_pre_and_post_20261103T113339.log`:
    before and after, from `scripts/m1_1_repro_review_issues.py`.
  - `results/m1_1/logs/regressions_vs_pre_fix_20261103T111458.log`: the new
    tests failing on the pre-fix source.
  - `results/m1_1/logs/pytest_cpu_final_20261103T113545.log`: 306 passed.
  - `results/m1_1/logs/pytest_gpu_m1_20261103T112801.log`: 12 passed.

## BUG-002: A symlinked marker let the disk store overwrite a file outside its directory
- **Status:** FIXED in M1.1 (2026-11-03). Found by external review of M1 and
  reproduced locally.
- **Component:** `kvlab/disk_store.py` (`DiskStore._open_root`, and the chunk
  read in `read_into`).
- **Trigger:** constructing a `DiskStore` on an existing directory whose
  `.kvlab-disk-cache-v1` entry is a symlink. Repro: an external file
  containing "KEEP", a cache directory with one stale `.kvc` file, and
  `MARKER` → symlink to the external file.
- **Root cause:**
  - `_open_root` validated entry *names* only.
  - It then deleted stale files.
  - It then rewrote the marker on every start with
    `Path.write_text`, which follows symlinks and hard links.
  - No entry *types* were checked, and validation was not completed before
    the first deletion.
- **Impact (pre-fix):**
  - The external file was overwritten with the marker text, and existing
    cache files were deleted before that write.
  - The same rewrite went through a hard-linked marker; observed in the
    pre-fix run of
    `test_an_existing_marker_is_never_rewritten_and_a_new_one_is_a_plain_file`.
  - With a directory as the marker, stale files were deleted before the
    construction failed with `IsADirectoryError`.
  - Related lower-impact gap: chunk reads followed a symlink at a chunk path.
    The outside file was only read, and the result was still a miss
    (`disk_truncated`).
  - Not observed in M1 runs, which used fresh or own directories.
- **Fix:** see decisions D20.
  - Validate the whole existing directory first, without following symlinks:
    - the root is a real directory;
    - the marker is a regular file;
    - every other entry has our name pattern and is a file or a symlink.

    Any failure raises `ValueError` before anything is deleted or written.
  - Create a missing marker with `O_CREAT|O_EXCL|O_NOFOLLOW`; never rewrite
    an existing one.
  - Open chunk files with `O_NOFOLLOW`.
  - Trust boundary: single process, trusted directory, no TOCTOU defence.
- **Regression tests** (`tests/test_disk_store.py`; all PASS after the fix;
  starred ones fail on the pre-fix code):
  - `test_marker_symlink_is_refused_before_anything_is_written_or_deleted[existing|dangling]`\*
  - `test_non_regular_marker_or_entry_is_refused_and_nothing_is_deleted`\*
  - `test_an_existing_marker_is_never_rewritten_and_a_new_one_is_a_plain_file`\*
  - `test_symlinked_chunk_file_is_not_followed_on_read`\*
  - Unchanged and still passing: `test_refuses_foreign_directories_and_deletes_nothing`,
    `test_stale_cleanup_only_touches_own_files_and_never_follows_symlinks`.
- **Evidence:** the same repro, pre-fix and final-run logs as BUG-001.

## RPT-003: Host memory figures mixed logical occupancy with allocated memory
- **Type:** reporting definition. There was no functional defect: the
  allocator never exceeded its budget.
- **Status:** FIXED in M1.1 (2026-11-03). Raised by external review of M1.
- **Component:** `kvlab/host_store.py` (`used_bytes`, `peak_bytes`),
  `kvlab/tiered_cache.py` (`budgets()`), `scripts/m1_run.py`
  (`host_used_bytes`), and the wording of the M1 review report (§6).
- **Trigger:** any report of host usage.
- **Root cause:**
  - The whole slab is allocated, and registered with `cudaHostRegister`, at
    construction.
  - `used_bytes` meant occupied resident slots + staging, and `peak_bytes`
    was its maximum, so one number mixed logical occupancy with reserved
    memory.
  - The metadata figure was not clearly labeled as an estimate.
- **Impact:**
  - Misleading evidence, no wrong behavior.
  - The M1 review report (§6) presented host "observed peak 22,020,096". That is the slab
    size with both slots occupied, not a measurement.
  - Before the slots fill, `used_bytes` understated the memory actually held.
- **Fix:** see decisions D21.
  - Separate figures: `slab_allocated_bytes`, `cuda_host_registered_bytes`,
    `staging_reserved_bytes`, `resident_logical_bytes`,
    `resident_logical_peak_bytes`.
  - Metadata is reported as an estimate, with a note in `budgets()`.
  - The ambiguous properties were removed. Historical evidence files keep
    their old keys.
- **Regression tests:**
  - `tests/test_host_store.py::test_capacity_accounting_includes_staging`
  - `tests/test_tiered_cpu.py::test_budget_accounting_reported`
  - `tests/gpu/test_m1_tiered.py::test_host_hit_restoration` (registered
    bytes = slab when pinned)
  - Every GPU request, via `Harness.run`, checks slab ≤ budget and
    resident + staging ≤ slab.
- **Evidence:**
  - `results/m1/m1_run_20261103T023956.json` (old keys);
  - `results/m1_1/m1_run_20261103T112930.json` (new keys; sequence: slab
    22,020,096, staging 7,340,032, resident 14,680,064).

## VAL-004: Evidence runner reported 19 strict requests but validated only 18
- **Type:** validation/reporting defect in the evidence script. It is not
  evidence that inference was repeated: the historical runs recorded 8 forward
  calls for the ENOSPC request (expected 8).
- **Status:** FIXED in M1.2 (2026-11-03). M1.2 is a narrowly scoped follow-up
  to M1.1, not a feature milestone. Found by independent review of M1.1.
- **Component:** `scripts/m1_run.py` (ENOSPC scenario, per-request strict
  checks, summary and exit code). No cache implementation changed.
- **Trigger:** any `scripts/m1_run.py` run.
- **Root cause:**
  - The ENOSPC request called `engine.run()` directly. It discarded the
    resident-reference outputs and produced no request record, and its forward
    count was stored only as a scenario field that no check read.
  - The summary computed `"requests": len(records) + 1`, and set exit 0 /
    `strict_pass_all` from `all(strict_pass)` over the 18 records that existed.
  - Step logits were compared with `all(torch.equal …) for … zip(…)`, which
    truncates silently. Token and step counts were never checked against STEPS.
  - Missing scenarios or records, and fault scenarios whose fault did not
    occur, were not detected.
- **Impact:**
  - The historical runs reported "19 requests, strict_pass_all true", but only
    18 had strict request records. The ENOSPC request's tokens and logits were
    never compared with its resident replay.
  - Editing the recorded ENOSPC forward count from 8 to 800 left acceptance
    true.
  - The historical evidence files are kept unchanged:
    - `results/m1/m1_run_20261103T023850.json`
    - `results/m1/m1_run_20261103T023956.json`
    - `results/m1_1/m1_run_20261103T112930.json`
    - `results/m1/manual_rerun_20261103_111038.json` (a manual rerun with the same pre-fix runner)

    Their 18 strict records are still valid evidence for those requests.
- **Fix:**
  - Every request goes through `Run.request`, ENOSPC included, with the real
    ENOSPC `write_fn` injection and an injection counter. Model forward calls
    are counted only around `engine.run`, not around the resident replay or the
    report-only full recompute.
  - `strict_checks()` first checks that token and step-logit counts (engine and
    reference) equal STEPS. Then it requires exact token equality, bitwise
    step-logit equality and, if L > 0, the restored-K/V byte check, plus
    forward calls == STEPS.
  - `validate_report()` re-derives acceptance from the recorded fields: every
    expected scenario with its exact record count, every strict field true,
    counts == STEPS, and the intended fault observed (the expected miss reason,
    or ENOSPC injections == ENOSPC spill failures ≥ 1, with an empty
    directory).
  - `summarize()` derives `requests` and `strict_records` from the records.
    The exit code is `0` only if `acceptance` holds.
  - `--validate <json>` re-checks a saved report without loading the model.
  - D10 is unchanged: full recompute stays report-only.
- **Regression tests** (`tests/test_m1_run_validation.py`, 19 cases, CPU,
  using the production functions and the real `--validate` CLI exit code):
  - `test_valid_complete_report_passes_in_memory_and_after_json_round_trip`
  - `test_incorrect_enospc_forward_call_count_fails_including_exit_code`
  - `test_token_or_step_logit_mismatch_fails[token|step_logit_value|step_logit_shape]`
  - `test_truncated_token_or_logit_sequences_fail[both_tokens|engine_logits|ref_logits|both_logits]`
  - `test_missing_enospc_record_or_scenario_fails`
  - `test_fault_scenario_without_its_miss_fails[4 fault scenarios]`
  - `test_enospc_scenario_without_the_injected_fault_fails[no_injection|no_spill_failure|files_left]`
  - `test_restore_without_kv_byte_check_fails`
  - `test_historical_evidence_lacks_the_enospc_strict_record`: on the preserved
    M1.1 JSON, the old rule accepts the 800 edit and the new validation
    rejects it.

  **Gap detection:** five mutants reintroduce the old runner behaviors into
  the new script (`+ 1` count, `zip()`-only logits, forward calls not in
  acceptance, acceptance = strict_pass of present records, fault occurrence
  not checked). All are killed. The `zip()` mutant first SURVIVED; the
  truncation test was then tightened to check each comparison field.
- **Evidence:**
  - `results/m1_2/m1_run_20261103T121800.json` (+ `logs/m1_run_20261103T121800.log`):
    19 requests, 19 strict records, acceptance true, exit 0. ENOSPC record:
    forward_calls 8 / expected 8, 8 tokens and 8 step logits on both sides,
    strict PASS. 2 ENOSPC injections = 2 `write_oserror:ENOSPC` spill
    failures, directory empty afterwards. The 18 shared records and all
    counters are identical to `results/m1_1/m1_run_20261103T112930.json`.
  - `results/m1_2/logs/validate_cli_20261103T121800.log`: `--validate` exits
    0 on the new run, 1 on a scratch copy with forward_calls = 800, and 1 on
    the historical M1.1 JSON (18 records, no ENOSPC record, pre-fix schema).
  - `results/m1_2/logs/pytest_m1_run_validation_20261103T121350.log`: first
    run, 1 failure from a wrong test assumption.
  - `results/m1_2/logs/pytest_m1_run_validation_20261103T121409.log`: 19 passed.
  - `results/m1_2/mutation_check_20261103T121449.txt`: the `zip()` mutant SURVIVED.
  - `results/m1_2/mutation_check_20261103T121623.txt`: 21/21 KILLED.
  - `results/m1_2/logs/pytest_cpu_20261103T121734.log`: 325 passed.
  - Summary: `docs/results.md` (M1–M1.2).

---

## VAL-005: M2 scoring treated clustered repetitions as independent and summed penalties over scorable cases only
- **Type:** validation/scoring defect in `kvlab/perf.py` (`paired_difference`,
  `decisions`) and `scripts/m2_score.py` (`dec_stats`). No measured sample is
  wrong; the uncertainty attached to the samples, and the penalty summary,
  were.
- **Status:** FIXED for reporting in M2.1 (2026-11-03). The fix is additive
  (`kvlab/perf_stats.py`, `scripts/m2_1_reanalyze.py`). `perf.py` and the
  saved M2 scores are unchanged, so the historical files keep exactly what
  M2 reported. Found in the M2.1 audit.
- **Trigger:** any M2 score of a series whose slow and fast samples come in
  time-clustered regimes. That is every restore series on this platform.
- **Root cause:**
  - `paired_difference` resampled per-repetition differences i.i.d. The
    held-out restore − full series have lag-1 autocorrelation up to +0.93,
    and AR(1) effective sample sizes as low as 1–3 of 40.
  - `decisions` set the penalty to `None` for every non-scorable ("tie")
    decision, and `dec_stats` summed penalties over scorable decisions only.
    The reported "0 ms penalty" therefore excluded every decision whose
    interval contained 0, including some whose point estimates favored
    recompute.
- **Impact on historical claims** (the M2 results, qualified in
  `docs/results.md`, M2):
  - **Overconfident restore wins.** "11 of 11 scorable held-out decisions
    correct" rests on i.i.d. intervals. Two of the 11 (host and disk at
    L=1024, S=64) are inconclusive with block length 8. Their i.i.d. median
    CI was [−27.0, −26.0] ms for host; with blocks of 8 it is
    [−27.0, +30.2] ms.
  - **Hidden penalties.** "0 ms penalty" excluded 7 inconclusive decisions.
    By the mean (expected latency), recompute is established faster in all 6
    held-out L = 512 decisions, at every block length. The frozen model chose
    restore there, with a total point penalty of 92 ms (110 ms over all 18
    decisions).
  - **Never claimed:** "tie" was not offered as evidence of equivalence, but
    the word invited that reading. M2.1 uses "inconclusive".
  - **Unaffected:**
    - the freeze chain;
    - the correctness results;
    - the latency medians themselves;
    - the restore-favoring direction at L = 2048 and 4096 (stable at every
      block length, for the median and the mean).
- **Fix:**
  - circular moving-block bootstrap with block lengths 1/2/4/8;
  - a direction is established only if all block lengths agree;
  - "inconclusive" wording;
  - a point penalty (mean-based) with a status, for every decision;
  - policy totals against always-restore and always-recompute.
- **Regression tests** (`tests/test_perf_stats.py`):
  - `test_block_bootstrap_widens_the_interval_for_a_clustered_series`
  - `test_classification_never_reads_an_interval_containing_zero_as_a_winner`
  - `test_selection_penalty_metric_and_status`: an inconclusive case still
    reports its positive point penalty.
  - `test_m2_iid_interval_was_overconfident_for_a_clustered_held_out_series`:
    on the preserved M2 evidence, the M2 call says `restore` and a block
    length of 8 says `inconclusive`.
  - `test_m2_zero_penalty_excluded_cases_with_a_positive_point_penalty`:
    on the preserved `results.json`, the 0 ms total excludes host L=512 S=1,
    whose paired median is +33 ms.
- **Evidence:**
  - `results/m2_1/reanalysis/reanalysis.json` and `.md`, written by
    `scripts/m2_1_reanalyze.py`. The script first reproduces every original
    M2 interval exactly (`all_original_cis_reproduced: true`).
  - Summary: `docs/results.md` (M2).

---

## VAL-006: M3 analysis inferred the experiment from the files present, so missing evidence could report supports_adoption=true
- **Type:** validation/reporting defect in the M3 analysis script. It is not
  evidence of an incorrect cache restore. No measured sample, restore,
  cache, model or tolerance code is involved, and the preserved M3 evidence
  is complete (validated below), so the M3 verdict is unchanged.
- **Status:** FIXED (2026-11-03), as an analysis-only follow-up to M3. Found
  by external review. Both reproductions were confirmed locally with the
  pre-fix script.
  - **Follow-up, same day: warmup evidence.** The first fix still did not
    validate retained warmup records. This is part of the same defect, so it
    gets no new entry. It is now fixed; see the "Warmup evidence" bullets
    below.
- **Component:** `scripts/m3_analyze.py` (loading and decision rule); new
  `kvlab/m3_validate.py`.
- **Trigger:** a run directory that lacks a planned session, case, block,
  repetition or sample, or whose correctness/A-B records are missing or
  inconsistent with their summary flags.
- **Root cause:**
  - The experiment's contents were inferred from the data that happened to
    load: the sessions were whatever `session_s*.json` existed, and the cases
    were the (L, S) values of the loaded samples.
  - The decision rule iterated over those cases only. A missing required case
    could neither fail clause (3) nor be reported.
  - Gate (1) was `all(correctness_all and not failures and 0 exclusions)` over
    the loaded sessions, using the stored flags. `all([])` is true for no
    sessions, and the flags were never recomputed from the detailed checks.
  - Pairs were formed only where both halves existed. Incomplete pairs were
    dropped silently.
- **Impact:**
  - An empty directory exited 0 with `supports_adoption: true`.
  - Removing every (1024, 1) sample from copies of the five sessions turned
    `supports_adoption` from false to true. The only unmet required case
    simply disappeared from the decision.
  - **The historical M3 result is unaffected.** The preserved evidence passes
    the new validation. Its reanalysis reproduces all 5,938 historical values
    exactly, and the verdict stays false. `results/m3/eval_20261103T173632/`
    is byte-identical.
  - **Warmup evidence (follow-up).** The first fix checked measured
    identities only:
    - process-warmup records were checked only for their warmup flag;
    - block-warmup records were checked for flag and range, but never for
      completeness or duplicates.

    Reproduced from copies of the five sessions:
    - **All 1,080 warmup rows removed**, with only `meta.n_samples` updated:
      `status: valid`, no problems, acceptance true, and the analyzer exited 0.
    - **One block warmup, or one process warmup, replaced by a copy of another**
      (row count unchanged): also `valid`, exit 0.

    `supports_adoption` stayed false in all three only because (1024, 1) is
    inconclusive.
- **Fix:**
  - **Explicit spec.** `kvlab/m3_validate.py` holds the spec
    `m3_eval_20261103T173632`, taken from the frozen `docs/M3_PROTOCOL.md`
    and the recorded commands. It fixes:
    - 5 sessions and their seeds, and 5 cases;
    - 2 blocks per case, each 2 warmup + 12 measured repetitions;
    - the 9 path/backend/mode samples per repetition, and G = 4;
    - the git HEAD, and the files that executed the measurement;
    - the optional profile session's planned rows.
  - **Checks before any statistic:**
    - session files: none missing, unexpected or unreadable;
    - the recorded arguments and block plan match the spec;
    - each (block, rep, path, backend, mode) identity occurs exactly once;
      warmup flags and order positions are consistent; every measured sample
      passes the M2/M3 validity rules and has the fields the analysis reads;
    - exactly one strict-correctness, A/B and restore-memory record per case,
      with every detailed field present;
    - stored counters and flags match the recomputed values;
    - the source hashes agree across sessions.
  - **Acceptance is recomputed from the detailed fields,** against counts
    taken from the spec: 5,400 measured samples, 50 strict checks
    (session × case × backend) and 50 A/B checks (session × case × tier).
    Stored summary flags are compared, never trusted. An A/B record also has
    to show that both requests restored the whole prefix from the tier.
  - **Three statuses, each with its own exit code:**
    - `valid` → 0, analysis written. The verdict may still be inconclusive.
    - `invalid` → 3: no analysis is written; `m3_validation.json` lists the
      problems and has `supports_adoption: false`.
    - `correctness_failed` → 4: complete, consistent evidence with a failed
      check. As for `invalid`, no analysis is written and adoption is false.
    - Refusal stays exit 1 and usage errors exit 2.
  - **`decide()` iterates over the spec's case list.** A spec case absent from
    the analysis blocks adoption. The analysis also checks that every case has
    10 strata and 120 pairs per tier and mode.
  - **`--out-dir`** writes a reanalysis to a new location.
  - **Warmup evidence (follow-up).** The protocol keeps warmups (§4: "kept,
    flagged, never used"). Leaving them out of the statistics does not leave
    them out of the completeness check:
    - **Block warmups:** each (block, rep < 2, path, backend, mode) identity
      must occur exactly once, with consistent order positions. That is
      900 = 5 sessions × 10 blocks × 2 reps × 9 combinations, counted
      separately from the measured samples.
    - **Process warmups:** these follow the spec's plan, which matches the
      frozen runner (`sorted({min(cases), max(cases)})`, every combination
      twice). That gives 36 records per session: (256, 32) and (4096, 64)
      × 9 combinations × 2.
      - The historical records have no repetition ID, so they are matched by
        multiplicity per (case, combination) on the recorded fields. Nothing
        is invented or rewritten.
      - They must be flagged `warmup: true`, carry no block/rep/order_pos, and
        come before the first block sample.
    - **Rejected:** missing, duplicate, surplus, substituted or unexpected
      warmup records, wrong warmup or stage flags, and records lacking the
      fields that `samples_chronological.csv` lists. The result is
      `invalid`, exit 3, no adoption.
    - **Statistics unaffected:** warmup timings are still excluded from every
      estimate, interval and decision.
  - **Unchanged:** R1–R3, the bootstrap, its seeds and all numerical code.
    `kvlab/m3_stats.py`, `kvlab/m3_bench.py`, `kvlab/tiered_cache.py`,
    `scripts/m3_measure.py` and the protocol are byte-identical.
  - **Hardware:** validation needs neither CUDA nor model weights. It imports
    torch only for the per-sample rules.
- **Regression tests** (`tests/test_m3_analyze_validation.py`, 42 cases, CPU;
  28 from the first fix, 14 from the follow-up).
  They use the production validation and the real CLI, in-process and as a
  subprocess with CUDA hidden, on scratch copies of the preserved run:
  - `test_empty_directory_is_invalid_via_the_real_cli` (subprocess; exit 3)
  - `test_missing_required_case_1024_1_is_invalid_via_the_real_cli`: the
    exact reproduction (subprocess; exit 3)
  - `test_decision_is_driven_by_the_spec_case_list_not_by_the_loaded_cases`:
    on the historical analysis with (1024, 1) removed, the pre-fix rule gave
    true; `decide()` gives false.
  - `test_missing_unexpected_or_unreadable_session_is_invalid[3]`
  - `test_missing_duplicate_or_unpaired_samples_are_invalid[7]`: missing,
    duplicate, warmup flags both ways, repetition outside the plan, a pair
    partner failing the validity rules, a missing `t_wall`.
  - `test_missing_or_failed_detailed_checks_with_stale_passing_summaries_are_invalid[11]`:
    - missing correctness, A/B or restore-memory records, including empty
      lists;
    - missing detailed fields;
    - failed detailed fields under passing `strict_pass` / `strict_pass_all` /
      `pass` / `correctness_all`;
    - an A/B record of two cache misses;
    - a stored flag contradicting passing details.
  - `test_consistently_recorded_correctness_failure_is_reported_as_such`
    (exit 4)
  - `test_complete_valid_report_reproduces_the_historical_analysis`: exit 0;
    every historical value equal; the CSV byte-identical; one added md line.
  - `test_refuses_to_overwrite_and_writes_nothing`
  - `test_spec_is_the_frozen_protocol_and_the_recorded_commands` (also checks
    the process-warmup plan against the runner and the protocol)
  - Follow-up, `test_missing_substituted_or_misflagged_warmup_evidence_is_invalid[12]`:
    - all warmups removed with `n_samples` adjusted;
    - one block warmup missing, and one process warmup missing;
    - block- and process-warmup substitutions with an unchanged row count;
    - a process warmup replaced by a relabelled measured row;
    - a process warmup flagged as measured;
    - a process warmup given the block stage, and a block warmup given the
      process stage;
    - an unknown stage;
    - a process warmup after the block samples;
    - a malformed block warmup.
  - Follow-up, `test_all_warmups_removed_is_rejected_by_the_real_cli_subprocess`
    (exit 3; all 5,400 measured samples valid, which is not enough).
  - Follow-up, `test_warmup_timings_never_enter_the_statistics`: with
    warmup timings inflated 100×, the evidence is still valid, and every case
    statistic and the decision equal the historical analysis.
- **Evidence** (`results/m3/reanalysis_val006_20261103T200427/`):
  - `logs/repro_before_prefix_analyzer.log`: with the pre-fix script (sha256
    `0ff942f2…`, as recorded by sessions s2–s5), both the empty directory and
    the no-(1024, 1) copies exit 0 with `supports_adoption: true`. A complete
    control copy reproduces the historical outputs.
  - `logs/repro_after_fixed_analyzer.log`: the same inputs exit 3 with
    `supports_adoption: false`, and the control exits 0 with the historical
    verdict.
  - `analysis/` (`m3_validation.json`, `m3_analysis.{json,md}`,
    `samples_chronological.csv`): the reanalysis of the preserved run, which
    is valid: 5,400/5,400 samples, 50/50 strict checks, 50/50 A/B checks.
    `logs/compare_with_historical.log` shows 0 differences in 5,938 values,
    the CSV byte-identical, and the same 4 unmet (1024, 1) requirements.
  - `logs/pytest_m3_validation_*.log`: 42 passed. `logs/pytest_cpu_*.log`:
    436 passed, 57 deselected.
  - `logs/preservation_check.log`: every pre-existing file under `results/`
    is byte-identical to `pre_fix_results_manifest.sha256`.
  - Summary: `docs/results.md` (M3 analysis validation).
- **Evidence, warmup follow-up** (`results/m3/reanalysis_val006_warmups_20261103T202825/`):
  - `logs/repro_before.log`: with the first-fix validator (`97f4a282…`), the
    three warmup variants from `make_warmup_variants.py` were `valid`, exit 0.
  - `logs/repro_after.log`: the same inputs are `invalid`, exit 3,
    `supports_adoption: false`, and no analysis is written.
  - `analysis/`: the preserved run is valid, with 900/900 block warmups and
    180/180 process warmups. `logs/compare_with_historical.log` shows 0
    differences in 5,938 historical values, the CSV byte-identical, the
    verdict false from the same 4 (1024, 1) entries, and everything equal to
    the first reanalysis except the validation block.
  - Focused tests:
    - `logs/pytest_m3_analyze_validation_20261103T203038.log`: 41 passed, 1
      failed. An existing test expected the old wording "duplicate sample
      identity", now "duplicate measured sample identity"; the expected text
      was updated.
    - `logs/pytest_m3_analyze_validation_20261103T203111.log`: 56 passed (42
      validation + 14 statistics).
    - The full suite was not rerun for the follow-up.
  - `logs/preservation_check.log`: every pre-existing file under `results/`
    is byte-identical.
