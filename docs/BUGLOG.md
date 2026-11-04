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

Earlier findings are recorded in `docs/M0_1_REVIEW.md`: three M0 hardening
gaps in `kv_snapshot.py`, and one audit-script indexing bug.

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
  - Details: `docs/M1_1_REVIEW.md` §2.

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
- **Evidence:** the same repro, pre-fix and final-run logs as BUG-001;
  `docs/M1_1_REVIEW.md` §3.

## RPT-003: Host memory figures mixed logical occupancy with allocated memory
- **Type:** reporting definition. There was no functional defect: the
  allocator never exceeded its budget.
- **Status:** FIXED in M1.1 (2026-11-03). Raised by external review of M1.
- **Component:** `kvlab/host_store.py` (`used_bytes`, `peak_bytes`),
  `kvlab/tiered_cache.py` (`budgets()`), `scripts/m1_run.py`
  (`host_used_bytes`), and the M1_REVIEW §6 wording.
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
  - M1_REVIEW §6 presented host "observed peak 22,020,096". That is the slab
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
    22,020,096, staging 7,340,032, resident 14,680,064);
  - `docs/M1_1_REVIEW.md` §4.

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
    - `results/m1/manual_rerun_20261103_111038.json` (your rerun, from the same pre-fix runner)

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
  - Summary: `docs/M1_1_REVIEW.md` §13.

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
- **Impact on historical claims** (`docs/M2_REVIEW.md`, qualified in its
  M2.1 note):
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
  - Summary: `docs/M2_1_REVIEW.md` §3.
