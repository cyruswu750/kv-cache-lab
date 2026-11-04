# M1.1 Review — targeted fixes from the external M1 review

Date: 2026-11-03.

**Status: M1.1 done.**
- Both reproduced disk-tier defects are fixed, and host accounting is reported
  in separate, labeled figures.
- D10 is recorded as you decided it. Inference behavior is unchanged.
- All CPU and M0.1/M1 GPU tests pass.
- The original **M0 strict suite still FAILs** on `nat_long` (10 passed,
  1 failed). That is the recorded result under D10; nothing was changed to
  hide it.

No cost model and no new cache features. No commits, pushes, tags,
dependency changes or AI-attribution prose.

Defect records: `docs/BUGLOG.md` (BUG-001, BUG-002, RPT-003). Decisions:
`docs/decisions.md` D10, D19–D21.

## 1. Identity and preserved state
- Repo `/home/cyrus/projects/kv-cache-lab`, branch `main`, no commits, no remote.
- Pre-M1.1 snapshot: `backups/pre-m1_1-20261103T111458.tar.gz`, sha256
  `48cda896c4137b319115a90cf9b86b98fb6732e33a2fbd128408392740cc7fab`
  (git-ignored).
- Per-file hashes: `results/m1_1/pre_m1_1_manifest_20261103T111458.sha256`,
  100 files. Its own entry is meaningless, because it hashed itself while
  being written.
- Unchanged:
  - `results/m0/`;
  - `results/m0_1/`;
  - `results/m1/`, including your rerun `results/m1/manual_rerun_20261103_111038.json`;
  - every historical log.

  Verified against the pre-M1.1 manifest.
- Environment unchanged: torch 2.14.1+cu130, transformers 5.22.1,
  Python 3.14.4, RTX 4080 Laptop GPU, WSL2. Model
  `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA.

## 2. Issue 1: a failed disk deletion broke accounting and escaped the cache layer
**Reproduced** on the pre-fix tree (`results/m1_1/logs/repro_pre_fix_20261103T111458.log`,
and before/after in `results/m1_1/logs/repro_review_issues_pre_and_post_20261103T113339.log`):
- 8-byte budget, then `put(a, b"", b"1234")` (8 B file), then EACCES on unlink
  of a's file, then `put(b, …)`:
  - **pre-fix:** `PermissionError` escapes; `committed_bytes 0, used_bytes 0`
    while the 8 B file still exists;
  - **M1.1:** `(False, 'evict_oserror:EACCES')`; `used_bytes 8`; the file is
    still there.

**Cause.** `_remove()` popped the entry and subtracted its size before calling
`unlink`. Every caller (eviction in `put`, invalid-entry cleanup in
`read_into`, `clear`) let the `OSError` escape. The temp-file cleanup in `put`'s
`finally` had the same order: it released `tmp_bytes`, then unlinked, and a
failure there replaced the return value with an exception.

**Fix (`kvlab/disk_store.py`, decisions D19).**
- `_unlink(name, context)` is the only deletion path. It catches only
  `OSError`:
  - `FileNotFoundError` means gone;
  - any other `OSError` is recorded in `unlink_failures["<context>:<errno>"]`
    and returned as an errno name;
  - anything else, e.g. `ValueError` from the path checks, propagates.
- `_remove()` unlinks first. Bytes are released only if the file is gone.
  Otherwise the entry stops being readable and its bytes move to
  `undeletable[name]`, which:
  - stays in `used_bytes = committed + tmp + undeletable`;
  - counts against `max_entries`;
  - is never read;
  - blocks any rewrite of that key (`undeletable_file_for_key`).
- **Eviction:** a victim that cannot be deleted ends that `put()` with
  `(False, "evict_oserror:<errno>")`. In the tiered store this is a spill
  failure: the chunk is dropped and the request continues. The victim is
  not retried automatically; the next `put()` evicts the next LRU entry.
  If temp reservations + undeletable bytes leave no room (or no entry slot),
  `put()` returns `disk_budget_unavailable` before evicting anything, so
  valid entries are not destroyed for nothing.
- **Invalid entry:** if a bad entry's file cannot be deleted, the read is
  still a miss. The caller falls back to recomputation; the bytes stay
  accounted; the invalid data is never read again.
- **Temp file:** if cleanup fails, the full reserved size stays accounted as
  undeletable. That is conservative: the partial file is at most that size.
  A non-`OSError` from the writer still propagates.
- `clear()` makes one explicit attempt per undeletable file and returns the
  names it could not delete. There is no background service and no retry loop.
- **Startup:** a stale file that cannot be deleted makes construction fail
  with that `OSError`. This happens before any request, with nothing accounted.

**Regression tests.** These inject through `os.unlink` (filtered by file
name; `Path.unlink` goes through it too), `write_fn` and `os.replace`. No real
permissions are changed and no disk is filled.

| Test | Checks |
|---|---|
| `test_disk_store::test_eviction_unlink_failure_is_a_bounded_write_skip` | the review repro exactly; `used_bytes == 8 ==` directory bytes; not readable; two later puts do not retry the victim (`attempts == [a]`); `clear()` frees it once the fault is gone |
| `…eviction_failure_keeps_other_entries_and_moves_on_to_the_next_victim` | the other entry is still readable; the next put evicts it rather than the stuck file; directory bytes = used ≤ budget after each step |
| `…undeletable_files_count_against_max_entries` | no growth past `max_entries` |
| `…invalid_entry_cleanup_failure_is_a_safe_miss[flip_payload, read_eio]` | miss; bytes accounted; `disk_absent` on the next read; key not rewritten; the rest of the budget is usable |
| `…temp_cleanup_failure_keeps_the_reservation[enospc, replace_exdev, non_oserror]` | reason returned (`TypeError` propagates); reservation kept; later writes bounded by it |
| `…stale_file_that_cannot_be_deleted_aborts_construction` | documents the startup rule (the behavior was already the same before the fix) |
| `test_tiered_cpu::test_spill_whose_disk_eviction_fails_is_a_reported_write_skip` | `store()` completes; `spill_failures == {evict_oserror:EACCES: 1}`; `budgets()` reports it |
| `test_engine_cpu::*` (3; tiny random Qwen2 on CPU, no download) | for eviction unlink failure, invalid-entry cleanup failure, and temp cleanup failure after ENOSPC: **every request makes exactly STEPS forward calls**, the results are bitwise equal to the resident replay, and directory bytes ≤ used ≤ budget |
| `tests/gpu/test_m1_tiered.py::test_disk_cleanup_failure_neither_aborts_nor_repeats_inference[evict_unlink, invalid_cleanup]` | same on the real model; strict resident replay |

The GPU `Harness.run` now asserts exactly STEPS forward calls on **every**
request in the M1 GPU suite.

## 3. Issue 2: a marker symlink let the store overwrite a file outside the cache
**Reproduced** (same logs). With `MARKER` a symlink to an external "KEEP" file:
- **pre-fix:** construction succeeds. The external file now contains the
  marker text, and the existing cache file was already deleted.
- **M1.1:** `ValueError: … is not a regular file; refusing to use …`. The
  external file still reads "KEEP", and the cache file is still present.

**Cause.** `_open_root` checked names only, deleted stale files, and then
called `(root / MARKER).write_text(...)`, which follows symlinks. It also
rewrote the marker on every start, which would also write through a hard link.

**Fix (D20).**
1. `os.lstat(root)` must be a directory.
2. `os.scandir` checks, without following symlinks:
   - a non-empty root has a marker, and the marker is a **regular file**
     (`is_file(follow_symlinks=False)`);
   - every other entry matches our name pattern and is a regular file or a
     symlink.

   Any failure raises `ValueError` before anything is deleted or written.
3. Then stale files are unlinked (unlink does not follow symlinks).
4. A missing marker is created with `O_CREAT|O_EXCL|O_NOFOLLOW`; an existing
   marker is never rewritten.
5. Chunk reads open with `O_NOFOLLOW`. A symlinked chunk file is a miss
   (`ELOOP`) and the link is removed.
6. `scan_dir_bytes` uses `lstat` sizes.

This is not a defence against concurrent or adversarial directory changes
(TOCTOU). The cache is single-process with a trusted directory.

**Regression tests.**
- `test_disk_store::test_marker_symlink_is_refused_before_anything_is_written_or_deleted[existing, dangling]`:
  - the external file is still "KEEP", and for the dangling case nothing is
    created;
  - the link is unchanged;
  - the existing `.kvc` and `.tmp` files are byte-identical;
  - the listing is unchanged.
- `…non_regular_marker_or_entry_is_refused_and_nothing_is_deleted`: the marker
  is a directory, or a cache-named entry is a directory; nothing is deleted.
- `…an_existing_marker_is_never_rewritten_and_a_new_one_is_a_plain_file`: the
  marker is a hard link to an outside "KEEP" file and is not rewritten.
- `…symlinked_chunk_file_is_not_followed_on_read`.

## 4. Issue 3: host memory accounting
**Before.** `HostStore.used_bytes = resident + staging`, and `peak_bytes` was
its maximum. The M1 evidence reported host "used/peak 22,020,096", which is
just the slab size once both slots were occupied. The slab itself is
allocated in full at construction.

**Now (D21).** The ambiguous `used_bytes` / `peak_bytes` were removed from
`HostStore`. `budgets()["host"]` reports separately:

| Key | Meaning | Sequence scenario (`results/m1_1/m1_run_20261103T112930.json`) |
|---|---|---|
| `slab_allocated_bytes` | requested size of the one slab, allocated at init and constant until `close()` | 22,020,096 |
| `cuda_host_registered_bytes` | bytes passed to a successful `cudaHostRegister` (0 with `pin=False`) | 22,020,096 |
| `staging_reserved_bytes` | part of the slab reserved as the staging buffer | 7,340,032 |
| `resident_logical_bytes` | entries × chunk_bytes, i.e. occupancy **inside** the slab, not extra memory | 14,680,064 |
| `resident_logical_peak_bytes` | maximum of the above | 14,680,064 |
| `metadata_bytes_estimate` + `metadata_note` | **estimate** of raw identity field bytes; not measured Python heap usage | 2,256 (estimate) |

None of these is an RSS or page-lock measurement.

The per-request record field `host_used_bytes` became
`host_resident_logical_bytes`. Disk budgets now also report `used_bytes`,
`peak_used_bytes`, `undeletable_bytes`, `undeletable_files` and
`unlink_failures`. M1 evidence files keep the old keys.

## 5. Issue 4: D10 recorded; inference unchanged
`docs/decisions.md` D10 now records your decision verbatim:
- standard argmax;
- the original M0 strict test and result are preserved;
- no approximate-tie rule, rounding, tolerance inflation or hidden xfail;
- storage correctness is proven by byte identity and by strict comparison
  with the resident-GPU reference under matched computation provenance;
- full-vs-split numerical differences and greedy divergences remain
  separately reported limitations.

`kvlab/metrics.py`, `kvlab/tolerances.py`, `kvlab/model.py` and
`tests/gpu/test_gpu_roundtrip.py` were not touched. The docs state explicitly
that restored output is **not** claimed to equal full-prefill output, and that
numerical differences are **not** claimed to be harmless in general. The same
note was added to design §9 and the README, plus a dated pointer in
M0_1_REVIEW and M1_REVIEW.

## 6. Commands run (in order) and exit codes
Logs are under `results/m1_1/logs/`.

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | snapshot: `tar … backups/pre-m1_1-20261103T111458.tar.gz` + `sha256sum` manifest | 0 | 100 files hashed |
| 2 | review repro against the unmodified tree (scratch script; later saved as `scripts/m1_1_repro_review_issues.py`) | 0 | both issues reproduced (`repro_pre_fix_20261103T111458.log`) |
| 3 | tiny random Qwen2 on CPU, `prefill` + `greedy_continue` (feasibility for the engine test) | 0 | works, no download |
| 4 | `.venv/bin/python -m pytest -q` after the host API change, before updating tests | **1** | 284 passed, 4 failed: the 4 tests that used the removed `used_bytes`/`peak_bytes`; moved to the new fields with constraints at least as strict, then 288 passed |
| 5 | `pytest -q tests/test_disk_store.py` | 0 | 28 passed |
| 6 | `pytest -q tests/test_tiered_cpu.py tests/test_engine_cpu.py` | **1**, then 0 | first run: 1 failure from a **test-setup error** (the flipped byte fell inside the tiny file's namespace metadata, so the miss was `header_crc`); flip moved into the payload, then 15 passed |
| 7 | `pytest -q` | 0 | 306 passed |
| 8 | new CPU tests against the **pre-fix** `kvlab/` extracted from the snapshot | **1** (expected) | first: 20 failed / 32 passed, two of them only because a test used a new constant; tests changed not to depend on it, then **19 failed / 33 passed** (`regressions_vs_pre_fix_20261103T111458.log`) |
| 9 | `scripts/m1_mutation_check.py <scratch>` | 0 | **16/16 KILLED** (9 earlier, one updated for the new temp-cleanup code; 7 new) (`results/m1_1/mutation_check_20261103T112602.txt`) |
| 10 | `HF_HUB_OFFLINE=1 pytest -m "gpu and m1" -v` | 0 | **12 passed** (10 earlier + 2 new) |
| 11 | `HF_HUB_OFFLINE=1 pytest -m "gpu and m0_1" -v` | 0 | 6 passed |
| 12 | `HF_HUB_OFFLINE=1 pytest -m "gpu and m0_strict" -v` | **1** (expected) | 10 passed, **1 failed** (`nat_long`), unchanged |
| 13 | `HF_HUB_OFFLINE=1 scripts/m1_run.py --out results/m1_1/m1_run_20261103T112930.json` | 0 | 19 requests, `strict_pass_all: true`. **Correction (§13):** only 18 had strict request records |
| 14 | `pytest -q` (logged) | 0 | **306 passed**, 29 deselected (`pytest_cpu_20261103T112801.log`) |
| 15 | `scripts/m1_1_repro_review_issues.py` on the pre-fix tree, then on the current tree | 0, 0 | before/after shown in §2–3 |
| 16 | final `pytest -q` after the doc edits (no code changed since #14) | 0 | 306 passed (`pytest_cpu_final_20261103T113545.log`) |

What failed on the pre-fix code (#8):
- 17 tests failed on behavior: `PermissionError` escaping (eviction, invalid
  entry, temp cleanup, tiered, engine ×3), `DID NOT RAISE ValueError` (marker
  symlink ×2), `IsADirectoryError` after stale files were already deleted, the
  marker rewritten through a hard link, and the symlinked chunk followed.
- 2 failed only because the new host accounting attributes and keys did not
  exist yet.
- `…stale_file_that_cannot_be_deleted_aborts_construction` passes on both
  versions (unchanged behavior).

New M1.1 mutants, all killed: budget released although the file remains;
temp reservation released although the temp file remains; unlink `OSError`
escapes; undeletable victim kept readable and retried; marker symlink
followed; chunk read follows symlinks; host peak not tracked.

## 7. Evidence rerun (`results/m1_1/m1_run_20261103T112930.json`)
> **Correction (2026-11-03, BUGLOG VAL-004):** this runner printed
> "19 requests" but produced strict request records for only **18**. The
> ENOSPC request had no strict record, and its resident replay was not
> compared. The count was `len(records) + 1`. The corrected run is in §13.
> The text below is unchanged; "19" in it means 18 strict records plus one
> unvalidated ENOSPC request, which recorded 8 forward calls.

- Compared with the M1 evidence `results/m1/m1_run_20261103T023956.json`, these
  are **identical** for all 18 records and all 6 counter sets: label, N,
  eligible L, L, sources, miss reason, stored chunks, strict pass, tiers after,
  report-only greedy and max_abs.
- Summary is the same: 19 requests, strict pass on all, 7 restored, 3 of 7
  greedy-identical to full recompute.
- The same 4 report-only divergences occur at step 4 (margins 0.000 / 0.125).
- ENOSPC scenario: 8 forward calls (expected 8), `spill_failures={write_oserror:ENOSPC: 2}`,
  empty directory afterwards.
- No unlink failures occur in this script (no new fault scenarios were added,
  so it stays the 19-request script). Every scenario reports
  `disk_files_left_after_clear: []`.

## 8. Changed files
- **Source:** `kvlab/disk_store.py` (issues 1–2), `kvlab/host_store.py` (issue 3),
  `kvlab/tiered_cache.py` (`budgets()` keys only).
- **Tests:**
  - modified: `tests/m1_toy.py` (`deny_unlink` helper),
    `tests/test_disk_store.py` (+14), `tests/test_host_store.py`,
    `tests/test_tiered_cpu.py` (+1), `tests/gpu/test_m1_tiered.py` (forward-call
    check in `Harness.run`, new accounting fields, +2);
  - new: `tests/test_engine_cpu.py` (+3).
- **Scripts:** `scripts/m1_run.py` (field rename, records the `clear()`
  result), `scripts/m1_mutation_check.py` (16 mutants); new
  `scripts/m1_1_repro_review_issues.py`.
- **Docs:**
  - `docs/decisions.md` (D10 decided, D19–D21), `docs/design.md` (§6, §7, §9),
    `docs/correctness-matrix.md`, `README.md`;
  - `docs/M1_WALKTHROUGH.md` (Chinese; only the changed behavior);
  - dated pointer notes in `docs/M1_REVIEW.md` and `docs/M0_1_REVIEW.md`;
  - new `docs/M1_1_REVIEW.md`.
- **Results:** new `results/m1_1/`, plus a hash manifest and the review
  archive hash (§10).

## 9. Remaining limitations
- An undeletable file reduces disk capacity for the rest of the process.
  Recovery is a caller-invoked `clear()` (one attempt per file); there is no
  background cleanup. Its key cannot be stored on disk again in that process.
- An undeletable temp file is accounted at its full reserved size, which may
  overstate the real partial size (the conservative direction).
- A stale file that cannot be deleted at startup aborts construction. There is
  no degraded mode.
- The directory checks assume a trusted, single-process directory:
  - there is no TOCTOU defence;
  - a FIFO or device created at a chunk path *during* a run is not
    specifically handled (at startup it is refused);
  - a hard-linked marker is accepted, but it is never written.
- Unlink failures were injected at `os.unlink`. Real permission errors were
  not produced, by instruction. With a real read-only directory, temp-file
  creation would usually fail too, which is a write skip.
- Host figures are requested and registered sizes, not RSS or page-lock
  measurements. WSL2 page-locking behavior was not measured. The metadata
  figure is an estimate.
- D10: full-prefill vs restored greedy output can differ in bf16 (4 of 7
  restored requests in `m1_run`, plus `nat_long`). This is reported and not
  gated, and it is not claimed to be harmless in general.
- Unchanged from M1 §8: single process; sequential; batch 1; one model
  configuration; no fsync; no promotion of disk hits; no write-through when
  the host has no slot; GPU restore peak about 2× the prefix; **no timings or
  performance claims**.

## 10. Review archive
- `backups/kvlab-review-m1_1.tar.gz`. It is local only and was not uploaded.
- Contents: source, tests, scripts, docs, configs, and all small results
  (`results/m0`, `m0_1`, `m0_reruns`, `m1`, `m1_1`).
- Excluded: `.git`, `.venv`, `.hf_cache` (weights), `.kvcache` (cache
  payloads), `backups/` (earlier archives), `__pycache__`, `.pytest_cache`.
  There are no credentials in the tree.
- Per-file hashes: `results/m1_1/post_m1_1_manifest.sha256`, written after
  the last edit to any listed file and included in the archive.
- Archive sha256: `results/m1_1/review_archive.sha256`, written after the
  archive was created, so it is not inside it.

## 11. Next acceptance commands
```bash
.venv/bin/python -m pytest -q                                                       # expect 306 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and (m0_1 or m1)" -v            # expect 18 passed
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_run.py --out results/m1_1/m1_run_rerun.json   # expect exit 0, strict_pass_all true
```
Optional:
- `.venv/bin/python scripts/m1_1_repro_review_issues.py .` shows the fixed
  behavior.
- `.venv/bin/python scripts/m1_mutation_check.py <empty scratch dir>`: expect
  16 KILLED.
- `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_strict" -v`:
  expect 10 passed, 1 failed (`nat_long`, kept by D10).

## 12. Documentation closure (after the runs above; no code or test changes)
- Added `docs/BUGLOG.md` with BUG-001 (disk deletion accounting), BUG-002
  (marker symlink) and RPT-003 (host reporting terms).
- Replaced the problem narratives in decisions D19–D21 with links to it. D10
  is unchanged.
- Added pointers here and in `README.md`.
- No tests were rerun for this step: only documentation changed. The results
  in §6–§7 are the real results from the runs listed there.
- The manifest and archive in §10 were regenerated afterwards so that they
  cover these edits.

## 13. Follow-up M1.2: evidence-runner validation fix (BUGLOG VAL-004)
An independent review found that `scripts/m1_run.py`:
- ran the ENOSPC request outside the strict per-request path;
- reported `records + 1` requests;
- accepted an ENOSPC forward count edited to 800;
- compared step logits only through `zip()`.

Fixed in the script only; no cache implementation, model, argmax or tolerance
change (D10). Details and test names are in BUGLOG VAL-004.

| Command | Exit | Result |
|---|---|---|
| `pytest -v tests/test_m1_run_validation.py` (first) | **1** | 18 passed, 1 failed: a wrong test assumption (the historical ENOSPC `records` key is `[]`, not absent); fixed (`results/m1_2/logs/pytest_m1_run_validation_20261103T121350.log`) |
| `pytest -v tests/test_m1_run_validation.py` | 0 | **19 passed** (`…_20261103T121409.log`) |
| `scripts/m1_mutation_check.py <scratch>` (with 5 runner mutants that reintroduce the old behaviors) | 0 | 20 KILLED, **1 SURVIVED** (`zip()`-only logits; `results/m1_2/mutation_check_20261103T121449.txt`); the truncation test was then tightened |
| `pytest -q tests/test_m1_run_validation.py`, then `scripts/m1_mutation_check.py` | 0, 0 | 19 passed; **21/21 KILLED** (`results/m1_2/mutation_check_20261103T121623.txt`) |
| `.venv/bin/python -m pytest -q` | 0 | **325 passed** (306 + 19), 29 deselected (`results/m1_2/logs/pytest_cpu_20261103T121734.log`) |
| `HF_HUB_OFFLINE=1 scripts/m1_run.py --out results/m1_2/m1_run_20261103T121800.json` | 0 | acceptance true; **19 requests, 19 strict records**; problems [] |
| `scripts/m1_run.py --validate` on: new run / scratch copy with ENOSPC forward_calls=800 / historical `results/m1_1/m1_run_20261103T112930.json` | 0 / **1** / **1** | new: accepted. Edited: the only problem is `forward_calls=800`. Historical: 18 records, no ENOSPC record, and pre-fix schema fields missing (`results/m1_2/logs/validate_cli_20261103T121800.log`) |

Corrected run (`results/m1_2/m1_run_20261103T121800.json`):
- The ENOSPC request now has a strict record:
  - forward calls **8 actual / 8 expected**, counted only around the engine
    request;
  - 8 tokens and 8 step logits on both the engine and reference sides;
  - tokens equal, step logits bitwise equal, strict PASS.
- Fault evidence: **2 ENOSPC injections** in `write_fn` =
  `spill_failures {write_oserror:ENOSPC: 2}`; `stored_chunks [0, 1, 2]` (all
  in host); the cache directory is empty afterwards.
- Every one of the 19 records shows 8/8 forward calls.
- The 18 records shared with the M1.1 run (L, sources, miss reasons, stored
  chunks, strict result, tiers, report-only greedy) and all counters are
  identical.
- Report-only vs full recompute is unchanged: 7 restored, 3 greedy-identical,
  and the same step-4 divergences.

Not rerun, because this change did not touch them: the GPU pytest suites
(`m0_strict`, `m0_1`, `m1`), whose results in §6 stand.

Remaining limits of the evidence runner:
- Its validation is specific to this fixed scenario list (`EXPECTED_RECORDS`,
  `EXPECTED_FAULT_MISS`). Adding a scenario means updating those tables.
- Forward calls are counted with a top-level `forward_pre_hook`. That counts
  model calls, not kernels or sub-module calls.
- `--validate` checks the recorded fields of a report. It cannot detect a
  report fabricated wholesale with consistent fields; the raw run log is the
  cross-check.
- Historical reports fail `--validate` because they predate the record
  schema. They remain valid evidence only for their 18 strict records.
