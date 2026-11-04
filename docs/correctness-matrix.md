# Correctness matrix (M0, M0.1, M1, M1.1, M1.2, M2, M2.1)

Columns:
- **Layer:** CPU = CPU-only unit test with synthetic tensors and real temp files.
  GPU = real model on CUDA with real pinned memory and real files.
  INJ = fault injected at a controlled I/O boundary (`write_fn` / `readinto_fn` /
  monkeypatched `os.replace` / `os.unlink`), never by filling the real disk or
  changing real permissions.
- **Actual:**
  - M0–M1 rows: the 2026-11-03 M1 runs (`results/m1/logs/*_20261103T024027.log`).
  - Every row was re-run in M1.1 (`results/m1_1/logs/pytest_*_20261103T112801.log`) with the same results.
  - The M1.1 rows are new.

## Identity and format (CPU)
| Scenario | Injection | Expected | Actual | Test |
|---|---|---|---|---|
| Namespace digest / chain keys identical across processes and `PYTHONHASHSEED` values | 4 subprocesses | identical | PASS | `test_identity::test_digest_is_deterministic_across_processes_and_hash_seeds` |
| Every namespace field changes digest and keys | 15 fields | different | PASS | `test_identity::test_every_namespace_field_changes_digest_and_keys` |
| Same chunk tokens after different prefixes | — | different keys | PASS | `test_identity::test_same_chunk_tokens_after_different_prefixes_do_not_alias` |
| Non-canonical namespace values (NaN, inf, set, bytes, non-str key) | — | raise | PASS | `test_identity::test_canonical_json_rejects_non_canonical_values` |
| Lossless round trip bf16/fp16/fp32; NaN payload, ±0, ±inf, subnormal | — | bytes equal | PASS | `test_chunk_format::test_lossless_round_trip`, `…special_values…` |
| Truncation at every byte offset, and one trailing byte | all offsets | miss (`truncated`/`size_mismatch`) | PASS | `…truncation_at_every_offset…` |
| Every single-bit flip in the header region | all bits | miss | PASS | `…every_header_bit_flip…` |
| Payload / trailer bit flips | sampled | miss `payload_crc` | PASS | `…payload_and_trailer_bit_flips…` |
| Wrong index / parent / tokens / key; inconsistent key | — | `identity_mismatch:*` | PASS | `…wrong_identity…`, `…inconsistent_expected_identity…` |
| Wrong namespace (5 fields) | file written under other namespace | `namespace_mismatch:<field>` | PASS | `…wrong_namespace_is_a_diagnosed_miss` |
| Meta altered, digest kept (CRC fixed) | — | `namespace_mismatch:meta_inconsistent` | PASS | `…meta_inconsistent…` |
| Wrong layout (layers, heads, head_dim, dtype, chunk size) | — | `layout_mismatch:*` | PASS | `…wrong_layout_is_a_miss` |
| payload_len = 2^60, meta_len = 2^31, token_count = 2^31 | header edit | miss; tracemalloc peak < 64 KiB | PASS | `…huge_lengths_rejected_before_any_large_allocation` |
| Short writes / short reads | partial I/O | completed correctly; zero-progress write raises EIO | PASS | `…short_writes…`, `…short_reads…` |

## Host tier (CPU, `pin=False`; GPU runs use the pinned slab)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Slab allocated once (≤ budget), never resized; staging reserved; resident = logical occupancy inside the slab, with peak (M1.1) | figures reported separately; slab size/pointer constant | PASS | `test_host_store::test_capacity_accounting_includes_staging`, `test_tiered_cpu::test_budget_accounting_reported` |
| LRU follows leases | least recently leased is spilled first | PASS | `…lru_eviction_order_follows_leases` |
| Pinned entries are never evicted; all pinned | `host_all_pinned`, no growth | PASS | `…pinned_entries_are_never_evicted…`, `test_tiered_cpu::test_all_pinned_skips_write…` |
| Lease / staging released when an exception occurs | pins 0, error propagates | PASS | `…lease_released_on_exception`, `test_tiered_cpu::test_pins_released…` |
| Oversized chunk (> budget) | no allocation (0-byte slab), `oversized_for_host` | PASS | `…oversized_chunk_bypassed_without_allocation` |
| Spill raises a non-OSError | propagates; entry kept | PASS | `…spill_exception_propagates…` |

## Disk tier (CPU, real files in `tmp_path`)
| Scenario | Injection | Expected | Actual | Test |
|---|---|---|---|---|
| LRU eviction; directory bytes = committed ≤ budget | — | as stated | PASS | `test_disk_store::test_round_trip_and_lru_eviction` |
| Temp file counted during the write | `write_fn` observer | used ≤ budget while the temp file exists | PASS | `…temp_file_counts_against_budget…` |
| ENOSPC mid-write | INJ `write_fn` | `write_oserror:ENOSPC`; no files left; accounting restored | PASS | `…enospc_skips_write_and_cleans_temp` |
| `os.replace` fails (EXDEV) | INJ monkeypatch | temp removed | PASS | `…replace_failure_cleans_temp` |
| Non-OSError during write | INJ | propagates; temp removed | PASS | `…non_oserror_propagates…` |
| Missing / truncated / payload-flipped / header-flipped file | file edit | miss; entry + file removed | PASS | `…read_faults_are_misses_and_remove_entry[4]` |
| EIO on read | INJ `readinto_fn` | `disk_read_oserror:EIO` | PASS | `…read_oserror_is_a_miss` |
| Foreign or non-empty directory; symlink root | — | refuse; nothing deleted | PASS | `…refuses_foreign_directories…` |
| Stale cleanup with a symlink pointing outside, plus a sibling outside the root | — | only own files removed; targets intact | PASS | `…stale_cleanup_only_touches_own_files…` |
| Path validation (`..`, absolute, wrong pattern) | — | ValueError | PASS | `…child_paths_validated` |

## Disk tier, M1.1 regressions (external review); all of these FAIL on the pre-fix code
`results/m1_1/logs/regressions_vs_pre_fix_20261103T111458.log`: 19 fail on the
pre-fix source.
- 17 fail on behavior: `PermissionError` escapes, DID NOT RAISE, the marker is
  rewritten, or a symlink is followed.
- 2 fail only because the new host accounting attributes don't exist yet.

| Scenario | Injection | Expected | Actual | Test |
|---|---|---|---|---|
| Review repro: 8 B budget, eviction victim unlink → EACCES | INJ `os.unlink` | `evict_oserror:EACCES`; file kept and still 8 B in `used`; not readable; later puts do not retry it; `clear()` frees it | PASS | `test_disk_store::test_eviction_unlink_failure_is_a_bounded_write_skip` |
| Eviction failure with other entries | INJ | other entries readable; next put evicts the next LRU; dir bytes = used ≤ budget | PASS | `…eviction_failure_keeps_other_entries…` |
| Undeletable files count against `max_entries` | INJ | `disk_budget_unavailable`, no growth | PASS | `…undeletable_files_count_against_max_entries` |
| Invalid entry (payload CRC / EIO) whose file cannot be deleted | INJ | miss; bytes stay accounted; never read again; key not rewritten (`undeletable_file_for_key`); rest of budget usable | PASS | `…invalid_entry_cleanup_failure_is_a_safe_miss[2]` |
| Temp cleanup fails after ENOSPC / EXDEV / TypeError | INJ | reason returned (TypeError propagates); reservation kept as undeletable; later writes bounded by it | PASS | `…temp_cleanup_failure_keeps_the_reservation[3]` |
| Stale file not deletable at startup | INJ | construction raises; file untouched | PASS (same on pre-fix) | `…stale_file_that_cannot_be_deleted_aborts_construction` |
| Marker is a symlink (to an existing file / dangling) | — | `ValueError` before any write/delete; external file still "KEEP" / not created; cache files unchanged | PASS | `…marker_symlink_is_refused…[2]` |
| Marker is a directory; cache-named directory entry | — | refuse; nothing deleted | PASS | `…non_regular_marker_or_entry_is_refused…` |
| Existing marker hard-linked to an outside file | — | never rewritten (outside file still "KEEP") | PASS | `…an_existing_marker_is_never_rewritten…` |
| Chunk file replaced by a symlink | — | miss `disk_read_oserror:ELOOP`; link removed; target untouched | PASS | `…symlinked_chunk_file_is_not_followed_on_read` |
| Spill whose disk eviction fails (tiered) | INJ | `store()` completes; `spill_failures={evict_oserror:EACCES: 1}`; chunk 0 not readable | PASS | `test_tiered_cpu::test_spill_whose_disk_eviction_fails_is_a_reported_write_skip` |
| Engine + tiny random Qwen2 on CPU: eviction unlink failure / invalid-entry cleanup failure / temp cleanup failure after ENOSPC | INJ | every request exactly STEPS forward calls; bitwise equal to the resident replay; dir bytes ≤ used ≤ budget | PASS (3) | `test_engine_cpu::*` |

## Tiered logic (CPU)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Host round trip | bytes equal | PASS | `test_tiered_cpu::test_host_only_round_trip` |
| Eviction spills; tiers exclusive; no promotion on disk hit | as stated | PASS | `…eviction_spills_to_disk_and_tiers_are_exclusive` |
| Invalid middle chunk | only verified prefix used; later chunks untouched | PASS | `…invalid_middle_chunk_stops_contiguous_prefix` |
| Spill ENOSPC (INJ) | chunk dropped, reason recorded, temp cleaned | PASS | `…spill_failure_drops_chunk…` |
| Host identity mismatch | **raise** (in-process invariant) | PASS | `…host_identity_mismatch_raises` |
| Model cache with wrong dtype/layout | **raise** | PASS | `…wrong_cache_layout_raises…` |

## GPU integration (real model, pinned slab, real files)
Every row below is also checked strictly against a resident-GPU replay: all
8 step logits bitwise equal, tokens equal, restored K/V byte-identical.

| Scenario | Expected | Actual | Test / evidence |
|---|---|---|---|
| Namespace from the loaded model | bf16, 28×2×128, sdpa; payload 7,340,032 B | PASS | `test_m1_tiered::test_namespace_reflects_loaded_model` |
| Host hit | L=512, sources host,host | PASS | `…test_host_hit_restoration` |
| Forced host eviction → disk hit | L=512 [disk,host]; L=256 [disk] | PASS | `…test_forced_host_eviction_then_disk_hit` |
| Partial prefix and boundaries (N = 513, 512, 257, 256, 513, 778, 769) | L = 0, 256, 256, 0, 512, 512, 768 | PASS | `…test_partial_prefix_and_chunk_boundaries` |
| Corrupt middle chunk (1 payload bit) | L=256, `disk_payload_crc`, chunk recomputed and re-stored, next request L=768 | PASS | `…test_corrupt_middle_chunk…`; `m1_run` fault_corrupt_middle |
| Missing file | L=0, `disk_file_missing` | PASS | `…test_missing_file…`; `m1_run` |
| Incompatible namespace (same name, written under `eager`) | L=0, `disk_namespace_mismatch:attn_implementation` | PASS | `…test_incompatible_entry_is_a_miss[namespace]` |
| Incompatible layout (header says 27 layers) | L=0, `disk_layout_mismatch:num_layers` | PASS | `…[layout]` |
| 9-request shared-prefix sequence, host 2 slots, disk 2 files | host and disk hits; both tiers evict; budgets held | PASS | `…test_multi_request_sequence…`; `m1_run` sequence |
| ENOSPC on every spill (INJ) | exactly 8 forward calls (no recompute); 2 spill failures; no files | PASS | `…test_write_failure_does_not_repeat_inference`; `m1_run` fault_enospc (strict record and forward-count acceptance only since M1.2, VAL-004) |
| M1.1: eviction victim unlink fails (INJ); invalid-entry cleanup fails (INJ) | write skip / miss; undeletable bytes accounted; never read; next request L=0 / L=768 | PASS (2) | `…test_disk_cleanup_failure_neither_aborts_nor_repeats_inference[2]` |
| M1.1: every GPU request makes exactly STEPS forward calls | asserted in `Harness.run` for all 12 tests | PASS | `-m "gpu and m1"` |
| Mutants: LSB flip in restored K; K copied into V slot | strict check fails | KILLED (2/2) | `results/m1/gpu_mutation_check_*.txt` |

## M0 / M0.1 (unchanged criteria)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Original M0 strict suite (D7 criteria incl. greedy identical vs full) | as in M0 | **FAIL 1/11** (nat_long, bf16 tie) | `-m "gpu and m0_strict"` |
| Restored == resident split, bitwise, repeated | equal | PASS 3/3 | `-m "gpu and m0_1"` |
| Corrupted snapshot / wrong-prefix snapshot / position offset | detected by byte check / byte check / guard | PASS 3/3 | `-m "gpu and m0_1"` |

## Full recomputation (report only, not gated; known limitation; D10 decided: no tie rule)
In `m1_run`, 7 requests were restored. Greedy matched full recomputation on 3
of them. The other 4 come from 2 distinct prompts, and all diverge at greedy
step 4. At that step the full path's top-2 margin is 0.000 (exact tie) or
0.125 (1 bf16 ulp). The mechanism is the one documented in M0_1_REVIEW.
M1.1 rerun (`results/m1_1/m1_run_20261103T112930.json`): identical records and
counters, and the same 4/7 divergences.

## Evidence runner validation (CPU; M1.2, BUGLOG VAL-004)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Complete valid report (in memory and after JSON) | acceptance, exit 0, requests = strict records = 19 | PASS | `test_m1_run_validation::test_valid_complete_report…` |
| ENOSPC forward_calls edited to 800 (strict_pass left true) | rejected, `--validate` exit 1 | PASS | `…incorrect_enospc_forward_call_count…` |
| Token / step-logit value (1 ulp) / step-logit shape mismatch | rejected | PASS (3) | `…token_or_step_logit_mismatch_fails` |
| Truncated tokens or logits (engine, reference, both) | rejected; each comparison refuses it on its own | PASS (4) | `…truncated_token_or_logit_sequences_fail` |
| Missing ENOSPC record or scenario | rejected; requests derived = 18 | PASS | `…missing_enospc_record_or_scenario_fails` |
| Fault scenario without its intended fault (4 miss scenarios; ENOSPC no injection / no spill failure / files left) | rejected | PASS (7) | `…fault_scenario_without_its_miss_fails`, `…enospc_scenario_without_the_injected_fault_fails` |
| Restore without K/V byte check | rejected | PASS | `…restore_without_kv_byte_check_fails` |
| Historical M1.1 evidence: old rule accepts the 800 edit; new validation rejects it | as stated | PASS | `…historical_evidence_lacks_the_enospc_strict_record` |
| GPU run `results/m1_2/m1_run_20261103T121800.json` | 19 requests, 19 strict records, ENOSPC 8/8 forward calls, 2 injections = 2 spill failures | PASS (exit 0) | `scripts/m1_run.py` |

## M2 measurement bookkeeping and cost model (CPU; `tests/test_perf.py`, 18 tests)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| ns in, ms out; median/IQR/mean/slow share | exact values | PASS | `test_units_are_nanoseconds_in_and_milliseconds_out` |
| Calibration/evaluation split disjoint and covering the grid; S = 64 only held out; invalid cases rejected | as stated | PASS | `test_splits_are_disjoint…` |
| Expected tier, L and admissions per path | as stated | PASS | `test_expected_tier_restore_length_and_admissions` |
| Disk sample served from host, wrong L, skipped admission, non-integer/zero time, cold-cache flags | excluded with a reason | PASS (6) | `test_wrong_tier_length_or_timing_is_excluded_with_a_reason` |
| Missing samples / unpaired repetitions | no statistics, `unknown` winner, fit refuses | PASS | `test_missing_samples_give_no_statistics_and_no_winner` |
| Paired winner needs a CI excluding 0 | restore / full / tie | PASS | `test_paired_winner_requires_the_ci_to_exclude_zero` |
| NNLS exact and non-negative | matches brute force | PASS | `test_nnls_small_matches_brute_force…` |
| Fit recovers known components; predicts held-out cases | exact on synthetic data | PASS | `test_cost_model_recovers_known_components…` |
| Fit refuses held-out rows; tampered model/predictions detected | raise / digest mismatch | PASS | `test_fit_refuses_held_out_cases…` |
| Decisions, penalties, ties; crossings only between reliable opposite winners | as stated | PASS | `test_decisions_score…`, `test_crossings_only…` |
| Engine `prefill_request(allow_restore=False)`, phase marks; `HostStore.discard` | 1 forward call; marks in order; pinned entries refused | PASS | `test_engine_cpu::test_prefill_request_without_restore_and_phase_marks`, `test_host_store::test_discard…` |

## M2 measured-path correctness (GPU; `results/m2/run_20261103T131543/`)
| Scenario | Expected | Actual |
|---|---|---|
| Every timed sample: tier, L, admissions; disk 100% / evicted 0% resident | all valid | PASS: 2,580 valid, 0 excluded |
| Per case: restored K/V bytes = source bytes (host, disk); first-token logits host == disk == resident split, bitwise | strict | PASS 15/15 |
| Full vs restored first token (report only, D10) | reported | 15/15 equal; max abs logit diff 0.39 |
| GPU M1 suite after the engine split | 12 passed | PASS |

## M2.1 statistics and diagnostic bookkeeping (CPU; `tests/test_perf_stats.py` 10, `tests/test_diag_bookkeeping.py` 16)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Paired series keeps only common repetitions, in order | as stated | PASS | `test_paired_series_keeps_only_common_reps_in_chronological_order` |
| Circular block indices are consecutive runs; block > n is clipped | as stated | PASS | `test_block_indices_are_circular_runs_of_consecutive_reps` |
| Clustered series: block CI wider than i.i.d.; too few blocks flagged | as stated | PASS | `test_block_bootstrap_widens_the_interval_for_a_clustered_series` |
| Too few pairs | no interval | PASS | `test_too_few_pairs_give_no_interval` |
| CI containing 0 | inconclusive | PASS | `test_classification_never_reads_an_interval_containing_zero_as_a_winner` |
| Lag-1 autocorrelation and AR(1) n_eff | as stated | PASS | `test_lag1_autocorrelation_and_effective_sample_size` |
| Penalty point estimate and status (incl. a positive point penalty on an inconclusive case) | as stated | PASS | `test_selection_penalty_metric_and_status` |
| Policy totals and oracle regret | exact | PASS | `test_policy_totals_and_oracle_regret` |
| Preserved M2 evidence: i.i.d. call `restore` vs block-8 `inconclusive` (host L=1024 S=64); "0 ms" excluded host L=512 S=1 | VAL-005 reproduced | PASS (2) | `test_m2_iid_interval_was_overconfident…`, `test_m2_zero_penalty_excluded…` |
| Diagnostic samples of every path × mode valid when well formed; `full_nocache` admits nothing; wrong tier, missing marks, negative event time, bad phases, cold disk excluded; probe work checks | as stated | PASS (16) | `tests/test_diag_bookkeeping.py` |

## M2.1 diagnostic-path correctness (GPU; `results/m2_1/diag_*`)
| Scenario | Expected | Actual |
|---|---|---|
| Every diagnostic sample validated (tier, L, admissions, residency, event marks, probe checks) | all valid | PASS: 0 excluded in all sessions |
| Per first-visit block: M2 strict check + `full_nocache` logits bitwise = `full` | strict | PASS 26/26 |
| Spike-probe cached forward starts at 256 and ends at 257 tokens; source K/V unchanged | as stated | PASS 10/10 sessions |
| M2 smoke after the `perf_harness` refactor | exit 0, strict | PASS (`results/m2_1/m2_smoke_after_refactor_20261103T152815/`) |
