# Correctness matrix (M0, M0.1, M1, M1.1, M1.2, M2, M2.1, M3)

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
0.125 (1 bf16 ulp). The mechanism is the one documented for M0.1 in `docs/results.md`.
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

## M3 restore backends, CPU differential (`tests/test_restore_backends_cpu.py`, 24 tests; twin stores, one per backend, identical history)
Each row compares the restore result, KV bytes (also against the source cache), counters,
tiers, host LRU order, pins, staging state and budgets between the backends.

| Scenario | Expected | Actual | Test |
|---|---|---|---|
| One chunk from host; one chunk from disk | identical; L = C; 1 placement | PASS | `test_one_chunk_host_and_disk` |
| 5 host chunks, G ∈ {1, 2, 3, 5, 8} | identical; batched copies = 5, placements = ⌈5/G⌉, scratch = min(5, G)·CB; baseline copies = 5·2·layers | PASS (5) | `test_multi_chunk_host_only_for_every_group_size` |
| Disk-only, 5 distinct payloads through the one staging slot | identical; no promotion | PASS | `test_disk_only_reads_reuse_one_staging_slot_with_distinct_payloads` |
| Mixed tiers (disk ×3, host ×2) | identical sources in stored order | PASS | `test_mixed_tiers_follow_the_stored_order` |
| Corrupt chunk at index 0 / 2 / 4 | L = index·C (0 → no cache), `disk_payload_crc`; only verified groups placed; later chunks untouched | PASS (3) | `test_corrupt_chunk_stops_at_the_verified_prefix` |
| Missing middle file; absent middle chunk | `disk_file_missing` at L = C; `absent` at L = 2C | PASS | `test_missing_middle_file_and_absent_middle_chunk` |
| Partial plans of 1–4 chunks (G = 3) | identical | PASS (4) | `test_partial_plans_and_chunk_boundaries` |
| Repeated requests with eviction; host slab overwritten afterwards | earlier returned caches unchanged | PASS | `test_repeated_requests_with_eviction_never_change_an_earlier_cache` |
| Returned tensors | contiguous, own exactly L tokens, no shared storage | PASS | `test_restored_tensors_own_exactly_their_bytes` |
| Copy failure at a disk chunk / a host chunk, per backend | error propagates; pins 0; staging free; hits count completed copies only; store reusable | PASS (4) | `test_copy_failure_releases_leases_and_staging_and_propagates` |
| Unknown backend, G < 1, invalid backend set later | `ValueError` | PASS | `test_backend_and_group_size_are_validated` |
| Engine sequences on a tiny Qwen2 (6 prompts, eviction, disk hits) | every step bitwise = resident replay; identical across backends | PASS | `test_engine_sequences_are_bitwise_identical_across_backends` |

## M3 GPU (`tests/gpu/test_m3_restore.py` 17; `tests/gpu/test_m1_tiered.py` parametrized over both backends)
| Scenario | Expected | Actual | Test / evidence |
|---|---|---|---|
| Every M1 GPU scenario with the batched backend | as in M1; strict resident replay; STEPS forward calls | PASS 11/11 (+11/11 baseline, +1 namespace) | `-m "gpu and m1"` (`results/m3/logs/pytest_gpu_m1_20261103T175241.log`) |
| A/B on one store, n ∈ {1, 3, 5, 6} × {host, disk, mixed} | KV bytes, logits, 8 greedy steps and tokens bitwise equal across backends and to the resident replay | PASS (12) | `test_batched_matches_baseline_on_identical_state` |
| Disk-only, 5 distinct payloads via the staging slot | bytes equal resident; 5 copies, 2 placements | PASS | `test_disk_reads_reuse_the_staging_slot_with_distinct_payloads` |
| Synchronizing copies for 6 chunks (torch sync debug mode) | baseline 336, batched 6 | PASS | `test_every_chunk_transfer_is_one_synchronizing_copy` |
| Returned cache after later requests, evictions and slab overwrite | unchanged; owns exactly [0, L) | PASS | `test_returned_cache_is_not_changed_by_later_requests_or_buffer_reuse` |
| Restore peak device memory | baseline ≤ 2n·CB (+2 MiB), batched ≤ (2n + min(n, G))·CB (+2 MiB); kept = n·CB | PASS | `test_transient_device_memory_is_bounded` |
| Copy failure on GPU | leases and staging released; next request restores fully | PASS | `test_copy_failure_on_gpu_releases_leases_and_staging` |
| M1 evidence runner with each backend | 19/19 strict, acceptance; records and counters identical across backends | PASS | `results/m3/m1_run_{baseline,batched}_20261103T175241.json` |
| Mutants (8 CPU, 2 GPU) | each killed | 10/10 KILLED | `results/m3/mutation_check_20261103T173143.txt` |
| M3 sessions: every timed sample (tier, L, admissions, residency, requested backend, copy count) | valid | PASS: 5,400 measured, 0 excluded | `results/m3/eval_20261103T173632/` |
| M3 sessions: M2 strict check per backend; A/B check (KV bytes, logits, 8 steps) | strict | PASS 25/25 and 25/25 | same |
| Full vs restored first token (report only, D10) | reported | 25/25 equal; max abs logit diff 0.20–0.39, identical for both backends | same |

## M3 statistics (`tests/test_m3_stats.py`, 14 tests)
| Scenario | Expected | Actual |
|---|---|---|
| Stratified indices stay within their stratum | as stated | PASS |
| Point estimates of the 5 statistics, incl. marginal vs paired ratios differing | exact | PASS |
| R1 / R2 / R3 each able to force "inconclusive"; "slower" detected | as stated | PASS |
| Regime durations | exact | PASS |
| Well-formed samples of all 9 path × backend × mode combinations valid; wrong backend, wrong copy count, wrong chunk count, missing record, restore on `full`, bad phases, wrong tier excluded | as stated | PASS (9 + 1) |

## M3 analysis evidence validation (CPU; BUGLOG VAL-006, `tests/test_m3_analyze_validation.py`, 42 tests)
Added after M3. These run the production validator (`kvlab/m3_validate.py`) and the real `scripts/m3_analyze.py` CLI on scratch copies of `results/m3/eval_20261103T173632/`. That directory is only read.

| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Spec equals the frozen protocol, the recorded commands and the runner's process-warmup plan; harness combinations equal | as stated | PASS | `test_spec_is_the_frozen_protocol_and_the_recorded_commands` |
| Empty directory (pre-fix: exit 0, adoption true) | `invalid`, exit 3, no analysis | PASS | `test_empty_directory_is_invalid_via_the_real_cli` |
| All (1024, 1) samples removed (pre-fix: adoption false → true) | `invalid`, exit 3 | PASS | `test_missing_required_case_1024_1_is_invalid_via_the_real_cli` |
| Decision on the historical cases with (1024, 1) removed, and on no cases | adoption false; the rule can still pass on valid all-faster input | PASS | `test_decision_is_driven_by_the_spec_case_list…` |
| Missing / unexpected / unreadable session file | `invalid` | PASS (3) | `test_missing_unexpected_or_unreadable_session_is_invalid` |
| Measured samples: missing, duplicate, warmup flag either way, rep outside the plan, pair partner failing the rules, missing `t_wall` | `invalid` | PASS (7) | `test_missing_duplicate_or_unpaired_samples_are_invalid` |
| Correctness / A/B / restore-memory records or fields missing or failed while stored summaries still pass; A/B of two misses; summary contradicting passing details | `invalid` (failed details also listed) | PASS (11) | `test_missing_or_failed_detailed_checks_with_stale_passing_summaries…` |
| Correctness failure recorded consistently | `correctness_failed`, exit 4 | PASS | `test_consistently_recorded_correctness_failure…` |
| Warmups (follow-up): all removed with `n_samples` adjusted; one block / one process warmup missing; same-count substitutions; relabelled measured row; flag or stage errors; out-of-order process warmup; malformed warmup | `invalid`, exit 3 | PASS (12 + 1 subprocess) | `test_missing_substituted_or_misflagged_warmup_evidence_is_invalid`, `test_all_warmups_removed_is_rejected_by_the_real_cli_subprocess` |
| Warmup timings inflated 100× | valid; every case statistic and the decision unchanged | PASS | `test_warmup_timings_never_enter_the_statistics` |
| Complete original evidence | `valid`, exit 0: 5,400 samples, 900 block and 180 process warmups, 50 strict and 50 A/B checks; all historical values equal; adoption false | PASS | `test_complete_valid_report_reproduces_the_historical_analysis` |
| Existing output, or a missing run directory | refused; nothing written | PASS | `test_refuses_to_overwrite_and_writes_nothing` |

## M4-A trace replay (CPU only; `tests/test_traces.py` 30 tests, `tests/test_trace_replay_cpu.py` 50 tests; tiny random model; correctness, not performance)
| Scenario | Expected | Actual | Test |
|---|---|---|---|
| Each generator: same seed gives the same trace and digest, another seed a different one; tokens in range; labeled synthetic | as stated | PASS (4) | `test_generation_is_deterministic_and_seed_sensitive` |
| JSONL round trip; no overwrite; tampered token or missing request | equal; refused; rejected | PASS (4) | `test_dump_load_round_trip_refuses_overwrite_and_detects_tampering` |
| Structures: T0 distinct first tokens; T1 shares exactly the prefix; T2 turn-major, each turn extends the last, conversations disjoint; T3 same document vs instruction only | as stated | PASS | `test_generator_structures` |
| Oracle reuse vs a brute-force common-prefix oracle, 4 traces × C ∈ {1, 2, 4, 8} | equal | PASS (16) | `test_oracle_reuse_equals_a_brute_force_prefix_oracle` |
| No-reuse control vs reuse traces | zero reuse vs some | PASS | `test_no_reuse_control_has_zero_reuse_and_reuse_traces_have_some` |
| First-token rule at exact chunk multiples; hand-worked LRU sequences (capacities 0, 2, 3, unbounded; storing does not refresh present chunks) | as hand-computed | PASS (2) | `test_first_token_rule_at_exact_chunk_multiples`, `test_lru_restore_lengths_on_hand_worked_sequences` |
| Working set, capacities, ρ; invalid generator parameters | as computed; refused | PASS (2) | `test_stats_working_set_budgets_and_rho`, `test_generators_reject_inconsistent_parameters` |
| C0, C1 (ample, 2 chunks), C2 (baseline, batched; host 2 slots spilling to disk) × 4 traces: every request bitwise equal to a resident replay of the same configuration's decisions; exactly 1 prefill + 2 decode calls; restore lengths eligible; reuse traces actually restore | as stated | PASS (20) | `test_every_configuration_matches_a_resident_replay_of_its_own_decisions` |
| Negative controls for the strict check: one greedy token or one logit off | detected | PASS | `test_the_strict_check_detects_a_wrong_output` |
| Identical requests across C0/C1/C2; equal request KV bytes (N × per-token bytes) | as stated | PASS | `test_compared_configurations_replay_identical_requests` |
| C1 restore lengths vs the independent key-level LRU model, capacities 0/1/2/3/100 × 3 reuse traces; capacity 100 equals the oracle | equal | PASS (15) | `test_gpu_only_restores_follow_the_independent_key_level_lru_model` |
| C1 budgets 0, 1 chunk − 1 B, 1 chunk, 3 chunks − 1 B, 3 chunks: capacity, retained storage ≤ budget and = entries × chunk bytes, one owned contiguous storage per entry, evictions, zero-capacity skips | as stated | PASS (5) | `test_capacity_is_a_hard_bound_on_owned_retained_storage` |
| C1 with capacity 0 vs C0 | bitwise equal | PASS | `test_zero_capacity_gpu_only_is_bitwise_the_no_cache_path` |
| No aliasing between the cache and the request KV (after restore and after prefill); in-place writes to the request KV leave every entry unchanged | as stated | PASS | `test_cached_prefixes_are_never_aliased_or_mutated_by_continuation` |
| Partial prefix, exact multiple (L ≤ N−1), full eligible restore; C1 and C2 | L = 0, 4, 4, 8 | PASS | `test_partial_prefix_exact_multiple_and_full_eligible_restores` |
| Storing skips present chunks without refreshing them (hand-worked; C1 and the key-level model) | 0, 4, 0, 4 | PASS | `test_storing_does_not_refresh_present_chunks` |
| Corrupted entry identity; model KV not matching the namespace | raise (not a miss) | PASS | `test_identity_and_model_mismatches_raise_instead_of_missing` |
| C2 wrapper vs `AlwaysRestoreEngine.run()` | identical records and outputs | PASS | `test_tiered_configuration_is_the_unchanged_engine` |
| Resource terms per configuration, both C2 backends: zeros for C0; C1 persistent ≤ budget, scratch = L × per-token bytes; C2 host slab, scratch = destination (+ batched staging), disk within budget | as stated | PASS (2) | `test_resource_terms_are_reported_separately` |
| CUDA never initialised by these tests | not initialised | PASS | module fixture `cpu_only` |
