# Design (M1 scope, with M1.1 fixes; M2 measurement additions in §10; M3 restore backends in §11; M4-A CPU-only trace replay in §12; M5a opt-in restore-length policy in §14; M5b measurement harness in §15)

This is the authoritative design for what is implemented. Anything not described here is
not implemented; planned but unimplemented work is listed under **Excluded** in §1.
Decisions and trade-offs are recorded in `docs/decisions.md` (D11+ for M1).

## 1. Scope
**Supported:** one pinned model/config (`Qwen/Qwen2.5-1.5B-Instruct@989aa79…`,
bf16, SDPA, all `full_attention` layers), text-only causal attention, batch 1,
a single process with sequential requests, complete chunks (C = 256), reuse of
the longest contiguous valid prefix, a bounded pinned-host tier and a bounded
project-owned disk tier, and a synchronous always-restore policy.

**Excluded:** LoRA, multimodal input, custom position schemes, sliding-window
layers, concurrency, async copies or pipelines, crash durability (no fsync),
cross-process sharing, startup recovery of previous-run caches, cost models,
admission policy, prefix-aware eviction, and compression or quantization.

## 2. Restore-length rule (unchanged from M0)
For prompt length N, L must be a complete chunk boundary with 0 < L ≤ N−1;
the largest such L whose chunks are all present and valid is used. At least
one prompt token is always recomputed, because the first output token's logits
come from the last prompt position, and caching those logits is out of scope.
Efficiency trade-off: when N is an exact multiple of C, the last stored chunk
is never restored for that same N (e.g. N = 512 restores only 256 tokens),
and C−1 extra tokens are recomputed. A later prompt that extends the prefix
can still restore that chunk.

## 3. Cache identity (`kvlab/identity.py`)
- **Namespace** fields: `layout_schema` (`kvlab-kv-layout-v1`), `model_id`,
  `revision`, `kv_dtype` (actual dtype of the KV tensors), `num_layers`,
  `num_kv_heads`, `head_dim`, `chunk_size`, `rope_parameters`, `layer_types`,
  `use_sliding_window`, `sliding_window`, `attn_implementation`,
  `transformers_version`, `torch_version`.
- **Canonical encoding:** JSON with sorted keys, no whitespace, ASCII only,
  `allow_nan=False`. Only str/int/float/bool/None/list/dict with string keys
  are allowed; anything else raises. Floats use Python's shortest round-trip
  `repr`.
- `ns_digest = SHA-256(b"kvlab-namespace-v1\0" + canonical_json)` (32 bytes).
- `key_i = SHA-256(b"kvlab-chunk-v1\0" + ns_digest + parent_key + u32le(i) + u32le(T) + T × u32le(token_id))`.
  The parent of chunk 0 is 32 zero bytes. Because of the chain, identical
  chunk tokens after different prefixes, or under a different namespace,
  get different keys. Python's process-randomized `hash()` is never used.
- On every read, the expected identity (namespace, key, parent, index, exact
  token ids) is compared field by field with what is stored. Host entries are
  checked the same way: a mismatch there means an in-process bug and raises.
  Disk mismatches are misses.

## 4. Chunk payload layout
`payload = concat over layer l = 0..num_layers−1 of [K_l bytes, V_l bytes]`.
Each tensor is `[1, num_kv_heads, C, head_dim]`, C-order (row-major), with
elements in **little-endian** byte order. This is the native order on the
supported x86-64 host; the module refuses to load on a big-endian host. bf16
is stored as its raw 2-byte patterns (lossless: NaN payloads, ±0, and inf
are preserved bitwise). For the baseline, one chunk is
28 × 2 × 2 × 256 × 128 × 2 B = **7,340,032 B**.

## 5. Disk chunk file format v1 (`kvlab/chunk_format.py`)
All integers are little-endian.

| Offset | Field | Type |
|---|---|---|
| 0 | magic `KVLCHNK1` | 8 bytes |
| 8 | format_version = 1 | u16 |
| 10 | dtype_code (1 bf16, 2 fp16, 3 fp32) | u16 |
| 12 | ns_digest | 32 bytes |
| 44 | chunk_key | 32 bytes |
| 76 | parent_key | 32 bytes |
| 108 | chunk_index, token_count, num_layers, num_kv_heads, head_dim, meta_len | 6 × u32 |
| 132 | payload_len | u64 |
| 140 | header_crc = CRC32(bytes[0:140] + tokens + meta) | u32 |
| 144 | token ids | token_count × u32 |
| … | meta = canonical namespace JSON | meta_len bytes (≤ 4096) |
| … | payload | payload_len bytes |
| end−4 | payload_crc = CRC32(payload) | u32 |

**Read validation order.** Every check runs before the payload is read, and
no allocation is ever sized by a header field.
1. File size ≥ 144.
2. Magic, version, known dtype.
3. Layout fields and dtype equal the *expected* namespace values.
4. token_count = C; meta_len ≤ 4096; payload_len = the expected payload length.
5. File size = 144 + 4·C + meta_len + payload_len + 4.
6. Bounded read of tokens + meta, then header CRC.
7. Namespace digest, with per-field diagnosis from meta; key, parent, index,
   tokens; recompute the key from the header contents.
8. `readinto` a caller-provided, fixed-size staging buffer; payload CRC.

CRC32 detects accidental corruption only. It is **not** authentication; the
cache directory is trusted as project-owned local storage.

## 6. Tiers and ownership (`kvlab/host_store.py`, `kvlab/disk_store.py`)
**Exclusive tiers:** a chunk lives in exactly one tier at a time.
- New chunks go to host.
- Host eviction spills to disk (synchronously), then frees the host slot.
- Disk eviction deletes the file.
- Disk hits are read through the host staging buffer and are **not**
  promoted, so a chunk never has host and disk copies at once.

**Host tier.**
- One slab, allocated once and pinned with `cudaHostRegister` at exactly
  `(slots + staging) × chunk_bytes`. PyTorch's caching pinned allocator is
  not used, because it rounds sizes up to powers of two and keeps freed
  blocks cached.
- `slots = min(max_entries, (budget − staging × chunk_bytes) // chunk_bytes)`,
  with one staging slot. The slab, `(slots + 1) × chunk_bytes` ≤ budget, is
  allocated (and registered) at construction and held until `close()`,
  whatever the occupancy.
- Reported separately (D21); none of these is an RSS or page-lock measurement:
  - `slab_allocated_bytes` (constant);
  - `cuda_host_registered_bytes` (0 with `pin=False`);
  - `staging_reserved_bytes` (constant);
  - `resident_logical_bytes` and its peak: entries × chunk_bytes, i.e.
    occupancy *inside* the slab, not extra memory.
- Metadata (identity + 4·C token bytes per entry) is bounded by `max_entries`.
  It is reported as `metadata_bytes_estimate`: an estimate of raw field bytes,
  not measured Python heap usage.
- LRU is at chunk granularity; a lease (`with host.lease(key)`) pins an
  entry and touches LRU, and pins are released in `finally`.
- Eviction considers only unpinned entries. If every candidate is pinned, the
  write is skipped (`host_all_pinned`) and capacity is never expanded.
- If a chunk is larger than the budget allows (no slot and no staging), the
  host tier is disabled. Every put is skipped (`oversized_for_host`), and
  nothing oversized is allocated.

**Disk tier.**
- A dedicated directory containing the marker `.kvlab-disk-cache-v1`.
- An existing root is validated **before anything is deleted or written**,
  without following symlinks (`lstat`/`scandir`, D20):
  - the root is a real directory;
  - a non-empty root must contain the marker, and the marker must be a
    regular file (a symlink, directory, etc. is refused);
  - every other entry has our name pattern and is a regular file or a symlink.

  Anything else raises `ValueError` and leaves the directory untouched.
- Stale files from earlier runs that match our name patterns are then deleted
  (no recovery). If one cannot be deleted, construction fails with that
  `OSError`.
- A missing marker is created with `O_CREAT|O_EXCL|O_NOFOLLOW`. An existing
  marker is never rewritten.
- Filenames are generated internally from the key: `<64 hex>.kvc`, and temp
  files `<64 hex>.kvc.tmp<n>`. Every path is checked to be a direct child of
  the resolved root before it is opened or unlinked.
- Budget: `used = committed (readable entries) + reserved temp bytes +
  undeletable bytes`. The exact file size is reserved before the temp file is
  created.
- **Bytes are released only once a file is gone** (D19). An already missing
  file counts as gone. If `unlink` raises `OSError`, the file becomes
  *undeletable*:
  - it stays in `used` and counts against `max_entries`;
  - it is never read, and its key is never rewritten
    (`undeletable_file_for_key`);
  - it is not retried automatically (`clear()` makes one explicit attempt).
- `max_entries` bounds readable entries + undeletable files; LRU deletes files.
- Before evicting, `put()` checks that eviction can make room. If temp
  reservations + undeletable bytes leave no room, it returns
  `disk_budget_unavailable` and evicts nothing. A victim that cannot be
  deleted ends that `put()` with `evict_oserror:<errno>`, a reported write skip.
- Writes go to a same-directory temp file, then `os.replace` (atomic
  rename). No fsync, so there is no durability claim.
- The temp file is removed in `finally` on any error. If that removal fails,
  its full reservation stays accounted as undeletable. `OSError` is reported
  as a write skip; other exceptions propagate.
- Chunk files are opened for reading with `O_NOFOLLOW`. A symlink is a miss
  (`disk_read_oserror:ELOOP`) and is removed.
- Scope: one process, sequential calls, a trusted directory. These checks are
  not a defence against concurrent or adversarial changes (TOCTOU).
- The OS page cache is not counted, and reads may be served from it. File
  reads are "file reads", never "NVMe reads".

## 7. Failure classification
| Situation | Handling |
|---|---|
| Absent entry; missing file; size/truncation; bad magic/version/dtype; header or payload CRC; namespace/key/parent/index/token mismatch; layout mismatch; `OSError` on read (incl. `ELOOP` for a symlink) | **miss** (counted with a reason); a bad disk entry is removed; if its file cannot be deleted it becomes undeletable (accounted, never read) and the result is still a miss |
| Middle chunk invalid | restore only the verified contiguous prefix before it; chunks after it are not used |
| `OSError` on write or spill (incl. ENOSPC), or `OSError` when unlinking an eviction victim (`evict_oserror:*`) | skip caching this chunk and record the reason; the request result is unaffected and inference is **not** repeated; a failed spill drops the chunk |
| `OSError` when unlinking a failed temp file | the write is still skipped; the temp file's reservation stays accounted as undeletable |
| Directory not owned or not valid (unknown entries, non-regular or symlinked marker, cache-named non-file); stale file not deletable at startup | refuse at construction (`ValueError` / `OSError`), before any request |
| Host identity mismatch, wrong tensor shape/dtype from the model cache, invariant violations, non-`OSError` exceptions, CUDA errors | **raise** |

Restore state: GPU destination tensors are filled chunk by chunk (with the M3
`batched` backend: chunk-major staging, placed group by group, §11). The
`DynamicCache` is built only from the verified prefix `[0, L)`, so no
partially invalid state is reachable. Leases are released on success and on
failure.

## 8. Always-restore request flow (`kvlab/always_restore.py`)
1. Compute chain keys for the complete chunks of the prompt.
   `n_plan = max_restore_len(N, C) / C`.
2. Restore the longest contiguous valid prefix (L = k·C, k ≤ n_plan) into a
   **fresh** `DynamicCache`.
3. Prefill `[L, N)` at positions `[L, N)`, or the full prompt if L = 0.
4. Store every complete prompt chunk not already cached (D2H directly into a
   host slot).
5. Greedy-decode a short continuation. Drop the GPU cache: the next request
   can reuse data only through the tiered store.

Counters: host/disk hits, misses with reasons, recovered prefix tokens,
selected L, restore vs full recompute, bytes D2H/H2D/disk read/disk written,
evictions (host spills, disk deletions), write skips with reasons, and
current/peak host and disk bytes.

## 9. Numerical verification contract
- **Storage correctness** is checked strictly: byte identity of restored K/V
  against what was written, and bitwise equality of all step logits against
  a resident-GPU reference that replays the same request sequence and splits
  (`kvlab/resident_reference.py`). The M0 tolerance (atol 1.45) is never used
  as evidence that storage is correct (`docs/results.md`, M0.1: it misses one-token
  prefix errors and position shifts).
- **Full recomputation** is compared separately and reported (D7 metrics). The
  known full-vs-split bf16 limitation (M0.1: exact ties can flip greedy
  tokens) is preserved and not gated in M1. The original M0 strict suite stays
  in place and still FAILs.
- **D10 (decided 2026-11-03):**
  - standard argmax;
  - no tie rule, rounding, tolerance inflation or xfail;
  - the M0 strict result is kept as recorded.

  This contract does not claim that restored output equals full-prefill
  output, or that every full-vs-split numerical difference is harmless.
  Those differences remain a separately reported limitation.
- **Every request makes exactly one prefill and `steps − 1` decode forward
  calls.** Cache write, spill, eviction and cleanup failures never add or
  remove a forward call (checked on CPU with a tiny random model, and on GPU).


## 10. M2: measurement hooks and the offline cost model (no runtime policy change)
- `AlwaysRestoreEngine.prefill_request(token_ids, allow_restore=True, on_restored=None, mark=None)`
  runs everything before first-token selection, in the M1 order: ingest
  (ids, chain keys), restore, prefill, store. `run()` = `prefill_request()` +
  greedy decode, with unchanged behavior.
  - `allow_restore=False` skips the lookup; it is used only to measure full
    recomputation offline.
  - `mark(phase)` is an optional callback for diagnostic phase timing.
- `HostStore.discard(key)` removes an unpinned entry and frees its slot
  without spilling, for benchmark state resets. It raises for a pinned entry.
- `kvlab/perf.py` and `kvlab/perf_harness.py` implement the protocol in
  `docs/M2_PROTOCOL.md`. The cost model is an offline predictor of
  tokenized-input first-token latency for full / host-restore / disk-restore
  at a given (L, S). Nothing in the runtime consults it.

**M2.1 note.**
- **New diagnostic modules:** `kvlab/perf_stats.py` (block-bootstrap
  statistics) and `kvlab/diag_harness.py` (`full_nocache` baseline, CUDA-event
  marks, probes) are diagnostic only.
- **Harness refactor:** `Bench.page_cache_before()` was extracted from
  `Bench.measure()` with no behavior change.
- **Unchanged:** the engine, tiers, runtime policy and the M2 cost model.
- **Reference:** `docs/results.md` (M2.1) and decisions D27–D29.

## 11. M3: restore backends (restoration mechanics only; default unchanged)
- `TieredKVCache(..., restore_backend="baseline" | "batched", restore_staging_chunks=4)`.
  The default is `baseline` (the M1 copies). `restore_backend` may be switched between
  requests on the same store; it is validated on every restore.
- **Shared by both backends, unchanged:** lookup (longest contiguous present prefix), leases
  and LRU touches, staging-slot disk reads with full validation, stop at the first invalid
  chunk, sources, counters (hits counted only after a completed copy), miss reasons, budgets,
  no promotion, and the D4 public-constructor cache build from `[0, L)`.
- **batched:**
  - per chunk, one blocking contiguous H2D of the whole 7,340,032 B slot into a chunk-major
    device staging tensor `X` of min(n, G) chunks;
  - per group of ≤ G chunks, one `copy_` of `X` permuted `[g, layer, K/V, 1, head, t, d] →
    [layer, K/V, 1, head, g, t, d]` into one layer-major destination `Y` (n chunks);
  - the last, possibly partial, group is placed before the cache is built.
- **Synchronization:** no copy is `non_blocking`. A blocking H2D returns after the transfer
  completed, so a lease or the staging slot is released only after its transfer finished. All
  device work is on the current stream: `X` is rewritten only by memcpys ordered after the
  placement that reads it, and `X`/`Y` are freed at return under the caching allocator's
  same-stream ordering.
- **Memory:** no new host memory. Transient device memory per restore of n chunks:
  - baseline: n·CB destination plus per-copy 128 KiB temporaries;
  - batched: n·CB destination plus min(n, G)·CB scratch;
  - both: plus the k·CB cache, which owns exactly its L tokens.

  `TieredKVCache.last_restore` reports the copy calls, placement ops, H2D bytes and the
  destination/scratch sizes of the latest restore (diagnostic, not a counter).
- Details and the options considered: `docs/M3_DESIGN.md`; decision D30; results
  `docs/results.md` (M3).


## 12. M4-A: synthetic traces, a GPU-only prefix cache and trace replay
Status: implemented and tested on CPU with a tiny random model. When M4-A was completed, nothing
here had run on a GPU. Since then, the M4-B correctness gate (§13) has exercised its replay paths
on the GPU. No timing, calibration or performance claim comes from M4-A. Decision D32.

- **Traces** (`kvlab/traces.py`):
  - **Synthetic, seeded token sequences.** They model the structure of prefix reuse, not
    natural language or production traffic.
  - **T0 `no_reuse`:** a distinct first token per request, so there is no shared prefix at any
    chunk size.
  - **T1 `shared_prefix`:** one shared prefix, then unique suffixes.
  - **T2 `multi_turn`:** conversations interleaved turn by turn. Each prompt extends the
    previous turn's prompt by a generated "answer" and a new user turn. The answers are drawn
    by the generator, not produced by a model.
  - **T3 `doc_reuse`:** a shared instruction, a Zipf-weighted document and a unique question.
  - **Identity and files:** a trace is identified by the SHA-256 of its canonical JSON. Files
    are JSONL, never overwritten, and checked on load.
  - **Offline statistics:**
    - the oracle reusable prefix under an unbounded cache (first-token rule; every complete
      prompt chunk is stored);
    - the unique-chunk working set;
    - per budget: the capacity, ρ = working set / budget, and the key-level LRU restore lengths
      of the C1 policy.
- **C1, the GPU-only baseline** (`kvlab/gpu_prefix_cache.py`): a bounded cache on the model's
  device.
  - **Shared with the tiered engine:** the namespace, the chain keys, the longest leading run
    of eligible present chunks (first-token rule, §2), storage of every complete prompt chunk
    after prefill, and chunk-granular LRU (hits touched in order; storing skips present chunks
    without touching them).
  - **Capacity:** `budget // chunk_bytes`. With 0, nothing is stored.
  - **Identity mismatch on a hit:** raises (an invariant, as for the host tier). It is never a
    miss.
  - **Ownership:**
    - each entry is one owned contiguous tensor in the payload layout (§4), filled by copy;
    - `retained_bytes()` sums the actual storages, so views cannot hide retained memory;
    - a hit gathers with `torch.cat` (temporary scratch, L tokens), then the D4 constructor
      copies again;
    - so a request never holds a stored tensor, and continuing the request cannot modify the
      cache.
- **C1 is not the correctness oracle:** `ResidentReference` is unbounded and replays another
  configuration's decisions. It stays separate, and it checks C1.
- **Replay** (`kvlab/trace_replay.py`): one interface for three configurations.
  - **The configurations:**
    - C0 no cache (full prefill; no keys, lookup or admission; D28 `full_nocache` semantics);
    - C1;
    - C2, the unchanged always-restore engine, driven through `prefill_request()` and
      `greedy_continue()` exactly as `run()` does. Its backend is the one its `TieredKVCache`
      was built with: `baseline` by default, `batched` only when selected.
  - **Per request:** exactly one prefill and `steps − 1` decode calls (checked), the restore
    length and sources, the stored chunks, the tokens and step logits.
  - **Resource terms, reported separately and never summed:**
    - persistent device cache (C1);
    - the request's own prompt KV;
    - temporary restore scratch;
    - the host slab and its logical occupancy (C2);
    - disk used and peak (C2).
  - **Not counted anywhere:** the OS page cache serving warm disk reads.
  - **Unequal resources:** the configurations do not use equal total resources.
- **Strict check** (`strict_mismatches`): each configuration's outputs must be bitwise equal to
  a `ResidentReference` replay of the **same** configuration's restore lengths and stored chunks
  (D12).
  - Different configurations are not compared bitwise where their splits differ (D10).
  - The check verifies outputs *given* the declared decisions. The decisions themselves are
    checked separately, against the key-level LRU model and the first-token rule.
  - A declared split one chunk shorter may give bitwise-equal outputs (it does on the tiny test
    model).
- **Deferred to GPU validation (M4-B; not done):**
  - a strict C1 smoke on the real model;
  - device-memory measurement (allocator-rounded blocks, activation peaks);
  - pinned-host registration;
  - any timing;
  - budget selection from trace statistics;
  - the evaluation protocol.

## 13. M4-B preparation: C1-PA, the trace-evaluation runner, validator and analysis
Status, 2026-11-04. The protocol is `docs/M4B_PROTOCOL.md`, approved for freeze (D35), with
decisions D33–D35. The freeze manifest (`results/m4/M4_EVAL_V1.freeze.json`) is created from the
commit that records this status, and its digest is kept outside the repository.
- **Implementation:** implemented and tested on CPU.
- **GPU correctness gate:** passed on its third attempt (below).
- **Timing pilot:** one timing-only feasibility pilot ran on calibration traces (D34).
- **Formal evaluation** (status updated 2026-11-05; results and limits in `docs/results.md`, M4):
  - **M4_EVAL_V1:** ran once. It is invalid under its frozen whole-tier warm-disk rule (§6 of the
    protocol), so it has no performance conclusion. The cause is unresolved.
  - **M4_EVAL_V2** (`docs/M4_EVAL_V2_PROTOCOL.md`, D36–D38): ran once and is valid. It replaces only
    the warm-disk preparation with a request-scoped one (`kvlab/warm_disk.py`) and adds a page-cache
    interference rule.
  - **Archive:** both evaluations are archived under `results/m4/` (D39).

The design choices in this section preceded all GPU work.

- **C1-PA** (`kvlab/prefix_aware_cache.py`): a subclass of the unchanged plain-LRU
  `GPUPrefixCache`.
  - **Inherited:** lookup, identity checks, owned storage, the copy semantics on a hit (gather,
    then the D4 constructor copy) and the byte budget.
  - **New admission:** it preserves prefix closure, evicts only unprotected leaves in LRU
    order, and stops admitting when no victim exists.
  - **Protected set:** the current request's chunks, from the moment the store loop reaches
    each of them until `store()` returns.
- **Independent key-level models** (`kvlab/trace_models.py`): C1-PA, C1-LRU (with dead
  entries) and C2's two LRU tiers (spill, no promotion). The validator compares every recorded
  decision and counter with them.
- **Session driver** (`kvlab/m4_bench.py`): the warmup traces, then the four evaluation traces.
  - **Instances:** fresh ones of C0, C1-PA(W), C1-PA(⌈W/2⌉), C2-base and C2-batch per trace.
  - **Order:** per-request interleaving in a seeded order.
  - **Timed region (D22):** synchronize, then `prefill()`, then argmax on the device; `.item()`
    is the end.
  - **Untimed:** the strict check against `ResidentReference` of each configuration's own
    decisions, recorded as logit digests. Nothing is caught: errors end the session.
- **The terms are kept distinct in every record and summary:**
  - **persistent cache bytes:** C1 `retained_bytes()` before and after each request, so growth
    during the request is explicit; C2's pinned slab, host occupancy and disk used and peak;
  - **transient storage:** C1's gather scratch; C2's device destination and batched staging;
    the request KV;
  - **logical copy traffic:** bytes moved per path. It covers the gather, the D4 constructor,
    admission, H2D, batched placement, D2H, and disk reads and writes. These are counted from
    the code paths, not profiled;
  - **allocator observations:** CUDA only. A0 before the request and P, the peak, are
    whole-process figures that include every co-resident configuration. P − A0 is the
    incremental process-allocation peak, including persistent C1 growth. It is never a
    per-configuration GPU peak;
  - **storage and page-cache observations:** the mincore resident fraction of every C2 chunk
    file before each C2 request, with files warmed if below 1.0. It is a point observation, not
    a budget.
- **Comparisons** are made within one process, with both C1 caches resident throughout. They
  describe this co-resident, interleaved experiment, not standalone deployments, and equal GPU
  budgets do not mean equal total resources.
- **Validator** (`kvlab/m4_validate.py`): fail-closed, recomputing everything from the spec.
  - **Schema pass:** types and ranges of every field are checked before any semantic check.
    Malformed evidence is exit 3, never a crash or a correctness failure. Records that fail the
    schema never reach a semantic check. A fail-closed backstop reports anything still
    uninterpretable as malformed; it can never make evidence valid.
  - **Exit reconciliation:** each session's exit must equal 4 if a strict mismatch is
    recomputed there, else 0. So an unexplained nonzero exit can never yield verdicts.
  - **Exit 3:** any missing, malformed, inconsistent or out-of-order record; a decision or
    counter that disagrees with the models; an unintended fault (spill or admission failures
    such as ENOSPC, read failures, temp or undeletable bytes); residency below 1.0; missing
    warmup coverage; or a batch-record problem (`kvlab.m3_confirm.batch_problems`: deadline,
    stop, launch order, contradictions).
  - **Exit 4:** a strict mismatch in complete evidence.
  - **Without a freeze manifest:** only explicitly unfrozen analyses run, labelled as not
    evaluation evidence.
- **Statistics** (`kvlab/m4_stats.py`, `kvlab/m4_analysis.py`):
  - **Unit:** the session.
  - **Primary family:** 15 comparisons. A verdict needs all session differences to share one
    sign, and the t-interval at level 1 − 0.05/15 to exclude 0.
  - **Assumptions:** independence and approximate normality, stated with every result. Fresh
    processes do not prove independence.
  - **Secondary results:** reported without verdicts, including the M3 block bootstrap as a
    sensitivity analysis.
- **GPU gate** (`tests/gpu/test_m4_replay.py`): run only through `scripts/m4_gate.py`.
  - One 900 s deadline covers the M1, M3 and M4 GPU suites.
  - `pytest -x` per suite, with a stop at the first failure.
  - On a stop signal or a wrapper error, the running group is still terminated, and a FAIL
    record is still written.
  - Cleanup is of the process group, independent of whether its leader has exited. Descendants
    left by a suite make the gate FAIL.
  - Each suite has its own pytest `--basetemp`, `<out>/basetemp/<suite>`.
    - `run_gate()` creates the common parent, because pytest's mkdir of an explicit basetemp is
      not recursive. It does so only after creating the fresh output directory.
    - The output directory itself is never a basetemp.
    - The first GPU attempt failed because that parent was missing.
  - The wrapper passes the fresh output directory to the suites as `KVLAB_M4_GATE_OUT_DIR`. The
    M4 suite writes structured per-request diagnostics under it (`m4_diagnostics/<case>.json`):
    the allocator figures A0, P and P − A0, labelled whole-process, and the logical storage terms.
    A final test fails the gate if any case's file is missing or incomplete.
  - **Checks**, beyond the strict replay against `ResidentReference`:
    - C1 restored bytes compared directly with the cached entries at the instant `restore()`
      returns: lengths, order, identity and no shared storage;
    - all C1 and C2 counters reconciled with the key-level models;
    - fault counters zero; prefix closure; budgets; storage overlap.
  - Elapsed times are recorded as a feasibility diagnostic.
  - **Outcome:**
    - attempt 1 failed (the basetemp defect);
    - attempt 2 passed, with three coverage gaps found in its review;
    - attempt 3 passed at the strengthened version: 46 tests, 311 s of 900 s. Its evidence is
      kept locally, not in the repository.
- **Batch control** (`kvlab/m4_freeze.py`, `scripts/m4_eval.py`): freeze, verify and run.
  - `run` reuses `kvlab.m3_confirm.run_batch` unchanged: one deadline, a stop on any nonzero
    exit, and no retries.
  - The freeze refuses a DRAFT protocol.
- **Timing-only feasibility pilot (D34):** one fresh session of the full calibration traces
  and the warmup, run through a private local adapter (not committed) around the unchanged
  `run_session`. It took 189 s. Ten-session projections are 1,891–1,935 s, below 80% of the
  2,700 s deadline. One session does not establish variability or guarantee completion.
- **At D35 approval, not yet done:** the freeze, any evaluation session and any performance
  measurement. The runtime restore-length policy, cost-model refit and asynchronous restoration
  are out of scope.

## 14. M5a: an opt-in restore-length policy and a versioned cost model (CPU only)
Status 2026-11-05: implemented and CPU-tested. The GPU gate, one calibration and one held-out evaluation then ran
and are valid: CostAware chose the same lengths as AlwaysRestore on all 8 evaluation cases (`docs/results.md`,
M5). Details: `docs/M5_DESIGN.md`; decisions D40–D41; protocol `docs/M5_PROTOCOL_DRAFT.md`.
- **Default unchanged:** `AlwaysRestoreEngine(model, tiered, *, policy=None)`; with `policy=None` the §8 flow
  is unchanged, and `allow_restore=False` never consults a policy.
- **With a policy:** a side-effect-free plan (`kvlab/restore_policy.py`: index membership only, candidates
  `0, C, …` under §2, stopping at the first absent chunk or, without staging, the first disk chunk), a
  validated choice, then `TieredKVCache.restore()` of exactly the chosen prefix. Validation failures keep the
  verified prefix (§7); there is exactly one prefill from the actual length.
- **Policies:** `AlwaysRestore`, `Recompute`, `FixedChunks` and `CostAware`. The last minimizes the objective of
  a `kvlab/restore_cost.py` artifact (schema `kvlab-restore-cost-v1`: digest, compatibility, a box domain; a plan
  with a positive candidate outside its projection onto the objective's inputs falls back to always-restore). It is separate from the frozen
  M2 model, which nothing at runtime loads.
- **Unchanged:** the storage format, tiers, eviction, budgets, counters' schema and the default backend.

## 15. M5b preparation: the candidate-measurement harness (CPU-tested; executed once on the GPU)
Implemented and CPU-tested 2026-11-05. The GPU gate, calibration and evaluation then ran once each (protocol
`docs/M5_PROTOCOL_DRAFT.md`, stages A–F; results `docs/results.md`, M5). Decisions D42–D43.
- **Spec:** `kvlab/m5_spec.py` declares the grid (fully specified chunk states, every feasible candidate, admissions,
  capacities, provenance), the seeded schedule, the domain, the design check and the bounded counts.
- **State:** `kvlab/m5_state.py` (benchmark only) builds a case's stores from one source bank and resets them after
  every sample with public store operations, verified by a structural fingerprint; production stores are unchanged.
- **Session:** `kvlab/m5_bench.py` measures FixedChunks candidates and real policy runs. The timed region is exactly
  synchronize, then `prefill_request`, then the first-token argmax. The strict checks of the actual split run
  untimed.
- **Validation, fit and analysis:** `kvlab/m5_validate.py` (fail-closed), `kvlab/m5_fit.py` (fit and frozen
  predictions) and `kvlab/m5_analysis.py` (multi-candidate statistics).
