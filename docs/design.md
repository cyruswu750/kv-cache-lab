# Design (M1 scope, with M1.1 fixes; M2 measurement additions in §10; M3 restore backends in §11)

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
