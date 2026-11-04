# M3 design note: batched KV restoration (written 2026-11-03, before implementation)

Goal: reduce the host-side CUDA calls and synchronizations of `TieredKVCache.restore()`.
Everything else stays as it is: the chunk format, the prefix identity, the tier policy,
longest-valid-prefix restore, admission, counters, budgets and miss reasons. M2.1 associated
the restore-path slow mode with a higher host-side cost per CUDA call in this environment
(D29); it did not establish a root cause. So the target is the number of calls, not bytes.

Numbers marked **measured** come from `results/m3/design_probe/` (synthetic tensors with the
real layout, torch 2.14.1+cu130, no model). Numbers marked **estimate** are derived from them
and will be re-measured on the real request path (§7).

## 1. Current layouts
- **Host payload (unchanged, design §4):** one chunk is one contiguous slot of 7,340,032 B
  holding `[layer 0..27][K, V][1, kv_heads=2, C=256, head_dim=128]` bf16, C-order. Each
  per-layer K or V block is contiguous (131,072 B). Disk chunks are read into the single host
  staging slot with the same layout.
- **GPU destination (baseline, M1):** 56 separate tensors, per layer `K_l`, `V_l` shaped
  `[1, 2, n·C, 128]` for a planned prefix of n chunks. Chunk i of `K_l` is a strided slice:
  2 runs (one per head) of 64 KiB, at pitch n·64 KiB.
- **Restored cache (D4):** `DynamicCache(ddp_cache_data=[(K_l[:, :, :L], V_l[:, :, :L])])`.
  `DynamicLayer.update` concatenates onto an empty tensor, so the cache owns fresh tensors of
  exactly L tokens. That is 56 device copy kernels (**measured**), independent of n.

## 2. Baseline cost per restore (measured, synthetic, `probe_copy_calls_*.json`)
The baseline issues 56 `copy_` calls per chunk (28 layers × K/V), each blocking:
- n = 1: the destination slice is contiguous, so each copy is `cudaMemcpyAsync` +
  `cudaStreamSynchronize`: **56 memcpy + 56 syncs**.
- n > 1: the slice is strided. PyTorch then allocates a contiguous device temporary, copies
  into it and launches a copy kernel into the slice: **56 memcpy + 56 syncs + 56 kernels per
  chunk** (n = 4: 224 + 224 + 224).
- Plus the constructor: 56 kernels.

## 3. Options evaluated
| Option | Per chunk | Per restore | Device scratch | Verdict |
|---|---|---|---|---|
| (a) one strided `copy_` per chunk into a layer-major destination `Y[..., i·C:(i+1)·C, :]` | 1 memcpy + 1 sync + 1 kernel (n > 1) | + 56 constructor kernels | implicit PyTorch temporary of 1 chunk per copy | correct, simplest code; placement count grows with n |
| **(b) contiguous H2D per chunk into chunk-major device staging, then one permuted placement per group of ≤ G chunks** | 1 memcpy + 1 sync | + ⌈n/G⌉ placement ops + 56 constructor kernels | explicit, G chunks | **chosen** |
| (c) `cudaMemcpy2DAsync` straight into the strided destination | 1 call | + 56 | none | not exposed by PyTorch; needs ctypes into the runtime; rejected |
| (d) `non_blocking=True` for host-tier chunks, leases held until one sync per group | 1 memcpy | fewer syncs | as (b) | deferred: PyTorch records events only for its own caching host allocator, not for our `cudaHostRegister`-ed slab, so all safety would rest on our own syncs; disk chunks would still need a sync before the staging slot is reused |
| (e) skip the constructor copy by writing `DynamicLayer` fields | — | −56 kernels, −1 prefix-sized D2D copy | none | conflicts with D4 (no private-field writes); deferred and measured as a remaining cost |

Why (b) over (a): both remove the same 55 of 56 transfers and syncs per chunk. (b) keeps the
scratch explicit and bounded with a lifetime we control, and it needs ⌈n/G⌉ placement ops
instead of n. The difference between (a) and (b) is at most n − ⌈n/G⌉ kernel launches. It is
small next to the shared gain and is not claimed as a separate improvement.

The layout permits bulk assembly with standard PyTorch operations. Chunk-major staging
`X[g, layer, kv, 1, head, t, d]` is permuted to `[layer, kv, 1, head, g, t, d]`. That matches
the destination slice `Y[layer, kv, 1, head, s:s+g·C, d]` unflattened to `(g, C)`, so one
`copy_` between two strided device tensors does the whole group (**measured**: 1 kernel for
n = 4, byte-identical to the baseline). This is not 56 D2D copies per chunk; it is one
device op per group.

## 4. Estimated calls per restore (host or disk; G = 4; estimate from §2/§3)
| n (L) | baseline memcpy / sync / kernels | batched memcpy / sync / kernels |
|---|---|---|
| 1 (256) | 56 / 56 / 56 | 2 / 1 / 56 (the 1-chunk placement is a D2D memcpy) |
| 4 (1024) | 224 / 224 / 280 | 4 / 4 / 57 |
| 6 (1536) | 336 / 336 / 392 | 6 / 6 / 58 |
| 12 (3072) | 672 / 672 / 728 | 12 / 12 / 59 |
| 16 (4096) | 896 / 896 / 952 | 16 / 16 / 60 |

Kernels include the 56 constructor kernels. Not counted: the caching allocator (no
`cudaMalloc` once warm; M2.1 recorded 0 device allocations per sample) and
`cudaStreamIsCapturing` checks.

## 5. Memory
- **Host:** nothing new. The existing slots (via leases) and the single staging slot are the
  only H2D sources. No second pinned cache and no host-side staging of a whole prefix.
- **Device during one restore** (CB = 7,340,032 B, n planned, k ≤ n loaded):

  | | baseline | batched |
  |---|---|---|
  | destination | n·CB (56 tensors) | n·CB (one tensor `Y`) |
  | scratch | per-copy temporary, 131,072 B (n > 1) | `X`: min(n, G)·CB, at most 29,360,128 B for G = 4 |
  | restored cache | k·CB (constructor copies) | k·CB (constructor copies) |

  The extra transient device memory is min(n, G)·CB. Both destinations and `X` are freed
  when `restore()` returns.
- **Retained capacity:** none. The returned cache's tensors are the constructor's fresh
  copies of exactly L tokens; no view keeps `Y` or `X` alive. A test asserts that each
  layer's storage size equals its L-token payload.

## 6. Ownership, lifetime, synchronization
- All copies use `non_blocking=False` on the current stream. PyTorch implements a blocking
  H2D as `cudaMemcpyAsync` + `cudaStreamSynchronize` (**measured**), so the transfer is
  complete when `copy_` returns. `non_blocking` is not used anywhere: by itself it would not
  make reuse of our registered slab safe.
- **Host lease:** held around the chunk's H2D and released (in `finally`) after it returns,
  i.e. after the transfer has finished.
- **Staging slot (disk chunks):** `read_into` (CPU) → blocking H2D → released. The next disk
  read starts only after the previous transfer out of the slot finished. At most one transfer
  from the staging slot exists at any time.
- **`X` (chunk-major device staging):** allocated per restore from the caching allocator, on
  the current stream. H2D memcpys write it; one placement op per group reads it. The next
  group's memcpy is ordered after the placement on the same stream, and it blocks the host
  until done. `X` is released at return; the caching allocator only hands that block to later
  work on the same stream, which is ordered after the queued reads.
- **`Y` (layer-major destination):** per restore. The constructor's copy kernels read it;
  released at return under the same stream-order rule.
- **Restored cache:** owned by the returned `DynamicCache`. Later requests, buffer reuse or
  host-slot reuse cannot change it. Tests check this.
- **Failure:** a CUDA or programming error in a copy propagates (never a miss). Leases and
  staging are released by their context managers, `X`/`Y` are dropped, and no cache is built.
- **Partial prefix:** at the first missing, corrupt or incompatible chunk, the loaded part of
  the current group is placed, and the cache is built from `[0, L)` only. The unwritten part of
  `Y` is never exposed, because the constructor copies only `[0, L)`.
- Counters, sources, miss reasons, LRU touches (leases), budgets and tier states are updated
  exactly where the baseline updates them. Disk hits are not promoted.

## 7. What M3 will measure, and what it will not change
- **Measured:** e2e first-token latency at the M2 boundary; restore-phase time from separate
  synchronized samples; CUDA API and device-op counts from a separate profiler run; H2D
  bytes and copy counts; peak allocated and reserved device memory; explicit scratch sizes.
- **Unchanged and still costly:** the suffix forward (~1,340 kernel launches for one
  prefill, M2.1), admission (store: 56 D2H copies per admitted chunk), the 56 constructor
  kernels (option e), ingest, and disk read with CRC.
- **Not claimed:** cold-NVMe performance (WSL), p99, a root cause of the slow mode.
