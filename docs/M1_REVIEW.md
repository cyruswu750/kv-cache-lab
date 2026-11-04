# M1 Review — cache identity, validated chunk format, bounded host/disk tiers, synchronous always-restore

Date: 2026-11-03. Status: **M1 implemented; all M1 acceptance tests PASS.**
The original **M0 strict suite still FAILs** (1/11, unchanged; see §2).

> **M1.1 note (2026-11-03).** External review reproduced:
> - two disk-tier defects: a failed `unlink` released budget and escaped the
>   cache layer, and a symlinked marker was written through;
> - an ambiguous host `used_bytes`.
>
> All are fixed in M1.1 (see `docs/M1_1_REVIEW.md`, decisions D19–D21). D10 is
> now decided (decisions D10).
>
> **Correction (BUGLOG VAL-004):** "strict pass 19/19" in §4 and the
> 19-request count mean 18 strict request records plus an ENOSPC request that
> was not strictly validated (it recorded 8 forward calls). The corrected
> runner and run are in `docs/M1_1_REVIEW.md` §13.
>
> The text below is the original M1 review and has not been edited. In §6,
> host "observed peak 22,020,096" is the slab size with both slots occupied,
> not a measurement.

## 1. Identity of what was reviewed
- Repo: `/home/cyrus/projects/kv-cache-lab` (the KV-cache project; not the KV store).
  Git branch `main`, **no commits**; no remote, no tags.
- Pre-edit snapshot: `backups/pre-m0_1-20261103T020921.tar.gz`, sha256
  `064868f2370ebce116b85040a7d6a0e951cd9eb0d936c61be6619f3f3234a218`
  (git-ignored). Per-file hashes: `results/m0_1/pre_edit_manifest_20261103T020921.sha256`.
- Post-M1 per-file hashes (94 files): `results/m1/post_m1_manifest.sha256`.
- Local review archive (not uploaded): `backups/kvlab-review-m1.tar.gz`, 136 KiB,
  sha256 `30a666f88c3ac20b367f8418606e7fb8932005710d09e7fdfeb8efe6e155bebe`.
  It contains source, tests, configs, docs and small results, and excludes
  .git, envs, weights, backups, cache payloads and credentials. The manifest
  was written before this review's final text edit, so `docs/M1_REVIEW.md`
  hashes differently from its manifest entry.
- Environment (unchanged): torch 2.14.1+cu130, transformers 5.22.1, Python 3.14.4,
  RTX 4080 Laptop 12 GiB, driver 596.49, WSL2, 7.4 GiB host RAM.
  Model: `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  **bf16**, **SDPA**.

## 2. Gate evidence and original M0 status
- M0.1 gate **PASS** (`docs/M0_1_REVIEW.md`):
  - restored == resident split, bitwise, on every greedy step, 3× repeated, partial restore included;
  - every byte check passes;
  - the `nat_long` mismatch is isolated to an exact bf16 tie
    (`0x41be`/`0x41be`; fp32 pre-rounding 23.769 vs 23.796; the fp32 instrument
    removes it);
  - negative controls are detected.
- Original M0 strict suite: `pytest -m "gpu and m0_strict"` → 10 passed,
  **1 failed** (`nat_long`, greedy 6/8). Criteria unchanged; no xfail;
  still included in `-m gpu`. The D10 tie policy remains **your open decision**.

## 3. Implemented / incomplete / blocked / excluded
| Item | Status |
|---|---|
| Identity: canonical-JSON namespace + SHA-256 chain keys (`kvlab/identity.py`) | implemented |
| Chunk file format v1: magic, version, bounded meta, layout fields, identity, CRC32 header + payload, LE, no pickle (`kvlab/chunk_format.py`) | implemented |
| Pinned host tier: exact `cudaHostRegister` slab, LRU, leases, all-pinned refusal, oversized bypass (`kvlab/host_store.py`) | implemented |
| Disk tier: owned dir + marker, generated names, temp+rename, temp bytes in budget, LRU, I/O injection points (`kvlab/disk_store.py`) | implemented |
| Synchronous spill on host eviction; failed spill drops the chunk | implemented |
| Longest contiguous valid prefix; stop at the first invalid chunk (`kvlab/tiered_cache.py`) | implemented |
| Always-restore engine with counters (`kvlab/always_restore.py`) | implemented |
| Resident-GPU replay reference for strict checks (`kvlab/resident_reference.py`) | implemented |
| Timings / TTFT / calibration grid | **NOT RUN** (deferred by instruction) |
| Promotion of disk hits back into host | deliberately excluded (D15) |
| Write-through to disk when the host has no slot | not implemented (the write is skipped with a reason) |
| Disk pins | not needed: single-threaded; disk eviction happens only in the store phase, never during a restore |
| Cost model, async/pipelines, concurrency, startup recovery, fsync durability, vLLM, LoRA/multimodal, UI | excluded by scope |
| Blocked items | none |

## 4. Commands run and exit codes (2026-11-03, in order; logs in `results/m0_1/logs/`, `results/m1/logs/`)
| Command | Exit | Result |
|---|---|---|
| `.venv/bin/python -m pytest -q` (after M0.1 hardening) | 0 | 193 passed |
| `scripts/m0_1_audit.py` (bf16 run 1, run 2) and `--fp32-diagnostic` | 0, 0, 0 | gate evidence |
| `scripts/m0_experiment.py --out-dir results/m0_reruns/20261103T021916` | **1** | nat_long FAIL (unchanged) |
| `.venv/bin/python -m pytest -m gpu -v` (after M0.1) | **1** | 16 passed, 1 failed (m0_strict nat_long) |
| `scripts/m1_mutation_check.py <scratch>` | 0 | 9/9 CPU mutants killed |
| `pytest -m "gpu and m1"` (first M1 run) | 0 | 10 passed |
| GPU mutants (LSB flip in restored K; K→V slot) | — | 2/2 killed |
| `scripts/m1_run.py --out results/m1/m1_run_20261103T023850.json` | 0 | strict pass 19/19 (first run, without divergence margins) |
| `scripts/m1_run.py --out results/m1/m1_run_20261103T023956.json` | 0 | strict pass 19/19 (final evidence) |
| `.venv/bin/python -m pytest -q` (final) | 0 | **288 passed**, 27 deselected |
| `pytest -m "gpu and m0_strict" -v` (final) | **1** | 10 passed, **1 failed** (nat_long) |
| `pytest -m "gpu and m0_1" -v` (final) | 0 | 6 passed |
| `pytest -m "gpu and m1" -v` (final) | 0 | 10 passed |
| `scripts/m0_experiment.py --smoke --out-dir results/m0_reruns/smoke_…` | 0 | PASS |

Test counts: 288 CPU + 27 GPU (11 m0_strict + 6 m0_1 + 10 m1).
One audit-tool bug was found and fixed during M0.1 (hidden-state index in the
audit script); see M0_1_REVIEW. No product-code failure occurred during M1
test runs.

## 5. Proof that each path actually executed (`results/m1/m1_run_20261103T023956.json`)
Sequence scenario: 9 requests; host = 2 slots + 1 staging (22,020,096 B, pinned); disk = 2 files (14,684,242 B).

| # | Prompt | N | L / eligible | Sources | Miss reason | Stored | Strict |
|---|---|---|---|---|---|---|---|
| 1 | X | 622 | 0 / 512 | — | absent | [0,1] | PASS |
| 2 | Y | 642 | 512 / 512 | host, host | — | [] | PASS |
| 3 | W | 600 | 0 / 512 | — | absent | [0,1] (spills X0, X1) | PASS |
| 4 | Z | 366 | 256 / 256 | disk | — | [] | PASS |
| 5 | X | 622 | 512 / 512 | disk, disk | — | [] | PASS |
| 6 | L3 | 779 | 512 / 768 | disk, disk | absent | [2] | PASS |
| 7 | Y | 642 | 0 / 512 | — | absent (X0 evicted from disk) | [0,1] | PASS |
| 8 | W | 600 | 0 / 512 | — | absent | [0,1] | PASS |
| 9 | X | 622 | 512 / 512 | disk, disk | — | [] | PASS |

Counters for the sequence:

| Counter | Value |
|---|---|
| host_hits | 2 |
| disk_hits | 7 |
| chunk_misses | 5 (all `absent`) |
| recovered_prefix_tokens | 2304 |
| chunks_stored | 9 |
| host spills (= host evictions) | 7 |
| disk evictions (deletions) | 5 |
| bytes_d2h = bytes_h2d | 66,060,288 |
| disk file bytes written = read | 51,394,847 (file reads; the page cache may serve them) |
| write_skips | 0 |
| spill failures | 0 |

Fault scenarios (fresh store each):

| Scenario | Outcome |
|---|---|
| corrupt middle chunk | L=256, `disk_payload_crc`, chunk 1 recomputed and re-stored; the next request restores L=768 from disk, host, disk |
| missing file | L=0, `disk_file_missing` |
| incompatible namespace | L=0, `disk_namespace_mismatch:attn_implementation` |
| incompatible layout | L=0, `disk_layout_mismatch:num_layers` |
| ENOSPC on every spill | **8 forward calls** (1 prefill + 7 decode, i.e. no recompute); `spill_failures={write_oserror:ENOSPC: 2}`; directory empty afterwards |

## 6. Budgets: definitions and observed peaks
| Tier | Definition | Budget | Observed peak |
|---|---|---|---|
| host | resident slots + 1 staging buffer, one exact `cudaHostRegister` slab (slab bytes = accounted bytes) | sequence: 22,020,096 B (3 × 7,340,032) | 22,020,096 (slab is fixed; never exceeded) |
| host metadata | identity + token ids per entry, bounded by `max_entries` = 64; reported separately | — | 2,256 B (sequence, 2 entries) |
| disk | committed files + reserved temp-file bytes; file = 7,342,121 B (payload 7,340,032 + 2,089 header/CRC) | sequence: 14,684,242 B (2 files) | 14,684,242; directory scan = committed |
| disk (ENOSPC run) | same | 58,736,968 B | peak 7,342,121 = temp reservation only, committed 0 |

**Not counted** (and stated as such):
- the OS page cache;
- GPU memory (the restore destination plus the `DynamicCache` copy, about 2× the prefix transiently; GPU-tier budgets are out of M1 scope);
- the Python/torch process baseline.

## 7. Full recomputation (report only)
Of the 7 restored requests, 3 matched full-recompute greedy 8/8. The other 4
come from **2 distinct prompts** (X; L3 = same prefix and tail). All 4 diverge
at step 4, where the **full** path's top-2 margin is **0.000** (X, exact tie)
or **0.125** (L3, 1 bf16 ulp). The first-token max_abs vs full is 0.27–0.375.
This is the M0.1 execution-shape bf16 effect, not storage: every one of those
requests is bitwise equal to the resident split replay. It is not gated, and
no tie rule was adopted (D10 open). Consequence for later milestones:
full-vs-restore output equality cannot be promised in bf16. The guarantee M1
gives is that restored == resident split, bit for bit.

## 8. Limitations
- Single process, sequential, batch 1, one model configuration. The disk cache is cleared at startup (no recovery).
- Host-tier writes need a free or evictable slot. With no slots, the write is
  skipped; there is no write-through to disk.
- Disk hits are not promoted (D15).
- CRC32 is not authentication; the cache directory must be trusted.
- No fsync: a crash may leave missing or zero-length files. Those become
  misses, but this was only tested via truncation/missing-file injection,
  not real crashes.
- `cudaHostRegister` slab: `close()` must be called (tests and scripts do). If
  a process exits without it, the OS reclaims the memory anyway.
- Restore allocates GPU destination tensors, and then `DynamicCache` copies
  them (D4), so peak GPU memory is about 2× the restored prefix.
- No performance numbers. Wall-clock or bandwidth figures seen during
  development (e.g. one-shot pinned vs pageable H2D) are **not** claims.
- `nat_long` (M0 strict) still FAILs, and full-vs-restore greedy divergence at
  bf16 near-ties is expected (§7).

## 9. Changed files for external review
- **Modified:** `kvlab/kv_snapshot.py`, `scripts/m0_experiment.py`,
  `tests/test_snapshot_cpu.py`, `tests/gpu/test_gpu_roundtrip.py` (marker only),
  `pyproject.toml`, `.gitignore`, `docs/decisions.md`, `docs/M0_WALKTHROUGH.md`,
  `README.md`, `docs/environment.md`.
- **New source:** `kvlab/identity.py`, `kvlab/chunk_format.py`,
  `kvlab/host_store.py`, `kvlab/disk_store.py`, `kvlab/tiered_cache.py`,
  `kvlab/always_restore.py`, `kvlab/resident_reference.py`.
- **New tests:** `tests/m1_toy.py`, `tests/test_identity.py`,
  `tests/test_chunk_format.py`, `tests/test_host_store.py`,
  `tests/test_disk_store.py`, `tests/test_tiered_cpu.py`,
  `tests/gpu/test_m0_1_controls.py`, `tests/gpu/test_m1_tiered.py`.
- **New scripts:** `scripts/m0_1_audit.py`, `scripts/m1_run.py`,
  `scripts/m1_mutation_check.py`.
- **New docs:** `docs/design.md`, `docs/M0_1_REVIEW.md`, `docs/M1_REVIEW.md`,
  `docs/M1_WALKTHROUGH.md`, `docs/correctness-matrix.md`.
- **Unchanged and read-only:** `results/m0/` (original M0 evidence).

Suggested review order:
1. `kvlab/chunk_format.py` (validation order)
2. `kvlab/disk_store.py` (`put`/`finally`, path checks)
3. `kvlab/tiered_cache.py` (`restore` stop-at-first-invalid, `store` slot release)
4. `kvlab/host_store.py` (`allocate`/`lease`)
5. `kvlab/identity.py`
6. `tests/gpu/test_m1_tiered.py`

## 10. Three commands to run tomorrow
```bash
.venv/bin/python -m pytest -q                                   # CPU: expect 288 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and (m0_1 or m1)" -v   # expect 16 passed
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_run.py --out results/m1/m1_run_rerun.json  # expect exit 0, strict_pass_all true
```
(The original strict suite: `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_strict" -v`
→ expect 10 passed, 1 failed.)

## 11. 验收前需要理解的要点（中文）
1. **M1 证明的是"存储无损"，不是"与完整重算一致"。**
   - 判据是两条：恢复出的 K/V 与写入时逐字节相同；每一步 logits 与"GPU 常驻、同样切分"的重放逐位相同。
   - atol=1.45 不能作为存储正确的证据（M0.1 已证明它漏检一 token 错前缀、位置偏移等错误）。
2. **完整重算 vs 恢复的贪心分叉是已知现象。**
   - 原因是 bf16 下执行形状不同导致的舍入平局（margin 0 或 1 ulp）。M0 strict 仍判 FAIL，平局规则（D10）等你决定。
3. **身份链。**
   - key = SHA-256(namespace 摘要 + 父 key + 块序号 + token ids)；同样的 token 在不同前缀后得到不同 key。
   - namespace 包含 dtype、层/头/维、RoPE、注意力实现和框架版本，任何一项变化都会变成未命中。
4. **读文件时先校验，再分配。**
   - 先查 magic/版本/布局/长度/文件大小 → 有界读 header → header CRC → 身份 → 再读 payload 到固定的 staging 缓冲 → payload CRC。
   - 任何已知问题都算未命中并删除该文件；程序错误和 CUDA 错误直接抛出。
5. **预算与所有权。**
   - 两层互斥，一个块只在一层。host 是一次性注册的 pinned slab（含 1 个 staging）；disk 预算 = 已提交文件 + 临时文件。
   - pin 保护恢复中的块；全部被 pin 时跳过写入，绝不超预算。
   - 写失败只跳过缓存，不会重算（ENOSPC 用例里前向调用次数正好是 8）。
