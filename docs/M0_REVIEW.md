# M0 Review — environment verification + minimal real KV round trip

Date: 2026-11-02. Scope: M0 only (see your M0 instruction). Nothing from M1 or Week 2 was started.

## Overall status: **FAIL (1 of 10 acceptance cases)**, with root cause diagnosed
- All byte-equality checks: **PASS** (every K/V tensor, every case, export and restore).
- Boundary cases N = 255, 256, 257, 511, 512, 513: **PASS**.
- Natural prompts: `nat_short`, `nat_medium` **PASS** (nothing eligible to restore),
  `nat_xlong` **PASS** (full + partial restore), `nat_long` **FAIL**
  (greedy 6/8 because of an exact bf16 tie; details below).
- The tolerances were not changed after any comparison was seen.

## Completed items
| Item | Status | Evidence |
|---|---|---|
| Project CLAUDE.md | done | `CLAUDE.md` |
| Environment inspected and recorded | done | `docs/environment.md`, `results/m0/environment.json` |
| Isolated env (`.venv`, no sudo, no global changes) | done | `docs/environment.md` |
| CUDA works from the venv PyTorch | **PASS** | `results/m0/logs/cuda_check.log` |
| Pinned model revision downloaded; dtype/backend checked after load | done | `results/m0/logs/model_download.log`, `model` block in `m0_results.json` |
| Installed cache API inspected (not assumed) | done | `docs/environment.md` "Cache API observed" |
| Tolerance rule fixed before any comparison; calibrated on a disjoint text | done | `docs/decisions.md` D7/D8, `results/m0/calibration.json` |
| CPU tests | **PASS** 190/190 | `results/m0/logs/pytest_cpu.log` |
| Smoke test | **PASS** | `results/m0/m0_smoke.json`, `results/m0/logs/m0_smoke.log` |
| Full experiment run1 / run2 | **FAIL** (nat_long only) | `results/m0/run1/`, `results/m0/m0_results.json`, `logs/m0_experiment_run2.log` |
| GPU pytest | **FAIL** 10 passed / 1 failed (nat_long) | `results/m0/logs/pytest_gpu.log` |
| Divergence root-cause analysis | done | `results/m0/divergence_nat_long.json`, `logs/m0_diagnose_nat_long.log` |
| Diagnostic timings | done (diagnostic only) | `diagnostic_timings_not_benchmarks` in `m0_results.json` |

## Commands actually run (in order) and exit statuses
| # | Command | Exit |
|---|---|---|
| 1 | `python3 -m venv --without-pip .venv` | 0 |
| 2 | `curl -sSfL https://bootstrap.pypa.io/get-pip.py -o <scratch>/get-pip.py && .venv/bin/python -I <scratch>/get-pip.py` | 0 |
| 3 | `.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130"` | 0 |
| 4 | CUDA check one-liner (`results/m0/logs/cuda_check.log`) | 0 |
| 5 | `.venv/bin/python -m pip install "transformers==5.22.1" pytest` | 0 |
| 6 | `snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa79…')` with `HF_HOME=.hf_cache` | 0 |
| 7 | `.venv/bin/python -m pytest -q` (CPU, first run) | 0 (190 passed) |
| 8 | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_calibrate.py` | 0 |
| 9 | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py --smoke` | 0 (PASS) |
| 10 | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py` (run1) | **1** (nat_long FAIL) |
| 11 | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_diagnose_divergence.py nat_long` | 0 |
| 12 | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py` (run2, adds nat_xlong) | **1** (nat_long FAIL) |
| 13 | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m gpu -v` | **1** (10 passed, 1 failed) |
| 14 | `.venv/bin/python -m pytest -v` (CPU, final) | 0 (190 passed, 11 deselected) |
| 15 | `.venv/bin/python scripts/env_report.py` | 0 |

## Acceptance results (run2; run1 numbers are identical for every case run1 had)
Chunk C = 256; tolerances atol 1.45, rtol 0.02; greedy 8 tokens.
Every restore had byte-equal export and restore checks. Every K/V tensor is
`[1, 2, 256, 128]` bf16 = 131,072 B per chunk per layer; one chunk = 56
tensors = 7,340,032 B.

| Case | N | Eligible L | L tested | Suffix | max_abs | rel_l2 | argmax | greedy | bitwise = GPU-chunked | neg-ctrl max_abs | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary_255 | 255 | [] | — | — | — | — | — | — | — | — | PASS |
| boundary_256 | 256 | [] | — | — | — | — | — | — | — | — | PASS |
| boundary_257 | 257 | [256] | 256 | 1 | 0.1875 | 1.17e-2 | eq | 8/8 | yes | 18.32 | PASS |
| boundary_511 | 511 | [256] | 256 | 255 | 0.5625 | 2.73e-2 | eq | 8/8 | yes | 23.62 | PASS |
| boundary_512 | 512 | [256] | 256 | 256 | 0.2500 | 1.46e-2 | eq | 8/8 | yes | 22.56 | PASS |
| boundary_513 | 513 | [256, 512] | 256 (partial) | 257 | 0.2188 | 1.42e-2 | eq | 8/8 | yes | — | PASS |
| | | | 512 | 1 | 0.2500 | 1.87e-2 | eq | 8/8 | yes | 23.09 | |
| nat_short | 40 | [] | — | — | — | — | — | — | — | — | PASS |
| nat_medium | 166 | [] | — | — | — | — | — | — | — | — | PASS |
| nat_long | 426 | [256] | 256 | 170 | 0.3125 | 1.75e-2 | eq | **6/8** | yes | 16.28 | **FAIL** |
| nat_xlong | 611 | [256, 512] | 256 (partial) | 355 | 0.2969 | 1.78e-2 | eq | 8/8 | yes | — | PASS |
| | | | 512 | 99 | 0.3047 | 2.33e-2 | eq | 8/8 | yes | 16.73 | |

For N = 256 and 512 (exact multiples of C), the last chunk is correctly **not**
restored: 256 → L = 0, 512 → L = 256.

### How the GPU test proves restoration was used
1. The suffix forward receives only `N − L` token ids. For N = 513, L = 512
   that is a single token. The cache length is asserted to be `L` before the
   forward and `N` after it.
2. The prefix GPU cache is deleted and `torch.cuda.empty_cache()` is called
   **before** restore. Restored tensors are rebuilt only from the CPU
   snapshot's fresh allocations (CPU tests check that storage pointers are
   disjoint).
3. Negative control: restoring an all-zero snapshot of the same shape moves
   first-token logits by 16–24 (far above atol 1.45). So the suffix
   computation really depends on the restored K/V.
4. The snapshot's per-tensor SHA-256 is unchanged after all restores and decodes.

### FAIL analysis: nat_long
- At greedy step 6, the **reference** (path A) has an exact bf16 tie:
  `' re'` = `' described'` = 23.75 (top-2 margin 0.0). argmax picks the lower id.
- In path B, the chunked prefill changes `' described'` by +0.125 (1 bf16 ulp
  at that magnitude). It wins, and decoding diverges. When B is teacher-forced
  on A's tokens, its argmax matches A at all other 7 steps, and the per-step
  max_abs is ≤ 0.5.
- The restored path is **bitwise identical** to a GPU-resident chunked prefill
  that never touches the CPU. So the gap comes from chunked vs unchunked bf16
  compute, not from the storage round trip.
- The case is still reported as FAIL because criterion 4 (greedy identical)
  was fixed in advance. Proposed rule options for you to choose:
  `docs/decisions.md` D10.

## Diagnostic timings (NOT benchmarks, NOT TTFT)
Timing boundary: `torch.cuda.synchronize()` → CUDA event + `perf_counter` →
operation → CUDA event → `torch.cuda.synchronize()`. There are 2 warmups and
5 timed reps, and the table shows the median. GPU-event ms is used for
forwards and host wall ms for copies, since copies from pageable memory are
host-staged. Restore includes the H2D copy, construction of the
`DynamicCache`, and its internal D2D copy (D4). Run2 values:

| Case | N | L | full prefill (gpu ms) | export GPU→CPU L (wall ms) | restore CPU→GPU L (wall ms) | suffix prefill n=N−L (gpu ms) |
|---|---|---|---|---|---|---|
| boundary_257 | 257 | 256 | 35.93 (25.1–48.4) | 2.46 | 3.25 | 16.20 |
| boundary_511 | 511 | 256 | 42.72 | 5.55 | 8.53 | 21.68 |
| boundary_512 | 512 | 256 | 31.21 | 2.44 | 3.12 | **66.06** |
| boundary_513 | 513 | 512 | 38.58 | 8.37 | 9.63 | 16.30 |
| nat_long | 426 | 256 | 30.37 | 2.38 | 3.16 | 19.89 |
| nat_xlong | 611 | 512 | 40.89 | 15.87 | 20.85 | 19.42 |

Caveats, all unexplained and not investigated in M0:
- Run-to-run variation is large. In run1, boundary_512's suffix median was
  21.58 ms; in run2 it was 66.06 ms, with tight within-run spread both times.
  Restore for 512 tokens took 9.6 ms in one case and 20.9 ms in another.
  Likely causes to check later: laptop GPU clocks and power state, WSL2
  scheduling, pageable-memory staging. **Do not use these numbers for claims.**
- A 1-token suffix costs about 16 ms. At these lengths the per-forward fixed
  cost (28 layers, Python launch overhead) dominates. This matters for the
  future T_prefill(n | L) model, which likely has a large intercept.

## Limitations
- bf16, not the plan's fp16 (D2). Batch size 1, no padding. SDPA only.
- Only first-token logits are gated by atol/rtol. Later greedy steps are
  checked for finiteness and token identity only.
- atol = 1.45 is wide in absolute terms (D8). The stronger evidence is byte
  equality plus bitwise equality with the GPU-resident chunked path.
- `nat_short` and `nat_medium` are shorter than 257 tokens, so they only test
  the "nothing eligible" path. `nat_xlong` was added after run1 (D9).
- Snapshot lives in pageable CPU memory. No pinned buffers, files, budgets,
  or async copies (excluded from M0).
- Restore does one extra D2D copy because it uses the public `DynamicCache`
  constructor (D4).
- WSL2 shows only 7.4 GiB of host RAM. That will constrain the Week-2 host tier
  (model load also passes through host RAM).
- `docs/design.md` does not exist yet. The plan's Week-1 items (2-D
  calibration grid, NVMe timing, feasibility note, EXPERIMENTS.md) are **NOT RUN**.

## Blockers
- None blocking M0. The one workaround was the missing `ensurepip`, solved
  inside the project with `venv --without-pip` + official `get-pip.py`.
- Your decision is needed on D10 (greedy-tie criterion) before M1 acceptance.

## Known issues in `kv-cache-plan.md` to correct later
1. **Suffix-prefill calibration needs both restored-prefix length and suffix
   length.** The plan does say T_prefill(n | L), but Week-3's `T_prefill(N−L | L)`
   and the feasibility check should state explicitly that the grid is indexed
   by (L, n). M0 already shows suffix cost is not a function of n alone.
2. **Only expected cache/storage failures should fall back.**
   Programming/CUDA errors must surface. The failure taxonomy says this, but
   the read-failure bullet ("读取/校验失败 → … 重算") must not be implemented
   as a broad `except`. Only enumerated exception types map to a miss.
3. **Memory/disk budgets must include staging buffers and temporary files.**
   That covers the host pinned staging, the device-side staging tensor that
   restore allocates before the cache copies it (D4), and NVMe tmp files
   before rename.
4. **File reads must distinguish OS page-cache effects from cold-device I/O.**
   Under WSL2 the Windows host cache sits underneath as well. Report the
   three categories and do not claim cold NVMe without verification.
5. **Request hit rate ≠ matched-prefix fraction.** The plan lists both under
   workload records, but result tables and resume bullets must not
   interchange them.
6. **Pipeline optimization is optional. Final correctness and documentation
   are required.** Week-4 P1 should never displace P0 verification; the plan's
   "整块砍P1" should be the default stance.
7. (Found in M0) The plan's dtype is fp16, but the checkpoint is bf16 (D2).
   The plan's greedy-agreement metric needs an explicit tie policy (D10).

## Next steps (not started)
- Decide D10. Then M1 per your next instruction. Candidates from the plan:
  `docs/design.md` draft, a 2-D (L, n) calibration grid, and a vLLM
  feasibility time-box.
- Investigate timing variance before any timing is used (clock locking is
  impossible under WSL. Record clocks via `nvidia-smi` during runs instead.)
