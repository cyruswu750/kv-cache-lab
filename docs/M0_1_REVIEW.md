# M0.1 Review — focused M0 audit and progression gate

Date: 2026-11-03. Repo: `/home/cyrus/projects/kv-cache-lab` (KV-cache project, not
the KV store). Git: `main`, **no commits** (as instructed). Pre-edit snapshot:
`backups/pre-m0_1-20261103T020921.tar.gz` (sha256 `064868f2…a218`, git-ignored).
Per-file manifest: `results/m0_1/pre_edit_manifest_20261103T020921.sha256` (45 files).
Original M0 evidence in `results/m0/` was made read-only (`chmod -R a-w`) and is unchanged.

Environment (re-checked, unchanged from M0): torch 2.14.1+cu130, transformers 5.22.1,
Python 3.14.4, RTX 4080 Laptop 12 GiB, driver 596.49, model
`Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, **bf16**, **SDPA**.

## Gate decision: **PASS** → M1 authorized
Passing the gate does **not** mean every M0 acceptance criterion passed. The
original strict M0 result is unchanged: `nat_long` **FAIL** (greedy 6/8),
reported separately below.

| # | Gate criterion | Result | Evidence |
|---|---|---|---|
| 1 | No unresolved aliasing/layout/position/mask/restore bug | **PASS** | code audit (below); default-position equality; position guard; 3 hardening fixes with regression tests |
| 2 | Export/restore byte checks pass | **PASS** | every C run in 9 bf16 splits × 3 repeats and 5 fp32 splits × 2 repeats (4,032 bf16 tensor comparisons, 0 mismatches) |
| 3 | Restored (C) and resident (B) agree in repeated controlled comparisons, no tolerance used | **PASS** | `torch.equal` on **all 8 step logits** and tokens, 3 repeats per split, including partial restore, and teacher-forced runs |
| 4 | Full-vs-split divergence fixed or reproducibly isolated to execution-shape numerics | **PASS (isolated)** | exact bf16 tie in A; 0.019 pre-rounding gap crosses a bf16 rounding boundary; fp32 instrument removes the gap (~1e-4) and the divergence |
| 5 | Negative controls show the relevant checks detect faults | **PASS** | 9 storage controls + 1 position control; byte check catches every storage fault, including ones invisible in logits |

No residual nondeterminism was observed. A, B, and C each repeated bitwise in
every split, in both bf16 and fp32.

## Commands run (exit codes)
| Command | Exit |
|---|---|
| `.venv/bin/python -m pytest -q` (CPU, after hardening) | 0 — 193 passed |
| pre-edit `kv_snapshot.py` + new regression tests (scratch copy) | 1 — the 3 new tests **fail on the old code**, as intended |
| `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --out results/m0_1/audit_bf16_20261103T021514.json` (audit run 1) | 0 |
| same, run 2 after the audit-tool fix → `audit_bf16_20261103T021650.json` | 0 |
| `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --fp32-diagnostic --out results/m0_1/audit_fp32_20261103T021748.json` | 0 |
| `HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py --out-dir results/m0_reruns/20261103T021916` (strict M0) | **1** — nat_long FAIL; every correctness line identical to the original M0 run2 |
| `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m gpu -v` | **1** — 16 passed, 1 failed (`m0_strict` nat_long) |

Logs: `results/m0_1/logs/`.

## Code audit (inspection + execution)
| Item | Finding |
|---|---|
| CPU snapshot independence | `torch.empty(device="cpu")` + `copy_` per chunk and layer; CPU test checks disjoint storage pointers; GPU source cache deleted and `empty_cache()` before restore. OK. |
| Hidden reuse of the original GPU cache | The C path only sees the `KVSnapshot`; `make_snapshot` deletes the source cache before `restore_to_device`. OK. |
| K/V layout, layer order | `[1, kv_heads=2, seq, 128]`; layers restored in index order; K/V not swapped. Controls show layer/KV swaps would be caught. OK. |
| dtype | Restore used `k0.dtype` for V and `copy_` silently casts → **hardened**: per-chunk dtype/shape check (regression test). |
| Shapes | `copy_` would silently **broadcast** a `[1,1,C,D]` chunk into `[1,2,C,D]` → **hardened** (regression test fails on the old code). The post-hoc byte check would have caught it; no M0 run triggered it. |
| Positions | Explicit `arange(L, N)`; HF default positions (none passed) give **bitwise identical** logits (`default_positions_bitwise_equal=True` at every Lmax). Guard rejects `start_pos != cache length`. OK. |
| Masks | 2D all-ones of length `L+n`. Verified in source: q_len == kv_len → `is_causal`; q_len = 1 → no mask; split (L>0, n>1) → explicit offset mask. OK. |
| Cache lengths, suffix slicing, chunk boundaries | asserted before/after each forward; suffix `ids[:, L:]`; boundary rule unchanged (CPU tests). OK. |
| Independent mutable caches | Every path and repeat builds its own cache; B with P == L extends the original prefix cache object, which is never reused. OK. |
| `compare_snapshot_to_cache` layer count | iterated cache layers without checking the count → **hardened** (raises; regression test). |
| Test collection / markers | `addopts = -m 'not gpu'`; `-m gpu` overrides it. Added explicit `m0_strict`, `m0_1`, `m1` markers. The M0 strict suite is still in `-m gpu`; no xfail anywhere. |
| Broad exception handling | Only `except (OSError, CalledProcessError)` in `env_report.py` (recorded) and `except ValueError` in the audit's guard-probe. No silent recompute. OK. |
| Rerun safety | `m0_experiment.py` wrote into `results/m0/` → added `--out-dir`, default `results/m0_reruns/<ts>/`. |

Audit-tool bug, found and fixed: in audit run 1, the fp32 dot products for
path B were read from the wrong `lm_head` call (B's prefix prefill also runs
inside the hook, shifting the index by one). Fixed by indexing from the end
(run 2). Every other field of runs 1 and 2 is identical (`diff` of the logs).
Run 1 files are kept.

## A / B / C comparison (bf16 baseline, 3 repeats each)
A = full prefill, B = split with resident GPU cache, C = same split, CPU export
→ delete → fresh-cache restore. P = prefix length actually computed (P > L = partial restore).

| Case | N | L | P | B==C all 8 steps ×3 | bytes | A−B first max_abs | A−B rel_l2 | A−B greedy (free) | argmax eq. teacher-forced on A | max_abs teacher-forced |
|---|---|---|---|---|---|---|---|---|---|---|
| nat_long | 426 | 256 | 256 | yes | ok | 0.3125 | 1.75e-2 | **6/8** | 7/8 | 0.50 |
| boundary_257 | 257 | 256 | 256 | yes | ok | 0.1875 | 1.17e-2 | 8/8 | 8/8 | 0.375 |
| boundary_511 | 511 | 256 | 256 | yes | ok | 0.5625 | 2.73e-2 | 8/8 | 8/8 | 0.5625 |
| boundary_513 | 513 | 256 | 256 | yes | ok | 0.2656 | 1.90e-2 | 8/8 | 8/8 | 0.375 |
| boundary_513 | 513 | 256 | 512 | yes | ok | 0.2188 | 1.42e-2 | 8/8 | 8/8 | 0.4375 |
| boundary_513 | 513 | 512 | 512 | yes | ok | 0.2500 | 1.87e-2 | 8/8 | 8/8 | 0.28 |
| nat_xlong | 611 | 256 | 256 | yes | ok | 0.2969 | 1.94e-2 | 8/8 | 8/8 | 0.406 |
| nat_xlong | 611 | 256 | 512 | yes | ok | 0.2969 | 1.78e-2 | 8/8 | 8/8 | 0.50 |
| nat_xlong | 611 | 512 | 512 | yes | ok | 0.3047 | 2.33e-2 | 8/8 | 8/8 | 0.4375 |

fp32 diagnostic: the same bf16 weights, upcast exactly. This is an
instrument, not a baseline change. Over 5 splits, A−B first-token max_abs
is **2.2e-5 … 1.1e-4** (rel_l2 ≤ 6e-6), greedy is 8/8 everywhere
(nat_long included), and B==C stays bitwise.

## The nat_long divergence (bf16), exact values
Shared history: prompt (426 tokens) + `"The autoregressive generation system"`
(6 generated tokens, identical on every path). Step 6 candidates: `312 ' re'`
and `7481 ' described'`.

| Path | logit(312) | logit(7481) | bf16 bits | fp32 dot of pre-lm_head hidden | argmax |
|---|---|---|---|---|---|
| A full | 23.75 | 23.75 | 0x41be / 0x41be (**exact tie**) | 23.76917 / 23.79568 | 312 (first index wins ties) |
| B resident split | 23.75 | 23.875 | 0x41be / 0x41bf | 23.76196 / 23.81442 | 7481 |
| C restored split | 23.75 | 23.875 | 0x41be / 0x41bf | 23.76196 / 23.81442 | 7481 |
| unchunked prefill of the full history (bf16) | 23.75 | 23.75 | tie | — | 312 |
| fp32 instrument, same history (A / B / C) | 23.74788 | 23.79324 | not bf16 values | — | 7481 (all three) |

- The tie is **exact** at the bf16 output: identical bit patterns, not a
  display-rounding effect.
- Before the bf16 output rounding, every path ranks `' described'` higher.
  bf16 spacing at 16–32 is 0.125, so 23.75 represents [23.6875, 23.8125].
  A's 23.7957 rounds into it and ties. B/C's 23.8144 crosses 23.8125 and
  rounds to 23.875.
- A−B differ by 0.28 (max over the vocab) at step 6. B−C differ by **0** (bitwise).
- Mechanism: the full prefill (q_len == kv_len) runs SDPA with `is_causal`
  and no mask. The split suffix runs with an explicit offset mask, i.e. a
  different kernel path and accumulation order. In fp32, that difference
  is ~1e-4 and the ranking is stable.
- Conclusion: the full-vs-split mismatch is an execution-shape bf16 rounding
  effect. Storage contributes nothing to it (B == C bitwise). It is **not
  fixed**: the strict M0 criterion still fails, and we did not change
  argmax or tie behavior, round logits, or loosen the tolerance.

## Audit of atol = 1.45 / rtol = 0.02
- **Origin:** `scripts/m0_calibrate.py`, rule fixed in decisions D7 *before*
  calibration: `atol = max(0.25, ceil_0.05(4·M))`, `rtol = 0.02`.
  M = 0.3535 is the largest first-token max_abs of full vs GPU-resident split.
- **Inputs:** `CALIBRATION_TEXT` (189 tokens) cycled to N = 300/400/600/700;
  disjoint from every acceptance fixture.
- **Identical histories:** yes. Calibration compares first-token logits after
  prefilling the same prompt, with no decoding, so post-divergence errors could
  not inflate M.
- **Can it conceal implementation errors? Yes.** In the bf16 audit, all of
  these stay within atol:

  | Fault | A−C max_abs | atol detects? |
  |---|---|---|
  | wrong prefix, one token changed (nat_long) | 0.94 | **no** |
  | query positions shifted +1 (nat_long / boundary_513) | 0.33 / 0.99 | **no** |
  | chunk 0 ↔ chunk 1 swapped (boundary_513) | 0.25 | **no** |
  | one-bit LSB flip | 0.25 | **no** |

  The tolerance is a weak gross-error check only. It must not be used as
  evidence that storage is correct. M1 uses byte identity, identity keys, and
  bitwise restored-vs-resident checks instead.

## Negative controls (bf16)
In the fp32 run, every byte-check outcome is identical. Logit visibility
differs slightly: for example, the nat_long LSB flip was bitwise invisible
in fp32 (`audit_fp32_*.json`).
Byte check = compare the restored snapshot with the K/V that was written
(the resident prefix cache).

| Control | nat_long L=256: byte check | ==B bitwise | atol detects | boundary_513 L=512: byte check | ==B bitwise | atol detects | Detecting invariant |
|---|---|---|---|---|---|---|---|
| valid restore | pass (0/56) | yes | — | pass (0/112) | yes | — | — |
| LSB flip, 1 element | **fail** (1) | no | no | **fail** (1) | **yes** (no logit changed at all) | no | byte check / checksum |
| 1 element := 1e3 | fail (1) | no | yes | fail (1) | no | yes | byte check / checksum |
| wrong prefix, other text | fail (56) | no | yes | fail (112) | no | yes | identity key; byte check vs written |
| wrong prefix, 1 token changed | fail (56) | no | **no** | fail (56: chunk 1 only) | no | yes | identity key (token chain); byte check vs written |
| layers 0↔1 swapped | fail (4) | no | yes | fail (8) | no | yes | byte check (per-layer) |
| K↔V swapped, layer 5 | fail (2) | no | yes | fail (4) | no | yes | byte check |
| chunks 0↔1 swapped | — | — | — | fail (112) | no | **no** | byte check / per-chunk identity |
| zeroed K/V | fail (56) | no | yes | fail (112) | no | yes | byte check |
| query positions +1 | (K/V valid) | no | **no** | (K/V valid) | no | **no** | `prefill` guard: `ValueError: cache length L != start_pos L+1` |

Why some faults are invisible in the logits:
- Attention over cached keys is order-independent once RoPE is baked into
  K, so a chunk swap only reorders a softmax sum.
- A low mantissa bit can be absorbed by bf16 rounding.

Not every perturbation changes a greedy token. The right detector is the
invariant listed in the last column.

## Original M0 status (unchanged)
`scripts/m0_experiment.py` exits 1, and `pytest -m "gpu and m0_strict"` has
10 passed, 1 failed (nat_long greedy 6/8). The criteria in decisions D7 are
unchanged. D10 (tie policy) is still **your open decision**. This audit
supplies the evidence for it, but no rule was adopted.
(M1.1 note, 2026-11-03: D10 is now decided. There is no tie rule, and the
M0 strict FAIL is kept as recorded; see decisions D10.)

## Changed files in M0.1
- `kvlab/kv_snapshot.py`: restore validates per-chunk shape/dtype/device;
  compare validates layer count and length.
- `tests/test_snapshot_cpu.py`: +3 regression tests.
- `tests/gpu/test_m0_1_controls.py` (new, `gpu and m0_1`): 6 tests.
- `tests/gpu/test_gpu_roundtrip.py`: marker `m0_strict` added; assertions unchanged.
- `pyproject.toml`: marker definitions.
- `scripts/m0_1_audit.py` (new).
- `scripts/m0_experiment.py`: `--out-dir`, so `results/m0/` is never overwritten.
- `docs/M0_WALKTHROUGH.md`: premature attribution note removed.
- `.gitignore`: `backups/`.
