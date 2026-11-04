# kv-cache-lab — Cost-Aware Tiered KV Cache for LLM Inference

**Status: M0, M0.1, M1, M1.1, M1.2 (runner validation fix), M2 (performance calibration + offline cost model), M2.1 (measurement audit + slow-mode diagnosis) and M3 (batched KV restoration, opt-in; analysis validation fix VAL-006) done. M3_CONFIRM_V1 (preregistered confirmation of the batched backend) measured: confirmed; the default backend stays `baseline`.**

| Milestone | What it covers | Status |
|---|---|---|
| M0 | Environment check plus one real GPU→CPU→GPU KV round trip | done; strict suite still **FAILs** on one case (`nat_long`, a bf16 tie) |
| M0.1 | Audit of M0 and the progression gate | gate **PASS** |
| M1 | Cache identity, validated chunk files, bounded pinned-host and disk tiers, synchronous always-restore | done; all M1 tests pass |
| M1.1 | Fixes from external review: disk deletion accounting, marker symlink, host accounting terms; D10 recorded | done; all tests pass except the known M0 strict case |
| M1.2 | Evidence-runner validation fix (BUGLOG VAL-004): strict ENOSPC record, forward-call acceptance, derived counts | done; corrected run `results/m1_2/m1_run_20261103T121800.json` |
| M2 | Synchronous first-token latency for full / host / disk paths; cost model fitted on calibration, scored on held-out cases | done (no break-even established; model describes the baseline backend; claims qualified by M2.1) |
| M2.1 | Audit of M2 timing/scoring; dependence-aware reanalysis (BUGLOG VAL-005); interleaved diagnosis of the restore slow mode | done (slow mode associated with higher host-side CUDA-call cost; root cause open) |
| M3 | Opt-in `batched` restore backend: one contiguous H2D per chunk into bounded chunk-major device staging, one placement per group; strict A/B against the baseline; interleaved 5-session comparison | done (restore calls 896 → 16 memcpy at 16 chunks; median e2e 1.07–2.29× faster in this sample). `supports_adoption` is **false**: under the frozen rule, (1024, 1) is inconclusive (R3, measured stretch too short), so the default stays `baseline`. The analysis now validates evidence against the frozen spec first (BUGLOG VAL-006); the reanalysis reproduced every number and the verdict |
| M3_CONFIRM_V1 | Preregistered confirmation of (1024, 1) and the fresh (1280, 96): own validator spec, premeasurement freeze manifest, budgeted batch wrapper | done: one batch, 8,640 measured samples, valid; all 8 primary verdicts batched faster (median paired d −14.0 to −16.8 ms), so `m3_confirm_v1_confirmed` is true. Restoration microbenchmark only; not pooled with M3, whose `supports_adoption` stays **false**; the default stays `baseline` (`docs/results.md`; D31) |

Not implemented yet: a runtime cost-aware policy (M2's model is offline only),
async pipelines, concurrency, service-level TTFT, and vLLM integration (scope: `docs/design.md` §1).

- Results, scope and limitations per milestone: `docs/results.md`
- Design: `docs/design.md`
- Trade-offs: `docs/decisions.md`
- Bugs found and fixed: `docs/BUGLOG.md`
- Test-by-scenario table: `docs/correctness-matrix.md`
- Environment and observed cache API: `docs/environment.md`
- M2 performance protocol: `docs/M2_PROTOCOL.md`
- M2.1 diagnostic plan: `docs/M2_1_DIAGNOSTIC_PLAN.md`
- M3 design note: `docs/M3_DESIGN.md`; frozen comparison protocol: `docs/M3_PROTOCOL.md`
- M3_CONFIRM_V1 confirmation protocol (frozen before measurement): `docs/M3_CONFIRM_PROTOCOL.md`; dated clarifications: `docs/M3_CONFIRM_CLARIFICATIONS.md`

Reference harness: Hugging Face Transformers + PyTorch (decisions D1).
Model: `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, **bf16**, **SDPA**, chunk = 256 tokens.

## Environment (observed; details in `docs/environment.md`)
- **OS/CPU:** WSL2 Ubuntu 26.04.1, Python 3.14.4, 7.4 GiB RAM.
- **GPU:** RTX 4080 Laptop GPU, 12 GiB, Windows driver 596.49.
- **Packages:** `torch==2.14.1+cu130` (official cu130 index), `transformers==5.22.1`, `pytest==9.2.0`.
  The full list is in `results/m0/pip_freeze.txt`.

## Layout
```
kvlab/boundaries.py        restore-length rule: L % C == 0 and 0 < L <= N-1
kvlab/model.py             pinned model loading; explicit-position prefill; greedy decode
kvlab/kv_snapshot.py       M0: DynamicCache <-> chunked CPU snapshot, byte comparison
kvlab/identity.py          M1: namespace (canonical JSON + SHA-256), chained chunk keys
kvlab/chunk_format.py      M1: chunk file v1 (LE header, CRC32, raw payload), validated reads
kvlab/host_store.py        M1: bounded pinned-host slab, LRU, leases
kvlab/disk_store.py        M1: bounded owned disk dir, temp+rename, LRU, I/O injection points
kvlab/tiered_cache.py      M1: store/restore across tiers, counters, budgets; M3: restore backends (baseline default, batched)
kvlab/always_restore.py    M1: synchronous always-restore request flow
kvlab/resident_reference.py  GPU-only replay used for strict (bitwise) verification
kvlab/perf.py              M2: grid/split, sample checks, statistics, cost model, scoring (CPU)
kvlab/perf_harness.py      M2: GPU benchmark harness (tier states, timed path, page-cache handling)
kvlab/perf_stats.py        M2.1: block bootstrap, inconclusive classification, penalties, policy totals (CPU)
kvlab/diag_harness.py      M2.1: diagnostic paths/modes (full_nocache, CUDA-event marks, probes); diagnostic only
kvlab/m3_bench.py          M3: baseline vs batched benchmark harness on the unchanged M2 timed region
kvlab/m3_stats.py          M3: stratified block bootstrap, R1-R3 verdicts, regime durations (CPU)
kvlab/m3_validate.py       M3 / VAL-006: fail-closed evidence validation against explicit specs, incl. warmups (CPU)
kvlab/m3_confirm.py        M3_CONFIRM_V1: premeasurement freeze manifest, budgeted batch wrapper, their checks (CPU, no torch)
scripts/                   m0_*, m0_1_audit, m1_run, m1_mutation_check, m2_measure, m2_fit_predict, m2_score, m2_spike_probe, m2_1_reanalyze, m2_1_diagnose, m2_1_analyze_diag, m2_1_regimes, m3_measure, m3_analyze, m3_mutation_check, m3_confirm, m3_confirm_reconstruct, env_report
tests/                     CPU tests; tests/gpu/ GPU tests (markers: gpu + m0_strict | m0_1 | m1 | m3)
results/m0/ (read-only), results/m0_1/, results/m1/, results/m1_1/, results/m1_2/, results/m2/, results/m2_1/, results/m3/, results/m3_confirm/   machine-readable evidence and logs
```

## Setup (no sudo; nothing outside the project is modified)
```bash
cd ~/projects/kv-cache-lab
python3 -m venv --without-pip .venv               # system python has no ensurepip
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python -I /tmp/get-pip.py
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130"
.venv/bin/python -m pip install "transformers==5.22.1" "pytest==9.2.0"
HF_HOME=$PWD/.hf_cache .venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
```
- `kvlab/model.py` uses `HF_HOME=.hf_cache/` unless you set `HF_HOME` yourself.
- The cache directories `.hf_cache/` and `.kvcache/` are git-ignored, as are `.venv/` and `backups/`.
- After the download, prefix the commands below with `HF_HUB_OFFLINE=1`.

## Tests and reproduction
```bash
# CPU: boundaries, snapshots, identity, file format, host/disk stores, tiered logic (no GPU)
.venv/bin/python -m pytest -q                                          # 408 passed at M3; 436 passed after the VAL-006 fix (last full CPU run)
.venv/bin/python -m pytest -q tests/test_m3_analyze_validation.py tests/test_m3_stats.py   # 56 passed after the VAL-006 warmup follow-up (focused run; full suite not rerun)

# GPU suites (each selectable; `-m gpu` runs all of them)
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m1" -v          # M1 integration, both restore backends: 23 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m3" -v          # M3 batched-restore checks: 17 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_1" -v        # M0.1 regressions: 6 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_strict" -v   # original M0: 10 passed, 1 FAILED (known)

# Evidence scripts
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_run.py --out results/m1_2/m1_run_<ts>.json [--restore-backend batched]  # exit 0 = acceptance (19 strict records)
.venv/bin/python scripts/m1_run.py --validate results/m1_2/m1_run_<ts>.json                 # re-check a saved report
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --out results/m0_1/audit_bf16_<ts>.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --fp32-diagnostic --out results/m0_1/audit_fp32_<ts>.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py --smoke     # writes results/m0_reruns/<ts>/
.venv/bin/python scripts/m1_mutation_check.py /tmp/kvlab-mutants       # CPU mutants: expect all 21 KILLED
.venv/bin/python scripts/m3_mutation_check.py /tmp/kvlab-m3-mutants --gpu  # restore-backend mutants: expect 10/10 KILLED
```

What the GPU acceptance checks mean (design §9):
- Storage is verified **strictly**: restored K/V bytes must match what was
  written, and every step's logits must match a resident-GPU replay of the
  same request sequence bitwise.
- Every request makes exactly one prefill plus `steps − 1` decode forward
  calls. Cache failures never repeat or abort inference.
- Comparison with full recomputation is report-only. In bf16, exact ties can
  flip greedy tokens; see `docs/results.md` (M0.1 and M1).
- Decision D10 (2026-11-03):
  - standard argmax; no tie rule, rounding, tolerance inflation or xfail;
  - the original M0 strict FAIL is kept as recorded.

  Restored output is **not** claimed to equal full-prefill output.
- Performance numbers exist only for M2 and M3. They are synchronous, single-request,
  tokenized-input first-token latencies on one WSL2 laptop GPU, with platform
  drift documented in `docs/results.md` (M2). They are not service TTFT and
  not cold-NVMe or p99 figures. M2.1 qualifies the M2 decision claims (block-bootstrap
  intervals; mean-based penalties) and shows that restore-path latency on this
  platform switches between regimes of different CPU-side CUDA call cost.
- M3 compares the baseline and batched restore backends within the same
  sessions (`docs/M3_PROTOCOL.md`). Its numbers describe that sample only. The
  slow mode remains, and its root cause is not established.

```bash
# M2 (one GPU process at a time; new run directory each time)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m2_measure.py --split calibration --run-dir results/m2/run_<ts>
.venv/bin/python scripts/m2_fit_predict.py --run-dir results/m2/run_<ts>          # freezes model + predictions
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m2_measure.py --split evaluation --run-dir results/m2/run_<ts> --reps 40
.venv/bin/python scripts/m2_score.py --run-dir results/m2/run_<ts>

# M2.1 (CPU reanalysis of a saved M2 run; diagnostic sessions, one GPU process at a time)
.venv/bin/python scripts/m2_1_reanalyze.py --m2-run results/m2/run_20261103T131543 --out results/m2_1/<dir>/reanalysis.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m2_1_diagnose.py --out-dir results/m2_1/diag_<ts> --session s1 [--probes] [--no-monitor] [--profile]
.venv/bin/python scripts/m2_1_analyze_diag.py --run-dir results/m2_1/diag_<ts>

# M3 (one GPU process at a time; new run directory each time)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m3_measure.py --out-dir results/m3/eval_<ts> --session s1 --cases 256:32,1024:1,1536:128,3072:256,4096:64 --blocks 2 --warmup 2 --reps 12 --seed 31001
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m3_measure.py --out-dir results/m3/eval_<ts> --session prof --profile --cases 256:32,1536:128,4096:64
.venv/bin/python scripts/m3_analyze.py --run-dir results/m3/eval_<ts> [--out-dir <new dir>]   # validates first: 0 valid, 3 invalid evidence, 4 failed correctness check

# M3_CONFIRM_V1 frozen analysis (CPU). Current tree, valid while the analysis files keep their frozen bytes:
.venv/bin/python scripts/m3_analyze.py --spec M3_CONFIRM_V1 --run-dir results/m3_confirm/M3_CONFIRM_V1_run --out-dir <new dir> --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216
# After later changes: reconstruct the frozen tree (verified against the manifest), then run its own analyzer (docs/results.md)
.venv/bin/python scripts/m3_confirm_reconstruct.py --out <new dir> --tree-rev 13afde2 --expect-freeze-sha256 dd58b894ebc9fc5d8d4e3c5bdc837d88637f3b2a224f1a1ce6e316f3e98b8216
```
