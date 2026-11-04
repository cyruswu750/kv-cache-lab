# kv-cache-lab — Cost-Aware Tiered KV Cache for LLM Inference

**Status: M0, M0.1, M1, M1.1, M1.2 (runner validation fix) M2 (performance calibration + offline cost model) and M2.1 (measurement audit + slow-mode diagnosis) done.**

| Milestone | What it covers | Status |
|---|---|---|
| M0 | Environment check plus one real GPU→CPU→GPU KV round trip | done; strict suite still **FAILs** on one case (`nat_long`, a bf16 tie) |
| M0.1 | Audit of M0 and the progression gate | gate **PASS** |
| M1 | Cache identity, validated chunk files, bounded pinned-host and disk tiers, synchronous always-restore | done; all M1 tests pass |
| M1.1 | Fixes from external review: disk deletion accounting, marker symlink, host accounting terms; D10 recorded | done; all tests pass except the known M0 strict case |
| M1.2 | Evidence-runner validation fix (BUGLOG VAL-004): strict ENOSPC record, forward-call acceptance, derived counts | done; corrected run `results/m1_2/m1_run_20261103T121800.json` |
| M2 | Synchronous first-token latency for full / host / disk paths; cost model fitted on calibration, scored on held-out cases | done; `docs/M2_REVIEW.md` (no break-even established; claims qualified by M2.1) |
| M2.1 | Audit of M2 timing/scoring; dependence-aware reanalysis (BUGLOG VAL-005); interleaved diagnosis of the restore slow mode | done; `docs/M2_1_REVIEW.md` (slow mode narrowed to CPU-side CUDA call cost; root cause open) |

Not implemented yet: a runtime cost-aware policy (M2's model is offline only),
async pipelines, concurrency, service-level TTFT, and vLLM integration (see `kv-cache-plan.md`).

- Design: `docs/design.md`
- Trade-offs: `docs/decisions.md`
- Bugs found and fixed: `docs/BUGLOG.md`
- Results and limitations: `docs/M0_REVIEW.md`, `docs/M0_1_REVIEW.md`, `docs/M1_REVIEW.md`, `docs/M1_1_REVIEW.md`, `docs/M2_REVIEW.md`, `docs/M2_1_REVIEW.md`
- M2 performance protocol: `docs/M2_PROTOCOL.md`; Chinese walkthrough: `docs/M2_WALKTHROUGH.md`
- M2.1 diagnostic plan: `docs/M2_1_DIAGNOSTIC_PLAN.md`; Chinese explanation: `docs/M2_1_WALKTHROUGH.md`
- Test-by-scenario table: `docs/correctness-matrix.md`

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
kvlab/tiered_cache.py      M1: store/restore across tiers, counters, budgets
kvlab/always_restore.py    M1: synchronous always-restore request flow
kvlab/resident_reference.py  GPU-only replay used for strict (bitwise) verification
kvlab/perf.py              M2: grid/split, sample checks, statistics, cost model, scoring (CPU)
kvlab/perf_harness.py      M2: GPU benchmark harness (tier states, timed path, page-cache handling)
kvlab/perf_stats.py        M2.1: block bootstrap, inconclusive classification, penalties, policy totals (CPU)
kvlab/diag_harness.py      M2.1: diagnostic paths/modes (full_nocache, CUDA-event marks, probes); diagnostic only
scripts/                   m0_*, m0_1_audit, m1_run, m1_mutation_check, m2_measure, m2_fit_predict, m2_score, m2_spike_probe, m2_1_reanalyze, m2_1_diagnose, m2_1_analyze_diag, m2_1_regimes, env_report
tests/                     CPU tests; tests/gpu/ GPU tests (markers: gpu + m0_strict | m0_1 | m1)
results/m0/ (read-only), results/m0_1/, results/m1/, results/m1_1/, results/m1_2/, results/m2/, results/m2_1/   machine-readable evidence and logs
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
.venv/bin/python -m pytest -q                                          # 370 collected (345 passed at M2; M2.1 ran only its focused tests)

# GPU suites (each selectable; `-m gpu` runs all of them)
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m1" -v          # M1 integration: 12 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_1" -v        # M0.1 regressions: 6 passed
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m "gpu and m0_strict" -v   # original M0: 10 passed, 1 FAILED (known)

# Evidence scripts
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m1_run.py --out results/m1_2/m1_run_<ts>.json  # exit 0 = acceptance (19 strict records)
.venv/bin/python scripts/m1_run.py --validate results/m1_2/m1_run_<ts>.json                 # re-check a saved report
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --out results/m0_1/audit_bf16_<ts>.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_1_audit.py --fp32-diagnostic --out results/m0_1/audit_fp32_<ts>.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/m0_experiment.py --smoke     # writes results/m0_reruns/<ts>/
.venv/bin/python scripts/m1_mutation_check.py /tmp/kvlab-mutants       # CPU mutants: expect all 21 KILLED
```

What the GPU acceptance checks mean (design §9):
- Storage is verified **strictly**: restored K/V bytes must match what was
  written, and every step's logits must match a resident-GPU replay of the
  same request sequence bitwise.
- Every request makes exactly one prefill plus `steps − 1` decode forward
  calls. Cache failures never repeat or abort inference.
- Comparison with full recomputation is report-only. In bf16, exact ties can
  flip greedy tokens; see M0_1_REVIEW and M1_REVIEW §7.
- Decision D10 (2026-11-03):
  - standard argmax; no tie rule, rounding, tolerance inflation or xfail;
  - the original M0 strict FAIL is kept as recorded.

  Restored output is **not** claimed to equal full-prefill output.
- Performance numbers exist only for M2. They are synchronous, single-request,
  tokenized-input first-token latencies on one WSL2 laptop GPU, with platform
  drift documented in `docs/M2_REVIEW.md` §6. They are not service TTFT and
  not cold-NVMe figures. M2.1 qualifies the M2 decision claims (block-bootstrap
  intervals; mean-based penalties) and shows that restore-path latency on this
  platform switches between regimes of different CPU-side CUDA call cost.

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
```
