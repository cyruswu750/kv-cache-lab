#!/usr/bin/env python3
"""The bounded M5 GPU correctness gate (docs/M5_PROTOCOL_DRAFT.md §2 A). NOT RUN as of M5b preparation; each
execution needs separate owner approval and runs exclusively on the GPU.

  HF_HOME=<weights cache> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m5_gate.py --out-dir local/m5_gate_<ts>

Runs, in order, under ONE monotonic deadline of GATE_CAP_S, reusing the M4 gate runner (scripts/m4_gate.py
run_gate: fail-fast, process-group cleanup, gate_record.json):
  1. pytest -x -m "gpu and m1"   (the M1 tiered suite: the policy=None engine path on the real model, both backends)
  2. pytest -x -m "gpu and m3"   (batched-restore checks)
  3. pytest -x -m "gpu and m5" tests/gpu/test_m5_policy.py   (the M5 policy paths and the harness smoke session)
Exit 0 = PASS; 3 = FAIL (any failure, timeout or stop). No retry. Elapsed times are a feasibility diagnostic only.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE_CAP_S = 900


def _m4_gate():
    spec = importlib.util.spec_from_file_location("m4_gate", ROOT / "scripts" / "m4_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate_commands(out_dir: Path, python: str = sys.executable) -> list:
    bt = Path(out_dir) / "basetemp"
    base = [python, "-m", "pytest", "-x", "-v", "-p", "no:cacheprovider"]
    return [("m1", base + [f"--basetemp={bt}/m1", "-m", "gpu and m1"]),
            ("m3", base + [f"--basetemp={bt}/m3", "-m", "gpu and m3"]),
            ("m5", base + [f"--basetemp={bt}/m5", "-m", "gpu and m5", "tests/gpu/test_m5_policy.py"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists():
        ap.error(f"{out} exists")
    rec = _m4_gate().run_gate(gate_commands(out), GATE_CAP_S, out)
    print(json.dumps({k: rec[k] for k in ("status", "stop", "total_elapsed_s")}, default=str))
    return 0 if rec["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
