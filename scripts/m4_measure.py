#!/usr/bin/env python3
"""One M4-B evaluation session (GPU). NOT EXECUTED in M4-B preparation: GPU work needs separate
approval, and normally runs only through `scripts/m4_eval.py run` after the freeze.

  HF_HOME=<weights cache> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m4_measure.py \\
      --spec M4_EVAL_V1 --out-dir results/m4/M4_EVAL_V1_run --session e01 --seed 44001

Writes <out-dir>/session_<id>.json (never overwrites) and a read-only nvidia-smi monitor CSV.
Exit 0: the session finished and its own digest comparison found no strict mismatch. Exit 4:
a strict mismatch was recorded (the report is still written; the batch then stops, D31). Any
exception propagates (exit 1, no report). Validity is decided only by scripts/m4_analyze.py, which
requires every session's exit to equal 4 if and only if it recomputes a mismatch there, else 0.
"""
import argparse
import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--seed", required=True, type=int)
    a = ap.parse_args()
    from kvlab import m4_spec as ms
    spec = ms.SPECS[a.spec]
    out = ROOT / a.out_dir / f"session_{a.session}.json"
    if out.exists():
        ap.error(f"{out} exists")
    import torch
    from kvlab import m3_confirm as mc
    from kvlab import m4_bench as mb
    from kvlab import m4_freeze as mf
    from kvlab import model as km
    from kvlab.identity import namespace_for
    from kvlab.perf_harness import environment, start_gpu_monitor, stop_gpu_monitor
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for an evaluation session")
    source = mc.source_identity(ROOT)  # before anything runs
    stack = mf.current_stack(ROOT)
    model, _tok = km.load_model()
    ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, spec["chunk_size"])
    disk_root = ROOT / ".kvcache" / "m4" / Path(a.out_dir).name / a.session
    monitor = start_gpu_monitor(out.with_name(f"session_{a.session}.gpu_monitor.csv"))
    try:
        env = {"stack": stack, "harness": environment(ROOT, disk_root.parent), "hf_home": os.environ.get("HF_HOME")}
        rep = mb.run_session(model, ns, spec, a.session, a.seed, disk_root, source=source, environment=env)
    finally:
        stop_gpu_monitor(monitor)
    mism = rep["strict_mismatches"]  # recorded by run_session; the validator recomputes it
    with open(out, "x") as f:
        f.write(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"session": a.session, "strict_mismatches": mism}))
    return 4 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
