#!/usr/bin/env python3
"""M5: validate calibration evidence, fit the cost model, and freeze predictions for the evaluation cases (CPU).

  .venv/bin/python scripts/m5_fit.py --cal-run-dir results/m5/M5_CAL_V1_run --out-dir results/m5/M5_FIT_V1

Writes into a NEW directory (never overwrites):
- validation.json, always;
- calibration_means.json, cost_model.json and predictions.json, only if the calibration evidence is valid.

The predictions are computed from the cost model as reloaded from disk, so the evaluation sessions decide with
exactly those bytes. Exit 0 = fitted; 3 = invalid evidence (nothing fitted); 4 = strict mismatch (nothing fitted).
Executed once, for M5_FIT_V1 (docs/results.md, M5).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m5_fit as mf  # noqa: E402
from kvlab import m5_spec as sp  # noqa: E402
from kvlab import m5_validate as mv  # noqa: E402
from kvlab import restore_cost as rc  # noqa: E402


def write_new(path: Path, obj) -> None:
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def run(cal_run_dir: Path, out_dir: Path, *, require_cuda: bool = True, cal=None, ev=None) -> int:
    cal = cal or sp.calibration_spec()
    ev = ev or sp.evaluation_spec(cal)
    out_dir.mkdir(parents=True)  # refuses an existing directory
    v = mv.validate(cal_run_dir, cal, require_cuda=require_cuda)
    write_new(out_dir / "validation.json", {k: v[k] for k in ("exit", "valid", "problems", "strict_problems")})
    if v["exit"] != 0:
        return v["exit"]
    write_new(out_dir / "calibration_means.json", mf.calibration_cases(cal, v["sessions"]))
    artifact = mf.fit_artifact(cal, v["sessions"], ev, run_dir=cal_run_dir)
    rc.save_artifact(artifact, out_dir / "cost_model.json")
    model = rc.load_model(out_dir / "cost_model.json")
    write_new(out_dir / "predictions.json", mf.make_predictions(ev, model, created_unix=time.time()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cal-run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    return run(Path(a.cal_run_dir), Path(a.out_dir))


if __name__ == "__main__":
    sys.exit(main())
