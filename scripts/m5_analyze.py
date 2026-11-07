#!/usr/bin/env python3
"""M5: validate evaluation evidence against the frozen cost model and predictions, then analyse it (CPU).

  .venv/bin/python scripts/m5_analyze.py --eval-run-dir results/m5/M5_EVAL_V1_run \\
      --fit-dir results/m5/M5_FIT_V1 --out-dir results/m5/M5_EVAL_V1_analysis

Writes into a NEW directory:
- validation.json, always;
- analysis.json and analysis.md, only for valid evidence. Invalid evidence gets no performance analysis.

Exit codes: the validator's (0 valid, 3 invalid, 4 strict mismatch).
Executed once, for M5_EVAL_V1 (docs/results.md, M5).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m5_analysis as ma  # noqa: E402
from kvlab import m5_spec as sp  # noqa: E402
from kvlab import m5_validate as mv  # noqa: E402
from kvlab import restore_cost as rc  # noqa: E402


def write_new(path: Path, text: str) -> None:
    with open(path, "x") as f:
        f.write(text)


def run(eval_run_dir: Path, fit_dir: Path, out_dir: Path, *, require_cuda: bool = True, ev=None) -> int:
    ev = ev or sp.evaluation_spec()
    out_dir.mkdir(parents=True)
    artifact = rc.load_model(fit_dir / "cost_model.json").artifact      # validated (digest, schema) on load
    preds = mc.load_json_finite((fit_dir / "predictions.json").read_text())
    v = mv.validate(eval_run_dir, ev, predictions=preds, artifact=artifact, require_cuda=require_cuda)
    write_new(out_dir / "validation.json", json.dumps({k: v[k] for k in ("exit", "valid", "problems",
                                                                          "strict_problems")}, indent=1))
    if v["exit"] != 0:
        return v["exit"]
    res = ma.analyze(ev, v["sessions"], preds, v)
    write_new(out_dir / "analysis.json", json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n")
    write_new(out_dir / "analysis.md", ma.render_md(res))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-run-dir", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    return run(Path(a.eval_run_dir), Path(a.fit_dir), Path(a.out_dir))


if __name__ == "__main__":
    sys.exit(main())
