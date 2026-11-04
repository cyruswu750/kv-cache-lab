"""M2: fit the cost model on calibration samples only, freeze it, and write predictions
for the held-out evaluation cases BEFORE any evaluation sample exists.

Usage: .venv/bin/python scripts/m2_fit_predict.py --run-dir results/m2/<run>
Writes model.json and predictions.json (never overwritten). CPU only.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kvlab import perf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    if (run / "evaluation_samples.json").exists() or (run / "evaluation_gpu_monitor.csv").exists():
        sys.exit("evaluation data already exists in this run; refusing to fit (the model must be frozen first)")
    for name in ("model.json", "predictions.json"):
        if (run / name).exists():
            sys.exit(f"refusing to overwrite {run / name}")
    cal_path = run / "calibration_samples.json"
    cal = json.loads(cal_path.read_text())
    valid, excluded = perf.partition(cal["samples"])
    valid = [s for s in valid if (s["L"], s["S"]) in perf.CALIBRATION]
    med = perf.case_medians(valid)
    model = perf.fit_cost_model(med)
    model["calibration_samples_sha256"] = hashlib.sha256(cal_path.read_bytes()).hexdigest()
    model["fitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model["n_valid_calibration_samples"], model["n_excluded_calibration_samples"] = len(valid), len(excluded)
    model["sha256"] = perf.model_digest(model)
    created = time.time()
    preds = perf.make_predictions(model, perf.EVALUATION, extra={
        "created_unix": created, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "in_sample_calibration": [perf.predict(model, path, L, S) for (L, S) in perf.CALIBRATION
                                  for path in perf.PATHS],
        "scan": perf.scan_predictions(model)})
    (run / "model.json").write_text(json.dumps(model, indent=1))
    (run / "predictions.json").write_text(json.dumps(preds, indent=1))
    print(f"model sha256 {model['sha256']}\npredictions sha256 {preds['sha256']}")
    for name, comp in model["components"].items():
        coefs = ", ".join(f"{f}={c:.4g}" for f, c in zip(comp["features"], comp["coef_ms"]))
        err = comp.get("in_sample_rel_err")
        print(f"  {name:16s} {coefs}" + (f"   in-sample rel err max {max(abs(e) for e in err):.1%}" if err else ""))
    print(f"  offsets {model['instrumentation_offset_ms']}")


if __name__ == "__main__":
    main()
