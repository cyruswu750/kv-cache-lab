#!/usr/bin/env python3
"""M5 calibration / evaluation sessions and batches (docs/M5_PROTOCOL_DRAFT.md).
Each formal batch (M5_CAL_V1, M5_EVAL_V1) was executed once (docs/results.md, M5).

  .venv/bin/python scripts/m5_run.py plan --split calibration|evaluation          # CPU only: counts, design, domain
  HF_HOME=<weights> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m5_run.py batch --split calibration \\
      --run-dir results/m5/M5_CAL_V1_run
  HF_HOME=<weights> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m5_run.py batch --split evaluation \\
      --run-dir results/m5/M5_EVAL_V1_run --cost-model <fit dir>/cost_model.json --predictions <fit dir>/predictions.json

`batch` refuses unless every pre-launch check passes:
- the source tree is clean;
- the run directory is new;
- for evaluation, the frozen cost model and predictions are bound to this evaluation spec.

It then writes spec.json and batch_plan.json and runs the planned `session` commands through
kvlab.m3_confirm.run_batch: one deadline, a stop on the first nonzero exit, no retries. `session` (one GPU
process) is launched only by `batch`. It writes session_<sid>.json even when it fails (status "failed", with the
error) and never overwrites anything.
"""
import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m5_spec as sp  # noqa: E402

SCRATCH = ROOT / ".kvcache" / "m5"


def spec_for(split: str) -> dict:
    cal = sp.calibration_spec()
    return cal if split == "calibration" else sp.evaluation_spec(cal)


def load_frozen(cost_model_path, predictions_path, spec) -> tuple:
    """(model, predictions, problems): the frozen files, their bindings to `spec`, and the predictions recomputed
    from the cost model (kvlab.m5_validate.prediction_problems, the same check the validator applies)."""
    from kvlab import m5_validate as mv
    from kvlab import restore_cost as rc
    model = rc.load_model(cost_model_path)
    preds = mc.load_json_finite(Path(predictions_path).read_text())
    return model, preds, mv.prediction_problems(spec, preds, model.artifact)


def cmd_plan(a) -> int:
    spec = spec_for(a.split)
    out = {"name": spec["name"], "spec_sha256": sp.digest(spec), "counts": spec["counts"], "budget": spec["budget"],
           "cases": [{"id": c["case_id"], "N": c["prompt"]["length"], "prefix": "".join(t[0].upper() for t in
                                                                                       c["feasible_tiers"]),
                      "candidates": len(c["candidates"]), "admitted": len(c["admitted"])} for c in spec["cases"]]}
    if spec["split"] == "calibration":
        out["design"], out["domain"] = spec["design"], spec["domain"]
    print(json.dumps(out, indent=1))
    return 0


def cmd_session(a) -> int:
    import torch
    from kvlab import m5_bench as mb
    from kvlab import model as km
    from kvlab.identity import namespace_for
    spec = spec_for(a.split)
    run_dir = ROOT / a.run_dir
    sess_path = run_dir / f"session_{a.session}.json"
    if sess_path.exists():
        print(f"{sess_path} exists", file=sys.stderr)
        return 2
    recorded = mc.load_json_finite((run_dir / "spec.json").read_text())
    if recorded != json.loads(mb.dumps(spec)):
        print("spec.json differs from the spec in code", file=sys.stderr)
        return 2
    out = {"argv": sys.argv, "source": mc.source_identity(ROOT), "packages": mc.package_versions(),
           "model_blobs": mc.weight_blobs(ROOT, sp.MODEL["id"], sp.MODEL["revision"])}
    scratch = SCRATCH / run_dir.name / a.session
    rc_ = 0
    try:
        model, _tok = km.load_model("cuda")
        out["gpu"] = torch.cuda.get_device_name(0)
        ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, spec["chunk_size"])
        cost_model = preds = None
        if spec["split"] == "evaluation":
            cost_model, preds, problems = load_frozen(a.cost_model, a.predictions, spec)
            if problems:
                raise RuntimeError(f"frozen files not bound to this spec: {problems}")
        scratch.mkdir(parents=True)  # refuses a leftover scratch directory
        mb.run_session(model, ns, spec, a.session, scratch, out=out, pin=True, cost_model=cost_model,
                       predictions=preds, log=lambda *m: print(*m, flush=True))
    except BaseException as e:  # noqa: BLE001  (recorded in the session file, then a nonzero exit)
        out["status"] = "failed"
        out["error"] = "".join(traceback.format_exception(e))[-4000:]
        rc_ = 4
    finally:
        try:  # clean up BEFORE the report, so a cleanup failure is recorded as a failed session
            if scratch.is_dir():
                shutil.rmtree(scratch)
        except OSError as e:
            out["status"] = "failed"
            out["error"] = (out.get("error") or "") + f"\nscratch cleanup failed: {e!r}"
            rc_ = 4
        with open(sess_path, "x") as f:
            f.write(mb.dumps(out))
    return rc_


def pre_launch_problems(spec, run_dir: Path, a) -> list:
    out = []
    src = mc.source_identity(ROOT)
    if src["git_status_porcelain"]:
        out.append(f"source tree not clean: {src['git_status_porcelain'][:5]}")
    if run_dir.exists():
        out.append(f"{run_dir} exists")
    if spec["split"] == "evaluation":
        if not (a.cost_model and a.predictions):
            out.append("evaluation needs --cost-model and --predictions")
        else:
            from kvlab import restore_cost as rc
            try:
                out += load_frozen(a.cost_model, a.predictions, spec)[2]
            except (OSError, ValueError, rc.ArtifactError) as e:   # missing, unreadable or malformed frozen files
                out.append(f"frozen files unusable: {type(e).__name__}: {e}")
    return out


def cmd_batch(a) -> int:
    from kvlab import m5_bench as mb
    spec = spec_for(a.split)
    run_dir = ROOT / a.run_dir
    problems = pre_launch_problems(spec, run_dir, a)
    if problems:
        print(json.dumps({"refused": problems}, indent=1), file=sys.stderr)
        return 2
    bindings = {}
    if spec["split"] == "evaluation":
        model, preds, _ = load_frozen(a.cost_model, a.predictions, spec)
        bindings = {"cost_model_digest": model.digest, "predictions_sha256": preds["sha256"]}
    plan = sp.batch_plan(spec, a.run_dir, sys.executable, cost_model_path=a.cost_model,
                         predictions_path=a.predictions, bindings=bindings)
    run_dir.mkdir(parents=True)
    with open(run_dir / "spec.json", "x") as f:
        f.write(mb.dumps(spec))
    with open(run_dir / "batch_plan.json", "x") as f:
        f.write(mb.dumps(plan))
    rec = mc.run_batch(ROOT, plan, mc.sha256_file(run_dir / "batch_plan.json"),
                       verification={"problems": [], "checks": "clean source, new run dir, frozen-file bindings"})
    print(json.dumps({k: rec.get(k) for k in ("execution_status", "stop", "elapsed_s")}, default=str))
    return mc.exit_code(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "session", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=("calibration", "evaluation"), required=True)
        if name != "plan":
            p.add_argument("--run-dir", required=True, help="relative to the repository root")
            p.add_argument("--cost-model")
            p.add_argument("--predictions")
        if name == "session":
            p.add_argument("--session", required=True)
    a = ap.parse_args()
    return {"plan": cmd_plan, "session": cmd_session, "batch": cmd_batch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
