"""M2 measurement: tokenized-input first-token latency for full / host / disk paths.

Usage:
  .venv/bin/python scripts/m2_measure.py --split smoke       --run-dir results/m2/<run>
  .venv/bin/python scripts/m2_measure.py --split calibration --run-dir results/m2/<run>
  .venv/bin/python scripts/m2_measure.py --split evaluation  --run-dir results/m2/<run>   (needs predictions.json)
Writes <run-dir>/<split>_samples.json (never overwritten): every raw sample (warmups
included and flagged), correctness checks, GPU state per L group, environment, budgets.
Protocol: docs/M2_PROTOCOL.md. One GPU process at a time.
"""

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402  (sets HF_HOME before transformers import)
import torch  # noqa: E402
from kvlab import perf  # noqa: E402
from kvlab.perf_harness import Bench, environment, gpu_state, start_gpu_monitor, stop_gpu_monitor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(perf.SPLITS), required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--instr-warmup", type=int, default=2)
    ap.add_argument("--instr-reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--budget-min", type=float, default=30.0, help="stop starting new L groups after this")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{args.split}_samples.json"
    if out_path.exists():
        sys.exit(f"refusing to overwrite {out_path}")
    meta = {"split": args.split, "args": vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "started_unix": time.time()}
    if args.split == "evaluation":
        pred_path = run_dir / "predictions.json"
        if not pred_path.exists():
            sys.exit("evaluation needs a frozen predictions.json in the run directory (scripts/m2_fit_predict.py)")
        preds = json.loads(pred_path.read_text())
        if preds["sha256"] != perf.predictions_digest(preds):
            sys.exit("predictions.json digest mismatch")
        meta["predictions_sha256"] = preds["sha256"]
        meta["predictions_file_sha256"] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        meta["predictions_mtime_unix"] = pred_path.stat().st_mtime
    cases = list(perf.SPLITS[args.split])
    for L, S in cases:
        perf.check_case(L, S)
    rng = random.Random(args.seed)
    disk_root = km.PROJECT_ROOT / ".kvcache" / "m2" / run_dir.name / args.split
    if disk_root.exists():
        shutil.rmtree(disk_root)  # our own scratch area under .kvcache/ only
    model, tok = km.load_model()
    max_L, max_S = max(L for L, _ in cases), max(S for _, S in cases)
    bench = Bench(model, tok, disk_root, max_L=max(max_L, perf.C), max_S=perf.C)
    monitor_csv = run_dir / f"{args.split}_gpu_monitor.csv"
    if monitor_csv.exists():
        sys.exit(f"refusing to overwrite {monitor_csv}")
    monitor = start_gpu_monitor(monitor_csv)
    meta["gpu_monitor_csv"] = monitor_csv.name
    report = {"meta": meta, "environment": environment(km.PROJECT_ROOT, disk_root), "model": km.model_info(model),
              "namespace": bench.ns.fields(), "gpu_state": [{"when": "start", **gpu_state()}],
              "samples": [], "correctness": [], "failures": [], "not_run": []}
    t_start = time.time()

    # Process-level warmup (kernel selection, allocator growth): smallest and largest L.
    for L in sorted({min(L for L, _ in cases), max_L}):
        bench.prepare(L)
        for S in sorted({S for _, S in cases}):
            for path in perf.paths_for(S):
                for _ in range(2):
                    report["samples"].append({**bench.measure(path, S), "warmup": True, "stage": "process_warmup"})

    groups = sorted({L for L, _ in cases})
    rng.shuffle(groups)
    report["meta"]["L_group_order"] = groups
    for L in groups:
        if (time.time() - t_start) / 60 > args.budget_min:
            report["not_run"] += [{"L": L, "S": S, "reason": "time budget reached"} for (l2, S) in cases if l2 == L]
            continue
        S_values = sorted(S for l2, S in cases if l2 == L)
        try:
            bench.prepare(L)
            report["gpu_state"].append({"when": f"before L={L}", **gpu_state()})
            for S in S_values:
                report["correctness"].append(bench.correctness(S))
            for instrumented, n_warm, n_reps in ((False, args.warmup, args.reps),
                                                 (True, args.instr_warmup, args.instr_reps)):
                for rep in range(n_warm + n_reps):
                    gc.collect()  # once per repetition, outside every timed region
                    order = [(S, path) for S in S_values for path in perf.paths_for(S)]
                    rng.shuffle(order)
                    for pos, (S, path) in enumerate(order):
                        s = bench.measure(path, S, instrumented=instrumented)
                        report["samples"].append({**s, "rep": rep, "order_pos": pos, "warmup": rep < n_warm,
                                                  "stage": "instrumented" if instrumented else "e2e"})
        except torch.cuda.OutOfMemoryError as e:
            report["failures"].append({"L": L, "reason": f"CUDA OOM: {e}"[:500]})
            torch.cuda.empty_cache()
            continue
        print(f"L={L}: done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    if args.split in ("smoke", "calibration"):
        bench.prepare(min(L for L, _ in cases))
        report["micro"] = bench.micro()
    report["budgets"] = bench.budgets()
    report["gpu_state"].append({"when": "end", **gpu_state()})
    bench.close()
    stop_gpu_monitor(monitor)
    shutil.rmtree(disk_root)
    valid, excluded = perf.partition(report["samples"])
    report["meta"].update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), elapsed_min=(time.time() - t_start) / 60,
                          n_samples=len(report["samples"]), n_valid=len(valid), n_excluded=len(excluded),
                          correctness_strict_all=all(c["strict_pass"] for c in report["correctness"]))
    report["excluded"] = [{k: e.get(k) for k in ("path", "L", "S", "rep", "stage", "excluded_because")} for e in excluded]
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report["meta"][k] for k in ("n_samples", "n_valid", "n_excluded", "elapsed_min",
                                                     "correctness_strict_all")}))
    print(f"failures {report['failures']} not_run {report['not_run']}")
    print(f"wrote {out_path}")
    sys.exit(0 if report["meta"]["correctness_strict_all"] and not report["failures"] else 1)


if __name__ == "__main__":
    main()
