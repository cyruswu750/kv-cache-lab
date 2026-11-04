"""M2.1 diagnostic session: one fresh process, interleaved paths x modes (docs/M2_1_DIAGNOSTIC_PLAN.md).

Usage:
  .venv/bin/python scripts/m2_1_diagnose.py --out-dir results/m2_1/<run> --session s1 [--config baseline]
      [--no-monitor] [--cuda-schedule spin] [--cases 512:1,1024:64,4096:64] [--blocks 2] [--warmup 3] [--reps 12]
  .venv/bin/python scripts/m2_1_diagnose.py --out-dir results/m2_1/<run> --session prof --profile
Writes <out-dir>/session_<id>.json (never overwritten) and, unless --no-monitor, the GPU monitor CSV.
Diagnostic only: these samples are never used by the M2 model or scores.
"""

import argparse
import ctypes
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CUDA_SCHEDULE = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}


def set_cuda_schedule(name: str) -> dict:
    """Process-local diagnostic: set the device schedule flag before torch creates a context."""
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    lib = Path(list(spec.submodule_search_locations)[0]) / "cu13" / "lib" / "libcudart.so.13"
    rt = ctypes.CDLL(str(lib))
    rc = rt.cudaSetDeviceFlags(ctypes.c_uint(CUDA_SCHEDULE[name]))
    return {"requested": name, "cudaSetDeviceFlags_rc": rc, "lib": str(lib), "_rt": rt}


def get_cuda_flags(rt) -> int:
    f = ctypes.c_uint(0)
    rc = rt.cudaGetDeviceFlags(ctypes.byref(f))
    return f.value if rc == 0 else -rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--cases", default="512:1,1024:64,4096:64")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--cuda-schedule", choices=sorted(CUDA_SCHEDULE), default=None)
    ap.add_argument("--profile", action="store_true", help="profiler session: host vs full_nocache at L=512 S=1")
    ap.add_argument("--profile-reps", type=int, default=40)
    ap.add_argument("--probes", action="store_true", help="Amendment 1: add CPU-only and CUDA-call probes to each rep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"session_{args.session}.json"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    sched = set_cuda_schedule(args.cuda_schedule) if args.cuda_schedule else None

    import kvlab.model as km  # noqa: E402  (sets HF_HOME)
    import torch  # noqa: E402
    from kvlab import diag_harness as dh  # noqa: E402
    from kvlab import perf  # noqa: E402
    from kvlab.perf_harness import environment, gpu_state, start_gpu_monitor, stop_gpu_monitor  # noqa: E402

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases.split(",")]
    for L, S in cases:
        perf.check_case(L, S)
    meta = {"session": args.session, "config": args.config, "args": vars(args), "seed": seed,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "started_unix": time.time()}
    disk_root = km.PROJECT_ROOT / ".kvcache" / "m2_1" / out_dir.name / args.session
    if disk_root.exists():
        shutil.rmtree(disk_root)  # our own scratch area under .kvcache/ only
    model, tok = km.load_model()
    if sched:
        meta["cuda_schedule"] = {"requested": sched["requested"], "set_rc": sched["cudaSetDeviceFlags_rc"],
                                 "flags_after_torch_init": get_cuda_flags(sched["_rt"]), "lib": sched["lib"]}
    bench = dh.DiagBench(model, tok, disk_root, max_L=max(L for L, _ in cases), max_S=perf.C)
    monitor = None
    if not args.no_monitor:
        mcsv = out_dir / f"session_{args.session}.gpu_monitor.csv"
        if mcsv.exists():
            sys.exit(f"refusing to overwrite {mcsv}")
        monitor = start_gpu_monitor(mcsv)
        meta["gpu_monitor_csv"] = mcsv.name
    report = {"meta": meta, "environment": environment(km.PROJECT_ROOT, disk_root),
              "packages": dh.package_versions(), "weights": dh.weight_blobs(km.PROJECT_ROOT, km.MODEL_ID, km.MODEL_REVISION),
              "process": dh.process_record(), "model": km.model_info(model), "gpu_state": [{"when": "start", **gpu_state()}],
              "checks": {}, "samples": [], "correctness": [], "failures": [], "blocks": []}
    t_start = time.time()

    if args.profile:
        run_profile(bench, report, args, rng, out_dir)
    else:
        bench.prepare(min(L for L, _ in cases))
        report["checks"]["spike_probe_audit"] = bench.spike_probe_audit()
        report["checks"]["sync_debug"] = bench.sync_debug()
        order = cases[:]
        rng.shuffle(order)
        block_cases = order * args.blocks
        report["meta"]["block_order"] = block_cases
        if args.probes:
            for p in dh.PROBES:
                for _ in range(3):
                    report["samples"].append({**bench.measure_probe(p), "warmup": True, "stage": "process_warmup",
                                              "session": args.session})
        # Process warmup: every (path, mode) twice at the smallest and largest case.
        for L, S in sorted({min(cases), max(cases)}):
            bench.prepare(L)
            for path in dh.DIAG_PATHS:
                for mode in dh.MODES:
                    for _ in range(2):
                        report["samples"].append({**bench.measure_diag(path, S, mode), "warmup": True,
                                                  "stage": "process_warmup", "session": args.session})
        for bi, (L, S) in enumerate(block_cases):
            try:
                bench.prepare(L)
                report["gpu_state"].append({"when": f"before block {bi} L={L}", **gpu_state()})
                if bi < len(order):
                    c = bench.correctness(S)
                    c["nocache_vs_full"] = bench.nocache_matches_full(S)
                    report["correctness"].append(c)
                tb = time.time()
                for rep in range(args.warmup + args.reps):
                    gc.collect()
                    pm = [(p, m) for p in dh.DIAG_PATHS for m in dh.MODES]
                    if args.probes:
                        pm += [(p, "plain") for p in dh.PROBES]
                    rng.shuffle(pm)
                    for pos, (path, mode) in enumerate(pm):
                        s = bench.measure_probe(path) if path in dh.PROBES else bench.measure_diag(path, S, mode)
                        report["samples"].append({**s, "session": args.session, "config": args.config, "block": bi,
                                                  "rep": rep, "order_pos": pos, "warmup": rep < args.warmup,
                                                  "stage": "diag"})
                report["blocks"].append({"block": bi, "L": L, "S": S, "t_start": tb, "t_end": time.time()})
            except torch.cuda.OutOfMemoryError as e:
                report["failures"].append({"block": bi, "L": L, "reason": f"CUDA OOM: {e}"[:500]})
                torch.cuda.empty_cache()
            print(f"block {bi} L={L} S={S} done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    report["budgets"] = bench.budgets()
    report["gpu_state"].append({"when": "end", **gpu_state()})
    report["process_end"] = dh.process_record()
    bench.close()
    if monitor:
        stop_gpu_monitor(monitor)
    shutil.rmtree(disk_root, ignore_errors=True)
    meas = [s for s in report["samples"] if not s.get("warmup")]
    excluded = [{"i": i, "path": s["path"], "mode": s["mode"], "why": dh.sample_problems(s)}
                for i, s in enumerate(meas) if dh.sample_problems(s)]
    strict = all(c["strict_pass"] and c["nocache_vs_full"]["bitwise_equal"] for c in report["correctness"])
    report["meta"].update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), elapsed_min=(time.time() - t_start) / 60,
                          n_samples=len(report["samples"]), n_measured=len(meas), n_excluded=len(excluded),
                          correctness_all=strict,
                          spike_probe_audit_pass=report["checks"].get("spike_probe_audit", {}).get("pass"))
    report["excluded"] = excluded
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report["meta"][k] for k in ("n_samples", "n_measured", "n_excluded", "elapsed_min",
                                                      "correctness_all", "spike_probe_audit_pass")}))
    print(f"failures {report['failures']}; wrote {out}")
    ok = (strict or args.profile) and not report["failures"] and not excluded
    sys.exit(0 if ok else 1)


def run_profile(bench, report, args, rng, out_dir):
    """Short torch.profiler capture; per-sample CPU-side CUDA API time and GPU kernel time."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    bench.prepare(512)
    S = 1
    for _ in range(3):  # unprofiled warmup
        for path in ("host", "full_nocache"):
            bench.measure_diag(path, S, "plain")
    plain_ref = []
    labels = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for rep in range(args.profile_reps):
            gc.collect()
            order = ["host", "full_nocache"]
            rng.shuffle(order)
            for path in order:
                lab = f"m21_sample_{len(labels)}"
                with record_function(lab):
                    s = bench.measure_diag(path, S, "plain")
                labels.append(lab)
                report["samples"].append({**s, "session": args.session, "rep": rep, "label": lab, "warmup": False,
                                          "stage": "profiled"})
    for _ in range(args.profile_reps // 2):  # unprofiled reference after, same process
        gc.collect()
        for path in ("host", "full_nocache"):
            s = bench.measure_diag(path, S, "plain")
            plain_ref.append({**s, "session": args.session, "warmup": False, "stage": "unprofiled_reference"})
    report["samples"] += plain_ref
    evs = prof.events()
    ranges = {e.name: (e.time_range.start, e.time_range.end) for e in evs if e.name.startswith("m21_sample_")}
    per = {lab: {"api": {}, "kernel_us": 0.0, "n_kernels": 0, "memcpy_us": 0.0} for lab in ranges}
    starts = sorted((v[0], v[1], k) for k, v in ranges.items())
    import bisect
    keys = [s[0] for s in starts]

    def owner(t):
        i = bisect.bisect_right(keys, t) - 1
        if i >= 0 and starts[i][0] <= t <= starts[i][1] + 2000:  # 2 ms tail for GPU work finishing after the CPU range
            return starts[i][2]
        return None
    for e in evs:
        if e.name.startswith("m21_sample_"):
            continue
        dev = str(getattr(e, "device_type", ""))
        lab = owner(e.time_range.start)
        if lab is None:
            continue
        dur = e.time_range.end - e.time_range.start
        if "CUDA" in dev:
            if "Memcpy" in e.name or "Memset" in e.name:
                per[lab]["memcpy_us"] += dur
            else:
                per[lab]["kernel_us"] += dur
                per[lab]["n_kernels"] += 1
        elif e.name.startswith("cuda") or e.name.startswith("cu"):
            a = per[lab]["api"].setdefault(e.name, [0, 0.0])
            a[0] += 1
            a[1] += dur
    report["profile"] = {"per_sample": per, "ranges_us": {k: list(v) for k, v in ranges.items()},
                         "key_averages_top": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)}


if __name__ == "__main__":
    main()
