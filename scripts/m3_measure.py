"""M3 session: baseline vs batched restore backend, interleaved in one fresh process (docs/M3_PROTOCOL.md).

Usage:
  .venv/bin/python scripts/m3_measure.py --out-dir results/m3/<run> --session s1 --cases 256:32,1024:1 [--blocks 2]
      [--warmup 2] [--reps 12] [--seed N] [--staging-chunks 4]
  .venv/bin/python scripts/m3_measure.py --out-dir results/m3/<run> --session prof --profile --cases 256:1,4096:1
Writes <out-dir>/session_<id>.json (never overwritten) and the GPU monitor CSV.
Each repetition runs, in a random order: full (prepared-state recompute, context), host and disk
with each backend (plain = the M2 timed region), and host and disk with each backend in the M2
synchronized-phase mode (restore-phase diagnostic timing). Every sample is kept.
--profile: a separate torch.profiler run that counts CUDA API calls and device ops, per restore
call and per whole request, for each backend and tier. Its timings are not used.
"""

import argparse
import gc
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--staging-chunks", type=int, default=4)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-reps", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"session_{args.session}.json"
    mcsv = out_dir / f"session_{args.session}.gpu_monitor.csv"
    if out.exists() or mcsv.exists():
        sys.exit(f"refusing to overwrite {out} / {mcsv}")

    import kvlab.model as km  # noqa: E402  (sets HF_HOME)
    import torch  # noqa: E402
    from kvlab import diag_harness as dh  # noqa: E402
    from kvlab import m3_bench as mb  # noqa: E402
    from kvlab import perf  # noqa: E402
    from kvlab.perf_harness import environment, gpu_state, start_gpu_monitor, stop_gpu_monitor  # noqa: E402

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases.split(",")]
    for L, S in cases:
        perf.check_case(L, S)
    meta = {"session": args.session, "args": vars(args), "seed": seed,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "started_unix": time.time()}
    disk_root = km.PROJECT_ROOT / ".kvcache" / "m3" / out_dir.name / args.session
    if disk_root.exists():
        shutil.rmtree(disk_root)  # our own scratch area under .kvcache/ only
    source = mb.source_identity(km.PROJECT_ROOT)  # before anything runs
    model, tok = km.load_model()
    bench = mb.M3Bench(model, tok, disk_root, max_L=max(L for L, _ in cases), max_S=perf.C,
                       staging_chunks=args.staging_chunks)
    monitor = start_gpu_monitor(mcsv)
    meta["gpu_monitor_csv"] = mcsv.name
    report = {"meta": meta, "source": source, "environment": environment(km.PROJECT_ROOT, disk_root),
              "packages": dh.package_versions(), "weights": dh.weight_blobs(km.PROJECT_ROOT, km.MODEL_ID, km.MODEL_REVISION),
              "process": dh.process_record(), "model": km.model_info(model),
              "restore_config": bench.tiered_A.restore_config(), "gpu_state": [{"when": "start", **gpu_state()}],
              "samples": [], "correctness": [], "ab_checks": [], "restore_memory": [], "failures": [], "blocks": []}
    t_start = time.time()
    if args.profile:
        run_profile(bench, report, cases, args.profile_reps)
    else:
        order = cases[:]
        rng.shuffle(order)
        block_cases = [c for _ in range(args.blocks) for c in order]
        report["meta"]["block_order"] = block_cases
        for L, S in sorted({min(cases), max(cases)}):  # process warmup: every combo twice
            bench.prepare(L)
            for path, b, m in mb.COMBOS:
                for _ in range(2):
                    report["samples"].append({**bench.measure_m3(path, S, b, m), "warmup": True,
                                              "stage": "process_warmup", "session": args.session})
        for bi, (L, S) in enumerate(block_cases):
            try:
                bench.prepare(L)
                report["gpu_state"].append({"when": f"before block {bi} L={L} S={S}", **gpu_state()})
                if bi < len(order):  # once per case per session, untimed
                    report["correctness"].append(bench.correctness_both(S))
                    report["ab_checks"].append(bench.ab_check(S))
                    report["restore_memory"].append({"L": L, "S": S, **bench.restore_memory()})
                tb = time.time()
                for rep in range(args.warmup + args.reps):
                    gc.collect()
                    combos = list(mb.COMBOS)
                    rng.shuffle(combos)
                    for pos, (path, b, m) in enumerate(combos):
                        s = bench.measure_m3(path, S, b, m)
                        report["samples"].append({**s, "session": args.session, "block": bi, "rep": rep,
                                                  "order_pos": pos, "warmup": rep < args.warmup, "stage": "m3"})
                report["blocks"].append({"block": bi, "L": L, "S": S, "t_start": tb, "t_end": time.time()})
            except torch.cuda.OutOfMemoryError as e:
                report["failures"].append({"block": bi, "L": L, "reason": f"CUDA OOM: {e}"[:500]})
                torch.cuda.empty_cache()
            print(f"block {bi} L={L} S={S} done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    report["budgets"] = bench.budgets()
    report["gpu_state"].append({"when": "end", **gpu_state()})
    report["process_end"] = dh.process_record()
    bench.close()
    stop_gpu_monitor(monitor)
    shutil.rmtree(disk_root, ignore_errors=True)
    meas = [s for s in report["samples"] if not s.get("warmup")]
    nl = model.config.num_hidden_layers
    excluded = [{"i": i, "path": s["path"], "backend": s["backend"], "mode": s["mode"], "why": mb.sample_problems(s, nl)}
                for i, s in enumerate(meas) if mb.sample_problems(s, nl)]
    strict = all(c["strict_pass_all"] for c in report["correctness"]) and all(a["pass"] for a in report["ab_checks"])
    report["meta"].update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), elapsed_min=(time.time() - t_start) / 60,
                          n_samples=len(report["samples"]), n_measured=len(meas), n_excluded=len(excluded),
                          correctness_all=strict)
    report["excluded"] = excluded
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report["meta"][k] for k in ("n_samples", "n_measured", "n_excluded", "elapsed_min",
                                                      "correctness_all")}))
    print(f"failures {report['failures']}; wrote {out}")
    sys.exit(0 if strict and not report["failures"] and not excluded else 1)


def run_profile(bench, report, cases, reps):
    """Count CUDA runtime API calls and device ops with torch.profiler: one profiler context per
    (case, tier, backend, unit), `reps` calls inside it, counts divided by `reps`. Units:
    restore = TieredKVCache.restore() alone; request = the M2 timed region (Bench.measure)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    rows = []
    for L, S in cases:
        bench.prepare(L)
        n = L // 256
        for path in ("host", "disk"):
            t = bench.tiered_B if path == "disk" else bench.tiered_A
            for backend in ("baseline", "batched"):
                for unit in ("restore", "request"):
                    bench.set_backend(backend)

                    def once():
                        if unit == "restore":
                            if path == "disk":
                                bench.page_cache_before("disk")
                            cache, Lr, src, _ = t.restore(bench.base[:L + S], bench.keys[:n], bench.device,
                                                          bench.model.config)
                            assert Lr == L and src == [path] * n
                            del cache
                        else:
                            s = bench.measure(path, S)
                            assert s["restore_len"] == L
                        torch.cuda.synchronize()
                    for _ in range(2):
                        once()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(reps):
                            once()
                    api, dev = Counter(), Counter()
                    for e in prof.events():
                        if "CUDA" in str(e.device_type):
                            nm = e.name
                            kind = ("memcpy_HtoD" if "HtoD" in nm else "memcpy_DtoD" if "DtoD" in nm else
                                    "memcpy_DtoH" if "DtoH" in nm else "memcpy_other" if "Memcpy" in nm else
                                    "memset" if "Memset" in nm else "kernel")
                            dev[kind] += 1
                        elif e.name.startswith("cuda") or e.name.startswith("cu"):
                            api[e.name] += 1
                    rows.append({"L": L, "S": S, "n_chunks": n, "path": path, "backend": backend, "unit": unit,
                                 "reps": reps, "api_per_call": {k: v / reps for k, v in sorted(api.items())},
                                 "device_ops_per_call": {k: v / reps for k, v in sorted(dev.items())},
                                 "note": "cudaDeviceSynchronize includes the profiler loop's own torch.cuda.synchronize() "
                                         "(1 per call); request units include Bench.measure's synchronize calls"})
                    print(f"profile L={L} {path} {backend} {unit}: {rows[-1]['api_per_call']}", flush=True)
    bench.set_backend("baseline")
    report["profile"] = rows


if __name__ == "__main__":
    main()
