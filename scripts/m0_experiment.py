"""M0: full recomputation vs CPU-snapshot KV restore, on a real GPU.

Usage:
  .venv/bin/python scripts/m0_experiment.py --smoke   # one case (N=257), no timings
  .venv/bin/python scripts/m0_experiment.py           # all cases + diagnostic timings

Writes <out-dir>/m0_results.json (or m0_smoke.json); default out-dir is
results/m0_reruns/<timestamp>/. The original M0 record in results/m0/ is never overwritten. Exit status 0 only if
every case passes; 1 if any case fails.

Timing boundary (diagnostic only, not TTFT): before each timed region we call
torch.cuda.synchronize(), then record a CUDA event and a host perf_counter
timestamp; after it we record an event and synchronize again. `wall_ms` is
host time between the two synchronizations (includes Python and CPU work, e.g.
the pageable-memory copy staging); `gpu_ms` is CUDA-event elapsed time on the
current stream.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402  (sets HF_HOME before transformers import)
import torch  # noqa: E402
import transformers  # noqa: E402
from kvlab.boundaries import max_restore_len  # noqa: E402
from kvlab.experiment import case_passed, run_case  # noqa: E402
from kvlab.fixtures import BOUNDARY_TEXT, NATURAL_PROMPTS, chat_ids, exact_length_ids  # noqa: E402
from kvlab.kv_snapshot import export_prefix_to_cpu, restore_to_device  # noqa: E402
from kvlab.tolerances import ATOL, RTOL  # noqa: E402

CHUNK = 256
BOUNDARY_LENGTHS = [255, 256, 257, 511, 512, 513]
GREEDY_STEPS = 8
WARMUP, REPS = 2, 5


def build_cases(tok, smoke):
    if smoke:
        return [("boundary_257", exact_length_ids(tok, BOUNDARY_TEXT, 257))]
    cases = [(f"boundary_{n}", exact_length_ids(tok, BOUNDARY_TEXT, n)) for n in BOUNDARY_LENGTHS]
    cases += [(name, chat_ids(tok, msgs)) for name, msgs in NATURAL_PROMPTS.items()]
    return cases


def timed(fn, setup=None):
    wall, gpu = [], []
    for i in range(WARMUP + REPS):
        state = setup() if setup else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        result = fn(state)
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del result, state
        if i >= WARMUP:
            wall.append((t1 - t0) * 1e3)
            gpu.append(start.elapsed_time(end))
    return {"reps": REPS, "warmup": WARMUP,
            "wall_ms_median": statistics.median(wall), "wall_ms_min": min(wall), "wall_ms_max": max(wall),
            "gpu_ms_median": statistics.median(gpu), "gpu_ms_min": min(gpu), "gpu_ms_max": max(gpu)}


def time_case(model, token_ids):
    ids = torch.tensor([token_ids], device="cuda")
    n = ids.shape[1]
    lmax = max_restore_len(n, CHUNK)
    _, prefix_cache = km.prefill(model, ids[:, :lmax], None, 0)
    snapshot = export_prefix_to_cpu(prefix_cache, lmax, CHUNK)
    out = {
        "N": n, "L": lmax, "suffix_len": n - lmax, "snapshot_bytes": snapshot.nbytes(),
        "full_prefill_N": timed(lambda _: km.prefill(model, ids, None, 0)),
        "export_gpu_to_cpu_L": timed(lambda _: export_prefix_to_cpu(prefix_cache, lmax, CHUNK)),
        "restore_cpu_to_gpu_L": timed(lambda _: restore_to_device(snapshot, lmax, model.config, "cuda")),
        "suffix_prefill_after_restore": timed(
            lambda cache: km.prefill(model, ids[:, lmax:], cache, lmax),
            setup=lambda: restore_to_device(snapshot, lmax, model.config, "cuda")),
    }
    del prefix_cache, snapshot
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/m0_reruns/<timestamp>/ (results/m0/ holds the original M0 record)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("BLOCKED: torch.cuda.is_available() is False")
        sys.exit(2)

    torch.manual_seed(0)
    model, tok = km.load_model()
    cases = build_cases(tok, args.smoke)
    results = []
    for name, token_ids in cases:
        rec = run_case(model, name, token_ids, CHUNK, ATOL, RTOL, GREEDY_STEPS)
        rec["status"] = "PASS" if case_passed(rec) else "FAIL"
        results.append(rec)
        line = f"{rec['status']} {name:14s} N={rec['prompt_len']:4d} eligible_L={rec['eligible_restore_lens']}"
        for r in rec["restores"]:
            fl = r["first_logits"]
            neg = r.get("negative_control_zero_kv")
            line += (f"\n     L={r['restore_len']:4d} suffix={r['suffix_len']:4d} "
                     f"bytes_eq(export,restore)={rec['export_check']['all_bytes_equal']},"
                     f"{r['restore_check']['all_bytes_equal']} max_abs={fl['max_abs_err']:.4f} "
                     f"rel_l2={fl['rel_l2_err']:.2e} argmax_eq={fl['argmax_equal']} "
                     f"greedy={r['greedy']['matching_positions']}/{r['greedy']['steps']} "
                     f"bitwise_vs_gpu_chunked={r.get('bitwise_equal_to_gpu_resident_chunked')}"
                     + (f" neg_ctrl_max_abs={neg['max_abs_err']:.2f}" if neg else ""))
        print(line, flush=True)

    timings = []
    if not args.smoke:
        for name, token_ids in cases:
            if max_restore_len(len(token_ids), CHUNK) > 0:
                t = time_case(model, token_ids)
                t["name"] = name
                timings.append(t)
                print(f"TIMING {name:14s} N={t['N']} L={t['L']} "
                      f"full={t['full_prefill_N']['gpu_ms_median']:.2f}ms "
                      f"restore={t['restore_cpu_to_gpu_L']['wall_ms_median']:.2f}ms(wall) "
                      f"suffix={t['suffix_prefill_after_restore']['gpu_ms_median']:.2f}ms", flush=True)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    out = {
        "milestone": "M0", "mode": "smoke" if args.smoke else "full", "overall": overall,
        "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "transformers": transformers.__version__, "python": sys.version.split()[0]},
        "gpu": torch.cuda.get_device_name(0),
        "model": km.model_info(model),
        "chunk_size": CHUNK, "greedy_steps": GREEDY_STEPS,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "cases": results,
        "diagnostic_timings_not_benchmarks": timings,
    }
    out_dir = Path(args.out_dir) if args.out_dir else (
        km.PROJECT_ROOT / "results" / "m0_reruns" / time.strftime("%Y%m%dT%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("m0_smoke.json" if args.smoke else "m0_results.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"OVERALL {overall}; wrote {path}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
