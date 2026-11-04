"""M2.1: analyse diagnostic sessions (CPU only). Contrasts from docs/M2_1_DIAGNOSTIC_PLAN.md.

Usage: .venv/bin/python scripts/m2_1_analyze_diag.py --run-dir results/m2_1/<diag run>
Reads every session_*.json in the run directory (never modifies them) and writes
diag_analysis.json and diag_analysis.md (refuses to overwrite).
Slow sample: e2e > 1.3 x the minimum of its (config, path, mode, L, S) over all sessions of
that config (the M2 definition, applied per config).
"""

import argparse
import bisect
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kvlab import diag_harness as dh  # noqa: E402
from kvlab import perf_stats  # noqa: E402

RESTORE = ("host", "disk")


def load_monitor(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((ts, int(p[1].split()[0]), float(p[3].split()[0]), p[7]))
        except (ValueError, IndexError):
            continue
    return sorted(rows)


def nearest(rows, t):
    if not rows:
        return None
    ts = [r[0] for r in rows]
    i = bisect.bisect_left(ts, t)
    cand = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    j = min(cand, key=lambda j: abs(rows[j][0] - t))
    return rows[j] if abs(rows[j][0] - t) < 1.0 else None


def ms(ns):
    return ns / 1e6


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def share(xs):
    return float(np.mean(xs)) if len(xs) else None


def event_features(s):
    """Phase features of an events-mode sample (ms). submit = wall of the forward submission
    (restore mark -> prefill mark); gpu_lag = GPU event time - wall time at the prefill mark
    (how long after submission the forward's last kernel finished; clock alignment approximate)."""
    m = {x["phase"]: x for x in s["events"]["marks"]}
    pre = "restore" if "restore" in m else "ingest"
    f = {"submit_wall": ms(m["prefill"]["wall_ns"]) if pre == "restore" else ms(m["prefill"]["wall_ns"]),
         "submit_thread_cpu": ms(m["prefill"]["thread_cpu_ns"]),
         "forward_gpu_span": m["prefill"]["gpu_ms"] - m[pre]["gpu_ms"],
         "gpu_lag_at_prefill": m["prefill"]["gpu_ms"] - ms(m["prefill"]["wall_from_t0_ns"]),
         "final_wait_wall": ms(s["events"]["final_wait_wall_ns"])}
    if "restore" in m:
        f["restore_wall"] = ms(m["restore"]["wall_ns"])
        f["restore_thread_cpu"] = ms(m["restore"]["thread_cpu_ns"])
        f["store_wall"] = ms(m["store"]["wall_ns"])
    f["thread_cpu_total"] = ms(s["thread_cpu_ns"])
    f["e2e"] = ms(s["e2e_ns"])
    ru = s["rusage_delta"]
    f["runqueue_wait"] = ru["sched_runqueue_wait_ns"] / 1e6 if "sched_runqueue_wait_ns" in ru else None
    f["nivcsw"] = ru["ru_nivcsw"]
    f["nvcsw"] = ru["ru_nvcsw"]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out_json, out_md = run / "diag_analysis.json", run / "diag_analysis.md"
    for p in (out_json, out_md):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    sessions = {}
    for f in sorted(run.glob("session_*.json")):
        sessions[f.stem.removeprefix("session_")] = json.loads(f.read_text())
    res = {"sessions": {}, "checks": {}}
    samples = []
    for sid, d in sessions.items():
        mon = load_monitor(run / d["meta"]["gpu_monitor_csv"]) if d["meta"].get("gpu_monitor_csv") else []
        cfg = d["meta"]["config"] + ("+profiler" if d["meta"]["args"].get("profile") else "")
        res["sessions"][sid] = {k: d["meta"].get(k) for k in ("config", "seed", "started_at", "elapsed_min", "n_measured",
                                                              "n_excluded", "correctness_all", "spike_probe_audit_pass",
                                                              "cuda_schedule", "block_order")}
        res["sessions"][sid]["checks"] = d.get("checks")
        res["sessions"][sid]["monitor_rows"] = len(mon)
        res["sessions"][sid]["correctness"] = [{"L": c["L"], "S": c["S"], "strict_pass": c["strict_pass"],
                                               "nocache_bitwise_equal_full": c["nocache_vs_full"]["bitwise_equal"],
                                               "report_only_full_vs_restored": c["report_only_full_vs_restored"]}
                                              for c in d["correctness"]]
        for s in d["samples"]:
            if s.get("warmup"):
                continue
            probs = dh.sample_problems(s)
            if probs:
                continue  # counted in the session's n_excluded
            c = nearest(mon, s["t_wall"])
            samples.append({**s, "session": sid, "cfg": cfg, "clock_sm": c[1] if c else None,
                            "power_w": c[2] if c else None, "event_reasons": c[3] if c else None})
    # slow labels
    mins = defaultdict(lambda: float("inf"))
    for s in samples:
        k = (s["cfg"], s["path"], s["mode"], s["L"], s["S"], s.get("stage"))
        mins[k] = min(mins[k], s["e2e_ns"])
    for s in samples:
        s["slow"] = s["e2e_ns"] > 1.3 * mins[(s["cfg"], s["path"], s["mode"], s["L"], s["S"], s.get("stage"))]
    diag_all = [s for s in samples if s.get("stage") == "diag"]
    diag = [s for s in diag_all if s["path"] not in dh.PROBES]  # probes only in the Amendment 1 contrast
    cfgs = sorted({s["cfg"] for s in diag})

    # 1. slow share and latency by config x path x mode x case (H4: plain vs events vs synced)
    table = {}
    for cfg in cfgs:
        for (L, S) in sorted({(s["L"], s["S"]) for s in diag if s["cfg"] == cfg}):
            for path in dh.DIAG_PATHS:
                for mode in dh.MODES:
                    g = [s for s in diag if (s["cfg"], s["L"], s["S"], s["path"], s["mode"]) == (cfg, L, S, path, mode)]
                    if not g:
                        continue
                    e = np.array([ms(s["e2e_ns"]) for s in g])
                    table[f"{cfg}|{L}|{S}|{path}|{mode}"] = {
                        "n": len(g), "median_ms": float(np.median(e)), "mean_ms": float(e.mean()), "min_ms": float(e.min()),
                        "p25_ms": float(np.percentile(e, 25)), "p75_ms": float(np.percentile(e, 75)),
                        "slow_share": share([s["slow"] for s in g])}
    res["by_case_path_mode"] = table

    # 2. H7: slow share by session and block (restore paths, all modes)
    by_block = {}
    for s in diag:
        if s["path"] in RESTORE:
            by_block.setdefault(f"{s['session']}|block{s['block']}|L{s['L']}", []).append(s["slow"])
    res["restore_slow_share_by_session_block"] = {k: {"n": len(v), "slow_share": share(v)} for k, v in by_block.items()}

    # 3. temporal clustering: lag-1 of the slow indicator over chronological restore samples per session
    clus = {}
    for sid in sessions:
        seq = sorted((s for s in diag if s["session"] == sid and s["path"] in RESTORE), key=lambda s: s["t0_mono_ns"]
                     if "t0_mono_ns" in s else s["t_wall"])
        x = [float(s["slow"]) for s in seq]
        if len(x) > 3:
            prev = [b for a, b in zip(x, x[1:]) if a == 1.0]
            clus[sid] = {"n": len(x), "slow_share": share(x), "p_slow_given_prev_slow": share(prev) if prev else None,
                         "lag1_autocorr": perf_stats.lag1_autocorr(x)}
    res["restore_slow_clustering_by_session"] = clus

    # 4. H1: GPU clock join
    clk = {}
    for cfg in cfgs:
        g = [s for s in diag if s["cfg"] == cfg and s["path"] in RESTORE and s["clock_sm"] is not None]
        for name, cond in (("sm_ge_2200", lambda c: c >= 2200), ("sm_lt_2200", lambda c: c < 2200)):
            sub = [s for s in g if cond(s["clock_sm"])]
            clk[f"{cfg}|{name}"] = {"n": len(sub), "slow_share": share([s["slow"] for s in sub]),
                                    "clock_median_slow": med([s["clock_sm"] for s in sub if s["slow"]]),
                                    "clock_median_fast": med([s["clock_sm"] for s in sub if not s["slow"]])}
        clk[f"{cfg}|n_without_clock"] = len([s for s in diag if s["cfg"] == cfg and s["clock_sm"] is None])
        clk[f"{cfg}|clock_distribution_restore"] = {
            str(b): len([s for s in g if b <= s["clock_sm"] < b + 100]) for b in range(1200, 3200, 100)
            if any(b <= s["clock_sm"] < b + 100 for s in g)}
    res["clock_join"] = clk

    # 5. H2/H3: events-mode features, slow vs fast, per path and case
    feats = {}
    for cfg in cfgs:
        for (L, S) in sorted({(s["L"], s["S"]) for s in diag if s["cfg"] == cfg}):
            for path in dh.DIAG_PATHS:
                g = [s for s in diag if (s["cfg"], s["L"], s["S"], s["path"], s["mode"]) == (cfg, L, S, path, "events")]
                if not g:
                    continue
                rows = [(s["slow"], event_features(s)) for s in g]
                rec = {}
                for lab, want in (("fast", False), ("slow", True)):
                    fs = [f for sl, f in rows if sl == want]
                    rec[lab] = {"n": len(fs), **{k: med([f.get(k) for f in fs]) for k in rows[0][1]}}
                feats[f"{cfg}|{L}|{S}|{path}"] = rec
    res["events_features_slow_vs_fast"] = feats

    # 6. H5: recompute paths in repetitions where restore samples are slow vs fast
    h5 = {}
    for cfg in cfgs:
        reps = defaultdict(list)
        for s in diag:
            if s["cfg"] == cfg:
                reps[(s["session"], s["block"], s["rep"])].append(s)
        for (L, S) in sorted({(s["L"], s["S"]) for s in diag if s["cfg"] == cfg}):
            out = {}
            for state in ("restore_slow_rep", "restore_fast_rep"):
                sel = []
                for k, g in reps.items():
                    if g[0]["L"] != L:
                        continue
                    r = [s["slow"] for s in g if s["path"] in RESTORE]
                    if not r:
                        continue
                    is_slow = np.mean(r) >= 0.5
                    if is_slow == (state == "restore_slow_rep"):
                        sel += [s for s in g if s["path"] in ("full", "full_nocache")]
                ev = [event_features(s) for s in sel if s["mode"] == "events"]
                out[state] = {"n_recompute_samples": len(sel),
                              "recompute_e2e_slow_share": share([s["slow"] for s in sel]),
                              "recompute_plain_e2e_median": med([ms(s["e2e_ns"]) for s in sel if s["mode"] == "plain"]),
                              "recompute_events_submit_wall_median": med([f["submit_wall"] for f in ev]),
                              "recompute_events_gpu_lag_median": med([f["gpu_lag_at_prefill"] for f in ev]),
                              "recompute_events_final_wait_median": med([f["final_wait_wall"] for f in ev])}
            h5[f"{cfg}|{L}|{S}"] = out
    res["recompute_in_restore_slow_vs_fast_reps"] = h5

    # 6b. Amendment 1 (H8a vs H8b): probes in repetitions where restore samples are slow vs fast
    h8 = {}
    for cfg in cfgs:
        reps = defaultdict(list)
        for s in diag_all:
            if s["cfg"] == cfg:
                reps[(s["session"], s["block"], s["rep"])].append(s)
        for probe in dh.PROBES:
            out = {}
            for state in ("restore_slow_rep", "restore_fast_rep"):
                sel = []
                for g in reps.values():
                    r = [s["slow"] for s in g if s["path"] in RESTORE]
                    if r and (np.mean(r) >= 0.5) == (state == "restore_slow_rep"):
                        sel += [s for s in g if s["path"] == probe]
                out[state] = {"n": len(sel), "wall_median_ms": med([ms(s["e2e_ns"]) for s in sel]),
                              "thread_cpu_median_ms": med([ms(s["thread_cpu_ns"]) for s in sel]),
                              "runqueue_wait_median_ms": med([s["rusage_delta"]["sched_runqueue_wait_ns"] / 1e6 for s in sel])}
            if out["restore_fast_rep"]["n"] and out["restore_slow_rep"]["n"]:
                out["slow_over_fast_wall_ratio"] = out["restore_slow_rep"]["wall_median_ms"] / out["restore_fast_rep"]["wall_median_ms"]
                h8[f"{cfg}|{probe}"] = out
    res["probes_in_restore_slow_vs_fast_reps"] = h8

    # 7. H6: allocator activity
    al = defaultdict(lambda: {"n": 0, "with_device_alloc": 0, "with_retries": 0})
    for s in diag:
        k = f"{s['cfg']}|{'slow' if s['slow'] else 'fast'}"
        al[k]["n"] += 1
        al[k]["with_device_alloc"] += int(s["alloc_delta"]["num_device_alloc"] > 0)
        al[k]["with_retries"] += int(s["alloc_delta"]["num_alloc_retries"] > 0)
    res["allocator"] = dict(al)

    # 8. profiler sessions
    prof = {}
    for sid, d in sessions.items():
        if "profile" not in d:
            continue
        per = d["profile"]["per_sample"]
        prs = [s for s in samples if s["session"] == sid and s.get("stage") == "profiled"]
        ref = [s for s in samples if s["session"] == sid and s.get("stage") == "unprofiled_reference"]
        out = {}
        for path in ("host", "full_nocache"):
            g = [s for s in prs if s["path"] == path]
            rows = []
            for s in g:
                p = per[s["label"]]
                launch = p["api"].get("cudaLaunchKernel", [0, 0.0])
                sync = p["api"].get("cudaStreamSynchronize", [0, 0.0])
                mcp = p["api"].get("cudaMemcpyAsync", [0, 0.0])
                rows.append({"slow": s["slow"], "e2e": ms(s["e2e_ns"]), "kernel_ms": p["kernel_us"] / 1e3,
                             "n_kernels": p["n_kernels"], "gpu_memcpy_ms": p["memcpy_us"] / 1e3,
                             "launch_api_ms": launch[1] / 1e3, "n_launch": launch[0],
                             "launch_api_us_per_call": launch[1] / launch[0] if launch[0] else None,
                             "sync_api_ms": sync[1] / 1e3, "memcpy_api_ms": mcp[1] / 1e3})
            rec = {"n": len(rows), "unprofiled_reference_median_ms": med([ms(s["e2e_ns"]) for s in ref if s["path"] == path]),
                   "profiled_median_ms": med([r["e2e"] for r in rows])}
            for lab, want in (("fast", False), ("slow", True)):
                fs = [r for r in rows if r["slow"] == want]
                rec[lab] = {"n": len(fs), **{k: med([r[k] for r in fs]) for k in rows[0] if k != "slow"}} if rows else {}
            rec["per_sample"] = rows
            out[path] = rec
        prof[sid] = out
    res["profiler"] = prof
    res["n_valid_samples"] = len(samples)
    out_json.write_text(json.dumps(res, indent=1, default=str))
    out_md.write_text(md(res))
    print(md(res))


def fmt(x, nd=1):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def md(res):
    L = ["# M2.1 diagnostic analysis (generated by scripts/m2_1_analyze_diag.py)", "",
         "Slow = e2e > 1.3× the minimum of its (config, path, mode, L, S). Diagnostic only.", "",
         "## Sessions", "", "```", json.dumps({k: {kk: v.get(kk) for kk in ("config", "elapsed_min", "n_measured", "n_excluded",
                                                                          "correctness_all", "spike_probe_audit_pass",
                                                                          "cuda_schedule")}
                                                for k, v in res["sessions"].items()}, indent=1, default=str), "```", "",
         "## Latency and slow share by path and mode (ms)", "",
         "| config | L | S | path | mode | n | median | mean | p25–p75 | min | slow share |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["by_case_path_mode"].items():
        cfg, L_, S_, path, mode = k.split("|")
        L.append(f"| {cfg} | {L_} | {S_} | {path} | {mode} | {v['n']} | {v['median_ms']:.1f} | {v['mean_ms']:.1f} | "
                 f"{v['p25_ms']:.1f}–{v['p75_ms']:.1f} | {v['min_ms']:.1f} | {v['slow_share']:.0%} |")
    L += ["", "## Events mode: medians of fast vs slow samples (ms)", "",
          "| config | L | S | path | class | n | e2e | restore wall | restore thread CPU | submit wall | submit thread CPU | "
          "forward GPU span | GPU lag at prefill mark | store wall | final wait | runqueue wait | invol. csw |",
          "|" + "---|" * 17]
    for k, v in res["events_features_slow_vs_fast"].items():
        cfg, L_, S_, path = k.split("|")
        for lab in ("fast", "slow"):
            r = v[lab]
            if not r["n"]:
                continue
            L.append(f"| {cfg} | {L_} | {S_} | {path} | {lab} | {r['n']} | {fmt(r.get('e2e'))} | {fmt(r.get('restore_wall'))} | "
                     f"{fmt(r.get('restore_thread_cpu'))} | {fmt(r.get('submit_wall'))} | {fmt(r.get('submit_thread_cpu'))} | "
                     f"{fmt(r.get('forward_gpu_span'))} | {fmt(r.get('gpu_lag_at_prefill'))} | {fmt(r.get('store_wall'))} | "
                     f"{fmt(r.get('final_wait_wall'))} | {fmt(r.get('runqueue_wait'), 2)} | {fmt(r.get('nivcsw'))} |")
    for title, key in (("Recompute paths in repetitions where restore is slow vs fast (H5)", "recompute_in_restore_slow_vs_fast_reps"),
                       ("Probes in repetitions where restore is slow vs fast (Amendment 1: H8a vs H8b)", "probes_in_restore_slow_vs_fast_reps"),
                       ("GPU clock join (H1)", "clock_join"), ("Restore slow clustering per session", "restore_slow_clustering_by_session"),
                       ("Restore slow share by session and block (H7)", "restore_slow_share_by_session_block"),
                       ("Allocator (H6)", "allocator")):
        L += ["", f"## {title}", "", "```", json.dumps(res[key], indent=1, default=str), "```"]
    if res["profiler"]:
        L += ["", "## Profiler sessions (perturbed; not latency estimates)", ""]
        for sid, out in res["profiler"].items():
            for path, rec in out.items():
                L.append(f"- `{sid}` {path}: profiled median {fmt(rec['profiled_median_ms'])} ms vs unprofiled "
                         f"{fmt(rec['unprofiled_reference_median_ms'])} ms")
                for lab in ("fast", "slow"):
                    r = rec.get(lab) or {}
                    if r.get("n"):
                        L.append(f"  - {lab} (n={r['n']}): e2e {fmt(r['e2e'])}, GPU kernels {fmt(r['kernel_ms'], 2)} ms "
                                 f"({fmt(r['n_kernels'])}), GPU memcpy {fmt(r['gpu_memcpy_ms'], 2)}, launch API {fmt(r['launch_api_ms'])} ms "
                                 f"({fmt(r['launch_api_us_per_call'], 1)} µs/call), sync API {fmt(r['sync_api_ms'])} ms, "
                                 f"memcpy API {fmt(r['memcpy_api_ms'], 2)} ms")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
