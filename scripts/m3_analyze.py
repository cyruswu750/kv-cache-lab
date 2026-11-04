"""M3: analyse the evaluation sessions (CPU only; docs/M3_PROTOCOL.md §6-§8).

Usage: .venv/bin/python scripts/m3_analyze.py --run-dir results/m3/<eval run> [--out-dir <new dir>]
           [--spec m3_eval_20261103T173632]
Reads session_*.json without modifying them.

- **Validation first** (kvlab/m3_validate.py, BUGLOG VAL-006). The run is checked against an
  explicit experiment specification: planned sessions, cases, blocks, repetitions, every
  path/backend/mode sample, and the correctness, A/B and restore-memory records. Correctness
  acceptance is recomputed from the detailed checks. m3_validation.json is always written.
- **Analysis.** Only if the evidence is valid, the analysis writes m3_analysis.json,
  m3_analysis.md and samples_chronological.csv.
- **Output.** Everything goes to --out-dir (default: the run directory). The script refuses to
  overwrite.

Exit codes:
- 0: valid evidence, analysis written. supports_adoption may still be false, for example
  when a required case is inconclusive.
- 1: refused (an output exists, or the run directory is missing).
- 2: usage error.
- 3: invalid or incomplete evidence. No analysis is written, and supports_adoption is false.
- 4: complete evidence, but a required correctness or A/B check failed. No analysis is
  written, and supports_adoption is false.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvlab import m3_stats as ms  # noqa: E402
from kvlab import m3_validate as mv  # noqa: E402

TIERS = ("host", "disk")
BACKENDS = ("baseline", "batched")
C = 256
EXIT = {"valid": 0, "invalid": 3, "correctness_failed": 4}


def sample_problems(s, nl=28):
    from kvlab import m3_bench as mb  # imported lazily (pulls in torch); CPU only here
    return mb.sample_problems(s, nl)


def value(s):
    return s["e2e_ns"] / 1e6 if s["mode"] == "plain" else s["phases_ns"]["restore"] / 1e6


def summarize(xs):
    xs = sorted(xs)
    if not xs:
        return None
    q = np.quantile(xs, [0.25, 0.75])
    return {"n": len(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "p25": float(q[0]), "p75": float(q[1]), "min": xs[0], "max": xs[-1]}


def decide(cases_out: dict, spec: dict, validation: dict) -> dict:
    """Protocol §8, driven by the spec's case list, never by the cases that happen to be in the
    analysis. A spec case missing from `cases_out` blocks adoption. Gate (1) is the recomputed
    acceptance from the validation, never a stored summary flag."""
    gates = validation.get("status") == "valid" and validation.get("correctness_and_ab_acceptance_recomputed") is True
    regress, need, missing = [], [], []
    for L, S in spec["cases"]:
        key = f"{L}:{S}"
        for tier in spec["tiers"]:
            for st in ("median_d", "mean_d"):
                try:
                    verdict = cases_out[key]["tiers"][tier]["e2e"][st]["verdict"]
                except (KeyError, TypeError):
                    missing.append(f"{key} {tier} {st}")
                    continue
                if verdict == "batched_slower":
                    regress.append(f"{key} {tier} {st}")
                if L // C >= spec["adoption_min_chunks"] and verdict != "batched_faster":
                    need.append(f"{key} {tier} {st}: {verdict}")
    return {"all_correctness_and_ab_checks_pass": gates, "established_regressions": regress,
            "required_but_not_established": need,
            "supports_adoption": gates and not regress and not need and not missing,
            "default_backend_after_m3": "baseline",
            "evidence_status": validation.get("status"), "spec": spec["name"], "missing_from_analysis": missing}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="where to write the outputs (default: the run directory)")
    ap.add_argument("--spec", default=mv.DEFAULT_SPEC, choices=sorted(mv.SPECS))
    args = ap.parse_args(argv)
    rd = Path(args.run_dir)
    od = Path(args.out_dir) if args.out_dir else rd
    if not rd.is_dir():
        sys.exit(f"run directory {rd} does not exist")
    outs = [od / "m3_validation.json", od / "m3_analysis.json", od / "m3_analysis.md", od / "samples_chronological.csv"]
    for o in outs:
        if o.exists():
            sys.exit(f"refusing to overwrite {o}")
    spec = mv.SPECS[args.spec]
    validation, sessions, prof = mv.load_and_validate(rd, spec)
    od.mkdir(parents=True, exist_ok=True)
    if validation["status"] != "valid":
        validation["decision"] = decide({}, spec, validation)
        (od / "m3_validation.json").write_text(json.dumps(validation, indent=1))
        print(render_rejected(validation))
        return EXIT[validation["status"]]
    out = {"run_dir": str(rd), "sessions": {}, "cases": {}, "decision": {}}
    if od != rd:
        out["out_dir"] = str(od)
    expected_pairs = len(spec["sessions"]) * spec["blocks"] * spec["reps"]
    expected_strata = len(spec["sessions"]) * spec["blocks"]

    # ---- sessions, validity, correctness
    all_samples, valid = [], []
    for sid, rep in sessions.items():
        m = rep["meta"]
        excl = []
        for s in rep["samples"]:
            s = {**s, "session": sid}
            all_samples.append(s)
            if s.get("warmup"):
                continue
            pr = sample_problems(s)
            if pr:
                excl.append(pr)
            else:
                valid.append(s)
        corr = [{"L": c["baseline"]["L"], "S": c["baseline"]["S"],
                 "strict_pass": {b: c[b]["strict_pass"] for b in BACKENDS},
                 "report_only_full_vs_restored": {b: c[b]["report_only_full_vs_restored"] for b in BACKENDS}}
                for c in rep["correctness"]]
        ab = [{"L": a["L"], "S": a["S"], "pass": a["pass"],
               **{t: {k: v for k, v in a[t].items() if isinstance(v, bool)} for t in TIERS}} for a in rep["ab_checks"]]
        clocks = []
        mon = rd / m.get("gpu_monitor_csv", "")
        if mon.is_file():
            for line in mon.read_text().splitlines():
                try:
                    clocks.append(float(line.split(",")[1].strip().split()[0]))
                except (IndexError, ValueError):
                    pass
        out["sessions"][sid] = {
            "seed": m["seed"], "started_at": m["started_at"], "elapsed_min": m["elapsed_min"],
            "n_measured": m["n_measured"], "n_excluded_recorded": m["n_excluded"], "n_excluded_recomputed": len(excl),
            "correctness_all": m["correctness_all"],
            "correctness_all_recomputed": validation["per_session"][sid]["correctness_and_ab_pass_recomputed"],
            "failures": rep["failures"], "block_order": m["block_order"],
            "correctness": corr, "ab_checks": ab, "git_head": rep["source"]["git_head"],
            "git_diff_head_sha256": rep["source"]["git_diff_head_sha256"],
            "tiered_cache_sha256": rep["source"]["files_sha256"].get("kvlab/tiered_cache.py"),
            "sm_clock_mhz": summarize(clocks), "restore_config": rep["restore_config"]}

    # ---- slow labels (protocol: e2e > 1.3x min of its (case, tier, backend, mode), all sessions)
    mins = defaultdict(lambda: float("inf"))
    for s in valid:
        k = (s["L"], s["S"], s["path"], s["backend"], s["mode"])
        mins[k] = min(mins[k], s["e2e_ns"])
    for s in valid:
        s["slow"] = s["e2e_ns"] > ms.SLOW_FACTOR * mins[(s["L"], s["S"], s["path"], s["backend"], s["mode"])]

    cases = sorted({(s["L"], s["S"]) for s in valid})
    for L, S in cases:
        cs = [s for s in valid if (s["L"], s["S"]) == (L, S)]
        strata_keys = sorted({(s["session"], s["block"]) for s in cs},
                             key=lambda k: min(x["t_wall"] for x in cs if (x["session"], x["block"]) == k))
        span = 0.0
        longest = 0.0
        for k in strata_keys:
            st = sorted((x for x in cs if (x["session"], x["block"]) == k), key=lambda x: x["t_wall"])
            span += st[-1]["t_wall"] - st[0]["t_wall"]
            rp = [x for x in st if x["path"] in TIERS]
            longest = max([longest, *ms.regimes([x["t_wall"] for x in rp], [x["slow"] for x in rp])])
        crec = {"L": L, "S": S, "n_chunks": L // C, "span_s": span, "longest_slow_regime_s": longest,
                "n_strata": len(strata_keys), "tiers": {}}
        if len(strata_keys) != expected_strata:  # cannot happen after validation; never analyse a partial case
            raise RuntimeError(f"{L}:{S}: {len(strata_keys)} strata, spec {expected_strata}")
        full = [s for s in cs if s["path"] == "full"]
        crec["full_context"] = summarize([value(s) for s in full])
        for tier in TIERS:
            trec = {}
            for mode, label in (("plain", "e2e"), ("synced", "restore_phase")):
                by = defaultdict(dict)
                for s in cs:
                    if s["path"] == tier and s["mode"] == mode:
                        by[(s["session"], s["block"], s["rep"])][s["backend"]] = s
                sa, sb, sess_of = [], [], []
                for k in strata_keys:
                    reps = sorted(r for (ss, bb, r) in by if (ss, bb) == k)
                    pairs = [by[(k[0], k[1], r)] for r in reps if len(by[(k[0], k[1], r)]) == 2]
                    if pairs:
                        sa.append([value(p["baseline"]) for p in pairs])
                        sb.append([value(p["batched"]) for p in pairs])
                        sess_of.append(k[0])
                rec = {"n_pairs": sum(len(x) for x in sa),
                       "baseline": summarize([v for x in sa for v in x]),
                       "batched": summarize([v for x in sb for v in x])}
                if rec["n_pairs"] != expected_pairs or len(sa) != expected_strata:  # as above
                    raise RuntimeError(f"{L}:{S} {tier} {mode}: {rec['n_pairs']} pairs in {len(sa)} strata, "
                                       f"spec {expected_pairs} in {expected_strata}")
                for stat in ("median_d", "mean_d"):
                    rec[stat] = ms.classify_case(sa, sb, sess_of, span, longest, stat=stat, seed=L * 1000 + S)
                rec["speedups"] = {st: {str(bl): ms.boot_ci(sa, sb, st, bl, seed=L * 1000 + S + bl)
                                        for bl in ms.BLOCK_LENGTHS}
                                   for st in ("marginal_median_ratio", "marginal_mean_ratio", "median_paired_ratio")}
                if mode == "plain":
                    rec["slow_share"] = {b: statistics.fmean(s["slow"] for s in cs if s["path"] == tier and
                                                             s["mode"] == mode and s["backend"] == b) for b in BACKENDS}
                trec[label] = rec
            # context: restore path minus full, per backend (plain, paired by repetition)
            ctx = {}
            for b in BACKENDS:
                by = defaultdict(dict)
                for s in cs:
                    if s["mode"] == "plain" and (s["path"] == "full" or (s["path"] == tier and s["backend"] == b)):
                        by[(s["session"], s["block"], s["rep"])]["full" if s["path"] == "full" else "r"] = s
                fa, rb = [], []
                for k in strata_keys:
                    reps = sorted(r for (ss, bb, r) in by if (ss, bb) == k and len(by[(ss, bb, r)]) == 2)
                    if reps:
                        fa.append([value(by[(k[0], k[1], r)]["full"]) for r in reps])
                        rb.append([value(by[(k[0], k[1], r)]["r"]) for r in reps])
                ctx[b] = {st: {str(bl): ms.boot_ci(fa, rb, st, bl, seed=L + bl) for bl in ms.BLOCK_LENGTHS}
                          for st in ("median_d", "mean_d")}
                for st in ("median_d", "mean_d"):
                    dirs = {ms.direction(r["ci"]) for r in ctx[b][st].values()}
                    ctx[b][f"{st}_direction_all_b"] = next(iter(dirs)) if len(dirs) == 1 else "mixed"
            trec["restore_minus_full_context"] = ctx
            # copies, bytes, memory
            lr = {b: sorted({json.dumps(s["last_restore"], sort_keys=True) for s in cs
                             if s["path"] == tier and s["backend"] == b}) for b in BACKENDS}
            trec["restore_record"] = {b: [json.loads(x) for x in v] for b, v in lr.items()}
            trec["request_peak_alloc_increment_bytes"] = {
                b: summarize([s["mem"]["peak_allocated"] - s["mem"]["allocated_before"] for s in cs
                              if s["path"] == tier and s["backend"] == b and s["mode"] == "plain"]) for b in BACKENDS}
            trec["peak_reserved_bytes_max"] = {
                b: max(s["mem"]["peak_reserved"] for s in cs if s["path"] == tier and s["backend"] == b) for b in BACKENDS}
            rm = [m for rep in sessions.values() for m in rep["restore_memory"] if (m["L"], m["S"]) == (L, S)]
            trec["restore_only_memory"] = {b: {"peak_above_start_bytes": sorted({m[f"{tier}_{b}"]["peak_above_start"] for m in rm}),
                                               "kept_after_return_bytes": sorted({m[f"{tier}_{b}"]["kept_after_return"] for m in rm})}
                                           for b in BACKENDS}
            crec["tiers"][tier] = trec
        out["cases"][f"{L}:{S}"] = crec

    # ---- decision rule (protocol §8): spec-driven, gate (1) recomputed by the validation
    out["decision"] = decide(out["cases"], spec, validation)
    out["validation"] = {k: v for k, v in validation.items() if k != "per_session"}
    (od / "m3_validation.json").write_text(json.dumps({**validation, "decision": out["decision"]}, indent=1))
    if prof:
        out["profile"] = prof["profile"]
        out["profile_session"] = {"git_diff_head_sha256": prof["source"]["git_diff_head_sha256"],
                                  "n_excluded": prof["meta"]["n_excluded"], "failures": prof["failures"]}
    (od / "m3_analysis.json").write_text(json.dumps(out, indent=1))

    # ---- chronological CSV (every sample, warmups flagged)
    with open(od / "samples_chronological.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "t_wall", "block", "rep", "order_pos", "warmup", "stage", "L", "S", "path", "backend",
                    "mode", "e2e_ms", "restore_phase_ms", "slow", "restore_len", "sources", "stored_chunks"])
        for s in sorted(all_samples, key=lambda x: (x["session"], x["t_wall"])):
            w.writerow([s["session"], f"{s['t_wall']:.6f}", s.get("block"), s.get("rep"), s.get("order_pos"),
                        s.get("warmup"), s.get("stage"), s["L"], s["S"], s["path"], s["backend"], s["mode"],
                        f"{s['e2e_ns'] / 1e6:.4f}",
                        f"{s['phases_ns']['restore'] / 1e6:.4f}" if s["mode"] == "synced" else "",
                        s.get("slow", ""), s["restore_len"],
                        "|".join(s["sources"]), "|".join(map(str, s["stored_chunks"]))])
    (od / "m3_analysis.md").write_text(render(out))
    print(render(out))
    return EXIT["valid"]


def fmt_ci(r, scale=1.0, nd=2):
    return f"{r['estimate'] * scale:.{nd}f} [{r['ci'][0] * scale:.{nd}f}, {r['ci'][1] * scale:.{nd}f}]"


def render(out) -> str:
    L = ["# M3 analysis (generated by scripts/m3_analyze.py)", "",
         "d = batched − baseline (ms), paired within a repetition. Intervals: stratified circular block "
         "bootstrap within contiguous blocks, 95% percentile. Verdicts follow docs/M3_PROTOCOL.md §6 (R1–R3).", ""]
    L += ["## Sessions", "", "| session | seed | elapsed min | measured | excluded | correctness + A/B | SM clock median MHz | tiered_cache sha256 |",
          "|---|---|---|---|---|---|---|---|"]
    for sid, s in out["sessions"].items():
        clk = s["sm_clock_mhz"]["median"] if s["sm_clock_mhz"] else "n/a"
        L.append(f"| {sid} | {s['seed']} | {s['elapsed_min']:.1f} | {s['n_measured']} | {s['n_excluded_recomputed']} | "
                 f"{s['correctness_all_recomputed']} | {clk} | `{(s['tiered_cache_sha256'] or '')[:12]}` |")
    L += ["", "## End-to-end first-token latency (plain, ms)", "",
          "| case | tier | baseline median / mean | batched median / mean | median d, b=4 [CI] | mean d, b=4 [CI] | median verdict | mean verdict | slow share base / batched |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            e = c["tiers"][tier]["e2e"]
            L.append(f"| {key} | {tier} | {e['baseline']['median']:.1f} / {e['baseline']['mean']:.1f} | "
                     f"{e['batched']['median']:.1f} / {e['batched']['mean']:.1f} | {fmt_ci(e['median_d']['by_block_len']['4'])} | "
                     f"{fmt_ci(e['mean_d']['by_block_len']['4'])} | {e['median_d']['verdict']} | {e['mean_d']['verdict']} | "
                     f"{e['slow_share']['baseline']:.0%} / {e['slow_share']['batched']:.0%} |")
    L += ["", "## Restore phase (synced diagnostic samples, ms)", "",
          "| case | tier | baseline median / mean | batched median / mean | median d, b=4 [CI] | mean d, b=4 [CI] | median verdict | mean verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            e = c["tiers"][tier]["restore_phase"]
            L.append(f"| {key} | {tier} | {e['baseline']['median']:.2f} / {e['baseline']['mean']:.2f} | "
                     f"{e['batched']['median']:.2f} / {e['batched']['mean']:.2f} | {fmt_ci(e['median_d']['by_block_len']['4'])} | "
                     f"{fmt_ci(e['mean_d']['by_block_len']['4'])} | {e['median_d']['verdict']} | {e['mean_d']['verdict']} |")
    L += ["", "## Speedups (plain e2e; b = 4; three different definitions)", "",
          "| case | tier | marginal median ratio | marginal mean ratio | median of paired ratios |", "|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            sp = c["tiers"][tier]["e2e"]["speedups"]
            L.append(f"| {key} | {tier} | {fmt_ci(sp['marginal_median_ratio']['4'], nd=3)} | "
                     f"{fmt_ci(sp['marginal_mean_ratio']['4'], nd=3)} | {fmt_ci(sp['median_paired_ratio']['4'], nd=3)} |")
    L += ["", "## Rules R1–R3 and block-length sensitivity (plain e2e)", "",
          "| case | tier | stat | CI b=1 | CI b=2 | CI b=4 | CI b=8 | per-session range | R1 | R2 | span s / longest regime s | n_eff | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            for st in ("median_d", "mean_d"):
                r = c["tiers"][tier]["e2e"][st]
                cis = " | ".join(f"[{r['by_block_len'][b]['ci'][0]:.2f}, {r['by_block_len'][b]['ci'][1]:.2f}]" for b in ("1", "2", "4", "8"))
                L.append(f"| {key} | {tier} | {st} | {cis} | [{r['per_session_range'][0]:.2f}, {r['per_session_range'][1]:.2f}] | "
                         f"{r['R1_all_block_lengths_agree']} | {r['R2_all_sessions_agree']} | "
                         f"{r['R3']['span_s']:.0f} / {r['R3']['longest_slow_regime_s']:.1f} | {r['R3']['ar1_effective_n']:.1f} | {r['verdict']} |")
    L += ["", "## Context: restore path − full (prepared-state recompute), plain e2e, b = 4 (report only)", "",
          "| case | full median / mean | tier | baseline: median d / mean d | batched: median d / mean d |", "|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            x = c["tiers"][tier]["restore_minus_full_context"]
            L.append(f"| {key} | {c['full_context']['median']:.1f} / {c['full_context']['mean']:.1f} | {tier} | "
                     f"{fmt_ci(x['baseline']['median_d']['4'])} / {fmt_ci(x['baseline']['mean_d']['4'])} | "
                     f"{fmt_ci(x['batched']['median_d']['4'])} / {fmt_ci(x['batched']['mean_d']['4'])} |")
    L += ["", "## Copies, bytes and device memory", "",
          "| case | tier | backend | H2D copy calls | placement ops | H2D bytes | scratch bytes | restore-only peak above start | kept after return | request peak increment (median) | peak reserved (max) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in out["cases"].items():
        for tier in TIERS:
            t = c["tiers"][tier]
            for b in BACKENDS:
                lr = t["restore_record"][b][0]
                rm = t["restore_only_memory"][b]
                L.append(f"| {key} | {tier} | {b} | {lr['h2d_copy_calls']} | {lr['placement_ops']} | {lr['h2d_bytes']:,} | "
                         f"{lr['device_scratch_bytes']:,} | {', '.join(f'{v:,}' for v in rm['peak_above_start_bytes'])} | "
                         f"{', '.join(f'{v:,}' for v in rm['kept_after_return_bytes'])} | "
                         f"{t['request_peak_alloc_increment_bytes'][b]['median']:,.0f} | {t['peak_reserved_bytes_max'][b]:,} |")
    if "profile" in out:
        L += ["", "## Measured CUDA API calls and device ops per call (torch.profiler, separate session)", "",
              "| L | tier | backend | unit | cudaMemcpyAsync | cudaStreamSynchronize | cudaLaunchKernel | device memcpy HtoD | DtoD | DtoH | kernels |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in out["profile"]:
            a, d = r["api_per_call"], r["device_ops_per_call"]
            L.append(f"| {r['L']} | {r['path']} | {r['backend']} | {r['unit']} | {a.get('cudaMemcpyAsync', 0):g} | "
                     f"{a.get('cudaStreamSynchronize', 0):g} | {a.get('cudaLaunchKernel', 0):g} | {d.get('memcpy_HtoD', 0):g} | "
                     f"{d.get('memcpy_DtoD', 0):g} | {d.get('memcpy_DtoH', 0):g} | {d.get('kernel', 0):g} |")
    L += ["", "## Per-session paired differences (plain e2e, ms)", "", "| case | tier | stat | " +
          " | ".join(out["sessions"]) + " |", "|---|---|---|" + "---|" * len(out["sessions"])]
    for key, c in out["cases"].items():
        for tier in TIERS:
            for st in ("median_d", "mean_d"):
                ps = c["tiers"][tier]["e2e"][st]["per_session"]
                L.append(f"| {key} | {tier} | {st} | " + " | ".join(f"{ps.get(s, float('nan')):.2f}" for s in out["sessions"]) + " |")
    d, v = out["decision"], out["validation"]
    c = v["counts"]
    L += ["", "## Decision rule (protocol §8)", "",
          f"- evidence validation against spec `{v['spec']}`: **{v['status']}** "
          f"({len(v['sessions_expected'])} planned sessions; measured samples "
          f"{c['measured_samples']['passed']}/{c['measured_samples']['expected']}; strict correctness checks "
          f"(session × case × backend) {c['correctness_checks']['passed']}/{c['correctness_checks']['expected']}; "
          f"A/B checks (session × case × tier) {c['ab_checks']['passed']}/{c['ab_checks']['expected']}; "
          f"warmup records kept as evidence, never used in statistics: block "
          f"{c['block_warmup_samples']['passed']}/{c['block_warmup_samples']['expected']}, process "
          f"{c['process_warmup_samples']['passed']}/{c['process_warmup_samples']['expected']}; "
          f"recomputed from the detailed records)",
          f"- all correctness and A/B checks pass: {d['all_correctness_and_ab_checks_pass']}",
          f"- established regressions: {d['established_regressions'] or 'none'}",
          f"- required (≥ 4 chunks, both tiers, median and mean) but not established: {d['required_but_not_established'] or 'none'}",
          f"- **supports adoption: {d['supports_adoption']}**; default backend after M3: `{d['default_backend_after_m3']}` (pending review)", ""]
    return "\n".join(L)


def render_rejected(v) -> str:
    c = v["counts"]
    L = [f"# M3 analysis NOT run: evidence status `{v['status']}` (spec `{v['spec']}`, run dir {v['run_dir']})", "",
         f"- session files found: {v['session_files_found'] or 'none'}; planned: {v['sessions_expected']}",
         *(f"- {k}: {x['passed']} passed / {x['present']} present / {x['expected']} expected" for k, x in c.items()),
         f"- problems ({len(v['problems'])}):", *(f"  - {p}" for p in v["problems"][:40]),
         *([f"  - ... {len(v['problems']) - 40} more in m3_validation.json"] if len(v["problems"]) > 40 else []),
         f"- failed correctness / A/B checks ({len(v['correctness_failures'])}):",
         *(f"  - {p}" for p in v["correctness_failures"][:40]),
         "", f"**supports adoption: {v['decision']['supports_adoption']}** (no performance result is drawn from "
             f"{'invalid or incomplete evidence' if v['status'] == 'invalid' else 'a run with a failed correctness check'})"]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
