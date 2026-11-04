"""M2: score frozen predictions against held-out measurements (CPU only).

Usage: .venv/bin/python scripts/m2_score.py --run-dir results/m2/<run>
Verifies the freeze chain (model digest -> predictions digest -> evaluation run that
recorded that digest and started after the predictions were written), then joins.
Writes results.json and results.md (never overwritten).
"""

import argparse
import bisect
import datetime
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kvlab import perf  # noqa: E402


def load_monitor(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((ts, int(p[1].split()[0])))
        except (ValueError, IndexError):
            continue
    return sorted(rows)


def nearest_clock(rows, t):
    if not rows:
        return None
    ts = [r[0] for r in rows]
    i = bisect.bisect_left(ts, t)
    cand = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    j = min(cand, key=lambda j: abs(rows[j][0] - t))
    return rows[j][1] if abs(rows[j][0] - t) < 1.0 else None


def e2e_by_case(valid):
    out = {}
    for s in valid:
        if s.get("stage") == "e2e":
            out.setdefault((s["path"], s["L"], s["S"]), {})[s["rep"]] = perf.ns_to_ms(s["e2e_ns"])
    return out


def analyse(split_name, data, valid, preds_list, monitor_rows):
    by_case = e2e_by_case(valid)
    summary = {k: perf.summarize(v.values(), seed=zlib.crc32(repr(k).encode())) for k, v in by_case.items()}
    med = {k: v["median_ms"] for k, v in summary.items() if v["status"] == "ok"}
    diffs = {}
    for (path, L, S) in by_case:
        if path in perf.RESTORE_TIERS and ("full", L, S) in by_case:
            diffs[(path, L, S)] = perf.paired_difference(by_case[(path, L, S)], by_case[("full", L, S)], seed=L * 7 + S)
    pred = {(p["path"], p["L"], p["S"]): p["predicted_ms"] for p in preds_list}
    errors = []
    for (path, L, S), pm in sorted(pred.items()):
        m = med.get((path, L, S))
        errors.append({"path": path, "L": L, "S": S, "predicted_ms": pm, "measured_median_ms": m,
                       "error_ms": None if m is None else pm - m,
                       "rel_error": None if m is None else (pm - m) / m})
    dec = perf.decisions(pred, med, {k: v for k, v in diffs.items() if ("full",) + k[1:] in pred and k in pred})
    clocks = {}
    for s in valid:
        if s.get("stage") == "e2e":
            c = nearest_clock(monitor_rows, s["t_wall"])
            clocks.setdefault((s["path"], s["L"], s["S"]), []).append((c, perf.ns_to_ms(s["e2e_ns"])))
    clock_summary = {}
    for k, v in clocks.items():
        known = [c for c, _ in v if c is not None]
        hi = [ms for c, ms in v if c is not None and c >= 2200]
        lo = [ms for c, ms in v if c is not None and c < 2200]
        clock_summary[k] = {"n": len(v), "n_clock_known": len(known),
                            "frac_sm_ge_2200MHz": (len(hi) / len(known)) if known else None,
                            "median_ms_sm_ge_2200": float(np.median(hi)) if hi else None,
                            "median_ms_sm_lt_2200": float(np.median(lo)) if lo else None}
    phases = perf.case_medians(valid)
    return {"split": split_name, "summary": summary, "paired_restore_minus_full": diffs, "errors": errors,
            "decisions": dec, "clock": clock_summary, "phase_medians": phases}


def keystr(d):
    return {"|".join(map(str, k)): v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    for name in ("results.json", "results.md"):
        if (run / name).exists():
            sys.exit(f"refusing to overwrite {run / name}")
    model = json.loads((run / "model.json").read_text())
    preds = json.loads((run / "predictions.json").read_text())
    ev = json.loads((run / "evaluation_samples.json").read_text())
    cal = json.loads((run / "calibration_samples.json").read_text())
    chain = {
        "model_digest_ok": model["sha256"] == perf.model_digest(model),
        "predictions_digest_ok": preds["sha256"] == perf.predictions_digest(preds),
        "predictions_reference_model": preds["model_sha256"] == model["sha256"],
        "evaluation_used_these_predictions": ev["meta"].get("predictions_sha256") == preds["sha256"],
        "predictions_written_before_evaluation_started": preds["created_unix"] < ev["meta"]["started_unix"],
        "prediction_cases_equal_evaluation_split": sorted(map(tuple, preds["cases"])) == sorted(perf.EVALUATION),
        "no_calibration_overlap": not (set(perf.CALIBRATION) & set(perf.EVALUATION)),
    }
    if not all(chain.values()):
        sys.exit(f"freeze chain broken: {chain}")
    v_ev, x_ev = perf.partition(ev["samples"])
    v_cal, x_cal = perf.partition(cal["samples"])
    held = analyse("evaluation (held-out)", ev, v_ev, preds["predictions"], load_monitor(run / ev["meta"]["gpu_monitor_csv"]))
    ins = analyse("calibration (in-sample)", cal, v_cal, preds["in_sample_calibration"],
                  load_monitor(run / cal["meta"]["gpu_monitor_csv"]))

    def err_stats(rows):
        out = {}
        for path in perf.PATHS:
            e = [abs(r["rel_error"]) for r in rows if r["path"] == path and r["rel_error"] is not None]
            if e:
                out[path] = {"n": len(e), "median_abs_rel_error": float(np.median(e)), "max_abs_rel_error": float(max(e))}
        return out

    def dec_stats(rows):
        scorable = [r for r in rows if r["correct"] is not None]
        wrong = [r for r in scorable if not r["correct"]]
        return {"n": len(rows), "scorable": len(scorable), "ties_or_unknown": len(rows) - len(scorable),
                "correct": sum(r["correct"] for r in scorable), "wrong": len(wrong),
                "total_penalty_ms": float(sum(r["penalty_ms"] for r in wrong)),
                "max_penalty_ms": float(max((r["penalty_ms"] for r in wrong), default=0.0))}

    all_dec = held["decisions"] + ins["decisions"]
    measured_x = perf.measured_crossings(all_dec)
    scan = {(p["path"], p["L"], p["S"]): p["predicted_ms"] for p in preds["scan"]}
    predicted_x = {}
    for tier in perf.RESTORE_TIERS:
        for S in perf.S_GRID:
            vals = {L: scan[(tier, L, S)] - scan[("full", L, S)] for L in perf.SCAN_L}
            predicted_x[(tier, S)] = {"changes": perf.sign_changes(vals),
                                      "restore_minus_full_ms": {str(L): v for L, v in vals.items()}}
    evicted = {}
    for an in (held, ins):
        for (path, L, S), sm in an["summary"].items():
            if path == "disk_evicted" and sm["status"] == "ok":
                warm = an["summary"].get(("disk", L, S), {})
                evicted[f"{L}|{S}"] = {"split": an["split"], "evicted_median_ms": sm["median_ms"],
                                       "warm_disk_median_ms": warm.get("median_ms"),
                                       "full_median_ms": an["summary"].get(("full", L, S), {}).get("median_ms")}
    results = {
        "freeze_chain": chain,
        "model_sha256": model["sha256"], "predictions_sha256": preds["sha256"],
        "exclusions": {"evaluation": len(x_ev), "calibration": len(x_cal),
                       "evaluation_reasons": [x["excluded_because"] for x in x_ev][:50],
                       "calibration_reasons": [x["excluded_because"] for x in x_cal][:50]},
        "not_run": ev.get("not_run", []) + cal.get("not_run", []), "failures": ev.get("failures", []) + cal.get("failures", []),
        "correctness": {"evaluation_strict_all": all(c["strict_pass"] for c in ev["correctness"]),
                        "calibration_strict_all": all(c["strict_pass"] for c in cal["correctness"]),
                        "cases": ev["correctness"] + cal["correctness"]},
        "held_out_error": err_stats(held["errors"]), "in_sample_error": err_stats(ins["errors"]),
        "held_out_decisions": dec_stats(held["decisions"]), "in_sample_decisions": dec_stats(ins["decisions"]),
        "measured_crossings": keystr(measured_x), "predicted_crossings": keystr(predicted_x),
        "disk_evicted_diagnostic": evicted,
        "held_out": {k: (keystr(v) if isinstance(v, dict) else v) for k, v in held.items()},
        "in_sample": {k: (keystr(v) if isinstance(v, dict) else v) for k, v in ins.items()},
        "micro": cal.get("micro"),
    }
    (run / "results.json").write_text(json.dumps(results, indent=1, default=str))

    lines = ["# M2 results (generated by scripts/m2_score.py)", "",
             f"model `{model['sha256'][:16]}…`, predictions `{preds['sha256'][:16]}…`; freeze chain: all checks true", "",
             "## Held-out evaluation: per-case medians (ms), predictions, winners", "",
             "| L | S | path | measured median [IQR] | predicted | error | SM≥2200 MHz share |", "|---|---|---|---|---|---|---|"]
    for r in sorted(held["errors"], key=lambda r: (r["L"], r["S"], r["path"])):
        sm = held["summary"].get((r["path"], r["L"], r["S"]), {})
        ck = held["clock"].get((r["path"], r["L"], r["S"]), {})
        frac = ck.get("frac_sm_ge_2200MHz")
        lines.append(f"| {r['L']} | {r['S']} | {r['path']} | {sm.get('median_ms', float('nan')):.2f} "
                     f"[{sm.get('p25_ms', float('nan')):.2f}–{sm.get('p75_ms', float('nan')):.2f}] | "
                     f"{r['predicted_ms']:.2f} | {r['error_ms']:+.2f} ({r['rel_error']:+.1%}) | "
                     f"{'n/a' if frac is None else f'{frac:.0%}'} |")
    for title, an in (("Held-out decisions", held), ("In-sample (calibration) decisions", ins)):
        lines += ["", f"## {title}", "", "| tier | L | S | predicted Δ (restore−full) | measured paired Δ [95% CI] | predicted | measured | correct | penalty |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for d in sorted(an["decisions"], key=lambda d: (d["tier"], d["S"], d["L"])):
            md = d["measured_paired_diff"]
            ci = md.get("ci95_ms", [float("nan")] * 2)
            pen = "" if d["penalty_ms"] is None else f"{d['penalty_ms']:.2f} ms ({d['penalty_rel']:.0%})"
            lines.append(f"| {d['tier']} | {d['L']} | {d['S']} | {d['predicted_restore_minus_full_ms']:+.2f} | "
                         f"{md.get('median_ms', float('nan')):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {d['predicted_winner']} | "
                         f"{d['measured_winner']} | {d['correct']} | {pen} |")
    lines += ["", "## Error summary", "", "```", json.dumps({"held_out": results["held_out_error"],
              "in_sample": results["in_sample_error"], "held_out_decisions": results["held_out_decisions"],
              "in_sample_decisions": results["in_sample_decisions"]}, indent=1), "```",
              "", "## Crossings", "", "```", json.dumps({"measured": results["measured_crossings"],
              "predicted_changes": {k: v["changes"] for k, v in results["predicted_crossings"].items()}}, indent=1), "```",
              "", "## disk_evicted (diagnostic; Linux page cache evicted, WSL2 host cache unknown)", "", "```",
              json.dumps(evicted, indent=1), "```"]
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
