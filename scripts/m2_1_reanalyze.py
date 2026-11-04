"""M2.1: dependence-aware reanalysis of the saved M2 run (CPU only; reads, never writes, the M2 run).

Usage: .venv/bin/python scripts/m2_1_reanalyze.py --m2-run results/m2/run_20261103T131543 \
           --out results/m2_1/<dir>/reanalysis.json
Writes reanalysis.json and reanalysis.md next to --out (refuses to overwrite).

Rules (fixed in this file before it was first run; docs/M2_1_REVIEW.md §3):
  - pairs: per repetition, restore tier vs the M2 "full" path of the same L group,
    in repetition order (= chronological order, checked against t_wall);
  - intervals: 95% percentile CIs from a circular moving-block bootstrap with block
    lengths 1 (= i.i.d., the M2 method), 2, 4, 8 repetitions, for the median and the
    mean of the paired differences;
  - a direction is "established" only if the CI excludes 0 with the same sign at
    every block length; otherwise "inconclusive" (never read as equivalence);
  - penalty metric: mean over paired repetitions of latency(chosen) - latency(other);
    point penalty = max(0, that); "established" only if the CI (block length 4 and 8
    both) is above 0;
  - policies: always-restore, always-recompute, frozen model (from predictions.json).
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kvlab import perf, perf_stats  # noqa: E402

N_BOOT = 4000


def by_case(samples):
    """{(path, L, S): {rep: ms}} for valid uninstrumented samples, plus t_wall per (L, rep, path, S)."""
    valid, excluded = perf.partition(samples)
    out, tw = {}, {}
    for s in valid:
        if s.get("stage") == "e2e":
            out.setdefault((s["path"], s["L"], s["S"]), {})[s["rep"]] = perf.ns_to_ms(s["e2e_ns"])
            tw[(s["path"], s["L"], s["S"], s["rep"])] = s["t_wall"]
    return out, tw, len(valid), len(excluded)


def chronological(tw, key, reps):
    t = [tw[key + (r,)] for r in reps]
    return all(b > a for a, b in zip(t, t[1:]))


def established(sens, stat):
    classes = {r["class"] for r in sens[stat].values() if r["status"] == "ok"}
    if len(classes) == 1 and len([r for r in sens[stat].values() if r["status"] == "ok"]) == len(perf_stats.BLOCK_LENGTHS):
        c = classes.pop()
        return c if c != "inconclusive" else "inconclusive"
    return "inconclusive"


def analyse_split(name, samples, preds_list, original_decisions):
    cases, tw, n_valid, n_excl = by_case(samples)
    pred = {(p["path"], p["L"], p["S"]): p["predicted_ms"] for p in preds_list}
    orig = {(d["tier"], d["L"], d["S"]): d for d in original_decisions}
    rows = []
    for (tier, L, S) in sorted(k for k in pred if k[0] in perf.RESTORE_TIERS):
        a, b = cases.get((tier, L, S)), cases.get(("full", L, S))
        if not a or not b:
            rows.append({"tier": tier, "L": L, "S": S, "status": "missing"})
            continue
        reps, d = perf_stats.paired_series(a, b)
        seed = zlib.crc32(f"{name}|{tier}|{L}|{S}".encode())
        sens = perf_stats.sensitivity(d, seed=seed, n_boot=N_BOOT)
        # Reproduce the original M2 interval exactly (same function, same seed as m2_score).
        o = perf.paired_difference(a, b, seed=L * 7 + S)
        o_saved = orig.get((tier, L, S), {}).get("measured_paired_diff", {})
        p_r, p_f = pred[(tier, L, S)], pred[("full", L, S)]
        chosen = "restore" if p_r < p_f else "full"
        ra, fa = np.array([a[r] for r in reps]), np.array([b[r] for r in reps])
        pen = {}
        for blk in (4, 8):
            ci = sens["mean"][str(blk)].get("ci")
            pen[str(blk)] = perf_stats.selection_penalty(chosen, d, ci)
        pen_status = {p["penalty_status"] for p in pen.values()}
        rows.append({
            "tier": tier, "L": L, "S": S, "n_pairs": len(reps),
            "chronological_order_verified": chronological(tw, (tier, L, S), reps) and chronological(tw, ("full", L, S), reps),
            "predicted_restore_ms": p_r, "predicted_full_ms": p_f, "predicted_choice": chosen,
            "restore_median_ms": float(np.median(ra)), "restore_mean_ms": float(ra.mean()),
            "full_median_ms": float(np.median(fa)), "full_mean_ms": float(fa.mean()),
            "paired_median_ms": float(np.median(d)), "paired_mean_ms": float(d.mean()),
            "original_iid_median_ci_ms": o.get("ci95_ms"), "original_ci_reproduced": o.get("ci95_ms") == o_saved.get("ci95_ms"),
            "original_measured_winner": orig.get((tier, L, S), {}).get("measured_winner"),
            "sensitivity": sens,
            "median_direction": established(sens, "median"), "mean_direction": established(sens, "mean"),
            "penalty_of_predicted_choice": pen,
            "penalty_status": pen_status.pop() if len(pen_status) == 1 else "inconclusive",
            "median_based_point_penalty_ms": max(0.0, (1 if chosen == "restore" else -1) * float(np.median(d))),
        })
    return {"split": name, "n_valid": n_valid, "n_excluded": n_excl, "rows": rows}, cases


def joint_policy_bootstrap(cases, rows, block_len, seed, n_boot=N_BOOT):
    """CI of sum over decisions of mean(restore - full), resampling repetition blocks jointly
    within each L group (cases of one group share repetitions); groups independent."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_boot)
    for L in sorted({r["L"] for r in rows}):
        group = [r for r in rows if r["L"] == L and "n_pairs" in r]
        reps = sorted(set.intersection(*[set(cases[(r["tier"], L, r["S"])]) & set(cases[("full", L, r["S"])])
                                          for r in group]))
        idx = perf_stats.block_bootstrap_indices(len(reps), block_len, n_boot, rng)
        for r in group:
            d = np.array([cases[(r["tier"], L, r["S"])][k] - cases[("full", L, r["S"])][k] for k in reps])
            total += d[idx].mean(axis=1)
    return [float(np.quantile(total, 0.025)), float(np.quantile(total, 0.975))]


def policies(rows, cases):
    ok = [r for r in rows if "n_pairs" in r]
    means = {(r["tier"], r["L"], r["S"]): {"restore": r["restore_mean_ms"], "full": r["full_mean_ms"]} for r in ok}
    medians = {(r["tier"], r["L"], r["S"]): {"restore": r["restore_median_ms"], "full": r["full_median_ms"]} for r in ok}
    choices = {"always_restore": {k: "restore" for k in means}, "always_recompute": {k: "full" for k in means},
               "frozen_model": {(r["tier"], r["L"], r["S"]): r["predicted_choice"] for r in ok}}
    out = {"by_mean": perf_stats.policy_totals(means, choices), "by_median": perf_stats.policy_totals(medians, choices),
           "model_equals_always_restore": choices["frozen_model"] == choices["always_restore"],
           "always_restore_minus_always_recompute_sum_of_mean_diffs_ms": float(sum(r["paired_mean_ms"] for r in ok))}
    out["always_restore_minus_always_recompute_ci_ms"] = {
        str(b): joint_policy_bootstrap(cases, ok, b, seed=1000 + b) for b in perf_stats.BLOCK_LENGTHS}
    for tier in perf.RESTORE_TIERS:
        sub = [r for r in ok if r["tier"] == tier]
        out[f"{tier}_only"] = {
            "n_cases": len(sub),
            "always_restore_minus_always_recompute_ms": float(sum(r["paired_mean_ms"] for r in sub)),
            "ci_ms": {str(b): joint_policy_bootstrap(cases, sub, b, seed=2000 + b) for b in perf_stats.BLOCK_LENGTHS}}
    return out


def fmt_ci(r):
    return "n/a" if r.get("status") != "ok" else f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]"


def md_report(res):
    L = ["# M2.1 reanalysis of the saved M2 run (generated by scripts/m2_1_reanalyze.py)", "",
         f"Source run: `{res['m2_run']}` (read only). Model `{res['model_sha256'][:16]}…`, "
         f"predictions `{res['predictions_sha256'][:16]}…`; nothing was refitted.", "",
         "Paired difference d = restore tier − M2 `full` path (recompute from the prepared cache state), per repetition, "
         "chronological. CIs: 95% circular moving-block bootstrap; b = block length in repetitions (b = 1 is the i.i.d. "
         "bootstrap M2 used). A direction is established only if all of b = 1, 2, 4, 8 agree.", ""]
    for an in (res["held_out"], res["in_sample"]):
        L += [f"## {an['split']}", "",
              "| tier | L | S | n | predicted | restore med / mean | full med / mean | d median | d mean | lag-1 ρ | n_eff | "
              "M2 i.i.d. CI (median) | median CI b=4 | median CI b=8 | mean CI b=4 | mean CI b=8 | M2 winner | median dir. | mean dir. | "
              "penalty (mean, point) | penalty status |",
              "|" + "---|" * 21]
        for r in an["rows"]:
            if "n_pairs" not in r:
                L.append(f"| {r['tier']} | {r['L']} | {r['S']} | missing |" + " |" * 17)
                continue
            s = r["sensitivity"]
            oc = r["original_iid_median_ci_ms"]
            rho = s["lag1_autocorr"]
            L.append(
                f"| {r['tier']} | {r['L']} | {r['S']} | {r['n_pairs']} | {r['predicted_choice']} | "
                f"{r['restore_median_ms']:.1f} / {r['restore_mean_ms']:.1f} | {r['full_median_ms']:.1f} / {r['full_mean_ms']:.1f} | "
                f"{r['paired_median_ms']:+.1f} | {r['paired_mean_ms']:+.1f} | {'n/a' if rho is None else f'{rho:+.2f}'} | "
                f"{s['ar1_effective_n']:.0f} | [{oc[0]:+.1f}, {oc[1]:+.1f}] | {fmt_ci(s['median']['4'])} | {fmt_ci(s['median']['8'])} | "
                f"{fmt_ci(s['mean']['4'])} | {fmt_ci(s['mean']['8'])} | {r['original_measured_winner']} | {r['median_direction']} | "
                f"{r['mean_direction']} | {r['penalty_of_predicted_choice']['4']['point_penalty_ms']:.1f} ms | {r['penalty_status']} |")
        c = an["counts"]
        L += ["", "```", json.dumps(c, indent=1), "```", ""]
    L += ["## Policies on the held-out decisions", "", "```", json.dumps(res["held_out_policies"], indent=1), "```", "",
          "## Checks", "", "```", json.dumps(res["checks"], indent=1), "```"]
    return "\n".join(L) + "\n"


def counts(rows):
    ok = [r for r in rows if "n_pairs" in r]
    c = {"decisions": len(rows), "with_data": len(ok)}
    for k in ("median_direction", "mean_direction", "penalty_status", "original_measured_winner"):
        vals = [r[k] for r in ok]
        c[k] = {v: vals.count(v) for v in sorted(set(map(str, vals)))}
    c["unreliable_block_count_at_b8"] = sum(not r["sensitivity"]["mean"]["8"].get("reliable_block_count", False) for r in ok)
    c["sum_point_penalty_mean_based_ms"] = float(sum(r["penalty_of_predicted_choice"]["4"]["point_penalty_ms"] for r in ok))
    c["sum_point_penalty_median_based_ms"] = float(sum(r["median_based_point_penalty_ms"] for r in ok))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.m2_run), Path(args.out)
    md = out.with_suffix(".md")
    for p in (out, md):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    model = json.loads((run / "model.json").read_text())
    preds = json.loads((run / "predictions.json").read_text())
    ev = json.loads((run / "evaluation_samples.json").read_text())
    cal = json.loads((run / "calibration_samples.json").read_text())
    saved = json.loads((run / "results.json").read_text())
    checks = {"model_digest_ok": model["sha256"] == perf.model_digest(model),
              "predictions_digest_ok": preds["sha256"] == perf.predictions_digest(preds),
              "evaluation_used_these_predictions": ev["meta"]["predictions_sha256"] == preds["sha256"],
              "predictions_before_evaluation": preds["created_unix"] < ev["meta"]["started_unix"]}
    held, ev_cases = analyse_split("held-out evaluation", ev["samples"], preds["predictions"], saved["held_out"]["decisions"])
    ins, _ = analyse_split("in-sample calibration", cal["samples"], preds["in_sample_calibration"],
                           saved["in_sample"]["decisions"])
    for an in (held, ins):
        an["counts"] = counts(an["rows"])
    checks["all_original_cis_reproduced"] = all(r.get("original_ci_reproduced") for r in held["rows"] + ins["rows"])
    checks["all_series_chronological"] = all(r.get("chronological_order_verified") for r in held["rows"] + ins["rows"])
    res = {"m2_run": str(run), "model_sha256": model["sha256"], "predictions_sha256": preds["sha256"],
           "n_boot": N_BOOT, "block_lengths": list(perf_stats.BLOCK_LENGTHS), "checks": checks,
           "held_out": held, "in_sample": ins, "held_out_policies": policies(held["rows"], ev_cases)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    md.write_text(md_report(res))
    print(md_report(res))


if __name__ == "__main__":
    main()
