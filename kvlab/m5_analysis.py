"""M5b: multi-candidate analysis of VALIDATED evaluation evidence (CPU only, no torch).
Definitions: docs/M5_PROTOCOL_DRAFT.md §8.

Three quantities are kept apart:
- **A. Candidate selection quality.** The FixedChunks latency of the candidate a policy selects, compared with
  the best measured candidate. It isolates the choice from the policy's own overhead.
  - *Naive regret:* mean(selected) − min over candidates of mean(candidate). A noisy, descriptive point
    estimate: the minimum of noisy means is biased low, so it is biased high.
  - *Cross-evaluated difference:* the reference candidate is selected on one predeclared session partition,
    and it and the policy's candidate are evaluated on the other; the two directions are combined with
    predeclared weights. The sign is kept: a negative value means the policy's candidate measured faster on
    held-out sessions than the separately selected reference. It is not an unbiased regret against a true
    oracle.
- **B. Delivered policy latency.** The policy runs themselves, including planning, prediction and validation.
  The primary reporting is the paired per-repetition difference against AlwaysRestore and against Recompute,
  whichever way it falls.
- **C. Prediction error.** `predict_ms` (with the actual m) minus the candidate mean.

**Best measured candidate:** the smallest candidate mean, with exact ties going to the larger L (the policy's
own tie rule). It is never the fastest single sample.

**Uncertainty:**
- **Resampling unit:** whole repetitions, so every candidate and policy of a repetition stays together and the
  pairing is preserved.
- **Within sessions:** circular moving blocks of `block_len` consecutive repetitions within each (case, session)
  stream (`kvlab.m3_stats.stratified_indices`). Blocks never join sessions.
- **Sessions:** fixed strata; they are not resampled. This assumes the sessions are representative and
  independent of each other; fresh processes do not prove it.
- **Cases:** a fixed, declared grid; they are not resampled. Aggregates are equal-weight means over the grid,
  and their intervals describe this grid on this machine, not arbitrary workloads.
- **Inadequate data:** too few sessions per partition half or too few blocks per session is flagged
  "inadequate", and no interval is reported.
- **No filtering:** slow samples are kept.

**Ambiguity (per case):**
- **Procedure:** simultaneous bootstrap intervals over the family of all pairwise candidate mean differences of
  that case, using a studentized max statistic at level 1 − alpha.
- **Possibly-best set:** the candidates that no other candidate beats with a simultaneous interval entirely
  below zero. A case is ambiguous if the set has more than one member.
- **Scope:** approximate simultaneous coverage within one case only; no error control across cases.
"""

import math

import numpy as np

from kvlab import m5_spec as sp
from kvlab.m3_stats import stratified_indices

ANALYSIS_FORMAT = "kvlab-m5-analysis-v1"
DIFFS = (("cost_aware", "always_restore"), ("cost_aware", "recompute"), ("always_restore", "recompute"))


# ---- point definitions (hand-testable) -----------------------------------------------------

def best_candidate(means: dict) -> int:
    """argmin over candidates; exact ties -> larger L."""
    best_L, best = None, math.inf
    for L in sorted(means):
        if means[L] <= best:
            best_L, best = L, means[L]
    return best_L


def naive_regret(means: dict, chosen: int) -> float:
    return means[chosen] - means[best_candidate(means)]


def cross_evaluated(means_a: dict, means_b: dict, chosen: int, w_a: float, w_b: float) -> float:
    """w_a: select the reference on A, evaluate both on B; w_b: the reverse. Signed."""
    ref_a, ref_b = best_candidate(means_a), best_candidate(means_b)
    return w_a * (means_b[chosen] - means_b[ref_a]) + w_b * (means_a[chosen] - means_a[ref_b])


def prediction_error(predicted, mean_ms: float) -> dict:
    if predicted is None:
        return {"error_ms": None, "relative_error": None}
    return {"error_ms": predicted - mean_ms, "relative_error": (predicted - mean_ms) / mean_ms}


# ---- data ----------------------------------------------------------------------------------

def collect(spec: dict, sessions: dict) -> dict:
    """{case_id: {"labels", "sessions": [sid...], "X": {sid: (reps, labels) ms}}}; measured samples only."""
    out = {}
    for c in spec["cases"]:
        labels = sp.run_labels(spec, c)
        col = {lab: j for j, lab in enumerate(labels)}
        X = {}
        for sid in spec["schedule"]["sessions"]:
            reps = spec["schedule"]["reps"]
            m = np.full((reps, len(labels)), np.nan)
            for s in sessions[sid]["samples"]:
                if s["case_id"] == c["case_id"] and s["phase"] == "measured":
                    if not np.isnan(m[s["rep"], col[s["run"]]]):
                        raise ValueError(f"{c['case_id']} {sid}: duplicate measured sample {s['rep']} {s['run']}")
                    m[s["rep"], col[s["run"]]] = s["latency_ns"] / 1e6
            if np.isnan(m).any():
                raise ValueError(f"{c['case_id']} {sid}: incomplete measured repetitions")
            X[sid] = m
        out[c["case_id"]] = {"labels": labels, "sessions": list(spec["schedule"]["sessions"]), "X": X}
    return out


def _cand_cols(labels):
    return [(int(lab.split(":")[1]), j) for j, lab in enumerate(labels) if lab.startswith("cand:")]


def _case_point(labels, Xcat, mask_a, chosen: dict, weights: dict) -> dict:
    """Point statistics from rows Xcat (n, labels); mask_a marks partition-A rows (or None)."""
    cols = _cand_cols(labels)
    means = {L: float(Xcat[:, j].mean()) for L, j in cols}
    res = {"means": means, "best": best_candidate(means), "regret": {}, "cross": {}, "delivered": {}, "diffs": {}}
    for P, L in chosen.items():
        res["regret"][P] = naive_regret(means, L)
        if mask_a is not None:
            ma = {Lc: float(Xcat[mask_a, j].mean()) for Lc, j in cols}
            mb = {Lc: float(Xcat[~mask_a, j].mean()) for Lc, j in cols}
            res["cross"][P] = cross_evaluated(ma, mb, L, weights["A"], weights["B"])
    pcol = {lab.split(":")[1]: j for j, lab in enumerate(labels) if lab.startswith("policy:")}
    for P, j in pcol.items():
        res["delivered"][P] = float(Xcat[:, j].mean())
    for a, b in DIFFS:
        if a in pcol and b in pcol:
            res["diffs"][f"{a}_minus_{b}"] = float((Xcat[:, pcol[a]] - Xcat[:, pcol[b]]).mean())
    return res


def _interval(boot, level):
    q = (1 - level) / 2
    return [float(np.quantile(boot, q)), float(np.quantile(boot, 1 - q))]


def analyze_case(case: dict, data: dict, chosen: dict, an: dict, rng) -> dict:
    labels, sids = data["labels"], data["sessions"]
    X = [data["X"][s] for s in sids]
    lengths = [x.shape[0] for x in X]
    Xcat = np.concatenate(X, axis=0)
    part = an.get("partition")
    mask_a = None
    if part:
        mask_a = np.concatenate([np.full(n, s in part["A"]) for s, n in zip(sids, lengths)])
    point = _case_point(labels, Xcat, mask_a, chosen, an.get("partition_weights") or {"A": 0.5, "B": 0.5})
    b, n_boot = an["block_len"], an["n_boot"]
    adequate = min(lengths) >= an["min_blocks_per_session"] * b
    cross_adequate = bool(part) and adequate and \
        min(len(part["A"]), len(part["B"])) >= an["min_sessions_per_half"]
    boot = None
    if adequate:
        idx = stratified_indices(lengths, b, n_boot, rng)       # (n_boot, n): whole repetitions, within sessions
        boot = Xcat[idx]                                        # (n_boot, n, labels)
    cols = _cand_cols(labels)
    res = {"point": point, "adequate": adequate, "cross_adequate": cross_adequate, "n_reps": lengths}
    if boot is None:
        res["possibly_best"], res["ambiguous"], res["ambiguity_status"] = None, None, "inadequate"
        return res
    cand_boot = np.stack([boot[:, :, j].mean(axis=1) for _L, j in cols], axis=1)   # (n_boot, K)
    res["_boot_means"] = cand_boot
    # simultaneous pairwise intervals (studentized max over the family of this case's pairs)
    K = len(cols)
    mu = np.array([point["means"][L] for L, _j in cols])
    if K == 1:
        res["possibly_best"], res["ambiguous"], res["ambiguity_status"] = [cols[0][0]], False, "single_candidate"
    else:
        pairs = [(i, j) for i in range(K) for j in range(K) if i < j]
        d = np.array([mu[i] - mu[j] for i, j in pairs])
        db = np.stack([cand_boot[:, i] - cand_boot[:, j] for i, j in pairs], axis=1)
        se = db.std(axis=0, ddof=1)
        var = se > 0  # a difference that never varies across replicates has a zero-width interval
        tmax = np.max(np.abs(db[:, var] - d[var]) / se[var], axis=1) if var.any() else np.zeros(n_boot)
        q = float(np.quantile(tmax, 1 - an["ambiguity_alpha"]))
        half = np.where(var, q * np.where(var, se, 0.0), 0.0)
        lo = {(i, j): d[k] - half[k] for k, (i, j) in enumerate(pairs)}
        hi = {(i, j): d[k] + half[k] for k, (i, j) in enumerate(pairs)}
        beaten = set()
        for (i, j) in pairs:
            if lo[(i, j)] > 0:   # mu_i - mu_j > 0 simultaneously: i is beaten by j
                beaten.add(i)
            if hi[(i, j)] < 0:   # mu_i - mu_j < 0: j is beaten by i
                beaten.add(j)
        res["possibly_best"] = [cols[i][0] for i in range(K) if i not in beaten]
        res["ambiguous"] = len(res["possibly_best"]) > 1
        res["ambiguity_status"] = "ok"
        res["simultaneous_q"] = q
    # bootstrap of the per-case statistics (selection re-done inside every replicate)
    reg, crs, dif = {P: [] for P in chosen}, {P: [] for P in chosen}, {}
    for r in range(n_boot):
        pr = _case_point(labels, boot[r], mask_a, chosen, an.get("partition_weights") or {"A": 0.5, "B": 0.5})
        for P in chosen:
            reg[P].append(pr["regret"][P])
            if mask_a is not None:
                crs[P].append(pr["cross"][P])
        for k, v in pr["diffs"].items():
            dif.setdefault(k, []).append(v)
    res["_boot"] = {"regret": reg, "cross": crs, "diffs": dif}
    return res


def chosen_lengths(case: dict, predictions) -> dict:
    out = {"always_restore": case["candidates"][-1], "recompute": 0}
    if predictions is not None:
        out["cost_aware"] = predictions["cases"][case["case_id"]]["decision"]["selected_len"]
    return out


def analyze(spec: dict, sessions: dict, predictions: dict, validation: dict) -> dict:
    """Refuses anything but valid evidence (validation exit 0)."""
    if validation.get("exit") != 0 or not validation.get("valid"):
        raise ValueError("invalid evidence: no performance analysis")
    if spec["split"] != "evaluation":
        raise ValueError("the policy analysis is for evaluation evidence")
    an = spec["analysis"]
    data = collect(spec, sessions)
    cases, agg_boot = {}, {}
    for ci, c in enumerate(spec["cases"]):
        rng = np.random.default_rng([an["seed"], ci])
        chosen = chosen_lengths(c, predictions)
        r = analyze_case(c, data[c["case_id"]], chosen, an, rng)
        pc = predictions["cases"][c["case_id"]]
        cand = {}
        for L, mean in r["point"]["means"].items():
            pred = pc["predictions"][str(L)]
            cand[str(L)] = {"mean_ms": mean, "predicted_ms": pred["predicted_ms"],
                            "unavailable": pred.get("unavailable"), **prediction_error(pred["predicted_ms"], mean)}
        pol = {P: {"selected_len": L, "agrees_with_best": L == r["point"]["best"],
                   "naive_regret_ms": r["point"]["regret"][P],
                   "cross_evaluated_ms": r["point"]["cross"].get(P),
                   "delivered_mean_ms": r["point"]["delivered"].get(P),
                   "in_possibly_best_set": None if r["possibly_best"] is None else L in r["possibly_best"]}
               for P, L in chosen.items()}
        cases[c["case_id"]] = {"candidates": cand, "best_measured": r["point"]["best"],
                               "possibly_best": r["possibly_best"], "ambiguous": r["ambiguous"],
                               "ambiguity_status": r["ambiguity_status"], "policies": pol,
                               "delivered_differences_ms": r["point"]["diffs"],
                               "cost_aware_decision_reason": pc["decision"]["reason"],
                               "n_reps_per_session": r["n_reps"], "adequate": r["adequate"],
                               "cross_adequate": r["cross_adequate"]}
        if "_boot" in r:
            agg_boot[c["case_id"]] = r["_boot"]
    n = len(cases)
    w = 1.0 / n  # equal case weights (spec: analysis.case_weight = "equal")
    agg = {"case_weight": "equal", "n_cases": n, "policies": {}, "delivered_differences_ms": {},
           "prediction_error": _pred_agg(cases)}
    complete_boot = len(agg_boot) == n
    for P in chosen_lengths(spec["cases"][0], predictions):
        regs = [cases[c]["policies"][P]["naive_regret_ms"] for c in cases]
        crs = [cases[c]["policies"][P]["cross_evaluated_ms"] for c in cases]
        agrees = [cases[c]["policies"][P]["agrees_with_best"] for c in cases]
        entry = {"agreement_fraction": sum(agrees) / n, "agreement_note": "noisy point estimate",
                 "naive_mean_regret_ms": w * sum(regs),
                 "cross_evaluated_mean_ms": None if None in crs else w * sum(crs)}
        if complete_boot:
            entry["naive_mean_regret_interval_ms"] = _interval(
                np.mean([agg_boot[c]["regret"][P] for c in cases], axis=0), an["level"])
            if entry["cross_evaluated_mean_ms"] is not None and all(cases[c]["cross_adequate"] for c in cases):
                entry["cross_evaluated_interval_ms"] = _interval(
                    np.mean([agg_boot[c]["cross"][P] for c in cases], axis=0), an["level"])
            else:
                entry["cross_interval_status"] = "inadequate"
        else:
            entry["interval_status"] = "inadequate"
        agg["policies"][P] = entry
    for key in cases[next(iter(cases))]["delivered_differences_ms"]:
        e = {"mean_ms": w * sum(cases[c]["delivered_differences_ms"][key] for c in cases)}
        if complete_boot:
            e["interval_ms"] = _interval(np.mean([agg_boot[c]["diffs"][key] for c in cases], axis=0), an["level"])
            e["direction"] = "below 0" if e["interval_ms"][1] < 0 else "above 0" if e["interval_ms"][0] > 0 \
                else "inconclusive"
        else:
            e["interval_status"] = "inadequate"
        agg["delivered_differences_ms"][key] = e
    agg["ambiguous_cases"] = [c for c in cases if cases[c]["ambiguous"]]
    agg["fallback_cases"] = [c for c in cases if cases[c]["cost_aware_decision_reason"].startswith("out_of_domain")]
    return {"format": ANALYSIS_FORMAT, "spec_sha256": sp.digest(spec), "predictions_sha256": predictions["sha256"],
            "artifact_digest": predictions["artifact_digest"], "analysis_parameters": an, "cases": cases,
            "aggregate": agg, "scope": SCOPE}


def _pred_agg(cases: dict) -> dict:
    rows = [v for c in cases.values() for v in c["candidates"].values()]
    avail = [r for r in rows if r["error_ms"] is not None]
    if not avail:
        return {"n_rows": len(rows), "n_unavailable": len(rows)}
    e = np.array([r["error_ms"] for r in avail])
    rel = np.array([r["relative_error"] for r in avail])
    return {"n_rows": len(rows), "n_unavailable": len(rows) - len(avail), "row_weight": "equal per candidate",
            "mean_signed_error_ms": float(e.mean()), "mean_abs_error_ms": float(np.abs(e).mean()),
            "mean_abs_relative_error": float(np.abs(rel).mean()), "max_abs_relative_error": float(np.abs(rel).max())}


SCOPE = ("One machine, model and software stack; batch 1; synchronous first-token latency from a controlled, "
         "warm-page-cache state; the declared case grid only (fixed, not sampled). The holdout tests new request/"
         "state configurations and token content inside the calibrated box, not other workloads, machines, cold "
         "storage, concurrency, trace-level cache dynamics, tail latency or service-level TTFT.")


def render_md(res: dict) -> str:
    agg = res["aggregate"]
    lines = [f"# M5 evaluation analysis ({res['format']})", "", f"Scope: {res['scope']}", "",
             "## Delivered policy latency (paired per repetition; equal case weights)", "",
             "| Difference | Mean ms | Interval ms | Direction |", "|---|---:|---|---|"]
    for k, e in agg["delivered_differences_ms"].items():
        lines.append(f"| {k} | {e['mean_ms']:.3f} | {e.get('interval_ms', e.get('interval_status'))} | "
                     f"{e.get('direction', '—')} |")
    lines += ["", "## Candidate selection quality (FixedChunks latencies of the selected candidates)", "",
              "| Policy | Agreement (noisy point estimate) | Naive mean regret ms | Cross-evaluated mean ms |",
              "|---|---:|---:|---:|"]
    for P, e in agg["policies"].items():
        ce = e["cross_evaluated_mean_ms"]
        lines.append(f"| {P} | {e['agreement_fraction']:.3f} | {e['naive_mean_regret_ms']:.3f} | "
                     f"{'—' if ce is None else f'{ce:.3f}'} |")
    pe = agg["prediction_error"]
    lines += ["", f"Prediction error: {pe}", "", f"Ambiguous cases: {agg['ambiguous_cases']}",
              f"Out-of-domain fallbacks: {agg['fallback_cases']}", ""]
    return "\n".join(lines)

