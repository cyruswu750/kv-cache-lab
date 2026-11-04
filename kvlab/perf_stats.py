"""M2.1: dependence-aware statistics for paired latency series (pure CPU, numpy only).

M2 (`kvlab/perf.py`) bootstraps per-repetition paired differences as if the
repetitions were independent. The M2 data shows slow samples clustered in time,
so this module adds:
  - a circular moving-block bootstrap of the paired median and mean, over the
    chronologically ordered repetitions (block length 1 = the i.i.d. bootstrap);
  - lag-1 autocorrelation and an AR(1) effective-sample-size estimate;
  - three-way classification with "inconclusive" for an interval containing 0
    (never read as equivalence);
  - per-decision selection penalties and fixed-policy totals.
Units: milliseconds in, milliseconds out. Nothing here refits or changes the
frozen M2 model; `perf.py` is left byte-identical.
"""

import numpy as np

MIN_PAIRS = 10            # fewer paired repetitions than this -> no interval
MIN_BLOCKS = 5            # fewer than n / block_len blocks than this -> interval flagged unreliable
BLOCK_LENGTHS = (1, 2, 4, 8)


def paired_series(a_by_rep: dict, b_by_rep: dict) -> tuple[list, np.ndarray]:
    """(reps, a - b) over repetitions present in both, in repetition (= chronological) order."""
    reps = sorted(set(a_by_rep) & set(b_by_rep))
    return reps, np.asarray([a_by_rep[r] - b_by_rep[r] for r in reps], dtype=float)


def block_bootstrap_indices(n: int, block_len: int, n_boot: int, rng) -> np.ndarray:
    """Circular moving-block bootstrap: (n_boot, n) index matrix built from
    ceil(n / block_len) blocks of consecutive indices (wrapping), truncated to n."""
    if n < 1 or block_len < 1:
        raise ValueError(f"need n >= 1 and block_len >= 1 (n={n}, block_len={block_len})")
    block_len = min(block_len, n)
    n_blocks = -(-n // block_len)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    return idx.reshape(n_boot, n_blocks * block_len)[:, :n]


def block_bootstrap_ci(x, block_len: int, stat: str = "median", n_boot: int = 4000, seed: int = 0,
                       level: float = 0.95) -> dict:
    """Percentile CI of `stat` ("median" or "mean") of the chronological series x."""
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    out = {"n": n, "block_len": block_len, "stat": stat}
    if n < MIN_PAIRS:
        return {**out, "status": "insufficient_samples"}
    f = {"median": np.median, "mean": np.mean}[stat]
    rng = np.random.default_rng(seed)
    boot = f(x[block_bootstrap_indices(n, block_len, n_boot, rng)], axis=1)
    a = (1 - level) / 2
    lo, hi = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
    n_blocks = n / block_len
    return {**out, "status": "ok", "estimate": float(f(x)), "ci": [lo, hi], "n_blocks": n_blocks,
            "reliable_block_count": bool(n_blocks >= MIN_BLOCKS)}


def classify(ci) -> str:
    """For a restore - recompute difference: which path an interval establishes as faster."""
    lo, hi = ci
    if hi < 0:
        return "restore_faster"
    if lo > 0:
        return "recompute_faster"
    return "inconclusive"


def lag1_autocorr(x) -> float | None:
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]):
        return None
    xc = x - x.mean()
    return float(np.dot(xc[:-1], xc[1:]) / np.dot(xc, xc))


def ar1_effective_n(n: int, rho1) -> float:
    """n (1 - rho) / (1 + rho) for rho > 0; n otherwise (rough, assumes AR(1))."""
    if rho1 is None or rho1 <= 0:
        return float(n)
    return float(n * (1 - rho1) / (1 + rho1))


def sensitivity(x, block_lengths=BLOCK_LENGTHS, seed: int = 0, n_boot: int = 4000) -> dict:
    """Median and mean CIs of a chronological paired series at several block lengths."""
    out = {"n": int(np.asarray(x).size), "lag1_autocorr": lag1_autocorr(x)}
    out["ar1_effective_n"] = ar1_effective_n(out["n"], out["lag1_autocorr"])
    for stat in ("median", "mean"):
        rows = {}
        for b in block_lengths:
            r = block_bootstrap_ci(x, b, stat=stat, n_boot=n_boot, seed=seed + 7919 * b)
            if r["status"] == "ok":
                r["class"] = classify(r["ci"])
            rows[str(b)] = r
        out[stat] = rows
        classes = {r.get("class") for r in rows.values()}
        out[f"{stat}_class_stable_across_blocks"] = len(classes) == 1 and None not in classes
    return out


def selection_penalty(chosen: str, diff_restore_minus_full, ci_mean=None) -> dict:
    """Observed penalty of choosing `chosen` ("restore" | "full") for one case and tier.
    Metric: mean over paired repetitions of latency(chosen) - latency(other), clipped
    at 0 for the point estimate = mean(chosen) - min(mean(restore), mean(full)).
    `ci_mean` is a CI of mean(restore - full); the penalty is *established* only if
    the CI shows the chosen path slower, *established absent* only if it shows the
    chosen path faster, and *inconclusive* otherwise."""
    d = np.asarray(diff_restore_minus_full, dtype=float)
    sign = 1.0 if chosen == "restore" else -1.0
    chosen_minus_other = sign * float(d.mean())
    rec = {"chosen": chosen, "mean_chosen_minus_other_ms": chosen_minus_other,
           "point_penalty_ms": max(0.0, chosen_minus_other)}
    if ci_mean is not None:
        lo, hi = sorted((sign * ci_mean[0], sign * ci_mean[1]))
        rec["chosen_minus_other_ci_ms"] = [lo, hi]
        rec["penalty_status"] = "established" if lo > 0 else "established_absent" if hi < 0 else "inconclusive"
    return rec


def policy_totals(cases: dict, choices: dict) -> dict:
    """cases[key] = {"restore": mean ms, "full": mean ms}; choices[policy][key] = "restore" | "full".
    Returns per policy the sum of mean latencies, its oracle regret and the case count."""
    oracle = sum(min(v["restore"], v["full"]) for v in cases.values())
    out = {"oracle_sum_ms": oracle, "n_cases": len(cases)}
    for name, ch in choices.items():
        total = sum(cases[k][ch[k]] for k in cases)
        out[name] = {"sum_ms": total, "regret_vs_oracle_ms": total - oracle}
    return out
