"""M3 statistics (docs/M3_PROTOCOL.md §6), pure CPU (numpy). Reuses kvlab/perf_stats.py unchanged.

Paired series are split into strata: contiguous, chronologically ordered blocks (one per
session x block). The bootstrap resamples circular moving blocks *within* each stratum and
concatenates the strata, so no resampled block joins two separate stretches of time.
"""

import numpy as np

from kvlab.perf_stats import ar1_effective_n, block_bootstrap_indices, lag1_autocorr

BLOCK_LENGTHS = (1, 2, 4, 8)
N_BOOT = 4000
SLOW_FACTOR = 1.3


def stratified_indices(lengths, block_len: int, n_boot: int, rng) -> np.ndarray:
    """(n_boot, sum(lengths)) indices into the concatenated strata."""
    parts, off = [], 0
    for n in lengths:
        parts.append(block_bootstrap_indices(n, block_len, n_boot, rng) + off)
        off += n
    return np.concatenate(parts, axis=1)


STATS = {
    "median_d": lambda a, b: np.median(b - a, axis=-1),
    "mean_d": lambda a, b: np.mean(b - a, axis=-1),
    "marginal_median_ratio": lambda a, b: np.median(a, axis=-1) / np.median(b, axis=-1),
    "marginal_mean_ratio": lambda a, b: np.mean(a, axis=-1) / np.mean(b, axis=-1),
    "median_paired_ratio": lambda a, b: np.median(a / b, axis=-1),
}


def boot_ci(strata_a, strata_b, stat: str, block_len: int, n_boot: int = N_BOOT, seed: int = 0,
            level: float = 0.95) -> dict:
    """Stratified block-bootstrap percentile CI of STATS[stat](a, b); a = reference (baseline),
    b = candidate (batched); pairs share an index. d = b - a."""
    a = np.concatenate([np.asarray(x, float) for x in strata_a])
    b = np.concatenate([np.asarray(x, float) for x in strata_b])
    lengths = [len(x) for x in strata_a]
    if [len(x) for x in strata_b] != lengths:
        raise ValueError("strata of a and b differ in length")
    f = STATS[stat]
    rng = np.random.default_rng(seed)
    idx = stratified_indices(lengths, block_len, n_boot, rng)
    boot = f(a[idx], b[idx])
    q = (1 - level) / 2
    return {"stat": stat, "block_len": block_len, "estimate": float(f(a, b)),
            "ci": [float(np.quantile(boot, q)), float(np.quantile(boot, 1 - q))], "n": int(a.size)}


def direction(ci, null: float = 0.0) -> str:
    lo, hi = ci
    if hi < null:
        return "below"
    if lo > null:
        return "above"
    return "inconclusive"


def effective_n(strata_d) -> float:
    """Sum over strata of the AR(1) effective sample size of the paired differences."""
    return float(sum(ar1_effective_n(len(x), lag1_autocorr(x)) for x in strata_d))


def regimes(times, slow) -> list[float]:
    """Durations (s) of maximal chronological runs of slow samples; a run of one sample has
    duration 0. `times` are wall-clock seconds, `slow` booleans, both chronological."""
    out, start, last = [], None, None
    for t, s in zip(times, slow):
        if s and start is None:
            start = t
        if s:
            last = t
        if not s and start is not None:
            out.append(last - start)
            start = None
    if start is not None:
        out.append(last - start)
    return out


def classify_case(strata_a, strata_b, session_of_stratum, span_s: float, longest_regime_s: float,
                  stat: str = "median_d", seed: int = 0) -> dict:
    """Protocol §6 rules R1-R3 for one (case, tier, statistic). d = b - a; 'faster' means b < a."""
    rows = {str(bl): boot_ci(strata_a, strata_b, stat, bl, seed=seed + 7919 * bl) for bl in BLOCK_LENGTHS}
    dirs = {direction(r["ci"]) for r in rows.values()}
    r1 = dirs == {"below"} or dirs == {"above"}
    f = STATS[stat]
    sess = {}
    for s, a, b in zip(session_of_stratum, strata_a, strata_b):
        sess.setdefault(s, ([], []))
        sess[s][0].extend(a)
        sess[s][1].extend(b)
    per_session = {s: float(f(np.asarray(a, float), np.asarray(b, float))) for s, (a, b) in sorted(sess.items())}
    want = next(iter(dirs)) if r1 else None
    r2 = r1 and all((v < 0) if want == "below" else (v > 0) for v in per_session.values())
    d_strata = [np.asarray(b, float) - np.asarray(a, float) for a, b in zip(strata_a, strata_b)]
    n_eff = effective_n(d_strata)
    r3 = span_s >= 10 * longest_regime_s and n_eff >= 10
    if r1 and r2 and r3:
        verdict = "batched_faster" if want == "below" else "batched_slower"
    else:
        verdict = "inconclusive"
    return {"stat": stat, "by_block_len": rows, "per_session": per_session,
            "per_session_range": [min(per_session.values()), max(per_session.values())],
            "R1_all_block_lengths_agree": r1, "R2_all_sessions_agree": r2,
            "R3": {"span_s": span_s, "longest_slow_regime_s": longest_regime_s, "ar1_effective_n": n_eff,
                   "pass": r3},
            "verdict": verdict}
