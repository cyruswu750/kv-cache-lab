"""Session-level statistics for M4-B (docs/M4B_PROTOCOL.md §8). CPU only; numpy only (no scipy).

The unit is the session (one fresh process). For a comparison (trace, A, B), each session s gives
D_s = trace-mean latency of A - trace-mean latency of B, over the same requests of the fixed trace.

Verdict rule for the declared primary family (fixed in the spec):
- "A faster" iff every D_s < 0 (unanimity, an observed stability rule; no probability bound is
  attached to it) AND the two-sided Student-t interval for mean(D_s) at level 1 - alpha/m
  (Bonferroni over the m primary comparisons, df = R - 1) lies entirely below 0;
- "A slower" symmetrically; otherwise "inconclusive". There is no equivalence verdict.

Assumptions, stated with every result: the D_s are treated as independent and approximately
normal. Fresh processes do not establish independence (machine, driver and thermal regimes can
persist across processes), so every interval and verdict is conditional on that assumption.
"""

import math

import numpy as np

ASSUMPTIONS = ("Sessions are treated as independent, approximately normal replicates. Fresh processes do not "
               "prove independence: machine, driver or thermal regimes can persist across processes run back "
               "to back. Every interval and verdict is conditional on this assumption.")


def t_cdf(t: float, df: int) -> float:
    """Student-t CDF for an integer df >= 1 (closed form; Abramowitz & Stegun 26.7.3/26.7.4)."""
    if not (isinstance(df, int) and df >= 1):
        raise ValueError("df must be a positive integer")
    th = math.atan(t / math.sqrt(df))
    s, c = math.sin(th), math.cos(th)
    if df % 2 == 1:
        acc, term = 0.0, s * c
        if df > 1:
            acc = term
            for k in range(1, (df - 3) // 2 + 1):
                term *= c * c * (2 * k) / (2 * k + 1)
                acc += term
        a = (2 / math.pi) * (th + (acc if df > 1 else 0.0))
    else:
        acc, term = s, s
        for k in range(1, (df - 2) // 2 + 1):
            term *= c * c * (2 * k - 1) / (2 * k)
            acc += term
        a = acc
    return 0.5 + 0.5 * a


def t_quantile(p: float, df: int) -> float:
    """Inverse of t_cdf by bisection (monotone), to ~1e-12."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_quantile(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_interval(x, level: float) -> list:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("a t-interval needs at least two sessions")
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = t_quantile(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n)
    return [m - h, m + h]


def compare(d_sessions, n_family: int, alpha: float, primary: bool) -> dict:
    """Statistics of one comparison from its per-session differences D_s (ms)."""
    d = np.asarray(d_sessions, dtype=float)
    if d.size < 2 or not np.all(np.isfinite(d)):
        raise ValueError("need >= 2 finite session differences")
    level = 1 - alpha / n_family
    out = {"n_sessions": int(d.size), "D_s_ms": [float(v) for v in d], "mean_ms": float(d.mean()),
           "median_ms": float(np.median(d)), "min_ms": float(d.min()), "max_ms": float(d.max()),
           "sd_ms": float(d.std(ddof=1)), "ci95_ms": t_interval(d, 0.95),
           "bonferroni_level": level, "ci_bonferroni_ms": t_interval(d, level),
           "unanimous_negative": bool(np.all(d < 0)), "unanimous_positive": bool(np.all(d > 0))}
    if primary:
        lo, hi = out["ci_bonferroni_ms"]
        if out["unanimous_negative"] and hi < 0:
            out["verdict"] = "A_faster"
        elif out["unanimous_positive"] and lo > 0:
            out["verdict"] = "A_slower"
        else:
            out["verdict"] = "inconclusive"
    else:
        out["verdict"] = None  # secondary: reported, not part of the verdict family
    return out
