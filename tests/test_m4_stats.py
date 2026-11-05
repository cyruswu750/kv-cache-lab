"""Session-level statistics (kvlab/m4_stats.py) and the declared family (kvlab/m4_spec.py)."""

import math

import numpy as np
import pytest

from kvlab import m4_spec as ms
from kvlab import m4_stats as st


def _t_density(x, df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def _numeric_cdf(t, df, n=20000):
    """Independent check: 0.5 + Simpson integral of the density over [0, t]."""
    h = t / n
    s = _t_density(0, df) + _t_density(t, df) + sum((4 if i % 2 else 2) * _t_density(i * h, df) for i in range(1, n))
    return 0.5 + s * h / 3


@pytest.mark.parametrize("df, p, q", [(1, 0.975, 12.7062047), (2, 0.975, 4.3026527), (4, 0.975, 2.7764451),
                                      (9, 0.975, 2.2621572), (9, 0.995, 3.2498355), (30, 0.975, 2.0422725)])
def test_t_quantiles_match_published_values(df, p, q):
    assert st.t_quantile(p, df) == pytest.approx(q, abs=2e-7)
    assert st.t_quantile(1 - p, df) == pytest.approx(-q, abs=2e-7)


@pytest.mark.parametrize("df", [3, 8, 9, 10])
def test_t_cdf_matches_numeric_integration_at_the_bonferroni_level(df):
    p = 1 - 0.05 / 15 / 2
    q = st.t_quantile(p, df)
    assert _numeric_cdf(q, df) == pytest.approx(p, abs=1e-9)
    for x in (0.3, 1.1, 2.7):
        assert st.t_cdf(x, df) == pytest.approx(_numeric_cdf(x, df), abs=1e-9)


def test_hand_calculated_interval_and_verdicts():
    d = [-2.0, -3.0, -2.5, -1.5, -2.0, -3.0, -2.2, -2.8, -1.9, -2.1]
    r = st.compare(d, 15, 0.05, primary=True)
    mean, sd = np.mean(d), np.std(d, ddof=1)                        # -2.3, 0.4876...
    h = st.t_quantile(1 - 0.05 / 15 / 2, 9) * sd / math.sqrt(10)
    assert r["mean_ms"] == pytest.approx(-2.3) and r["sd_ms"] == pytest.approx(sd)
    assert r["ci_bonferroni_ms"] == pytest.approx([mean - h, mean + h])
    assert r["bonferroni_level"] == pytest.approx(1 - 0.05 / 15)
    assert r["unanimous_negative"] and r["verdict"] == "A_faster"
    assert st.compare([-x for x in d], 15, 0.05, True)["verdict"] == "A_slower"


def test_unanimity_without_an_excluding_interval_is_inconclusive_and_vice_versa():
    skewed = [-0.01] * 9 + [-40.0]           # all negative, but the Bonferroni interval spans 0
    r = st.compare(skewed, 15, 0.05, True)
    assert r["unanimous_negative"] and r["ci_bonferroni_ms"][1] > 0 and r["verdict"] == "inconclusive"
    mixed = [-5.0] * 9 + [0.5]               # interval below 0, but one session disagrees
    r = st.compare(mixed, 15, 0.05, True)
    assert r["ci_bonferroni_ms"][1] < 0 and not r["unanimous_negative"] and r["verdict"] == "inconclusive"
    assert st.compare([-1.0, -2.0, -3.0], 15, 0.05, primary=False)["verdict"] is None
    with pytest.raises(ValueError):
        st.compare([1.0], 15, 0.05, True)
    with pytest.raises(ValueError):
        st.compare([1.0, float("nan")], 15, 0.05, True)


def test_declared_family_is_exactly_the_protocol_s_fifteen():
    fam = [tuple(x) for x in ms.M4_EVAL_V1["family"]]
    assert len(fam) == 15 and len(set(fam)) == 15
    want = {(t, b, o) for t in ("T1", "T2", "T3") for b in ("C2_base", "C2_batch") for o in ("C0", "C1PA_HALF")}
    want |= {("T0", "C2_base", "C0"), ("T0", "C2_batch", "C0"), ("T0", "C1PA_HALF", "C0")}
    assert set(fam) == want
    assert not set(map(tuple, ms.M4_EVAL_V1["secondary"])) & want
    s = ms.M4_EVAL_V1
    assert (len(s["sessions"]), s["budget_s"], s["alpha"], s["steps"], s["chunk_size"]) == (10, 2700, 0.05, 1, 256)
    assert s["configs"] == ["C0", "C1PA_W", "C1PA_HALF", "C2_base", "C2_batch"]


def test_verdicts_use_the_bonferroni_level_not_95_percent():
    """Unanimous, and the 95% interval excludes 0 (t = mean / se is about 3.3, above t_{0.975,9}
    = 2.262), but the level-(1 - 0.05/15) interval does not (t_{1-0.05/30,9} is about 4.0). So the
    verdict is inconclusive."""
    d = [-0.5, -2.0, -1.0, -2.0, -0.2, -5.0, -1.0, -2.0, -0.3, -1.0]
    r = st.compare(d, 15, 0.05, True)
    assert r["unanimous_negative"] and r["ci95_ms"][1] < 0 < r["ci_bonferroni_ms"][1]
    assert r["verdict"] == "inconclusive"


def test_steady_state_means_and_ratios_use_the_same_sliced_series():
    """Review P3: A = 100, 100, 1 x 6 and B = 2 x 8 in two sessions, steady fraction 1/4 (2 excluded):
    D = -1, steady A mean 1, B mean 2, ratio 0.5 (not the full-trace 25.75 and 12.875)."""
    from kvlab import m4_analysis as ma
    A, B = [100.0, 100.0] + [1.0] * 6, [2.0] * 8
    lat = {("T1", "a"): {"s1": A, "s2": A}, ("T1", "b"): {"s1": B, "s2": B}}
    r = ma._cmp(lat, ["s1", "s2"], "T1", "a", "b", 15, 0.05, False, 0.25)
    assert r["excluded_leading_requests"] == 2 and r["D_s_ms"] == [-1.0, -1.0]
    assert (r["mean_trace_mean_A_ms"], r["mean_trace_mean_B_ms"], r["ratio_of_trace_means_A_over_B"]) == (1.0, 2.0, 0.5)
    full = ma._cmp(lat, ["s1", "s2"], "T1", "a", "b", 15, 0.05, False)
    assert full["mean_trace_mean_A_ms"] == 25.75 and full["excluded_leading_requests"] == 0
