"""M5b analysis definitions on hand-calculated, synthetic multi-candidate examples (nothing measured).

Two evaluation cases from the tiny spec, two sessions (s01 = partition A, s02 = partition B), two measured
repetitions each, latencies written out by hand below. Expected values are computed by hand in the comments.
"""

import copy

import numpy as np
import pytest

import m5b_toy as t
from kvlab import m5_analysis as ma
from kvlab import m5_spec as sp
from kvlab.m3_stats import stratified_indices

# per case: label -> (s01 value, s02 value), identical in both repetitions of a session
TABLE = {
    "te-02": {"cand:0": (10.0, 10.0), "cand:4": (7.0, 9.0), "cand:8": (9.0, 8.0),
              "policy:cost_aware": (7.5, 9.5), "policy:always_restore": (9.2, 8.2), "policy:recompute": (10.1, 10.1)},
    "te-01": {"cand:0": (5.0, 5.0), "cand:4": (6.0, 6.0),
              "policy:cost_aware": (6.1, 6.1), "policy:always_restore": (6.05, 6.05), "policy:recompute": (5.02, 5.02)},
}
DECISIONS = {"te-01": 4, "te-02": 4}
PREDICTED = {"te-01": {"0": 6.5, "4": 6.0}, "te-02": {"0": 10.5, "4": 7.5, "8": 9.0}}


def mini_spec(reps=2, block_len=1, min_blocks=2):
    cal, _ = t.tiny_specs()
    cases = [c for c in sp.grid_cases(t.TINY_EVAL_GRID, "evaluation", t.C, t.TINY_VOCAB) if c["case_id"] in TABLE]
    an = {**t.TINY_ANALYSIS, "block_len": block_len, "min_blocks_per_session": min_blocks, "n_boot": 400}
    return sp.build_spec("HAND", "evaluation", cases, {"sessions": {"s01": 1, "s02": 2}, "warmup_reps": 1, "reps": reps},
                         C_=t.C, backend="batched", group=2, model=t.TINY_MODEL, analysis=an, deadline_s=60,
                         calibration=cal)


def sessions_from(spec, table=TABLE, warmup_ms=5000.0, per_rep=None):
    out = {}
    for si, sid in enumerate(spec["schedule"]["sessions"]):
        samples = []
        for cid, phase, rep, pos, label in sp.schedule(spec, sid):
            v = table[cid][label][si] if phase == "measured" else warmup_ms
            if per_rep is not None and phase == "measured":
                v += per_rep(sid, rep)
            samples.append({"case_id": cid, "phase": phase, "rep": rep, "pos": pos, "run": label,
                            "latency_ns": int(round(v * 1e6))})
        out[sid] = {"samples": samples}
    return out


def predictions():
    cases = {cid: {"decision": {"selected_len": L, "reason": "min_predicted_objective", "objective_ms": []},
                   "predictions": {k: {"predicted_ms": v} for k, v in PREDICTED[cid].items()}} for cid, L in DECISIONS.items()}
    return {"sha256": "p" * 64, "artifact_digest": "a" * 64, "cases": cases}


VALID = {"exit": 0, "valid": True}


def test_best_candidate_ties_and_naive_regret():
    assert ma.best_candidate({0: 10.0, 4: 8.0, 8: 8.0}) == 8          # exact tie -> larger L
    assert ma.best_candidate({0: 5.0, 4: 6.0}) == 0
    m = {0: 10.0, 4: 7.0, 8: 9.0}
    assert ma.naive_regret(m, 8) == 2.0 and ma.naive_regret(m, 4) == 0.0


def test_cross_evaluated_difference_is_signed():
    a, b = {0: 10.0, 4: 7.0, 8: 9.0}, {0: 10.0, 4: 9.0, 8: 8.0}
    # chosen 8: reference picked on A is 4 -> on B: 8 - 9 = -1; picked on B is 8 -> on A: 9 - 9 = 0 => -0.5
    assert ma.cross_evaluated(a, b, 8, 0.5, 0.5) == pytest.approx(-0.5)
    # chosen 4: on B 9 - 9 = 0; on A 7 - 9 = -2 => -1.0 (negative: the policy's candidate won on held-out data)
    assert ma.cross_evaluated(a, b, 4, 0.5, 0.5) == pytest.approx(-1.0)
    assert ma.cross_evaluated(a, b, 0, 0.5, 0.5) == pytest.approx(1.0)
    assert ma.cross_evaluated(a, b, 4, 1.0, 0.0) == pytest.approx(0.0)


def test_hand_calculated_analysis():
    spec = mini_spec()
    res = ma.analyze(spec, sessions_from(spec), predictions(), VALID)
    c2, c1 = res["cases"]["te-02"], res["cases"]["te-01"]
    # te-02 means: 0 -> 10, 4 -> (7+9)/2 = 8, 8 -> (9+8)/2 = 8.5; best 4
    assert {k: v["mean_ms"] for k, v in c2["candidates"].items()} == pytest.approx({"0": 10.0, "4": 8.0, "8": 8.5})
    assert c2["best_measured"] == 4
    p = c2["policies"]
    assert (p["cost_aware"]["naive_regret_ms"], p["always_restore"]["naive_regret_ms"],
            p["recompute"]["naive_regret_ms"]) == pytest.approx((0.0, 0.5, 2.0))
    assert (p["cost_aware"]["cross_evaluated_ms"], p["always_restore"]["cross_evaluated_ms"],
            p["recompute"]["cross_evaluated_ms"]) == pytest.approx((-1.0, -0.5, 1.0))
    # candidate quality vs delivered latency: CostAware's candidate (L = 4) measures 8.0 through FixedChunks,
    # while the CostAware policy runs themselves deliver (7.5 + 9.5) / 2 = 8.5 (its own planning/prediction cost)
    assert c2["candidates"]["4"]["mean_ms"] == pytest.approx(8.0) and p["cost_aware"]["delivered_mean_ms"] == pytest.approx(8.5)
    assert c2["delivered_differences_ms"] == pytest.approx(
        {"cost_aware_minus_always_restore": -0.2, "cost_aware_minus_recompute": -1.6, "always_restore_minus_recompute": -1.4})
    # te-01 means 0 -> 5, 4 -> 6; best 0; CostAware chose 4: regret 1, cross 1
    assert c1["best_measured"] == 0 and c1["policies"]["cost_aware"]["agrees_with_best"] is False
    assert c1["policies"]["cost_aware"]["cross_evaluated_ms"] == pytest.approx(1.0)
    agg = res["aggregate"]["policies"]
    assert agg["cost_aware"]["agreement_fraction"] == 0.5                                  # equal case weights
    assert agg["cost_aware"]["naive_mean_regret_ms"] == pytest.approx((0.0 + 1.0) / 2)
    assert agg["always_restore"]["naive_mean_regret_ms"] == pytest.approx((0.5 + 1.0) / 2)
    assert agg["recompute"]["naive_mean_regret_ms"] == pytest.approx((2.0 + 0.0) / 2)
    assert agg["cost_aware"]["cross_evaluated_mean_ms"] == pytest.approx((-1.0 + 1.0) / 2)
    d = res["aggregate"]["delivered_differences_ms"]
    assert d["cost_aware_minus_always_restore"]["mean_ms"] == pytest.approx((-0.2 + 0.05) / 2)
    assert d["cost_aware_minus_recompute"]["mean_ms"] == pytest.approx((-1.6 + 1.08) / 2)
    # prediction errors: te-02 +0.5, -0.5, +0.5; te-01 +1.5, 0.0 (rows weighted equally)
    pe = res["aggregate"]["prediction_error"]
    assert pe["mean_signed_error_ms"] == pytest.approx(2.0 / 5) and pe["mean_abs_error_ms"] == pytest.approx(3.0 / 5)
    assert pe["mean_abs_relative_error"] == pytest.approx((0.5 / 10 + 0.5 / 8 + 0.5 / 8.5 + 1.5 / 5 + 0) / 5)
    assert c2["candidates"]["8"]["error_ms"] == pytest.approx(0.5)
    ma.render_md(res)


def test_warmup_timings_never_affect_the_statistics():
    spec = mini_spec()
    a = ma.analyze(spec, sessions_from(spec, warmup_ms=5000.0), predictions(), VALID)
    b = ma.analyze(spec, sessions_from(spec, warmup_ms=0.001), predictions(), VALID)
    assert a == b


def test_resampling_keeps_whole_repetitions_and_sessions():
    """Repetition r of session s is shifted by a large common amount (all labels alike). Paired policy
    differences are then constant, so their interval must have zero width; an unpaired resampling could not
    produce that. Each case's sessions stay separate strata."""
    spec = mini_spec(reps=6, block_len=2, min_blocks=2)
    shift = lambda sid, rep: 100.0 * rep + (1000.0 if sid == "s02" else 0.0)  # noqa: E731
    res = ma.analyze(spec, sessions_from(spec, per_rep=shift), predictions(), VALID)
    for key, e in res["aggregate"]["delivered_differences_ms"].items():
        assert e["interval_ms"][0] == pytest.approx(e["interval_ms"][1]) == pytest.approx(e["mean_ms"]), key
    rng = np.random.default_rng(0)
    idx = stratified_indices([6, 6], 2, 50, rng)
    assert (idx[:, :6] < 6).all() and (idx[:, 6:] >= 6).all()   # never across the session boundary


def test_ambiguity_uses_simultaneous_intervals_and_flags_inadequate_data():
    spec = mini_spec(reps=6, block_len=1, min_blocks=2)
    noise = {("s01", r): x for r, x in enumerate([0.0, 3.0, -2.0, 1.0, -1.5, 2.5])}
    noise.update({("s02", r): x for r, x in enumerate([1.0, -3.0, 2.0, -1.0, 0.5, -2.5])})
    table = copy.deepcopy(TABLE)
    table["te-01"]["cand:0"] = (5.0, 5.0)
    table["te-01"]["cand:4"] = (5.2, 5.2)       # 0.2 ms apart, under per-label noise of several ms: ambiguous
    sess = sessions_from(spec, table)
    for sid, s in sess.items():                 # independent noise per label breaks the shared shift
        for x in s["samples"]:
            if x["phase"] == "measured" and x["case_id"] == "te-01":
                k = 1.0 if x["run"] == "cand:0" else -1.0
                x["latency_ns"] += int(k * noise[(sid, x["rep"])] * 1e6)
    res = ma.analyze(spec, sess, predictions(), VALID)
    assert res["cases"]["te-01"]["ambiguous"] is True and sorted(res["cases"]["te-01"]["possibly_best"]) == [0, 4]
    assert res["cases"]["te-02"]["ambiguous"] is False and res["cases"]["te-02"]["possibly_best"] == [4]
    small = mini_spec(reps=2, block_len=3, min_blocks=2)          # 2 repetitions < 2 blocks of 3
    r2 = ma.analyze(small, sessions_from(small), predictions(), VALID)
    assert r2["cases"]["te-02"]["ambiguity_status"] == "inadequate"
    assert r2["aggregate"]["policies"]["cost_aware"]["interval_status"] == "inadequate"
    assert "interval_ms" not in r2["aggregate"]["delivered_differences_ms"]["cost_aware_minus_recompute"]


def test_stratified_resampling_never_mixes_sessions():
    """Within each session the repetitions are identical, but the sessions differ by candidate (TABLE). A
    resampling that stays within sessions reproduces the point estimates exactly in every replicate, so every
    interval, the cross-evaluated ones included, has zero width; mixing sessions would widen them."""
    spec = mini_spec(reps=4, block_len=2, min_blocks=2)
    res = ma.analyze(spec, sessions_from(spec), predictions(), VALID)
    for P, e in res["aggregate"]["policies"].items():
        for key, point in (("naive_mean_regret_interval_ms", "naive_mean_regret_ms"),
                           ("cross_evaluated_interval_ms", "cross_evaluated_mean_ms")):
            assert e[key][0] == pytest.approx(e[key][1]) == pytest.approx(e[point]), (P, key)


def test_invalid_evidence_gets_no_analysis():
    spec = mini_spec()
    for bad in ({"exit": 3, "valid": False}, {"exit": 4, "valid": False}, {"exit": 0, "valid": False}):
        with pytest.raises(ValueError, match="invalid evidence"):
            ma.analyze(spec, sessions_from(spec), predictions(), bad)


def test_incomplete_measured_repetitions_are_refused():
    spec = mini_spec()
    sess = sessions_from(spec)
    sess["s01"]["samples"] = [x for x in sess["s01"]["samples"] if not (x["phase"] == "measured" and x["rep"] == 1)]
    with pytest.raises(ValueError, match="incomplete"):
        ma.analyze(spec, sess, predictions(), VALID)
    sess = sessions_from(spec)
    first = next(x for x in sess["s01"]["samples"] if x["phase"] == "measured")
    sess["s01"]["samples"].append(dict(first))
    with pytest.raises(ValueError, match="duplicate measured sample"):
        ma.analyze(spec, sess, predictions(), VALID)


def test_review_too_few_sessions_per_half_gives_no_cross_interval():
    spec = mini_spec(reps=6, block_len=2, min_blocks=2)
    spec["analysis"]["min_sessions_per_half"] = 2          # one session per half here
    res = ma.analyze(spec, sessions_from(spec), predictions(), VALID)
    for P, e in res["aggregate"]["policies"].items():
        assert "cross_evaluated_interval_ms" not in e and e["cross_interval_status"] == "inadequate", P
        assert "naive_mean_regret_interval_ms" in e
    assert all(c["cross_adequate"] is False for c in res["cases"].values())
