"""M2 bookkeeping and cost-model calculations (kvlab/perf.py), CPU only."""

import itertools

import numpy as np
import pytest

from kvlab import perf


def sample(path, L, S, e2e_ms=10.0, rep=0, instrumented=False, **over):
    want = perf.expected(path, L, S)
    s = {"path": path, "L": L, "S": S, "rep": rep, "warmup": False, "instrumented": instrumented,
         "e2e_ns": int(e2e_ms * 1e6), **want}
    if path == "disk":
        s["resident_fraction_before"] = 1.0
    if path == "disk_evicted":
        s["resident_fraction_before"] = 0.0
    if instrumented:
        s["phases_ns"] = {p: int(e2e_ms * 1e6 / 5) for p in perf.PHASES}
    s.update(over)
    return s


# ---- units, grid, workload split

def test_units_are_nanoseconds_in_and_milliseconds_out():
    assert perf.ns_to_ms(1_500_000) == 1.5
    s = perf.summarize([1.0] * 5 + [3.0] * 6, seed=0)
    assert s["status"] == "ok" and s["median_ms"] == 3.0 and s["n"] == 11 and s["min_ms"] == 1.0
    assert s["mean_ms"] == pytest.approx(23 / 11) and s["slow_share_gt_1p3x_min"] == pytest.approx(6 / 11)


def test_splits_are_disjoint_cover_the_grid_and_are_valid_cases():
    grid = set(itertools.product(perf.L_GRID, perf.S_GRID))
    cal, ev = set(perf.CALIBRATION), set(perf.EVALUATION)
    assert not cal & ev and cal | ev == grid and len(ev) == 9 and len(cal) == 6
    assert {S for _, S in ev} - {S for _, S in cal} == {64}  # S = 64 is only ever held out
    for L, S in grid | set(perf.SMOKE):
        perf.check_case(L, S)
    for bad in [(0, 1), (300, 1), (256, 0), (256, 257)]:
        with pytest.raises(ValueError):
            perf.check_case(*bad)


def test_expected_tier_restore_length_and_admissions():
    assert perf.expected("full", 1024, 1) == {"restore_len": 0, "sources": [], "stored_chunks": []}
    assert perf.expected("host", 1024, 256) == {"restore_len": 1024, "sources": ["host"] * 4, "stored_chunks": [4]}
    assert perf.expected("disk_evicted", 512, 64)["sources"] == ["disk", "disk"]
    assert perf.expected("full", 512, 256)["stored_chunks"] == [2]  # the same admission on every path
    assert perf.paths_for(1) == ("full", "host", "disk", "disk_evicted") and perf.paths_for(64) == perf.PATHS


# ---- sample validation and missing samples

@pytest.mark.parametrize("edit, reason", [
    ({"sources": ["host", "host"]}, "sources"),            # a disk sample served from host
    ({"restore_len": 256}, "restore_len"),                 # restored the wrong prefix
    ({"stored_chunks": []}, "stored_chunks"),              # admission skipped
    ({"e2e_ns": 0}, "e2e_ns"),
    ({"e2e_ns": 1.5}, "e2e_ns"),                           # not integer nanoseconds
    ({"resident_fraction_before": 0.5}, "page cache not warm"),
])
def test_wrong_tier_length_or_timing_is_excluded_with_a_reason(edit, reason):
    s = sample("disk", 512, 256, **edit)
    valid, excluded = perf.partition([s, sample("disk", 512, 256, rep=1)])
    assert len(valid) == 1 and len(excluded) == 1
    assert any(reason in p for p in excluded[0]["excluded_because"])


def test_evicted_sample_must_be_cold_and_instrumented_sample_needs_all_phases():
    _, ex = perf.partition([sample("disk_evicted", 256, 1, resident_fraction_before=0.2)])
    assert "page cache not evicted" in ex[0]["excluded_because"][0]
    bad = sample("host", 256, 1, instrumented=True)
    del bad["phases_ns"]["store"]
    _, ex = perf.partition([bad])
    assert "phases_ns" in ex[0]["excluded_because"][0]


def test_warmups_are_neither_used_nor_counted_as_exclusions():
    valid, excluded = perf.partition([sample("host", 256, 1, warmup=True, sources=[]), sample("host", 256, 1)])
    assert len(valid) == 1 and excluded == []


def test_missing_samples_give_no_statistics_and_no_winner():
    assert perf.summarize([1.0] * (perf.MIN_SAMPLES - 1))["status"] == "insufficient_samples"
    a = {r: 1.0 for r in range(20)}
    b = {r: 2.0 for r in range(5, 20)}  # only 15 reps in common
    d = perf.paired_difference(a, b)
    assert d["n"] == 15 and d["median_ms"] == -1.0
    short = perf.paired_difference(a, {r: 2.0 for r in range(5)})
    assert short["status"] == "insufficient_samples" and perf.measured_winner(short) == "unknown"
    med = perf.case_medians([sample("host", 256, 1, rep=r) for r in range(perf.MIN_SAMPLES - 1)])
    assert "e2e" not in med[("host", 256, 1)]
    with pytest.raises(ValueError, match="lacks medians"):
        perf.fit_cost_model(med, calibration_cases=[(256, 1)])


def test_paired_winner_requires_the_ci_to_exclude_zero():
    rng = np.random.default_rng(0)
    full = {r: 50 + rng.normal() for r in range(20)}
    assert perf.measured_winner(perf.paired_difference({r: v - 10 for r, v in full.items()}, full)) == "restore"
    assert perf.measured_winner(perf.paired_difference({r: v + 10 for r, v in full.items()}, full)) == "full"
    noisy = {r: full[r] + rng.normal(scale=3) for r in range(20)}
    assert perf.measured_winner(perf.paired_difference(noisy, full)) == "tie"


# ---- NNLS and the cost model

def test_nnls_small_matches_brute_force_and_respects_signs():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 2, size=(12, 3))
    y = X @ np.array([2.0, 0.5, 1.0])
    assert np.allclose(perf.nnls_small(X, y), [2.0, 0.5, 1.0])
    y2 = X @ np.array([2.0, -1.0, 0.0])  # unconstrained optimum has a negative term
    coef = perf.nnls_small(X, y2)
    assert (coef >= 0).all()
    grid = np.linspace(0, 3, 61)  # coarse brute force: no non-negative point does better
    best = min(np.sum((X @ np.array(c) - y2) ** 2) for c in itertools.product(grid, repeat=3))
    assert np.sum((X @ coef - y2) ** 2) <= best + 1e-9


TRUE = {"ingest": [0.3, 0.1], "restore_host": [0.5, 5.0], "restore_disk": [0.6, 7.0],
        "prefill_suffix": [18.0, 20.0, 9.0], "prefill_full": [17.0, 40.0, 3.0], "store": [0.05, 1.2]}


def synthetic_medians(cases, offset=-0.5):
    med = {}
    for L, S in cases:
        n_new = len(perf.expected("full", L, S)["stored_chunks"])
        f = {k: float(np.dot(v, perf.COMPONENTS[k][1](L, S, n_new))) for k, v in TRUE.items()}
        for path in perf.PATHS:
            restore = f[f"restore_{path}"] if path != "full" else 0.001
            prefill = f["prefill_full"] if path == "full" else f["prefill_suffix"]
            ph = {"ingest": f["ingest"], "restore": restore, "prefill": prefill, "store": f["store"], "select": 0.2}
            total = sum(ph.values())
            med[(path, L, S)] = {"phases": ph, "instrumented_total": total, "e2e": total + offset}
    return med


def test_cost_model_recovers_known_components_and_predicts_held_out_cases():
    model = perf.fit_cost_model(synthetic_medians(perf.CALIBRATION))
    for name, coef in TRUE.items():
        assert np.allclose(model["components"][name]["coef_ms"], coef, rtol=1e-6, atol=1e-9), name
    truth = synthetic_medians(perf.EVALUATION)
    for L, S in perf.EVALUATION:
        for path in perf.PATHS:
            p = perf.predict(model, path, L, S)
            assert p["predicted_ms"] == pytest.approx(truth[(path, L, S)]["e2e"], rel=1e-6)
            assert set(p["components_ms"]) == set(perf.PATH_COMPONENTS[path]) | {"instrumentation_offset"}


def test_fit_refuses_held_out_cases_and_frozen_artifacts_are_tamper_evident():
    med = synthetic_medians(perf.CALIBRATION + ((512, 64),))
    with pytest.raises(ValueError, match="not a calibration case"):
        perf.fit_cost_model(med)
    model = perf.fit_cost_model(synthetic_medians(perf.CALIBRATION))
    preds = perf.make_predictions(model, perf.EVALUATION, extra={"created_unix": 1.0})
    assert preds["sha256"] == perf.predictions_digest(preds) and len(preds["predictions"]) == 27
    preds["predictions"][0]["predicted_ms"] += 1
    assert preds["sha256"] != perf.predictions_digest(preds)
    model["components"]["store"]["coef_ms"][1] = 0.0
    with pytest.raises(ValueError, match="digest mismatch"):
        perf.predict(model, "full", 512, 64)


# ---- scoring

def test_decisions_score_winners_penalties_and_ties():
    pred = {("host", 512, 1): 20.0, ("full", 512, 1): 30.0, ("disk", 512, 1): 35.0}
    med = {("host", 512, 1): 25.0, ("full", 512, 1): 28.0, ("disk", 512, 1): 26.0}
    diffs = {("host", 512, 1): {"status": "ok", "median_ms": -3.0, "ci95_ms": [-4.0, -2.0]},
             ("disk", 512, 1): {"status": "ok", "median_ms": -2.0, "ci95_ms": [-3.0, -1.0]}}
    rows = {r["tier"]: r for r in perf.decisions(pred, med, diffs)}
    assert rows["host"]["correct"] is True and rows["host"]["penalty_ms"] == 0.0
    assert rows["disk"]["predicted_winner"] == "full" and rows["disk"]["correct"] is False
    assert rows["disk"]["penalty_ms"] == pytest.approx(2.0) and rows["disk"]["penalty_rel"] == pytest.approx(2 / 26)
    diffs["disk"] = diffs.pop(("disk", 512, 1)) | {"ci95_ms": [-3.0, 0.5]}
    tie = perf.decisions(pred, med, {("disk", 512, 1): diffs["disk"]})[0]
    assert tie["measured_winner"] == "tie" and tie["correct"] is None and tie["penalty_ms"] is None


def test_crossings_only_between_reliable_opposite_winners():
    assert perf.sign_changes({256: 1.0, 512: 0.5, 1024: -2.0, 2048: -3.0}) == [
        {"L_low": 512, "L_high": 1024, "low_side": "full", "high_side": "restore"}]
    assert perf.sign_changes({256: -1.0, 512: -2.0}) == []
    rows = [{"tier": "disk", "S": 1, "L": L, "measured_winner": w}
            for L, w in ((256, "full"), (512, "tie"), (1024, "restore"), (2048, "restore"))]
    rows += [{"tier": "host", "S": 1, "L": L, "measured_winner": "restore"} for L in (256, 512)]
    x = perf.measured_crossings(rows)
    assert all(c.get("inconclusive") for c in x[("disk", 1)]["changes"])  # the tie breaks the comparison
    assert x[("host", 1)]["changes"] == []
