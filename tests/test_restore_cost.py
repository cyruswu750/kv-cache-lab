"""M5a: restore cost model fitting, artifact validation, compatibility, domain and CostAware choices.

CPU only, synthetic data only (tests/m5_toy.py): no coefficient here is measured, and no fitted
value is a performance result. Hand-computed choices use the model equation written out
independently in m5_toy.truth_ms.
"""

import copy
import json
import math

import numpy as np
import pytest

from kvlab import perf
from kvlab import restore_cost as rc
from kvlab.restore_policy import CostAware
from m5_toy import SPECS, artifact, coefficients, compat, domain, plan, redigest, synthetic_cases, truth_ms

TRUTH = coefficients(full=(13.0, 40.0, 8.0), partial=(17.0, 0.8, 2.5, 30.0, 9.0), admit=4.5)
CAL_IDS = [s[0] for s in SPECS]
EVAL_IDS = ["e01", "e02"]


def fit(cases=None, **over):
    kw = dict(calibration_case_ids=CAL_IDS, evaluation_case_ids=EVAL_IDS, compatibility=compat(),
              weighting="uniform", provenance={"source": "synthetic test data"})
    kw.update(over)
    return rc.fit_restore_cost(synthetic_cases(TRUTH) if cases is None else cases, **kw)


def test_backend_names_match_the_store():
    from kvlab.tiered_cache import RESTORE_BACKENDS
    assert rc.BACKENDS == RESTORE_BACKENDS


# ---- fitting -------------------------------------------------------------------------------

@pytest.mark.parametrize("weighting", rc.WEIGHTINGS)
def test_fit_recovers_known_coefficients_and_records_domain_and_provenance(weighting):
    art = fit(weighting=weighting)
    for g, names in rc.FEATURES.items():
        for f in names:
            assert art["coefficients_ms"][g][f] == pytest.approx(TRUTH[g][f], rel=1e-7, abs=1e-7), (g, f)
    assert art["fit"]["rank"] == 9 and art["fit"]["weighting"] == weighting
    assert art["fit"]["max_abs_residual_ms"] < 1e-6 and art["fit"]["n_cases"] == len(SPECS)
    assert art["fit"]["n_rows"] == sum(1 + len(s[2]) for s in SPECS)
    assert art["domain"] == {"prompt_len": [300, 2100], "suffix_len": [44, 1844], "restore_chunks_max": 5,
                             "host_chunks_max": 2, "disk_chunks_max": 5, "admitted_chunks": [0, 3]}
    assert art["calibration"]["case_ids"] == sorted(CAL_IDS) and art["calibration"]["evaluation_case_ids"] == EVAL_IDS
    assert art["provenance"] == {"source": "synthetic test data"} and art["digest"] == rc.artifact_digest(art)
    m = rc.RestoreCostModel(art)
    assert m.predict_ms(1100, 512, 1, 1, 1) == pytest.approx(truth_ms(TRUTH, 1100, 512, 1, 1, 1), rel=1e-9)


def test_case_order_does_not_change_the_artifact():
    cases = synthetic_cases(TRUTH)
    assert fit(cases[::-1]) == fit(cases)


def test_nnls_matches_perf_nnls_small_and_satisfies_kkt_with_nine_columns():
    rng = np.random.default_rng(5)
    for _ in range(20):
        X = rng.normal(size=(30, 6))
        y = rng.normal(size=30)
        assert np.allclose(rc._nnls_exact(X, y), perf.nnls_small(X, y), atol=1e-10)
    X, y = rng.normal(size=(40, 9)), rng.normal(size=40)
    b = rc._nnls_exact(X, y)
    grad = X.T @ (X @ b - y)
    assert np.all(b >= 0) and np.all(grad[b > 0] == pytest.approx(0, abs=1e-8)) and np.all(grad[b == 0] >= -1e-8)


def test_noisy_fit_keeps_coefficients_non_negative():
    rng = np.random.default_rng(1)
    zeroed = coefficients(full=(13.0, 40.0, 0.0), partial=(17.0, 0.0, 2.5, 30.0, 9.0), admit=4.5)
    cases = synthetic_cases(zeroed)
    for c in cases:
        for cand in c["candidates"]:
            cand["mean_ms"] *= float(1 + 0.02 * rng.normal())
    art = fit(cases)
    assert all(v >= 0 for g in art["coefficients_ms"].values() for v in g.values())
    assert art["fit"]["max_abs_residual_ms"] > 0


# ---- fit isolation (no leakage, nothing filtered) -----------------------------------------

def test_fit_refuses_evaluation_unexpected_duplicate_or_missing_cases():
    cases = synthetic_cases(TRUTH)
    leaked = copy.deepcopy(cases)
    leaked[0]["case_id"] = "e01"
    with pytest.raises(rc.FitError, match="declared evaluation case"):
        fit(leaked)
    stray = copy.deepcopy(cases) + [{**copy.deepcopy(cases[0]), "case_id": "x99"}]
    with pytest.raises(rc.FitError, match="not a declared calibration case"):
        fit(stray)
    with pytest.raises(rc.FitError, match="duplicate case"):
        fit(cases + [copy.deepcopy(cases[0])])
    with pytest.raises(rc.FitError, match="without data"):
        fit(cases[1:])
    with pytest.raises(rc.FitError, match="both calibration and evaluation"):
        fit(evaluation_case_ids=["c01"])
    with pytest.raises(rc.FitError, match="duplicates"):
        fit(calibration_case_ids=CAL_IDS + ["c01"])
    with pytest.raises(rc.FitError, match="collection"):
        fit(evaluation_case_ids="e01")


@pytest.mark.parametrize("edit, match", [
    (lambda c: c["candidates"][1].update(mean_ms=float("nan")), "finite"),
    (lambda c: c["candidates"][1].update(mean_ms=float("inf")), "finite"),
    (lambda c: c["candidates"][1].update(mean_ms=0.0), "> 0"),
    (lambda c: c["candidates"][1].update(mean_ms=True), "finite number"),
    (lambda c: c["candidates"][1].update(n_samples=0), "n_samples"),
    (lambda c: c.update(prompt_len=1100.0), "prompt_len"),
    (lambda c: c.update(prompt_len=True), "prompt_len"),
    (lambda c: c["candidates"][1].update(restore_len=255), "every legal length"),
    (lambda c: c["candidates"].pop(), "every legal length"),
    (lambda c: c["candidates"].append(dict(c["candidates"][0])), "every legal length"),
    (lambda c: c.update(tiers=["host", "gpu", "host"]), "tiers"),
    (lambda c: c.update(tiers=["host"] * 5), "first-token rule"),
    (lambda c: c.update(admitted_chunks=2), "admitted_chunks"),
    (lambda c: c.update(extra=1), "unexpected"),
    (lambda c: c["candidates"][0].update(std_ms=1.0), "unexpected"),
])
def test_fit_rejects_malformed_case_inputs(edit, match):
    cases = synthetic_cases(TRUTH)
    edit(cases[2])  # c03: N=1100, tiers host/disk/host, m=1
    with pytest.raises(rc.FitError, match=match):
        fit(cases)


def test_rank_is_checked_not_row_counts():
    many_rows_one_n = [(f"s{i}", 2100, ["host", "disk", "host", "disk"][: 1 + i % 4], i % 3) for i in range(40)]
    cases = synthetic_cases(TRUTH, many_rows_one_n)
    assert sum(len(c["candidates"]) for c in cases) > 9 * 10
    with pytest.raises(rc.FitError, match="rank"):
        fit(cases, calibration_case_ids=[s[0] for s in many_rows_one_n])


@pytest.mark.parametrize("specs, match", [
    ([(cid, N, ["host"] * len(t), m) for cid, N, t, m in SPECS], "never exercised"),   # no disk chunk anywhere
    ([(cid, N, t, 1) for cid, N, t, m in SPECS if N // 256 - len(t) >= 1], "rank"),   # constant admission
    ([(cid, N, t, 0) for cid, N, t, m in SPECS], "never exercised"),                  # no admission at all
])
def test_unidentifiable_calibration_designs_are_rejected(specs, match):
    cases = synthetic_cases(TRUTH, specs)
    with pytest.raises(rc.FitError, match=match):
        fit(cases, calibration_case_ids=[s[0] for s in specs])


def test_fit_rejects_bad_configuration():
    with pytest.raises(rc.FitError, match="weighting"):
        fit(weighting="median")
    with pytest.raises(rc.FitError, match="restore_backend"):
        fit(compatibility={**compat(), "restore_backend": "fast"})
    with pytest.raises(rc.FitError, match="null for the baseline"):
        fit(compatibility={**compat(), "restore_staging_chunks": 4})
    with pytest.raises(rc.FitError, match="restore_staging_chunks"):
        fit(compatibility=compat(backend="batched", group=0))
    with pytest.raises(rc.FitError, match="hex"):
        fit(compatibility={**compat(), "namespace_digest": "AB" * 32})
    with pytest.raises(rc.FitError, match="provenance"):
        fit(provenance={"x": float("nan")})


# ---- artifacts -----------------------------------------------------------------------------

def test_artifact_round_trip_and_tamper_evidence(tmp_path):
    art = fit()
    path = tmp_path / "model.json"
    rc.save_artifact(art, path)
    with pytest.raises(FileExistsError):
        rc.save_artifact(art, path)
    m = rc.load_model(path)
    assert m.digest == art["digest"] and m.artifact == art
    bad = json.loads(path.read_text())
    bad["coefficients_ms"]["partial"]["disk_chunks"] *= 2
    with pytest.raises(rc.ArtifactError, match="digest mismatch"):
        rc.RestoreCostModel(bad)
    nan_file = tmp_path / "nan.json"
    nan_file.write_text(path.read_text().replace(repr(art["fit"]["condition_number"]), "NaN", 1))
    with pytest.raises(rc.ArtifactError, match="non-finite JSON constant"):
        rc.load_model(nan_file)


@pytest.mark.parametrize("edit, match", [
    (lambda a: a.update(schema="kvlab-restore-cost-v2"), "unsupported schema"),
    (lambda a: a.update(extra=1), "unexpected"),
    (lambda a: a.pop("domain"), "missing"),
    (lambda a: a["units"].update(latency="s"), "units"),
    (lambda a: a.update(objective="all terms"), "objective"),
    (lambda a: a["features"]["partial"].reverse(), "features"),
    (lambda a: a["coefficients_ms"]["partial"].update(disk_chunks=-1.0), ">= 0"),
    (lambda a: a["coefficients_ms"]["partial"].update(disk_chunks="1"), "finite number"),
    (lambda a: a["coefficients_ms"]["full"].pop("n_kt"), "missing"),
    (lambda a: a["compatibility"].update(restore_backend="batched"), "restore_staging_chunks"),
    (lambda a: a["fit"].update(rank=8), "fit.rank"),
    (lambda a: a["fit"].update(method="ols"), "fit method"),
    (lambda a: a["domain"].update(prompt_len=[900, 300]), "min 900 > max 300"),
    (lambda a: a["domain"].update(host_chunks_max=99), "exceeds"),
    (lambda a: a["calibration"].update(evaluation_case_ids=["synthetic-1"]), "disjoint"),
    (lambda a: a["calibration"].update(case_ids=["b", "a"]), "sorted"),
    (lambda a: a.update(provenance=["x"]), "provenance"),
])
def test_malformed_artifacts_are_rejected_even_when_redigested(edit, match):
    art = artifact()
    edit(art)
    with pytest.raises(rc.ArtifactError, match=match):
        rc.RestoreCostModel(redigest(art))


def test_non_json_values_are_rejected():
    art = artifact()
    art["coefficients_ms"]["full"]["const"] = float("nan")
    with pytest.raises(rc.ArtifactError, match="canonical JSON"):
        rc.RestoreCostModel(art)


def test_model_is_validated_once_not_per_candidate(monkeypatch):
    m = rc.RestoreCostModel(artifact())

    def boom(*a, **k):
        raise AssertionError("re-validation or hashing per decision")
    monkeypatch.setattr(rc, "artifact_digest", boom)
    monkeypatch.setattr(rc, "canonical_json", boom)
    monkeypatch.setattr(rc, "_validate_artifact", boom)
    d = CostAware(m).choose(plan(1000, ["host", "host", "disk"]))
    assert d.selected_len == 512


# ---- compatibility, domain, predictions ----------------------------------------------------

def test_incompatible_plans_raise_instead_of_falling_back():
    base = CostAware(rc.RestoreCostModel(artifact()))
    with pytest.raises(rc.IncompatibleModelError, match="namespace"):
        base.choose(plan(1000, ["host"], ns="cd" * 32))
    with pytest.raises(rc.IncompatibleModelError, match="chunk size"):
        base.choose(plan(1000, ["host"], C=128))
    with pytest.raises(rc.IncompatibleModelError, match="restore backend"):
        base.choose(plan(1000, ["host"], backend="batched"))
    with pytest.raises(rc.IncompatibleModelError, match="namespace"):  # checked even with no positive candidate
        base.choose(plan(1000, [], ns="cd" * 32))
    assert base.choose(plan(1000, ["host"], group=9)).selected_len >= 0  # baseline ignores the staging group
    batched = CostAware(rc.RestoreCostModel(artifact(comp=compat(backend="batched", group=4))))
    assert batched.choose(plan(1000, ["host"], backend="batched", group=4)).reason == "min_predicted_objective"
    with pytest.raises(rc.IncompatibleModelError, match="staging group 2 != 4"):
        batched.choose(plan(1000, ["host"], backend="batched", group=2))


BOX = dict(prompt=(500, 1500), suffix=(1, 1000), kmax=4, hmax=2, dmax=1)


@pytest.mark.parametrize("p, dom, field", [
    (plan(2000, ["host"]), BOX, "prompt_len"),
    (plan(400, []), BOX, None),                                   # no positive candidate: model not consulted
    (plan(1200, ["host", "disk", "host"]), {**BOX, "kmax": 2}, "restore_chunks"),
    (plan(1200, ["host", "host", "host"]), BOX, "host_chunks"),
    (plan(1200, ["disk", "disk"]), BOX, "disk_chunks"),
    (plan(1300, ["host"]), BOX, "suffix_len"),                   # S = 1044 > 1000
])
def test_out_of_domain_plans_fall_back_to_always_restore_with_a_reason(p, dom, field):
    m = rc.RestoreCostModel(artifact(dom=domain(**dom)))
    d = CostAware(m).choose(p)
    if field is None:
        assert (d.selected_len, d.reason, d.objective_ms) == (0, "no_feasible_restore", None)
    else:
        assert d.selected_len == p.max_feasible_len and d.objective_ms is None
        assert d.reason == f"out_of_domain_always_restore:{field}"
        assert m.domain_violation(p) == field


def test_predict_ms_checks_inputs_and_domain_and_includes_admission():
    coef = coefficients()
    m = rc.RestoreCostModel(artifact(coef, dom=domain(admitted=(0, 3))))
    # N = 2000: 7 complete chunks; after L = 512 (2 chunks) up to 5 can be admitted
    assert m.predict_ms(2000, 512, 2, 0, 3) == pytest.approx(truth_ms(coef, 2000, 512, 2, 0, 3))
    assert m.predict_ms(2000, 512, 2, 0, 3) - m.predict_ms(2000, 512, 2, 0, 0) == pytest.approx(3 * 4.5)
    with pytest.raises(rc.OutOfDomainError, match="admitted_chunks"):
        m.predict_ms(2000, 512, 2, 0, 4)  # consistent, but outside the declared range [0, 3]
    for args in [(1000, 500, 2, 0, 0), (1000, 512, 1, 0, 0), (1000, 1024, 4, 0, 0), (1000, True, 0, 0, 0)]:
        with pytest.raises(ValueError):
            m.predict_ms(*args)
    # N = 300 has one complete chunk; restored at L = 256 it is indexed, so nothing can be admitted
    with pytest.raises(ValueError, match="complete chunks after L=256") as e:
        m.predict_ms(300, 256, 1, 0, 1)
    assert not isinstance(e.value, rc.OutOfDomainError)
    assert m.predict_ms(300, 0, 0, 0, 1) == pytest.approx(truth_ms(coef, 300, 0, 0, 0, 1))  # L = 0: chunk 0 admissible


def test_admitted_range_is_a_prediction_domain_not_a_decision_domain():
    """m does not enter the objective and is the same for every candidate (controlled state), so the
    decision domain is the box projected onto N, S and the chunk counts; predict_ms checks the full box."""
    coef = coefficients(full=(1.0, 0.0, 0.0), partial=(99.0, 0.0, 0.0, 0.0, 0.0))
    m = rc.RestoreCostModel(artifact(coef, dom=domain(admitted=(0, 1))))
    # N = 1500: 5 complete chunks; 0-1 indexed in host, 2-4 absent, so the request admits m = 3 whatever L is
    p = plan(1500, ["host", "host"])
    assert m.domain_violation(p) is None
    d = CostAware(m).choose(p)
    assert (d.selected_len, d.reason) == (0, "min_predicted_objective")
    for L, h in ((0, 0), (256, 1), (512, 2)):
        with pytest.raises(rc.OutOfDomainError, match="admitted_chunks"):
            m.predict_ms(1500, L, h, 0, 3)


# ---- hand-computed CostAware choices -------------------------------------------------------

def test_hand_computed_interior_choice_and_recorded_objectives():
    # N = 1000, C = 256, tiers host, host, disk; coefficients from m5_toy.coefficients():
    #   L=0:   10 + 40*1.0 + 8*1.0^2/2                                  = 54.0
    #   L=256: 15 + 1*1 + 40*0.744 + 8*0.744*(0.256 + 0.372)           = 49.497856
    #   L=512: 15 + 1*2 + 40*0.488 + 8*0.488*(0.512 + 0.244)           = 39.471424   <- minimum
    #   L=768: 15 + 1*2 + 20*1 + 40*0.232 + 8*0.232*(0.768 + 0.116)    = 47.920704
    d = CostAware(rc.RestoreCostModel(artifact())).choose(plan(1000, ["host", "host", "disk"]))
    assert d.selected_len == 512 and d.reason == "min_predicted_objective"
    want = [54.0, 49.497856, 39.471424, 47.920704]
    assert [L for L, _ in d.objective_ms] == [0, 256, 512, 768]
    assert [ms for _, ms in d.objective_ms] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("coef, want", [
    (coefficients(partial=(60.0, 1.0, 20.0, 40.0, 8.0)), 0),        # suffix path overhead dominates: recompute
    (coefficients(partial=(15.0, 1.0, 1.0, 40.0, 8.0)), 768),       # cheap disk: longest prefix
    (coefficients(full=(5.0, 0.0, 0.0), partial=(5.0, 0.0, 0.0, 0.0, 0.0)), 768),   # all tied: largest L
    (coefficients(full=(12.0, 0.0, 0.0), partial=(10.0, 0.0, 100.0, 0.0, 0.0)), 512),  # 256/512 tie at 10
])
def test_hand_computed_choices_and_deterministic_ties(coef, want):
    d = CostAware(rc.RestoreCostModel(artifact(coef))).choose(plan(1000, ["host", "host", "disk"]))
    assert d.selected_len == want
    objs = dict(d.objective_ms)
    assert objs[want] == min(objs.values()) and all(L <= want for L, v in objs.items() if v == objs[want])


def test_feasibility_limits_the_candidates_the_model_sees():
    d = CostAware(rc.RestoreCostModel(artifact())).choose(
        plan(1000, ["host", "host", "disk"], feasible=1, limit="no_staging_buffer", staging=False))
    assert [L for L, _ in d.objective_ms] == [0, 256] and d.selected_len == 256
    assert math.isclose(dict(d.objective_ms)[256], 49.497856, rel_tol=1e-12)
