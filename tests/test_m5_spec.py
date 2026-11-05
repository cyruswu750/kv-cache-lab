"""M5b specification: the declared grid, its identifiability and domain, configuration-level holdout separation,
and the seeded schedule (CPU, no torch, no measurement)."""

import subprocess
import sys

import numpy as np
import pytest

import m5b_toy as t
from kvlab import m5_spec as sp
from kvlab import restore_cost as rc
from kvlab.boundaries import max_restore_len
from m5_toy import artifact, coefficients, compat


@pytest.fixture(scope="module")
def specs():
    cal = sp.calibration_spec()
    return cal, sp.evaluation_spec(cal)


def test_cases_declare_every_feasible_candidate_and_their_admissions(specs):
    for spec in specs:
        for c in spec["cases"]:
            N = c["prompt"]["length"]
            assert len(c["chunks"]) == N // sp.C and len(sp.prompt_tokens(c["prompt"])) == N
            k = c["feasible_chunks"]
            assert c["candidates"] == list(range(0, k * sp.C + 1, sp.C)) and k * sp.C <= max_restore_len(N, sp.C)
            assert all(s != "absent" for s in c["chunks"][:k])
            assert c["admitted"] == [i for i, s in enumerate(c["chunks"]) if s == "absent"]
            assert c["capacities"] == {"host_slots": c["chunks"].count("host") + len(c["admitted"]),
                                       "disk_files": c["chunks"].count("disk")}
            assert c["source"]["P"] == (max([i for i, s in enumerate(c["chunks"]) if s != "absent"], default=-1) + 1) * sp.C


def test_grid_coverage_and_counts(specs):
    cal, ev = specs
    for spec in specs:
        tiers = [c["feasible_tiers"] for c in spec["cases"]]
        assert any(0 < len(x) <= 2 for x in tiers)                                       # short prefixes
        assert any(c["prompt"]["length"] - sp.C > sp.C for c in spec["cases"] if c["feasible_chunks"])  # S > C
        assert any(x and set(x) == {"host"} for x in tiers) and any(x and set(x) == {"disk"} for x in tiers)
        assert any(x and x[0] == "host" and "disk" in x for x in tiers)                  # host then disk
        assert any(x and x[0] == "disk" and "host" in x for x in tiers)                  # disk then host
        assert len({c["prompt"]["length"] for c in spec["cases"]}) == len(spec["cases"])     # varying N
        assert spec["restore_backend"] == "batched" and spec["restore_staging_chunks"] == 4
    assert len({len(c["admitted"]) for c in cal["cases"]}) >= 3
    assert cal["counts"]["candidate_rows"] == 81 and ev["counts"]["candidate_rows"] == 47
    assert cal["counts"]["samples_per_session"] == 81 * 14 and ev["counts"]["samples_per_session"] == 71 * 12
    assert cal["counts"]["measured_samples_total"] == 81 * 12 * 3 and ev["counts"]["measured_samples_total"] == 71 * 10 * 4
    assert cal["budget"]["estimate_note"] == "planning estimate, not measured"
    assert cal["counts"]["estimated_batch_s"] < cal["budget"]["deadline_s"]
    assert ev["counts"]["estimated_batch_s"] < ev["budget"]["deadline_s"]


def test_calibration_design_is_identifiable_before_any_measurement(specs):
    cal, _ = specs
    d = cal["design"]
    assert d["rank"] == 9 and d["n_columns"] == 9 and d["condition_number"] < 100 and not d["zero_features"]
    assert sum(d["rows_per_case"].values()) == d["n_rows"] == 81
    assert d["row_share_per_case"]["cal-10"] == pytest.approx(15 / 81)   # uniform rows: larger cases weigh more
    # the fit itself accepts the design (synthetic means; nothing measured) and records the declared domain
    truth = coefficients(full=(13.0, 40.0, 8.0), partial=(17.0, 0.8, 2.5, 30.0, 9.0), admit=4.5)
    cases = []
    for c in cal["cases"]:
        cands, h, dd = [], 0, 0
        for k, L in enumerate(c["candidates"]):
            if k:
                h, dd = h + (c["feasible_tiers"][k - 1] == "host"), dd + (c["feasible_tiers"][k - 1] == "disk")
            x = rc.design_row(c["prompt"]["length"], L, h, dd, len(c["admitted"]))
            coef = [truth[g][f] for g, f in rc._COLUMNS]
            cands.append({"restore_len": L, "mean_ms": float(np.dot(x, coef)), "n_samples": 12})
        cases.append({"case_id": c["case_id"], "prompt_len": c["prompt"]["length"], "tiers": c["feasible_tiers"],
                      "admitted_chunks": len(c["admitted"]), "candidates": cands})
    art = rc.fit_restore_cost(cases, calibration_case_ids=[c["case_id"] for c in cal["cases"]],
                              evaluation_case_ids=[c["case_id"] for c in specs[1]["cases"]],
                              compatibility=compat(), weighting=cal["fit"]["weighting"], provenance={"t": "synthetic"})
    assert art["domain"] == cal["domain"]
    assert art["fit"]["condition_number"] == pytest.approx(d["condition_number"])


def test_evaluation_cases_are_inside_both_domains_by_the_model_itself(specs):
    cal, ev = specs
    m = rc.RestoreCostModel(artifact(dom=cal["domain"], comp=compat()))
    from kvlab.m5_fit import case_plan
    for c in ev["cases"]:
        assert sp.domain_problems(cal["domain"], c) == []
        assert m.domain_violation(case_plan(c, ev, m.namespace_digest)) is None
        h = d = 0
        for k, L in enumerate(c["candidates"]):
            if k:
                h, d = h + (c["feasible_tiers"][k - 1] == "host"), d + (c["feasible_tiers"][k - 1] == "disk")
            m.predict_ms(c["prompt"]["length"], L, h, d, len(c["admitted"]))   # raises if out of domain


def test_holdout_is_separated_by_configuration_not_by_ids(specs):
    cal, _ = specs
    base = sp.grid_cases(sp.EVALUATION_GRID, "evaluation")
    renamed = sp.make_case("ev-99", "evaluation", 1100, cal["cases"][2]["chunks"], 59999)   # cal-03's config
    with pytest.raises(sp.SpecError, match="duplicate calibration configurations"):
        sp.build_spec("X", "evaluation", base + [renamed], sp.EVAL_SCHEDULE, deadline_s=1, calibration=cal)
    same_prompt = sp.make_case("ev-98", "evaluation", 1100, ["host", "host", "absent", "absent"], 51003)
    with pytest.raises(sp.SpecError, match="duplicate calibration configurations or prompts"):
        sp.build_spec("X", "evaluation", base + [same_prompt], sp.EVAL_SCHEDULE, deadline_s=1, calibration=cal)
    twin = dict(base[0], case_id="ev-97")
    with pytest.raises(sp.SpecError, match="duplicate case ids or duplicate configurations"):
        sp.build_spec("X", "evaluation", base + [twin], sp.EVAL_SCHEDULE, deadline_s=1, calibration=cal)


@pytest.mark.parametrize("N, prefix, field", [(4200, "H", "prompt_len"), (3900, "H" * 15, "restore_chunks"),
                                             (3900, "H" * 13, "host_chunks"), (3900, "D" * 13, "disk_chunks"),
                                             (2500, "H", "admitted_chunks")])
def test_out_of_domain_evaluation_cases_are_refused_not_dropped(specs, N, prefix, field):
    cal, _ = specs
    bad = sp.grid_cases((("ev-x", N, prefix, 59000),), "evaluation")
    with pytest.raises(sp.SpecError, match=field):
        sp.build_spec("X", "evaluation", bad, sp.EVAL_SCHEDULE, deadline_s=1, calibration=cal)


def test_unidentifiable_or_inconsistent_specs_are_refused(specs):
    cal, _ = specs
    host_only = [(cid, N, p.replace("D", "H"), s) for cid, N, p, s in sp.CALIBRATION_GRID]
    with pytest.raises(sp.SpecError, match="not identifiable"):
        sp.build_spec("X", "calibration", sp.grid_cases(host_only, "calibration"), sp.CAL_SCHEDULE, deadline_s=1)
    ev_cases = sp.grid_cases(sp.EVALUATION_GRID, "evaluation")
    with pytest.raises(sp.SpecError, match="restore_backend"):
        sp.build_spec("X", "evaluation", ev_cases, sp.EVAL_SCHEDULE, deadline_s=1, calibration=cal, backend="baseline")
    with pytest.raises(sp.SpecError, match="partition"):
        sp.build_spec("X", "evaluation", ev_cases, {**sp.EVAL_SCHEDULE, "sessions": {"s01": 1, "s09": 2}},
                      deadline_s=1, calibration=cal)
    with pytest.raises(sp.SpecError, match="needs its calibration"):
        sp.build_spec("X", "evaluation", ev_cases, sp.EVAL_SCHEDULE, deadline_s=1)
    with pytest.raises(sp.SpecError, match="one state"):
        sp.make_case("x", "calibration", 600, ["host"], 1)


def test_schedule_is_seeded_complete_and_reproducible(specs):
    for spec in specs:
        for sid in spec["schedule"]["sessions"]:
            sched = sp.schedule(spec, sid)
            assert sched == sp.schedule(spec, sid) and len(sched) == spec["counts"]["samples_per_session"]
            assert sorted(sp.case_order(spec, sid)) == sorted(c["case_id"] for c in spec["cases"])
            by = {}
            for cid, phase, rep, pos, label in sched:
                by.setdefault((cid, phase, rep), []).append((pos, label))
            for (cid, phase, rep), runs in by.items():
                labels = sp.run_labels(spec, sp.case_by_id(spec)[cid])
                assert [p for p, _ in runs] == list(range(len(labels))) and sorted(x for _, x in runs) == sorted(labels)
            assert {p for _c, p, *_ in sched} == {"warmup", "measured"}
        orders = {tuple(sp.case_order(spec, sid)) for sid in spec["schedule"]["sessions"]}
        assert len(orders) > 1
    _, ev = specs
    assert {lab for c in ev["cases"] for lab in sp.run_labels(ev, c) if lab.startswith("policy:")} == \
        {"policy:cost_aware", "policy:always_restore", "policy:recompute"}


def test_spec_binding_and_digest(specs):
    cal, ev = specs
    assert ev["calibration_spec_sha256"] == sp.digest(cal) and sp.digest(cal) == sp.digest(sp.calibration_spec())
    assert ev["calibration_case_ids"] == [c["case_id"] for c in cal["cases"]]
    tiny_cal, tiny_ev = t.tiny_specs()
    assert tiny_cal["design"]["rank"] == 9 and tiny_ev["calibration_spec_sha256"] == sp.digest(tiny_cal)


def test_spec_validator_fit_and_analysis_modules_do_not_import_torch():
    code = ("import sys; import kvlab.m5_spec, kvlab.m5_validate, kvlab.m5_fit, kvlab.m5_analysis; "
            "print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False"


def test_plan_cli_runs_on_cpu():
    r = subprocess.run([sys.executable, "scripts/m5_run.py", "plan", "--split", "calibration"],
                       capture_output=True, text=True, check=True)
    import json
    out = json.loads(r.stdout)
    assert out["design"]["rank"] == 9 and out["counts"]["candidate_rows"] == 81
