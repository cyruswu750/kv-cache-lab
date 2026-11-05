"""M5b end to end on the CPU with synthetic timings: tiny sessions -> the real run_batch -> validation -> fit ->
frozen predictions -> evaluation sessions with real CostAware/AlwaysRestore/Recompute runs -> validation ->
analysis. Then tampered, incomplete and failed evidence must be rejected.

Latencies come from FakeTiming (a known truth model plus policy overheads; warmups +1000 ms). Nothing here is a
measurement or a performance result."""

import copy
import importlib.util
import json
import shutil
import time
from pathlib import Path

import pytest

import m5b_toy as t
from kvlab import m3_confirm as mc
from kvlab import m5_analysis as ma
from kvlab import m5_bench as mb
from kvlab import m5_fit as mf
from kvlab import m5_spec as sp
from kvlab import m5_validate as mv
from kvlab import restore_cost as rc
from kvlab.disk_store import DiskStore
from kvlab.resident_reference import ResidentReference

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5pipe")
    model = t.tiny_model()
    ns = t.tiny_ns(model)
    cal, ev = t.tiny_specs()
    cal_out = {sid: t.run_tiny_session(model, ns, cal, sid, root / "scratch" / "cal" / sid) for sid in cal["schedule"]["sessions"]}
    cal_dir = t.make_run_dir(root / "calroot", cal, cal_out)
    v = mv.validate(cal_dir, cal, require_cuda=False)
    assert v["exit"] == 0, (v["problems"][:5], v["strict_problems"][:5])
    art = mf.fit_artifact(cal, v["sessions"], ev, run_dir=cal_dir)
    rc.save_artifact(art, root / "cost_model.json")
    model_c = rc.load_model(root / "cost_model.json")
    preds = mf.make_predictions(ev, model_c, created_unix=time.time())
    (root / "predictions.json").write_text(json.dumps(preds))
    ev_out = {sid: t.run_tiny_session(model, ns, ev, sid, root / "scratch" / "ev" / sid, cost_model=model_c,
                                      predictions=preds) for sid in ev["schedule"]["sessions"]}
    ev_dir = t.make_run_dir(root / "evroot", ev, ev_out, cost_model_path=str(root / "cost_model.json"),
                            predictions_path=str(root / "predictions.json"),
                            bindings={"cost_model_digest": model_c.digest, "predictions_sha256": preds["sha256"]})
    return {"root": root, "model": model, "ns": ns, "cal": cal, "ev": ev, "cal_dir": cal_dir, "ev_dir": ev_dir,
            "art": model_c.artifact, "preds": preds, "cal_out": cal_out, "ev_out": ev_out}


def vcal(p, run_dir=None):
    return mv.validate(run_dir or p["cal_dir"], p["cal"], require_cuda=False)


def vev(p, run_dir=None, preds=None, art=None):
    return mv.validate(run_dir or p["ev_dir"], p["ev"], predictions=preds or p["preds"], artifact=art or p["art"],
                       require_cuda=False)


def copy_run(p, which, tmp_path):
    dst = tmp_path / which
    shutil.copytree(p[f"{which}_dir"], dst)
    return dst


# ---- the valid pipeline --------------------------------------------------------------------

def test_valid_calibration_fit_predictions_and_evaluation(pipe):
    p = pipe
    v = vcal(p)
    assert v["exit"] == 0 and v["valid"]
    # warmups were 1000 ms slower; the means are the truth: warmups never reach the fit
    means = mf.calibration_cases(p["cal"], v["sessions"])
    by_id = sp.case_by_id(p["cal"])
    for c in means:
        for cand in c["candidates"]:
            assert cand["mean_ms"] == pytest.approx(t.truth_ms(by_id[c["case_id"]], cand["restore_len"]), abs=1e-5)
            assert cand["n_samples"] == 2 * 3
    for g, names in rc.FEATURES.items():
        for f in names:
            assert p["art"]["coefficients_ms"][g][f] == pytest.approx(t.TRUTH[g][f], rel=2e-3, abs=2e-3), (g, f)
    assert p["art"]["calibration"]["evaluation_case_ids"] == sorted(c["case_id"] for c in p["ev"]["cases"])
    assert p["art"]["provenance"]["calibration_spec_sha256"] == sp.digest(p["cal"])
    assert set(p["art"]["provenance"]["calibration_evidence_sha256"]) == mv.expected_files(p["cal"])
    # predictions: CostAware's choice equals the truth's argmin wherever the truth is not (nearly) tied
    ev_by = sp.case_by_id(p["ev"])
    for cid, pc in p["preds"]["cases"].items():
        truth = {L: t.truth_ms(ev_by[cid], L) for L in ev_by[cid]["candidates"]}
        best = ma.best_candidate(truth)
        if sorted(truth.values())[1:2] and sorted(truth.values())[1] - min(truth.values()) > 0.01:
            assert pc["decision"]["selected_len"] == best, cid
        assert pc["decision"]["reason"] == "min_predicted_objective"
    ve = vev(p)
    assert ve["exit"] == 0, (ve["problems"][:5], ve["strict_problems"][:5])
    res = ma.analyze(p["ev"], ve["sessions"], p["preds"], ve)
    agg = res["aggregate"]
    assert agg["prediction_error"]["mean_abs_error_ms"] < 1e-2
    # the delivered CostAware-minus-AlwaysRestore difference = candidate difference + overhead difference
    want = sum(t.truth_ms(ev_by[c], p["preds"]["cases"][c]["decision"]["selected_len"])
               - t.truth_ms(ev_by[c], ev_by[c]["candidates"][-1]) for c in ev_by) / len(ev_by)
    want += t.OVERHEAD_MS["cost_aware"] - t.OVERHEAD_MS["always_restore"]
    assert agg["delivered_differences_ms"]["cost_aware_minus_always_restore"]["mean_ms"] == pytest.approx(want, abs=1e-5)
    assert agg["policies"]["recompute"]["naive_mean_regret_ms"] >= 0


def test_scripts_fit_and_analyze_write_new_outputs_only(pipe, tmp_path):
    p = pipe
    fit = _script("m5_fit")
    out = tmp_path / "fit"
    assert fit.run(p["cal_dir"], out, require_cuda=False, cal=p["cal"], ev=p["ev"]) == 0
    assert {x.name for x in out.iterdir()} == {"validation.json", "calibration_means.json", "cost_model.json",
                                                "predictions.json"}
    with pytest.raises(FileExistsError):
        fit.run(p["cal_dir"], out, require_cuda=False, cal=p["cal"], ev=p["ev"])
    bad = copy_run(p, "cal", tmp_path)
    t.rewrite_session(bad, "s01", lambda o: o["samples"][3].update(latency_ns=-5))
    out_bad = tmp_path / "fit_bad"
    assert fit.run(bad, out_bad, require_cuda=False, cal=p["cal"], ev=p["ev"]) == 3
    assert {x.name for x in out_bad.iterdir()} == {"validation.json"}       # nothing fitted from invalid evidence
    an = _script("m5_analyze")
    # analysing the pipeline's evaluation run with a DIFFERENT (freshly frozen) fit directory: bindings fail
    assert an.run(p["ev_dir"], out, tmp_path / "an_other", require_cuda=False, ev=p["ev"]) == 3
    fit_dir = tmp_path / "fit_pipe"
    fit_dir.mkdir()
    shutil.copy(p["root"] / "cost_model.json", fit_dir / "cost_model.json")
    shutil.copy(p["root"] / "predictions.json", fit_dir / "predictions.json")
    assert an.run(p["ev_dir"], fit_dir, tmp_path / "an", require_cuda=False, ev=p["ev"]) == 0
    assert {x.name for x in (tmp_path / "an").iterdir()} == {"validation.json", "analysis.json", "analysis.md"}


# ---- tampering, incompleteness, failure -----------------------------------------------------

def _first(o, **match):
    for s in o["samples"]:
        if all(s.get(k) == v if not callable(v) else v(s.get(k)) for k, v in match.items()):
            return s
    raise AssertionError(match)


SAMPLE_EDITS = {
    "drop": (lambda o: o["samples"].pop(7), "sample identities differ"),
    "duplicate": (lambda o: o["samples"].insert(7, copy.deepcopy(o["samples"][7])), "sample identities differ"),
    "swap": (lambda o: o["samples"].__setitem__(slice(3, 5), o["samples"][4:2:-1]), "identities differ"),
    "relabel": (lambda o: o["samples"][2].update(run="cand:999"), "identities differ"),
    "warmup_flag": (lambda o: _first(o, phase="warmup").update(phase="measured"), "identities differ"),
    "selected": (lambda o: _first(o, phase="measured", run="cand:4").update(selected_len=8), "selected_len"),
    "truncated": (lambda o: _first(o, phase="measured", run="cand:4").update(actual_len=0, restore_len=0), "actual_len"),
    "stored": (lambda o: _first(o, stored_chunks=lambda x: bool(x)).update(stored_chunks=[]), "stored_chunks"),
    "write_skip": (lambda o: o["samples"][5]["counters_delta"]["write_skips"].update(host_all_pinned=1), "counter delta"),
    "spill": (lambda o: o["samples"][5]["counters_delta"].update(host_spills_to_disk=1), "counter delta"),
    "latency_zero": (lambda o: o["samples"][5].update(latency_ns=0, t1_ns=o["samples"][5]["t0_ns"]), "invalid timing"),
    "latency_bool": (lambda o: o["samples"][5].update(latency_ns=True), "invalid timing"),
    "latency_float": (lambda o: o["samples"][5].update(latency_ns=1.5), "invalid timing"),
    "fingerprint": (lambda o: o["samples"][6].update(fingerprint="0" * 64), "fingerprint"),
    "not_resident": (lambda o: _first(o, page_cache=lambda pc: bool(pc["files"]))["page_cache"]["files"][0]["final"]
                     .update(resident_pages=0), "page-cache outcome"),
    "drop_pagecache": (lambda o: o["samples"][4]["page_cache"]["counters_after_request"]["vmstat"]
                       .update(drop_pagecache=8), "drop_pagecache"),
    "status": (lambda o: o.update(status="failed"), "status"),
    "spec_sha": (lambda o: o.update(spec_sha256="0" * 64), "spec_sha256"),
    "dirty": (lambda o: o["source"].update(git_status_porcelain=[" M kvlab/x.py"]), "not clean"),
    "namespace": (lambda o: o["namespace"].update(digest="0" * 64), "malformed"),
    "content": (lambda o: o["cases"][0]["setup_content"].update(ok=False), "setup_content"),
    "case_missing": (lambda o: o["cases"].pop(1), "case records"),
    "plan": (lambda o: o["samples"][3]["decision"]["plan"].update(limit_reason="corrupted"), "decision plan"),
    "reason": (lambda o: o["samples"][3]["decision"].update(reason="other"), "decision reason"),
    "miss": (lambda o: o["samples"][3].update(miss_reason="disk_payload_crc"), "miss_reason"),
}


@pytest.mark.parametrize("name", sorted(SAMPLE_EDITS))
def test_tampered_calibration_sessions_are_invalid(pipe, tmp_path, name):
    edit, match = SAMPLE_EDITS[name]
    d = copy_run(pipe, "cal", tmp_path)
    t.rewrite_session(d, "s02", edit)
    v = vcal(pipe, d)
    assert v["exit"] == 3 and not v["valid"] and v["sessions"] is None
    assert any(match in x for x in v["problems"]), v["problems"][:6]


@pytest.mark.parametrize("edit, match", [
    (lambda o: _first(o, run="cand:0", phase="measured")["strict"].update(logits_equal=False), "logits differ"),
    (lambda o: _first(o, run="cand:4", phase="measured")["strict"]["restored_bytes"].update(all_bytes_equal=False,
                                                                                             mismatched=1),
     "restored bytes differ"),
    (lambda o: _first(o, run="cand:0", phase="measured").update(first_token=-1), "first token"),
])
def test_strict_mismatches_in_complete_evidence_are_exit_4(pipe, tmp_path, edit, match):
    d = copy_run(pipe, "cal", tmp_path)
    t.rewrite_session(d, "s01", edit)
    v = vcal(pipe, d)
    assert v["exit"] == 4 and not v["problems"] and any(match in x for x in v["strict_problems"])


def test_evaluation_specific_bindings(pipe, tmp_path):
    p = pipe
    d = copy_run(p, "ev", tmp_path)
    t.rewrite_session(d, "s01", lambda o: _first(o, run="policy:cost_aware").update(selected_len=-4))
    assert any("selected_len" in x for x in vev(p, d)["problems"])
    d = copy_run(p, "ev", tmp_path / "b")
    t.rewrite_session(d, "s02", lambda o: _first(o, run="policy:cost_aware")["decision"].update(objective_ms=[[0, 1.0]]))
    assert any("objectives differ" in x for x in vev(p, d)["problems"])
    preds = copy.deepcopy(p["preds"])
    preds["cases"]["te-01"]["decision"]["selected_len"] = 4 if preds["cases"]["te-01"]["decision"]["selected_len"] == 0 else 0
    v = vev(p, preds=preds)
    assert any("predictions digest mismatch" in x for x in v["problems"])
    late = copy.deepcopy(p["preds"])
    late["created_unix"] = time.time() + 1e6
    late["sha256"] = sp.digest({k: x for k, x in late.items() if k != "sha256"})
    assert any("not frozen before" in x for x in vev(p, preds=late)["problems"])
    art = copy.deepcopy(p["art"])
    art["digest"] = "f" * 64
    assert any("artifact_digest" in x or "cost model / predictions" in x for x in vev(p, art=art)["problems"])
    v = mv.validate(p["ev_dir"], p["ev"], require_cuda=False)
    assert v["exit"] == 3 and "needs the frozen cost model" in v["problems"][0]
    cal_bad = copy_run(p, "cal", tmp_path / "c")
    t.rewrite_session(cal_bad, "s01", lambda o: o.update(cost_model_digest="x"))
    assert any("must not load a cost model" in x for x in vcal(p, cal_bad)["problems"])


def test_file_level_and_batch_level_problems(pipe, tmp_path):
    p = pipe
    d = copy_run(p, "cal", tmp_path / "a")
    (d / "session_s02.json").unlink()
    v = vcal(p, d)
    assert any("missing files" in x for x in v["problems"]) and any("no session file" in x for x in v["problems"])
    d = copy_run(p, "cal", tmp_path / "b")
    (d / "notes.txt").write_text("x")
    assert any("unexpected files" in x for x in vcal(p, d)["problems"])
    d = copy_run(p, "cal", tmp_path / "c")
    sj = json.loads((d / "spec.json").read_text())
    sj["schedule"]["reps"] = 2
    (d / "spec.json").write_text(json.dumps(sj))
    assert any("spec.json differs" in x for x in vcal(p, d)["problems"])
    d = copy_run(p, "cal", tmp_path / "e")
    rec = json.loads((d / mc.BATCH_RECORD).read_text())
    rec["execution_status"] = "failed"
    (d / mc.BATCH_RECORD).write_text(json.dumps(rec))
    assert any("execution_status" in x for x in vcal(p, d)["problems"])
    d = copy_run(p, "cal", tmp_path / "f")
    rec = json.loads((d / mc.BATCH_RECORD).read_text())
    rec["elapsed_s"] = rec["sessions"][-1]["ended_s"] = 10_000.0
    (d / mc.BATCH_RECORD).write_text(json.dumps(rec))
    assert any("deadline" in x or "budget" in x for x in vcal(p, d)["problems"])
    d = copy_run(p, "cal", tmp_path / "g")
    plan = json.loads((d / "batch_plan.json").read_text())
    plan["commands"][0]["env"]["EXTRA"] = "1"
    (d / "batch_plan.json").write_text(json.dumps(plan))
    v = vcal(p, d)
    assert any("batch plan differs" in x for x in v["problems"])
    d = copy_run(p, "cal", tmp_path / "h")
    (d / "session_s01.json").write_text("{not json")
    assert any("unreadable" in x for x in vcal(p, d)["problems"])


def test_a_strict_failure_stops_the_session_and_keeps_its_record(pipe, tmp_path, monkeypatch):
    p = pipe
    real = ResidentReference.run

    def perturbed(self, *a, **k):
        tokens, steps = real(self, *a, **k)
        steps[0] = steps[0] + 1.0
        return tokens, steps
    monkeypatch.setattr(ResidentReference, "run", perturbed)
    out = copy.deepcopy(t.FAKE_IDENTITY)
    timing = t.FakeTiming(t.scheduled_latencies_ns(p["cal"], "s01"))
    with pytest.raises(RuntimeError, match="strict check failed"):
        mb.run_session(p["model"], p["ns"], p["cal"], "s01", tmp_path / "s", out=out, pin=False,
                       clock=timing.clock, sync=timing.sync, read_counters=t.fake_counters)
    assert out["status"] == "running" and out["samples"][-1]["strict"]["logits_equal"] is False


def test_an_unintended_read_fault_in_a_performance_case_makes_the_evidence_invalid(pipe, tmp_path, monkeypatch):
    """A read failure inside the selected prefix truncates (the fallback works), but performance evidence must
    then be rejected: the sample's actual length, miss reason and counters differ from the declared case."""
    p = pipe
    real, calls = DiskStore.read_into, []

    def flaky(self, key, *a):
        calls.append(1)
        if len(calls) == 25:
            return False, "disk_read_oserror:EIO", 0   # the entry stays indexed, so the reset still succeeds
        return real(self, key, *a)
    monkeypatch.setattr(DiskStore, "read_into", flaky)
    out = t.run_tiny_session(p["model"], p["ns"], p["cal"], "s01", tmp_path / "s")
    monkeypatch.setattr(DiskStore, "read_into", real)
    bad = [s for s in out["samples"] if s["actual_len"] != s["selected_len"]]
    assert len(bad) == 1 and bad[0]["restore_miss_reason"] == "disk_read_oserror:EIO"
    d = t.make_run_dir(tmp_path / "root", p["cal"], {"s01": out, "s02": p["cal_out"]["s02"]})
    v = vcal(p, d)
    assert v["exit"] == 3 and any("actual_len" in x for x in v["problems"])


def test_batch_refuses_before_creating_anything(tmp_path, monkeypatch):
    run = _script("m5_run")
    monkeypatch.setattr(run.mc, "source_identity", lambda root: {"git_status_porcelain": []})
    ev = run.spec_for("evaluation")
    existing = tmp_path / "exists"
    existing.mkdir()
    args = type("A", (), {"cost_model": str(tmp_path / "missing.json"), "predictions": str(tmp_path / "p.json")})()
    problems = run.pre_launch_problems(ev, existing, args)
    assert any("exists" in x for x in problems) and any("frozen files unusable" in x for x in problems)
    args.cost_model = args.predictions = None
    assert any("needs --cost-model" in x for x in run.pre_launch_problems(ev, tmp_path / "new", args))
    monkeypatch.setattr(run.mc, "source_identity", lambda root: {"git_status_porcelain": [" M x"]})
    assert any("not clean" in x for x in run.pre_launch_problems(run.spec_for("calibration"), tmp_path / "n2", args))
    assert not (tmp_path / "new").exists() and not (tmp_path / "n2").exists()


# ---- regressions from the focused M5b review (each reproduces a reported acceptance of invalid evidence) ----

@pytest.mark.parametrize("name, value", [("batch_record.json", []), ("batch_plan.json", []), ("spec.json", None)])
def test_review_non_object_files_never_bypass_validation(pipe, tmp_path, name, value):
    d = copy_run(pipe, "cal", tmp_path)
    (d / name).write_text(json.dumps(value))
    v = vcal(pipe, d)
    assert v["exit"] == 3 and any("not a JSON object" in x for x in v["problems"])


def test_review_a_recorded_final_nonzero_exit_is_not_a_successful_batch(pipe, tmp_path):
    d = copy_run(pipe, "cal", tmp_path)
    rec = json.loads((d / mc.BATCH_RECORD).read_text())
    last = rec["sessions"][-1]
    last["returncode"] = 4
    rec["execution_status"] = "failed"
    rec["stop"] = {"reason": "nonzero_exit", "session": last["session"], "returncode": 4, "detail": "test"}
    (d / mc.BATCH_RECORD).write_text(json.dumps(rec))
    v = vcal(pipe, d)
    assert v["exit"] == 3 and any("batch did not succeed" in x for x in v["problems"])


def test_review_predictions_must_be_what_the_bound_model_produces(pipe, tmp_path):
    p = pipe
    bad = copy.deepcopy(p["preds"])
    first = next(iter(bad["cases"].values()))["predictions"]
    first[next(iter(first))]["predicted_ms"] = -123456.0
    bad["sha256"] = sp.digest({k: x for k, x in bad.items() if k != "sha256"})   # a consistent digest
    assert any("differ from those the bound cost model produces" in x
               for x in mv.prediction_problems(p["ev"], bad, p["art"]))
    assert mv.prediction_problems(p["ev"], p["preds"], p["art"]) == []
    run = _script("m5_run")                                   # the pre-launch loader applies the same check
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    _m, _p, problems = run.load_frozen(p["root"] / "cost_model.json", tmp_path / "bad.json", p["ev"])
    assert any("differ from those the bound cost model produces" in x for x in problems)
    assert run.load_frozen(p["root"] / "cost_model.json", p["root"] / "predictions.json", p["ev"])[2] == []


def test_review_impossible_or_late_page_cache_records_are_invalid(pipe, tmp_path):
    def zero_pages(o):
        pc = _first(o, page_cache=lambda pc: bool(pc["files"]))["page_cache"]
        for ph in ("initial", "final"):
            pc["files"][0][ph].update(size=0, pages=0, resident_pages=0)

    def late(o):
        s = _first(o, page_cache=lambda pc: bool(pc["files"]))
        s["page_cache"]["t_verified_ns"] = s["t1_ns"] + 5
        s["page_cache"]["files"][-1]["final"]["t_ns"] = s["t1_ns"] + 4

    def gap(o):
        s = _first(o, page_cache=lambda pc: bool(pc["files"]))
        s["page_cache"]["timer_start_minus_verified_ns"] += 1
    for edit, match in ((zero_pages, "malformed or impossible"), (late, "out of order"), (gap, "inconsistent")):
        d = copy_run(pipe, "cal", tmp_path / match.split()[0])
        t.rewrite_session(d, "s01", edit)
        v = vcal(pipe, d)
        assert v["exit"] == 3 and any(match in x for x in v["problems"]), (match, v["problems"][:4])


def test_review_missing_identity_is_not_matching_provenance(pipe, tmp_path):
    d = copy_run(pipe, "cal", tmp_path)

    def strip(o):
        for k in ("git_head", "files_sha256", "git_diff_head_sha256"):
            o["source"].pop(k)
        o.pop("model_blobs")
        o.pop("packages")
    for sid in pipe["cal"]["schedule"]["sessions"]:
        t.rewrite_session(d, sid, strip)
    v = vcal(pipe, d)
    for match in ("source identity missing", "model weight identity", "package versions"):
        assert any(match in x for x in v["problems"]), match
    v = mv.validate(pipe["cal_dir"], pipe["cal"], require_cuda=True)
    assert any("GPU identity missing" in x for x in v["problems"]) and any("device" in x for x in v["problems"])
    d = copy_run(pipe, "cal", tmp_path / "nulls")                      # non-empty containers of null entries
    for sid in pipe["cal"]["schedule"]["sessions"]:
        t.rewrite_session(d, sid, lambda o: o.update(model_blobs={"model.safetensors": None}, packages={"torch": None}))
    v = vcal(pipe, d)
    assert v["exit"] == 3 and any("model weight identity" in x for x in v["problems"]) and \
        any("package versions missing" in x for x in v["problems"])


def test_review_malformed_predictions_are_reported_not_raised(pipe):
    bad = copy.deepcopy(pipe["preds"])
    bad["cases"] = None
    v = vev(pipe, preds=bad)
    assert v["exit"] == 3 and v["problems"]
