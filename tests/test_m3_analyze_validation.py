"""VAL-006: the M3 analysis fails closed on invalid or incomplete evidence (CPU only).

These tests run the production validation (kvlab/m3_validate.py) and the real CLI
(scripts/m3_analyze.py: main(argv) in-process, and a subprocess with CUDA hidden for the two
confirmed reproductions). Inputs are scratch copies of the preserved M3 evaluation run
results/m3/eval_20261103T173632, which is only read.
"""

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kvlab import m3_validate as mv

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "m3" / "eval_20261103T173632"
_spec = importlib.util.spec_from_file_location("m3_analyze", ROOT / "scripts" / "m3_analyze.py")
m3_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3_analyze)
SPEC = mv.FROZEN_M3_EVAL
EVAL_FILES = [f"session_s{i}.json" for i in range(1, 6)]
HOST_BATCHED_PLAIN = ("host", "batched", "plain")


@pytest.fixture(scope="module")
def texts():
    """Raw text of the preserved session files, read once."""
    return {p.name: p.read_text() for p in sorted(EVAL.glob("session_*.json"))}


def make_run(tmp_path, texts, edits=None, drop=(), extra=None, monitor=False):
    """A scratch run directory. It holds copies of the preserved session files, with
    `edits[name](report)` applied to parsed copies; files in `drop` are left out, and
    `extra` {name: text} is added."""
    run = tmp_path / "run"
    run.mkdir()
    for name, text in texts.items():
        if name in drop:
            continue
        if edits and name in edits:
            rep = json.loads(text)
            edits[name](rep)
            text = json.dumps(rep, indent=1)
        (run / name).write_text(text)
    for name, text in (extra or {}).items():
        (run / name).write_text(text)
    if monitor:
        for p in EVAL.glob("session_*.gpu_monitor.csv"):
            shutil.copy2(p, run / p.name)
    return run


def cli(run, out=None):
    """The real CLI entry point, in-process. Returns (exit code, m3_validation.json)."""
    code = m3_analyze.main(["--run-dir", str(run)] + (["--out-dir", str(out)] if out else []))
    return code, json.loads(((out or run) / "m3_validation.json").read_text())


def cli_subprocess(run):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "m3_analyze.py"), "--run-dir", str(run)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    return r, json.loads((run / "m3_validation.json").read_text())


def assert_rejected(code, val, out_dir, status="invalid", expect=()):
    assert code == m3_analyze.EXIT[status] and code != 0
    assert val["status"] == status
    assert val["decision"]["supports_adoption"] is False
    assert val["decision"]["evidence_status"] == status
    assert val["decision"]["all_correctness_and_ab_checks_pass"] is False
    assert val["correctness_and_ab_acceptance_recomputed"] is False
    for name in ("m3_analysis.json", "m3_analysis.md", "samples_chronological.csv"):
        assert not (out_dir / name).exists()  # no performance result is drawn
    for e in expect:
        assert any(e in p for p in val["problems"]), (e, val["problems"][:10])


def sample_index(rep, block, r, combo):
    return next(i for i, s in enumerate(rep["samples"]) if s.get("stage") == "m3" and
                (s["block"], s["rep"], s["path"], s["backend"], s["mode"]) == (block, r, *combo))


def case_index(records, case, key=lambda x: (x["L"], x["S"])):
    return next(i for i, x in enumerate(records) if key(x) == case)


def corr_index(rep, case):
    return case_index(rep["correctness"], case, key=lambda c: (c["baseline"]["L"], c["baseline"]["S"]))


# ---------------------------------------------------------------------------------------------
def test_spec_is_the_frozen_protocol_and_the_recorded_commands():
    from kvlab import m3_bench as mb
    assert tuple(SPEC["combos"]) == tuple(mb.COMBOS)
    assert SPEC["sessions"] == {f"s{i}": 31000 + i for i in range(1, 6)}
    cmds = (EVAL / "logs" / "commands.txt").read_text()
    assert "for i in 1 2 3 4 5" in cmds and "--session s$i" in cmds and "--seed 3100$i" in cmds
    assert "--cases " + ",".join(f"{L}:{S}" for L, S in SPEC["cases"]) + " " in cmds
    for flag, v in (("--blocks", SPEC["blocks"]), ("--warmup", SPEC["warmup"]), ("--reps", SPEC["reps"]),
                    ("--staging-chunks", SPEC["staging_chunks"])):
        assert re.search(rf"{flag} {v}\b", cmds), flag
    protocol = (ROOT / "docs" / "M3_PROTOCOL.md").read_text()
    assert "**5 sessions**" in protocol and "Seeds 31001–31005" in protocol and "12 measured repetitions" in protocol
    for L, S in SPEC["cases"]:
        assert f"({L}, {S})" in protocol
    # process warmup as the frozen runner does it: every combination twice at min and max case
    runner = (ROOT / "scripts" / "m3_measure.py").read_text()
    assert "for L, S in sorted({min(cases), max(cases)}):" in runner and '"stage": "process_warmup"' in runner
    assert "for _ in range(2):" in runner and "every combination twice at the smallest and largest" in protocol
    assert list(SPEC["process_warmup"]["cases"]) == sorted({min(SPEC["cases"]), max(SPEC["cases"])})
    assert SPEC["process_warmup"]["repeats"] == 2 and "2 warmup repetitions (kept, flagged, never used)" in protocol
    counts = mv._new_counts(SPEC)
    assert {k: v["expected"] for k, v in counts.items()} == {
        "measured_samples": 5 * 5 * 2 * 12 * 9, "block_warmup_samples": 5 * 5 * 2 * 2 * 9,
        "process_warmup_samples": 5 * 2 * 9 * 2, "correctness_checks": 5 * 5 * 2, "ab_checks": 5 * 5 * 2}


def test_empty_directory_is_invalid_via_the_real_cli(tmp_path):
    # before VAL-006: exit 0 and supports_adoption=true (all([]) over no sessions and no cases)
    r, val = cli_subprocess(tmp_path)
    assert_rejected(r.returncode, val, tmp_path,
                    expect=[f"missing session file session_s{i}.json" for i in range(1, 6)])
    assert all(c["present"] == 0 for c in val["counts"].values())
    assert "supports adoption: False" in r.stdout


def test_missing_required_case_1024_1_is_invalid_via_the_real_cli(tmp_path, texts):
    """The confirmed reproduction: every (1024, 1) sample is removed from copies of the five
    evaluation sessions. Before VAL-006 the case vanished from the decision, and
    supports_adoption went from false to true."""
    def drop_case(rep):
        rep["samples"] = [s for s in rep["samples"] if (s["L"], s["S"]) != (1024, 1)]
    run = make_run(tmp_path, texts, edits={f: drop_case for f in EVAL_FILES}, drop={"session_prof.json"})
    r, val = cli_subprocess(run)
    assert_rejected(r.returncode, val, run)
    for i in range(1, 6):  # 2 blocks x 12 measured reps x 9 samples per session
        assert any(p.startswith(f"s{i}: 216 of 1080 planned measured samples missing") and "(1024, 1): 216" in p
                   for p in val["problems"]), val["problems"]


def test_decision_is_driven_by_the_spec_case_list_not_by_the_loaded_cases():
    hist = json.loads((EVAL / "m3_analysis.json").read_text())
    ok = {"status": "valid", "correctness_and_ab_acceptance_recomputed": True}
    d = m3_analyze.decide(hist["cases"], SPEC, ok)
    assert d["supports_adoption"] is False
    assert d["required_but_not_established"] == hist["decision"]["required_but_not_established"]
    # the historical analysis with the inconclusive required case removed (pre-fix: true)
    d = m3_analyze.decide({k: v for k, v in hist["cases"].items() if k != "1024:1"}, SPEC, ok)
    assert d["supports_adoption"] is False
    assert d["missing_from_analysis"] == [f"1024:1 {t} {s}" for t in ("host", "disk") for s in ("median_d", "mean_d")]
    assert m3_analyze.decide({}, SPEC, ok)["supports_adoption"] is False
    # the rule can pass: every verdict established faster, valid evidence
    fast = copy.deepcopy(hist["cases"])
    for c in fast.values():
        for t in c["tiers"].values():
            for st in ("median_d", "mean_d"):
                t["e2e"][st]["verdict"] = "batched_faster"
    assert m3_analyze.decide(fast, SPEC, ok)["supports_adoption"] is True
    for bad in ({"status": "invalid", "correctness_and_ab_acceptance_recomputed": True},
                {"status": "valid", "correctness_and_ab_acceptance_recomputed": False}, {}):
        assert m3_analyze.decide(fast, SPEC, bad)["supports_adoption"] is False


@pytest.mark.parametrize("which", ["missing", "unexpected", "unreadable"])
def test_missing_unexpected_or_unreadable_session_is_invalid(tmp_path, texts, which):
    if which == "missing":
        run, expect = make_run(tmp_path, texts, drop={"session_s3.json"}), "missing session file session_s3.json"
    elif which == "unexpected":
        run = make_run(tmp_path, texts, extra={"session_s6.json": texts["session_s5.json"]})
        expect = "unexpected session file session_s6.json"
    else:
        run = make_run(tmp_path, texts, extra={"session_s2.json": texts["session_s2.json"][:5000]})
        expect = "s2: unreadable session file"
    code, val = cli(run)
    assert_rejected(code, val, run, expect=[expect])


def _drop_one(rep):
    del rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)]


def _duplicate(rep):
    rep["samples"].append(copy.deepcopy(rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)]))


def _measured_as_warmup(rep):
    rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)]["warmup"] = True


def _warmup_as_measured(rep):
    rep["samples"][sample_index(rep, 3, 0, HOST_BATCHED_PLAIN)]["warmup"] = False


def _outside_plan(rep):
    s = copy.deepcopy(rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)])
    rep["samples"].append({**s, "rep": SPEC["warmup"] + SPEC["reps"]})


def _partner_fails_rules(rep):  # the batched half of a pair did not run the batched backend
    s = rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)]
    s["last_restore"] = {**s["last_restore"], "backend": "baseline"}


def _no_t_wall(rep):
    del rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)]["t_wall"]


@pytest.mark.parametrize("edit, expect", [
    (_drop_one, "s2: 1 of 1080 planned measured samples missing, so their pairs are incomplete"),
    (_duplicate, "s2: duplicate measured sample identity (block, rep, path, backend, mode) = (3, 5, 'host', 'batched', 'plain')"),
    (_measured_as_warmup, "warmup flag True for repetition 5"),
    (_warmup_as_measured, "warmup flag False for repetition 0"),
    (_outside_plan, "block/rep 3/14 outside the plan"),
    (_partner_fails_rules, "fails the validity rules, so its pair is incomplete"),
    (_no_t_wall, "fails the validity rules, so its pair is incomplete: ['t_wall']"),
], ids=["missing", "duplicate", "measured_flagged_warmup", "warmup_flagged_measured", "rep_outside_plan",
        "pair_partner_fails_rules", "missing_t_wall"])
def test_missing_duplicate_or_unpaired_samples_are_invalid(tmp_path, texts, edit, expect):
    code, val = cli(make_run(tmp_path, texts, edits={"session_s2.json": edit}))
    assert_rejected(code, val, tmp_path / "run", expect=[expect])


CASE = (1024, 1)
CHECK_EDITS = {
    # missing records or fields, while every stored summary flag still says pass
    "correctness_record_missing": (lambda r: r["correctness"].pop(corr_index(r, CASE)),
                                   ["s4: no correctness record for case (1024, 1)", "s4: stored correctness_all=True"]),
    "all_correctness_records_missing": (lambda r: r["correctness"].clear(),
                                        ["s4: no correctness record for case (256, 32)", "s4: stored correctness_all=True"]),
    "correctness_detail_missing": (
        lambda r: r["correctness"][corr_index(r, CASE)]["batched"].pop("strict_host_eq_resident_bitwise"),
        ["missing detailed fields ['strict_host_eq_resident_bitwise']", "stored strict_pass_all=True"]),
    "ab_record_missing": (lambda r: r["ab_checks"].pop(case_index(r["ab_checks"], CASE)),
                          ["s4: no A/B record for case (1024, 1)", "s4: stored correctness_all=True"]),
    "all_ab_records_missing": (lambda r: r["ab_checks"].clear(), ["s4: no A/B record for case (4096, 64)"]),
    "ab_detail_missing": (lambda r: r["ab_checks"][case_index(r["ab_checks"], CASE)]["host"].pop("greedy_tokens_equal"),
                          ["A/B (1024, 1) host: missing detailed fields ['greedy_tokens_equal']", "stored pass=True"]),
    "restore_memory_missing": (lambda r: r["restore_memory"].clear(), ["s4: no restore_memory record for case (1024, 1)"]),
    # failed detailed checks, while every stored summary flag still says pass
    "correctness_detail_failed": (
        lambda r: r["correctness"][corr_index(r, CASE)]["batched"]["restored_kv_bytes_equal"].update(disk=False),
        ["correctness (1024, 1) batched: stored strict_pass=True disagrees with the detailed checks (False)",
         "stored strict_pass_all=True", "s4: stored correctness_all=True"]),
    "ab_detail_failed": (lambda r: r["ab_checks"][case_index(r["ab_checks"], CASE)]["disk"].update(step_logits_bitwise_equal=False),
                         ["A/B (1024, 1): stored pass=True, detailed checks give False"]),
    "ab_compared_two_misses": (lambda r: r["ab_checks"][case_index(r["ab_checks"], CASE)]["host"].update(restore_len=0, sources=[]),
                               ["A/B (1024, 1): stored pass=True, detailed checks give False"]),
    # a stored summary that contradicts passing details is also stale evidence
    "summary_contradicts_details": (lambda r: r["meta"].update(correctness_all=False),
                                    ["s4: stored correctness_all=False"]),
}


@pytest.mark.parametrize("name", list(CHECK_EDITS))
def test_missing_or_failed_detailed_checks_with_stale_passing_summaries_are_invalid(tmp_path, texts, name):
    edit, expect = CHECK_EDITS[name]
    code, val = cli(make_run(tmp_path, texts, edits={"session_s4.json": edit}))
    assert_rejected(code, val, tmp_path / "run", expect=expect)
    if name.endswith("_failed") or name == "ab_compared_two_misses":
        assert val["correctness_failures"] and all(f.startswith("s4: ") for f in val["correctness_failures"])


def test_consistently_recorded_correctness_failure_is_reported_as_such(tmp_path, texts):
    """A detailed check failed and every summary says so: the evidence is complete and
    consistent, but gate (1) fails. Exit 4, never adoption, no performance analysis."""
    def edit(rep):
        c = rep["correctness"][corr_index(rep, CASE)]
        c["batched"]["strict_host_eq_resident_bitwise"] = False
        c["batched"]["strict_pass"] = False
        c["strict_pass_all"] = False
        rep["meta"]["correctness_all"] = False
    code, val = cli(make_run(tmp_path, texts, edits={"session_s4.json": edit}))
    assert_rejected(code, val, tmp_path / "run", status="correctness_failed")
    assert val["problems"] == []
    assert val["correctness_failures"] == ["s4: correctness (1024, 1) batched: failed ['host_eq_resident_bitwise']"]


def test_complete_valid_report_reproduces_the_historical_analysis(tmp_path, texts):
    run, out = make_run(tmp_path, texts, monitor=True), tmp_path / "out"
    code, val = cli(run, out)
    assert code == 0 and val["status"] == "valid" and val["problems"] == [] and val["correctness_failures"] == []
    assert val["correctness_and_ab_acceptance_recomputed"] is True and val["profile_session_present"] is True
    assert val["counts"] == {"measured_samples": {"expected": 5400, "present": 5400, "passed": 5400},
                             "block_warmup_samples": {"expected": 900, "present": 900, "passed": 900},
                             "process_warmup_samples": {"expected": 180, "present": 180, "passed": 180},
                             "correctness_checks": {"expected": 50, "present": 50, "passed": 50},
                             "ab_checks": {"expected": 50, "present": 50, "passed": 50}}
    new, hist = json.loads((out / "m3_analysis.json").read_text()), json.loads((EVAL / "m3_analysis.json").read_text())
    d = new["decision"]
    assert d["supports_adoption"] is False and d["evidence_status"] == "valid" and d["missing_from_analysis"] == []
    assert d["all_correctness_and_ab_checks_pass"] is True and d["established_regressions"] == []
    assert d["required_but_not_established"] == hist["decision"]["required_but_not_established"] == [
        f"1024:1 {t} {s}: inconclusive" for t in ("host", "disk") for s in ("median_d", "mean_d")]
    # every historical number and verdict is unchanged; the fix only adds keys
    for k in ("run_dir", "out_dir", "validation"):
        new.pop(k, None)
        hist.pop(k, None)
    for k in ("evidence_status", "spec", "missing_from_analysis"):
        new["decision"].pop(k)
    for s in new["sessions"].values():
        assert s.pop("correctness_all_recomputed") is True
    assert new == hist
    assert (out / "samples_chronological.csv").read_bytes() == (EVAL / "samples_chronological.csv").read_bytes()
    old_md = (EVAL / "m3_analysis.md").read_text().splitlines()
    new_md = (out / "m3_analysis.md").read_text().splitlines()
    added = [x for x in new_md if x not in old_md]
    assert [x for x in new_md if x in old_md] == old_md and len(added) == 1 and "**valid**" in added[0]


def test_refuses_to_overwrite_and_writes_nothing(tmp_path, texts):
    run = make_run(tmp_path, texts)
    shutil.copy2(EVAL / "m3_analysis.json", run / "m3_analysis.json")
    before = sorted(p.name for p in run.iterdir())
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        m3_analyze.main(["--run-dir", str(run)])
    assert sorted(p.name for p in run.iterdir()) == before
    with pytest.raises(SystemExit, match="does not exist"):
        m3_analyze.main(["--run-dir", str(tmp_path / "nonexistent")])
    assert not (tmp_path / "nonexistent").exists()


# ---- warmup evidence (VAL-006 follow-up): kept as evidence of the frozen procedure, never used
def _pw_index(rep, L, combo, nth=0):
    return [i for i, s in enumerate(rep["samples"]) if s.get("stage") == "process_warmup" and s["L"] == L and
            (s["path"], s["backend"], s["mode"]) == combo][nth]


def _remove(rep, i):
    del rep["samples"][i]
    rep["meta"]["n_samples"] = len(rep["samples"])  # keep the stored count consistent: only identity can tell


def _no_warmups(rep):  # the reproduction: every warmup row removed, only meta.n_samples adjusted
    rep["samples"] = [s for s in rep["samples"] if not s.get("warmup")]
    rep["meta"]["n_samples"] = len(rep["samples"])


def _block_warmup_substituted(rep):  # total row count unchanged
    rep["samples"][sample_index(rep, 3, 1, HOST_BATCHED_PLAIN)] = copy.deepcopy(
        rep["samples"][sample_index(rep, 3, 0, HOST_BATCHED_PLAIN)])


def _process_warmup_substituted(rep):  # total row count unchanged
    rep["samples"][_pw_index(rep, 256, ("disk", "batched", "synced"))] = copy.deepcopy(
        rep["samples"][_pw_index(rep, 256, ("host", "batched", "synced"))])


def _process_warmup_replaced_by_measured_copy(rep):  # count unchanged; a measured row relabelled
    s = copy.deepcopy(rep["samples"][sample_index(rep, 3, 5, HOST_BATCHED_PLAIN)])
    for k in ("block", "rep", "order_pos"):
        s.pop(k)
    rep["samples"][_pw_index(rep, 4096, ("full", "none", "plain"), 1)] = {**s, "warmup": True, "stage": "process_warmup"}


def _process_warmup_not_flagged(rep):
    rep["samples"][_pw_index(rep, 4096, HOST_BATCHED_PLAIN)]["warmup"] = False


def _process_warmup_as_block_stage(rep):
    rep["samples"][_pw_index(rep, 4096, HOST_BATCHED_PLAIN)]["stage"] = "m3"


def _block_warmup_as_process_stage(rep):
    rep["samples"][sample_index(rep, 3, 0, HOST_BATCHED_PLAIN)]["stage"] = "process_warmup"


def _warmup_unknown_stage(rep):
    rep["samples"][sample_index(rep, 3, 0, HOST_BATCHED_PLAIN)]["stage"] = "warmup"


def _process_warmup_after_blocks(rep):
    rep["samples"].append(rep["samples"].pop(_pw_index(rep, 256, HOST_BATCHED_PLAIN)))


def _block_warmup_malformed(rep):
    del rep["samples"][sample_index(rep, 3, 0, HOST_BATCHED_PLAIN)]["t_wall"]


WARMUP_EDITS = {
    "all_warmups_removed_n_samples_adjusted": (
        _no_warmups, True, ["s1: 180 of 180 planned block-warmup samples missing",
                            "s5: process-warmup records differ from the plan (36 = 2 cases x 9 combinations x 2)"]),
    "one_block_warmup_missing": (lambda r: _remove(r, sample_index(r, 3, 1, HOST_BATCHED_PLAIN)), False,
                                 ["s2: 1 of 180 planned block-warmup samples missing (by case {(1024, 1): 1}; "
                                  "first (3, 1, 'host', 'batched', 'plain'))"]),
    "one_process_warmup_missing": (lambda r: _remove(r, _pw_index(r, 4096, ("full", "none", "plain"))), False,
                                   ["missing {((4096, 64), ('full', 'none', 'plain')): '1/2'}; surplus none"]),
    "block_warmup_substituted_same_count": (
        _block_warmup_substituted, False,
        ["s2: duplicate block-warmup sample identity (block, rep, path, backend, mode) = (3, 0, 'host', 'batched', 'plain')",
         "s2: 1 of 180 planned block-warmup samples missing"]),
    "process_warmup_substituted_same_count": (
        _process_warmup_substituted, False,
        ["missing {((256, 32), ('disk', 'batched', 'synced')): '1/2'}; "
         "surplus {((256, 32), ('host', 'batched', 'synced')): '3/2'}"]),
    "process_warmup_replaced_by_relabelled_measured_row": (
        _process_warmup_replaced_by_measured_copy, False,
        ["process-warmup session/case/combination 's2' ((1024, 1), ('host', 'batched', 'plain')) not in the spec",
         "missing {((4096, 64), ('full', 'none', 'plain')): '1/2'}"]),
    "process_warmup_flagged_measured": (_process_warmup_not_flagged, False,
                                        ["process-warmup sample not flagged as warmup (False)"]),
    "process_warmup_with_block_stage": (_process_warmup_as_block_stage, False, ["block/rep None/None outside the plan"]),
    "block_warmup_with_process_stage": (_block_warmup_as_process_stage, False,
                                        ["process-warmup sample carries block/rep/order_pos fields",
                                         "1 of 180 planned block-warmup samples missing"]),
    "warmup_with_unknown_stage": (_warmup_unknown_stage, False, ["unknown stage 'warmup'"]),
    "process_warmup_after_block_samples": (_process_warmup_after_blocks, False,
                                           ["comes after the first block sample"]),
    "block_warmup_malformed": (_block_warmup_malformed, False,
                               ["malformed block-warmup record (3, 0, 'host', 'batched', 'plain'): ['t_wall']"]),
}


@pytest.mark.parametrize("name", list(WARMUP_EDITS))
def test_missing_substituted_or_misflagged_warmup_evidence_is_invalid(tmp_path, texts, name):
    edit, all_sessions, expect = WARMUP_EDITS[name]
    edits = {f: edit for f in EVAL_FILES} if all_sessions else {"session_s2.json": edit}
    code, val = cli(make_run(tmp_path, texts, edits=edits))
    assert_rejected(code, val, tmp_path / "run", expect=expect)
    assert val["correctness_failures"] == []  # an evidence defect, not a failed correctness check


def test_all_warmups_removed_is_rejected_by_the_real_cli_subprocess(tmp_path, texts):
    """The reported reproduction through the real CLI. Before the fix: status valid, exit 0."""
    run = make_run(tmp_path, texts, edits={f: _no_warmups for f in EVAL_FILES}, drop={"session_prof.json"})
    r, val = cli_subprocess(run)
    assert_rejected(r.returncode, val, run)
    assert r.returncode == 3 and val["counts"]["block_warmup_samples"]["present"] == 0
    assert val["counts"]["process_warmup_samples"]["present"] == 0
    assert val["counts"]["measured_samples"]["passed"] == 5400  # measured evidence alone is not enough


def test_warmup_timings_never_enter_the_statistics(tmp_path, texts):
    """Well-formed warmups with 100x inflated timings: the evidence stays valid, and every
    statistic, verdict and the decision equal the historical analysis."""
    def inflate(rep):
        for s in rep["samples"]:
            if s.get("warmup"):
                s["e2e_ns"] *= 100
                if s["mode"] == "synced":
                    s["phases_ns"] = {k: v * 100 for k, v in s["phases_ns"].items()}
    run, out = make_run(tmp_path, texts, edits={f: inflate for f in EVAL_FILES}, monitor=True), tmp_path / "out"
    code, val = cli(run, out)
    assert code == 0 and val["status"] == "valid"
    new, hist = json.loads((out / "m3_analysis.json").read_text()), json.loads((EVAL / "m3_analysis.json").read_text())
    assert new["cases"] == hist["cases"]
    assert {k: new["decision"][k] for k in hist["decision"]} == hist["decision"]
    assert new["decision"]["supports_adoption"] is False
