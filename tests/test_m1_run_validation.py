"""Validation/summary path of scripts/m1_run.py (CPU only; no model is loaded).

These tests call the production functions strict_checks / validate_report /
summarize / exit_code, and run the real `--validate` CLI in a subprocess to
check the process exit code. Records are built by strict_checks from small CPU
tensors, in the same form Run.request writes them.
Regression for BUGLOG VAL-004: the ENOSPC request had no strict record, the
summary reported "+ 1" requests, and an edited forward-call count still passed.
"""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("m1_run", ROOT / "scripts" / "m1_run.py")
m1_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1_run)
STEPS = m1_run.STEPS

FAULT_MISS = {name: reason for name, (_, reason) in m1_run.EXPECTED_FAULT_MISS.items()}


def steps(seed=0, n=STEPS):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(16, generator=g) for _ in range(n)]


def record(label, L=0, miss_reason=None, tokens=None, step_logits=None, ref_tokens=None, ref_steps=None,
           forward_calls=STEPS):
    tokens = list(range(STEPS)) if tokens is None else tokens
    ref_tokens = list(tokens) if ref_tokens is None else ref_tokens
    step_logits = steps() if step_logits is None else step_logits
    ref_steps = [x.clone() for x in step_logits] if ref_steps is None else ref_steps
    kv = {"tensors_checked": 56, "mismatched": 0, "all_bytes_equal": True} if L else {}
    strict = m1_run.strict_checks(tokens, step_logits, ref_tokens, ref_steps, L, kv, forward_calls)
    return {"label": label, "N": 600, "eligible_L": 512, "L": L, "restore_used": L > 0, "sources": [],
            "miss_reason": miss_reason, "stored_chunks": [], **strict,
            "report_only_vs_full_recompute": {"greedy": {"identical": True, "matching_positions": STEPS}}}


def valid_report() -> dict:
    sc = {}
    for name, n in m1_run.EXPECTED_RECORDS.items():
        recs = [record(f"{name}-{i}", L=256 if i == 0 and name == "sequence" else 0) for i in range(n)]
        if name in FAULT_MISS:
            recs[1]["miss_reason"] = FAULT_MISS[name]
        sc[name] = {"records": recs, "counters": {"spill_failures": {}}}
    sc["fault_enospc"].update(enospc_injections=2, dir_after=[0, []],
                              counters={"spill_failures": {"write_oserror:ENOSPC": 2}})
    return {"scenarios": sc}


def cli_validate(tmp_path, report) -> subprocess.CompletedProcess:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "m1_run.py"), "--validate", str(path)],
                          capture_output=True, text=True, timeout=300)


def assert_rejected(report, fragment):
    summary = m1_run.summarize(report)
    assert summary["acceptance"] is False and m1_run.exit_code(summary) == 1
    assert any(fragment in p for p in summary["problems"]), summary["problems"]
    return summary


def test_valid_complete_report_passes_in_memory_and_after_json_round_trip(tmp_path):
    report = valid_report()
    for form in (report, json.loads(json.dumps(report))):
        s = m1_run.summarize(form)
        assert s["problems"] == [] and s["acceptance"] is True and m1_run.exit_code(s) == 0
        assert s["requests"] == s["strict_records"] == s["expected_requests"] == 19  # derived from records
        assert s["strict_pass_all"] is True
    r = cli_validate(tmp_path, report)
    assert r.returncode == 0, r.stdout + r.stderr


def test_incorrect_enospc_forward_call_count_fails_including_exit_code(tmp_path):
    report = valid_report()
    rec = report["scenarios"]["fault_enospc"]["records"][0]
    rec["forward_calls"] = 800  # the independent review's edit; strict_pass is left True on purpose
    assert rec["strict_pass"] is True
    assert_rejected(report, "forward_calls=800")
    r = cli_validate(tmp_path, report)
    assert r.returncode == 1 and "forward_calls=800" in r.stdout
    # Produced by the production check, not only by the validator's re-derivation:
    assert m1_run.strict_checks(list(range(STEPS)), steps(), list(range(STEPS)), steps(), 0, {}, 9)["strict_pass"] \
        is False


@pytest.mark.parametrize("kind", ["token", "step_logit_value", "step_logit_shape"])
def test_token_or_step_logit_mismatch_fails(kind):
    ref_steps = steps()
    test_steps = [x.clone() for x in ref_steps]
    tokens = list(range(STEPS))
    if kind == "token":
        tokens[STEPS - 1] += 1
    elif kind == "step_logit_value":
        test_steps[3][5] = torch.nextafter(test_steps[3][5], torch.tensor(float("inf")))  # one float32 ulp
    else:
        test_steps[3] = test_steps[3][:-1]
    strict = m1_run.strict_checks(tokens, test_steps, list(range(STEPS)), ref_steps, 0, {}, STEPS)
    assert strict["strict_pass"] is False and strict["lengths_ok"] is True
    report = valid_report()
    report["scenarios"]["sequence"]["records"][4] = record(
        "mismatch", tokens=tokens, step_logits=test_steps, ref_tokens=list(range(STEPS)), ref_steps=ref_steps)
    assert_rejected(report, "sequence[4] 'mismatch': strict checks not true")


@pytest.mark.parametrize("which", ["both_tokens", "engine_logits", "ref_logits", "both_logits"])
def test_truncated_token_or_logit_sequences_fail(which):
    full, toks = steps(), list(range(STEPS))
    args = dict(tokens=toks, step_logits=full, ref_tokens=list(toks), ref_steps=[x.clone() for x in full])
    if which == "both_tokens":  # equal but short: plain == would accept it
        args["tokens"], args["ref_tokens"] = toks[:-1], toks[:-1]
    elif which == "engine_logits":  # zip() alone would compare only the common prefix
        args["step_logits"] = full[:-2]
    elif which == "ref_logits":
        args["ref_steps"] = args["ref_steps"][:-2]
    else:
        args["step_logits"], args["ref_steps"] = full[:-1], args["ref_steps"][:-1]
    strict = m1_run.strict_checks(args["tokens"], args["step_logits"], args["ref_tokens"], args["ref_steps"],
                                  0, {}, STEPS)
    assert strict["lengths_ok"] is False and strict["strict_pass"] is False
    # Each comparison must refuse a truncated sequence on its own (not only via lengths_ok):
    field = "strict_tokens_equal_resident" if which == "both_tokens" else "strict_all_step_logits_bitwise_equal_resident"
    assert strict[field] is False
    report = valid_report()
    report["scenarios"]["fault_enospc"]["records"][0] = record("truncated", **args)
    assert_rejected(report, "expected 8")


def test_missing_enospc_record_or_scenario_fails(tmp_path):
    report = valid_report()
    report["scenarios"]["fault_enospc"]["records"] = []
    s = assert_rejected(report, "fault_enospc: 0 request records, expected 1")
    assert s["requests"] == 18  # derived, no "+ 1"
    assert cli_validate(tmp_path, report).returncode == 1
    report = valid_report()
    del report["scenarios"]["fault_enospc"]
    assert_rejected(report, "fault_enospc: scenario missing")
    assert_rejected({"scenarios": {}}, "sequence: scenario missing")


@pytest.mark.parametrize("scenario", sorted(FAULT_MISS))
def test_fault_scenario_without_its_miss_fails(scenario):
    report = valid_report()
    report["scenarios"][scenario]["records"][1]["miss_reason"] = "absent"
    assert_rejected(report, f"{scenario}: intended fault not observed")


@pytest.mark.parametrize("edit", ["no_injection", "no_spill_failure", "files_left"])
def test_enospc_scenario_without_the_injected_fault_fails(edit):
    report = valid_report()
    sc = report["scenarios"]["fault_enospc"]
    if edit == "no_injection":
        sc["enospc_injections"] = 0
    elif edit == "no_spill_failure":
        sc["counters"]["spill_failures"] = {}
    else:
        sc["dir_after"] = [7342121, ["a" * 64 + ".kvc"]]
    assert_rejected(report, "fault_enospc")


def test_restore_without_kv_byte_check_fails():
    report = valid_report()
    rec = report["scenarios"]["sequence"]["records"][0]
    assert rec["L"] == 256
    rec["strict_restored_kv_bytes"] = None
    assert_rejected(report, "without a passing K/V byte check")
    strict = m1_run.strict_checks(list(range(STEPS)), steps(), list(range(STEPS)), steps(), 256, {}, STEPS)
    assert strict["restored_kv_ok"] is False and strict["strict_pass"] is False


def test_historical_evidence_lacks_the_enospc_strict_record():
    """The gap, shown on preserved raw evidence (read only): the M1.1 run reported
    19 requests but has only 18 request records, none for ENOSPC."""
    old = json.loads((ROOT / "results/m1_1/m1_run_20261103T112930.json").read_text())
    assert old["summary"]["requests"] == 19 and old["summary"]["strict_pass_all"] is True
    assert sum(len(sc.get("records", [])) for sc in old["scenarios"].values()) == 18
    assert old["scenarios"]["fault_enospc"]["records"] == []  # finish() wrote an empty list; no strict record
    edited = copy.deepcopy(old)
    edited["scenarios"]["fault_enospc"]["forward_calls_for_request"] = 800
    # The old summary rule (all strict_pass over records) is blind to that edit ...
    assert all(rec["strict_pass"] for sc in edited["scenarios"].values() for rec in sc.get("records", []))
    # ... the corrected validation is not.
    s = m1_run.summarize(edited)
    assert s["acceptance"] is False and s["requests"] == 18
    assert any("fault_enospc: 0 request records" in p for p in s["problems"])
