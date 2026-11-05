"""The M4-B batch semantics with dummy child processes (no GPU, no model).

M4-B reuses kvlab.m3_confirm.run_batch unchanged (scripts/m4_eval.py run). These tests run it with
an M4-shaped manifest and check the deadline, stopping, interruption and no-retry rules, and that
kvlab.m3_confirm.batch_problems (used by kvlab.m4_validate) rejects every failed execution."""

import json
import os
import sys
from pathlib import Path

import pytest

from kvlab import m3_confirm as mc
from kvlab import m4_freeze as mf
from kvlab import m4_spec as ms

ROOT = Path(__file__).resolve().parent.parent


def manifest(sessions, budget_s=60):
    cmds = [{"session": sid, "argv": [sys.executable, "-c", code], "env": {}} for sid, code in sessions]
    return {"experiment": "M4_DUMMY", "run_dir": "run", "budget_s": budget_s, "commands": cmds, "frozen_unix": 0.0}


def ok(sid, rc=0, report=True, extra=""):
    w = f"open('run/session_{sid}.json', 'w').write('{{}}');" if report else ""
    return f"import os, sys, time; {w} {extra} sys.exit({rc})"


def run(tmp_path, m):
    rec = mc.run_batch(tmp_path, m, "a" * 64, verification={"problems": []}, log=lambda *_: None, grace_s=2.0)
    probs = mc.batch_problems(rec, m, "a" * 64, tmp_path / "run", {"name": "M4_DUMMY", "budget_s": m["budget_s"]})
    return rec, probs


def test_successful_batch_has_no_problems(tmp_path):
    rec, probs = run(tmp_path, manifest([("e01", ok("e01")), ("e02", ok("e02")), ("e03", ok("e03"))]))
    assert rec["execution_status"] == "succeeded" and mc.exit_code(rec) == 0 and probs == []
    assert [e["session"] for e in rec["sessions"]] == ["e01", "e02", "e03"]


def test_nonzero_exit_stops_without_retry_or_replacement(tmp_path):
    rec, probs = run(tmp_path, manifest([("e01", ok("e01")), ("e02", ok("e02", rc=1)), ("e03", ok("e03"))]))
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3
    assert rec["stop"]["reason"] == "nonzero_exit" and rec["stop"]["session"] == "e02"
    assert [e["session"] for e in rec["sessions"]] == ["e01", "e02"] and rec["not_launched"] == ["e03"]
    assert not (tmp_path / "run" / "session_e03.json").exists()
    assert any("never launched" in p for p in probs)


def test_deadline_kills_the_running_session_and_is_never_success(tmp_path):
    m = manifest([("e01", ok("e01", extra="time.sleep(30);")), ("e02", ok("e02"))], budget_s=1.5)
    rec, probs = run(tmp_path, m)
    assert rec["execution_status"] == "failed" and rec["stop"]["reason"] == "deadline"
    e = rec["sessions"][0]
    assert e["killed"]["group_gone"] is True and rec["not_launched"] == ["e02"]
    assert probs and mc.exit_code(rec) == 3


def test_operator_signal_stops_the_batch_and_cleans_up(tmp_path):
    code = ok("e01", report=False, extra="import signal; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30);")
    rec, probs = run(tmp_path, manifest([("e01", code), ("e02", ok("e02"))]))
    assert rec["stop"]["reason"] == "operator_signal" and rec["sessions"][0]["killed"]["group_gone"] is True
    assert rec["execution_status"] == "failed" and rec["not_launched"] == ["e02"] and probs


def test_exit_zero_without_a_report_stops_the_batch(tmp_path):
    rec, probs = run(tmp_path, manifest([("e01", ok("e01", report=False)), ("e02", ok("e02"))]))
    assert rec["stop"]["reason"] == "missing_report" and rec["not_launched"] == ["e02"] and probs


def test_m4_commands_and_the_freeze_refuse_a_draft_protocol(tmp_path):
    spec = ms.M4_EVAL_V1
    cmds = ms.commands_for(spec)
    assert [c["session"] for c in cmds] == [f"e{i:02d}" for i in range(1, 11)]
    assert cmds[0]["argv"][1:] == ["scripts/m4_measure.py", "--spec", "M4_EVAL_V1", "--out-dir",
                                   "results/m4/M4_EVAL_V1_run", "--session", "e01", "--seed", "44001"]
    assert "DRAFT" in (ROOT / spec["protocol_doc"]).read_text().splitlines()[0]
    with pytest.raises(mc.Refused, match="DRAFT"):
        mf.build_manifest(ROOT, spec, stack={"python": "x"})
    assert not (ROOT / mf.manifest_path(spec)).exists()  # nothing written


def test_manifest_problems_detect_changed_analysis_bytes():
    spec = ms.M4_EVAL_V1
    m, _diff = mf.build_manifest(ROOT, spec, stack={"python": "x"}, require_frozen_protocol=False, now=1.0e9)
    assert mf.manifest_problems(m, spec, ROOT) == []
    bad = json.loads(json.dumps(m))
    bad["analysis_files_sha256"]["kvlab/m4_stats.py"] = "0" * 64
    assert any("kvlab/m4_stats.py differs" in p for p in mf.manifest_problems(bad, spec, ROOT))
    bad = {**m, "spec_sha256": "0" * 64}
    assert any("spec_sha256" in p for p in mf.manifest_problems(bad, spec, ROOT))
    assert "kvlab/m4_freeze.py" in m["analysis_files_sha256"]  # the analysis-time check covers its own code
    bad = json.loads(json.dumps(m))
    bad["analysis_files_sha256"]["kvlab/m4_freeze.py"] = "0" * 64
    assert any("kvlab/m4_freeze.py differs" in p for p in mf.manifest_problems(bad, spec, ROOT))


def _gate():
    sys.path.insert(0, str(ROOT / "scripts"))
    import m4_gate
    return m4_gate


def test_gate_wrapper_passes_fails_fast_and_enforces_one_deadline(tmp_path):
    g = _gate()
    py = sys.executable
    ok3 = [(n, [py, "-c", "import sys; sys.exit(0)"]) for n in ("m1", "m3", "m4")]
    rec = g.run_gate(ok3, 30, tmp_path / "pass")
    assert rec["status"] == "PASS" and [e["returncode"] for e in rec["suites"]] == [0, 0, 0]
    assert json.loads((tmp_path / "pass" / "gate_record.json").read_text())["status"] == "PASS"
    fail = [ok3[0], ("m3", [py, "-c", "import sys; sys.exit(1)"]), ok3[2]]
    rec = g.run_gate(fail, 30, tmp_path / "fail")
    assert rec["status"] == "FAIL" and rec["stop"]["suite"] == "m3" and [e["suite"] for e in rec["suites"]] == ["m1", "m3"]
    slow = [ok3[0], ("m3", [py, "-c", "import time; time.sleep(30)"]), ok3[2]]
    rec = g.run_gate(slow, 1.5, tmp_path / "slow", grace_s=2.0)
    assert rec["status"] == "FAIL" and rec["stop"]["reason"] == "deadline" and rec["suites"][-1]["killed"]["group_gone"]
    assert [e["suite"] for e in rec["suites"]] == ["m1", "m3"]
    with pytest.raises(FileExistsError):
        g.run_gate(ok3, 30, tmp_path / "pass")
    cmds = g.suites("PY", tmp_path / "bt")
    assert [c[0] for c in cmds] == ["m1", "m3", "m4"] and all("-x" in c[1] for c in cmds)
    assert cmds[2][1][-3:] == ["-m", "gpu and m4", "tests/gpu/test_m4_replay.py"]


def test_gate_wrapper_cleans_up_and_records_fail_when_interrupted(tmp_path):
    """Final-review P2 (post-review correction): a stop signal to the wrapper while a suite runs.
    The suite (which signals the wrapper, then sleeps) is terminated and reaped, nothing further
    runs, and a FAIL record is written."""
    g = _gate()
    py = sys.executable
    code = "import os, signal, time; os.kill(os.getppid(), signal.SIGINT); time.sleep(30)"
    cmds = [("m1", [py, "-c", code]), ("m3", [py, "-c", "import sys; sys.exit(0)"])]
    rec = g.run_gate(cmds, 30, tmp_path / "int", grace_s=2.0)
    assert rec["status"] == "FAIL" and rec["stop"] == {"reason": "operator_signal", "signal": "SIGINT"}
    assert [e["suite"] for e in rec["suites"]] == ["m1"] and rec["suites"][0]["killed"]["group_gone"] is True
    assert json.loads((tmp_path / "int" / "gate_record.json").read_text())["status"] == "FAIL"
    import signal as _s
    assert _s.getsignal(_s.SIGINT) is _s.default_int_handler  # handlers restored


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_no_owned_process_group_survives_an_interrupted_gate(tmp_path, sig):
    """Final-review P2: after the wrapper returns from a stop signal, the suite's process group no
    longer exists (probed with killpg(pgid, 0)); nothing further was launched; the record says FAIL."""
    import signal as _s
    g = _gate()
    py = sys.executable
    code = f"import os, signal, time; os.kill(os.getppid(), signal.{sig}); time.sleep(30)"
    launched = []
    real = g.subprocess.Popen

    def track(*a, **kw):
        p = real(*a, **kw)
        launched.append(p)
        return p
    g.subprocess.Popen = track
    try:
        rec = g.run_gate([("m1", [py, "-c", code]), ("m3", [py, "-c", "import sys; sys.exit(0)"])], 30,
                         tmp_path / "g", grace_s=2.0)
    finally:
        g.subprocess.Popen = real
    assert rec["status"] == "FAIL" and rec["stop"] == {"reason": "operator_signal", "signal": sig}
    assert len(launched) == 1 and launched[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)  # the owned group is gone
    assert _s.getsignal(_s.SIGINT) is _s.default_int_handler and _s.getsignal(_s.SIGTERM) == _s.SIG_DFL


SPAWN = ("import subprocess, sys; "
         "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); sys.exit(0)")  # same group


def test_leader_exit_does_not_release_a_group_with_live_descendants(tmp_path):
    """Correction-pass P2: the leader exits 0 while its descendant still runs in its group. The gate
    cleans up the GROUP, records a FAIL ("leftover_processes") and runs nothing further."""
    g = _gate()
    py = sys.executable
    launched, real = [], g.subprocess.Popen

    def track(*a, **kw):
        p = real(*a, **kw)
        launched.append(p)
        return p
    g.subprocess.Popen = track
    try:
        rec = g.run_gate([("m1", [py, "-c", SPAWN]), ("m3", [py, "-c", "import sys; sys.exit(0)"])], 30,
                         tmp_path / "g", grace_s=2.0)
    finally:
        g.subprocess.Popen = real
    assert rec["status"] == "FAIL" and rec["stop"] == {"reason": "leftover_processes", "suite": "m1"}
    e = rec["suites"][0]
    assert e["returncode"] == 0 and e["group_cleanup"]["sigterm"] is True and e["group_cleanup"]["group_gone"]
    assert [x["suite"] for x in rec["suites"]] == ["m1"]
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)


def test_wrapper_error_after_the_leader_exits_still_cleans_up_the_group(tmp_path):
    """The reviewer's case: an error in the wrapper after the leader exited, with a descendant alive.
    The error is re-raised; the record says FAIL (wrapper_error); the group is gone."""
    g = _gate()
    py = sys.executable
    launched, real = [], g.subprocess.Popen

    def track(*a, **kw):
        p = real(*a, **kw)
        real_wait = p.wait

        def wait(timeout=None):
            real_wait(timeout=timeout)
            raise RuntimeError("injected wrapper error after the leader exited")
        p.wait = wait
        launched.append(p)
        return p
    g.subprocess.Popen = track
    try:
        with pytest.raises(RuntimeError, match="injected"):
            g.run_gate([("m1", [py, "-c", SPAWN])], 30, tmp_path / "g", grace_s=2.0)
    finally:
        g.subprocess.Popen = real
    rec = json.loads((tmp_path / "g" / "gate_record.json").read_text())
    assert rec["status"] == "FAIL" and rec["stop"]["reason"] == "wrapper_error"
    assert rec["suites"][0]["killed"]["group_gone"] is True
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)


def test_gate_restores_the_callers_signal_mask_and_children_start_unmasked(tmp_path):
    import signal as _s
    g = _gate()
    py = sys.executable
    out = tmp_path / "mask.txt"
    code = ("import sys; line = [l for l in open('/proc/self/status') if l.startswith('SigBlk:')][0]; "
            f"open({str(out)!r}, 'w').write(line.split()[1]); sys.exit(0)")
    before = _s.pthread_sigmask(_s.SIG_BLOCK, [_s.SIGTERM])
    try:
        rec = g.run_gate([("m1", [py, "-c", code])], 30, tmp_path / "g", grace_s=2.0)
        assert _s.SIGTERM in _s.pthread_sigmask(_s.SIG_BLOCK, ())  # the caller's block is kept
    finally:
        _s.pthread_sigmask(_s.SIG_SETMASK, before)
    assert rec["status"] == "PASS" and int(out.read_text(), 16) == 0  # the suite started with an empty mask
