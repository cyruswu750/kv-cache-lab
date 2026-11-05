"""scripts/m4_gate.py through its real subprocess path with REAL pytest children (CPU only).

The gate's first GPU attempt failed because the wrapper gave each suite
`--basetemp=<out>/basetemp/<suite>` but never created `<out>/basetemp`. pytest creates an explicit
basetemp with a non-recursive mkdir, so the first test requesting `tmp_path` errored at setup. The
earlier wrapper tests used `python -c` children and could not see this.

Here the commands come from the production path-building (`gate_commands`). Only the pytest
selection (`-m "gpu and ..."` and the GPU test path) is replaced by tiny synthetic CPU suites, so
no GPU test is collected; every generated `--basetemp` argument is kept and checked. The synthetic
suites live in their own directory with their own pytest.ini, so the repository's configuration is
not involved. Nothing here touches a GPU or loads a model.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import m4_gate as g  # noqa: E402

OK = '''
def test_{name}(tmp_path, tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    assert tmp_path.is_relative_to(base)
    (tmp_path / "marker.json").write_text(__import__("json").dumps({{"suite": "{name}", "basetemp": str(base),
                                                                    "tmp_path": str(tmp_path),
                                                                    "env": __import__("os").environ.get("{env}")}}))
'''
FAIL = '''
def test_{name}(tmp_path):
    (tmp_path / "marker.json").write_text("{{}}")
    assert False, "synthetic failing suite"
'''
LINGER = '''
import subprocess, sys
def test_{name}(tmp_path):
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # left running in the group
'''


def write_suites(d: Path, kinds: dict) -> Path:
    d.mkdir(parents=True)
    (d / "pytest.ini").write_text("[pytest]\n")
    for name in ("m1", "m3", "m4"):
        (d / f"test_{name}.py").write_text({"ok": OK, "fail": FAIL, "linger": LINGER}[kinds.get(name, "ok")]
                                           .format(name=name, env=g.GATE_OUT_ENV))
    return d


def commands(out: Path, suite_dir: Path) -> list:
    """Production commands for `out`, with only the pytest selection replaced."""
    picked = []
    for name, argv in g.gate_commands(out, sys.executable):
        bt = [a for a in argv if a.startswith("--basetemp=")]
        assert bt == [f"--basetemp={out / g.BASETEMP_DIR / name}"], bt  # the production layout, kept as is
        assert Path(bt[0].split("=", 1)[1]) != out                        # never the output directory itself
        sel = argv.index("-m", argv.index("-p") + 2)                      # the "-m gpu and ..." selection
        assert argv[sel + 1].startswith("gpu and ")
        picked.append((name, argv[:sel] + [str(suite_dir / f"test_{name}.py")]))
    return picked


def markers(out: Path, name: str) -> list:
    return [json.loads(p.read_text()) for p in (out / g.BASETEMP_DIR / name).rglob("marker.json")]


def test_fresh_output_runs_three_real_pytest_suites_with_separate_tmp_dirs(tmp_path):
    suite = write_suites(tmp_path / "suite", {})
    out = tmp_path / "out"
    rec = g.run_gate(commands(out, suite), 120, out, cwd=suite, grace_s=2.0)
    assert rec["status"] == "PASS" and rec["stop"] is None, rec
    assert [(e["suite"], e["returncode"]) for e in rec["suites"]] == [("m1", 0), ("m3", 0), ("m4", 0)]
    bases = set()
    for name in ("m1", "m3", "m4"):
        m = markers(out, name)
        assert len(m) == 1 and m[0]["suite"] == name, (name, m)
        assert Path(m[0]["basetemp"]) == (out / g.BASETEMP_DIR / name).resolve()
        assert m[0]["env"] == str(out.resolve())                # every suite learns the fresh output directory
        bases.add(m[0]["basetemp"])
        log = (out / f"{name}.log").read_text()                 # the logs survive pytest's tmp handling
        assert "1 passed" in log and "FileNotFoundError" not in log, log[-800:]
        e = next(x for x in rec["suites"] if x["suite"] == name)
        assert e["group_cleanup"]["group_gone"] and not e["group_cleanup"]["sigterm"]
    assert len(bases) == 3                                       # three separate suite temp directories
    assert json.loads((out / "gate_record.json").read_text())["status"] == "PASS"
    assert rec["env_added"] == {g.GATE_OUT_ENV: str(out.resolve())}
    assert sorted(p.name for p in out.iterdir()) == ["basetemp", "gate_record.json", "m1.log", "m3.log", "m4.log"]


def test_existing_output_directory_is_refused_without_altering_it(tmp_path):
    suite = write_suites(tmp_path / "suite", {})
    out = tmp_path / "out"
    (out / "basetemp" / "m1").mkdir(parents=True)
    (out / "gate_record.json").write_text('{"previous": "attempt"}')
    (out / "basetemp" / "m1" / "keep.txt").write_text("evidence")
    before = {str(p.relative_to(out)): (p.read_bytes() if p.is_file() else None) for p in out.rglob("*")}
    with pytest.raises(FileExistsError):
        g.run_gate(commands(out, suite), 120, out, cwd=suite, grace_s=2.0)
    after = {str(p.relative_to(out)): (p.read_bytes() if p.is_file() else None) for p in out.rglob("*")}
    assert after == before                                       # nothing launched, nothing changed


def test_a_failing_real_pytest_suite_stops_the_later_suites(tmp_path):
    suite = write_suites(tmp_path / "suite", {"m3": "fail"})
    out = tmp_path / "out"
    rec = g.run_gate(commands(out, suite), 120, out, cwd=suite, grace_s=2.0)
    assert rec["status"] == "FAIL" and rec["stop"] == {"reason": "failure", "suite": "m3", "returncode": 1}
    assert [e["suite"] for e in rec["suites"]] == ["m1", "m3"]
    assert not (out / "m4.log").exists() and not (out / g.BASETEMP_DIR / "m4").exists()
    assert "1 failed" in (out / "m3.log").read_text() and "1 passed" in (out / "m1.log").read_text()
    assert json.loads((out / "gate_record.json").read_text())["stop"]["suite"] == "m3"


def test_a_real_pytest_suite_leaving_a_descendant_is_cleaned_up_and_fails(tmp_path):
    suite = write_suites(tmp_path / "suite", {"m1": "linger"})
    out = tmp_path / "out"
    launched, real = [], g.subprocess.Popen

    def track(*a, **kw):
        p = real(*a, **kw)
        launched.append(p)
        return p
    g.subprocess.Popen = track
    try:
        rec = g.run_gate(commands(out, suite), 120, out, cwd=suite, grace_s=2.0)
    finally:
        g.subprocess.Popen = real
    assert rec["status"] == "FAIL" and rec["stop"] == {"reason": "leftover_processes", "suite": "m1"}
    assert rec["suites"][0]["returncode"] == 0 and rec["suites"][0]["group_cleanup"]["group_gone"]
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)
    assert len(launched) == 1 and not (out / "m3.log").exists()
