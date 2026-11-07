#!/usr/bin/env python3
"""The bounded, fail-fast M4-B GPU correctness gate (docs/M4B_PROTOCOL.md §9). Each execution runs
exclusively on the GPU, once per reviewed version, into a new output directory.

  HF_HOME=<weights cache> HF_HUB_OFFLINE=1 .venv/bin/python scripts/m4_gate.py --out-dir <new dir>

Runs, in order, under ONE monotonic deadline of kvlab.m4_spec.GATE_CAP_S (900 s):
  1. pytest -x -m "gpu and m1"   (existing M1 tiered suite, both restore backends)
  2. pytest -x -m "gpu and m3"   (existing M3 batched-restore checks)
  3. pytest -x -m "gpu and m4" tests/gpu/test_m4_replay.py   (the new strict replays)
Stops at the first failing suite (nothing further runs) or at the deadline (the running suite's
process group is terminated and reaped). Writes <out-dir>/gate_record.json with the commands, exit
codes and per-suite / total elapsed seconds, and one log per suite. Every suite's environment also
carries KVLAB_M4_GATE_OUT_DIR=<absolute out-dir>; the M4 suite writes its required diagnostics under
<out-dir>/m4_diagnostics/ (outside pytest's temporary directories), and fails without them. The
elapsed times are a feasibility diagnostic only, never performance evidence. Exit 0 = PASS;
3 = FAIL (any failure, timeout or stop). No retry.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvlab import m3_confirm as mc  # noqa: E402
from kvlab import m4_spec as ms  # noqa: E402


BASETEMP_DIR = "basetemp"  # <out-dir>/basetemp/<suite> is each suite's own pytest --basetemp
GATE_OUT_ENV = "KVLAB_M4_GATE_OUT_DIR"  # every suite's environment: the absolute fresh output directory


def suites(python: str, basetemp: Path) -> list:
    base = [python, "-m", "pytest", "-x", "-v", "-p", "no:cacheprovider"]
    return [("m1", base + [f"--basetemp={basetemp}/m1", "-m", "gpu and m1"]),
            ("m3", base + [f"--basetemp={basetemp}/m3", "-m", "gpu and m3"]),
            ("m4", base + [f"--basetemp={basetemp}/m4", "-m", "gpu and m4", "tests/gpu/test_m4_replay.py"])]


class GateStopped(Exception):
    """A stop signal (SIGINT, SIGTERM, SIGHUP) received while the gate runs."""


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _stop(signum, _frame):
    raise GateStopped(signal.Signals(signum).name)


def gate_commands(out_dir: Path, python: str = sys.executable) -> list:
    """The gate's production commands for output directory `out_dir` (used by main()): each suite's
    pytest --basetemp is its own directory <out_dir>/basetemp/<suite>, never `out_dir` itself."""
    return suites(python, Path(out_dir) / BASETEMP_DIR)


def run_gate(cmds, cap_s: float, out_dir: Path, *, cwd=ROOT, clock=time.monotonic, grace_s: float = 10.0) -> dict:
    """Run the suites under one deadline. Whatever happens (failure, deadline, stop signal, wrapper
    error), the running suite's process group is terminated and reaped, and gate_record.json is
    written with status FAIL unless every suite passed in time. A wrapper error is re-raised
    after that cleanup.

    Ownership is of the process GROUP, not only its leader: after every suite, including one whose
    leader exited normally, the group is checked and any remaining descendant is terminated. A
    suite that left processes running is a FAIL ("leftover_processes"). The suite stays owned
    (`active`) until that cleanup is complete. The caller's signal mask is saved and restored."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)  # refuses an existing directory
    # pytest creates an explicit --basetemp with a NON-recursive mkdir, so the suites' common parent
    # must exist; it is created only now, inside the fresh output directory just accepted above.
    # Each suite's own basetemp (<out_dir>/basetemp/<suite>, gate_commands) is left to pytest; the
    # output directory itself, which holds the logs and gate_record.json, is never a basetemp.
    (out_dir / BASETEMP_DIR).mkdir()
    env = {**os.environ, GATE_OUT_ENV: str(out_dir.resolve())}  # where the M4 suite retains its diagnostics
    rec = {"cap_s": cap_s, "suites": [], "stop": None, "status": None, "signals_during_cleanup": [],
           "env_added": {GATE_OUT_ENV: env[GATE_OUT_ENV]},
           "note": "elapsed times are a feasibility diagnostic only, not performance evidence"}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())  # query only: the caller's mask, unchanged
    old = {sig: signal.signal(sig, _stop) for sig in STOP_SIGNALS}
    t0, active, error = clock(), None, None
    try:
        for name, argv in cmds:
            remaining = cap_s - (clock() - t0)
            if remaining <= 0:
                rec["stop"] = {"reason": "deadline", "before": name}
                break
            entry = {"suite": name, "argv": argv, "log": f"{name}.log", "returncode": None, "killed": None}
            rec["suites"].append(entry)
            entry["_start"] = clock()
            with open(out_dir / entry["log"], "wb") as log:
                # no stop signal may land between the launch and `active` (the child would go untracked);
                # the child itself starts with an empty signal mask
                signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
                try:
                    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True,
                                            preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, ()))
                    active = (proc, entry)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)  # back to the caller's mask
                try:
                    entry["returncode"] = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    rec["stop"] = {"reason": "deadline", "during": name}
                    break
            # the leader exited; its group may still hold descendants: clean up the GROUP before release
            entry["group_cleanup"] = mc._kill_group(proc.pid, grace_s, proc)
            active = None
            entry["elapsed_s"] = clock() - entry.pop("_start")
            leftover = entry["group_cleanup"].get("sigterm") or entry["group_cleanup"].get("sigkill") or \
                not entry["group_cleanup"].get("group_gone")
            if entry["returncode"] != 0:
                rec["stop"] = {"reason": "failure", "suite": name, "returncode": entry["returncode"]}
                break
            if leftover:
                rec["stop"] = {"reason": "leftover_processes", "suite": name}
                break
    except GateStopped as e:
        rec["stop"] = {"reason": "operator_signal", "signal": str(e)}
    except BaseException as e:  # noqa: BLE001  (recorded and cleaned up, then re-raised)
        rec["stop"] = {"reason": "wrapper_error", "detail": repr(e)[:300]}
        error = e
    finally:
        for sig in STOP_SIGNALS:  # cleanup cannot be interrupted; later signals are only recorded
            signal.signal(sig, lambda signum, _f: rec["signals_during_cleanup"].append(signal.Signals(signum).name))
        if active is not None:  # the group, whether or not its leader has already exited
            proc, entry = active
            entry["killed"] = mc._kill_group(proc.pid, grace_s, proc)
            entry["returncode"] = proc.poll()
            entry["elapsed_s"] = clock() - entry.pop("_start", clock())
        rec["total_elapsed_s"] = clock() - t0
        passed = rec["stop"] is None and len(rec["suites"]) == len(cmds) and \
            all(e_["returncode"] == 0 for e_ in rec["suites"])
        rec["status"] = "PASS" if passed and rec["total_elapsed_s"] < cap_s else "FAIL"
        (out_dir / "gate_record.json").write_text(json.dumps(rec, indent=1, default=str))
        for sig, h in old.items():
            signal.signal(sig, h)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    if error is not None:
        raise error
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists():
        ap.error(f"{out} exists")
    rec = run_gate(gate_commands(out, sys.executable), ms.GATE_CAP_S, out)
    print(json.dumps({k: rec[k] for k in ("status", "stop", "total_elapsed_s")}, default=str))
    return 0 if rec["status"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
