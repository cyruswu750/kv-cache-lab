"""M3_CONFIRM_V1: spec, freeze binding, budgeted batch wrapper and fail-closed analysis (CPU only).

- **Evidence:** synthetic. A generator below mirrors the loop of scripts/m3_measure.py; it is
  written independently of the validator's replay. The provenance comes from a real freeze
  manifest of the working tree (`kvlab.m3_confirm.build_manifest`). The stack is synthetic,
  equal to the spec's required stack, so no GPU query is needed.
- **Wrapper:** tested with dummy child processes only. No test launches a GPU process.
- **Old frozen M3 spec:** its regression suite is tests/test_m3_analyze_validation.py. Here we
  check only that the new spec's replay reproduces the recorded M3 order.
"""

import copy
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kvlab import m3_confirm as mc
from kvlab import m3_validate as mv
from kvlab import perf

ROOT = Path(__file__).resolve().parent.parent
SPEC = mv.M3_CONFIRM_V1
OLD = mv.FROZEN_M3_EVAL
M3_EVAL = ROOT / "results" / "m3" / "eval_20261103T173632"
_spec = importlib.util.spec_from_file_location("m3_analyze_confirm_tests", ROOT / "scripts" / "m3_analyze.py")
m3_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3_analyze)
CB = 7340032
T_FREEZE = 1_800_000_000.0
T_LAUNCH = T_FREEZE + 60.0
SIDS = list(SPEC["sessions"])
COMBOS = [tuple(c) for c in SPEC["combos"]]
STACK = {"python": SPEC["required_stack"]["python"], "packages": dict(SPEC["required_packages"]),
         "weights": {"config.json": {"blob": "f81ead14ab072d65a07817f83a3ee0e5a1890d10", "bytes": 660},
                     "model.safetensors": {"blob": SPEC["required_stack"]["weights_blob"],
                                           "bytes": SPEC["required_stack"]["weights_bytes"]}},
         "gpu_name": SPEC["required_stack"]["gpu_name"], "driver_version": SPEC["required_stack"]["driver_version"],
         "kernel": "6.18.54.1-microsoft-standard-WSL2", "cpu": "AMD Ryzen 9 7945HX with Radeon Graphics"}


# ---------------------------------------------------------------------------------------------
# synthetic evidence
def _last_restore(backend, n):
    if backend == "baseline":
        return {"backend": "baseline", "chunks_planned": n, "chunks_loaded": n, "h2d_copy_calls": 56 * n,
                "h2d_bytes": n * CB, "placement_ops": 0, "device_destination_bytes": n * CB, "device_scratch_bytes": 0}
    return {"backend": "batched", "chunks_planned": n, "chunks_loaded": n, "h2d_copy_calls": n, "h2d_bytes": n * CB,
            "placement_ops": math.ceil(n / 4), "device_destination_bytes": n * CB,
            "device_scratch_bytes": min(n, 4) * CB}


def synth_session(sid, manifest, t_start, pid, speed=None):
    """One session report as scripts/m3_measure.py writes it. `speed(L, S, path, backend)`
    gives the e2e time in ms before noise (default: batched 25% faster on every case)."""
    seed = SPEC["sessions"][sid]
    rng, noise = random.Random(seed), random.Random(seed + 99)
    cases = [tuple(int(x) for x in c.split(":")) for c in mc.case_arg(SPEC).split(",")]
    order = cases[:]
    rng.shuffle(order)
    block_cases = [c for _ in range(SPEC["blocks"]) for c in order]
    speed = speed or (lambda L, S, path, b: {"full": 90.0, "host": 30.0, "disk": 34.0}[path] * (L / 1024) *
                      (0.75 if b == "batched" else 1.0))
    clock = [t_start + 6.0]

    def sample(path, backend, mode, L, S):
        n = L // 256
        e2e = int(speed(L, S, path, backend) * 1e6 * noise.uniform(0.98, 1.02))
        clock[0] += e2e / 1e9 + 0.004
        rec = {"path": path, "L": L, "S": S, "N": L + S, "e2e_ns": e2e, "t_wall": clock[0],
               **perf.expected(path, L, S), "miss_reason": "restore_not_attempted" if path == "full" else None,
               "first_token": 13, "backend": backend, "mode": mode,
               "last_restore": None if path == "full" else _last_restore(backend, n),
               "mem": {"allocated_before": 3096866816, "reserved_before": 4343201792,
                       "peak_allocated": 3184955904, "peak_reserved": 4343201792, "allocated_after": 3096866816},
               "rusage_delta": {}, "alloc_delta": {}}
        if path == "disk":
            rec["resident_fraction_before"] = 1.0
        if mode == "synced":
            rec["phases_ns"] = {"ingest": 200000, "restore": int(e2e * 0.2), "prefill": int(e2e * 0.7),
                                "store": 30000, "select": 150000}
        return rec

    def corr(L, S, backend):
        n = L // 256
        return {"L": L, "S": S, "host_restore_len": L, "host_sources": ["host"] * n, "disk_restore_len": L,
                "disk_sources": ["disk"] * n, "full_restore_len": 0, "full_sources": [],
                "restored_kv_bytes_equal": {"host": True, "disk": True}, "strict_host_eq_disk_bitwise": True,
                "strict_host_eq_resident_bitwise": True, "strict_pass": True,
                "report_only_full_vs_restored": {"first_token_equal": True, "max_abs_logit_diff": 0.25}}

    def ab(L, S):
        n = L // 256
        tier = lambda t: {"restore_len_sources_admission_equal": True, "kv_bytes_equal_after_prefill": True,  # noqa: E731
                          "first_logits_bitwise_equal": True, "greedy_tokens_equal": True,
                          "step_logits_bitwise_equal": True, "restore_len": L, "sources": [t] * n}
        return {"L": L, "S": S, "host": tier("host"), "disk": tier("disk"), "pass": True}

    def mem(L, S):
        n = L // 256
        return {"L": L, "S": S, **{f"{t}_{b}": {"L": L, "sources_ok": True, "peak_above_start": 2 * n * CB,
                                                "kept_after_return": n * CB, "last_restore": _last_restore(b, n),
                                                "reserved_after": 4053794816}
                                   for t in ("host", "disk") for b in ("baseline", "batched")}}

    samples, correctness, ab_checks, memory, blocks = [], [], [], [], []
    for L, S in sorted({min(cases), max(cases)}):  # process warmup, as the runner does it
        for path, b, m in COMBOS:
            for _ in range(2):
                samples.append({**sample(path, b, m, L, S), "warmup": True, "stage": "process_warmup", "session": sid})
    for bi, (L, S) in enumerate(block_cases):
        if bi < len(order):
            correctness.append({"baseline": corr(L, S, "baseline"), "batched": corr(L, S, "batched"),
                                "strict_pass_all": True})
            ab_checks.append(ab(L, S))
            memory.append(mem(L, S))
        tb = clock[0]
        for r in range(SPEC["warmup"] + SPEC["reps"]):
            combos = list(COMBOS)
            rng.shuffle(combos)
            for pos, (path, b, m) in enumerate(combos):
                samples.append({**sample(path, b, m, L, S), "session": sid, "block": bi, "rep": r, "order_pos": pos,
                                "warmup": r < SPEC["warmup"], "stage": "m3"})
        blocks.append({"block": bi, "L": L, "S": S, "t_start": tb, "t_end": clock[0]})
    n_meas = sum(not s["warmup"] for s in samples)
    meta = {"session": sid, "args": mc.expected_args(SPEC, sid), "seed": seed, "started_at": "2027-01-15T08:00:00-0800",
            "started_unix": t_start, "gpu_monitor_csv": f"session_{sid}.gpu_monitor.csv",
            "block_order": [list(c) for c in block_cases], "finished_at": "2027-01-15T08:03:00-0800",
            "elapsed_min": (clock[0] - t_start) / 60, "n_samples": len(samples), "n_measured": n_meas,
            "n_excluded": 0, "correctness_all": True}
    gpu_static = ("name, driver_version, power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz], memory.total [MiB]\n"
                  f"{STACK['gpu_name']}, {STACK['driver_version']}, [N/A], 3105 MHz, 9001 MHz, 12282 MiB")
    return {"meta": meta,
            "source": {"git_head": manifest["base_git_head"], "git_status_porcelain": [],
                       "git_diff_head_sha256": manifest["git_diff_head_sha256"],
                       "files_sha256": dict(manifest["files_sha256"])},
            "environment": {"torch": SPEC["required_stack"]["torch"], "transformers": SPEC["required_stack"]["transformers"],
                            "cuda": SPEC["required_stack"]["cuda"], "python": SPEC["required_stack"]["python"],
                            "gpu": STACK["gpu_name"], "gpu_static": gpu_static, "cpu": STACK["cpu"],
                            "kernel": STACK["kernel"]},
            "packages": dict(STACK["packages"]), "weights": copy.deepcopy(STACK["weights"]),
            "process": {"pid": pid, "env": dict(SPEC["required_process_env"])},
            "model": {**SPEC["required_model"], "num_key_value_heads": 2, "head_dim": 128},
            "restore_config": {"restore_backend": "baseline", "restore_staging_chunks": 4},
            "gpu_state": [], "samples": samples, "correctness": correctness, "ab_checks": ab_checks,
            "restore_memory": memory, "failures": [], "blocks": blocks, "budgets": {}, "process_end": {},
            "excluded": []}


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """A real freeze manifest of the current working tree (synthetic stack), and its file."""
    d = tmp_path_factory.mktemp("freeze")
    manifest, diff = mc.build_manifest(ROOT, SPEC, stack=STACK, now=T_FREEZE, require_frozen_protocol=False)
    manifest["git_diff_file_sha256"] = mc.sha256_bytes(diff.encode())
    path = d / "M3_CONFIRM_V1.freeze.json"
    path.write_text(json.dumps(manifest, indent=1))
    return manifest, path


@pytest.fixture(scope="module")
def base_reports(frozen):
    manifest, _ = frozen
    return {sid: synth_session(sid, manifest, T_LAUNCH + 1 + 200 * i, 50000 + i) for i, sid in enumerate(SIDS)}


def write_manifest(tmp_path, manifest):
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps(manifest, indent=1))
    return p


def make_run(tmp_path, frozen, reports, edits=None, batch_edit=None, after=None, manifest_edit=None, exits=None):
    """Write a run directory as the wrapper and runner leave it.
    - `exits[sid]`: that session's exit code (default 0). As the wrapper does, the first nonzero
      exit stops the batch: later sessions are never launched (no files, listed as not launched),
      and the record says stop = nonzero_exit, execution_status = failed.
    - `edits[sid](report)`: change a report before it is written. The batch record then hashes
      the changed file, as if the runner had produced it.
    - `batch_edit(record)`: change the batch record.
    - `after(run)`: change files after the batch record was written.
    - `manifest_edit(manifest)`: change the manifest before the batch record names its SHA-256.
    Returns (run_dir, manifest_path)."""
    manifest = copy.deepcopy(frozen[0])
    if manifest_edit:
        manifest_edit(manifest)
    mpath = write_manifest(tmp_path, manifest)
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    entries, stop, not_launched = [], None, []
    for i, c in enumerate(manifest["commands"]):
        sid = c["session"]
        if stop is not None:
            not_launched.append(sid)
            continue
        rep = reports.get(sid)
        if rep is not None:
            if edits and sid in edits:
                rep = copy.deepcopy(rep)
                edits[sid](rep)
            (run / f"session_{sid}.json").write_text(json.dumps(rep))
            (run / f"session_{sid}.gpu_monitor.csv").write_text("2027/01/15 08:00:01.000, 2505 MHz, 9001 MHz\n")
            (run / "logs" / f"session_{sid}.log").write_text("block 0 done\n")
        rc = (exits or {}).get(sid, 0)
        entries.append({"session": sid, "argv": c["argv"], "env": c["env"], "log": f"logs/session_{sid}.log",
                        "pid": 50000 + i, "started_s": 200.0 * i, "ended_s": 200.0 * i + 150.0, "returncode": rc,
                        "killed": False, "leftover_cleanup": None,
                        "session_json": mc._file_state(run / f"session_{sid}.json"),
                        "monitor_csv_present": True, "runner_reported": None, "scratch_removed": None})
        if rc != 0:
            stop = {"reason": "nonzero_exit", "session": sid, "returncode": rc, "detail": f"exit code {rc}"}
    rec = {"record_version": mc.BATCH_RECORD_VERSION, "experiment": SPEC["name"],
           "freeze_manifest_sha256": mc.sha256_file(mpath),
           "pre_launch_verification": {"problems": [], "checked_at": "2027-01-15T07:59:00-0800"},
           "budget_s": SPEC["budget_s"], "clock": "time.monotonic", "wrapper_pid": 49999,
           "execution_status": "failed" if stop else "succeeded", "stop": stop,
           "launched_at": "2027-01-15T08:00:00-0800", "launched_unix": T_LAUNCH, "sessions": entries,
           "not_launched": not_launched, "elapsed_s": entries[-1]["ended_s"], "finalized_s": entries[-1]["ended_s"] + 1,
           "signals_received": [], "finished_at": "2027-01-15T08:16:00-0800"}
    if batch_edit:
        batch_edit(rec)
    (run / mc.BATCH_RECORD).write_text(json.dumps(rec, indent=1))
    if after:
        after(run)
    return run, mpath


def cli(run, mpath, out, *extra):
    code = m3_analyze.main(["--spec", SPEC["name"], "--run-dir", str(run), "--out-dir", str(out),
                            "--freeze", str(mpath), *extra])
    return code, json.loads((out / "m3_validation.json").read_text())


def assert_rejected(code, val, out, status="invalid", expect=()):
    assert code == m3_analyze.EXIT[status] and code != 0
    assert val["status"] == status
    d = val["decision"]
    assert d["m3_confirm_v1_confirmed"] is False and "supports_adoption" not in d
    assert d["all_correctness_and_ab_checks_pass"] is False and d["authorizes_default_change"] is False
    assert val["correctness_and_ab_acceptance_recomputed"] is False
    for name in ("m3_analysis.json", "m3_analysis.md", "samples_chronological.csv"):
        assert not (out / name).exists()  # no performance result from invalid or failed evidence
    for e in expect:
        assert any(e in p for p in val["problems"]), (e, val["problems"][:12])


def m3_index(rep, block, r, combo):
    return next(i for i, s in enumerate(rep["samples"]) if s["stage"] == "m3" and
                (s["block"], s["rep"], s["path"], s["backend"], s["mode"]) == (block, r, *combo))


def pw_index(rep, L, combo, nth=0):
    return [i for i, s in enumerate(rep["samples"]) if s["stage"] == "process_warmup" and s["L"] == L and
            (s["path"], s["backend"], s["mode"]) == combo][nth]


HBP = ("host", "batched", "plain")
DBP = ("disk", "baseline", "plain")


# ---------------------------------------------------------------------------------------------
# the spec and plan
def test_spec_counts_commands_and_runner_semantics():
    from kvlab import m3_bench as mb
    assert {k: v["expected"] for k, v in mv._new_counts(SPEC).items()} == {
        "measured_samples": 8640, "block_warmup_samples": 720, "process_warmup_samples": 180,
        "correctness_checks": 20, "ab_checks": 20, "restore_memory_records": 10}
    assert tuple(SPEC["combos"]) == tuple(mb.COMBOS) and len(SPEC["combos"]) == 9
    assert SPEC["sessions"] == {f"c{i}": 32000 + i for i in range(1, 6)}
    assert list(SPEC["process_warmup"]["cases"]) == sorted({min(SPEC["cases"]), max(SPEC["cases"])})
    assert SPEC["profile_session"] is None and SPEC["budget_s"] == 1800 and SPEC["staging_chunks"] == 4
    for L, S in SPEC["cases"]:
        perf.check_case(L, S)
    assert (1280 + 96) // 256 == 1280 // 256  # the fresh case admits no new chunk
    cmd = mc.commands_for(SPEC)
    assert [c["session"] for c in cmd] == SIDS and all(c["env"] == {"HF_HUB_OFFLINE": "1"} for c in cmd)
    assert cmd[2]["argv"] == [".venv/bin/python", "scripts/m3_measure.py", "--out-dir", SPEC["run_dir"],
                              "--session", "c3", "--cases", "1024:1,1280:96", "--blocks", "4", "--warmup", "2",
                              "--reps", "24", "--seed", "32003", "--staging-chunks", "4"]
    runner = (ROOT / "scripts" / "m3_measure.py").read_text()
    assert "for L, S in sorted({min(cases), max(cases)}):" in runner and "rng.shuffle(order)" in runner
    assert 'ap.add_argument("--profile-reps", type=int, default=5)' in runner  # expected_args' profile_reps
    doc = (ROOT / mc.PROTOCOL_DOC).read_text()
    assert mc.DRAFT_MARKER in doc.splitlines()[0]  # still a draft: the freeze refuses
    for s in ("8,640", "720", "180", "1,800 s", "32001–32005", "(1280, 96)", "5 sessions × 2 cases × 4 blocks × 24 reps × 9 combinations"):
        assert s in doc, s
    # the frozen M3 spec is untouched
    assert OLD["git_head"] == "362ee40aeaaae204b2b1dc88398b46bc5826f410" and OLD["reps"] == 12
    assert "provenance" not in OLD and "decision_rule" not in OLD and "exact_sequence" not in OLD
    assert mv.DEFAULT_SPEC == OLD["name"]


def test_required_stack_is_the_recorded_m3_stack():
    rep = json.loads((M3_EVAL / "session_s1.json").read_text())
    assert rep["packages"] == SPEC["required_packages"]
    env, req = rep["environment"], SPEC["required_stack"]
    assert {k: env[k] for k in ("torch", "transformers", "cuda", "python")} == {k: req[k] for k in ("torch", "transformers", "cuda", "python")}
    assert mc.parse_gpu_static(env["gpu_static"]) == {"gpu_name": req["gpu_name"], "driver_version": req["driver_version"]}
    assert rep["weights"]["model.safetensors"] == {"blob": req["weights_blob"], "bytes": req["weights_bytes"]}
    assert {k: rep["model"][k] for k in SPEC["required_model"]} == SPEC["required_model"]
    assert mc.model_constant_problems(ROOT, SPEC) == []


def test_torch_free_identity_mirrors_the_runner():
    """The freeze and the launch check use torch-free copies of the runner's identity records."""
    from kvlab import diag_harness as dh
    from kvlab import m3_bench as mb
    a, b = mc.source_identity(ROOT), mb.source_identity(ROOT)
    assert a == b
    assert mc.package_versions() == dh.package_versions()
    m = SPEC["required_model"]
    assert mc.weight_blobs(ROOT, m["model_id"], m["revision"]) == dh.weight_blobs(ROOT, m["model_id"], m["revision"])
    r = subprocess.run([sys.executable, "-c", "import sys, kvlab.m3_validate, kvlab.m3_confirm; print('torch' in sys.modules)"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False"


def test_sequence_replay_reproduces_the_recorded_m3_order():
    """The replay the new spec relies on, checked against real runner output (the M3 evaluation)."""
    for i in range(1, 6):
        rep = json.loads((M3_EVAL / f"session_s{i}.json").read_text())
        block_cases, seq = mv.expected_sequence(OLD, f"s{i}")
        assert [tuple(x) for x in rep["meta"]["block_order"]] == block_cases
        got = [(s["stage"], s["L"], s["S"], s.get("block"), s.get("rep"), s.get("order_pos"),
                (s["path"], s["backend"], s["mode"])) for s in rep["samples"]]
        assert got == seq and len(seq) == 1296


# ---------------------------------------------------------------------------------------------
# complete valid evidence
@pytest.fixture(scope="module")
def valid_analysis(tmp_path_factory, frozen, base_reports):
    tmp = tmp_path_factory.mktemp("valid")
    run, mpath = make_run(tmp, frozen, base_reports)
    code, val = cli(run, mpath, tmp / "out")
    return code, val, tmp / "out", run


def test_complete_valid_evidence_confirms(valid_analysis):
    code, val, out, run = valid_analysis
    assert code == 0 and val["status"] == "valid", val["problems"][:10]
    assert val["problems"] == [] and val["correctness_failures"] == []
    assert val["counts"] == {k: {"expected": n, "present": n, "passed": n} for k, n in (
        ("measured_samples", 8640), ("block_warmup_samples", 720), ("process_warmup_samples", 180),
        ("correctness_checks", 20), ("ab_checks", 20), ("restore_memory_records", 10))}
    assert val["provenance"]["batch_record_sha256"] == mc.sha256_file(run / mc.BATCH_RECORD)
    a = json.loads((out / "m3_analysis.json").read_text())
    d = a["decision"]
    assert d["m3_confirm_v1_confirmed"] is True and "supports_adoption" not in d
    assert d["primary_verdicts"] == {f"{c} {t} {s}": "batched_faster" for c in ("1024:1", "1280:96")
                                     for t in ("host", "disk") for s in ("median_d", "mean_d")}
    assert d["default_backend"] == "baseline" and d["authorizes_default_change"] is False
    assert "not pooled" in d["scope"]
    assert set(a["cases"]) == {"1024:1", "1280:96"} and "profile" not in a
    for c in a["cases"].values():
        assert c["n_strata"] == 20
        for t in c["tiers"].values():
            assert t["e2e"]["n_pairs"] == t["restore_phase"]["n_pairs"] == 480
    md = (out / "m3_analysis.md").read_text()
    for s in ("# M3_CONFIRM_V1 analysis", "## Primary: end-to-end", "## Secondary: restore phase", "## Context:",
              "**M3_CONFIRM_V1 confirmed: True**", "does not by itself authorize a default change"):
        assert s in md, s
    assert "supports adoption" not in md
    assert sorted(p.name for p in run.iterdir()) == sorted({mc.BATCH_RECORD, "logs", *(f"session_{s}.json" for s in SIDS),
                                                            *(f"session_{s}.gpu_monitor.csv" for s in SIDS)})


def test_warmup_timings_never_enter_the_statistics(tmp_path, frozen, base_reports, valid_analysis):
    def inflate(rep):
        for s in rep["samples"]:
            if s["warmup"]:
                s["e2e_ns"] *= 100
                if s["mode"] == "synced":
                    s["phases_ns"] = {k: v * 100 for k, v in s["phases_ns"].items()}
    run, mpath = make_run(tmp_path, frozen, base_reports, edits={s: inflate for s in SIDS})
    code, val = cli(run, mpath, tmp_path / "out")
    assert code == 0 and val["status"] == "valid"
    new = json.loads((tmp_path / "out" / "m3_analysis.json").read_text())
    old = json.loads((valid_analysis[2] / "m3_analysis.json").read_text())
    assert new["cases"] == old["cases"] and new["decision"] == old["decision"]


def test_complete_but_not_confirmed_is_a_valid_final_outcome(tmp_path, frozen, base_reports):
    """No difference on (1280, 96): complete, valid evidence (exit 0), but not confirmed."""
    manifest = frozen[0]

    def speed(L, S, path, b):
        base = {"full": 90.0, "host": 30.0, "disk": 34.0}[path] * (L / 1024)
        return base * (0.75 if b == "batched" and L == 1024 else 1.0)
    reports = {sid: synth_session(sid, manifest, T_LAUNCH + 1 + 200 * i, 50000 + i, speed=speed)
               for i, sid in enumerate(SIDS)}
    run, mpath = make_run(tmp_path, frozen, reports)
    code, val = cli(run, mpath, tmp_path / "out")
    assert code == 0 and val["status"] == "valid"
    d = json.loads((tmp_path / "out" / "m3_analysis.json").read_text())["decision"]
    assert d["m3_confirm_v1_confirmed"] is False and d["all_correctness_and_ab_checks_pass"] is True
    assert all(v == "batched_faster" for k, v in d["primary_verdicts"].items() if k.startswith("1024:1"))
    assert d["primary_not_established"] and all(k.startswith("1280:96") for k in d["primary_not_established"])
    assert "**M3_CONFIRM_V1 confirmed: False**" in (tmp_path / "out" / "m3_analysis.md").read_text()


# ---------------------------------------------------------------------------------------------
# invalid or incomplete evidence: exit 3, no analysis, no positive verdict
def _drop_case(rep):
    rep["samples"] = [s for s in rep["samples"] if (s["L"], s["S"]) != (1280, 96)]
    rep["meta"]["n_samples"] = len(rep["samples"])
    rep["meta"]["n_measured"] = sum(not s["warmup"] for s in rep["samples"])


def _duplicate(rep):
    rep["samples"].insert(m3_index(rep, 5, 7, HBP) + 1, copy.deepcopy(rep["samples"][m3_index(rep, 5, 7, HBP)]))
    rep["meta"]["n_samples"] += 1
    rep["meta"]["n_measured"] += 1


def _substitute(rep):  # count unchanged: one measured sample replaced by a copy of its partner
    rep["samples"][m3_index(rep, 5, 7, HBP)] = {**copy.deepcopy(rep["samples"][m3_index(rep, 5, 7, ("host", "baseline", "plain"))]),
                                                "order_pos": rep["samples"][m3_index(rep, 5, 7, HBP)]["order_pos"]}


def _swap(rep):  # identities and fields intact, two samples of one repetition swap positions in time
    i, j = m3_index(rep, 5, 7, HBP), m3_index(rep, 5, 7, DBP)
    a, b = rep["samples"][i], rep["samples"][j]
    a["order_pos"], b["order_pos"], a["t_wall"], b["t_wall"] = b["order_pos"], a["order_pos"], b["t_wall"], a["t_wall"]
    rep["samples"][i], rep["samples"][j] = b, a


def _block_warmup_missing(rep):
    del rep["samples"][m3_index(rep, 2, 1, HBP)]
    rep["meta"]["n_samples"] -= 1


def _block_warmup_substituted(rep):
    rep["samples"][m3_index(rep, 2, 1, HBP)] = copy.deepcopy(rep["samples"][m3_index(rep, 2, 0, HBP)])


def _block_warmup_misflagged(rep):
    rep["samples"][m3_index(rep, 2, 0, HBP)]["warmup"] = False


def _pw_missing(rep):
    del rep["samples"][pw_index(rep, 1280, DBP)]
    rep["meta"]["n_samples"] -= 1


def _pw_substituted(rep):
    rep["samples"][pw_index(rep, 1280, DBP, 1)] = copy.deepcopy(rep["samples"][pw_index(rep, 1280, HBP)])


def _pw_misflagged(rep):
    rep["samples"][pw_index(rep, 1024, HBP)]["warmup"] = False


def _shuffled_blocks(rep):  # a different but self-consistent block order: not the seed's
    rep["meta"]["block_order"] = rep["meta"]["block_order"][1:] + rep["meta"]["block_order"][:1]


INVALID = {
    "missing_case_1280_96_in_all_sessions": ({s: _drop_case for s in SIDS}, [
        "c1: 864 of 1728 planned measured samples missing", "c5: sample sequence differs"]),
    "duplicate_row": ({"c2": _duplicate}, ["c2: duplicate measured sample identity (block, rep, path, backend, mode) = (5, 7, 'host', 'batched', 'plain')"]),
    "substituted_row_same_count": ({"c2": _substitute}, ["c2: 1 of 1728 planned measured samples missing",
                                                         "c2: duplicate measured sample identity"]),
    "reordered_rows_same_identities": ({"c2": _swap}, ["c2: sample sequence differs from the runner's order for seed 32002"]),
    "block_warmup_missing": ({"c3": _block_warmup_missing}, ["c3: 1 of 144 planned block-warmup samples missing"]),
    "block_warmup_substituted": ({"c3": _block_warmup_substituted},
                                 ["c3: duplicate block-warmup sample identity", "c3: 1 of 144 planned block-warmup samples missing"]),
    "block_warmup_misflagged": ({"c3": _block_warmup_misflagged}, ["c3: sample", "warmup flag False for repetition 0"]),
    "process_warmup_missing": ({"c4": _pw_missing}, ["c4: process-warmup records differ from the plan (36 = 2 cases x 9 combinations x 2)",
                                                     "missing {((1280, 96), ('disk', 'baseline', 'plain')): '1/2'}"]),
    "process_warmup_substituted": ({"c4": _pw_substituted},
                                   ["surplus {((1280, 96), ('host', 'batched', 'plain')): '3/2'}", "c4: sample sequence differs"]),
    "process_warmup_misflagged": ({"c4": _pw_misflagged}, ["c4: sample", "process-warmup sample not flagged as warmup (False)"]),
    "block_order_not_from_seed": ({"c5": _shuffled_blocks}, ["c5: block_order"]),
    "correctness_record_missing": ({"c1": lambda r: r["correctness"].pop()}, ["c1: no correctness record for case",
                                                                              "c1: stored correctness_all=True"]),
    "restore_memory_tier_missing": ({"c1": lambda r: r["restore_memory"][0].pop("disk_batched")},
                                    ["restore_memory", "disk_batched: missing, incomplete or not a full batched restore"]),
    "profile_section_in_session": ({"c1": lambda r: r.update(profile=[])}, ["c1: unexpected profiler evidence"]),
}


@pytest.mark.parametrize("name", list(INVALID))
def test_invalid_evidence_is_rejected(tmp_path, frozen, base_reports, name):
    edits, expect = INVALID[name]
    run, mpath = make_run(tmp_path, frozen, base_reports, edits=edits)
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=expect)
    assert val["correctness_failures"] == []


def _drop_session(reports, sid):
    return {k: v for k, v in reports.items() if k != sid}


def test_missing_session_is_invalid(tmp_path, frozen, base_reports):
    run, mpath = make_run(tmp_path, frozen, _drop_session(base_reports, "c3"))
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=["missing session file session_c3.json",
                                                         "batch c3: no session file",
                                                         "missing file in the run directory: logs/session_c3.log"])


@pytest.mark.parametrize("name, path, expect", [
    ("profiler_session", "session_prof.json", "unexpected session file session_prof.json (not in spec M3_CONFIRM_V1; "
                                               "this experiment plans no profiler session)"),
    ("profiler_trace", "trace.json", "unexpected file in the run directory: trace.json"),
    ("extra_session", "session_c6.json", "unexpected session file session_c6.json"),
    ("stale_analysis_in_run_dir", "m3_validation.json", "unexpected file in the run directory: m3_validation.json"),
])
def test_unexpected_files_and_profiler_evidence_are_rejected(tmp_path, frozen, base_reports, name, path, expect):
    run, mpath = make_run(tmp_path, frozen, base_reports, after=lambda r: (r / path).write_text("{}"))
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=[expect])


# ---- provenance and batch-record binding
def _set(path, value):
    def edit(obj):
        *head, last = path
        for k in head:
            obj = obj[k]
        obj[last] = value
    return edit


PROVENANCE = {
    "session_source_bytes_differ": (dict(edits={"c2": lambda r: r["source"]["files_sha256"].update({"kvlab/tiered_cache.py": "0" * 64})}),
                                    ["c2: source files differ from the freeze: ['kvlab/tiered_cache.py']"]),
    "session_new_untracked_file": (dict(edits={"c2": lambda r: r["source"]["files_sha256"].update({"scripts/extra.py": "1" * 64})}),
                                   ["c2: source files differ from the freeze: ['scripts/extra.py']"]),
    "session_git_head_differs": (dict(edits={"c2": _set(("source", "git_head"), "f" * 40)}), ["c2: git_head 'fff"]),
    "session_diff_differs": (dict(edits={"c2": _set(("source", "git_diff_head_sha256"), "2" * 64)}),
                             ["c2: git diff HEAD differs from the frozen diff"]),
    "session_transformers_differs": (dict(edits={"c2": _set(("environment", "transformers"), "5.23.0")}),
                                     ["c2: environment transformers '5.23.0', required '5.22.1'"]),
    "session_packages_differ": (dict(edits={"c2": lambda r: r["packages"].update(numpy="2.6.0")}),
                                ["c2: package versions differ from the frozen and required stack"]),
    "session_driver_differs": (dict(edits={"c2": lambda r: r["environment"].update(
        gpu_static=r["environment"]["gpu_static"].replace("596.49", "600.10"))}), ["c2: GPU"]),
    "session_weights_differ": (dict(edits={"c2": lambda r: r["weights"]["model.safetensors"].update(blob="3" * 64)}),
                               ["c2: model weight files differ from the freeze"]),
    "session_arguments_differ": (dict(edits={"c2": _set(("meta", "args", "profile_reps"), 6)}),
                                 ["c2: recorded arguments"]),
    "session_offline_env_missing": (dict(edits={"c2": _set(("process", "env"), {})}), ["c2: process environment"]),
    "session_alloc_conf_set": (dict(edits={"c2": lambda r: r["process"]["env"].update(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")}),
                               ["c2: process environment"]),
    "session_started_before_freeze": (dict(manifest_edit=_set(("frozen_unix",), T_LAUNCH + 100)),
                                      ["c1: session started at", "batch launched at"]),
    "session_file_edited_after_batch": (dict(after=lambda run: (run / "session_c4.json").write_text(
        (run / "session_c4.json").read_text() + " ")), ["batch c4: session file differs from the one the batch recorded"]),
    "manifest_edited_after_batch": (dict(after=lambda run: (run.parent / "freeze.json").write_text(
        (run.parent / "freeze.json").read_text() + "\n")), ["batch record freeze_manifest_sha256"]),
    "manifest_analysis_bytes_differ": (dict(manifest_edit=lambda m: m["files_sha256"].update({"scripts/m3_analyze.py": "4" * 64})),
                                       ["scripts/m3_analyze.py differs from its frozen bytes (analysis code changed after the freeze)",
                                        "source files differ from the freeze: ['scripts/m3_analyze.py']"]),
    "manifest_protocol_bytes_differ": (dict(manifest_edit=lambda m: m["documents_sha256"].update({mc.PROTOCOL_DOC: "5" * 64})),
                                       [f"{mc.PROTOCOL_DOC} differs from its frozen bytes"]),
    "manifest_commands_differ": (dict(manifest_edit=lambda m: m["commands"][0]["argv"].__setitem__(-1, "8")),
                                 ["freeze manifest commands"]),
    "manifest_stack_differs": (dict(manifest_edit=lambda m: m["stack"].update(python="3.14.5")),
                               ["freeze manifest stack: python '3.14.5', required '3.14.4'"]),
    "batch_stopped_at_deadline": (dict(batch_edit=lambda b: (b["sessions"].pop(), b.update(
        execution_status="failed", not_launched=["c5"],
        stop={"reason": "deadline", "session": None, "detail": "deadline reached before launching c5"}))),
        ["batch execution stopped by", "'deadline'", "planned sessions never launched, so their evidence is missing: ['c5']"]),
    "batch_not_finalized": (dict(batch_edit=_set(("execution_status",), "running")),
                            ["batch record not finalized (execution_status 'running')"]),
    "batch_over_budget": (dict(batch_edit=_set(("elapsed_s",), 1800.5)),
                          ["batch elapsed 1800.5 s is missing, not finite or not below the 1800 s budget"]),
    "session_ended_at_the_deadline": (dict(batch_edit=lambda b: b["sessions"][4].update(ended_s=1800.0)),
                                      ["batch c5: ended at 1800.0 s, not a finite time before the 1800 s deadline"]),
    "batch_session_killed": (dict(batch_edit=lambda b: b["sessions"][1].update(killed={"sigterm": True}, returncode=-15)),
                             ["batch c2: exit -15, killed {'sigterm': True}: the session did not exit by itself"]),
    "batch_exit_code_contradicts_session": (dict(exits={"c5": 1}),
                                            ["c5: exit code 1 disagrees with the session record (runner success True)"]),
    "launch_after_a_nonzero_exit": (dict(batch_edit=lambda b: b["sessions"][1].update(returncode=1)),
                                    ["batch c2: exit 1, yet a later session was launched (stopping rule violated)",
                                     "batch sessions ['c2'] exited nonzero, yet stop None and execution_status 'succeeded'",
                                     "batch c2: exit 1, session file"]),
    "status_succeeded_but_stopped": (dict(exits={"c5": 1}, batch_edit=_set(("execution_status",), "succeeded")),
                                     ["batch execution_status 'succeeded', yet stop",
                                      "batch execution_status 'succeeded' with a nonzero-exit stop"]),
    "batch_stopped_by_signal": (dict(batch_edit=_set(("stop",), {"reason": "operator_signal", "session": None,
                                                                   "detail": "SIGTERM before launching c6"})),
                                ["batch execution stopped by {'reason': 'operator_signal'"]),
    "batch_pid_not_the_session_process": (dict(batch_edit=lambda b: b["sessions"][1].update(pid=1)),
                                          ["c2: process id 50001 is not the process the batch launched (1)"]),
    "batch_no_pre_launch_verification": (dict(batch_edit=_set(("pre_launch_verification",), None)),
                                         ["batch record has no passing pre-launch verification"]),
    "batch_record_missing": (dict(after=lambda run: (run / mc.BATCH_RECORD).unlink()), ["batch record batch_record.json missing"]),
    "batch_record_malformed_field": (dict(batch_edit=lambda b: b["sessions"][0].update(session_json="x")),
                                     ["batch c1: session file differs from the one the batch recorded (recorded 'x'"]),
    "manifest_malformed_field": (dict(manifest_edit=_set(("stack",), "x")), ["malformed freeze manifest"]),
    "sessions_overlap_in_time": (dict(edits={"c3": _set(("meta", "started_unix"), T_LAUNCH + 1 + 200 * 1 + 10)}),
                                 ["c3: started at"]),
}


@pytest.mark.parametrize("name", list(PROVENANCE))
def test_provenance_or_batch_mismatch_is_invalid(tmp_path, frozen, base_reports, name):
    kw, expect = PROVENANCE[name]
    run, mpath = make_run(tmp_path, frozen, base_reports, **kw)
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=expect)


# ---- execution-record consistency (re-review P1): status, stop, sequence, exits and timing
# Each case starts from a complete, valid run (all five sessions exit 0) and overwrites batch
# record fields with hand-written literal values, independently of make_run()'s own logic.
def _literal(**fields):
    def edit(rec):
        for path, value in fields.items():
            obj, keys = rec, path.split("__")
            for k in keys[:-1]:
                obj = obj[int(k)] if isinstance(obj, list) else obj[k]
            last = keys[-1]
            if isinstance(obj, list):
                obj[int(last)] = value
            else:
                obj[last] = value
    return edit


NZ_C5 = {"reason": "nonzero_exit", "session": "c5", "returncode": 1, "detail": "exit code 1"}
CONTRADICTORY = {
    # the exact re-review trigger, and the same with the other status value
    "p1_nonzero_stop_all_exits_zero_failed": (
        dict(batch_edit=_literal(execution_status="failed", stop=NZ_C5)),
        ["batch stop records a nonzero exit of 'c5' (1), but the last launched session 'c5' exited 0 (killed False)"]),
    "p1_nonzero_stop_all_exits_zero_succeeded": (
        dict(batch_edit=_literal(execution_status="succeeded", stop=NZ_C5)),
        ["batch stop records a nonzero exit of 'c5' (1), but the last launched session 'c5' exited 0",
         "batch execution_status 'succeeded', yet stop", "batch execution_status 'succeeded' with a nonzero-exit stop"]),
    # the inverse: c5 really exited 1 (its report records a strict failure), but no stop is recorded
    "inverse_nonzero_exit_without_stop_succeeded": (
        dict(edits={"c5": "fail_strict"}, batch_edit=_literal(sessions__4__returncode=1, execution_status="succeeded", stop=None)),
        ["batch sessions ['c5'] exited nonzero, yet stop None and execution_status 'succeeded'"]),
    "inverse_nonzero_exit_without_stop_failed": (
        dict(edits={"c5": "fail_strict"}, batch_edit=_literal(sessions__4__returncode=1, execution_status="failed", stop=None)),
        ["batch execution_status 'failed' without a stop reason",
         "batch sessions ['c5'] exited nonzero, yet stop None and execution_status 'failed'"]),
    "nonzero_stop_names_another_session": (
        dict(edits={"c5": "fail_strict"}, batch_edit=_literal(sessions__4__returncode=1, execution_status="failed",
                                                             stop={**NZ_C5, "session": "c4"})),
        ["batch stop names 'c4' (1), but the last launched session exited ('c5', 1)"]),
    "nonzero_stop_with_another_exit_code": (
        dict(edits={"c5": "fail_strict"}, batch_edit=_literal(sessions__4__returncode=1, execution_status="failed",
                                                             stop={**NZ_C5, "returncode": 2})),
        ["batch stop names 'c5' (2), but the last launched session exited ('c5', 1)"]),
    "boolean_exit_code": (  # JSON true equals 1 in Python comparisons; it is not an exit code
        dict(edits={"c5": "fail_strict"}, batch_edit=_literal(sessions__4__returncode=True, execution_status="failed",
                                                             stop={**NZ_C5, "returncode": True})),
        ["batch stop records a nonzero exit of 'c5' (True), but the last launched session 'c5' exited True"]),
    "nonzero_stop_but_session_killed": (
        dict(batch_edit=_literal(sessions__4__returncode=-15, sessions__4__killed={"sigterm": True, "group_gone": True},
                                 execution_status="failed", stop={**NZ_C5, "returncode": -15})),
        ["batch stop records a nonzero exit of 'c5' (-15), but the last launched session 'c5' exited -15 (killed"]),
    "failed_without_stop_all_exits_zero": (
        dict(batch_edit=_literal(execution_status="failed", stop=None)),
        ["batch execution_status 'failed' without a stop reason"]),
    "no_stop_but_a_session_never_launched": (
        dict(batch_edit=lambda b: (b["sessions"].pop(), b.update(not_launched=["c5"], execution_status="failed",
                                                                 stop=None, elapsed_s=750.0))),
        ["batch has no stop, yet sessions were never launched: ['c5']", "batch execution_status 'failed' without a stop"]),
    "exit_0_without_a_report_but_no_stop": (
        dict(batch_edit=_literal(sessions__2__session_json={"present": False, "parses": False, "sha256": None, "bytes": None})),
        ["batch c3: exit 0, session file {'present': False", "the wrapper would have stopped here"]),
    "elapsed_not_the_last_end": (
        dict(batch_edit=_literal(elapsed_s=900.0)),
        ["batch elapsed_s 900.0 is not the last session's end (950.0)"]),
    "overlapping_session_times": (
        dict(batch_edit=_literal(sessions__2__started_s=300.0)),
        ["are missing, not finite, negative or overlapping"]),
    # review 1 (correction cycle): the recorded session-file state must be the actual one
    "recorded_file_absent_but_present": (
        dict(batch_edit=_literal(sessions__2__session_json__present=False)),
        ["batch c3: session file differs from the one the batch recorded (recorded {'present': False"]),
    "recorded_file_size_differs": (
        dict(batch_edit=_literal(sessions__2__session_json__bytes=1)),
        ["batch c3: session file differs from the one the batch recorded"]),
    "c5_recorded_unparseable_with_ab_failure": (  # complete c5 A/B failure would be exit 4; this is exit 3
        dict(edits={"c5": "fail_ab"}, exits={"c5": 1}, batch_edit=_literal(sessions__4__session_json__parses=False)),
        ["batch c5: session file differs from the one the batch recorded"]),
    # review 1 (correction cycle): non-finite times defeat every comparison
    "nan_intermediate_end": (
        dict(batch_edit=_literal(sessions__2__ended_s=float("nan"))),
        ["batch record batch_record.json missing or unreadable: non-finite number NaN"]),
    "infinite_elapsed": (
        dict(batch_edit=_literal(elapsed_s=float("inf"))),
        ["batch record batch_record.json missing or unreadable: non-finite number Infinity"]),
    "nan_launch_time": (
        dict(batch_edit=_literal(launched_unix=float("nan"))),
        ["non-finite number NaN"]),
    "nan_freeze_time": (
        dict(manifest_edit=_set(("frozen_unix",), float("nan"))),
        ["missing or unreadable: non-finite number NaN"]),
    "nan_session_start": (
        dict(edits={"c3": _set(("meta", "started_unix"), float("nan"))}),
        ["c3: session started at nan"]),
}


@pytest.mark.parametrize("name", list(CONTRADICTORY))
def test_contradictory_execution_record_is_rejected_before_analysis(tmp_path, frozen, base_reports, name):
    kw, expect = CONTRADICTORY[name]
    kw = dict(kw)
    named = {"fail_strict": _fail_strict, "fail_ab": _fail_ab}
    if kw.get("edits") and all(isinstance(v, str) for v in kw["edits"].values()):
        kw["edits"] = {sid: named[v] for sid, v in kw["edits"].items()}
    run, mpath = make_run(tmp_path, frozen, base_reports, **kw)
    code, val = cli(run, mpath, tmp_path / "out", "--expect-freeze-sha256", mc.sha256_file(mpath))
    assert_rejected(code, val, tmp_path / "out", expect=expect)  # exit 3, no performance output, not confirmed


def test_wrapper_records_satisfy_the_consistency_rules_and_tampering_fails(tmp_path):
    """Records written by the real wrapper (dummy children): the two accepted shapes pass the
    consistency rules; a deadline stop is a different rejection (not a contradiction); and the
    re-review's contradiction, applied to a real record, is caught."""
    def real(modes, sub, **kw):
        d = tmp_path / sub
        d.mkdir()
        m = dummy_manifest(d, modes)
        m.update(kw)
        return run(d, m, grace_s=2.0), m, d
    ok, m_ok, d_ok = real(["ok", "ok"], "ok")
    nz, m_nz, d_nz = real(["ok", "exit1_oom", "ok"], "nz")
    dl, m_dl, d_dl = real(["ok", "hang"], "dl", budget_s=1.0)
    for rec in (ok, nz, dl):
        assert mc.execution_consistency_problems(rec, rec["sessions"]) == [], rec
    assert ok["execution_status"] == "succeeded" and nz["stop"]["reason"] == "nonzero_exit"
    assert mc.batch_problems(ok, {**m_ok, "frozen_unix": 0}, "a" * 64, d_ok / "run", dummy_spec(m_ok)) == []
    nz_problems = mc.batch_problems(nz, {**m_nz, "frozen_unix": 0}, "a" * 64, d_nz / "run", dummy_spec(m_nz))
    assert nz_problems == ["planned sessions never launched, so their evidence is missing: ['d3']"]
    assert any("stopped by" in p for p in mc.batch_problems(dl, {**m_dl, "frozen_unix": 0}, "a" * 64, d_dl / "run",
                                                             dummy_spec(m_dl)))
    forged = {**ok, "execution_status": "failed",
              "stop": {"reason": "nonzero_exit", "session": "d2", "returncode": 1, "detail": "exit code 1"}}
    assert any("but the last launched session 'd2' exited 0" in p
               for p in mc.execution_consistency_problems(forged, forged["sessions"]))
    hidden = {**nz, "execution_status": "succeeded", "stop": None, "not_launched": []}
    assert any("exited nonzero, yet stop None" in p for p in mc.execution_consistency_problems(hidden, hidden["sessions"]))


def test_manifest_must_match_the_out_of_band_sha256(tmp_path, frozen, base_reports):
    run, mpath = make_run(tmp_path, frozen, base_reports)
    code, val = cli(run, mpath, tmp_path / "out", "--expect-freeze-sha256", "6" * 64)
    assert_rejected(code, val, tmp_path / "out", expect=["!= the expected 6666"])
    code, val = cli(run, mpath, tmp_path / "out2", "--expect-freeze-sha256", mc.sha256_file(mpath))
    assert code == 0 and val["status"] == "valid"


def test_missing_manifest_is_invalid(tmp_path, frozen, base_reports):
    run, _ = make_run(tmp_path, frozen, base_reports)
    code, val = cli(run, tmp_path / "nonexistent.json", tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=["freeze manifest", "missing or unreadable"])


def _fail_strict(rep):
    """A detailed strict check fails, and every summary records it (as the runner writes it)."""
    c = rep["correctness"][1]
    c["batched"]["strict_host_eq_resident_bitwise"] = False
    c["batched"]["strict_pass"] = False
    c["strict_pass_all"] = False
    rep["meta"]["correctness_all"] = False


def _fail_ab(rep):
    """A detailed A/B check fails, and every summary records it (as the runner writes it)."""
    a = rep["ab_checks"][0]
    a["host"]["first_logits_bitwise_equal"] = False
    a["pass"] = False
    rep["meta"]["correctness_all"] = False


def test_ab_failure_in_the_last_session_exits_4(tmp_path, frozen, base_reports):
    """Positive control for the file-state check: complete, consistent c5 A/B failure, exit 4."""
    run, mpath = make_run(tmp_path, frozen, base_reports, edits={"c5": _fail_ab}, exits={"c5": 1})
    code, val = cli(run, mpath, tmp_path / "out", "--expect-freeze-sha256", mc.sha256_file(mpath))
    assert_rejected(code, val, tmp_path / "out", status="correctness_failed")
    assert code == 4 and val["problems"] == [] and len(val["correctness_failures"]) == 1
    assert "A/B" in val["correctness_failures"][0] and val["correctness_failures"][0].startswith("c5: ")


def test_non_finite_times_are_rejected_by_the_checks_themselves():
    """Direct callers (no JSON parsing) get the same rejection: NaN never passes a comparison."""
    entries = [{"session": "d1", "returncode": 0, "killed": False, "started_s": 0.0, "ended_s": float("nan"),
                "session_json": {"present": True, "parses": True, "sha256": "x", "bytes": 1}},
               {"session": "d2", "returncode": 0, "killed": False, "started_s": 10.0, "ended_s": 20.0,
                "session_json": {"present": True, "parses": True, "sha256": "y", "bytes": 1}}]
    rec = {"execution_status": "succeeded", "stop": None, "not_launched": [], "elapsed_s": 20.0, "sessions": entries}
    assert any("not finite" in p for p in mc.execution_consistency_problems(rec, entries))
    assert not mc.finite(float("nan")) and not mc.finite(float("inf")) and not mc.finite(True) and mc.finite(1.5)
    with pytest.raises(ValueError, match="non-finite number NaN"):
        mc.load_json_finite('{"elapsed_s": NaN}')


# ---- review 2 (P2): a JSON integer beyond the float range in a time field. Before the fix,
# finite() passed it to math.isfinite(), which raised OverflowError: the analyzer exited 1 with
# a traceback and no m3_validation.json. Every such time is now malformed evidence (exit 3).
HUGE = 10 ** 400


def _replace_in_batch_record(old, new):
    """Edit the written batch record's text: for number literals json.dumps cannot emit."""
    def after(run):
        p = run / mc.BATCH_RECORD
        text = p.read_text()
        assert text.count(old) == 1, old
        p.write_text(text.replace(old, new))
    return after


def _t_wall(index, value):
    def edit(rep):
        rep["samples"][index]["t_wall"] = value
    return edit


NUMERIC = {
    "huge_integer_intermediate_end": (  # the exact review-2 trigger
        dict(batch_edit=_literal(sessions__2__ended_s=HUGE)),
        ["batch c3: ended at 10000000000", "s, not a finite time before the 1800 s deadline"]),
    "huge_negative_integer_start": (
        dict(batch_edit=_literal(sessions__2__started_s=-HUGE)),
        ["batch session times", "are missing, not finite, negative or overlapping"]),
    "huge_integer_elapsed": (
        dict(batch_edit=_literal(elapsed_s=HUGE)),
        ["batch elapsed 10000000000", "is missing, not finite or not below the 1800 s budget"]),
    "huge_integer_launch_time": (dict(batch_edit=_literal(launched_unix=HUGE)), ["batch launched at 10000000000"]),
    "huge_integer_freeze_time": (
        dict(manifest_edit=_set(("frozen_unix",), HUGE)), ["freeze manifest frozen_unix 10000000000", "missing or not finite"]),
    "huge_integer_session_start": (
        dict(edits={"c3": _set(("meta", "started_unix"), HUGE)}), ["c3: session started at 10000000000"]),
    "huge_integer_sample_wall_time": (  # the last sample: still in file order
        dict(edits={"c3": _t_wall(-1, HUGE)}), ["c3: sample", "fails the validity rules, so its pair is incomplete: ['t_wall']"]),
    "boolean_sample_wall_time": (  # the first sample: True (== 1) is in file order; it was accepted before the fix
        dict(edits={"c1": _t_wall(0, True)}), ["c1: sample 0: malformed process-warmup record: ['t_wall']"]),
    "boolean_session_end": (dict(batch_edit=_literal(sessions__2__ended_s=True)), ["batch c3: ended at True s"]),
    "largest_float_integer_end": (  # converts to a float: rejected by the deadline comparison, not by overflow
        dict(batch_edit=_literal(sessions__2__ended_s=int(sys.float_info.max))),
        [f"batch c3: ended at {int(sys.float_info.max)} s, not a finite time before the 1800 s deadline"]),
    "overflowing_float_literal_end": (  # P3: 1e400 parses to inf; finite() rejects it at the comparison
        dict(after=_replace_in_batch_record('"ended_s": 550.0', '"ended_s": 1e400')),
        ["batch c3: ended at inf s, not a finite time before the 1800 s deadline"]),
    "integer_beyond_the_parser_digit_limit": (  # Python refuses to parse it (int max str digits)
        dict(after=_replace_in_batch_record('"ended_s": 550.0', '"ended_s": 1' + "0" * 5000)),
        ["batch record batch_record.json missing or unreadable: Exceeds the limit"]),
}


@pytest.mark.parametrize("name", list(NUMERIC))
def test_out_of_range_or_boolean_times_are_invalid_evidence_not_a_crash(tmp_path, frozen, base_reports, name):
    kw, expect = NUMERIC[name]
    run, mpath = make_run(tmp_path, frozen, base_reports, **kw)
    code, val = cli(run, mpath, tmp_path / "out", "--expect-freeze-sha256", mc.sha256_file(mpath))
    assert_rejected(code, val, tmp_path / "out", expect=expect)  # exit 3, report written, no performance output
    assert val["correctness_failures"] == []


def test_huge_integer_end_through_the_real_cli_subprocess(tmp_path, frozen, base_reports):
    """The review-2 reproduction through the real CLI process. Before the fix: exit 1,
    OverflowError traceback, no m3_validation.json."""
    run, mpath = make_run(tmp_path, frozen, base_reports, batch_edit=_literal(sessions__2__ended_s=HUGE))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "m3_analyze.py"), "--spec", SPEC["name"],
                        "--run-dir", str(run), "--out-dir", str(out), "--freeze", str(mpath),
                        "--expect-freeze-sha256", mc.sha256_file(mpath)],
                       cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 3 and "Traceback" not in r.stderr, r.stderr[-2000:]
    assert sorted(p.name for p in out.iterdir()) == ["m3_validation.json"]
    val = json.loads((out / "m3_validation.json").read_text())
    assert_rejected(r.returncode, val, out, expect=["batch c3: ended at 10000000000"])


def test_finite_accepts_exactly_the_numbers_that_convert_to_finite_floats():
    fmax = sys.float_info.max
    for x in (0, 1800, -3, 1.5, T_LAUNCH, int(T_LAUNCH), int(fmax), fmax, 2 ** 53 + 1):
        assert mc.finite(x), x
    for x in (HUGE, -HUGE, 2 ** 1024, int(fmax) * 2, True, False, float("nan"), float("inf"), float("-inf"),
              None, "1", [1]):
        assert not mc.finite(x), repr(x)[:40]
    # the checks compare such values without converting them: no exception, a reported problem
    entries = [{"session": "d1", "returncode": 0, "killed": False, "started_s": 0, "ended_s": HUGE,
                "session_json": {"present": True, "parses": True, "sha256": "x", "bytes": 1}}]
    rec = {"execution_status": "succeeded", "stop": None, "not_launched": [], "elapsed_s": HUGE, "sessions": entries}
    assert any("not finite" in p for p in mc.execution_consistency_problems(rec, entries))
    # P3: an overflowing literal is not rejected while parsing; it parses to inf, which finite() rejects
    parsed = mc.load_json_finite('{"ended_s": 1e400}')
    assert parsed["ended_s"] == float("inf") and not mc.finite(parsed["ended_s"])


def test_ordinary_integer_times_remain_valid(tmp_path, frozen, base_reports):
    """Positive control: integral JSON times (no fraction) are ordinary finite numbers."""
    def int_times(b):
        for i, e in enumerate(b["sessions"]):
            e["started_s"], e["ended_s"] = 200 * i, 200 * i + 150
        b["elapsed_s"], b["launched_unix"] = 950, int(T_LAUNCH)
    run, mpath = make_run(tmp_path, frozen, base_reports, batch_edit=int_times,
                          manifest_edit=_set(("frozen_unix",), int(T_FREEZE)))
    code, val = cli(run, mpath, tmp_path / "out", "--expect-freeze-sha256", mc.sha256_file(mpath))
    assert code == 0 and val["status"] == "valid" and val["problems"] == [], val["problems"][:5]
    assert json.loads((tmp_path / "out" / "m3_analysis.json").read_text())["decision"]["m3_confirm_v1_confirmed"] is True


def test_correctness_failure_in_the_last_session_exits_4(tmp_path, frozen, base_reports, capsys):
    """c5 records a correctness failure and exits 1. The wrapper stops (nothing is left to launch)
    and reports an execution failure, but every planned record is present and consistent and
    the batch stayed within its budget: complete evidence with a failed check, exit 4."""
    run, mpath = make_run(tmp_path, frozen, base_reports, edits={"c5": _fail_strict}, exits={"c5": 1})
    rec = json.loads((run / mc.BATCH_RECORD).read_text())
    assert rec["execution_status"] == "failed" and rec["stop"]["reason"] == "nonzero_exit" and mc.exit_code(rec) == 3
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", status="correctness_failed")
    assert code == 4 and val["problems"] == []
    case = tuple(base_reports["c5"]["meta"]["block_order"][1])
    assert val["correctness_failures"] == [f"c5: correctness {case} batched: failed ['host_eq_resident_bitwise']"]
    assert val["correctness_failures_observed"] is True
    assert val["provenance"]["execution"]["stop"]["session"] == "c5"
    assert val["decision"]["observed_correctness_failures"] == val["correctness_failures"]


def test_correctness_failure_before_the_last_session_is_invalid_but_reported(tmp_path, frozen, base_reports, capsys):
    """c2 records a correctness failure and exits 1: the wrapper launches nothing further, so the
    evidence of c3-c5 is missing (exit 3). The recorded failure is still reported explicitly."""
    run, mpath = make_run(tmp_path, frozen, base_reports, edits={"c2": _fail_strict}, exits={"c2": 1})
    assert sorted(p.name for p in run.glob("session_*.json")) == ["session_c1.json", "session_c2.json"]
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", status="invalid", expect=[
        "planned sessions never launched, so their evidence is missing: ['c3', 'c4', 'c5']",
        "missing session file session_c3.json"])
    case = tuple(base_reports["c2"]["meta"]["block_order"][1])
    failure = f"c2: correctness {case} batched: failed ['host_eq_resident_bitwise']"
    assert val["correctness_failures"] == [failure] and val["correctness_failures_observed"] is True
    assert val["decision"]["observed_correctness_failures"] == [failure]
    assert not any("nonzero" in p and "c2" in p for p in val["problems"])  # the exit itself is not an evidence problem
    out = capsys.readouterr().out
    assert "observed correctness / A/B failures: 1" in out and failure in out and "'nonzero_exit'" in out


def test_oom_report_in_the_last_session_is_invalid(tmp_path, frozen, base_reports):
    """The runner caught an OOM, kept going and exited 1 with a report: exit 3, not exit 4."""
    def oom(rep):
        rep["failures"] = [{"block": 7, "L": 1280, "reason": "CUDA OOM: synthetic"}]
    run, mpath = make_run(tmp_path, frozen, base_reports, edits={"c5": oom}, exits={"c5": 1})
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=["c5: recorded block failures"])
    assert val["correctness_failures"] == []


def test_all_processes_ran_but_a_report_is_empty(tmp_path, frozen, base_reports):
    """Five processes exiting 0 with five parseable JSON files is not completeness by itself."""
    run, mpath = make_run(tmp_path, frozen, {**base_reports, "c3": {}})
    rec = json.loads((run / mc.BATCH_RECORD).read_text())
    assert rec["execution_status"] == "succeeded" and len(list(run.glob("session_*.json"))) == 5
    code, val = cli(run, mpath, tmp_path / "out")
    assert_rejected(code, val, tmp_path / "out", expect=["c3: missing sections"])


def test_cli_usage_guards():
    with pytest.raises(SystemExit) as e:  # the confirm run directory has a closed listing
        m3_analyze.main(["--spec", SPEC["name"], "--run-dir", "results/m3_confirm/x"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        m3_analyze.main(["--spec", SPEC["name"], "--run-dir", "/tmp/x", "--out-dir", "/tmp/x/analysis"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:  # the old spec has no freeze manifest
        m3_analyze.main(["--run-dir", str(M3_EVAL), "--freeze", "x.json"])
    assert e.value.code == 2


# ---------------------------------------------------------------------------------------------
# premeasurement freeze
def test_freeze_refuses_a_draft_protocol_and_a_different_stack():
    with pytest.raises(mc.Refused) as e:
        mc.build_manifest(ROOT, SPEC, stack=STACK, now=T_FREEZE)
    assert any("still marked DRAFT" in p for p in e.value.problems)
    other = {**STACK, "packages": {**STACK["packages"], "transformers": "5.23.0"}, "driver_version": "600.10"}
    with pytest.raises(mc.Refused) as e:
        mc.build_manifest(ROOT, SPEC, stack=other, now=T_FREEZE, require_frozen_protocol=False)
    assert any("transformers" in p and "('5.23.0', '5.22.1')" in p for p in e.value.problems)
    assert any("driver_version '600.10', required '596.49'" in p for p in e.value.problems)
    assert not (ROOT / SPEC["freeze_manifest"]).exists() and not (ROOT / SPEC["run_dir"]).exists()


def test_manifest_records_the_tree_without_hashing_itself(frozen):
    m, _ = frozen
    src = mc.source_identity(ROOT)
    assert m["base_git_head"] == src["git_head"] and m["git_diff_head_sha256"] == src["git_diff_head_sha256"]
    assert m["files_sha256"] == src["files_sha256"]
    for f in ("kvlab/m3_confirm.py", "scripts/m3_confirm.py", "tests/test_m3_confirm.py", *mc.ANALYSIS_FILES,
              *SPEC["executed_sources"]):
        assert m["files_sha256"][f] == mc.sha256_file(ROOT / f), f  # actual bytes, untracked files included
    assert set(m["documents_sha256"]) == set(mc.DOCUMENTS)
    assert not any(p.startswith("results/m3_confirm/") for p in m["untracked_files_sha256"])
    untracked = [p for p in subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.splitlines() if p]
    assert set(m["untracked_files_sha256"]) == {p for p in untracked if not p.startswith("results/m3_confirm/")}
    assert m["commands"] == mc.commands_for(SPEC) and m["budget_s"] == 1800 and m["stack"] == STACK
    assert mc.manifest_problems(m, SPEC, ROOT) == []


def test_freeze_writes_once_and_verify_detects_drift(tmp_path):
    spec = {**SPEC, "freeze_manifest": str(tmp_path / "f" / "M3_CONFIRM_V1.freeze.json"), "run_dir": str(tmp_path / "run")}
    path = mc.freeze(ROOT, spec, stack=STACK, now=T_FREEZE, require_frozen_protocol=False)
    side = path.parent / "M3_CONFIRM_V1.freeze.diff"
    manifest = json.loads(path.read_text())
    assert side.is_file() and mc.sha256_file(side) == manifest["git_diff_file_sha256"]
    clean_env = {k: v for k, v in os.environ.items() if k not in spec["required_process_env"] and k != "CUDA_VISIBLE_DEVICES"}
    assert mc.verify_tree(ROOT, manifest, spec, stack=STACK, environ=clean_env) == []
    assert mc.verify_tree(ROOT, manifest, spec, stack=STACK, environ={**clean_env, "CUDA_LAUNCH_BLOCKING": "1",
                                                                      "CUDA_VISIBLE_DEVICES": "0"}) == [
        "environment CUDA_LAUNCH_BLOCKING='1', required None",
        "CUDA_VISIBLE_DEVICES is set ('0'); the M3 sessions ran without it"]
    with pytest.raises(mc.Refused, match="already exists"):
        mc.freeze(ROOT, spec, stack=STACK, now=T_FREEZE, require_frozen_protocol=False)
    drift = mc.verify_tree(ROOT, manifest, spec, stack={**STACK, "kernel": "6.19.0"}, environ=clean_env)
    assert drift == ["current kernel '6.19.0' != frozen '6.18.54.1-microsoft-standard-WSL2'"]
    bad = {**manifest, "files_sha256": {**manifest["files_sha256"], "kvlab/m3_bench.py": "7" * 64}}
    assert any("source files differ from the freeze: ['kvlab/m3_bench.py']" in p
               for p in mc.verify_tree(ROOT, bad, spec, stack=STACK, environ=clean_env))
    side.write_text("changed")
    assert mc.verify_tree(ROOT, manifest, spec, stack=STACK, environ=clean_env) == [
        f"diff sidecar {side.name} is missing or differs"]


def test_run_refuses_without_a_matching_manifest(tmp_path, frozen, capsys):
    """The CLI refusal paths launch nothing. A real manifest with one wrong hash, and a wrong
    expected SHA-256, guarantee the refusal, so no GPU process can start here."""
    bad = copy.deepcopy(frozen[0])
    bad["files_sha256"]["scripts/m3_measure.py"] = "8" * 64
    p = write_manifest(tmp_path, bad)
    assert mc.main_run(SPEC, ROOT, str(p), expect_sha="9" * 64) == 1
    out = capsys.readouterr().out
    assert "launch REFUSED" in out and "scripts/m3_measure.py" in out and "!= expected 9999" in out
    assert not (ROOT / SPEC["run_dir"]).exists()
    assert mc.main_run(SPEC, ROOT, str(tmp_path / "none.json")) == 1
    assert mc.main_verify(SPEC, ROOT, str(tmp_path / "none.json")) == 1


# ---------------------------------------------------------------------------------------------
# budgeted batch wrapper, with dummy child processes only (no model, no CUDA)
DUMMY = textwrap.dedent("""
    import json, os, signal, subprocess, sys, time
    sid, mode, run, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]

    def report(obj):
        with open(os.path.join(run, f"session_{sid}.json"), "w") as f:
            f.write(json.dumps(obj))
    open(os.path.join(run, f"session_{sid}.gpu_monitor.csv"), "w").write("x\\n")
    print(f"start {sid}", flush=True)
    if mode == "grandchild":  # a helper in the session's process group, like the GPU monitor
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open(os.path.join(run, f"grandchild_{sid}.pid"), "w").write(str(g.pid))
        time.sleep(120)
    elif mode == "hang":
        time.sleep(120)
    elif mode == "signal_wrapper":
        os.kill(int(extra[0]), signal.SIGTERM)
        time.sleep(120)
    elif mode == "ignore_term_and_signal":  # needs SIGKILL; keeps signalling the wrapper meanwhile
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for _ in range(6000):
            os.kill(int(extra[0]), signal.SIGTERM)
            time.sleep(0.02)
    elif mode == "wait_release":  # exits 0 with a report once the test releases it
        while not os.path.exists(os.path.join(run, "release")):
            time.sleep(0.005)
        report({"meta": {"correctness_all": True, "n_excluded": 0}, "failures": []})
    elif mode == "crash":
        sys.exit(2)
    elif mode == "no_report":
        sys.exit(0)
    elif mode == "exit1_oom":  # what the runner writes after a caught OOM: parseable but incomplete
        report({"meta": {"correctness_all": True, "n_excluded": 0, "n_measured": 0},
                "failures": [{"block": 0, "reason": "CUDA OOM"}], "samples": []})
        sys.exit(1)
    elif mode == "exit1_empty":
        report({})
        sys.exit(1)
    else:  # ok
        report({"meta": {"correctness_all": True, "n_excluded": 0}, "failures": []})
""")


def dummy_manifest(tmp_path, modes, extra=()):
    script = tmp_path / "dummy_session.py"
    script.write_text(DUMMY)
    return {"experiment": SPEC["name"], "run_dir": "run", "budget_s": 1800,
            "commands": [{"session": f"d{i}", "argv": [sys.executable, str(script), f"d{i}", m, "run", *extra],
                          "env": {"HF_HUB_OFFLINE": "1"}} for i, m in enumerate(modes, 1)]}


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers kill(0); it is dead for our purpose
        return Path(f"/proc/{pid}/stat").read_text().split(")")[1].split()[0] != "Z"
    except OSError:
        return False


def dummy_spec(m):
    return {**SPEC, "budget_s": m["budget_s"], "sessions": {c["session"]: 0 for c in m["commands"]}}


def run(tmp_path, m, **kw):
    kw.setdefault("log", lambda s: None)
    return mc.run_batch(tmp_path, m, "a" * 64, verification={"problems": []}, **kw)


def current_mask():
    return signal.pthread_sigmask(signal.SIG_BLOCK, [])


def assert_no_live_session(rec):
    for e in rec["sessions"]:
        assert e["pid"] is None or not _alive(e["pid"]), e
        assert e["pid"] is None or e["returncode"] is not None, e  # reaped


def test_batch_runs_every_session_in_order(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "ok", "ok"])
    mask = current_mask()
    rec = run(tmp_path, m)
    assert current_mask() == mask  # the signal mask is restored
    assert rec["execution_status"] == "succeeded" and rec["stop"] is None and rec["not_launched"] == []
    assert mc.exit_code(rec) == 0
    assert [(e["session"], e["returncode"], e["killed"]) for e in rec["sessions"]] == [
        ("d1", 0, False), ("d2", 0, False), ("d3", 0, False)]
    assert all(e["session_json"]["parses"] and e["monitor_csv_present"] for e in rec["sessions"])
    starts = [e["started_s"] for e in rec["sessions"]]
    assert starts[0] < 1.0 and all(e["ended_s"] <= n for e, n in zip(rec["sessions"], starts[1:]))  # sequential
    assert rec["elapsed_s"] == rec["sessions"][-1]["ended_s"] <= rec["finalized_s"]
    assert json.loads((tmp_path / "run" / mc.BATCH_RECORD).read_text()) == rec
    assert (tmp_path / "run" / "logs" / "session_d2.log").read_text() == "start d2\n"
    assert mc.batch_problems(rec, {**m, "frozen_unix": rec["launched_unix"] - 1}, "a" * 64, tmp_path / "run",
                             dummy_spec(m)) == []
    with pytest.raises(FileExistsError):  # never reuses a run directory
        run(tmp_path, m)


@pytest.mark.parametrize("mode, rc", [("exit1_oom", 1), ("exit1_empty", 1), ("crash", 2)])
def test_any_nonzero_exit_stops_the_batch(tmp_path, mode, rc):
    """Review finding 1: a parseable report (an OOM-like incomplete one, or an empty object) does
    not let the batch continue after exit 1; nor does any other nonzero exit."""
    m = dummy_manifest(tmp_path, ["ok", mode, "ok"])
    rec = run(tmp_path, m)
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3
    assert rec["stop"]["reason"] == "nonzero_exit" and rec["stop"]["session"] == "d2" and rec["stop"]["returncode"] == rc
    assert [e["session"] for e in rec["sessions"]] == ["d1", "d2"] and rec["not_launched"] == ["d3"]
    assert not (tmp_path / "run" / "logs" / "session_d3.log").exists()  # never launched
    e = rec["sessions"][1]
    assert e["returncode"] == rc and e["killed"] is False
    assert (tmp_path / "run" / "logs" / "session_d2.log").read_text() == "start d2\n" and e["monitor_csv_present"]
    if mode == "crash":
        assert e["session_json"]["present"] is False
    else:  # the report is kept and hashed; what the runner recorded is shown, not trusted
        assert e["session_json"]["parses"] and e["session_json"]["sha256"] == mc.sha256_file(tmp_path / "run" / "session_d2.json")
        assert e["runner_reported"] == ({"correctness_all": True, "n_block_failures": 1, "n_excluded": 0,
                                         "n_measured": 0, "n_samples": None} if mode == "exit1_oom" else
                                        {"correctness_all": None, "n_block_failures": None, "n_excluded": None,
                                         "n_measured": None, "n_samples": None})
    assert_no_live_session(rec)


def test_nonzero_exit_of_the_last_session_is_an_execution_failure(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "exit1_oom"])
    rec = run(tmp_path, m)
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3 and rec["not_launched"] == []
    assert rec["stop"] == {"reason": "nonzero_exit", "session": "d2", "returncode": 1,
                           "detail": rec["stop"]["detail"]} and "n_block_failures': 1" in rec["stop"]["detail"]
    # not by itself an evidence problem: the analysis judges the report (here it would be invalid)
    assert mc.batch_problems(rec, {**m, "frozen_unix": 0}, "a" * 64, tmp_path / "run", dummy_spec(m)) == []


def test_exit_0_without_a_report_stops_the_batch(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "no_report", "ok"])
    rec = run(tmp_path, m)
    assert rec["execution_status"] == "failed" and rec["stop"]["reason"] == "missing_report"
    assert rec["not_launched"] == ["d3"] and rec["sessions"][1]["returncode"] == 0


def test_deadline_while_running_kills_only_the_session_group(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "grandchild", "ok"])
    m["budget_s"] = 3.0
    scratch = mc.scratch_dir(tmp_path, "run", "d2")
    scratch.mkdir(parents=True)
    (scratch / "chunk.kvc").write_bytes(b"x")
    keep = [mc.scratch_dir(tmp_path, "run", "d3"), mc.scratch_dir(tmp_path, "other_run", "d2")]  # not this session's
    for k in keep:
        k.mkdir(parents=True)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        t0 = time.monotonic()
        rec = run(tmp_path, m, grace_s=2.0)
        took = time.monotonic() - t0
        assert rec["execution_status"] == "failed" and rec["not_launched"] == ["d3"] and mc.exit_code(rec) == 3
        assert rec["stop"]["reason"] == "deadline" and rec["stop"]["session"] == "d2"
        assert "still running at the deadline" in rec["stop"]["detail"]
        e = rec["sessions"][1]
        assert e["killed"]["sigterm"] is True and e["killed"]["group_gone"] is True and e["returncode"] < 0
        assert e["session_json"]["present"] is False and e["monitor_csv_present"] is True
        assert (tmp_path / "run" / "logs" / "session_d2.log").read_text() == "start d2\n"
        assert not _alive(int((tmp_path / "run" / "grandchild_d2.pid").read_text()))  # its helper went too
        assert _alive(bystander.pid)  # a process outside the group is untouched
        assert not scratch.exists() and all(k.exists() for k in keep) and e["scratch_removed"] == ".kvcache/m3/run/d2"
        assert 3.0 <= rec["elapsed_s"] < 3.5 and took < 3.0 + 2 * 2.0 + 5
        assert_no_live_session(rec)
        problems = mc.batch_problems(rec, {**m, "frozen_unix": 0}, "a" * 64, tmp_path / "run", dummy_spec(m))
        assert any("stopped by" in p and "deadline" in p for p in problems)
    finally:
        bystander.kill()
        bystander.wait()


class ExitDrivenClock:
    """A deterministic clock for the deadline race (review finding 3). It reads 0.0 until the
    session's process has exited, and far past the deadline afterwards. Its first reading after
    the launch was recorded releases the dummy session, so the session can exit only while the
    wrapper is waiting, i.e. between two observations."""

    def __init__(self, run_dir):
        self.run_dir, self.pid = run_dir, None

    def __call__(self):
        if self.pid is None:
            try:
                self.pid = json.loads((self.run_dir / mc.BATCH_RECORD).read_text())["sessions"][0]["pid"]
            except (OSError, ValueError, IndexError, KeyError):
                return 0.0
            if self.pid is None:
                return 0.0
        if not _alive(self.pid):
            return 1e6
        (self.run_dir / "release").touch()
        return 0.0


def test_exit_observed_after_the_deadline_is_not_success(tmp_path):
    """The reviewed trigger: the last session exits 0 with a report, but only after the deadline
    (here: between two observations). Before the fix this returned status complete."""
    m = dummy_manifest(tmp_path, ["wait_release"])
    m["budget_s"] = 10.0
    rec = run(tmp_path, m, clock=ExitDrivenClock(tmp_path / "run"))
    e = rec["sessions"][0]
    assert e["returncode"] == 0 and e["killed"] is False and e["session_json"]["parses"]  # it did finish
    assert rec["stop"]["reason"] == "deadline" and "first observed after the deadline" in rec["stop"]["detail"]
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3
    problems = mc.batch_problems(rec, {**m, "frozen_unix": 0}, "a" * 64, tmp_path / "run", dummy_spec(m))
    assert any("ended at 1000000.0 s, not a finite time before the 10.0 s deadline" in p for p in problems)
    assert any("stopped by" in p for p in problems)


def test_deadline_is_rechecked_before_the_next_launch(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "ok"])
    m["budget_s"] = 10.0
    rec_path = tmp_path / "run" / mc.BATCH_RECORD

    def clock():  # past the deadline as soon as the first session's end has been recorded
        try:
            return 1e6 if json.loads(rec_path.read_text())["sessions"][0]["ended_s"] is not None else 0.0
        except (OSError, ValueError, IndexError, KeyError):
            return 0.0
    rec = run(tmp_path, m, clock=clock)
    assert rec["sessions"][0]["returncode"] == 0 and rec["not_launched"] == ["d2"]
    assert rec["stop"] == {"reason": "deadline", "session": None, "detail": "deadline reached before launching d2"}
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP], ids=lambda s: s.name)
def test_signal_right_after_launch_terminates_and_reaps_the_session(tmp_path, sig):
    """Review finding 2: a stop signal delivered in the launch-to-wait interval (here by the
    launch-log callback, as in the review) ends with the session's group terminated and reaped,
    before run_batch returns. Signals are blocked and received synchronously, so none can land
    inside Popen or the cleanup."""
    m = dummy_manifest(tmp_path, ["hang", "ok"])
    mask = current_mask()

    def log(s):
        if s.startswith("launched"):
            os.kill(os.getpid(), sig)
    rec = run(tmp_path, m, grace_s=2.0, log=log)
    assert current_mask() == mask
    assert rec["stop"]["reason"] == "operator_signal" and rec["stop"]["session"] == "d1"
    assert sig.name in rec["stop"]["detail"] and rec["not_launched"] == ["d2"]
    e = rec["sessions"][0]
    assert e["killed"]["sigterm"] is True and e["killed"]["group_gone"] is True and e["returncode"] < 0
    assert_no_live_session(rec)
    assert rec["execution_status"] == "failed" and mc.exit_code(rec) == 3


def test_signal_between_sessions_launches_nothing_further(tmp_path):
    m = dummy_manifest(tmp_path, ["ok", "ok"])

    def log(s):
        if s.startswith("finished d1"):
            os.kill(os.getpid(), signal.SIGTERM)
    rec = run(tmp_path, m, log=log)
    assert rec["stop"] == {"reason": "operator_signal", "session": None, "detail": "SIGTERM before launching d2"}
    assert rec["not_launched"] == ["d2"] and rec["sessions"][0]["returncode"] == 0
    assert_no_live_session(rec)


def test_wrapper_error_after_launch_cleans_up_then_raises(tmp_path):
    m = dummy_manifest(tmp_path, ["hang", "ok"])

    def log(s):
        if s.startswith("launched"):
            raise RuntimeError("synthetic wrapper failure")
    with pytest.raises(RuntimeError, match="synthetic wrapper failure"):
        run(tmp_path, m, grace_s=2.0, log=log)
    rec = json.loads((tmp_path / "run" / mc.BATCH_RECORD).read_text())
    assert rec["stop"]["reason"] == "wrapper_error" and rec["stop"]["session"] == "d1"
    assert rec["execution_status"] == "failed" and rec["not_launched"] == ["d2"]
    assert rec["sessions"][0]["killed"]["group_gone"] is True
    assert_no_live_session(rec)


def test_cleanup_escalates_and_further_signals_cannot_abort_it(tmp_path):
    """The session ignores SIGTERM and keeps signalling the wrapper: the wrapper escalates to
    SIGKILL within the bounded grace, finishes its record and returns normally; later signals
    are only recorded."""
    m = dummy_manifest(tmp_path, ["ignore_term_and_signal", "ok"], extra=(str(os.getpid()),))
    mask = current_mask()
    rec = run(tmp_path, m, grace_s=1.0)
    assert current_mask() == mask
    e = rec["sessions"][0]
    assert rec["stop"]["reason"] == "operator_signal" and rec["not_launched"] == ["d2"]
    assert e["killed"]["sigterm"] is True and e["killed"]["sigkill"] is True and e["killed"]["group_gone"] is True
    assert e["returncode"] == -signal.SIGKILL and rec["signals_received"] and set(rec["signals_received"]) == {"SIGTERM"}
    assert_no_live_session(rec)


def test_operator_signal_while_waiting(tmp_path):
    m = dummy_manifest(tmp_path, ["signal_wrapper", "ok"], extra=(str(os.getpid()),))
    rec = run(tmp_path, m, grace_s=2.0)
    assert rec["execution_status"] == "failed" and rec["stop"]["reason"] == "operator_signal"
    assert rec["not_launched"] == ["d2"] and rec["sessions"][0]["killed"]["group_gone"] is True
    assert_no_live_session(rec)


def test_killed_wrapper_takes_its_session_down(tmp_path):
    """SIGKILL of the wrapper cannot be handled: no final record, no escalation, no scratch
    cleanup. What is enforced: the session gets SIGTERM through its parent-death signal, and
    the record stays 'running', which the analysis rejects."""
    m = dummy_manifest(tmp_path, ["hang", "ok"])
    (tmp_path / "m.json").write_text(json.dumps(m))
    code = ("import json, sys; from pathlib import Path; from kvlab import m3_confirm as mc; "
            f"r = Path({str(tmp_path)!r}); "
            "mc.run_batch(r, json.loads((r / 'm.json').read_text()), 'a' * 64, verification={'problems': []})")
    w = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.DEVNULL)
    rec_path = tmp_path / "run" / mc.BATCH_RECORD
    pid, deadline = None, time.monotonic() + 30
    while pid is None and time.monotonic() < deadline:
        time.sleep(0.05)
        try:
            pid = json.loads(rec_path.read_text())["sessions"][0]["pid"]
        except (OSError, ValueError, IndexError, KeyError):
            pass
    assert pid is not None and _alive(pid)
    w.kill()
    w.wait()
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)
    rec = json.loads(rec_path.read_text())
    assert rec["execution_status"] == "running" and rec["sessions"][0]["returncode"] is None
    problems = mc.batch_problems(rec, {**m, "frozen_unix": 0}, "a" * 64, tmp_path / "run", dummy_spec(m))
    assert any("batch record not finalized (execution_status 'running')" in p for p in problems)


# ---------------------------------------------------------------------------------------------
# the documented shell procedure (review finding 4)
def _documented_block():
    doc = (ROOT / mc.PROTOCOL_DOC).read_text()
    sec = doc[doc.index("## 11."):]
    return sec[sec.index("```bash\n") + len("```bash\n"):sec.index("\n```", sec.index("```bash\n"))]


def test_documented_procedure_preserves_the_failing_wrapper_status(tmp_path):
    """Runs the documented §11 block with bash, on the real CLIs. No freeze manifest and no run
    directory exist, so `run` refuses (exit 1) and the analysis refuses (exit 1) without
    launching anything. The documented capture must report those statuses although `tee`
    succeeds; the old form (`cmd | tee log` then `$?`) reports 0."""
    if (ROOT / SPEC["freeze_manifest"]).exists() or (ROOT / SPEC["run_dir"]).exists():
        pytest.skip("a real freeze manifest or run directory exists; this test must not touch it")
    block = _documented_block()
    assert "m3_confirm.py freeze" not in block  # the executed block can never freeze
    lines = block.splitlines()
    run_line = next(i for i, ln in enumerate(lines) if "scripts/m3_confirm.py run" in ln)
    assert "| tee " in lines[run_line] and lines[run_line + 1].strip() == "run_status=${PIPESTATUS[0]}"
    assert lines[0].strip() == "set -o pipefail"
    an_end = next(i for i, ln in enumerate(lines) if "scripts/m3_analyze.py" in ln)
    while lines[an_end].rstrip().endswith("\\"):
        an_end += 1
    assert "| tee " in lines[an_end] and lines[an_end + 1].strip() == "analyze_status=${PIPESTATUS[0]}"
    script = block.replace("<the freeze manifest SHA-256>", "0" * 64).replace('"local/', f'"{tmp_path}/')
    assert "local/" not in script.replace(str(tmp_path), "")
    r = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert "wrapper exit status: 1" in r.stdout and "analysis exit status: 1" in r.stdout, r.stdout + r.stderr
    logs = {p.name: p.read_text() for p in tmp_path.glob("m3_confirm_*.log")}
    assert any("REFUSED" in x for n, x in logs.items() if n.startswith("m3_confirm_run_")), logs
    assert any("does not exist" in x for n, x in logs.items() if n.startswith("m3_confirm_analyze_")), logs
    assert not (ROOT / SPEC["run_dir"]).exists() and not (ROOT / SPEC["freeze_manifest"]).exists()
    old = subprocess.run(["bash", "-c", f".venv/bin/python scripts/m3_confirm.py verify 2>&1 | tee {tmp_path}/old.log; "
                                        "echo status=$?"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "status=0" in old.stdout  # the review's reproduction: tee's status hides the failure
