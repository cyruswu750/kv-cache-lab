"""M4-B runner, validator and analysis end to end on CPU (tiny random bf16 Qwen2, C = 4, toy spec).

Real sessions are produced by kvlab.m4_bench.run_session on the CPU (their timings are not
evidence). A batch record with the exact shape kvlab.m3_confirm.run_batch writes is added (the
real wrapper is exercised with dummy processes in tests/test_m4_batch.py). Every tamper must make
the evidence fail closed; a strict mismatch in complete evidence gives exit 4; an injected ENOSPC
spill failure gives exit 3 although every output stays bitwise correct."""

import errno
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import m3_confirm as mc
from kvlab import m4_bench as mb
from kvlab import m4_spec as ms
from kvlab import m4_validate as mv
from kvlab.identity import namespace_for

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import m4_analyze  # noqa: E402

P = {"T0": {"n_requests": 4, "min_len": 5, "max_len": 12},
     "T1": {"n_requests": 5, "prefix_len": 9, "suffix_min": 1, "suffix_max": 3},
     "T2": {"n_conversations": 2, "n_turns": 3, "opening_len": 5, "user_min": 1, "user_max": 3, "answer_min": 1,
            "answer_max": 3},
     "T3": {"n_requests": 6, "n_docs": 3, "instruction_len": 4, "doc_min": 4, "doc_max": 9, "question_min": 1,
            "question_max": 3, "zipf_s": 1.0}}
WARM = {"T0": {"n_requests": 3}, "T1": {"n_requests": 3}, "T2": {"n_conversations": 2, "n_turns": 2},
        "T3": {"n_requests": 4, "n_docs": 2}}
SPEC = ms.make_spec(
    name="M4_TOY", chunk_size=4, params=P, warmup_overrides=WARM,
    seeds={"calibration": {"T0": 1, "T1": 2, "T2": 3, "T3": 4}, "evaluation": {"T0": 11, "T1": 12, "T2": 13, "T3": 14},
           "warmup": {"T0": 21, "T1": 22, "T2": 23, "T3": 24}},
    sessions={"s1": 101, "s2": 102, "s3": 103}, budget_s=600, run_dir="run", vocab=(10, 500))
SOURCE = {"git_head": "0" * 40, "git_diff_head_sha256": "1" * 64, "files_sha256": {"kvlab/x.py": "2" * 64}}


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, SPEC["chunk_size"])


def write_batch(run_dir: Path, spec=SPEC, edit=None):
    """A batch record of the shape run_batch() writes for a fully successful batch."""
    entries, t = [], 0.0
    for c in ms.commands_for(spec):
        sid = c["session"]
        entries.append({"session": sid, "argv": c["argv"], "env": c["env"], "log": f"logs/session_{sid}.log",
                        "pid": 4000 + len(entries), "started_s": t, "ended_s": t + 1.0, "returncode": 0,
                        "killed": False, "leftover_cleanup": None,
                        "session_json": mc._file_state(run_dir / f"session_{sid}.json"), "monitor_csv_present": False,
                        "runner_reported": None, "scratch_removed": None})
        t += 1.5
    rec = {"record_version": mc.BATCH_RECORD_VERSION, "experiment": spec["name"], "freeze_manifest_sha256": "f" * 64,
           "pre_launch_verification": {"problems": []}, "budget_s": spec["budget_s"], "clock": "time.monotonic",
           "wrapper_pid": 1, "execution_status": "succeeded", "stop": None, "launched_at": "2026-11-04T12:00:00-0800",
           "launched_unix": 1.79e9, "sessions": entries, "not_launched": [], "elapsed_s": entries[-1]["ended_s"],
           "finalized_s": entries[-1]["ended_s"], "signals_received": []}
    if edit:
        edit(rec)
    (run_dir / "batch_record.json").write_text(json.dumps(rec))


@pytest.fixture(scope="module")
def base_run(model, ns, tmp_path_factory):
    d = tmp_path_factory.mktemp("m4run") / "run"
    d.mkdir()
    for sid, seed in SPEC["sessions"].items():
        rep = mb.run_session(model, ns, SPEC, sid, seed, d.parent / "disk" / sid, source=SOURCE,
                             environment={"stack": {"python": "x"}})
        (d / f"session_{sid}.json").write_text(json.dumps(rep))
    write_batch(d)
    return d


def copy_run(base, tmp_path) -> Path:
    d = tmp_path / "run"
    shutil.copytree(base, d)
    return d


def edit_session(d: Path, sid: str, fn, rebatch=True):
    p = d / f"session_{sid}.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))
    if rebatch:  # keep the batch record consistent, so the tamper itself is what is detected
        write_batch(d)


def ev(r, trace="T1"):
    return next(b for b in r["evaluation"] if b["name"] == trace)


def test_complete_toy_evidence_is_valid_and_analysed(base_run, tmp_path):
    v = mv.validate(base_run, SPEC, allow_unfrozen=True)
    assert v["status"] == "valid", v["problems"][:5]
    assert v["correctness_failures"] == [] and mv.exit_code(v) == 0
    n_eval = sum(len(ms.trace(SPEC, t, "evaluation").requests) for t in ms.TRACE_NAMES) * 5
    assert v["counts"] == {"sessions": 3, "evaluation_records": 3 * n_eval,
                           "warmup_records": 3 * sum(len(ms.trace(SPEC, t, "warmup").requests) for t in ms.TRACE_NAMES) * 5}
    assert mv.validate(base_run, SPEC)["status"] == "invalid"           # no manifest and not explicitly unfrozen
    out = tmp_path / "an"
    code = m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(base_run), "--out-dir", str(out), "--allow-unfrozen"],
                           specs={"M4_TOY": SPEC})
    assert code == 0
    a = json.loads((out / "m4_analysis.json").read_text())
    assert a["frozen"] is False and a["evidence_label"].startswith("UNFROZEN")
    assert len(a["primary"]) == 15 and {p["verdict"] for p in a["primary"]} <= {"A_faster", "A_slower", "inconclusive"}
    assert all(p["n_sessions"] == 3 and p["bonferroni_level"] == pytest.approx(1 - 0.05 / 15) for p in a["primary"])
    assert all(s["verdict"] is None for s in a["secondary"])
    r = a["resources"]["T1 C2_batch"]
    assert "process_allocator" not in r                                  # CPU: no allocator figures, none invented
    assert r["logical_copy_traffic_bytes_total"]["d2d_placement"] == r["logical_copy_traffic_bytes_total"]["h2d"]
    assert a["resources"]["T1 C2_base"]["logical_copy_traffic_bytes_total"]["d2d_placement"] == 0
    assert "per-configuration" in " ".join(a["claim_limits"])
    with pytest.raises(SystemExit):                                      # never overwrites an output directory
        m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(base_run), "--out-dir", str(out), "--allow-unfrozen"],
                        specs={"M4_TOY": SPEC})


def test_hand_calculated_copy_accounting_in_the_records(base_run, ns):
    """T1 (toy): prefix 9 tokens = 2 complete chunks shared. For the second request every cached
    configuration restores L = 8 (2 chunks): C1 gathers 2 x payload and the constructor copies
    2 x payload; C2 copies 2 x payload H2D, the constructor 2 x payload, and batched also places
    2 x payload. C0 moves nothing. Persistent C1 growth equals (stored - evicted) x payload."""
    from kvlab import chunk_format as cf
    pl = cf.payload_len(ns)
    r = json.loads((base_run / "session_s1.json").read_text())
    recs = {x["config"]: x for x in ev(r)["records"] if x["index"] == 1}
    for c in ("C1PA_W", "C2_base", "C2_batch"):
        assert recs[c]["restore_len"] == 8
    t = recs["C1PA_W"]["traffic"]
    assert (t["d2d_gather"], t["d2d_constructor"], t["h2d"]) == (2 * pl, 2 * pl, 0)
    for c, place in (("C2_base", 0), ("C2_batch", 2 * pl)):
        t = recs[c]["traffic"]
        assert (t["h2d"], t["d2d_constructor"], t["d2d_placement"], t["d2d_gather"]) == (2 * pl, 2 * pl, place, 0)
    assert all(v == 0 for v in recs["C0"]["traffic"].values())
    first = {x["config"]: x for x in ev(r)["records"] if x["index"] == 0}["C1PA_W"]
    growth = first["c1_retained_after"]["C1PA_W"] - first["c1_retained_before"]["C1PA_W"]
    assert growth == len(first["stored"]) * pl == first["traffic"]["d2d_admission"]


def last_session_exit(code):
    """The batch shape run_batch() writes when the last session exits `code`."""
    def f(rec):
        rec["sessions"][-1]["returncode"] = code
        rec.update(execution_status="failed", stop={"reason": "nonzero_exit", "session": rec["sessions"][-1]["session"],
                                                    "returncode": code, "detail": f"exit code {code}"})
    return f


def mismatch(r, trace="T3", i=7):
    ev(r, trace)["records"][i].update(logits_sha256="0" * 64)
    r["strict_mismatches"] = 1


def test_strict_mismatch_in_complete_evidence_exits_4(base_run, tmp_path):
    """Consistently recorded: the digest differs, the runner counted it, and the last session exited 4
    (scripts/m4_measure.py), which stopped the batch with nothing left to launch."""
    d = copy_run(base_run, tmp_path)
    edit_session(d, "s3", mismatch, rebatch=False)
    write_batch(d, edit=last_session_exit(4))
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "valid" and len(v["correctness_failures"]) == 1 and mv.exit_code(v) == 4
    out = tmp_path / "an"
    assert m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(d), "--out-dir", str(out), "--allow-unfrozen"],
                           specs={"M4_TOY": SPEC}) == 4
    assert (out / "m4_validation.json").is_file() and not (out / "m4_analysis.json").exists()


@pytest.mark.parametrize("case", ["final exit 7, consistent stop (review P1)", "final exit 4 without a mismatch",
                                  "mismatch not counted by the runner", "mismatch in a non-final session"])
def test_session_exits_must_be_explained_by_the_recorded_result(base_run, tmp_path, case):
    d = copy_run(base_run, tmp_path)
    if case.startswith("final exit 7"):
        write_batch(d, edit=last_session_exit(7))
    elif case == "final exit 4 without a mismatch":
        write_batch(d, edit=last_session_exit(4))
    elif case == "mismatch not counted by the runner":
        edit_session(d, "s3", lambda r: ev(r, "T3")["records"][7].update(logits_sha256="0" * 64), rebatch=False)
        write_batch(d, edit=last_session_exit(4))
    else:
        edit_session(d, "s2", mismatch, rebatch=False)
        (d / "session_s3.json").unlink()

        def f(rec):
            rec["sessions"] = rec["sessions"][:2]
            last_session_exit(4)(rec)
            rec.update(not_launched=["s3"], elapsed_s=rec["sessions"][-1]["ended_s"])
        write_batch(d, edit=f)
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid" and mv.exit_code(v) == 3, (case, v["problems"][:3])
    out = tmp_path / "an"
    assert m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(d), "--out-dir", str(out), "--allow-unfrozen"],
                           specs={"M4_TOY": SPEC}) == 3
    assert not (out / "m4_analysis.json").exists()
    if case == "mismatch in a non-final session":
        assert v["correctness_failures"]  # still listed explicitly, although exit 3 takes precedence


def test_injected_enospc_spill_failure_is_invalid_although_outputs_stay_correct(model, ns, base_run, tmp_path):
    def enospc(f, mv_):
        raise OSError(errno.ENOSPC, "No space left on device")
    d = copy_run(base_run, tmp_path)
    (d / "session_s2.json").unlink()
    rep = mb.run_session(model, ns, SPEC, "s2", 102, tmp_path / "disk", source=SOURCE,
                         environment={"stack": {"python": "x"}}, disk_write_fn=enospc)
    (d / "session_s2.json").write_text(json.dumps(rep))
    write_batch(d)
    assert all(r["ref_logits_sha256"] == r["logits_sha256"] for b in rep["evaluation"] for r in b["records"])
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid" and mv.exit_code(v) == 3 and v["correctness_failures"] == []
    assert any("unintended cache/infrastructure fault" in p and "ENOSPC" in p for p in v["problems"])


def _set(path, value):
    def f(r):
        obj = r
        for k in path[:-1]:
            obj = obj[k] if not callable(k) else k(obj)
        obj[path[-1]] = value
    return f


TAMPERS = {
    "restore length off by a chunk": lambda r: ev(r)["records"][6].update(restore_len=ev(r)["records"][6]["restore_len"] + 4),
    "residency below 1.0": lambda r: [x.update(residency={**x["residency"], "resident_fraction_before": 0.5})
                                      for x in ev(r)["records"] if x["config"] == "C2_base"][:1],
    "two forward calls": lambda r: ev(r)["records"][3].update(forward_calls=2),
    "synthetic spill failure": lambda r: [x["c2"]["counters"].update(spill_failures={"write_oserror:ENOSPC": 1})
                                          for x in ev(r, "T2")["records"] if x["config"] == "C2_batch"][:1],
    "CRC read failure recorded": lambda r: [x["c2"]["counters"].update(miss_reasons={"absent": 1, "disk_crc": 1})
                                            for x in ev(r, "T3")["records"] if x["config"] == "C2_base"][:1],
    "undeletable disk bytes": lambda r: [x["c2"].update(disk_undeletable_bytes=10)
                                         for x in ev(r, "T0")["records"] if x["config"] == "C2_base"][:1],
    "copy traffic edited": lambda r: ev(r)["records"][2]["traffic"].update(h2d=1),
    "C1 eviction counter edited": lambda r: [x["c1"]["counters"].update(evictions=99)
                                             for x in ev(r, "T3")["records"] if x["config"] == "C1PA_HALF"][:1],
    "record dropped": lambda r: ev(r)["records"].pop(),
    "records swapped": lambda r: ev(r)["records"].insert(0, ev(r)["records"].pop(1)),
    "trace digest": lambda r: ev(r).update(trace_sha256="0" * 64),
    "budgets": lambda r: ev(r)["budgets"].update(C1PA_HALF=99),
    "spec digest": lambda r: r.update(spec_sha256="0" * 64),
    "unfinished": lambda r: r.update(finished=False),
    "trace order": lambda r: r.update(trace_order=list(reversed(r["trace_order"]))),
    "reference digest missing": lambda r: ev(r)["records"][0].pop("ref_logits_sha256"),
    "allocator figures on a CPU run": lambda r: ev(r)["records"][0].update(memory={"A0_allocated": 1}),
    "latency not a number": lambda r: ev(r)["records"][0].update(latency_ns=float("nan")),
    "wall clock not chronological": lambda r: ev(r)["records"][5].update(t_wall=1.0),
    "request KV size": lambda r: ev(r)["records"][1].update(request_kv_bytes=1),
    "warmup block missing": lambda r: r.update(warmup=r["warmup"][:3]),
    # malformed inputs found by the independent review (each must be exit 3, never a crash or exit 4)
    "missing restore_scratch_bytes": lambda r: ev(r)["records"][0].pop("restore_scratch_bytes"),
    "c1_retained_after not a mapping": lambda r: ev(r)["records"][0].update(c1_retained_after=[]),
    "latency beyond the float range": lambda r: ev(r)["records"][0].update(latency_ns=10 ** 400),
    "reference digest not a string": lambda r: ev(r)["records"][0].update(ref_logits_sha256=[]),
    "batched hit reports the baseline backend": lambda r: [x["c2"]["last_restore"].update(backend="baseline")
                                                           for x in ev(r)["records"]
                                                           if x["config"] == "C2_batch" and x["restore_len"]][:1],
    "negative destination storage": lambda r: [x["c2"]["last_restore"].update(device_destination_bytes=-1)
                                               for x in ev(r)["records"]
                                               if x["config"] == "C2_base" and x["restore_len"]][:1],
    "negative disk usage": lambda r: [x["c2"].update(disk_used_bytes=-5) for x in ev(r)["records"]
                                      if x["config"] == "C2_base"][:1],
    "negative slab": lambda r: [x["c2"].update(host_slab_bytes=-1) for x in ev(r)["records"]
                                if x["config"] == "C2_batch"][:1],
    "miss_reasons missing": lambda r: [x["c2"]["counters"].pop("miss_reasons") for x in ev(r)["records"]
                                       if x["config"] == "C2_base"][:1],
    "host hits inflated": lambda r: [x["c2"]["counters"].update(host_hits=999999) for x in ev(r)["records"]
                                     if x["config"] == "C2_base"][:1],
    "device removed": lambda r: r.pop("device"),
    "namespace removed": lambda r: r.pop("namespace_sha256"),
    "session start removed": lambda r: r.pop("started_unix"),
    "runner mismatch count removed": lambda r: r.pop("strict_mismatches"),
    # further malformed-record and counter-consistency cases
    "last_restore removed": lambda r: [x["c2"].pop("last_restore") for x in ev(r)["records"]
                                       if x["config"] == "C2_base"][:1],
    "namespace digest not a string": lambda r: r.update(namespace_sha256=[]),
    "warmup C2 counters not a mapping": lambda r: [x["c2"].update(counters=["bad"]) for x in r["warmup"][1]["records"]
                                                   if x["config"] == "C2_batch"][:1],
    "full recomputes inflated": lambda r: [x["c2"]["counters"].update(full_recompute_requests=999999)
                                           for x in ev(r)["records"] if x["config"] == "C2_base"][:1],
    "chunk misses inflated": lambda r: [x["c2"]["counters"].update(chunk_misses=999999)
                                        for x in ev(r)["records"] if x["config"] == "C2_batch"][:1],
    "host evictions inflated": lambda r: [x["c2"].update(host_evictions=999999)
                                          for x in ev(r)["records"] if x["config"] == "C2_base"][:1],
    "absent misses inflated": lambda r: [x["c2"]["counters"].update(miss_reasons={"absent": 999999})
                                         for x in ev(r)["records"] if x["config"] == "C2_base"][:1],
    "C1 misses inflated": lambda r: [x["c1"]["counters"].update(chunk_misses=999999)
                                     for x in ev(r)["records"] if x["config"] == "C1PA_W"][:1],
}
REASON = {  # each tamper must be rejected for its own reason (checked by message), not incidentally
    "restore length off by a chunk": "restored or stored", "residency below 1.0": "page residency",
    "two forward calls": "forward calls", "synthetic spill failure": "unintended cache/infrastructure fault",
    "CRC read failure recorded": "unintended cache/infrastructure fault",
    "undeletable disk bytes": "unintended cache/infrastructure fault", "copy traffic edited": "copy traffic",
    "C1 eviction counter edited": "C1 counters", "record dropped": "records, expected",
    "records swapped": "seeded interleaved order", "trace digest": "trace_sha256", "budgets": "budgets",
    "spec digest": "spec_sha256", "unfinished": "finished", "trace order": "trace_order",
    "reference digest missing": "ref_logits_sha256 missing or malformed", "allocator figures on a CPU run": "memory record",
    "latency not a number": "unparseable", "wall clock not chronological": "not chronological",
    "request KV size": "request KV", "warmup block missing": "warmup blocks",
    "missing restore_scratch_bytes": "restore_scratch_bytes missing or malformed",
    "c1_retained_after not a mapping": "c1_retained_after missing or malformed",
    "latency beyond the float range": "latency_ns missing or malformed",
    "reference digest not a string": "ref_logits_sha256 missing or malformed",
    "batched hit reports the baseline backend": "restore bookkeeping", "negative destination storage": "restore bookkeeping",
    "negative disk usage": "C2 state malformed", "negative slab": "C2 state malformed",
    "miss_reasons missing": "C2 state malformed", "host hits inflated": "C2 cumulative counters",
    "device removed": "device missing", "namespace removed": "namespace digest",
    "session start removed": "start/end times", "runner mismatch count removed": "strict_mismatches",
    "last_restore removed": "C2 state malformed", "namespace digest not a string": "namespace digest",
    "warmup C2 counters not a mapping": "C2 state malformed",
    "full recomputes inflated": "(full recomputes, misses, miss reasons, host evictions)",
    "chunk misses inflated": "(full recomputes, misses, miss reasons, host evictions)",
    "host evictions inflated": "(full recomputes, misses, miss reasons, host evictions)",
    "absent misses inflated": "(full recomputes, misses, miss reasons, host evictions)",
    "C1 misses inflated": "C1 cumulative",
}


@pytest.mark.parametrize("name", list(TAMPERS))
def test_tampered_or_inconsistent_evidence_is_invalid(base_run, tmp_path, name):
    d = copy_run(base_run, tmp_path)
    edit_session(d, "s2", TAMPERS[name])
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid" and mv.exit_code(v) == 3, name
    assert any(REASON[name] in p for p in v["problems"]), (name, v["problems"][:3])
    out = tmp_path / "an"  # the real CLI: exit 3, a validation report, no analysis, never a crash
    assert m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(d), "--out-dir", str(out), "--allow-unfrozen"],
                           specs={"M4_TOY": SPEC}) == 3
    assert (out / "m4_validation.json").is_file() and not (out / "m4_analysis.json").exists()


def test_allocator_schema_and_device_requirements():
    rec = {"memory": {"A0_allocated": -1, "P_max_allocated": 5,
                      "persistent_c1_retained_at_A0": {"C1PA_W": 0, "C1PA_HALF": 0}}}
    assert any("allocator record malformed" in p for p in mv._schema(rec, "C0", False, cuda=True))
    rec["memory"].update(A0_allocated=6)                     # P below A0
    assert any("allocator record malformed" in p for p in mv._schema(rec, "C0", False, cuda=True))
    assert ms.M4_EVAL_V1["require_cuda"] is True and ms.M4_EVAL_V1["expected_payload_bytes"] == 7_340_032


def test_warmup_coverage_rule():
    def blk(recs):
        return {"records": recs}
    full = [{"config": c, "restore_len": 4, "stored": [0], "sources": ["host", "disk"],
             "c1": {"counters": {"evictions": 1}}, "c2": {"counters": {"host_spills_to_disk": 1}}}
            for c in ("C1PA_W", "C1PA_HALF", "C2_base", "C2_batch")]
    cx = mv._Ctx()
    mv._coverage(cx, "s", [blk(full)])
    assert cx.problems == []
    no_disk = [{**r, "sources": ["host"]} for r in full]
    cx = mv._Ctx()
    mv._coverage(cx, "s", [blk(no_disk)])
    assert any("disk hit" in p for p in cx.problems)
    cx = mv._Ctx()
    mv._coverage(cx, "s", [blk([{**r, "c1": {"counters": {}}} for r in full])])
    assert any("no eviction or prefix-capacity stop" in p for p in cx.problems)


@pytest.mark.parametrize("case", ["missing session", "truncated json", "extra file", "batch says deadline",
                                  "batch nonzero exit, rest not launched", "elapsed over budget",
                                  "session file changed after the batch"])
def test_incomplete_or_contradictory_runs_are_invalid(base_run, tmp_path, case):
    d = copy_run(base_run, tmp_path)
    if case == "missing session":
        (d / "session_s3.json").unlink()
        write_batch(d)
    elif case == "truncated json":
        p = d / "session_s1.json"
        p.write_text(p.read_text()[:-100])
        write_batch(d)
    elif case == "extra file":
        (d / "notes.txt").write_text("x")
    elif case == "batch says deadline":
        write_batch(d, edit=lambda rec: rec.update(execution_status="failed", stop={
            "reason": "deadline", "session": "s3", "detail": "still running"}))
    elif case == "batch nonzero exit, rest not launched":
        def f(rec):
            rec["sessions"] = rec["sessions"][:2]
            rec["sessions"][1]["returncode"] = 1
            rec.update(execution_status="failed", stop={"reason": "nonzero_exit", "session": "s2", "returncode": 1,
                                                         "detail": "exit 1"}, not_launched=["s3"],
                       elapsed_s=rec["sessions"][1]["ended_s"])
        write_batch(d, edit=f)
    elif case == "elapsed over budget":
        write_batch(d, edit=lambda rec: [e.update(started_s=e["started_s"] + 700, ended_s=e["ended_s"] + 700)
                                         for e in rec["sessions"]] and rec.update(elapsed_s=rec["sessions"][-1]["ended_s"]))
    elif case == "session file changed after the batch":
        edit_session(d, "s1", lambda r: r.update(environment={"stack": {"python": "y"}}), rebatch=False)
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid", case


def test_freeze_manifest_binding(base_run, tmp_path):
    d = copy_run(base_run, tmp_path)
    good = {"experiment": SPEC["name"], "run_dir": SPEC["run_dir"], "budget_s": SPEC["budget_s"],
            "commands": ms.commands_for(SPEC), "spec_sha256": SPEC["spec_sha256"], "frozen_unix": 1.0e9,
            "files_sha256": SOURCE["files_sha256"], "base_git_head": SOURCE["git_head"],
            "git_diff_head_sha256": SOURCE["git_diff_head_sha256"], "stack": {"python": "x"}}
    v = mv.validate(d, SPEC, manifest=good, manifest_sha256="f" * 64)
    assert v["status"] == "valid" and v["frozen"] is True, v["problems"][:3]
    for k, val in (("spec_sha256", "0" * 64), ("files_sha256", {"kvlab/x.py": "3" * 64}), ("stack", {"python": "z"}),
                   ("commands", [])):
        assert mv.validate(d, SPEC, manifest={**good, k: val}, manifest_sha256="f" * 64)["status"] == "invalid", k
    assert mv.validate(d, SPEC, manifest=good, manifest_sha256="e" * 64)["status"] == "invalid"  # batch bound to another


COUNTERS = {  # final-review P2: (where in a C2 record, key)
    "full_recompute_requests": ("counters", "full_recompute_requests"),
    "chunk_misses": ("counters", "chunk_misses"),
    "miss_reasons_absent": ("counters", "miss_reasons"),
    "host_evictions": ("state", "host_evictions"),
}


def _counter_rec(r):
    """The last C2_base record of evaluation T2 (multi-turn): every one of the four counters is
    nonzero there, so a mismatch cannot pass by coincidence of zeros."""
    return [x for x in ev(r, "T2")["records"] if x["config"] == "C2_base"][-1]


def test_the_four_counters_are_nonzero_in_the_positive_control(base_run):
    r = json.loads((base_run / "session_s2.json").read_text())
    c2 = _counter_rec(r)["c2"]
    assert c2["counters"]["full_recompute_requests"] > 0 and c2["counters"]["chunk_misses"] > 0
    assert c2["counters"]["miss_reasons"].get("absent", 0) > 0 and c2["host_evictions"] > 0
    assert mv.validate(base_run, SPEC, allow_unfrozen=True)["status"] == "valid"  # unmodified: valid


@pytest.mark.parametrize("name", list(COUNTERS))
@pytest.mark.parametrize("change", ["missing", "+1", "-1"])
def test_each_counter_must_reconcile_missing_or_off_by_one(base_run, tmp_path, name, change):
    where, key = COUNTERS[name]

    def f(r):
        c2 = _counter_rec(r)["c2"]
        d = c2["counters"] if where == "counters" else c2
        if change == "missing":
            d.pop(key)
        elif key == "miss_reasons":
            d[key] = {"absent": d[key]["absent"] + (1 if change == "+1" else -1)}
        else:
            d[key] += 1 if change == "+1" else -1
    d = copy_run(base_run, tmp_path)
    edit_session(d, "s2", f)
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid" and mv.exit_code(v) == 3, (name, change)
    want = "C2 state malformed" if change == "missing" else "(full recomputes, misses, miss reasons, host evictions)"
    assert any(want in p for p in v["problems"]), (name, change, v["problems"][:3])
    out = tmp_path / "an"
    assert m4_analyze.main(["--spec", "M4_TOY", "--run-dir", str(d), "--out-dir", str(out), "--allow-unfrozen"],
                           specs={"M4_TOY": SPEC}) == 3
    assert (out / "m4_validation.json").is_file() and not (out / "m4_analysis.json").exists()
