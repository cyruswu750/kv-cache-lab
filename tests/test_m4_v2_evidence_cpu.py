"""M4_EVAL_V2 on CPU: the runner's request-scoped warm-disk preparation, its records, the validator's V2 rules,
the batch drop_pagecache record (scripts/m4_eval.py) and the V2 spec/freeze binding. Tiny random bf16 Qwen2,
C = 4, a toy spec with `warm_disk="request_candidates"`. CPU timings are not evidence.

The positive-control sessions read scripted kernel counters, so a real WSL page-cache drop during the test run
cannot change the result. The real counter reader is tested in tests/test_warm_disk.py."""

import copy
import functools
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import m3_confirm as mc
from kvlab import m4_analysis as ma
from kvlab import m4_bench as mb
from kvlab import m4_freeze as mf
from kvlab import m4_spec as ms
from kvlab import m4_validate as mv
from kvlab import warm_disk as wd
from kvlab.identity import namespace_for
from test_m4_evidence_cpu import P, SOURCE, WARM, write_batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import m4_analyze  # noqa: E402
import m4_eval  # noqa: E402

SEEDS = {"calibration": {"T0": 1, "T1": 2, "T2": 3, "T3": 4}, "evaluation": {"T0": 11, "T1": 12, "T2": 13, "T3": 14},
         "warmup": {"T0": 21, "T1": 22, "T2": 23, "T3": 24}}
SPEC = ms.make_spec(name="M4_TOY_V2", chunk_size=4, params=P, warmup_overrides=WARM, seeds=SEEDS,
                    sessions={"s1": 101, "s2": 102}, budget_s=600, run_dir="run", vocab=(10, 500),
                    warm_disk="request_candidates")
SPEC1 = ms.make_spec(name="M4_TOY_V2_ONE", chunk_size=4, params=P, warmup_overrides=WARM, seeds=SEEDS,
                     sessions={"s1": 101}, budget_s=600, run_dir="run", vocab=(10, 500), warm_disk="request_candidates")
DROP = 3


def fake_counters(drop=DROP):
    return {"status": "ok", "vmstat": {"drop_pagecache": drop, "pgsteal_kswapd": 10}, "meminfo_kib": {}}


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


LAUNCHED_UNIX = 1.79e9  # write_batch's first-launch time; the toy sessions run later (real wall clock)


def write_interference(d: Path, start=DROP, end=DROP, edit=None, spec=SPEC):
    """A record bracketing the batch: the start read before the first launch, the end read now (after every
    session of the run directory has ended)."""
    rec = {"format": wd.INTERFERENCE_FORMAT, "experiment": spec["name"], "counter": "drop_pagecache",
           "start": fake_counters(start), "end": fake_counters(end), "start_unix": LAUNCHED_UNIX - 1.0,
           "end_unix": time.time()}
    if edit:
        edit(rec)
    (d / wd.INTERFERENCE_RECORD).write_text(json.dumps(rec))


def run_sessions(model, ns, d: Path, spec=SPEC, prepare=None):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(wd, "counters", fake_counters)
        if prepare is not None:
            m.setattr(mb.wd, "prepare", prepare)
        for sid in spec["sessions"]:
            rep = mb.run_session(model, ns, spec, sid, spec["sessions"][sid], d.parent / "disk" / sid, source=SOURCE,
                                 environment={"stack": {"python": "x"}})
            (d / f"session_{sid}.json").write_text(json.dumps(rep))
    write_batch(d, spec)


@pytest.fixture(scope="module")
def base_run(model, ns, tmp_path_factory):
    d = tmp_path_factory.mktemp("m4v2") / "run"
    d.mkdir()
    run_sessions(model, ns, d)
    write_interference(d)
    return d


def copy_run(base, tmp_path) -> Path:
    d = tmp_path / "run"
    shutil.copytree(base, d)
    return d


def edit_session(d: Path, sid: str, fn):
    p = d / f"session_{sid}.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))
    write_batch(d, SPEC)  # keep the batch record consistent: the tamper itself must be what is detected


def c2_records(r):
    return [x for blk in r["warmup"] + r["evaluation"] for x in blk["records"] if x["config"].startswith("C2")]


def invalid_with(d, text):
    v = mv.validate(d, SPEC, allow_unfrozen=True)
    assert v["status"] == "invalid" and mv.exit_code(v) == 3
    assert any(text in p for p in v["problems"]), (text, v["problems"][:3])
    return v


# ---- positive control

def test_v2_toy_evidence_is_valid_request_scoped_and_matches_the_actual_reads(base_run, tmp_path):
    v = mv.validate(base_run, SPEC, allow_unfrozen=True)
    assert v["status"] == "valid", v["problems"][:5]
    seen = {"candidates": 0, "empty_with_tier": 0, "mixed": 0}
    for sid in SPEC["sessions"]:
        r = json.loads((base_run / f"session_{sid}.json").read_text())
        assert r["interference"]["start"]["vmstat"]["drop_pagecache"] == DROP
        for x in c2_records(r):
            res = x["residency"]
            assert res["protocol"] == "request_candidates" and res["outcome"] in ("verified", "empty")
            # independent of the planner: the bytes the restore actually read, and its sources
            assert x["traffic"]["disk_read"] == len(res["candidates"]) * r["file_bytes"]
            assert len(res["candidates"]) == x["sources"].count("disk")
            assert res["timer_start_minus_verified_ns"] >= 0
            seen["candidates"] += bool(res["candidates"])
            seen["empty_with_tier"] += not res["candidates"] and res["tier_entries"] > 0
            seen["mixed"] += "host" in x["sources"] and "disk" in x["sources"]
    assert all(n > 0 for n in seen.values()), seen
    out = tmp_path / "an"
    assert m4_analyze.main(["--spec", SPEC["name"], "--run-dir", str(base_run), "--out-dir", str(out),
                            "--allow-unfrozen"], specs={SPEC["name"]: SPEC}) == 0
    a = json.loads((out / "m4_analysis.json").read_text())
    assert ma.REQUEST_SCOPED_LIMIT in a["claim_limits"] and ma.WHOLE_TIER_LIMIT not in a["claim_limits"]


def test_v1_shaped_spec_keeps_the_whole_tier_rule(base_run):
    assert ms.warm_disk_protocol(ms.M4_EVAL_V1) == "whole_tier" and "warm_disk" not in ms.M4_EVAL_V1
    v1_toy = {k: v for k, v in SPEC.items() if k != "warm_disk"}
    v1_toy["spec_sha256"] = ms.spec_digest(v1_toy)
    v = mv.validate(base_run, v1_toy, allow_unfrozen=True)  # V2 records are not V1 evidence
    assert v["status"] == "invalid" and any("residency record malformed" in p for p in v["problems"])
    assert ma.claim_limits(ms.M4_EVAL_V1) == ma.CLAIM_LIMITS


# ---- the runner: order, and failures recorded without retry

class Clock:
    """Stands in for the `time` module inside kvlab.m4_bench, logging its calls."""

    def __init__(self, log):
        self.log = log

    def time(self):
        self.log.append("time.time")
        return time.time()

    def perf_counter_ns(self):
        self.log.append("perf_counter_ns")
        return time.perf_counter_ns()


def test_preparation_is_the_last_step_before_the_timer(model, ns, tmp_path, monkeypatch):
    log = []
    real_prepare, real_complete, real_counters = wd.prepare, wd.complete, fake_counters

    def prepare(*a, **k):
        rec = real_prepare(*a, **k)
        assert "outcome" not in rec  # review P2-2: no bookkeeping before the timer
        log.append("prepare")
        return rec

    def complete(*a, **k):
        log.append("complete")
        return real_complete(*a, **k)

    def counters():
        log.append("counters")
        return real_counters()
    monkeypatch.setattr(mb, "time", Clock(log))
    monkeypatch.setattr(wd, "prepare", prepare)
    monkeypatch.setattr(wd, "complete", complete)
    monkeypatch.setattr(wd, "counters", counters)
    monkeypatch.setattr(mb, "_residency", lambda t: (_ for _ in ()).throw(AssertionError("V1 scan in V2")))
    trace = ms.trace(SPEC, "T3", "warmup")
    b = ms.budgets(SPEC, "warmup", {t: ms.trace(SPEC, t, "warmup") for t in ms.TRACE_NAMES})["T3"]
    blk = mb.replay_trace(model, ns, SPEC, trace, "T3", "warmup", 101, b, tmp_path / "disk", strict=False)
    n_c2 = sum(1 for x in blk["records"] if x["config"].startswith("C2"))
    starts = [i for i, e in enumerate(log) if e == "prepare"]
    assert len(starts) == n_c2 > 0
    for i in starts:  # the wall-clock read, t0, t1, then the untimed counter snapshot and completion
        assert log[i + 1:i + 6] == ["time.time", "perf_counter_ns", "perf_counter_ns", "counters", "complete"], \
            log[i - 2:i + 7]
    assert log.count("counters") == 2 * n_c2  # one inside each preparation (before its plan), one after each request


def test_not_resident_after_one_prewarm_is_recorded_and_makes_the_evidence_invalid(model, ns, tmp_path):
    """Scripted observations: every candidate is half resident in both passes. The runner records it, reads each
    candidate exactly once, finishes the session, and the validator rejects the evidence."""
    reads = {}

    def observe(path):
        r, n, size = wd.resident_pages(path)
        return n // 2, n, size  # n = 1 for these toy files: 0 of 1 page

    def read(path):
        reads[(path.parent, path.name)] = reads.get((path.parent, path.name), 0) + 1
        return wd.read_file(path)
    d = tmp_path / "run"
    d.mkdir()
    run_sessions(model, ns, d, spec=SPEC1, prepare=functools.partial(wd.prepare, observe=observe, read=read))
    write_interference(d, spec=SPEC1)
    r = json.loads((d / "session_s1.json").read_text())
    outs = [x["residency"]["outcome"] for x in c2_records(r)]
    assert "not_resident" in outs and set(outs) <= {"not_resident", "empty"} and r["finished"] is True
    n_read = sum(len(x["residency"]["candidates"]) for x in c2_records(r))
    assert sum(reads.values()) == n_read and n_read > 0  # one prewarm per candidate per request, no retry
    v = mv.validate(d, SPEC1, allow_unfrozen=True)
    assert v["status"] == "invalid" and v["correctness_failures"] == []
    assert all("warm-disk" in p for p in v["problems"]), v["problems"][:3]  # nothing else is wrong
    assert any("warm-disk preparation not_resident" in p and "after one prewarm" in p for p in v["problems"])


# ---- validator tampers (each on a copy of the valid positive control)

def _first(r, pred):
    return next(x for x in c2_records(r) if pred(x["residency"]))


def _with_candidates(n=1):
    return lambda res: len(res["candidates"]) >= n


def _drop_last_candidate(r):
    res = _first(r, _with_candidates())["residency"]
    res["candidates"].pop()
    res["files"].pop()


def _extra_candidate(r):
    res = _first(r, _with_candidates())["residency"]
    f = copy.deepcopy(res["files"][-1])
    f.update(file="0" * 64 + ".kvc", index=f["index"] + 1)
    res["files"].append(f)
    res["candidates"].append(f["file"])


def _set(path, value, pred=_with_candidates()):
    def f(r):
        res = _first(r, pred)["residency"]
        obj = res
        for k in path[:-1]:
            obj = obj[k]
        obj[path[-1]] = value(obj[path[-1]]) if callable(value) else value
    return f


def _final_low(consistent):
    def f(r):
        res = _first(r, _with_candidates())["residency"]
        res["files"][0]["final"]["resident_pages"] -= 1
        if consistent:
            res["outcome"] = "not_resident"
    return f


def _initial_low_no_prewarm(r):
    res = _first(r, _with_candidates())["residency"]
    res["files"][0]["initial"]["resident_pages"] = 0


def _prewarm_short(r):
    res = _first(r, _with_candidates())["residency"]
    f = res["files"][0]
    f["initial"]["resident_pages"] = 0
    f["prewarm"] = {"status": "read", "bytes_read": f["initial"]["size"] - 1, "t_start_ns": f["initial"]["t_ns"],
                    "t_end_ns": f["initial"]["t_ns"]}
    res["prewarmed_files"], res["phases"]["prewarm"] = 1, "done"


def _unneeded_prewarm(r):
    res = _first(r, _with_candidates())["residency"]
    f = res["files"][0]
    f["prewarm"] = {"status": "read", "bytes_read": f["initial"]["size"], "t_start_ns": f["initial"]["t_ns"],
                    "t_end_ns": f["initial"]["t_ns"]}
    res["prewarmed_files"], res["phases"]["prewarm"] = 1, "done"


def _observation_error(r):
    res = _first(r, _with_candidates())["residency"]
    res["files"][0]["final"] = {"status": "error", "error": "OSError:EIO: x", "t_ns": res["t_verified_ns"]}
    res["outcome"] = "error"


def _final_before_initial(r):
    res = _first(r, _with_candidates())["residency"]
    f = res["files"][0]
    f["final"]["t_ns"] = f["initial"]["t_ns"] - 1


def _empty_claims_files(r):
    res = _first(r, lambda res: not res["candidates"])["residency"]
    res["outcome"] = "verified"


def _wrong_size(r):
    res = _first(r, _with_candidates())["residency"]
    for ph in ("initial", "final"):
        res["files"][0][ph]["size"] += 1


TAMPERS = {
    "candidate dropped": (_drop_last_candidate, "candidate files"),
    "extra candidate": (_extra_candidate, "candidate files"),
    "candidate index changed": (_set(("files", 0, "index"), lambda i: i + 1), "candidate files"),
    "final pass below full, outcome still verified": (_final_low(False), "recorded outcome 'verified' != recomputed"),
    "final pass below full, outcome consistent": (_final_low(True), "warm-disk preparation not_resident"),
    "initially below full but not prewarmed": (_initial_low_no_prewarm, "one whole-file read required"),
    "prewarm read fewer bytes than the file": (_prewarm_short, "one whole-file read required"),
    "prewarm of a fully resident file": (_unneeded_prewarm, "fully resident initially, but prewarm"),
    "observation error": (_observation_error, "warm-disk preparation error"),
    "final pass timestamp before the initial pass": (_final_before_initial, "phase timestamps"),
    "phase label": (_set(("phases", "prewarm"), "done"), "phases"),
    "prewarm count": (_set(("prewarmed_files",), 1), "prewarmed_files"),
    "plan host count": (_set(("plan", "host_chunks"), lambda n: n + 1), "plan"),
    "plan stop": (_set(("plan", "stop"), "no_staging_buffer"), "plan"),
    "empty set recorded as verified": (_empty_claims_files, "recorded outcome 'verified' != recomputed 'empty'"),
    "file size": (_wrong_size, "a chunk file has"),
    "record missing": (lambda r: c2_records(r)[0].update(residency=None), "warm-disk record malformed"),
    "V1-shaped record": (lambda r: c2_records(r)[0].update(residency={"files": 1, "prewarmed_files": 0,
                                                                      "min_fraction_before": 1.0,
                                                                      "resident_fraction_before": 1.0}),
                         "warm-disk record malformed"),
    "page size": (_set(("page_size",), 3000), "warm-disk record malformed"),
    "unknown outcome": (_set(("outcome",), "warm"), "warm-disk record malformed"),
    "timer interval missing": (lambda r: c2_records(r)[0]["residency"].pop("timer_start_minus_verified_ns"),
                               "warm-disk record malformed"),
}


@pytest.mark.parametrize("name", list(TAMPERS))
def test_tampered_warm_disk_records_are_invalid(base_run, tmp_path, name):
    fn, text = TAMPERS[name]
    d = copy_run(base_run, tmp_path)
    edit_session(d, "s2", fn)
    invalid_with(d, text)


# ---- the drop_pagecache interference rule

def _request_counters(status=None, drop=None, where="counters_before"):
    def f(r):
        snap = c2_records(r)[5]["residency"][where]
        if status:
            snap["status"] = status
        if drop is not None:
            snap["vmstat"]["drop_pagecache"] = drop
    return f


SESSION_CASES = {
    "session snapshots missing": (lambda r: r.pop("interference"), "interference counter snapshots"),
    "session end differs": (lambda r: r["interference"]["end"]["vmstat"].update(drop_pagecache=DROP + 1),
                            "drop_pagecache observations"),
    "request snapshot unreadable": (_request_counters(status="error"), "drop_pagecache observations"),
    "request snapshot unavailable": (_request_counters(status="unavailable"), "drop_pagecache observations"),
    "request snapshot differs after the request": (_request_counters(drop=DROP + 1, where="counters_after_request"),
                                                   "drop_pagecache observations"),
}


@pytest.mark.parametrize("name", list(SESSION_CASES))
def test_session_drop_pagecache_observations_must_equal_the_batch_start(base_run, tmp_path, name):
    fn, text = SESSION_CASES[name]
    d = copy_run(base_run, tmp_path)
    edit_session(d, "s1", fn)
    invalid_with(d, text)


BATCH_CASES = {
    "record missing": (None, "batch interference record missing"),
    "increment": (dict(end=DROP + 1), "increased by 1"),
    "decrease (reset)": (dict(end=DROP - 1), "decreased"),
    "start unavailable": (dict(edit=lambda rec: rec.update(start={"status": "unavailable", "vmstat": {}})),
                          "not read at the start"),
    "end unreadable": (dict(edit=lambda rec: rec.update(end={"status": "error", "error": "x"})),
                       "not read at the end"),
    "end never written": (dict(edit=lambda rec: rec.update(end=None)), "not read at the end"),
    "format": (dict(edit=lambda rec: rec.update(format="other")), "batch interference record malformed"),
    # review P2-1: identity and the bracket of the batch
    "identity fields missing": (dict(edit=lambda rec: [rec.pop("experiment"), rec.pop("counter")]),
                                "batch interference record names None/None"),
    "another experiment": (dict(edit=lambda rec: rec.update(experiment="M4_OTHER")), "batch interference record names"),
    "another counter": (dict(edit=lambda rec: rec.update(counter="drop_slab")), "batch interference record names"),
    "times missing": (dict(edit=lambda rec: [rec.pop("start_unix"), rec.pop("end_unix")]), "read times"),
    "times reversed": (dict(edit=lambda rec: rec.update(start_unix=rec["end_unix"] + 1.0)), "read times"),
    "start read after the first launch": (dict(edit=lambda rec: rec.update(start_unix=LAUNCHED_UNIX + 1.0)),
                                          "do not bracket the batch"),
    "end read before the sessions ended": (dict(edit=lambda rec: rec.update(end_unix=LAUNCHED_UNIX)),
                                           "do not bracket the batch"),
}


@pytest.mark.parametrize("name", list(BATCH_CASES))
def test_batch_drop_pagecache_rule(base_run, tmp_path, name):
    kw, text = BATCH_CASES[name]
    d = copy_run(base_run, tmp_path)
    (d / wd.INTERFERENCE_RECORD).unlink()  # the copy's record, rewritten below
    if kw is not None:
        write_interference(d, **kw)
    invalid_with(d, text)


def test_an_unexpected_file_is_still_rejected_and_the_v2_record_is_expected(base_run, tmp_path):
    assert wd.INTERFERENCE_RECORD in mv.expected_files(SPEC)
    assert wd.INTERFERENCE_RECORD not in mv.expected_files(ms.M4_EVAL_V1)
    d = copy_run(base_run, tmp_path)
    (d / "extra.json").write_text("{}")
    invalid_with(d, "unexpected files")


def test_m4_eval_brackets_the_batch_with_two_counter_reads(tmp_path):
    spec = {**SPEC, "run_dir": "results/m4/X_run"}
    seq = iter([fake_counters(4), fake_counters(4)])
    calls = []

    def run_batch():
        calls.append(json.loads((tmp_path / spec["run_dir"] / wd.INTERFERENCE_RECORD).read_text()))
        return {"execution_status": "succeeded"}
    out = m4_eval.run_with_interference_record(tmp_path, spec, run_batch, counters=lambda: next(seq))
    assert out == {"execution_status": "succeeded"}
    assert calls[0]["start"]["vmstat"]["drop_pagecache"] == 4 and calls[0]["end"] is None  # written before the launch
    rec = json.loads((tmp_path / spec["run_dir"] / wd.INTERFERENCE_RECORD).read_text())
    assert rec["format"] == wd.INTERFERENCE_FORMAT and rec["end"]["vmstat"]["drop_pagecache"] == 4
    assert sorted(p.name for p in (tmp_path / spec["run_dir"]).iterdir()) == [wd.INTERFERENCE_RECORD]
    with pytest.raises(FileExistsError):  # an existing run directory: refused before anything runs
        m4_eval.run_with_interference_record(tmp_path, spec, lambda: calls.append("ran"), counters=fake_counters)
    assert calls[-1] != "ran"


def test_m4_eval_records_the_end_even_when_the_batch_wrapper_raises(tmp_path):
    spec = {**SPEC, "run_dir": "r"}
    seq = iter([fake_counters(4), fake_counters(5)])

    def boom():
        raise RuntimeError("wrapper error")
    with pytest.raises(RuntimeError):
        m4_eval.run_with_interference_record(tmp_path, spec, boom, counters=lambda: next(seq))
    rec = json.loads((tmp_path / "r" / wd.INTERFERENCE_RECORD).read_text())
    assert rec["end"]["vmstat"]["drop_pagecache"] == 5
    cx = mv._Ctx()
    mv._interference(cx, tmp_path / "r", spec, None, {})
    assert any("increased by 1" in p for p in cx.problems)
    assert not any("names" in p or "read times" in p or "bracket" in p for p in cx.problems)  # the writer's fields


# ---- the spec and the freeze binding

V1_SPEC_SHA256 = "442e13ecd69cc7e13d9a3f3bf7eb1be616456feb576f7cbf5e604f51b0bab541"  # in V1's freeze manifest


def test_v1_spec_is_unchanged_and_v2_differs_only_in_its_identity_and_warm_disk_rule():
    assert ms.M4_EVAL_V1["spec_sha256"] == V1_SPEC_SHA256 == ms.spec_digest(ms.M4_EVAL_V1)
    v1, v2 = ms.M4_EVAL_V1, ms.M4_EVAL_V2
    assert sorted(k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)) == \
        ["name", "protocol_doc", "run_dir", "spec_sha256", "warm_disk"]
    assert (v2["name"], v2["run_dir"], v2["protocol_doc"], v2["warm_disk"]) == \
        ("M4_EVAL_V2", "results/m4/M4_EVAL_V2_run", "docs/M4_EVAL_V2_PROTOCOL.md", "request_candidates")
    assert ms.SPECS == {"M4_EVAL_V1": v1, "M4_EVAL_V2": v2}
    assert ms.commands_for(v2)[0]["argv"][1:] == ["scripts/m4_measure.py", "--spec", "M4_EVAL_V2", "--out-dir",
                                                  "results/m4/M4_EVAL_V2_run", "--session", "e01", "--seed", "44001"]
    with pytest.raises(ValueError):
        ms.make_spec(**{k: v1[k] for k in ("name", "chunk_size", "params", "warmup_overrides", "seeds", "sessions",
                                            "budget_s", "run_dir")}, warm_disk="whole_tier")


def _fixture_root(tmp_path, spec, first_line) -> Path:
    root = tmp_path / "root"
    files = [*mf.documents(spec), *mf.ANALYSIS_FILES, "pyproject.toml",
             *(str(p.relative_to(ROOT)) for pat in ("kvlab/*.py", "scripts/*.py", "tests/**/*.py") for p in ROOT.glob(pat))]
    for f in sorted(set(files)):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes((ROOT / f).read_bytes())
    if first_line is not None:
        doc = root / spec["protocol_doc"]
        doc.write_text(first_line + "\n" + doc.read_text().split("\n", 1)[1])
    return root


def test_v2_freeze_binds_both_protocols_and_refuses_a_draft_v2_protocol(tmp_path):
    """Fixture roots only: the refusal and the manifest do not depend on the real protocol's lifecycle state."""
    assert mf.documents(ms.M4_EVAL_V1) == mf.DOCUMENTS == ("docs/M4B_PROTOCOL.md", "requirements.txt")
    assert mf.documents(ms.M4_EVAL_V2) == ("docs/M4B_PROTOCOL.md", "requirements.txt", "docs/M4_EVAL_V2_PROTOCOL.md")
    draft = _fixture_root(tmp_path / "a", ms.M4_EVAL_V2, "# M4_EVAL_V2 protocol (DRAFT): fixture")
    with pytest.raises(mc.Refused, match="DRAFT"):
        mf.build_manifest(draft, ms.M4_EVAL_V2, stack={"python": "x"})
    assert not (draft / mf.manifest_path(ms.M4_EVAL_V2)).exists()  # nothing written
    root = _fixture_root(tmp_path / "b", ms.M4_EVAL_V2, "# M4_EVAL_V2 protocol (approved)")
    m, _diff = mf.build_manifest(root, ms.M4_EVAL_V2, stack={"python": "x"}, now=1.0e9)
    assert set(m["documents_sha256"]) == set(mf.documents(ms.M4_EVAL_V2))
    assert mf.manifest_problems(m, ms.M4_EVAL_V2, root) == []
    (root / "docs/M4B_PROTOCOL.md").write_text("changed\n")  # the amended V1 protocol stays bound
    assert any("docs/M4B_PROTOCOL.md differs" in p for p in mf.manifest_problems(m, ms.M4_EVAL_V2, root))


def test_the_validator_restates_the_record_constants_of_the_runner():
    assert ms.warm_disk_protocol(ms.M4_EVAL_V2) == mv.REQUEST_SCOPED == wd.PROTOCOL
    assert mv.INTERFERENCE_RECORD == wd.INTERFERENCE_RECORD and mv.INTERFERENCE_FORMAT == wd.INTERFERENCE_FORMAT
    assert mv.WD_STOPS == wd.STOPS and mv.WD_OUTCOMES == wd.OUTCOMES
