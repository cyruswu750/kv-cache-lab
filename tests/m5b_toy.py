"""Synthetic M5b test toolkit (imported by tests/test_m5_*.py). Nothing here is measured.

- a tiny random Qwen2 (CPU) and small specs with the production spec's structure (C = 4);
- `FakeTiming`: a fake clock whose timed intervals are SYNTHETIC latencies from a known truth model (plus
  policy overheads), fed in schedule order, so fitting and analysis have hand-checkable inputs;
- `make_run_dir`: assembles a run directory through the real `kvlab.m3_confirm.run_batch`, using a fake
  interpreter that copies pre-recorded session files instead of running GPU sessions.
"""

import copy
import json
import os
import stat
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import m3_confirm as mc
from kvlab import m5_bench as mb
from kvlab import m5_spec as sp
from kvlab.identity import namespace_for

C = 4
TINY_VOCAB = (10, 500)
TINY_MODEL = {"id": "toy/tiny-qwen2", "revision": "0" * 40, "dtype": "bfloat16", "attn_implementation": "sdpa"}
TINY_CAL_GRID = (("tc-01", 5, "H", 1), ("tc-02", 10, "DD", 2), ("tc-03", 17, "HDH", 3), ("tc-04", 23, "DH", 4),
                 ("tc-05", 33, "HHDD", 5), ("tc-06", 14, "", 6), ("tc-07", 27, "DDDDD", 7),
                 ("tc-08", 29, "HHHHHH", 8), ("tc-09", 35, "DDDDDD", 9), ("tc-10", 21, "DHHH", 10))
TINY_EVAL_GRID = (("te-01", 7, "D", 21), ("te-02", 11, "HH", 22), ("te-03", 19, "DHD", 23), ("te-04", 26, "HD", 24),
                  ("te-05", 31, "HHHDD", 25))
TINY_ANALYSIS = {"block_len": 1, "n_boot": 200, "level": 0.95, "ambiguity_alpha": 0.05, "seed": 5,
                 "partition": {"A": ["s01"], "B": ["s02"]}, "partition_weights": {"A": 0.5, "B": 0.5},
                 "case_weight": "equal", "min_sessions_per_half": 1, "min_blocks_per_session": 2}
TRUTH = {"full": {"const": 10.0, "n_kt": 2000.0, "n_pairs_kt2": 100.0},
         "partial": {"const": 12.0, "host_chunks": 1.0, "disk_chunks": 9.0, "s_kt": 2000.0, "s_pairs_kt2": 100.0},
         "shared": {"admitted_chunks": 0.5}}
OVERHEAD_MS = {"cost_aware": 0.07, "always_restore": 0.02, "recompute": 0.01}
FAKE_SOURCE = {"git_head": "0" * 40, "git_status_porcelain": [], "git_diff_head_sha256": "e" * 64,
               "files_sha256": {"kvlab/m5_bench.py": "a" * 64}}
FAKE_IDENTITY = {"source": FAKE_SOURCE, "packages": {"torch": "test"},
                 "model_blobs": {"model.safetensors": {"blob": "b" * 64, "bytes": 1}}}


def tiny_model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


def tiny_ns(model):
    return namespace_for(model, TINY_MODEL["id"], TINY_MODEL["revision"], C)


def tiny_specs(*, cal_sessions=None, eval_sessions=None, reps=3, warmup=1, backend="batched", group=2):
    cal_sched = {"sessions": cal_sessions or {"s01": 101, "s02": 102}, "warmup_reps": warmup, "reps": reps}
    ev_sched = {"sessions": eval_sessions or {"s01": 201, "s02": 202}, "warmup_reps": warmup, "reps": reps}
    kw = dict(C_=C, backend=backend, group=group, model=TINY_MODEL, analysis=TINY_ANALYSIS, deadline_s=600)
    cal = sp.build_spec("TINY_CAL", "calibration", sp.grid_cases(TINY_CAL_GRID, "calibration", C, TINY_VOCAB), cal_sched,
                        **{**kw, "analysis": {**TINY_ANALYSIS, "partition": None}})
    ev = sp.build_spec("TINY_EVAL", "evaluation", sp.grid_cases(TINY_EVAL_GRID, "evaluation", C, TINY_VOCAB), ev_sched,
                       calibration=cal, **kw)
    return cal, ev


def truth_ms(case, L, coef=TRUTH, C_=C) -> float:
    N, m = case["prompt"]["length"], len(case["admitted"])
    tiers = case["feasible_tiers"][:L // C_]
    h, d = tiers.count("host"), tiers.count("disk")
    f, p, a = coef["full"], coef["partial"], coef["shared"]["admitted_chunks"]
    if L == 0:
        n = N / 1000
        return f["const"] + f["n_kt"] * n + f["n_pairs_kt2"] * n * n / 2 + a * m
    s, lk = (N - L) / 1000, L / 1000
    return p["const"] + p["host_chunks"] * h + p["disk_chunks"] * d + p["s_kt"] * s + \
        p["s_pairs_kt2"] * s * (lk + s / 2) + a * m


def latency_ms(spec, case, label, session, rep, phase, predictions=None, noise=None) -> float:
    kind, v = sp.parse_label(label)
    L = sp.expected_selected(case, label, predictions)
    x = truth_ms(case, L) + (OVERHEAD_MS[v] if kind == "policy" else 0.0)
    if phase == "warmup":
        x += 1000.0  # warmups are deliberately huge: they must never reach any statistic
    if noise is not None:
        x += noise(session, rep, label)
    return x


class FakeTiming:
    """clock() advances 1 us per call, except that the call right after sync() is t0 and the next one is
    t0 + the scheduled synthetic latency (the timed region makes exactly these two calls)."""

    def __init__(self, latencies_ns):
        self.lat = list(latencies_ns)
        self.used, self.t, self.armed, self.stage = 0, 10 ** 12, None, 0

    def sync(self):
        self.armed, self.stage = self.lat[self.used], 1
        self.used += 1

    def clock(self):
        if self.stage == 1:
            self.stage = 2
            return self.t
        if self.stage == 2:
            self.t += self.armed
            self.stage, self.armed = 0, None
            return self.t
        self.t += 1000
        return self.t


def scheduled_latencies_ns(spec, session, predictions=None, noise=None) -> list:
    by_id = sp.case_by_id(spec)
    return [int(round(latency_ms(spec, by_id[cid], label, session, rep, phase, predictions, noise) * 1e6))
            for cid, phase, rep, _pos, label in sp.schedule(spec, session)]


def fake_counters():
    return {"status": "ok", "vmstat": {"drop_pagecache": 7}, "meminfo_kib": {}}


def run_tiny_session(model, ns, spec, sid, scratch: Path, *, cost_model=None, predictions=None, noise=None,
                     timing=None):
    timing = timing or FakeTiming(scheduled_latencies_ns(spec, sid, predictions, noise))
    out = {"argv": ["test"], **copy.deepcopy(FAKE_IDENTITY)}
    mb.run_session(model, ns, spec, sid, scratch, out=out, pin=False, cost_model=cost_model, predictions=predictions,
                   clock=timing.clock, sync=timing.sync, read_counters=fake_counters)
    assert timing.used == len(timing.lat), "every scheduled sample consumed exactly one synthetic latency"
    return out


COPIER = """import shutil, sys
args = sys.argv[1:]
run_dir, sid = args[args.index('--run-dir') + 1], args[args.index('--session') + 1]
shutil.copy({src!r} + '/session_' + sid + '.json', run_dir + '/session_' + sid + '.json')
"""


def make_run_dir(root: Path, spec, sessions: dict, rel="run", *, cost_model_path=None, predictions_path=None,
                 bindings=None, budget_s=None) -> Path:
    """A run directory produced by the real run_batch, whose 'python' copies the given session records."""
    src = root / "recorded"
    src.mkdir(parents=True)
    for sid, out in sessions.items():
        (src / f"session_{sid}.json").write_text(mb.dumps(out))
    fake_py = root / "fakepython"
    copier = root / "copier.py"
    copier.write_text(COPIER.format(src=str(src)))
    fake_py.write_text(f"#!/bin/sh\nexec {sys.executable} {copier} \"$@\"\n")
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IXUSR)
    run_dir = root / rel
    plan = sp.batch_plan(spec, str(run_dir), str(fake_py), cost_model_path=cost_model_path,
                         predictions_path=predictions_path, bindings=bindings)
    if budget_s is not None:
        plan["budget_s"] = budget_s
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(mb.dumps(spec))
    (run_dir / "batch_plan.json").write_text(mb.dumps(plan))
    mc.run_batch(root, plan, mc.sha256_file(run_dir / "batch_plan.json"),
                 verification={"problems": [], "checks": "test"}, log=lambda *_a: None)
    return run_dir


def rewrite_session(run_dir: Path, sid: str, edit) -> None:
    """Tamper with a session file AND keep the batch record's recorded digest consistent, so that only the
    sample-level checks can catch the edit."""
    p = run_dir / f"session_{sid}.json"
    obj = json.loads(p.read_text())
    edit(obj)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))
    rec_p = run_dir / mc.BATCH_RECORD
    rec = json.loads(rec_p.read_text())
    for e in rec["sessions"]:
        if e["session"] == sid:
            e["session_json"] = mc._file_state(p)
    rec_p.write_text(json.dumps(rec, indent=1))


def env_ok() -> bool:
    return os.name == "posix"
