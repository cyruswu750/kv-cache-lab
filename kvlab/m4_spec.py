"""The M4-B evaluation spec (DRAFT; docs/M4B_PROTOCOL.md). CPU only: no torch.

`make_spec()` builds a spec dict; `M4_EVAL_V1` is the draft evaluation spec. Tests build scaled-down
specs (tiny traces, C = 4) through the same function, so the runner, validator and analysis are
exercised on the same code paths. Nothing here is frozen: the freeze (kvlab/m4_freeze.py) refuses
while the protocol's first line says DRAFT.

Roles of the trace seeds, fixed before any GPU run:
- `calibration` (41001-41004): used ONLY by the bounded GPU correctness gate
  (tests/gpu/test_m4_replay.py) and by CPU tests. No timing from them is analysed. The one
  exception is the owner-authorized feasibility pilot (decision D34), used solely for the
  runtime-feasibility decision. Nothing measured on them selects a policy, metric, trace, budget,
  threshold or comparison;
- `evaluation` (42001-42004): the evaluation sessions only;
- `warmup` (43001-43004): untimed-for-analysis process warmup inside every session.
"""

import copy
import hashlib
import math
import random

from kvlab import traces as tr
from kvlab.identity import canonical_json

CONFIGS = ("C0", "C1PA_W", "C1PA_HALF", "C2_base", "C2_batch")
CONFIG_LABELS = {"C0": "C0 no cache", "C1PA_W": "C1-PA(W)", "C1PA_HALF": "C1-PA(W/2)",
                 "C2_base": "C2-base (restore_backend=baseline, the default)",
                 "C2_batch": "C2-batch (restore_backend=batched, opt-in)"}
TRACE_NAMES = ("T0", "T1", "T2", "T3")
GENERATOR = {"T0": "no_reuse", "T1": "shared_prefix", "T2": "multi_turn", "T3": "doc_reuse"}


def _family():
    """The 15 primary comparisons (A, B): d = mean_A - mean_B; d < 0 means A is faster."""
    fam = []
    for t in ("T1", "T2", "T3"):
        for b in ("C2_base", "C2_batch"):
            fam += [(t, b, "C0"), (t, b, "C1PA_HALF")]
    fam += [("T0", "C2_base", "C0"), ("T0", "C2_batch", "C0"), ("T0", "C1PA_HALF", "C0")]
    return fam


def _secondary():
    sec = []
    for t in ("T1", "T2", "T3"):
        sec += [(t, "C2_base", "C1PA_W"), (t, "C2_batch", "C1PA_W"), (t, "C1PA_HALF", "C0")]
    for t in TRACE_NAMES:
        sec.append((t, "C2_batch", "C2_base"))
    return sec


EVAL_PARAMS = {
    "T0": {"n_requests": 48, "min_len": 800, "max_len": 1900},
    "T1": {"n_requests": 64, "prefix_len": 1536, "suffix_min": 16, "suffix_max": 200},
    "T2": {"n_conversations": 8, "n_turns": 6, "opening_len": 384, "user_min": 32, "user_max": 160,
           "answer_min": 64, "answer_max": 320},
    "T3": {"n_requests": 64, "n_docs": 12, "instruction_len": 256, "doc_min": 384, "doc_max": 1536,
           "question_min": 16, "question_max": 96, "zipf_s": 1.0},
}
WARMUP_OVERRIDES = {"T0": {"n_requests": 4}, "T1": {"n_requests": 4},
                    "T2": {"n_conversations": 2, "n_turns": 3}, "T3": {"n_requests": 6, "n_docs": 2}}


def make_spec(*, name, chunk_size, params, warmup_overrides, seeds, sessions, budget_s, run_dir,
              staging_chunks=4, alpha=0.05, steady_fraction=0.25, vocab=(10, 150000), protocol_doc=None,
              python=".venv/bin/python", session_env=None, require_cuda=False,
              expected_payload_bytes=None) -> dict:
    spec = {
        "name": name, "chunk_size": chunk_size, "params": copy.deepcopy(params),
        "warmup_overrides": copy.deepcopy(warmup_overrides), "seeds": copy.deepcopy(seeds),
        "sessions": dict(sessions), "budget_s": budget_s, "run_dir": run_dir, "configs": list(CONFIGS),
        "staging_chunks": staging_chunks, "steps": 1, "alpha": alpha, "family": [list(x) for x in _family()],
        "secondary": [list(x) for x in _secondary()], "steady_fraction": steady_fraction, "vocab": list(vocab),
        "protocol_doc": protocol_doc, "python": python, "require_cuda": require_cuda,
        "expected_payload_bytes": expected_payload_bytes, "session_env": dict(session_env or {"HF_HUB_OFFLINE": "1"}),
        "budget_rule": "W = unique chunks (T0 uses T3's W of the same role); C1PA_W cap W; C1PA_HALF cap ceil(W/2); "
                       "C2 host slots ceil(W/2) (slab = (slots + 1) x payload); C2 disk (W + 2) files",
        "t_interval": "two-sided Student t over sessions, level 1 - alpha/len(family) for verdicts, 0.95 descriptive",
        "verdict_rule": "faster/slower iff all session differences share one sign AND the Bonferroni-level "
                        "t-interval excludes 0 in that direction; otherwise inconclusive; no equivalence claims",
    }
    spec["spec_sha256"] = spec_digest(spec)
    return spec


def spec_digest(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k != "spec_sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


M4_EVAL_V1 = make_spec(
    name="M4_EVAL_V1", chunk_size=256, params=EVAL_PARAMS, warmup_overrides=WARMUP_OVERRIDES,
    seeds={"calibration": {"T0": 41001, "T1": 41002, "T2": 41003, "T3": 41004},
           "evaluation": {"T0": 42001, "T1": 42002, "T2": 42003, "T3": 42004},
           "warmup": {"T0": 43001, "T1": 43002, "T2": 43003, "T3": 43004}},
    sessions={f"e{i:02d}": 44000 + i for i in range(1, 11)},
    budget_s=2700, run_dir="results/m4/M4_EVAL_V1_run", protocol_doc="docs/M4B_PROTOCOL.md",
    require_cuda=True, expected_payload_bytes=7_340_032)  # Qwen2.5-1.5B bf16, C = 256: 28 x 2 x 2 x 256 x 128 x 2 B
GATE_CAP_S = 900  # the bounded GPU correctness gate (docs/M4B_PROTOCOL.md §9)
SPECS = {M4_EVAL_V1["name"]: M4_EVAL_V1}


def trace(spec: dict, name: str, role: str) -> tr.Trace:
    params = dict(spec["params"][name])
    if role == "warmup":
        params.update(spec["warmup_overrides"][name])
    return tr.GENERATORS[GENERATOR[name]](spec["seeds"]["warmup" if role == "warmup" else role][name],
                                          vocab=tuple(spec["vocab"]), **params)


def budgets(spec: dict, role: str, traces: dict) -> dict:
    """Per trace: W, and the capacities/sizes of every configuration (§4)."""
    C = spec["chunk_size"]
    W = {t: tr.trace_stats(traces[t], C)["unique_chunks"] for t in traces}
    out = {}
    for t in traces:
        w = W["T3"] if t == "T0" else W[t]
        half = math.ceil(w / 2)
        out[t] = {"W": W[t], "budget_W": w, "C1PA_W": w, "C1PA_HALF": half, "host_slots": half, "disk_files": w + 2}
    return out


def trace_order(spec: dict, seed: int) -> list:
    r = random.Random(f"m4-trace-order:{spec['name']}:{seed}")
    order = list(TRACE_NAMES)
    r.shuffle(order)
    return order


def config_order(spec: dict, seed: int, trace_name: str, role: str, index: int) -> list:
    """The seeded per-request configuration order; the validator replays it."""
    r = random.Random(f"m4-config-order:{spec['name']}:{seed}:{role}:{trace_name}:{index}")
    order = list(spec["configs"])
    r.shuffle(order)
    return order


def commands_for(spec: dict) -> list:
    """The exact session commands, in launch order (run from the repository root)."""
    return [{"session": sid,
             "argv": [spec["python"], "scripts/m4_measure.py", "--spec", spec["name"], "--out-dir", spec["run_dir"],
                      "--session", sid, "--seed", str(seed)],
             "env": dict(spec["session_env"])}
            for sid, seed in spec["sessions"].items()]
