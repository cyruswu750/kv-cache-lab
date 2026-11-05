"""M5b: the explicit, bounded experiment specification (CPU only, no torch).

docs/M5_PROTOCOL_DRAFT.md defines the experiment; this module is its machine-readable form. Everything a
session does, and everything the validator expects, is derived from the dict built here:
- **Cases:** a case is one complete request/state:
  - the prompt: a seeded synthetic token sequence of length N;
  - the state of every complete prompt chunk ("host", "disk" or "absent");
  - the declared host and disk LRU orders and the capacities;
  - the source provenance: one full prefill of the prompt prefix that covers every indexed chunk.
- **Candidates:** every feasible candidate is measured: 0, C, ..., k*C, where k is the leading indexed run
  inside the first-token rule.
- **Admissions:** the expected newly stored chunks are exactly the absent complete chunks; indexed chunks
  beyond the feasible prefix are never re-admitted.
- **Split:** calibration and evaluation are separated by configuration, never by id strings or L values.
  The configuration signature excludes the id and the prompt seed, so a renamed or reseeded duplicate is
  rejected.
- **Domain:** the calibration domain is derived from the declared calibration inputs before any measurement
  (the same formulas as `kvlab.restore_cost.fit_restore_cost`). An evaluation case outside it is a build
  error; it is never dropped silently.
- **Schedule:** per session, a seeded case order; per (case, phase, repetition), a seeded order of the runs.
  Seeds are derived with SHA-256, never with Python's `hash()`.

Nothing here is a measurement. Runtime figures are planning estimates (`budget`), labelled as such.
"""

import copy
import hashlib
import math
import random
import struct
import time

import numpy as np

from kvlab import restore_cost as rc
from kvlab.boundaries import max_restore_len
from kvlab.identity import canonical_json

SPEC_VERSION = "kvlab-m5-spec-v1"
SESSION_FORMAT = "kvlab-m5-session-v1"
CAL_NAME, EVAL_NAME = "M5_CAL_V1", "M5_EVAL_V1"
MODEL = {"id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
         "dtype": "bfloat16", "attn_implementation": "sdpa"}
C = 256
BACKEND, GROUP = "batched", 4           # the explicitly selected performance configuration
VOCAB = (10, 150000)                    # as kvlab.traces: below the model's special tokens
STATES = ("host", "disk", "absent")
POLICIES = ("cost_aware", "always_restore", "recompute")
PROMPT_TAG = b"kvlab-m5-prompt-v1\0"
STATE_LETTERS = {"H": "host", "D": "disk", "A": "absent"}

# (case id, N, indexed prefix as letters, prompt seed); every complete chunk after the prefix is absent
CALIBRATION_GRID = (
    ("cal-01", 300, "H", 51001), ("cal-02", 600, "DD", 51002), ("cal-03", 1100, "HDH", 51003),
    ("cal-04", 1500, "DH", 51004), ("cal-05", 2100, "HHDD", 51005), ("cal-06", 900, "", 51006),
    ("cal-07", 1800, "DDDDD", 51007), ("cal-08", 3500, "H" * 12, 51008), ("cal-09", 3700, "D" * 12, 51009),
    ("cal-10", 4000, "D" * 6 + "H" * 8, 51010), ("cal-11", 2800, "HHHHHDDDDD", 51011),
    ("cal-12", 1300, "DHHH", 51012),
)
EVALUATION_GRID = (
    ("ev-01", 400, "D", 52001), ("ev-02", 700, "HH", 52002), ("ev-03", 1250, "DHD", 52003),
    ("ev-04", 1700, "HD", 52004), ("ev-05", 2400, "HHHDD", 52005), ("ev-06", 3000, "D" * 4 + "H" * 6, 52006),
    ("ev-07", 3850, "H" * 10 + "D" * 3, 52007), ("ev-08", 1000, "DDD", 52008),
)
CAL_SCHEDULE = {"sessions": {"s01": 61001, "s02": 61002, "s03": 61003}, "warmup_reps": 2, "reps": 12}
EVAL_SCHEDULE = {"sessions": {"s01": 62001, "s02": 62002, "s03": 62003, "s04": 62004}, "warmup_reps": 2, "reps": 10}
# Planning estimates only (not measured): per-sample wall time including reset, checks, page-cache
# preparation, the timed request and the strict check; per-case setup; per-session model load.
ESTIMATE = {"per_sample_s": 0.35, "per_case_setup_s": 3.0, "per_session_load_s": 30.0}
CAL_DEADLINE_S, EVAL_DEADLINE_S = 2400, 2400
ANALYSIS = {"block_len": 3, "n_boot": 2000, "level": 0.95, "ambiguity_alpha": 0.05, "seed": 7001,
            "partition": {"A": ["s01", "s03"], "B": ["s02", "s04"]}, "partition_weights": {"A": 0.5, "B": 0.5},
            "case_weight": "equal", "min_sessions_per_half": 2, "min_blocks_per_session": 2}
FIT = {"weighting": "uniform",
       "row_contribution": "one row per (case, candidate) mean; a case with k+1 candidates contributes k+1 rows"}


class SpecError(ValueError):
    """A specification that must not be executed (malformed, leaking, unidentifiable or out of domain)."""


# ---- prompts and identities ----------------------------------------------------------------

def prompt_tokens(prompt: dict) -> list:
    if prompt.get("generator") != "uniform_v1":
        raise SpecError(f"unknown prompt generator {prompt.get('generator')!r}")
    rng = random.Random(prompt["seed"])
    lo, hi = prompt["vocab"]
    return [rng.randrange(lo, hi) for _ in range(prompt["length"])]


def tokens_sha256(tokens) -> str:
    return hashlib.sha256(PROMPT_TAG + struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "little")


# ---- cases ---------------------------------------------------------------------------------

def make_case(case_id: str, split: str, N: int, chunks, seed: int, *, C_: int = C, host_lru=None, disk_lru=None,
              vocab=VOCAB) -> dict:
    """One fully specified case. `chunks` gives the state of every complete prompt chunk (len N // C)."""
    chunks = list(chunks)
    if len(chunks) != N // C_ or any(s not in STATES for s in chunks):
        raise SpecError(f"{case_id}: need one state of {STATES} per complete chunk ({N // C_}), got {chunks}")
    host = [i for i, s in enumerate(chunks) if s == "host"]
    disk = [i for i, s in enumerate(chunks) if s == "disk"]
    host_lru = host if host_lru is None else list(host_lru)
    disk_lru = disk if disk_lru is None else list(disk_lru)
    if sorted(host_lru) != host or sorted(disk_lru) != disk:
        raise SpecError(f"{case_id}: LRU orders must list exactly the host / disk chunks")
    n_elig = max_restore_len(N, C_) // C_
    k = 0
    while k < n_elig and chunks[k] != "absent":
        k += 1
    indexed_last = max(host + disk, default=-1)
    admitted = [i for i, s in enumerate(chunks) if s == "absent"]
    tokens = prompt_tokens({"generator": "uniform_v1", "seed": seed, "length": N, "vocab": list(vocab)})
    case = {
        "case_id": case_id, "split": split,
        "prompt": {"generator": "uniform_v1", "seed": seed, "length": N, "vocab": list(vocab),
                   "sha256": tokens_sha256(tokens)},
        "chunks": chunks, "host_lru": host_lru, "disk_lru": disk_lru,
        "feasible_chunks": k, "feasible_tiers": chunks[:k],
        "candidates": list(range(0, k * C_ + 1, C_)),
        "limit_reason": ("no_eligible_chunk" if n_elig == 0 else "eligible_limit" if k == n_elig else "absent"),
        "admitted": admitted,
        "capacities": {"host_slots": len(host) + len(admitted), "disk_files": len(disk)},
        "source": {"method": "one full prefill of prompt[:P]; chunk bytes copied by the M1 store path",
                   "P": (indexed_last + 1) * C_},
    }
    case["signature"] = configuration_signature(case, C_)
    return case


def configuration_signature(case: dict, C_: int) -> str:
    """The case's performance-relevant identity: everything but the id, split and prompt content."""
    return digest({"N": case["prompt"]["length"], "C": C_, "chunks": case["chunks"], "host_lru": case["host_lru"],
                   "disk_lru": case["disk_lru"], "capacities": case["capacities"], "P": case["source"]["P"]})


def grid_cases(grid, split: str, C_: int = C, vocab=VOCAB) -> list:
    out = []
    for cid, N, prefix, seed in grid:
        chunks = [STATE_LETTERS[x] for x in prefix] + ["absent"] * (N // C_ - len(prefix))
        out.append(make_case(cid, split, N, chunks, seed, C_=C_, vocab=vocab))
    return out


def run_labels(spec: dict, case: dict) -> list:
    labels = [f"cand:{L}" for L in case["candidates"]]
    if spec["split"] == "evaluation":
        labels += [f"policy:{p}" for p in POLICIES]
    return labels


# ---- design checks (before any measurement) ------------------------------------------------

def _prefix_counts(tiers):
    h = d = 0
    out = [(0, 0)]
    for t in tiers:
        h, d = h + (t == "host"), d + (t == "disk")
        out.append((h, d))
    return out


def design_report(cases, C_: int = C) -> dict:
    """Rank and condition number of the calibration design (uniform weights, the fit's column scaling and
    rank tolerance), and how many rows each case contributes."""
    rows, per_case = [], {}
    for c in cases:
        N, m = c["prompt"]["length"], len(c["admitted"])
        for L, (h, d) in zip(c["candidates"], _prefix_counts(c["feasible_tiers"])):
            rows.append(rc.design_row(N, L, h, d, m))
        per_case[c["case_id"]] = len(c["candidates"])
    X = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(X, axis=0)
    zero = [f"{g}.{f}" for (g, f), n in zip(rc._COLUMNS, norms) if n == 0]
    if zero:
        return {"rank": None, "zero_features": zero, "n_rows": len(rows), "rows_per_case": per_case}
    sv = np.linalg.svd(X / norms, compute_uv=False)
    total = len(rows)
    return {"rank": int(np.sum(sv > sv[0] * rc.RANK_RTOL)), "n_columns": len(rc._COLUMNS),
            "condition_number": float(sv[0] / sv[-1]), "n_rows": total, "zero_features": [],
            "rows_per_case": per_case, "row_share_per_case": {k: v / total for k, v in per_case.items()}}


def declared_domain(cal_cases, C_: int = C) -> dict:
    """The domain fit_restore_cost will record for these cases (the same formulas)."""
    pos = [c["prompt"]["length"] - L for c in cal_cases for L in c["candidates"] if L]
    return {"prompt_len": [min(c["prompt"]["length"] for c in cal_cases), max(c["prompt"]["length"] for c in cal_cases)],
            "suffix_len": [min(pos), max(pos)],
            "restore_chunks_max": max(c["feasible_chunks"] for c in cal_cases),
            "host_chunks_max": max(c["feasible_tiers"].count("host") for c in cal_cases),
            "disk_chunks_max": max(c["feasible_tiers"].count("disk") for c in cal_cases),
            "admitted_chunks": [min(len(c["admitted"]) for c in cal_cases), max(len(c["admitted"]) for c in cal_cases)]}


def domain_problems(dom: dict, case: dict, C_: int = C) -> list:
    """Decision-domain and prediction-domain violations of one case (restore_cost semantics: S bounds apply to
    L > 0 only; m is prediction-only; a case without a positive candidate needs only the prediction domain)."""
    N, out = case["prompt"]["length"], []
    if not dom["prompt_len"][0] <= N <= dom["prompt_len"][1]:
        out.append("prompt_len")
    for k, (h, d) in enumerate(_prefix_counts(case["feasible_tiers"])[1:], start=1):
        if k > dom["restore_chunks_max"]:
            out.append(f"restore_chunks at L={k * C_}")
        if h > dom["host_chunks_max"]:
            out.append(f"host_chunks at L={k * C_}")
        if d > dom["disk_chunks_max"]:
            out.append(f"disk_chunks at L={k * C_}")
        if not dom["suffix_len"][0] <= N - k * C_ <= dom["suffix_len"][1]:
            out.append(f"suffix_len at L={k * C_}")
    m = len(case["admitted"])
    if not dom["admitted_chunks"][0] <= m <= dom["admitted_chunks"][1]:
        out.append("admitted_chunks (prediction domain)")
    return out


# ---- specs ---------------------------------------------------------------------------------

def _counts(spec: dict) -> dict:
    sch = spec["schedule"]
    per_rep = sum(len(run_labels(spec, c)) for c in spec["cases"])
    n_s = len(sch["sessions"])
    per_session = per_rep * (sch["warmup_reps"] + sch["reps"])
    est = spec["budget"]["estimate"]
    session_s = per_session * est["per_sample_s"] + len(spec["cases"]) * est["per_case_setup_s"] + \
        est["per_session_load_s"]
    return {"runs_per_repetition": per_rep, "samples_per_session": per_session,
            "measured_samples_per_session": per_rep * sch["reps"],
            "warmup_samples_per_session": per_rep * sch["warmup_reps"],
            "samples_total": per_session * n_s, "measured_samples_total": per_rep * sch["reps"] * n_s,
            "candidate_rows": sum(len(c["candidates"]) for c in spec["cases"]),
            "estimated_session_s": round(session_s, 1), "estimated_batch_s": round(session_s * n_s, 1)}


def build_spec(name: str, split: str, cases: list, schedule: dict, *, C_: int = C, backend: str = BACKEND,
               group: int = GROUP, deadline_s: float, estimate=None, model=None, calibration=None,
               analysis=None, check_design: bool = True) -> dict:
    """Assemble and check a spec. `calibration` (evaluation only) is the calibration spec it is held out from.
    `check_design=False` is only for small correctness specs (the GPU gate's smoke session, CPU tests) that are
    never fitted; the production calibration spec is always checked."""
    if split not in ("calibration", "evaluation"):
        raise SpecError(f"split {split!r}")
    ids = [c["case_id"] for c in cases]
    sigs = [c["signature"] for c in cases]
    if len(set(ids)) != len(ids) or len(set(sigs)) != len(sigs):
        raise SpecError("duplicate case ids or duplicate configurations within the spec")
    if any(c["split"] != split for c in cases):
        raise SpecError("a case belongs to another split")
    spec = {"spec_version": SPEC_VERSION, "name": name, "split": split, "model": dict(model or MODEL),
            "chunk_size": C_, "restore_backend": backend,
            "restore_staging_chunks": group if backend == "batched" else None,
            "cases": copy.deepcopy(cases), "schedule": copy.deepcopy(schedule),
            "timed_region": "synchronize; t0 = perf_counter_ns(); engine.prefill_request(prompt); "
                            "int(argmax(logits)); t1 = perf_counter_ns()  (gc disabled inside)",
            "page_cache": {"procedure": "kvlab.warm_disk.prepare/complete: one observation pass, at most one "
                                        "prewarm per candidate file, one final pass, no retry",
                           "files": "disk files of the maximal feasible prefix (the union of all candidates' reads)",
                           "required_outcome": {"with_disk_files": "verified", "without": "empty"},
                           "drop_pagecache": "must be unchanged from before the preparation to after the request"},
            "budget": {"deadline_s": deadline_s, "estimate": dict(estimate or ESTIMATE),
                       "estimate_note": "planning estimate, not measured"},
            "analysis": copy.deepcopy(ANALYSIS if analysis is None else analysis), "fit": dict(FIT)}
    if split == "calibration":
        rep = design_report(cases, C_)
        if check_design and rep["rank"] != len(rc._COLUMNS):
            raise SpecError(f"calibration design is not identifiable: {rep}")
        spec["design_checked"] = check_design
        spec["design"] = rep
        spec["domain"] = declared_domain(cases, C_)
        spec["analysis"]["partition"] = None  # calibration has no policy analysis
    else:
        if calibration is None:
            raise SpecError("an evaluation spec needs its calibration spec")
        cal_sigs = {c["signature"] for c in calibration["cases"]}
        cal_prompts = {c["prompt"]["sha256"] for c in calibration["cases"]}
        clash = [c["case_id"] for c in cases if c["signature"] in cal_sigs or c["prompt"]["sha256"] in cal_prompts]
        if clash:
            raise SpecError(f"evaluation cases duplicate calibration configurations or prompts: {clash}")
        for k in ("chunk_size", "restore_backend", "restore_staging_chunks", "model"):
            if calibration[k] != spec[k]:
                raise SpecError(f"evaluation {k} differs from calibration")
        bad = {c["case_id"]: p for c in cases if (p := domain_problems(calibration["domain"], c, C_))}
        if bad:
            raise SpecError(f"evaluation cases outside the declared calibration domain: {bad}")
        spec["calibration_spec_sha256"] = digest(calibration)
        spec["calibration_case_ids"] = [c["case_id"] for c in calibration["cases"]]
        part = spec["analysis"]["partition"]
        if sorted(part["A"] + part["B"]) != sorted(schedule["sessions"]) or set(part["A"]) & set(part["B"]):
            raise SpecError("the split-sample partition must split the sessions")
    spec["counts"] = _counts(spec)
    return spec


def calibration_spec() -> dict:
    return build_spec(CAL_NAME, "calibration", grid_cases(CALIBRATION_GRID, "calibration"), CAL_SCHEDULE,
                      deadline_s=CAL_DEADLINE_S)


def evaluation_spec(cal: dict | None = None) -> dict:
    cal = cal or calibration_spec()
    return build_spec(EVAL_NAME, "evaluation", grid_cases(EVALUATION_GRID, "evaluation"), EVAL_SCHEDULE,
                      deadline_s=EVAL_DEADLINE_S, calibration=cal)


def case_by_id(spec: dict) -> dict:
    return {c["case_id"]: c for c in spec["cases"]}


# ---- schedule ------------------------------------------------------------------------------

def case_order(spec: dict, session: str) -> list:
    ids = [c["case_id"] for c in spec["cases"]]
    random.Random(_seed(spec["name"], spec["schedule"]["sessions"][session], "cases")).shuffle(ids)
    return ids


def run_order(spec: dict, session: str, case: dict, phase: str, rep: int) -> list:
    labels = run_labels(spec, case)
    random.Random(_seed(spec["name"], spec["schedule"]["sessions"][session], case["case_id"], phase, rep)).shuffle(labels)
    return labels


def schedule(spec: dict, session: str) -> list:
    """Every sample of a session in execution order: (case_id, phase, rep, pos, label)."""
    by_id, sch, out = case_by_id(spec), spec["schedule"], []
    for cid in case_order(spec, session):
        case = by_id[cid]
        for phase, reps in (("warmup", sch["warmup_reps"]), ("measured", sch["reps"])):
            for r in range(reps):
                out.extend((cid, phase, r, pos, label) for pos, label in enumerate(run_order(spec, session, case, phase, r)))
    return out


def parse_label(label: str) -> tuple:
    kind, _, value = label.partition(":")
    if kind == "cand":
        return "candidate", int(value)
    if kind == "policy" and value in POLICIES:
        return "policy", value
    raise SpecError(f"bad run label {label!r}")


def expected_selected(case: dict, label: str, predictions: dict | None = None) -> int:
    kind, v = parse_label(label)
    if kind == "candidate":
        return v
    if v == "always_restore":
        return case["candidates"][-1]
    if v == "recompute":
        return 0
    if predictions is None:
        raise SpecError("cost_aware runs need the frozen predictions")
    return predictions["cases"][case["case_id"]]["decision"]["selected_len"]


def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def expected_structure(case: dict, key_hex: list, file_bytes: int) -> dict:
    """The store structure a correctly prepared (or reset) case must have, from the spec alone:
    host chunks inserted in host-LRU order into slots 0, 1, ... (HostStore hands out slot 0 first), the
    remaining host slots free for exactly the expected admissions, disk files written in disk-LRU order, no
    pins, no busy staging buffer, no temp or undeletable bytes. File inode/size/mtime are not predictable:
    they are compared with the case's own setup observation instead."""
    h, d = len(case["host_lru"]), len(case["disk_lru"])
    slots = case["capacities"]["host_slots"]
    return {"host_lru": [key_hex[i] for i in case["host_lru"]], "host_slots": list(range(h)), "host_pins": [0] * h,
            "host_free": list(range(slots - 1, h - 1, -1)), "host_capacity": slots, "staging_slots": 1,
            "staging_busy": False,
            "disk_lru": [key_hex[i] for i in case["disk_lru"]], "disk_sizes": [file_bytes] * d,
            "disk_committed": d * file_bytes, "disk_tmp": 0, "disk_undeletable": [], "disk_budget": d * file_bytes,
            "disk_max_entries": d, "dir_names": sorted(f"{key_hex[i]}.kvc" for i in case["disk_lru"])}


def batch_plan(spec: dict, run_dir: str, python: str, *, cost_model_path=None, predictions_path=None,
               bindings=None, created_unix=None) -> dict:
    """The exact session commands (run from the repository root), in launch order, for
    kvlab.m3_confirm.run_batch (one deadline, stop on any nonzero exit, no retries)."""
    cmds = []
    for sid in spec["schedule"]["sessions"]:
        argv = [python, "scripts/m5_run.py", "session", "--split", spec["split"], "--run-dir", run_dir,
                "--session", sid]
        if spec["split"] == "evaluation":
            argv += ["--cost-model", cost_model_path, "--predictions", predictions_path]
        cmds.append({"session": sid, "argv": argv, "env": {"HF_HUB_OFFLINE": "1"}})
    return {"plan_version": "kvlab-m5-batch-plan-v1", "experiment": spec["name"], "spec_sha256": digest(spec),
            "run_dir": run_dir, "budget_s": spec["budget"]["deadline_s"], "commands": cmds,
            "bindings": dict(bindings or {}), "created_unix": time.time() if created_unix is None else created_unix}
