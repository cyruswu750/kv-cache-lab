"""M2: benchmark bookkeeping, statistics and the offline restore-vs-recompute cost model.

Pure CPU code (no model). Units: raw samples are integer nanoseconds from
time.perf_counter_ns(); everything reported, and every model coefficient, is in
milliseconds. L = restored prefix length, S = suffix length, N = L + S.
See docs/M2_PROTOCOL.md for timing boundaries and the experimental protocol.
"""

import hashlib
import itertools
import json

import numpy as np

C = 256
PATHS = ("full", "host", "disk")            # measured, modeled
DIAGNOSTIC_PATHS = ("disk_evicted",)        # measured only (S == 1), not modeled
RESTORE_TIERS = ("host", "disk")
PHASES = ("ingest", "restore", "prefill", "store", "select")
MIN_SAMPLES = 10                            # fewer valid samples than this -> no statistics

L_GRID = (256, 512, 1024, 2048, 4096)
S_GRID = (1, 64, 256)
CALIBRATION = tuple((L, S) for L in (256, 1024, 4096) for S in (1, 256))
EVALUATION = tuple((L, S) for L in L_GRID for S in S_GRID if (L, S) not in CALIBRATION)
SMOKE = ((256, 1), (1024, 64))
SPLITS = {"smoke": SMOKE, "calibration": CALIBRATION, "evaluation": EVALUATION}


def ns_to_ms(ns) -> float:
    return ns / 1e6


def check_case(L: int, S: int) -> None:
    if L <= 0 or L % C or S < 1:
        raise ValueError(f"invalid case L={L} S={S}: need chunk-aligned L > 0 and S >= 1")
    if S > C:
        raise ValueError(f"S={S} > {C}: the eligible restore length would exceed L")


def paths_for(S: int) -> tuple:
    return PATHS + (DIAGNOSTIC_PATHS if S == 1 else ())


def expected(path: str, L: int, S: int) -> dict:
    """What a correct measurement of `path` must report for the prefix state [0, L) cached."""
    check_case(L, S)
    n_prefix, n_complete = L // C, (L + S) // C
    new_chunks = list(range(n_prefix, n_complete))  # complete prompt chunks not in the prepared state
    if path == "full":
        return {"restore_len": 0, "sources": [], "stored_chunks": new_chunks}
    tier = {"host": "host", "disk": "disk", "disk_evicted": "disk"}[path]
    return {"restore_len": L, "sources": [tier] * n_prefix, "stored_chunks": new_chunks}


def sample_problems(sample: dict) -> list[str]:
    """Reasons a raw sample must be excluded: wrong tier, wrong restore length, wrong
    admissions, missing or non-positive timing, or a cold-read variant that was not cold."""
    problems = []
    want = expected(sample["path"], sample["L"], sample["S"])
    for k, v in want.items():
        if sample.get(k) != v:
            problems.append(f"{k}={sample.get(k)!r}, expected {v!r}")
    if not isinstance(sample.get("e2e_ns"), int) or sample["e2e_ns"] <= 0:
        problems.append(f"e2e_ns={sample.get('e2e_ns')!r}")
    if sample.get("instrumented"):
        phases = sample.get("phases_ns") or {}
        if set(phases) != set(PHASES) or any(not isinstance(v, int) or v < 0 for v in phases.values()):
            problems.append(f"phases_ns={phases!r}")
    if sample["path"] == "disk_evicted" and sample.get("resident_fraction_before", 1.0) > 0.01:
        problems.append(f"page cache not evicted (resident fraction {sample.get('resident_fraction_before')})")
    if sample["path"] == "disk" and sample.get("resident_fraction_before", 0.0) < 0.999:
        problems.append(f"page cache not warm (resident fraction {sample.get('resident_fraction_before')})")
    return problems


def partition(samples: list[dict]) -> tuple[list, list]:
    """(valid, excluded). Warmup samples are neither: they are kept raw but never used."""
    valid, excluded = [], []
    for s in samples:
        if s.get("warmup"):
            continue
        problems = sample_problems(s)
        if problems:
            excluded.append({**s, "excluded_because": problems})
        else:
            valid.append(s)
    return valid, excluded


def summarize(values_ms, seed: int = 0, n_boot: int = 2000) -> dict:
    v = np.asarray(list(values_ms), dtype=float)
    out = {"n": int(v.size)}
    if v.size < MIN_SAMPLES:
        return {**out, "status": "insufficient_samples"}
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {**out, "status": "ok", "median_ms": float(p50), "p25_ms": float(p25), "p75_ms": float(p75),
            "iqr_ms": float(p75 - p25), "min_ms": float(v.min()), "max_ms": float(v.max()), "mean_ms": float(v.mean()),
            "slow_share_gt_1p3x_min": float(np.mean(v > 1.3 * v.min())),
            "median_ci95_ms": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def paired_difference(a_by_rep: dict, b_by_rep: dict, seed: int = 0, n_boot: int = 2000) -> dict:
    """Median of per-repetition differences a - b (ms) over repetitions present in both,
    with a bootstrap 95% CI. Paths of one repetition run back to back (random order)."""
    reps = sorted(set(a_by_rep) & set(b_by_rep))
    d = np.asarray([a_by_rep[r] - b_by_rep[r] for r in reps], dtype=float)
    if d.size < MIN_SAMPLES:
        return {"n": int(d.size), "status": "insufficient_samples"}
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(d, size=(n_boot, d.size), replace=True), axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"n": int(d.size), "status": "ok", "median_ms": float(np.median(d)), "ci95_ms": [lo, hi]}


def measured_winner(diff_restore_minus_full: dict) -> str:
    """'restore' / 'full' when the 95% CI of the paired median difference excludes 0, else 'tie'."""
    if diff_restore_minus_full.get("status") != "ok":
        return "unknown"
    lo, hi = diff_restore_minus_full["ci95_ms"]
    return "restore" if hi < 0 else "full" if lo > 0 else "tie"


# ---- tiny exact non-negative least squares ------------------------------------------------

def nnls_small(X, y, weights=None) -> np.ndarray:
    """Exact NNLS for a handful of columns: the optimum's support is some column subset on
    which the unconstrained least-squares solution is non-negative; try all subsets."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        X, y = X * w[:, None], y * w
    k = X.shape[1]
    if k > 8:
        raise ValueError("nnls_small is for at most 8 columns")
    best, best_res = np.zeros(k), float(y @ y)
    for mask in itertools.product((False, True), repeat=k):
        cols = [i for i in range(k) if mask[i]]
        if not cols:
            continue
        sol, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        if np.any(sol < 0):
            continue
        coef = np.zeros(k)
        coef[cols] = sol
        r = y - X @ coef
        if float(r @ r) < best_res - 1e-15:
            best, best_res = coef, float(r @ r)
    return best


# ---- cost model ---------------------------------------------------------------------------
# Token counts enter in kilotokens (kt) so coefficients stay readable: e.g. "ms per kt".

def _kt(x):
    return x / 1000.0


COMPONENTS = {
    # name: (feature names, feature function of the case, description)
    "ingest": (("const", "N_kt"), lambda L, S, n_new: [1.0, _kt(L + S)],
               "token ids to device + chain keys (SHA-256 over the prompt)"),
    "restore_host": (("const", "chunks"), lambda L, S, n_new: [1.0, L / C],
                     "lookup + per-chunk lease/identity check + H2D + DynamicCache assembly"),
    "restore_disk": (("const", "chunks"), lambda L, S, n_new: [1.0, L / C],
                     "lookup + per-chunk file read/validation/CRC into staging + H2D + assembly"),
    # Attention work ~ query-key pairs: S*(L + S/2) for a suffix, N^2/2 for a full causal prefill.
    # (With calibration S in {1, 256}, separate S and S^2 terms would not be identifiable.)
    "prefill_suffix": (("const", "S_kt", "S_kt*(L_kt+S_kt/2)"),
                       lambda L, S, n_new: [1.0, _kt(S), _kt(S) * (_kt(L) + _kt(S) / 2)],
                       "forward over S new tokens with L cached tokens (explicit-mask SDPA path)"),
    "prefill_full": (("const", "N_kt", "N_kt^2/2"), lambda L, S, n_new: [1.0, _kt(L + S), _kt(L + S) ** 2 / 2],
                     "forward over all N tokens, empty cache (causal SDPA path)"),
    "store": (("const", "new_chunks"), lambda L, S, n_new: [1.0, float(n_new)],
              "admission of new complete chunks: D2H into free pinned host slots"),
}
PATH_COMPONENTS = {
    "full": ("ingest", "restore_skipped", "prefill_full", "store", "select"),
    "host": ("ingest", "restore_host", "prefill_suffix", "store", "select"),
    "disk": ("ingest", "restore_disk", "prefill_suffix", "store", "select"),
}
MODEL_VERSION = "kvlab-m2-cost-model-v1"


def _n_new(L, S):
    return len(expected("full", L, S)["stored_chunks"])


def case_medians(valid: list[dict]) -> dict:
    """{(path, L, S): {"e2e": median ms (uninstrumented), "phases": {phase: median ms},
    "instrumented_total": median ms}} from valid samples."""
    out = {}
    keys = sorted({(s["path"], s["L"], s["S"]) for s in valid})
    for key in keys:
        group = [s for s in valid if (s["path"], s["L"], s["S"]) == key]
        e2e = [ns_to_ms(s["e2e_ns"]) for s in group if not s.get("instrumented")]
        inst = [s for s in group if s.get("instrumented")]
        rec = {"n_e2e": len(e2e), "n_instrumented": len(inst)}
        if len(e2e) >= MIN_SAMPLES:
            rec["e2e"] = float(np.median(e2e))
        if len(inst) >= MIN_SAMPLES:
            rec["phases"] = {p: float(np.median([ns_to_ms(s["phases_ns"][p]) for s in inst])) for p in PHASES}
            rec["instrumented_total"] = float(np.median([ns_to_ms(s["e2e_ns"]) for s in inst]))
        out[key] = rec
    return out


def fit_cost_model(medians: dict, calibration_cases=CALIBRATION) -> dict:
    """Fit every component on calibration medians only (relative-error weighted NNLS).
    Raises if a case outside `calibration_cases` is present or a needed median is missing."""
    cal = set(calibration_cases)
    rows = {name: ([], []) for name in COMPONENTS}
    const_rows = {"restore_skipped": [], "select": []}
    offsets = {p: [] for p in PATHS}
    for (path, L, S), rec in sorted(medians.items()):
        if path not in PATHS:
            continue
        if (L, S) not in cal:
            raise ValueError(f"case L={L} S={S} is not a calibration case; refusing to fit on it")
        if "phases" not in rec or "e2e" not in rec:
            raise ValueError(f"calibration case {path} L={L} S={S} lacks medians: {rec}")
        ph, n_new = rec["phases"], _n_new(L, S)

        def add(name, y):
            rows[name][0].append(COMPONENTS[name][1](L, S, n_new))
            rows[name][1].append(y)
        add("ingest", ph["ingest"])
        add("store", ph["store"])
        const_rows["select"].append(ph["select"])
        if path == "full":
            const_rows["restore_skipped"].append(ph["restore"])
            add("prefill_full", ph["prefill"])
        else:
            add(f"restore_{path}", ph["restore"])
            add("prefill_suffix", ph["prefill"])
        offsets[path].append(rec["e2e"] - rec["instrumented_total"])
    model = {"version": MODEL_VERSION, "units": "ms; token features in kilotokens (kt); chunk = 256 tokens",
             "calibration_cases": [list(c) for c in sorted(cal)], "components": {}}
    for name, (X, y) in rows.items():
        if len(y) < len(COMPONENTS[name][0]):
            raise ValueError(f"component {name}: {len(y)} calibration rows for {len(COMPONENTS[name][0])} terms")
        y_arr = np.asarray(y)
        coef = nnls_small(X, y_arr, weights=1.0 / y_arr)
        pred = np.asarray(X) @ coef
        model["components"][name] = {
            "features": list(COMPONENTS[name][0]), "coef_ms": [float(c) for c in coef],
            "description": COMPONENTS[name][2], "n_rows": len(y),
            "in_sample_rel_err": [float(e) for e in (pred - y_arr) / y_arr]}
    for name, ys in const_rows.items():
        model["components"][name] = {"features": ["const"], "coef_ms": [float(np.median(ys))],
                                     "description": {"restore_skipped": "full path: restore not attempted",
                                                     "select": "argmax + transfer of the token id to the host"}[name],
                                     "n_rows": len(ys)}
    model["instrumentation_offset_ms"] = {p: float(np.median(v)) for p, v in offsets.items() if v}
    model["sha256"] = model_digest(model)
    return model


def model_digest(model: dict) -> str:
    body = {k: v for k, v in model.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def predict(model: dict, path: str, L: int, S: int) -> dict:
    """Predicted uninstrumented first-token latency (ms) and its components."""
    if model.get("sha256") != model_digest(model):
        raise ValueError("cost model digest mismatch (model was modified after it was frozen)")
    check_case(L, S)
    n_new = _n_new(L, S)
    parts = {}
    for name in PATH_COMPONENTS[path]:
        comp = model["components"][name]
        x = COMPONENTS[name][1](L, S, n_new) if name in COMPONENTS else [1.0]
        parts[name] = float(np.dot(comp["coef_ms"], x))
    parts["instrumentation_offset"] = model["instrumentation_offset_ms"][path]
    return {"path": path, "L": L, "S": S, "predicted_ms": sum(parts.values()), "components_ms": parts}


def make_predictions(model: dict, cases, extra: dict | None = None) -> dict:
    """Frozen predictions for `cases` (all PATHS), digested together with `extra`."""
    preds = [predict(model, path, L, S) for (L, S) in cases for path in PATHS]
    body = {"model_sha256": model["sha256"], "cases": [list(c) for c in cases], "predictions": preds,
            **(extra or {})}
    return {**body, "sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


SCAN_L = tuple(range(C, 4096 + 1, C))  # predicted break-even scan: the calibrated L range only


def scan_predictions(model: dict) -> list[dict]:
    return [predict(model, path, L, S) for S in S_GRID for L in SCAN_L for path in PATHS]


def predictions_digest(predictions: dict) -> str:
    body = {k: v for k, v in predictions.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# ---- scoring ------------------------------------------------------------------------------

def decisions(pred_by_case: dict, med: dict, diffs: dict) -> list[dict]:
    """Per (case, tier): predicted vs measured winner and the penalty of a wrong choice.
    pred_by_case[(path, L, S)] = predicted ms; med[(path, L, S)] = measured median ms;
    diffs[(tier, L, S)] = paired_difference(restore - full)."""
    out = []
    for (tier, L, S), diff in sorted(diffs.items()):
        p_r, p_f = pred_by_case[(tier, L, S)], pred_by_case[("full", L, S)]
        predicted = "restore" if p_r < p_f else "full"
        measured = measured_winner(diff)
        rec = {"tier": tier, "L": L, "S": S, "predicted_winner": predicted, "measured_winner": measured,
               "predicted_restore_minus_full_ms": p_r - p_f, "measured_paired_diff": diff}
        if measured in ("restore", "full"):
            rec["correct"] = predicted == measured
            chosen = med[(tier if predicted == "restore" else "full", L, S)]
            best = min(med[(tier, L, S)], med[("full", L, S)])
            rec["penalty_ms"] = chosen - best
            rec["penalty_rel"] = (chosen - best) / best
        else:
            rec["correct"] = None  # no reliable measured winner: not scorable
            rec["penalty_ms"] = None
        out.append(rec)
    return out


def sign_changes(values_by_L: dict) -> list[dict]:
    """Adjacent chunk-aligned L values whose sign (restore - full) differs; 0 counts as a tie."""
    Ls = sorted(values_by_L)
    out = []
    for a, b in zip(Ls, Ls[1:]):
        va, vb = values_by_L[a], values_by_L[b]
        if (va < 0) != (vb < 0) and va != 0 and vb != 0:
            out.append({"L_low": a, "L_high": b, "low_side": "restore" if va < 0 else "full",
                        "high_side": "restore" if vb < 0 else "full"})
    return out


def measured_crossings(decision_rows: list[dict]) -> dict:
    """{(tier, S): crossings} from measured winners; ties/unknowns break the comparison."""
    out = {}
    for tier in RESTORE_TIERS:
        for S in sorted({r["S"] for r in decision_rows}):
            rows = sorted((r for r in decision_rows if r["tier"] == tier and r["S"] == S), key=lambda r: r["L"])
            seq = [(r["L"], r["measured_winner"]) for r in rows]
            res = []
            for (la, wa), (lb, wb) in zip(seq, seq[1:]):
                if {wa, wb} == {"restore", "full"}:
                    res.append({"L_low": la, "L_high": lb, "low_side": wa, "high_side": wb})
                elif wa != wb:
                    res.append({"L_low": la, "L_high": lb, "low_side": wa, "high_side": wb, "inconclusive": True})
            out[(tier, S)] = {"measured_L": [L for L, _ in seq], "winners": [w for _, w in seq], "changes": res}
    return out
