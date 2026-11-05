"""M5a: a versioned, interpretable restore-vs-recompute latency model (CPU only, no torch).

Design: docs/M5_DESIGN.md §5. NO calibrated artifact exists. M5a provides the model form,
fitting, artifact validation and loading; coefficients must come from a future calibration
(docs/M5_PROTOCOL_DRAFT.md). Tests use synthetic data only. The M2 model (kvlab/perf.py:
baseline backend, S <= C, medians) is a different, frozen model and is never loaded here.

Units: ms; token features in kilotokens (kt = 1000 tokens); pair features in kt^2; chunk
counts. For one request with prompt length N and a candidate restore length L (S = N - L
recomputed tokens; h host and d disk chunks in [0, L), in the order restore() reads them;
m admitted chunks = complete prompt chunks not indexed before the request):

  L = 0:  T = f_const + f_n * N_kt + f_pairs * N_kt^2 / 2                          + a * m
  L > 0:  T = p_const + r_host * h + r_disk * d + p_s * S_kt
              + p_pairs * S_kt * (L_kt + S_kt / 2)                                 + a * m

The pair features count query-key pairs of a causal forward (N^2/2 for a full prefill,
S*L + S^2/2 for a suffix), so S^2 enters only through the pair count. Omitting a free S^2
coefficient is a chosen model constraint (attention work proportional to pairs), not an
identifiability limit: with varied suffix lengths such a column could be identified. Full and suffix prefill have separate coefficients (different attention paths,
causal vs explicit mask). `a * m` (admission) is shared by both paths and is the same for every
candidate of one request under the controlled state (candidates do not change what is
admitted), so the policy objective omits it; it stays in fitted predictions and in every
measurement.

Target: the case mean of synchronous first-token latency of each candidate (for mean regret).
Fit: non-negative least squares over all candidate rows, exact support enumeration on a
column-scaled design, with an explicit weighting ("uniform": absolute ms; "inverse_mean":
relative error). `perf.nnls_small` uses the same algorithm but caps at 8 columns; this basis has
9 (perf.py is frozen and unchanged).
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kvlab.boundaries import max_restore_len
from kvlab.identity import canonical_json

SCHEMA = "kvlab-restore-cost-v1"
_DIGEST_TAG = b"kvlab-restore-cost-v1\0"
BACKENDS = ("baseline", "batched")  # == kvlab.tiered_cache.RESTORE_BACKENDS (tested; avoids importing torch)
TIERS = ("host", "disk")
WEIGHTINGS = ("uniform", "inverse_mean")
FIT_METHOD = "nnls_exact_support_enumeration_column_scaled"
RANK_RTOL = 1e-9  # singular values below RANK_RTOL * largest (column-scaled design) count as rank-deficient
FEATURES = {"full": ("const", "n_kt", "n_pairs_kt2"),
            "partial": ("const", "host_chunks", "disk_chunks", "s_kt", "s_pairs_kt2"),
            "shared": ("admitted_chunks",)}
_COLUMNS = tuple((group, f) for group in ("full", "partial", "shared") for f in FEATURES[group])
UNITS = {"latency": "ms", "token_features": "kilotokens (1 kt = 1000 tokens)", "pair_features": "kt^2",
         "chunk_features": "chunks"}
TARGET = ("case mean of synchronous first-token latency (ms) of one candidate restore length, "
          "measured from a controlled pre-request state")
OBJECTIVE = ("policy objective = predicted latency minus the shared admitted_chunks term, which is identical "
             "for every candidate of one request when the candidate does not change what is admitted")
_TOP_KEYS = {"schema", "units", "target", "objective", "features", "compatibility", "coefficients_ms", "fit",
             "domain", "calibration", "provenance", "digest"}
_COMPAT_KEYS = {"namespace_digest", "chunk_size", "restore_backend", "restore_staging_chunks"}
_FIT_KEYS = {"method", "weighting", "n_cases", "n_rows", "rank", "condition_number", "max_abs_residual_ms",
             "max_abs_relative_residual"}
_DOMAIN_KEYS = {"prompt_len", "suffix_len", "restore_chunks_max", "host_chunks_max", "disk_chunks_max",
                "admitted_chunks"}
_CAL_KEYS = {"case_ids", "evaluation_case_ids", "cases_sha256"}
_CASE_KEYS = {"case_id", "prompt_len", "tiers", "admitted_chunks", "candidates"}
_CAND_KEYS = {"restore_len", "mean_ms", "n_samples"}


class FitError(ValueError):
    """Calibration input that must not be fitted (malformed, leaked, incomplete or not identifiable)."""


class ArtifactError(ValueError):
    """A malformed, tampered or unsupported model artifact."""


class IncompatibleModelError(ValueError):
    """A valid model used with a different namespace, chunk size, backend or staging group."""


class OutOfDomainError(ValueError):
    """A prediction requested outside the model's declared calibration domain."""


# ---- small validators ----------------------------------------------------------------------

def _int(v, name, lo, err):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise err(f"{name} must be an int >= {lo}, got {v!r}")
    return v


def _num(v, name, err, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise err(f"{name} must be a finite number, got {v!r}")
    if v < 0 or (positive and v == 0):
        raise err(f"{name} must be {'> 0' if positive else '>= 0'}, got {v!r}")
    return float(v)


def _hex64(v, name, err):
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise err(f"{name} must be 64 lowercase hex characters, got {v!r}")
    return v


def _keys(d, want, name, err):
    if not isinstance(d, dict):
        raise err(f"{name} must be a dict, got {type(d).__name__}")
    if set(d) != want:
        raise err(f"{name}: missing {sorted(want - set(d))}, unexpected {sorted(set(d) - want)}")


def _check_compatibility(c, err) -> dict:
    _keys(c, _COMPAT_KEYS, "compatibility", err)
    _hex64(c["namespace_digest"], "compatibility.namespace_digest", err)
    _int(c["chunk_size"], "compatibility.chunk_size", 1, err)
    if c["restore_backend"] not in BACKENDS:
        raise err(f"compatibility.restore_backend must be one of {BACKENDS}, got {c['restore_backend']!r}")
    g = c["restore_staging_chunks"]
    if c["restore_backend"] == "batched":
        _int(g, "compatibility.restore_staging_chunks", 1, err)
    elif g is not None:
        raise err(f"compatibility.restore_staging_chunks must be null for the baseline backend, got {g!r}")
    return dict(c)


def _id_list(ids, name, err) -> list:
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__iter__"):
        raise err(f"{name} must be a collection of case ids, got {type(ids).__name__}")
    out = list(ids)
    if any(not isinstance(i, str) or not i for i in out):
        raise err(f"{name} must contain non-empty strings")
    if len(set(out)) != len(out):
        raise err(f"{name} contains duplicates")
    return sorted(out)


def _prefix_counts(tiers):
    """(h, d) after each chunk of `tiers`, in order."""
    h = d = 0
    out = []
    for t in tiers:
        h, d = h + (t == "host"), d + (t == "disk")
        out.append((h, d))
    return out


def design_row(prompt_len: int, restore_len: int, host_chunks: int, disk_chunks: int, admitted_chunks: int):
    """Feature vector, in _COLUMNS order, of one candidate (inputs already validated)."""
    x = [0.0] * len(_COLUMNS)
    if restore_len == 0:
        n = prompt_len / 1000.0
        x[0:3] = [1.0, n, n * n / 2.0]
    else:
        s, l_kt = (prompt_len - restore_len) / 1000.0, restore_len / 1000.0
        x[3:8] = [1.0, float(host_chunks), float(disk_chunks), s, s * (l_kt + s / 2.0)]
    x[8] = float(admitted_chunks)
    return x


# ---- fitting -------------------------------------------------------------------------------

def _check_case(case, C) -> dict:
    """One calibration case: a request/state with every legal candidate measured once."""
    _keys(case, _CASE_KEYS, "case", FitError)
    cid = case["case_id"]
    if not isinstance(cid, str) or not cid:
        raise FitError(f"case_id must be a non-empty string, got {cid!r}")
    N = _int(case["prompt_len"], f"{cid}: prompt_len", 1, FitError)
    tiers = case["tiers"]
    if not isinstance(tiers, (list, tuple)) or any(t not in TIERS for t in tiers):
        raise FitError(f"{cid}: tiers must be a list of {TIERS}, got {tiers!r}")
    k = len(tiers)
    if k * C > max_restore_len(N, C):
        raise FitError(f"{cid}: {k} prepared chunks exceed the first-token rule for N={N} (C={C})")
    m = _int(case["admitted_chunks"], f"{cid}: admitted_chunks", 0, FitError)
    if m > N // C - k:
        raise FitError(f"{cid}: admitted_chunks={m} > the {N // C - k} complete chunks outside the prepared prefix")
    cands = case["candidates"]
    if not isinstance(cands, (list, tuple)):
        raise FitError(f"{cid}: candidates must be a list")
    out = []
    for c in cands:
        _keys(c, _CAND_KEYS, f"{cid}: candidate", FitError)
        L = _int(c["restore_len"], f"{cid}: restore_len", 0, FitError)
        out.append({"restore_len": L, "mean_ms": _num(c["mean_ms"], f"{cid}: L={L} mean_ms", FitError, True),
                    "n_samples": _int(c["n_samples"], f"{cid}: L={L} n_samples", 1, FitError)})
    out.sort(key=lambda c: c["restore_len"])
    want = list(range(0, k * C + 1, C))
    if [c["restore_len"] for c in out] != want:
        raise FitError(f"{cid}: candidates {[c['restore_len'] for c in out]} must be every legal length {want}, "
                       "each exactly once")
    return {"case_id": cid, "prompt_len": N, "tiers": list(tiers), "admitted_chunks": m, "candidates": out}


def _nnls_exact(X, y) -> np.ndarray:
    """Exact NNLS: the optimum's support is a column subset whose unconstrained least-squares
    solution is non-negative; try every subset (X has full column rank, checked by the caller)."""
    k = X.shape[1]
    if k > 10:
        raise ValueError("_nnls_exact is for at most 10 columns")
    best, best_res = np.zeros(k), float(y @ y)
    for mask in range(1, 1 << k):
        cols = [i for i in range(k) if mask >> i & 1]
        sol, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        if np.any(sol < 0):
            continue
        coef = np.zeros(k)
        coef[cols] = sol
        r = y - X @ coef
        if float(r @ r) < best_res - 1e-15:
            best, best_res = coef, float(r @ r)
    return best


def fit_restore_cost(cases, *, calibration_case_ids, evaluation_case_ids, compatibility: dict, weighting: str,
                     provenance: dict) -> dict:
    """Fit on calibration cases only and return a digested artifact (dict).

    Every case must be a declared calibration case; a declared evaluation case, an undeclared
    case, a duplicate or a missing declared case raises FitError (nothing is filtered).
    Each case must contain every legal candidate exactly once. The design must have full column
    rank; row counts alone are not accepted as identifiability."""
    compat = _check_compatibility(compatibility, FitError)
    C = compat["chunk_size"]
    if weighting not in WEIGHTINGS:
        raise FitError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    cal_ids = _id_list(calibration_case_ids, "calibration_case_ids", FitError)
    eval_ids = _id_list(evaluation_case_ids, "evaluation_case_ids", FitError)
    if not cal_ids:
        raise FitError("no calibration case ids declared")
    overlap = set(cal_ids) & set(eval_ids)
    if overlap:
        raise FitError(f"case ids declared both calibration and evaluation: {sorted(overlap)[:5]}")
    if not isinstance(provenance, dict):
        raise FitError("provenance must be a dict")
    try:
        canonical_json(provenance)
    except (TypeError, ValueError) as e:
        raise FitError(f"provenance is not canonical JSON: {e}") from None

    norm, seen = [], set()
    for case in cases:
        cid = case.get("case_id") if isinstance(case, dict) else None
        if cid in eval_ids:
            raise FitError(f"case {cid!r} is a declared evaluation case; refusing to fit on it")
        if cid not in cal_ids:
            raise FitError(f"case {cid!r} is not a declared calibration case; refusing to fit on it")
        if cid in seen:
            raise FitError(f"duplicate case {cid!r}")
        seen.add(cid)
        norm.append(_check_case(case, C))
    missing = sorted(set(cal_ids) - seen)
    if missing:
        raise FitError(f"declared calibration cases without data: {missing[:5]}")
    norm.sort(key=lambda c: c["case_id"])

    rows, y = [], []
    for c in norm:
        counts = [(0, 0)] + _prefix_counts(c["tiers"])
        for cand in c["candidates"]:
            h, d = counts[cand["restore_len"] // C]
            rows.append(design_row(c["prompt_len"], cand["restore_len"], h, d, c["admitted_chunks"]))
            y.append(cand["mean_ms"])
    X, y = np.asarray(rows, dtype=float), np.asarray(y, dtype=float)
    w = np.ones_like(y) if weighting == "uniform" else 1.0 / y
    Xw, yw = X * w[:, None], y * w
    norms = np.linalg.norm(Xw, axis=0)
    zero = [f"{g}.{f}" for (g, f), n in zip(_COLUMNS, norms) if n == 0]
    if zero:
        raise FitError(f"features never exercised by the calibration cases: {zero}")
    Xn = Xw / norms
    sv = np.linalg.svd(Xn, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * RANK_RTOL))
    if rank < len(_COLUMNS):
        raise FitError(f"design matrix rank {rank} < {len(_COLUMNS)} features: the calibration cases do not "
                       "identify the model (vary N, host and disk counts, suffix lengths and admitted chunks)")
    coef = _nnls_exact(Xn, yw) / norms
    resid = X @ coef - y
    coefficients = {g: {} for g in FEATURES}
    for (g, f), v in zip(_COLUMNS, coef):
        coefficients[g][f] = float(v)
    positive = [(c["prompt_len"] - cand["restore_len"]) for c in norm for cand in c["candidates"]
                if cand["restore_len"]]
    artifact = {
        "schema": SCHEMA, "units": dict(UNITS), "target": TARGET, "objective": OBJECTIVE,
        "features": {g: list(v) for g, v in FEATURES.items()},
        "compatibility": compat,
        "coefficients_ms": coefficients,
        "fit": {"method": FIT_METHOD, "weighting": weighting, "n_cases": len(norm), "n_rows": len(y),
                "rank": rank, "condition_number": float(sv[0] / sv[-1]),
                "max_abs_residual_ms": float(np.max(np.abs(resid))),
                "max_abs_relative_residual": float(np.max(np.abs(resid) / y))},
        "domain": {"prompt_len": [min(c["prompt_len"] for c in norm), max(c["prompt_len"] for c in norm)],
                   "suffix_len": [min(positive), max(positive)],
                   "restore_chunks_max": max(len(c["tiers"]) for c in norm),
                   "host_chunks_max": max(c["tiers"].count("host") for c in norm),
                   "disk_chunks_max": max(c["tiers"].count("disk") for c in norm),
                   "admitted_chunks": [min(c["admitted_chunks"] for c in norm),
                                       max(c["admitted_chunks"] for c in norm)]},
        "calibration": {"case_ids": cal_ids, "evaluation_case_ids": eval_ids,
                        "cases_sha256": _sha256(canonical_json(norm))},
        "provenance": copy.deepcopy(provenance),
    }
    artifact["digest"] = artifact_digest(artifact)
    RestoreCostModel(artifact)  # the fitted artifact must pass the loader's own validation
    return artifact


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def artifact_digest(artifact: dict) -> str:
    """SHA-256 over the canonical JSON of every field except `digest` (domain-separated)."""
    return _sha256(_DIGEST_TAG + canonical_json({k: v for k, v in artifact.items() if k != "digest"}))


# ---- artifact validation and the loaded model ----------------------------------------------

def _range(v, name, lo):
    if not isinstance(v, list) or len(v) != 2:
        raise ArtifactError(f"domain.{name} must be [min, max]")
    a, b = (_int(x, f"domain.{name}", lo, ArtifactError) for x in v)
    if a > b:
        raise ArtifactError(f"domain.{name}: min {a} > max {b}")
    return a, b


def _validate_artifact(artifact) -> dict:
    _keys(artifact, _TOP_KEYS, "artifact", ArtifactError)
    try:
        art = json.loads(canonical_json(artifact))  # a detached copy; also rejects non-JSON values
    except (TypeError, ValueError) as e:
        raise ArtifactError(f"artifact is not canonical JSON: {e}") from None
    if art["schema"] != SCHEMA:
        raise ArtifactError(f"unsupported schema {art['schema']!r}; expected {SCHEMA!r}")
    for key, want in (("units", UNITS), ("target", TARGET), ("objective", OBJECTIVE),
                      ("features", {g: list(v) for g, v in FEATURES.items()})):
        if art[key] != want:
            raise ArtifactError(f"{key} differs from this schema's definition")
    _check_compatibility(art["compatibility"], ArtifactError)
    co = art["coefficients_ms"]
    _keys(co, set(FEATURES), "coefficients_ms", ArtifactError)
    for g, names in FEATURES.items():
        _keys(co[g], set(names), f"coefficients_ms.{g}", ArtifactError)
        for f in names:
            _num(co[g][f], f"coefficients_ms.{g}.{f}", ArtifactError)
    fit = art["fit"]
    _keys(fit, _FIT_KEYS, "fit", ArtifactError)
    if fit["method"] != FIT_METHOD or fit["weighting"] not in WEIGHTINGS:
        raise ArtifactError(f"unsupported fit method/weighting {fit['method']!r}/{fit['weighting']!r}")
    for f in ("n_cases", "n_rows"):
        _int(fit[f], f"fit.{f}", 1, ArtifactError)
    if fit["rank"] != len(_COLUMNS) or isinstance(fit["rank"], bool):
        raise ArtifactError(f"fit.rank must be {len(_COLUMNS)}, got {fit['rank']!r}")
    for f in ("condition_number", "max_abs_residual_ms", "max_abs_relative_residual"):
        _num(fit[f], f"fit.{f}", ArtifactError)
    dom = art["domain"]
    _keys(dom, _DOMAIN_KEYS, "domain", ArtifactError)
    _range(dom["prompt_len"], "prompt_len", 1)
    _range(dom["suffix_len"], "suffix_len", 1)
    _range(dom["admitted_chunks"], "admitted_chunks", 0)
    kmax = _int(dom["restore_chunks_max"], "domain.restore_chunks_max", 1, ArtifactError)
    for f in ("host_chunks_max", "disk_chunks_max"):
        if _int(dom[f], f"domain.{f}", 1, ArtifactError) > kmax:
            raise ArtifactError(f"domain.{f} exceeds restore_chunks_max")
    cal = art["calibration"]
    _keys(cal, _CAL_KEYS, "calibration", ArtifactError)
    ids = _id_list(cal["case_ids"], "calibration.case_ids", ArtifactError)
    ev = _id_list(cal["evaluation_case_ids"], "calibration.evaluation_case_ids", ArtifactError)
    if ids != cal["case_ids"] or ev != cal["evaluation_case_ids"] or set(ids) & set(ev):
        raise ArtifactError("calibration case ids must be sorted, unique and disjoint from the evaluation ids")
    if len(ids) != fit["n_cases"]:
        raise ArtifactError("calibration.case_ids does not match fit.n_cases")
    _hex64(cal["cases_sha256"], "calibration.cases_sha256", ArtifactError)
    if not isinstance(art["provenance"], dict):
        raise ArtifactError("provenance must be a dict")
    _hex64(art["digest"], "digest", ArtifactError)
    if artifact_digest(art) != art["digest"]:
        raise ArtifactError("artifact digest mismatch (modified after it was produced)")
    return art


class RestoreCostModel:
    """A validated artifact, loaded once. Predictions are plain arithmetic on cached floats:
    nothing is re-serialized or re-hashed per request or candidate."""

    def __init__(self, artifact: dict):
        art = _validate_artifact(artifact)
        self.artifact, self.digest = art, art["digest"]
        co, c, dom = art["coefficients_ms"], art["compatibility"], art["domain"]
        self._full = tuple(co["full"][f] for f in FEATURES["full"])
        self._partial = tuple(co["partial"][f] for f in FEATURES["partial"])
        self._admit = co["shared"]["admitted_chunks"]
        self.namespace_digest, self.chunk_size = c["namespace_digest"], c["chunk_size"]
        self.restore_backend, self.restore_staging_chunks = c["restore_backend"], c["restore_staging_chunks"]
        self.prompt_range, self.suffix_range = tuple(dom["prompt_len"]), tuple(dom["suffix_len"])
        self.admitted_range = tuple(dom["admitted_chunks"])
        self.restore_chunks_max = dom["restore_chunks_max"]
        self.host_chunks_max, self.disk_chunks_max = dom["host_chunks_max"], dom["disk_chunks_max"]

    # ---- compatibility and domain
    def check_compatible(self, plan) -> None:
        problems = []
        if plan.namespace_digest != self.namespace_digest:
            problems.append("namespace digest")
        if plan.chunk_size != self.chunk_size:
            problems.append(f"chunk size {plan.chunk_size} != {self.chunk_size}")
        if plan.restore_backend != self.restore_backend:
            problems.append(f"restore backend {plan.restore_backend!r} != {self.restore_backend!r}")
        elif self.restore_backend == "batched" and plan.restore_staging_chunks != self.restore_staging_chunks:
            problems.append(f"staging group {plan.restore_staging_chunks} != {self.restore_staging_chunks}")
        if problems:
            raise IncompatibleModelError(f"cost model {self.digest[:12]} is incompatible: {'; '.join(problems)}")

    def _violation(self, N, k, h, d):
        if not self.prompt_range[0] <= N <= self.prompt_range[1]:
            return "prompt_len"
        if k > self.restore_chunks_max:
            return "restore_chunks"
        if h > self.host_chunks_max:
            return "host_chunks"
        if d > self.disk_chunks_max:
            return "disk_chunks"
        if k and not self.suffix_range[0] <= N - k * self.chunk_size <= self.suffix_range[1]:
            return "suffix_len"
        return None

    def domain_violation(self, plan):
        """The first *decision-domain* bound any feasible candidate of `plan` violates, else None.

        The decision domain is the declared box (per-field calibration ranges, not the calibration
        hull) projected onto the objective's inputs: N, and for each positive candidate S and the
        restore, host and disk chunk counts (the S range applies to L > 0 only). `CostAware` decides a
        plan without a positive candidate before calling this. The
        admitted-chunk range is a prediction-only field (checked by `predict_ms`): m does not enter
        the objective, and under the controlled-state assumption it is the same for every candidate
        of a request, so it cannot change the choice. A plan does not carry m."""
        N = plan.prompt_len
        if not self.prompt_range[0] <= N <= self.prompt_range[1]:
            return "prompt_len"
        for k, (h, d) in enumerate(_prefix_counts(plan.feasible_tiers), start=1):
            v = self._violation(N, k, h, d)
            if v is not None:
                return v
        return None

    # ---- predictions
    def _objective(self, N, L, h, d) -> float:
        if L == 0:
            n = N / 1000.0
            f0, f1, f2 = self._full
            return f0 + f1 * n + f2 * n * n / 2.0
        s, l_kt = (N - L) / 1000.0, L / 1000.0
        p0, rh, rd, ps, pp = self._partial
        return p0 + rh * h + rd * d + ps * s + pp * s * (l_kt + s / 2.0)

    def candidate_objectives(self, plan) -> tuple:
        """((L, objective ms), ...) for every feasible candidate, ascending L. The objective omits
        the candidate-independent admission term (module doc)."""
        N, C = plan.prompt_len, self.chunk_size
        out = [(0, self._objective(N, 0, 0, 0))]
        for k, (h, d) in enumerate(_prefix_counts(plan.feasible_tiers), start=1):
            out.append((k * C, self._objective(N, k * C, h, d)))
        return tuple(out)

    def predict_ms(self, prompt_len: int, restore_len: int, host_chunks: int, disk_chunks: int,
                   admitted_chunks: int) -> float:
        """Predicted mean latency of one candidate, including the admission term (for prediction-error
        evaluation). Inconsistent inputs raise ValueError; inputs outside the full declared domain,
        the admitted-chunk range included (the S range applies to L > 0 only), raise OutOfDomainError;
        nothing is extrapolated."""
        C = self.chunk_size
        N = _int(prompt_len, "prompt_len", 1, ValueError)
        L = _int(restore_len, "restore_len", 0, ValueError)
        h = _int(host_chunks, "host_chunks", 0, ValueError)
        d = _int(disk_chunks, "disk_chunks", 0, ValueError)
        m = _int(admitted_chunks, "admitted_chunks", 0, ValueError)
        if L % C or L > max_restore_len(N, C) or h + d != L // C:
            raise ValueError(f"inconsistent candidate N={N} L={L} h={h} d={d} (C={C})")
        if m > N // C - L // C:  # restored chunks are indexed, so they can never be admitted
            raise ValueError(f"admitted_chunks={m} > the {N // C - L // C} complete chunks after L={L} (C={C})")
        v = self._violation(N, L // C, h, d)
        if v is None and not self.admitted_range[0] <= m <= self.admitted_range[1]:
            v = "admitted_chunks"
        if v is not None:
            raise OutOfDomainError(f"N={N} L={L} h={h} d={d} m={m} is outside the calibration domain ({v})")
        return self._objective(N, L, h, d) + self._admit * m


def _reject_constant(name):
    raise ArtifactError(f"non-finite JSON constant {name}")


def load_model(path) -> RestoreCostModel:
    try:
        obj = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise ArtifactError(f"{path}: not JSON: {e}") from None
    return RestoreCostModel(obj)


def save_artifact(artifact: dict, path) -> None:
    """Validate, then write a new file (never overwrites)."""
    RestoreCostModel(artifact)
    with open(path, "x") as f:
        f.write(json.dumps(artifact, indent=1, sort_keys=True, allow_nan=False) + "\n")
