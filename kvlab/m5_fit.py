"""M5b: calibration export, model fitting and frozen predictions (CPU only, no torch).

Only VALIDATED calibration evidence is used (`kvlab.m5_validate.validate` exit 0); the caller passes its parsed
sessions. Then:
1. **Means (`calibration_cases`):** for each case and candidate, the arithmetic mean of the measured (never the
   warmup) FixedChunks latencies over every session, in ms, with the sample count. Each case has every
   feasible candidate, so `fit_restore_cost`'s complete-candidate rule holds.
2. **Fit (`fit_artifact`):** `kvlab.restore_cost.fit_restore_cost` with:
   - the spec's declared weighting (`uniform`: one row per (case, candidate) mean, so a case with k+1
     candidates contributes k+1 rows);
   - the evaluation case ids declared, so they cannot leak in;
   - the namespace, backend and staging group of the sessions;
   - provenance binding the calibration spec, batch plan, batch record, every session file and the source
     identity.
3. **Predictions (`make_predictions`):** for every evaluation case, written before any evaluation sample exists:
   - the CostAware decision on the case's plan (selected length, reason, objectives);
   - `predict_ms` for every candidate, with the case's actual admission count m. A prediction outside the
     declared domain is recorded as unavailable, with the reason; it is never fabricated or extrapolated.

   The file is digested, and its own digest covers the artifact digest and both spec digests.
"""

import hashlib
from pathlib import Path

import numpy as np

from kvlab import m5_spec as sp
from kvlab import restore_cost as rc
from kvlab.boundaries import max_restore_len
from kvlab.restore_policy import CostAware, RestorePlan

PREDICTIONS_FORMAT = "kvlab-m5-predictions-v1"


def measured_latencies_ms(spec: dict, sessions: dict) -> dict:
    """{case_id: {label: [ms, ...]}} of measured samples, session order then chronological."""
    out = {c["case_id"]: {} for c in spec["cases"]}
    for sid in spec["schedule"]["sessions"]:
        for s in sessions[sid]["samples"]:
            if s["phase"] == "measured":
                out[s["case_id"]].setdefault(s["run"], []).append(s["latency_ns"] / 1e6)
    return out


def calibration_cases(spec: dict, sessions: dict) -> list:
    if spec["split"] != "calibration":
        raise ValueError("calibration means come from calibration evidence only")
    lat = measured_latencies_ms(spec, sessions)
    cases = []
    for c in spec["cases"]:
        cands = []
        for L in c["candidates"]:
            xs = lat[c["case_id"]][f"cand:{L}"]
            cands.append({"restore_len": L, "mean_ms": float(np.mean(xs)), "n_samples": len(xs)})
        cases.append({"case_id": c["case_id"], "prompt_len": c["prompt"]["length"], "tiers": list(c["feasible_tiers"]),
                      "admitted_chunks": len(c["admitted"]), "candidates": cands})
    return cases


def evidence_digests(run_dir) -> dict:
    run_dir = Path(run_dir)
    return {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob("*")) if p.is_file()}


def fit_artifact(cal_spec: dict, sessions: dict, eval_spec: dict, *, run_dir) -> dict:
    if eval_spec.get("calibration_spec_sha256") != sp.digest(cal_spec):
        raise ValueError("the evaluation spec is not bound to this calibration spec")
    first = sessions[next(iter(cal_spec["schedule"]["sessions"]))]
    compat = {"namespace_digest": first["namespace"]["digest"], "chunk_size": cal_spec["chunk_size"],
              "restore_backend": cal_spec["restore_backend"],
              "restore_staging_chunks": cal_spec["restore_staging_chunks"]}
    cases = calibration_cases(cal_spec, sessions)
    provenance = {"calibration_spec_sha256": sp.digest(cal_spec), "evaluation_spec_sha256": sp.digest(eval_spec),
                  "calibration_evidence_sha256": evidence_digests(run_dir),
                  "source_git_head": first["source"]["git_head"],
                  "source_files_sha256": sp.digest(first["source"]["files_sha256"]),
                  "means": "arithmetic mean of measured FixedChunks latencies over all sessions (warmups excluded)"}
    return rc.fit_restore_cost(cases, calibration_case_ids=[c["case_id"] for c in cal_spec["cases"]],
                               evaluation_case_ids=[c["case_id"] for c in eval_spec["cases"]],
                               compatibility=compat, weighting=cal_spec["fit"]["weighting"], provenance=provenance)


def case_plan(case: dict, spec: dict, ns_hex: str) -> RestorePlan:
    """The plan the engine will build for the prepared case (staging is always available in these states)."""
    N, C = case["prompt"]["length"], spec["chunk_size"]
    n_elig = max_restore_len(N, C) // C
    indexed = []
    for s in case["chunks"][:n_elig]:
        if s == "absent":
            break
        indexed.append(s)
    return RestorePlan(prompt_len=N, chunk_size=C, eligible_len=max_restore_len(N, C), indexed_tiers=tuple(indexed),
                       feasible_lengths=tuple(case["candidates"]), limit_reason=case["limit_reason"],
                       namespace_digest=ns_hex, restore_backend=spec["restore_backend"],
                       restore_staging_chunks=spec["restore_staging_chunks"] if spec["restore_staging_chunks"] else 4,
                       disk_staging_available=True)


def make_predictions(eval_spec: dict, model: rc.RestoreCostModel, *, created_unix: float) -> dict:
    ns_hex = model.namespace_digest
    policy = CostAware(model)
    cases = {}
    for c in eval_spec["cases"]:
        d = policy.choose(case_plan(c, eval_spec, ns_hex))
        N, m, C = c["prompt"]["length"], len(c["admitted"]), eval_spec["chunk_size"]
        preds = {}
        h = dd = 0
        for k, L in enumerate(c["candidates"]):
            if k:
                t = c["feasible_tiers"][k - 1]
                h, dd = h + (t == "host"), dd + (t == "disk")
            try:
                preds[str(L)] = {"predicted_ms": model.predict_ms(N, L, h, dd, m)}
            except rc.OutOfDomainError as e:
                preds[str(L)] = {"predicted_ms": None, "unavailable": str(e)}
        cases[c["case_id"]] = {"decision": {"selected_len": d.selected_len, "reason": d.reason,
                                            "objective_ms": None if d.objective_ms is None else
                                            [list(x) for x in d.objective_ms]},
                               "predictions": preds, "admitted_chunks": m, "chunk_size": C}
    body = {"format": PREDICTIONS_FORMAT, "artifact_digest": model.digest, "eval_spec_sha256": sp.digest(eval_spec),
            "calibration_spec_sha256": eval_spec["calibration_spec_sha256"], "namespace_digest": ns_hex,
            "created_unix": created_unix, "cases": cases}
    return {**body, "sha256": sp.digest(body)}
