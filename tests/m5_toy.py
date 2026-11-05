"""Synthetic M5a test helpers (imported by tests/test_restore_*.py). Nothing here is measured:
coefficients and latencies are invented to exercise code paths and hand-computed choices."""

import copy

from kvlab import restore_cost as rc
from kvlab.restore_policy import RestorePlan

NS_HEX = "ab" * 32


def plan(prompt_len, tiers, C=256, *, feasible=None, backend="baseline", group=4, ns=NS_HEX, limit=None,
         staging=True):
    """A RestorePlan built by hand (no store): `tiers` are the indexed prefix; all feasible by default."""
    from kvlab.boundaries import max_restore_len
    k = len(tiers) if feasible is None else feasible
    eligible = max_restore_len(prompt_len, C)
    assert len(tiers) * C <= eligible
    return RestorePlan(prompt_len=prompt_len, chunk_size=C, eligible_len=eligible, indexed_tiers=tuple(tiers),
                       feasible_lengths=tuple(range(0, k * C + 1, C)), limit_reason=limit or "eligible_limit",
                       namespace_digest=ns, restore_backend=backend, restore_staging_chunks=group,
                       disk_staging_available=staging)


def coefficients(full=(10.0, 40.0, 8.0), partial=(15.0, 1.0, 20.0, 40.0, 8.0), admit=4.5) -> dict:
    return {"full": dict(zip(rc.FEATURES["full"], map(float, full))),
            "partial": dict(zip(rc.FEATURES["partial"], map(float, partial))),
            "shared": {"admitted_chunks": float(admit)}}


def domain(prompt=(1, 100_000), suffix=(1, 100_000), kmax=64, hmax=64, dmax=64, admitted=(0, 64)) -> dict:
    return {"prompt_len": list(prompt), "suffix_len": list(suffix), "restore_chunks_max": kmax,
            "host_chunks_max": hmax, "disk_chunks_max": dmax, "admitted_chunks": list(admitted)}


def compat(ns=NS_HEX, C=256, backend="baseline", group=None) -> dict:
    return {"namespace_digest": ns, "chunk_size": C, "restore_backend": backend,
            "restore_staging_chunks": group if backend == "batched" else None}


def artifact(coef=None, dom=None, comp=None) -> dict:
    """A hand-built, digested artifact with synthetic coefficients (test use only; not calibrated)."""
    art = {"schema": rc.SCHEMA, "units": dict(rc.UNITS), "target": rc.TARGET, "objective": rc.OBJECTIVE,
           "features": {g: list(v) for g, v in rc.FEATURES.items()},
           "compatibility": comp or compat(), "coefficients_ms": coef or coefficients(),
           "fit": {"method": rc.FIT_METHOD, "weighting": "uniform", "n_cases": 1, "n_rows": 9, "rank": 9,
                   "condition_number": 1.0, "max_abs_residual_ms": 0.0, "max_abs_relative_residual": 0.0},
           "domain": dom or domain(),
           "calibration": {"case_ids": ["synthetic-1"], "evaluation_case_ids": [], "cases_sha256": "0" * 64},
           "provenance": {"synthetic": "hand-built test artifact; not a calibration"}}
    art["digest"] = rc.artifact_digest(art)
    return art


def redigest(art: dict) -> dict:
    art = copy.deepcopy(art)
    art["digest"] = rc.artifact_digest(art)
    return art


def truth_ms(coef, N, L, h, d, m, C=256) -> float:
    """The model equation evaluated directly (independent of RestoreCostModel)."""
    f, p, a = coef["full"], coef["partial"], coef["shared"]["admitted_chunks"]
    if L == 0:
        n = N / 1000
        return f["const"] + f["n_kt"] * n + f["n_pairs_kt2"] * n * n / 2 + a * m
    s, l_kt = (N - L) / 1000, L / 1000
    return (p["const"] + p["host_chunks"] * h + p["disk_chunks"] * d + p["s_kt"] * s
            + p["s_pairs_kt2"] * s * (l_kt + s / 2) + a * m)


SPECS = (  # (case_id, N, prepared tiers, admitted chunks); C = 256
    ("c01", 300, ["host"], 0),
    ("c02", 700, ["disk", "disk"], 0),
    ("c03", 1100, ["host", "disk", "host"], 1),
    ("c04", 1500, ["disk", "host"], 3),
    ("c05", 2100, ["host", "host", "disk", "disk"], 2),
    ("c06", 900, [], 1),
    ("c07", 1800, ["disk"] * 5, 1),
)


def synthetic_cases(coef, specs=SPECS, C=256, n_samples=20) -> list:
    out = []
    for cid, N, tiers, m in specs:
        cands, h, d = [{"restore_len": 0, "mean_ms": truth_ms(coef, N, 0, 0, 0, m, C), "n_samples": n_samples}], 0, 0
        for k, t in enumerate(tiers, start=1):
            h, d = h + (t == "host"), d + (t == "disk")
            cands.append({"restore_len": k * C, "mean_ms": truth_ms(coef, N, k * C, h, d, m, C),
                          "n_samples": n_samples})
        out.append({"case_id": cid, "prompt_len": N, "tiers": list(tiers), "admitted_chunks": m,
                    "candidates": cands})
    return out
