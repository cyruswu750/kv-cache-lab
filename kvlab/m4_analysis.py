"""M4-B analysis of validated evidence (docs/M4B_PROTOCOL.md §7-§8, §12). CPU only; no torch.

Only called after kvlab.m4_validate returned status "valid" with no correctness failure. Results:
- primary: the 15 declared comparisons, session-level (kvlab.m4_stats.compare, Bonferroni over 15);
- secondary (no verdicts): the declared secondary comparisons, steady-state versions, ratios of
  trace means, per-session medians/p90, the M3 stratified block bootstrap of per-request paired
  differences (strata = sessions) as a sensitivity analysis, slow-sample shares/regimes;
- descriptive resources per (trace, configuration): logical copy traffic, persistent and
  transient storage, allocator observations (whole-process; never per configuration), disk and
  page-residency observations, restored/recomputed tokens and tier hits.
Every output carries the assumptions and the claim limits. Without a freeze manifest the result
is marked UNFROZEN and is not evaluation evidence.
"""

import math

import numpy as np

from kvlab import m3_stats
from kvlab import m4_stats as st

CLAIM_LIMITS = [
    "synthetic traces only; batch 1, sequential, synchronous first-token latency (D22), not service TTFT",
    "configurations use different resources: C1 holds persistent GPU KV; C2 holds none on the GPU but a pinned "
    "host slab, a disk tier and an uncounted page cache; equal GPU budgets do not mean equal total resources",
    "every G below the device's free memory is an artificial GPU cap",
    "conditional on an interleaved, co-resident experiment: both C1 caches stay allocated while every "
    "configuration runs, and all share one allocator, page cache and thermal state; not standalone deployment",
    "allocator peaks are whole-process figures, never per-configuration GPU peaks",
    "page residency is a mincore point observation before each C2 request (warmed if below 1.0), not a budget; "
    "warm page cache only, no cold-NVMe claim (D26)",
    "C1 is this project's GPU-only baseline (prefix-aware LRU), not vLLM or SGLang; C1-LRU is offline only",
    "no p99/tail, throughput, concurrency or production claims; this machine, model and stack only",
]
WHOLE_TIER_LIMIT = CLAIM_LIMITS[5]
REQUEST_SCOPED_LIMIT = (
    "warm disk (M4_EVAL_V2): each C2 request's candidate files (the disk files its restore reads on the normal "
    "path) were observed fully resident by mincore in one verification pass immediately before the request, "
    "after at most one prewarm read per file; these are per-file point observations, not a simultaneous snapshot, "
    "and not a guarantee at timer start or during the timed read (pages are not pinned); warm page cache only, no "
    "cold-NVMe, storage-device or production claim (D26); not procedurally comparable with M4_EVAL_V1's "
    "whole-tier preparation, and not a correction of V1")


def claim_limits(spec: dict) -> list:
    """CLAIM_LIMITS, with the warm-disk statement of the spec's rule."""
    if spec.get("warm_disk") == "request_candidates":
        return [REQUEST_SCOPED_LIMIT if x == WHOLE_TIER_LIMIT else x for x in CLAIM_LIMITS]
    return list(CLAIM_LIMITS)


def _series(sessions: dict, C: int) -> dict:
    """{(trace, config): {session: [latency_ms by request index]}} and the records by key."""
    lat, recs = {}, {}
    for sid, r in sessions.items():
        for blk in r["evaluation"]:
            for rec in blk["records"]:
                key = (blk["name"], rec["config"])
                lat.setdefault(key, {}).setdefault(sid, {})[rec["index"]] = rec["latency_ns"] / 1e6
                n = rec["request_kv_bytes"] * C // r["payload_bytes"]  # validated: N x per-token bytes
                recs.setdefault(key, []).append({**rec, "_sid": sid, "_n_tokens": n})
    return {k: {s: [v[i] for i in sorted(v)] for s, v in d.items()} for k, d in lat.items()}, recs


def _cmp(lat, sessions, t, a, b, n_family, alpha, primary, steady_frac=None):
    """One comparison. With `steady_frac`, the first ceil(n x frac) requests are excluded from the
    differences, the means and the ratio alike."""
    lo = math.ceil(len(lat[(t, a)][sessions[0]]) * steady_frac) if steady_frac else 0
    xa = {sid: lat[(t, a)][sid][lo:] for sid in sessions}
    xb = {sid: lat[(t, b)][sid][lo:] for sid in sessions}
    d = [float(np.mean(xa[sid])) - float(np.mean(xb[sid])) for sid in sessions]
    out = st.compare(d, n_family, alpha, primary)
    ma = float(np.mean([np.mean(xa[s]) for s in sessions]))
    mb = float(np.mean([np.mean(xb[s]) for s in sessions]))
    out.update({"trace": t, "A": a, "B": b, "excluded_leading_requests": lo, "mean_trace_mean_A_ms": ma,
                "mean_trace_mean_B_ms": mb, "ratio_of_trace_means_A_over_B": ma / mb})
    return out


def _bootstrap(lat, sessions, t, a, b):
    sa = [lat[(t, b)][s] for s in sessions]  # m3_stats: d = second - first, so first = B, second = A
    sb = [lat[(t, a)][s] for s in sessions]
    base = sum(map(ord, f"{t}:{a}:{b}"))  # deterministic, as M3's verdict seeds
    return {str(bl): m3_stats.boot_ci(sa, sb, "mean_d", bl, seed=base + 7919 * bl)["ci"]
            for bl in m3_stats.BLOCK_LENGTHS}


def _resources(recs) -> dict:
    def tot(k):
        return sum(r["traffic"][k] for r in recs)

    def mx(f):
        vals = [f(r) for r in recs]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    out = {"logical_copy_traffic_bytes_total": {k: tot(k) for k in recs[0]["traffic"]},
           "restored_tokens": sum(r["restore_len"] for r in recs),
           "recomputed_tokens": sum(r["_n_tokens"] - r["restore_len"] for r in recs),
           "requests": len(recs), "tier_hits": {s: sum(r["sources"].count(s) for r in recs) for s in ("gpu", "host", "disk")},
           "request_kv_bytes_max": mx(lambda r: r["request_kv_bytes"]),
           "transient_restore_storage_bytes_max": mx(lambda r: r["restore_scratch_bytes"])}
    c1 = [r["c1"] for r in recs if r.get("c1")]
    if c1:
        out["persistent_gpu_kv_bytes_max"] = max(x["retained_bytes"] for x in c1)
        out["persistent_gpu_kv_budget_bytes"] = c1[0]["budget_bytes"]
    c2 = [r["c2"] for r in recs if r.get("c2")]
    if c2:
        lrs = [x["last_restore"] for x in c2 if x.get("last_restore")]
        out.update(device_destination_bytes_max=max((x["device_destination_bytes"] for x in lrs), default=0),
                   batched_staging_bytes_max=max((x["device_scratch_bytes"] for x in lrs), default=0))
        out.update(pinned_host_slab_bytes=c2[0]["host_slab_bytes"], host_registered_bytes=c2[0]["host_registered_bytes"],
                   host_logical_occupancy_max=max(x["host_resident_logical_bytes"] for x in c2),
                   disk_peak_used_bytes=max(x["disk_peak_used_bytes"] for x in c2),
                   disk_evictions=c2[-1]["disk_evictions"], host_spills=c2[-1]["counters"]["host_spills_to_disk"],
                   page_residency_prewarmed_files=sum(r["residency"]["prewarmed_files"] for r in recs))
    mem = [r["memory"] for r in recs if r.get("memory")]
    if mem:
        out["process_allocator"] = {
            "note": "whole-process figures; include every co-resident configuration; not a per-configuration peak",
            "P_max_allocated_max": max(m["P_max_allocated"] for m in mem),
            "incremental_peak_above_A0_max": max(m["P_max_allocated"] - m["A0_allocated"] for m in mem),
            "incremental_peak_definition": "P - A0: everything allocated during the request, including persistent "
                                           "C1 growth from admission inside the timed region"}
    return out


def _slow(lat, sessions, recs_by_key) -> dict:
    out = {}
    for key, per in lat.items():
        n = len(next(iter(per.values())))
        mins = [min(per[s][i] for s in sessions) for i in range(n)]
        flags = {s: [per[s][i] > m3_stats.SLOW_FACTOR * mins[i] for i in range(n)] for s in sessions}
        walls = {}
        for r in recs_by_key[key]:
            walls.setdefault(r["_sid"], {})[r["index"]] = r["t_wall"]
        regimes = [x for s in sessions for x in m3_stats.regimes([walls[s][i] for i in range(n)], flags[s])]
        out[f"{key[0]} {key[1]}"] = {"slow_share": float(np.mean([f for s in sessions for f in flags[s]])),
                                     "longest_regime_s": max(regimes, default=0.0)}
    return out


def analyze(validation: dict, spec: dict) -> dict:
    if validation["status"] != "valid" or validation["correctness_failures"]:
        raise ValueError("the analysis runs only on valid evidence without correctness failures")
    sessions = validation["sessions"]
    sids = sorted(sessions)
    lat, recs = _series(sessions, spec["chunk_size"])
    fam, alpha = [tuple(x) for x in spec["family"]], spec["alpha"]
    primary = [_cmp(lat, sids, t, a, b, len(fam), alpha, True) for t, a, b in fam]
    for p in primary:
        p["block_bootstrap_mean_d_ci95_by_block_len"] = _bootstrap(lat, sids, p["trace"], p["A"], p["B"])
    secondary = [_cmp(lat, sids, t, a, b, len(fam), alpha, False) for t, a, b in [tuple(x) for x in spec["secondary"]]]
    steady = [_cmp(lat, sids, t, a, b, len(fam), alpha, False, spec["steady_fraction"]) for t, a, b in fam]
    steady_secondary = [_cmp(lat, sids, t, a, b, len(fam), alpha, False, spec["steady_fraction"])
                        for t, a, b in [tuple(x) for x in spec["secondary"]]]
    per_session = {f"{t} {c}": {s: {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)),
                                    "mean_ms": float(np.mean(v))} for s, v in per.items()} for (t, c), per in lat.items()}
    across = {}
    for key, per in per_session.items():  # the plan's convention: per-session statistic, then median and range
        across[key] = {stat: {"median": float(np.median([v[stat] for v in per.values()])),
                              "range": [float(min(v[stat] for v in per.values())),
                                        float(max(v[stat] for v in per.values()))]}
                       for stat in ("median_ms", "p90_ms", "mean_ms")}
    return {"spec": spec["name"], "spec_sha256": spec["spec_sha256"], "frozen": validation["frozen"],
            "evidence_label": "evaluation evidence" if validation["frozen"] else
            "UNFROZEN: not evaluation evidence; no performance claim may be made from it",
            "assumptions": st.ASSUMPTIONS, "claim_limits": claim_limits(spec), "n_sessions": len(sids),
            "primary_family_size": len(fam), "alpha": alpha, "primary": primary, "secondary": secondary,
            "steady_state": steady, "steady_state_secondary": steady_secondary, "per_session": per_session,
            "across_sessions": across,
            "resources": {f"{t} {c}": _resources(v) for (t, c), v in recs.items()},
            "slow_samples": _slow(lat, sids, recs),
            "not_reported": ["p99/tail latency", "throughput", "service TTFT"]}


def markdown(a: dict) -> str:
    lines = [f"# {a['spec']} analysis ({a['evidence_label']})", "", f"Assumptions: {a['assumptions']}", "",
             f"Primary family: {a['primary_family_size']} comparisons; verdicts need unanimity across "
             f"{a['n_sessions']} sessions and a Bonferroni-level t-interval excluding 0.", "",
             "| trace | A | B | mean D (ms) | 95% t-interval | Bonferroni interval | unanimous | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for p in a["primary"]:
        u = "neg" if p["unanimous_negative"] else "pos" if p["unanimous_positive"] else "no"
        lines.append(f"| {p['trace']} | {p['A']} | {p['B']} | {p['mean_ms']:.3f} | "
                     f"[{p['ci95_ms'][0]:.3f}, {p['ci95_ms'][1]:.3f}] | "
                     f"[{p['ci_bonferroni_ms'][0]:.3f}, {p['ci_bonferroni_ms'][1]:.3f}] | {u} | {p['verdict']} |")
    lines += ["", "Claim limits:"] + [f"- {c}" for c in a["claim_limits"]]
    return "\n".join(lines) + "\n"
