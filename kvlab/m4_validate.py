"""Fail-closed validation of M4-B evidence (docs/M4B_PROTOCOL.md §8, §10). CPU only; no torch.

Nothing recorded by the runner is trusted as a verdict. The validator regenerates the traces,
budgets, trace order, per-request configuration order and every key-level decision from the
spec (kvlab.m4_spec, kvlab.trace_models), and compares each record with them. It recomputes the
strict comparison from the recorded SHA-256 digests and tokens, not from a flag.

Outcomes (`status`, and the analysis exit code):
- invalid (exit 3): any problem with completeness, structure, identity, deadline/stopping, the
  freeze binding, decisions, budgets, an unintended fault (ENOSPC or any other spill, admission,
  read, unlink or temp failure; any miss reason other than "absent"), the warm-disk rule, forward
  calls, or warmup coverage. No statistic is computed;
- valid with correctness failures (exit 4): complete, consistent evidence in which a strict
  comparison failed. No statistic is computed. Correctness failures are listed in every outcome;
- valid (exit 0): the statistics may be computed. Without a freeze manifest the outcome is
  additionally marked `unfrozen`, and the analysis states that it is not evaluation evidence.

The warm-disk rule depends on the spec (kvlab.m4_spec.warm_disk_protocol):
- "whole_tier" (M4_EVAL_V1): `resident_fraction_before` is 1.0 for every C2 request;
- "request_candidates" (M4_EVAL_V2; docs/M4_EVAL_V2_PROTOCOL.md). Every C2 record must satisfy all of:
  - its candidate files equal, in order, the disk chunks of the restore that the key-level model
    (kvlab.trace_models.tiered_model) predicts, named by the real chain keys of the session's namespace
    (`<key hex>.kvc`), with the chunk indices and the plan counts;
  - every candidate was observed in the initial pass, read exactly once (all of its bytes) if and only if
    it was below full residency there, and observed fully resident in the final pass, with the file size
    and page count of a chunk file;
  - the phase timestamps are in the declared order (plan, initial pass, prewarms, final pass);
  - the recorded outcome equals the one recomputed here ("verified", or "empty" with no candidates).

  The batch interference record (batch_interference.json) must name this spec and the drop_pagecache
  counter, carry finite ordered wall-clock read times that bracket the batch (start at or before the first
  launch and every session start; end at or after every session end), and show the drop_pagecache count read
  before the first launch and after the batch, unchanged. Every drop_pagecache value recorded in the
  sessions (session start and end, before and after each C2 request) must have been read and must equal
  the batch start value. A missing or unreadable count, an increase, or a decrease (a reset) is invalid.
  Other kernel counters are context only and are not checked.
"""

import json
from pathlib import Path

from kvlab import m3_confirm as mc
from kvlab import m4_spec as ms
from kvlab import trace_models as tm
from kvlab.boundaries import max_restore_len
from kvlab.identity import chain_keys

BATCH_RECORD = mc.BATCH_RECORD
HEX64 = set("0123456789abcdef")
FAULT_FREE_DISK = ("disk_tmp_bytes", "disk_undeletable_bytes", "disk_stale_removed")
TRAFFIC_KEYS = ("d2d_gather", "d2d_constructor", "d2d_admission", "h2d", "d2d_placement", "d2h_admission",
                "disk_read", "disk_written")
# M4_EVAL_V2 records (written by kvlab/warm_disk.py; the names are restated here, not imported)
REQUEST_SCOPED = "request_candidates"
INTERFERENCE_RECORD = "batch_interference.json"
INTERFERENCE_FORMAT = "m4-batch-interference-v1"
WD_STOPS = ("all_eligible_present", "first_absent", "no_staging_buffer")
WD_OUTCOMES = ("empty", "verified", "not_resident", "error")
WD_OBS = ("observed", "error")
WD_PREWARM = ("not_needed", "read", "error", "skipped")


def _hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and set(x) <= HEX64


def _int(x) -> bool:
    return type(x) is int


def expected_files(spec: dict) -> set:
    out = {BATCH_RECORD}
    if ms.warm_disk_protocol(spec) == REQUEST_SCOPED:
        out.add(INTERFERENCE_RECORD)
    for sid in spec["sessions"]:
        out |= {f"session_{sid}.json", f"session_{sid}.gpu_monitor.csv", f"logs/session_{sid}.log"}
    return out


class _Ctx:
    def __init__(self):
        self.problems, self.failures = [], []
        self.drop_obs = []  # (where, recorded counter snapshot): M4_EVAL_V2's drop_pagecache observations

    def p(self, msg):
        self.problems.append(msg)


C2_COUNTERS = ("requests", "restored_requests", "full_recompute_requests", "host_hits", "disk_hits", "chunk_misses",
               "recovered_prefix_tokens", "chunks_stored", "host_spills_to_disk", "bytes_d2h", "bytes_h2d",
               "disk_file_bytes_read", "disk_file_bytes_written")
C2_DICTS = ("miss_reasons", "write_skips", "spill_failures")
C2_STATE_INTS = ("host_evictions", "host_entries", "host_slab_bytes", "host_registered_bytes",
                 "host_resident_logical_bytes", "disk_entries", "disk_evictions", "disk_used_bytes",
                 "disk_peak_used_bytes", "disk_tmp_bytes", "disk_undeletable_bytes", "disk_stale_removed")
C1_STATE_INTS = ("entries", "retained_bytes", "peak_retained_bytes", "budget_bytes", "capacity_chunks")
BIG = 1 << 62  # any recorded size or count must be a non-negative int below this


def _nn(x) -> bool:
    """A non-negative JSON integer of a sane size (not a bool, not beyond the float range)."""
    return type(x) is int and 0 <= x < BIG


def _int_map(d, keys=None) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) and _nn(v) for k, v in d.items()) and \
        (keys is None or set(d) == set(keys))


def _wd_obs_ok(o) -> bool:
    if not (isinstance(o, dict) and o.get("status") in WD_OBS and _nn(o.get("t_ns"))):
        return False
    if o["status"] == "observed":
        return _nn(o.get("resident_pages")) and _nn(o.get("pages")) and _nn(o.get("size")) \
            and o["resident_pages"] <= o["pages"]
    return isinstance(o.get("error"), str)


def _wd_prewarm_ok(w) -> bool:
    if not (isinstance(w, dict) and w.get("status") in WD_PREWARM):
        return False
    if w["status"] in ("read", "error"):
        if not (_nn(w.get("t_start_ns")) and _nn(w.get("t_end_ns")) and w["t_start_ns"] <= w["t_end_ns"]):
            return False
    return _nn(w.get("bytes_read")) if w["status"] == "read" else True


def _wd_schema(res) -> list:
    """Types of an M4_EVAL_V2 warm-disk record (kvlab/warm_disk.py prepare, plus the runner's two fields)."""
    if not isinstance(res, dict) or res.get("protocol") != REQUEST_SCOPED:
        return [f"warm-disk record missing or not {REQUEST_SCOPED!r}"]
    out = []
    ps = res.get("page_size")
    if not (_nn(ps) and ps > 0 and ps & (ps - 1) == 0):
        out.append("page_size")
    plan = res.get("plan")
    if not (isinstance(plan, dict) and all(_nn(plan.get(k)) for k in ("eligible_chunks", "present_chunks",
                                                                       "host_chunks"))
            and plan.get("stop") in WD_STOPS):
        out.append("plan")
    cand, files = res.get("candidates"), res.get("files")
    if not (isinstance(cand, list) and all(isinstance(x, str) for x in cand)):
        out.append("candidates")
    if not (isinstance(files, list) and all(
            isinstance(f, dict) and _nn(f.get("index")) and isinstance(f.get("file"), str)
            and _wd_obs_ok(f.get("initial")) and _wd_prewarm_ok(f.get("prewarm")) and _wd_obs_ok(f.get("final"))
            for f in files)):
        out.append("files")
    ph = res.get("phases")
    if not (isinstance(ph, dict) and set(ph) == {"initial", "prewarm", "final"}
            and all(isinstance(v, str) for v in ph.values())):
        out.append("phases")
    if res.get("outcome") not in WD_OUTCOMES:
        out.append("outcome")
    for k in ("prewarmed_files", "tier_entries", "t_start_ns", "t_planned_ns", "t_verified_ns",
              "timer_start_minus_verified_ns"):
        if not _nn(res.get(k)):
            out.append(k)
    for k in ("counters_before", "counters_after_request"):
        if not isinstance(res.get(k), dict):
            out.append(k)
    return out


def _schema(rec, cname, evaluation, cuda, protocol="whole_tier") -> list:
    """Types and ranges of one record, checked before any semantic use. Malformed evidence is
    reported as a problem (exit 3), never as a correctness failure or a crash."""
    out = []
    need = {"index": _nn, "request_id": lambda x: isinstance(x, str), "config": lambda x: x == cname, "pos": _nn,
            "t_wall": mc.finite, "latency_ns": lambda x: _nn(x) and x > 0 and mc.finite(x), "token": _nn,
            "logits_sha256": _hex64, "restore_len": _nn,
            "sources": lambda x: isinstance(x, list) and all(v in ("gpu", "host", "disk") for v in x),
            "stored": lambda x: isinstance(x, list) and all(_nn(v) for v in x), "forward_calls": _nn,
            "request_kv_bytes": _nn, "restore_scratch_bytes": _nn,
            "traffic": lambda x: _int_map(x, TRAFFIC_KEYS),
            "c1_retained_before": lambda x: _int_map(x, ("C1PA_W", "C1PA_HALF")),
            "c1_retained_after": lambda x: _int_map(x, ("C1PA_W", "C1PA_HALF"))}
    if evaluation:
        need.update(ref_logits_sha256=_hex64, ref_token=_nn)
    for k, ok in need.items():
        if k not in rec or not ok(rec[k]):
            out.append(f"field {k} missing or malformed ({type(rec.get(k)).__name__})")
    mem = rec.get("memory")
    if cuda:
        if not (isinstance(mem, dict) and _nn(mem.get("A0_allocated")) and _nn(mem.get("P_max_allocated"))
                and mem["P_max_allocated"] >= mem["A0_allocated"]
                and _int_map(mem.get("persistent_c1_retained_at_A0"), ("C1PA_W", "C1PA_HALF"))):
            out.append(f"allocator record malformed {mem!r}")
    elif mem is not None:
        out.append("memory record present for a CPU run")
    res = rec.get("residency")
    if cname.startswith("C2"):
        if protocol == REQUEST_SCOPED:
            bad = _wd_schema(res)
            if bad:
                out.append(f"warm-disk record malformed: {bad}")
        elif not (isinstance(res, dict) and _nn(res.get("files")) and _nn(res.get("prewarmed_files"))
                  and all(mc.finite(res.get(k)) and 0.0 <= res[k] <= 1.0
                          for k in ("min_fraction_before", "resident_fraction_before"))):
            out.append(f"residency record malformed {res!r}")
        st = rec.get("c2")
        c = st.get("counters") if isinstance(st, dict) else None
        if not (isinstance(c, dict) and all(_nn(c.get(k)) for k in C2_COUNTERS)
                and all(_int_map(c.get(k)) for k in C2_DICTS)
                and all(_nn(st.get(k)) for k in C2_STATE_INTS) and _int_map(st.get("disk_unlink_failures"))
                and "last_restore" in st and (st["last_restore"] is None or isinstance(st["last_restore"], dict))):
            out.append("C2 state malformed")
    else:
        if res is not None:
            out.append("residency recorded for a configuration without a disk tier")
    if cname.startswith("C1"):
        st = rec.get("c1")
        if not (isinstance(st, dict) and _int_map(st.get("counters")) and all(_nn(st.get(k)) for k in C1_STATE_INTS)):
            out.append("C1 state malformed")
    return out


def _models(spec, trace, b):
    C = spec["chunk_size"]
    prompts = [list(r.tokens) for r in trace.requests]
    return {"C1PA_W": tm.pa_model(prompts, C, b["C1PA_W"]), "C1PA_HALF": tm.pa_model(prompts, C, b["C1PA_HALF"]),
            "C2": tm.tiered_model(prompts, C, b["host_slots"], b["disk_files"]), "prompts": prompts}


def _check_record(cx, where, rec, cname, req, model, cum, b, spec, payload, file_bytes, cuda, evaluation):
    C = spec["chunk_size"]
    N = len(req.tokens)
    if rec["request_id"] != req.id:
        cx.p(f"{where}: request id {rec['request_id']!r} != trace {req.id!r}")
    if rec["forward_calls"] != spec["steps"]:
        cx.p(f"{where}: {rec['forward_calls']!r} forward calls, expected exactly {spec['steps']} (one prefill)")
    if rec["request_kv_bytes"] != N * (payload // C):
        cx.p(f"{where}: request KV {rec['request_kv_bytes']!r} B != N x per-token {N * (payload // C)}")
    L = rec["restore_len"]
    if not _int(L) or L % C or L > max_restore_len(N, C):
        cx.p(f"{where}: restore_len {L!r} breaks the first-token rule or the chunk grid (N = {N})")
        return
    if cuda and rec["memory"]["persistent_c1_retained_at_A0"] != rec["c1_retained_before"]:
        cx.p(f"{where}: persistent occupancy at A0 differs from the C1 retained bytes before the request")
    nr, tr_ = L // C, rec["traffic"]
    if cname == "C0":
        if L or rec["sources"] or rec["stored"]:
            cx.p(f"{where}: C0 restored or stored something")
        want_tr = dict.fromkeys(TRAFFIC_KEYS, 0)
    elif cname.startswith("C1"):
        m = model[cname]
        if (L, rec["stored"]) != (m["restore_len"], m["stored"]) or rec["sources"] != ["gpu"] * nr:
            cx.p(f"{where}: C1-PA decision L={L} stored={rec['stored']} != key-level model "
                 f"L={m['restore_len']} stored={m['stored']}")
        cum[cname]["hits"] += nr
        cum[cname]["stored_n"] += len(m["stored"])
        cum[cname]["req"] += 1
        cum[cname]["restored"] += bool(nr)
        cum[cname]["tokens"] += L
        cum[cname]["misses"] += nr < max_restore_len(N, C) // C
        cum[cname]["ev"] += m["evictions"]
        cum[cname]["stops"] += m["stopped"]
        cum[cname]["refused"] += m["refused"]
        cum[cname]["entries"] += len(m["stored"]) - m["evictions"]
        st = rec.get("c1") or {}
        c = st.get("counters", {})
        got = (c.get("evictions", 0), c.get("admission_stops_prefix_capacity", 0),
               c.get("chunks_refused_prefix_capacity", 0), c.get("write_skips_no_capacity", 0))
        want = (cum[cname]["ev"], cum[cname]["stops"], cum[cname]["refused"], 0)
        if got != want:
            cx.p(f"{where}: C1 counters (evictions, stops, refused, no-capacity skips) {got} != model {want}")
        cap = b[cname]
        got1 = tuple(c.get(x, 0) for x in ("chunk_hits", "chunks_stored", "lookups", "restored_requests",
                                            "recovered_prefix_tokens", "chunk_misses"))
        want1 = tuple(cum[cname][x] for x in ("hits", "stored_n", "req", "restored", "tokens", "misses"))
        if got1 != want1:
            cx.p(f"{where}: C1 cumulative (hits, stored, lookups, restored requests, tokens, misses) {got1} != "
                 f"model {want1}")
        if st["entries"] != cum[cname]["entries"] or st["retained_bytes"] != st["entries"] * payload \
                or st["budget_bytes"] != cap * payload or st["capacity_chunks"] != cap \
                or st["retained_bytes"] > cap * payload or st["peak_retained_bytes"] < st["retained_bytes"] \
                or st["peak_retained_bytes"] > cap * payload \
                or rec["c1_retained_after"][cname] != st["retained_bytes"]:
            cx.p(f"{where}: C1 storage {st.get('entries')!r} entries / {st.get('retained_bytes')!r} B; model "
                 f"{cum[cname]['entries']} entries, budget {cap * payload} B")
        growth = rec["c1_retained_after"].get(cname, 0) - rec["c1_retained_before"].get(cname, 0)
        if growth != (len(m["stored"]) - m["evictions"]) * payload:
            cx.p(f"{where}: persistent C1 growth {growth} B != (stored - evicted) x payload")
        want_tr = {"d2d_gather": nr * payload, "d2d_constructor": nr * payload,
                   "d2d_admission": len(m["stored"]) * payload, "h2d": 0, "d2d_placement": 0, "d2h_admission": 0,
                   "disk_read": 0, "disk_written": 0}
    else:
        m = model["C2"]
        if (L, rec["sources"], rec["stored"]) != (m["restore_len"], m["sources"], m["stored"]):
            cx.p(f"{where}: C2 decision L={L} sources={rec['sources']} stored={rec['stored']} != key-level model "
                 f"L={m['restore_len']} sources={m['sources']} stored={m['stored']}")
        cum[cname]["spills"] += m["spills"]
        cum[cname]["dev"] += m["disk_evictions"]
        n_disk = m["sources"].count("disk")
        k = cum[cname]
        k["req"] += 1
        k["restored"] += bool(nr)
        k["host"] += m["sources"].count("host")
        k["disk"] += n_disk
        k["stored_n"] += len(m["stored"])
        k["h2d"] += nr * payload
        k["d2h"] += len(m["stored"]) * payload
        k["read"] += n_disk * file_bytes
        k["written"] += m["spills"] * file_bytes
        k["tokens"] += L
        k["misses"] += nr < max_restore_len(N, C) // C
        st = rec.get("c2") or {}
        c = st.get("counters") or {}
        faults = {"write_skips": c.get("write_skips"), "spill_failures": c.get("spill_failures"),
                  "miss_reasons_other_than_absent": {k: v for k, v in (c.get("miss_reasons") or {}).items()
                                                     if k != "absent"},
                  "disk_unlink_failures": st.get("disk_unlink_failures"),
                  **{k: st.get(k) for k in FAULT_FREE_DISK}}
        if any(v not in ({}, 0) for v in faults.values()):
            cx.p(f"{where}: unintended cache/infrastructure fault {faults}")
        got = (c.get("host_spills_to_disk"), st.get("disk_evictions"), st.get("host_entries"), st.get("disk_entries"))
        want = (cum[cname]["spills"], cum[cname]["dev"], m["host_entries"], m["disk_entries"])
        if got != want:
            cx.p(f"{where}: C2 placement (spills, disk evictions, host entries, disk entries) {got} != model {want}")
        got_c = tuple(c[x] for x in ("requests", "restored_requests", "host_hits", "disk_hits", "chunks_stored",
                                     "bytes_h2d", "bytes_d2h", "disk_file_bytes_read", "disk_file_bytes_written",
                                     "recovered_prefix_tokens"))
        want_c = (k["req"], k["restored"], k["host"], k["disk"], k["stored_n"], k["h2d"], k["d2h"], k["read"],
                  k["written"], k["tokens"])
        if got_c != want_c:
            cx.p(f"{where}: C2 cumulative counters {got_c} != recomputed {want_c}")
        got_m = (c["full_recompute_requests"], c["chunk_misses"], c["miss_reasons"], st["host_evictions"])
        want_m = (k["req"] - k["restored"], k["misses"], {"absent": k["misses"]} if k["misses"] else {}, k["spills"])
        if got_m != want_m:
            cx.p(f"{where}: C2 (full recomputes, misses, miss reasons, host evictions) {got_m} != recomputed {want_m}")
        slab = (b["host_slots"] + 1) * payload
        if st["host_resident_logical_bytes"] != st["host_entries"] * payload or st["host_slab_bytes"] != slab \
                or st["host_registered_bytes"] != (slab if cuda else 0) \
                or st["disk_used_bytes"] != st["disk_entries"] * file_bytes \
                or not st["disk_used_bytes"] <= st["disk_peak_used_bytes"] <= b["disk_files"] * file_bytes:
            cx.p(f"{where}: C2 storage terms inconsistent (host {st['host_resident_logical_bytes']} / slab "
                 f"{st['host_slab_bytes']} / registered {st['host_registered_bytes']}; disk {st['disk_used_bytes']} / "
                 f"peak {st['disk_peak_used_bytes']}; budget {b['disk_files'] * file_bytes})")
        lr = st["last_restore"]
        backend = "batched" if cname == "C2_batch" else "baseline"
        if nr:
            g = min(nr, spec["staging_chunks"]) * payload if backend == "batched" else 0
            want_lr = {"backend": backend, "chunks_planned": nr, "chunks_loaded": nr, "h2d_bytes": nr * payload,
                       "device_destination_bytes": nr * payload, "device_scratch_bytes": g}
            if not isinstance(lr, dict) or any(lr.get(x) != v for x, v in want_lr.items()) \
                    or rec["restore_scratch_bytes"] != nr * payload + g:
                cx.p(f"{where}: restore bookkeeping {lr!r} / scratch {rec['restore_scratch_bytes']} != expected "
                     f"{want_lr} / {nr * payload + g}")
        elif lr is not None or rec["restore_scratch_bytes"] != 0:
            cx.p(f"{where}: a request without a restore recorded restore storage")
        res = rec["residency"]
        if ms.warm_disk_protocol(spec) == REQUEST_SCOPED:
            _check_warm_disk(cx, where, res, model["C2_candidates"], m, N, C, file_bytes)
        elif not isinstance(res, dict) or res.get("resident_fraction_before") != 1.0:
            cx.p(f"{where}: page residency before the request {res!r}, required 1.0")
        want_tr = {"d2d_gather": 0, "d2d_constructor": nr * payload, "d2d_admission": 0, "h2d": nr * payload,
                   "d2d_placement": nr * payload if cname == "C2_batch" else 0,
                   "d2h_admission": len(m["stored"]) * payload, "disk_read": n_disk * file_bytes,
                   "disk_written": m["spills"] * file_bytes}
    if not cname.startswith("C2") and cname != "C0" and rec["restore_scratch_bytes"] != nr * payload:
        cx.p(f"{where}: C1 gather scratch {rec['restore_scratch_bytes']} != L x per-token bytes {nr * payload}")
    if cname == "C0" and rec["restore_scratch_bytes"] != 0:
        cx.p(f"{where}: C0 recorded restore storage")
    if tr_ != want_tr:
        cx.p(f"{where}: copy traffic {tr_!r} != recomputed {want_tr!r}")
    if evaluation:
        if rec["ref_logits_sha256"] != rec["logits_sha256"] or rec["ref_token"] != rec["token"]:
            cx.failures.append(f"{where}: first-token logits/token differ from the ResidentReference replay of this "
                               f"configuration's own decisions (D12)")


def _wd_outcome(res, file_bytes) -> tuple:
    """(outcome, reasons), recomputed from the per-file phases of a schema-valid M4_EVAL_V2 record."""
    files, ps = res["files"], res["page_size"]
    if not files:
        return "empty", []
    pages = -(-file_bytes // ps)
    errors, low = [], []
    for f in files:
        ini, pw, fin = f["initial"], f["prewarm"], f["final"]
        name = f["file"]
        if ini["status"] != "observed" or fin["status"] != "observed":
            errors.append(f"{name}: observation failed ({ini.get('error') or fin.get('error')})")
            continue
        if (ini["size"], fin["size"], ini["pages"], fin["pages"]) != (file_bytes, file_bytes, pages, pages):
            errors.append(f"{name}: size/pages {ini['size']}/{ini['pages']} -> {fin['size']}/{fin['pages']}, "
                          f"a chunk file has {file_bytes} B / {pages} pages")
            continue
        need = ini["resident_pages"] < ini["pages"]
        if need and not (pw["status"] == "read" and pw["bytes_read"] == file_bytes):
            errors.append(f"{name}: initially {ini['resident_pages']}/{pages} pages, prewarm {pw!r} (one whole-file "
                          "read required)")
            continue
        if not need and pw["status"] != "not_needed":
            errors.append(f"{name}: fully resident initially, but prewarm {pw['status']!r}")
            continue
        if fin["resident_pages"] < fin["pages"]:
            low.append(f"{name}: {fin['resident_pages']}/{pages} pages resident in the final pass"
                       + (" after one prewarm" if need else ""))
    if errors:
        return "error", errors + low
    return ("not_resident", low) if low else ("verified", [])


def _check_warm_disk(cx, where, res, want, m, N, C, file_bytes):
    """M4_EVAL_V2: the candidate set, the per-file phases, the phase order and the outcome of one C2 record."""
    why = []
    if want is None:
        why.append("the candidate files cannot be recomputed (no valid namespace digest)")
    else:
        idx, names = want
        if res["candidates"] != names or [f["file"] for f in res["files"]] != names \
                or [f["index"] for f in res["files"]] != idx:
            why.append(f"candidate files {res['candidates']} (indices {[f['index'] for f in res['files']]}) != "
                       f"the restore's disk reads {names} (indices {idx}) from the key-level model")
    E, n = max_restore_len(N, C) // C, len(m["sources"])
    want_plan = {"eligible_chunks": E, "present_chunks": n, "host_chunks": m["sources"].count("host"),
                 "stop": "first_absent" if n < E else "all_eligible_present"}
    if res["plan"] != want_plan:
        why.append(f"plan {res['plan']} != {want_plan}")
    files = res["files"]
    reads = [f for f in files if f["prewarm"]["status"] in ("read", "error")]
    if res["prewarmed_files"] != sum(1 for f in files if f["prewarm"]["status"] == "read"):
        why.append(f"prewarmed_files {res['prewarmed_files']} != the reads recorded per file")
    want_ph = (dict.fromkeys(("initial", "prewarm", "final"), "not_applicable") if not files else
               {"initial": "done", "prewarm": "done" if reads else "not_needed", "final": "done"})
    if res["phases"] != want_ph:
        why.append(f"phases {res['phases']} != {want_ph}")
    order_ok = all(f["initial"]["t_ns"] >= res["t_planned_ns"] for f in files)
    last_ini = max((f["initial"]["t_ns"] for f in files), default=res["t_planned_ns"])
    last_pw = max((f["prewarm"]["t_end_ns"] for f in reads), default=last_ini)
    order_ok &= all(f["prewarm"]["t_start_ns"] >= last_ini for f in reads)
    order_ok &= all(f["final"]["t_ns"] >= last_pw for f in files)
    order_ok &= all(f["final"]["t_ns"] <= res["t_verified_ns"] for f in files)
    order_ok &= res["t_start_ns"] <= res["t_planned_ns"] <= res["t_verified_ns"]
    if not order_ok:
        why.append("phase timestamps are not in the order plan, initial pass, prewarms, final pass")
    got, reasons = _wd_outcome(res, file_bytes)
    if res["outcome"] != got:
        why.append(f"recorded outcome {res['outcome']!r} != recomputed {got!r}")
    if got not in ("verified", "empty"):
        why.append(f"warm-disk preparation {got}: {reasons}")
    if why:
        cx.p(f"{where}: warm-disk rule (request candidates): {why}")
    cx.drop_obs += [(f"{where} before the request", res["counters_before"]),
                    (f"{where} after the request", res["counters_after_request"])]


def _check_block(cx, sess, blk, spec, name, role, seed, trace, b, payload, file_bytes, cuda, ns_digest=None):
    where0 = f"{sess} {role} {name}"
    for k, want in (("name", name), ("role", role), ("trace_sha256", trace.digest),
                    ("n_requests", len(trace.requests)), ("budgets", b), ("strict_checked", role == "evaluation"),
                    ("seed", spec["seeds"]["warmup" if role == "warmup" else role][name])):
        if blk.get(k) != want:
            cx.p(f"{where0}: {k} {blk.get(k)!r} != expected {want!r}")
    recs = blk.get("records")
    n_cfg = len(spec["configs"])
    if not isinstance(recs, list) or len(recs) != len(trace.requests) * n_cfg:
        cx.p(f"{where0}: {len(recs) if isinstance(recs, list) else type(recs).__name__} records, expected "
             f"{len(trace.requests) * n_cfg}")
        return (recs if isinstance(recs, list) else []), []
    model = _models(spec, trace, b)
    request_scoped = ms.warm_disk_protocol(spec) == REQUEST_SCOPED
    good = []
    cum = {c: dict.fromkeys(("ev", "stops", "refused", "entries", "spills", "dev", "hits", "stored_n", "req",
                             "restored", "host", "disk", "h2d", "d2h", "read", "written", "tokens", "misses"), 0)
           for c in spec["configs"]}
    for idx, req in enumerate(trace.requests):
        order = ms.config_order(spec, seed, name, role, idx)
        for pos, cname in enumerate(order):
            rec = recs[idx * n_cfg + pos]
            where = f"{where0} r{idx} {cname}"
            if not isinstance(rec, dict) or (rec.get("index"), rec.get("config"), rec.get("pos")) != (idx, cname, pos):
                cx.p(f"{where}: record out of the seeded interleaved order")
                continue
            bad = _schema(rec, cname, role == "evaluation", cuda, ms.warm_disk_protocol(spec))
            if bad:
                cx.p(f"{where}: malformed record: {bad}")
                continue
            per = {"C1PA_W": model["C1PA_W"][idx], "C1PA_HALF": model["C1PA_HALF"][idx], "C2": model["C2"][idx]}
            if request_scoped and cname.startswith("C2"):
                per["C2_candidates"] = None
                if ns_digest is not None:
                    keys = chain_keys(ns_digest, list(req.tokens), spec["chunk_size"])
                    idx_disk = [i for i, s_ in enumerate(per["C2"]["sources"]) if s_ == "disk"]
                    per["C2_candidates"] = (idx_disk, [keys[i].hex() + ".kvc" for i in idx_disk])
            _check_record(cx, where, rec, cname, req, per, cum, b, spec, payload, file_bytes, cuda,
                          evaluation=role == "evaluation")
            good.append(rec)
    return recs, good


def _coverage(cx, sess, warm_blocks):
    tot = {}
    for blk in warm_blocks:
        for r in blk.get("records") or []:
            if not isinstance(r, dict):
                continue
            t = tot.setdefault(r.get("config"), {"hits": 0, "stored": 0, "host": 0, "disk": 0, "evict_or_stop": 0,
                                                 "spills": 0})
            t["hits"] += bool(r.get("restore_len"))
            t["stored"] += bool(r.get("stored"))
            t["host"] += (r.get("sources") or []).count("host")
            t["disk"] += (r.get("sources") or []).count("disk")
            c1 = (r.get("c1") or {}).get("counters", {})
            t["evict_or_stop"] = max(t["evict_or_stop"], c1.get("evictions", 0) + c1.get("admission_stops_prefix_capacity", 0))
            t["spills"] = max(t["spills"], ((r.get("c2") or {}).get("counters") or {}).get("host_spills_to_disk", 0))
    for c in ("C1PA_W", "C1PA_HALF", "C2_base", "C2_batch"):
        t = tot.get(c, {})
        if not (t.get("hits") and t.get("stored")):
            cx.p(f"{sess}: warmup coverage: {c} needs >= 1 hit and >= 1 admission, got {t}")
    if not tot.get("C1PA_HALF", {}).get("evict_or_stop"):
        cx.p(f"{sess}: warmup coverage: C1PA_HALF had no eviction or prefix-capacity stop")
    for c in ("C2_base", "C2_batch"):
        t = tot.get(c, {})
        if not (t.get("host") and t.get("disk") and t.get("spills")):
            cx.p(f"{sess}: warmup coverage: {c} needs >= 1 host hit, disk hit and spill, got {t}")


def validate(run_dir, spec: dict, *, manifest: dict | None = None, manifest_sha256: str | None = None,
             allow_unfrozen: bool = False) -> dict:
    """Validate a run directory against `spec` (and the freeze manifest, if given)."""
    run_dir = Path(run_dir)
    cx = _Ctx()
    if manifest is None and not allow_unfrozen:
        cx.p("no freeze manifest: the evidence is not bound to a frozen spec")
    if manifest is not None:
        for k, want in (("experiment", spec["name"]), ("run_dir", spec["run_dir"]), ("budget_s", spec["budget_s"]),
                        ("commands", ms.commands_for(spec)), ("spec_sha256", spec["spec_sha256"])):
            if manifest.get(k) != want:
                cx.p(f"freeze manifest {k} {manifest.get(k)!r} != spec {want!r}")
    present = {str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()} if run_dir.is_dir() else set()
    allowed = expected_files(spec)
    extra = sorted(present - allowed)
    if extra:
        cx.p(f"unexpected files in the run directory: {extra}")
    rec = None
    try:
        rec = mc.load_json_finite((run_dir / BATCH_RECORD).read_bytes())
    except (OSError, ValueError) as e:
        cx.p(f"batch record unreadable: {e!r}")
    if rec is not None:
        if manifest is not None:
            cx.problems += mc.batch_problems(rec, manifest, manifest_sha256, run_dir, spec)
        else:  # unfrozen (draft/tests only): the same execution checks against the spec's commands
            cx.problems += mc.batch_problems(rec, {"commands": ms.commands_for(spec), "frozen_unix": 0.0},
                                             rec.get("freeze_manifest_sha256") if isinstance(rec, dict) else None,
                                             run_dir, spec)
    sources, namespaces, fails = [], [], {}
    counts = {"sessions": 0, "evaluation_records": 0, "warmup_records": 0}
    sessions = {}
    for sid, seed in spec["sessions"].items():
        n_prob0 = len(cx.problems)
        try:
            path = run_dir / f"session_{sid}.json"
            if not path.is_file():
                cx.p(f"session {sid}: no session file")
                continue
            try:
                r = mc.load_json_finite(path.read_bytes())
            except ValueError as e:
                cx.p(f"session {sid}: unparseable ({e!r})")
                continue
            if not isinstance(r, dict):
                cx.p(f"session {sid}: not a JSON object")
                continue
            for k, want in (("format", "m4-session-v1"), ("spec", spec["name"]), ("spec_sha256", spec["spec_sha256"]),
                            ("session", sid), ("seed", seed), ("finished", True), ("chunk_size", spec["chunk_size"]),
                            ("trace_order", ms.trace_order(spec, seed))):
                if r.get(k) != want:
                    cx.p(f"session {sid}: {k} {r.get(k)!r} != expected {want!r}")
            payload, file_bytes = r.get("payload_bytes"), r.get("file_bytes")
            if not (_nn(payload) and _nn(file_bytes) and payload > 0 and payload % spec["chunk_size"] == 0
                    and file_bytes > payload):
                cx.p(f"session {sid}: payload/file sizes malformed")
                continue
            if spec.get("expected_payload_bytes") is not None and payload != spec["expected_payload_bytes"]:
                cx.p(f"session {sid}: chunk payload {payload} B is not the spec's model ({spec['expected_payload_bytes']} B)")
            dev = r.get("device")
            if not isinstance(dev, str) or not dev:
                cx.p(f"session {sid}: device missing")
                continue
            cuda = dev.startswith("cuda")
            if spec.get("require_cuda") and not cuda:
                cx.p(f"session {sid}: device {dev!r}; the spec requires a CUDA device")
            ns_digest = None
            if not _hex64(r.get("namespace_sha256")):
                cx.p(f"session {sid}: namespace digest missing or malformed")
            else:
                namespaces.append(r["namespace_sha256"])
                ns_digest = bytes.fromhex(r["namespace_sha256"])
            if ms.warm_disk_protocol(spec) == REQUEST_SCOPED:
                itf = r.get("interference")
                if not (isinstance(itf, dict) and isinstance(itf.get("start"), dict) and isinstance(itf.get("end"), dict)):
                    cx.p(f"session {sid}: interference counter snapshots (start/end) missing or malformed")
                else:
                    cx.drop_obs += [(f"session {sid} start", itf["start"]), (f"session {sid} end", itf["end"])]
            if not (mc.finite(r.get("started_unix")) and mc.finite(r.get("ended_unix"))
                    and r["started_unix"] <= r["ended_unix"]):
                cx.p(f"session {sid}: start/end times missing, not finite or reversed")
            if not (isinstance(r.get("source"), dict) and isinstance(r.get("environment"), dict)):
                cx.p(f"session {sid}: source or environment record missing")
            if not _nn(r.get("strict_mismatches")):
                cx.p(f"session {sid}: runner-recorded strict_mismatches missing or malformed")
            n_fail0 = len(cx.failures)
            sources.append(json.dumps(r.get("source"), sort_keys=True))
            if manifest is not None:
                src = r.get("source") or {}
                if src.get("files_sha256") != manifest.get("files_sha256") or \
                        src.get("git_head") != manifest.get("base_git_head") or \
                        src.get("git_diff_head_sha256") != manifest.get("git_diff_head_sha256"):
                    cx.p(f"session {sid}: source tree differs from the freeze manifest")
                if (r.get("environment") or {}).get("stack") != manifest.get("stack"):
                    cx.p(f"session {sid}: software/GPU stack differs from the freeze manifest")
            for role, key in (("warmup", "warmup"), ("evaluation", "evaluation")):
                traces = {t: ms.trace(spec, t, role) for t in ms.TRACE_NAMES}
                b = ms.budgets(spec, role, traces)
                order = list(ms.TRACE_NAMES) if role == "warmup" else ms.trace_order(spec, seed)
                blocks = r.get(key)
                if not isinstance(blocks, list) or len(blocks) != 4:
                    cx.p(f"session {sid}: {role} blocks {type(blocks).__name__}, expected 4")
                    continue
                good_blocks = []
                for name, blk in zip(order, blocks):
                    if not isinstance(blk, dict):
                        cx.p(f"session {sid}: {role} {name} block malformed")
                        continue
                    recs, good = _check_block(cx, sid, blk, spec, name, role, seed, traces[name], b[name], payload,
                                              file_bytes, cuda, ns_digest)
                    counts[f"{role}_records"] += len(recs)
                    good_blocks.append({"records": good})
                if role == "warmup":
                    _coverage(cx, sid, good_blocks)  # only schema-valid records are aggregated
            walls = [x.get("t_wall") for role in ("warmup", "evaluation") for blk in (r.get(role) or [])
                     if isinstance(blk, dict) for x in (blk.get("records") or []) if isinstance(x, dict)]
            if any(not mc.finite(w) for w in walls) or any(b_ < a for a, b_ in zip(walls, walls[1:])):
                cx.p(f"session {sid}: record wall times missing or not chronological")
            elif walls and mc.finite(r.get("started_unix")) and mc.finite(r.get("ended_unix")) and \
                    not r["started_unix"] <= walls[0] <= walls[-1] <= r["ended_unix"]:
                cx.p(f"session {sid}: record wall times outside the session's start/end")
            fails[sid] = len(cx.failures) - n_fail0
            if _nn(r.get("strict_mismatches")) and r["strict_mismatches"] != fails[sid]:
                cx.p(f"session {sid}: runner-recorded strict_mismatches {r['strict_mismatches']} != recomputed {fails[sid]}")
            sessions[sid] = r
            counts["sessions"] += 1
        except (KeyError, TypeError, AttributeError, ValueError, IndexError, OverflowError) as e:
            # Last-resort backstop, fail-closed: evidence that the validator cannot interpret is invalid
            # (exit 3, with a report). It can never make evidence valid. The schema pass is meant to catch
            # malformed records first; reaching this line is itself reported, with the exception text.
            cx.p(f"session {sid}: evidence could not be interpreted ({type(e).__name__}: {e}); treated as malformed")
            sessions.pop(sid, None)
            fails.pop(sid, None)
            if len(cx.problems) == n_prob0:
                cx.p(f"session {sid}: malformed")

    if len(set(sources)) > 1:
        cx.p("sessions ran from different source trees")
    if len(set(namespaces)) > 1:
        cx.p("sessions used different model namespaces")
    # P1: every session exit must be explained by that session's recomputed strict result. The
    # runner (scripts/m4_measure.py) exits 4 exactly when it recorded a strict mismatch, else 0. Any other exit, or an exit
    # that disagrees with the recomputed comparison, is invalid evidence (exit 3), even when the
    # batch record has the consistent shape kvlab.m3_confirm.batch_problems accepts.
    entries = rec.get("sessions") if isinstance(rec, dict) and isinstance(rec.get("sessions"), list) else []
    for e in entries:
        sid = e.get("session") if isinstance(e, dict) else None
        if sid not in fails:
            continue
        want = 4 if fails[sid] else 0
        if e.get("returncode") != want or type(e.get("returncode")) is not int:
            cx.p(f"session {sid}: exit {e.get('returncode')!r} is not explained by its recorded result "
                 f"(expected {want}: {fails[sid]} recomputed strict mismatches)")
    if ms.warm_disk_protocol(spec) == REQUEST_SCOPED:
        _interference(cx, run_dir, spec, rec, sessions)
    status = "valid" if not cx.problems else "invalid"
    return {"spec": spec["name"], "spec_sha256": spec["spec_sha256"], "run_dir": str(run_dir), "status": status,
            "frozen": manifest is not None, "problems": cx.problems, "correctness_failures": cx.failures,
            "counts": counts, "sessions": sessions if status == "valid" else None}


def _drop_count(snap):
    """The drop_pagecache count of a recorded snapshot (kvlab.warm_disk.counters), or None if it was not read."""
    if not (isinstance(snap, dict) and snap.get("status") == "ok" and isinstance(snap.get("vmstat"), dict)):
        return None
    v = snap["vmstat"].get("drop_pagecache")
    return v if _nn(v) else None


def _interference(cx, run_dir: Path, spec: dict, batch, sessions: dict) -> None:
    """M4_EVAL_V2's predeclared global-interference rule (drop_pagecache), over the whole batch. `batch` is the
    batch record (or None) and `sessions` the parsed session reports, for the bracket check."""
    try:
        rec = mc.load_json_finite((run_dir / INTERFERENCE_RECORD).read_bytes())
    except (OSError, ValueError) as e:
        cx.p(f"batch interference record missing or unreadable: {e!r}")
        return
    if not isinstance(rec, dict) or rec.get("format") != INTERFERENCE_FORMAT:
        cx.p("batch interference record malformed")
        return
    if (rec.get("experiment"), rec.get("counter")) != (spec["name"], "drop_pagecache"):
        cx.p(f"batch interference record names {rec.get('experiment')!r}/{rec.get('counter')!r}, expected "
             f"{spec['name']!r}/'drop_pagecache'")
    s_u, e_u = rec.get("start_unix"), rec.get("end_unix")
    if not (mc.finite(s_u) and mc.finite(e_u) and s_u <= e_u):
        cx.p(f"batch interference read times {s_u!r} -> {e_u!r} missing, not finite or reversed")
    else:
        launched = batch.get("launched_unix") if isinstance(batch, dict) else None
        starts = [r.get("started_unix") for r in sessions.values() if mc.finite(r.get("started_unix"))]
        ends = [r.get("ended_unix") for r in sessions.values() if mc.finite(r.get("ended_unix"))]
        if (mc.finite(launched) and not s_u <= launched <= e_u) or any(t < s_u for t in starts) \
                or any(t > e_u for t in ends):
            cx.p(f"batch interference reads ({s_u} -> {e_u}) do not bracket the batch (first launch {launched!r}, "
                 f"sessions {min(starts, default=None)!r} -> {max(ends, default=None)!r})")
    start, end = _drop_count(rec.get("start")), _drop_count(rec.get("end"))
    if start is None or end is None:
        cx.p(f"batch drop_pagecache count not read at the {'start' if start is None else 'end'} "
             "(missing, unavailable or unreadable): the interference rule cannot be checked")
        return
    if end < start:
        cx.p(f"batch drop_pagecache decreased {start} -> {end} (counter reset or reboot)")
    elif end > start:
        cx.p(f"batch drop_pagecache increased by {end - start}: a global page-cache drop during the batch")
    bad = [(w, _drop_count(s)) for w, s in cx.drop_obs if _drop_count(s) != start]
    if bad:
        cx.p(f"{len(bad)} drop_pagecache observations in the sessions are missing or differ from the batch start "
             f"value {start}; first: {bad[:3]}")


def exit_code(v: dict) -> int:
    if v["status"] != "valid":
        return 3
    return 4 if v["correctness_failures"] else 0
