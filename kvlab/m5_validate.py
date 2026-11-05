"""M5b: fail-closed validation of calibration and evaluation evidence (CPU only, no torch).

The specification is recomputed in code (kvlab/m5_spec.py) and passed in. The evidence is never trusted to
describe itself. Everything is recomputed from the spec, the recorded namespace (which gives the chunk keys and
file names) and the frozen predictions:
- the session list, the batch plan and the batch execution record (`kvlab.m3_confirm.batch_problems`);
- every sample's identity and its chronological position (case order, phase, repetition, position, run label),
  warmups included;
- the expected plan, selected and actual lengths, sources, admissions, decision reason and counter deltas;
- the page-cache files, outcome and drop_pagecache rule;
- the state fingerprints and the content checks;
- the strict checks;
- the source, spec, namespace, model, cost-model and prediction bindings.

Any problem makes the evidence invalid (exit 3). A strict mismatch in otherwise complete evidence is exit 4.
Malformed records are problems, never crashes: a per-record backstop reports anything uninterpretable, and it
can never make evidence valid. Invalid evidence gets no performance analysis (kvlab/m5_analysis.py refuses it).
"""

import hashlib
import json
from pathlib import Path

from kvlab import chunk_format as cf
from kvlab import m3_confirm as mc
from kvlab import m5_spec as sp
from kvlab import restore_cost as rc
from kvlab.m5_fit import make_predictions
from kvlab.m5_spec import SESSION_FORMAT, batch_plan
from kvlab import warm_disk as wd
from kvlab.boundaries import max_restore_len
from kvlab.identity import Namespace, canonical_json, chain_keys, namespace_digest_of_canonical

PLAN_FILE, SPEC_FILE = "batch_plan.json", "spec.json"


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p: Path, problems: list, what: str):
    """A JSON OBJECT from `p`, or None with a recorded problem (missing, unreadable, not finite, or any other
    JSON value such as null or a list)."""
    if not p.is_file():
        problems.append(f"{what}: missing")
        return None
    try:
        obj = mc.load_json_finite(p.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{what}: unreadable or not finite JSON ({type(e).__name__}: {e})")
        return None
    if not isinstance(obj, dict):
        problems.append(f"{what}: not a JSON object ({type(obj).__name__})")
        return None
    return obj


def expected_files(spec: dict) -> set:
    out = {PLAN_FILE, SPEC_FILE, mc.BATCH_RECORD}
    for sid in spec["schedule"]["sessions"]:
        out |= {f"session_{sid}.json", f"logs/session_{sid}.log"}
    return out


def namespace_from(rec_ns) -> tuple:
    """(Namespace, digest bytes) from a session's recorded namespace; raises on any inconsistency."""
    fields = dict(rec_ns["fields"])
    fields["layer_types"] = tuple(fields["layer_types"])
    ns = Namespace(**fields)
    d = namespace_digest_of_canonical(canonical_json(ns.fields()))
    if d.hex() != rec_ns["digest"] or ns.digest() != d:
        raise ValueError("recorded namespace digest does not match its fields")
    return ns, d


def expected_plan(case: dict, spec: dict, ns_hex: str) -> dict:
    N, C = case["prompt"]["length"], spec["chunk_size"]
    k = case["feasible_chunks"]
    n_elig = max_restore_len(N, C) // C
    indexed = []
    for s in case["chunks"][:n_elig]:
        if s == "absent":
            break
        indexed.append(s)
    return {"prompt_len": N, "chunk_size": C, "eligible_len": max_restore_len(N, C), "indexed_tiers": indexed,
            "feasible_lengths": list(range(0, k * C + 1, C)), "limit_reason": case["limit_reason"],
            "presence": "indexed_not_validated", "namespace_digest": ns_hex,
            "restore_backend": spec["restore_backend"],
            "restore_staging_chunks": spec["restore_staging_chunks"] if spec["restore_staging_chunks"] is not None else 4,
            "disk_staging_available": True}


def expected_reason(case: dict, label: str, predictions) -> str:
    kind, v = sp.parse_label(label)
    if kind == "candidate":
        return "fixed_chunks"
    if v == "always_restore":
        return "longest_feasible_prefix" if case["candidates"][-1] else "no_feasible_restore"
    if v == "recompute":
        return "policy_recompute"
    return predictions["cases"][case["case_id"]]["decision"]["reason"]


def expected_delta(case: dict, L: int, C: int, cb: int, file_bytes: int) -> dict:
    """The counter delta one sample must produce: hits and bytes of exactly the selected prefix, the declared
    admissions (and nothing else) stored, no miss, spill, write skip or disk write."""
    k = L // C
    tiers = case["feasible_tiers"][:k]
    m = len(case["admitted"])
    return {"requests": 1, "restored_requests": int(L > 0), "full_recompute_requests": int(L == 0),
            "host_hits": tiers.count("host"), "disk_hits": tiers.count("disk"), "chunk_misses": 0,
            "recovered_prefix_tokens": L, "chunks_stored": m, "host_spills_to_disk": 0, "bytes_d2h": m * cb,
            "bytes_h2d": k * cb, "disk_file_bytes_read": tiers.count("disk") * file_bytes,
            "disk_file_bytes_written": 0, "miss_reasons": {}, "write_skips": {}, "spill_failures": {}}


def _drop(snap):
    return snap.get("vmstat", {}).get(wd.RULE_COUNTER) if isinstance(snap, dict) and snap.get("status") == "ok" else None


def _obs_ok(o, file_bytes, pages) -> bool:
    return isinstance(o, dict) and o.get("status") == "observed" and o.get("size") == file_bytes and \
        o.get("pages") == pages and _nat(o.get("resident_pages")) and o["resident_pages"] <= pages and _nat(o.get("t_ns"))


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _page_cache_record_problems(pc, file_bytes, t0) -> list:
    """Schema, bounds and chronology of a warm-disk record (the producer is kvlab.warm_disk, unchanged): every file
    observed twice with the expected size and page count, a legal prewarm, the passes in order, and the final pass
    before the timer start."""
    pages = (file_bytes + wd.PAGE - 1) // wd.PAGE
    if pc.get("page_size") != wd.PAGE:
        return [f"page size {pc.get('page_size')!r} != {wd.PAGE}"]
    times = [pc.get("t_start_ns"), pc.get("t_planned_ns")]
    files = pc["files"]
    for f in files:
        if not _obs_ok(f.get("initial"), file_bytes, pages):
            return [f"initial observation of {f.get('file')} malformed or impossible: {f.get('initial')!r}"]
        times.append(f["initial"]["t_ns"])
    for f in files:
        pw, ini = f.get("prewarm"), f["initial"]
        if not isinstance(pw, dict):
            return [f"prewarm record of {f.get('file')} missing"]
        if pw.get("status") == "not_needed":
            if ini["resident_pages"] != pages:
                return [f"prewarm of {f.get('file')} skipped although the file was not resident"]
        elif pw.get("status") == "read":
            if pw.get("bytes_read") != file_bytes or not (_nat(pw.get("t_start_ns")) and _nat(pw.get("t_end_ns"))):
                return [f"prewarm of {f.get('file')} malformed: {pw!r}"]
            times += [pw["t_start_ns"], pw["t_end_ns"]]
        else:
            return [f"prewarm of {f.get('file')} has status {pw.get('status')!r}"]
    for f in files:
        if not _obs_ok(f.get("final"), file_bytes, pages):
            return [f"final observation of {f.get('file')} malformed or impossible: {f.get('final')!r}"]
        times.append(f["final"]["t_ns"])
    times += [pc.get("t_verified_ns"), t0]
    if not all(_nat(x) for x in times) or any(b < a for a, b in zip(times, times[1:])):
        return [f"page-cache passes out of order or after the timer start: {times}"]
    if pc.get("timer_start_minus_verified_ns") != t0 - pc["t_verified_ns"]:
        return ["recorded gap between the final pass and the timer start is inconsistent"]
    return []


def _check_page_cache(where, pc, want_files, out, file_bytes=None, t0=None):
    if not isinstance(pc, dict) or pc.get("protocol") != wd.PROTOCOL:
        out.append(f"{where}: page-cache record missing or not {wd.PROTOCOL}")
        return
    if pc.get("candidates") != want_files:
        out.append(f"{where}: page-cache files {pc.get('candidates')} != the maximal feasible prefix's disk files {want_files}")
    files = pc.get("files")
    if not isinstance(files, list) or [f.get("file") for f in files] != want_files:
        out.append(f"{where}: page-cache per-file records do not match the candidate files")
        return
    got = wd.outcome(files)
    want = "verified" if want_files else "empty"
    if got != want or pc.get("outcome") != got:
        out.append(f"{where}: page-cache outcome recomputed {got!r}, recorded {pc.get('outcome')!r}, required {want!r}")
    a, b = _drop(pc.get("counters_before")), _drop(pc.get("counters_after_request"))
    if a is None or b is None or a != b:
        out.append(f"{where}: drop_pagecache {a!r} -> {b!r}: unreadable or changed around the request")
    gap = pc.get("timer_start_minus_verified_ns")
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 0:
        out.append(f"{where}: timer start before the final page-cache pass ({gap!r})")
    if file_bytes is not None:
        if not _nat(t0):
            out.append(f"{where}: no timer start to order the page-cache passes against")
        else:
            out += [f"{where}: {x}" for x in _page_cache_record_problems(pc, file_bytes, t0)]


def _check_sample(where, s, case, label, spec, ctx, predictions, out, strict_out):
    C = spec["chunk_size"]
    want_L = sp.expected_selected(case, label, predictions)
    kind, v = sp.parse_label(label)
    want_policy = "fixed_chunks" if kind == "candidate" else v
    if s.get("fingerprint") != ctx["fingerprint"]:
        out.append(f"{where}: state fingerprint {s.get('fingerprint')!r} != the case setup's")
    if s.get("policy") != want_policy:
        out.append(f"{where}: policy {s.get('policy')!r}, expected {want_policy!r}")
    for k, want in (("selected_len", want_L), ("actual_len", want_L), ("restore_len", want_L),
                    ("sources", case["feasible_tiers"][:want_L // C]), ("stored_chunks", case["admitted"]),
                    ("miss_reason", None), ("restore_miss_reason", None)):
        if s.get(k) != want or type(s.get(k)) is bool:
            out.append(f"{where}: {k} {s.get(k)!r}, expected {want!r}")
    dec = s.get("decision") or {}
    if dec.get("plan") != ctx["plan"]:
        out.append(f"{where}: decision plan differs from the spec's: {dec.get('plan')!r}")
    if dec.get("reason") != expected_reason(case, label, predictions):
        out.append(f"{where}: decision reason {dec.get('reason')!r}, expected {expected_reason(case, label, predictions)!r}")
    if kind == "policy" and v == "cost_aware":
        want_obj = predictions["cases"][case["case_id"]]["decision"]["objective_ms"]
        if dec.get("objective_ms") != want_obj:
            out.append(f"{where}: cost_aware objectives differ from the frozen predictions")
    elif dec.get("objective_ms") is not None:
        out.append(f"{where}: objectives recorded for a non-model run")
    want_delta = expected_delta(case, want_L, C, ctx["cb"], ctx["file_bytes"])
    if s.get("counters_delta") != want_delta:
        out.append(f"{where}: counter delta {s.get('counters_delta')!r} != expected {want_delta!r}")
    t0, t1, lat = s.get("t0_ns"), s.get("t1_ns"), s.get("latency_ns")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (t0, t1, lat)) or lat != t1 - t0 or lat <= 0:
        out.append(f"{where}: invalid timing t0={t0!r} t1={t1!r} latency={lat!r}")
    if not sp.is_finite(s.get("t_wall")):
        out.append(f"{where}: t_wall missing or not finite")
    _check_page_cache(where, s.get("page_cache"), ctx["disk_files"], out, ctx["file_bytes"], t0)
    st = s.get("strict") or {}
    if st.get("split") != want_L:
        out.append(f"{where}: strict check of split {st.get('split')!r}, not the sample's {want_L}")
    if want_L:
        rb = st.get("restored_bytes") or {}
        if rb.get("all_bytes_equal") is not True or rb.get("mismatched") != 0 or \
                rb.get("tensors_checked") != want_L // C * 2 * ctx["layers"]:
            strict_out.append(f"{where}: restored bytes differ from the source bank: {rb!r}")
    elif "restored_bytes" in st:
        out.append(f"{where}: restored-byte record for L = 0")
    if st.get("logits_equal") is not True:
        strict_out.append(f"{where}: first-token logits differ from the same-split reference")
    if s.get("first_token") != st.get("reference_token") or not isinstance(s.get("first_token"), int):
        strict_out.append(f"{where}: first token {s.get('first_token')!r} != reference {st.get('reference_token')!r}")


def _hex(x, n) -> bool:
    return isinstance(x, str) and len(x) == n and all(c in "0123456789abcdef" for c in x)


def identity_problems(sess, require_cuda) -> list:
    """The provenance a session must record: a clean, identified source tree, the model weight files, the
    package versions and (for GPU evidence) the GPU. Equal missing values never count as matching provenance."""
    out = []
    src = sess.get("source")
    if not isinstance(src, dict) or src.get("git_status_porcelain") != [] or not _hex(src.get("git_head"), 40) \
            or not _hex(src.get("git_diff_head_sha256"), 64) or not isinstance(src.get("files_sha256"), dict) \
            or not src["files_sha256"] or not all(_hex(v, 64) for v in src["files_sha256"].values()):
        out.append("source identity missing, malformed or not clean")
    blobs = sess.get("model_blobs")
    if not isinstance(blobs, dict) or not blobs or "error" in blobs or not all(
            isinstance(v, dict) and isinstance(v.get("blob"), str) and v["blob"]
            and isinstance(v.get("bytes"), int) and not isinstance(v["bytes"], bool) and v["bytes"] > 0
            for v in blobs.values()):
        out.append(f"model weight identity missing, an error or without blob ids and sizes: {blobs!r}"[:200])
    pk = sess.get("packages")
    if not isinstance(pk, dict) or not pk or not all(isinstance(v, str) and v for v in pk.values()):
        out.append("package versions missing or empty")
    if require_cuda and (not isinstance(sess.get("gpu"), str) or not sess["gpu"]):
        out.append("GPU identity missing")
    return out


def _check_session(sid, sess, spec, predictions, artifact, require_cuda, out, strict_out) -> dict:
    """Returns {namespace_hex, source, model} for the cross-session comparison."""
    w = f"session {sid}"
    for k, want in (("format", SESSION_FORMAT), ("spec_name", spec["name"]), ("spec_sha256", sp.digest(spec)),
                    ("session", sid), ("session_seed", spec["schedule"]["sessions"][sid]), ("status", "complete")):
        if sess.get(k) != want:
            out.append(f"{w}: {k} {sess.get(k)!r}, expected {want!r}")
    if require_cuda and (not str(sess.get("device", "")).startswith("cuda") or sess.get("pinned_host") is not True):
        out.append(f"{w}: device {sess.get('device')!r}, pinned {sess.get('pinned_host')!r}: not the declared GPU run")
    if spec["split"] == "evaluation":
        if sess.get("cost_model_digest") != artifact["digest"] or sess.get("predictions_sha256") != predictions["sha256"]:
            out.append(f"{w}: cost model / predictions binding differs from the frozen files")
    elif sess.get("cost_model_digest") is not None or sess.get("predictions_sha256") is not None:
        out.append(f"{w}: a calibration session must not load a cost model or predictions")
    ns, nd = namespace_from(sess["namespace"])
    if ns.chunk_size != spec["chunk_size"]:
        out.append(f"{w}: namespace chunk size {ns.chunk_size} != spec {spec['chunk_size']}")
    model = spec["model"]
    if (ns.model_id, ns.revision, ns.kv_dtype, ns.attn_implementation) != \
            (model["id"], model["revision"], model["dtype"], model["attn_implementation"]):
        out.append(f"{w}: namespace model {ns.model_id}@{ns.revision} {ns.kv_dtype} {ns.attn_implementation} "
                   f"is not the spec's")
    src = sess.get("source")
    out += [f"{w}: {x}" for x in identity_problems(sess, require_cuda)]
    cb, file_bytes = cf.payload_len(ns), cf.file_size(ns)
    by_id = sp.case_by_id(spec)
    order = sp.case_order(spec, sid)
    cases = sess.get("cases") if isinstance(sess.get("cases"), list) else []
    if [c.get("case_id") for c in cases] != order:
        out.append(f"{w}: case records {[c.get('case_id') for c in cases]} != the seeded order {order}")
    ctx_by_case = {}
    for crec in cases:
        cid = crec.get("case_id")
        if cid not in by_id:
            continue
        case = by_id[cid]
        try:
            tokens = sp.prompt_tokens(case["prompt"])
            keys = chain_keys(nd, tokens, spec["chunk_size"])
            hexes = [k.hex() for k in keys]
            want = sp.expected_structure(case, hexes, file_bytes)
            setup = crec["setup_structure"]
            if {k: v for k, v in setup.items() if k != "file_stats"} != want:
                out.append(f"{w} {cid}: prepared structure differs from the spec")
            if crec.get("setup_fingerprint") != sp.digest(setup) or crec.get("end_fingerprint") != crec.get("setup_fingerprint"):
                out.append(f"{w} {cid}: setup/end fingerprints inconsistent")
            bank = crec["bank"]
            if bank.get("P") != case["source"]["P"] or len(bank["payload_sha256"]) != case["source"]["P"] // spec["chunk_size"]:
                out.append(f"{w} {cid}: bank does not cover the declared provenance prefix")
            want_host = {hexes[i]: bank["payload_sha256"][i] for i in case["host_lru"]}
            want_disk = {hexes[i]: bank["file_sha256"][i] for i in case["disk_lru"]}
            for tag in ("setup_content", "end_content"):
                cc = crec.get(tag) or {}
                if cc.get("ok") is not True or cc.get("host") != want_host or cc.get("disk") != want_disk:
                    out.append(f"{w} {cid}: {tag} does not match the bank")
            ctx_by_case[cid] = {"fingerprint": crec.get("setup_fingerprint"), "cb": cb, "file_bytes": file_bytes,
                                "layers": ns.num_layers, "plan": expected_plan(case, spec, nd.hex()),
                                "disk_files": [f"{hexes[i]}.kvc" for i, t in enumerate(case["feasible_tiers"]) if t == "disk"]}
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:  # fail-closed backstop
            out.append(f"{w} {cid}: malformed case record ({type(e).__name__}: {e})")
    samples = sess.get("samples") if isinstance(sess.get("samples"), list) else []
    sched = sp.schedule(spec, sid)
    ids = [(s.get("case_id"), s.get("phase"), s.get("rep"), s.get("pos"), s.get("run")) if isinstance(s, dict) else None
           for s in samples]
    if ids != sched:
        n_ok = next((i for i, (a, b) in enumerate(zip(ids, sched)) if a != b), min(len(ids), len(sched)))
        out.append(f"{w}: sample identities differ from the schedule at position {n_ok} "
                   f"({len(ids)} recorded, {len(sched)} scheduled): missing, extra, duplicate or out of order")
    for i, (s, ident) in enumerate(zip(samples, sched)):
        where = f"{w} sample {i} {ident[0]} {ident[1]} r{ident[2]} {ident[4]}"
        if not isinstance(s, dict) or s.get("seq") != i:
            out.append(f"{where}: seq {s.get('seq') if isinstance(s, dict) else None!r} != {i}")
            continue
        if ident[0] not in ctx_by_case:
            out.append(f"{where}: its case record is missing or malformed")
            continue
        try:
            _check_sample(where, s, by_id[ident[0]], ident[4], spec, ctx_by_case[ident[0]], predictions, out, strict_out)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:  # fail-closed backstop
            out.append(f"{where}: malformed sample ({type(e).__name__}: {e})")
    return {"namespace": nd.hex(), "source": src, "model_blobs": sess.get("model_blobs"),
            "packages": sess.get("packages"), "gpu": sess.get("gpu"), "device": sess.get("device")}


def prediction_problems(spec, predictions, artifact, ns_hex=None) -> list:
    """The frozen predictions must be exactly what the bound cost model produces for this evaluation spec (a
    digest only shows the file is unchanged, not that the model produced its numbers); used before launch and in
    validation."""
    out = []
    if not isinstance(predictions, dict) or not isinstance(artifact, dict):
        return ["predictions or cost model is not a JSON object"]
    try:
        model = rc.RestoreCostModel(artifact)
    except rc.ArtifactError as e:
        return [f"cost model artifact invalid: {e}"]
    if not sp.is_finite(predictions.get("created_unix")):
        return ["predictions have no finite creation time"]
    want = make_predictions(spec, model, created_unix=predictions["created_unix"])
    if predictions != want:
        diff = sorted(k for k in set(want) | set(predictions) if predictions.get(k) != want.get(k))
        out.append(f"predictions differ from those the bound cost model produces for this spec (fields {diff})")
    _check_bindings(spec, predictions, artifact, ns_hex, out)
    return out


def _check_bindings(spec, predictions, artifact, ns_hex, out):
    body = {k: v for k, v in predictions.items() if k != "sha256"}
    if predictions.get("sha256") != sp.digest(body):
        out.append("predictions digest mismatch (modified after they were frozen)")
    for k, want in (("eval_spec_sha256", sp.digest(spec)), ("calibration_spec_sha256", spec["calibration_spec_sha256"]),
                    ("artifact_digest", artifact.get("digest"))) + ((("namespace_digest", ns_hex),) if ns_hex else ()):
        if predictions.get(k) != want:
            out.append(f"predictions {k} {predictions.get(k)!r}, expected {want!r}")
    if not isinstance(predictions.get("cases"), dict) or \
            set(predictions["cases"]) != {c["case_id"] for c in spec["cases"]}:
        out.append("predictions do not cover exactly the evaluation cases")
    prov = artifact.get("provenance") or {}
    if prov.get("calibration_spec_sha256") != spec["calibration_spec_sha256"]:
        out.append("cost model provenance is not bound to this evaluation's calibration spec")
    cal = artifact.get("calibration") or {}
    if cal.get("evaluation_case_ids") != sorted(c["case_id"] for c in spec["cases"]) or \
            cal.get("case_ids") != sorted(spec["calibration_case_ids"]):
        out.append("cost model case declarations differ from the calibration/evaluation specs")
    if ns_hex is not None and (artifact.get("compatibility") or {}).get("namespace_digest") != ns_hex:
        out.append("cost model namespace differs from the evaluation sessions'")


def validate(run_dir, spec: dict, *, predictions=None, artifact=None, require_cuda: bool = True) -> dict:
    """Validate one batch run directory against `spec`. Returns {"exit", "valid", "problems",
    "strict_problems", "sessions"}; `sessions` (parsed) is returned for the analysis only if valid."""
    run_dir = Path(run_dir)
    out, strict_out, sessions = [], [], {}
    if spec["split"] == "evaluation" and (predictions is None or artifact is None):
        out.append("evaluation evidence needs the frozen cost model and predictions")
        return {"exit": 3, "valid": False, "problems": out, "strict_problems": [], "sessions": None}
    present = {str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()} if run_dir.is_dir() else set()
    want = expected_files(spec)
    if present - want:
        out.append(f"unexpected files in the run directory: {sorted(present - want)[:8]}")
    if want - present:
        out.append(f"missing files: {sorted(want - present)}")
    rec_spec = _load(run_dir / SPEC_FILE, out, SPEC_FILE)
    if rec_spec is not None and rec_spec != json.loads(canonical_json(spec)):
        out.append("recorded spec.json differs from the spec in code")
    plan = _load(run_dir / PLAN_FILE, out, PLAN_FILE)
    rec = _load(run_dir / mc.BATCH_RECORD, out, mc.BATCH_RECORD)
    if plan is None or rec is None:
        out.append("no valid batch plan or batch record: execution cannot be verified")
    else:
        try:
            cmds = plan["commands"]
            python = cmds[0]["argv"][0]
            ev = spec["split"] == "evaluation"
            want_plan = batch_plan(spec, plan["run_dir"], python,
                                   cost_model_path=cmds[0]["argv"][-3] if ev else None,
                                   predictions_path=cmds[0]["argv"][-1] if ev else None,
                                   bindings=plan["bindings"], created_unix=plan["created_unix"])
            if plan != want_plan:
                out.append("batch plan differs from the plan derived from the spec")
            want_bind = {"cost_model_digest": artifact["digest"], "predictions_sha256": predictions["sha256"]} if ev else {}
            if plan["bindings"] != want_bind:
                out.append(f"batch plan bindings {plan['bindings']!r} != {want_bind!r}")
            if ev and not (sp.is_finite(predictions.get("created_unix")) and sp.is_finite(rec.get("launched_unix"))
                           and predictions["created_unix"] < rec["launched_unix"]):
                out.append("predictions were not frozen before the evaluation batch was launched")
            out += mc.batch_problems(rec, {"frozen_unix": plan["created_unix"], "commands": cmds},
                                     _sha_file(run_dir / PLAN_FILE), run_dir,
                                     {"name": spec["name"], "budget_s": spec["budget"]["deadline_s"]})
            # M5 requires a fully successful batch; the M3 helper also accepts a recorded final nonzero exit
            exits = [(e.get("session"), e.get("returncode"), e.get("killed")) for e in rec.get("sessions") or []]
            if rec.get("execution_status") != "succeeded" or rec.get("stop") is not None or rec.get("not_launched") \
                    or [x[0] for x in exits] != [c["session"] for c in cmds] \
                    or any(type(rc_) is not int or rc_ != 0 or killed is not False for _s, rc_, killed in exits):
                out.append(f"batch did not succeed: status {rec.get('execution_status')!r}, stop {rec.get('stop')!r}, "
                           f"exits {exits}")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
            out.append(f"malformed batch plan or record ({type(e).__name__}: {e})")
    idents = {}
    for sid in spec["schedule"]["sessions"]:
        p = run_dir / f"session_{sid}.json"
        if not p.is_file():
            continue
        sess = _load(p, out, p.name)
        if sess is None:
            continue
        try:
            idents[sid] = _check_session(sid, sess, spec, predictions, artifact, require_cuda, out, strict_out)
            sessions[sid] = sess
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:  # fail-closed backstop
            out.append(f"session {sid}: malformed ({type(e).__name__}: {e})")
    if len({json.dumps(v, sort_keys=True) for v in idents.values()}) > 1:
        out.append("sessions differ in namespace, source identity, model files, packages or device")
    if spec["split"] == "evaluation":
        try:
            ns_hex = next(iter(idents.values()))["namespace"] if idents else None
            out += prediction_problems(spec, predictions, artifact, ns_hex)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:  # fail-closed backstop
            out.append(f"malformed predictions or cost model ({type(e).__name__}: {e})")
    exit_ = 0 if not out and not strict_out else 4 if not out else 3
    return {"exit": exit_, "valid": exit_ == 0, "problems": out, "strict_problems": strict_out,
            "sessions": sessions if exit_ == 0 else None}
