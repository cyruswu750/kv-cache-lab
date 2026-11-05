"""M5b: the candidate and policy measurement session (docs/M5_PROTOCOL_DRAFT.md §5-§7).

For each case, in the session's seeded order:
1. **Setup (untimed):**
   - one bank (`m5_state.ChunkBank`: one full prefill of the declared prefix);
   - fresh stores prepared from it (`m5_state.CaseState`), with a structure check and a content check;
   - a `ResidentReference` seeded with the bank's bytes.
2. **The samples:** warmup repetitions, then measured repetitions. Each repetition runs every run label of
   the case once, in the seeded order: every candidate through `FixedChunks`, plus, in evaluation, the
   `CostAware`, `AlwaysRestore` and `Recompute` policy runs. One sample:
   - (a) structure check (the fingerprint must equal the setup's), then the policy is attached;
   - (b) counters before, then the common page-cache preparation (`kvlab.warm_disk.prepare`: the disk files of
     the maximal feasible prefix, whatever the policy chooses);
   - (c) THE TIMED REGION: synchronize, t0, `engine.prefill_request(prompt)`, `int(argmax(logits))`, t1, with
     the collector disabled. Nothing else is inside it (no hooks, marks, synchronization, checks or
     reference work);
   - (d) untimed: `warm_disk.complete`, counter deltas, the strict check of this sample's ACTUAL split (the
     restored bytes against the bank, and the first-token logits bitwise against the reference for the same
     split, computed once per split and session), the release of outputs and the reset. The garbage collector
     runs once after every repetition (and case), as in the M2 harness.
3. **The end of the case:** a content check, then teardown.

Nothing is caught. A state, strict-check or programming error ends the session. The script writes what was
recorded, with the error, and exits nonzero. Exactly-one-prefill is verified by the CPU tests and the GPU
correctness gate rather than by a hook inside timed samples.
"""

import gc
import time

import torch

from kvlab import m5_spec as sp
from kvlab import warm_disk as wd
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.identity import canonical_json
from kvlab.m5_state import CaseState, ChunkBank
from kvlab.m5_spec import SESSION_FORMAT, batch_plan  # noqa: F401  (re-exported for the scripts)
from kvlab.restore_policy import AlwaysRestore, CostAware, FixedChunks, Recompute

_NUMERIC_COUNTERS = ("requests", "restored_requests", "full_recompute_requests", "host_hits", "disk_hits",
                     "chunk_misses", "recovered_prefix_tokens", "chunks_stored", "host_spills_to_disk", "bytes_d2h",
                     "bytes_h2d", "disk_file_bytes_read", "disk_file_bytes_written")
_COUNTER_MAPS = ("miss_reasons", "write_skips", "spill_failures")


def counter_delta(before: dict, after: dict) -> dict:
    out = {k: after[k] - before[k] for k in _NUMERIC_COUNTERS}
    for k in _COUNTER_MAPS:
        keys = set(before[k]) | set(after[k])
        out[k] = {x: after[k].get(x, 0) - before[k].get(x, 0) for x in sorted(keys)
                  if after[k].get(x, 0) != before[k].get(x, 0)}
    return out


def policy_for(label: str, C: int, cost_model=None):
    kind, v = sp.parse_label(label)
    if kind == "candidate":
        return FixedChunks(v // C)
    if v == "always_restore":
        return AlwaysRestore()
    if v == "recompute":
        return Recompute()
    if cost_model is None:
        raise RuntimeError("a cost_aware run needs the frozen cost model")
    return CostAware(cost_model)


def timed_request(engine, prompt, *, clock, sync):
    """The timed region, and nothing else."""
    gc.disable()
    try:
        sync()
        t0 = clock()
        p = engine.prefill_request(prompt)
        token = int(torch.argmax(p.logits))
        t1 = clock()
    finally:
        gc.enable()
    return p, token, t0, t1


def run_session(model, ns, spec: dict, session: str, scratch, *, out: dict, pin: bool, cost_model=None,
                predictions=None, clock=time.perf_counter_ns, sync=None, wall=time.time,
                observe=wd.resident_pages, read=wd.read_file, read_counters=wd.counters, collect=gc.collect,
                log=None) -> dict:
    """Run one session, filling `out` progressively (so a failure keeps what was recorded)."""
    device = next(model.parameters()).device
    sync = sync or ((lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None))
    log = log or (lambda *_a: None)
    if spec["split"] == "evaluation" and (cost_model is None or predictions is None):
        raise RuntimeError("an evaluation session needs the frozen cost model and predictions")
    C = spec["chunk_size"]
    out.update(format=SESSION_FORMAT, spec_name=spec["name"], spec_sha256=sp.digest(spec), session=session,
               session_seed=spec["schedule"]["sessions"][session], status="running",
               namespace={"fields": ns.fields(), "digest": ns.digest().hex()},
               device=str(device), pinned_host=pin, started_unix=wall(), cases=[], samples=[],
               cost_model_digest=None if cost_model is None else cost_model.digest,
               predictions_sha256=None if predictions is None else predictions["sha256"])
    by_id = sp.case_by_id(spec)
    seq = 0
    sched = sp.schedule(spec, session)
    for cid in sp.case_order(spec, session):
        case = by_id[cid]
        tokens = sp.prompt_tokens(case["prompt"])
        if sp.tokens_sha256(tokens) != case["prompt"]["sha256"]:
            raise RuntimeError(f"{cid}: prompt digest mismatch")
        bank = ChunkBank(model, ns, tokens, case["source"]["P"])
        state = CaseState(ns, case, spec, bank, scratch / cid, pin=pin)
        crec = {"case_id": cid, "bank": bank.record(), "setup_structure": state.setup,
                "setup_fingerprint": state.fingerprint, "setup_content": state.content_check()}
        out["cases"].append(crec)
        if not crec["setup_content"]["ok"]:
            raise RuntimeError(f"{cid}: prepared content differs from the bank")
        ref = bank.reference(model)
        ref_logits = {}
        engine = AlwaysRestoreEngine(model, state.tiered)
        log(f"case {cid}: {len(case['candidates'])} candidates")
        n_runs = len(sp.run_labels(spec, case))
        for (s_cid, phase, rep, pos, label) in [x for x in sched if x[0] == cid]:
            rec = {"seq": seq, "case_id": cid, "phase": phase, "rep": rep, "pos": pos, "run": label}
            rec["fingerprint"] = state.check()                     # (a) raises on any difference
            engine.policy = policy_for(label, C, cost_model)
            c_before = state.tiered.counters.to_dict()             # (b)
            wrec = wd.prepare(state.tiered, tokens, observe=observe, read=read, clock=clock,
                              read_counters=read_counters)
            rec["t_wall"] = wall()
            p, token, t0, t1 = timed_request(engine, tokens, clock=clock, sync=sync)   # (c)
            rec["page_cache"] = wd.complete(wrec, t0, read_counters())               # (d) untimed from here
            rec.update(t0_ns=t0, t1_ns=t1, latency_ns=t1 - t0, first_token=token)
            rec["counters_delta"] = counter_delta(c_before, state.tiered.counters.to_dict())
            d = p.decision
            rec.update(policy=d.policy, selected_len=d.selected_len, actual_len=d.actual_len,
                       sources=list(d.actual_sources), restore_miss_reason=d.restore_miss_reason,
                       miss_reason=p.miss_reason, stored_chunks=list(p.stored_chunks), restore_len=p.restore_len,
                       decision={"reason": d.reason, "plan": d.plan.to_dict(),
                                 "objective_ms": None if d.objective_ms is None else [list(x) for x in d.objective_ms]})
            L = p.restore_len
            strict = {"split": L}
            if L:
                strict["restored_bytes"] = ref.check_restored(tokens, p.cache, L)
            if L not in ref_logits:
                ref_tokens, ref_steps = ref.run(tokens, L, [], steps=1)  # reference of the same split, untimed
                ref_logits[L] = (ref_steps[0], ref_tokens[0])
            got = p.logits.detach().float().cpu()
            strict["logits_equal"] = bool(torch.equal(got, ref_logits[L][0]))
            strict["reference_token"] = ref_logits[L][1]
            rec["strict"] = strict
            p.cache = None
            del p, got
            if strict["logits_equal"] is not True or (L and strict["restored_bytes"]["all_bytes_equal"] is not True):
                out["samples"].append(rec)
                raise RuntimeError(f"strict check failed: {cid} {label} split {L}: {strict}")
            state.reset(rec["stored_chunks"])
            out["samples"].append(rec)
            seq += 1
            if pos == n_runs - 1:
                collect()  # once per repetition (as the M2 harness), never inside a timed region
        crec["end_content"] = state.content_check()
        crec["end_fingerprint"] = state.check()
        del ref, engine
        state.close()
        del state, bank
        collect()
        if not crec["end_content"]["ok"]:
            raise RuntimeError(f"{cid}: content changed during the case")
    out.update(status="complete", ended_unix=wall())
    return out


def dumps(obj) -> str:
    """Session/evidence JSON: canonical values (no NaN/Infinity), indented for reading."""
    import json
    canonical_json(obj)  # raises on non-finite floats or unsupported types
    return json.dumps(obj, indent=1, sort_keys=True)
