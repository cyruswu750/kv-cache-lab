"""M4-B session driver (docs/M4B_PROTOCOL.md §6-§8). Device-agnostic: CUDA in the evaluation
sessions, CPU with a tiny random model in the tests. CPU timings are not performance evidence.

One session = one process. It runs:
1. the process warmup: the four warmup traces, every configuration interleaved, with throwaway
   instances. The records are kept as evidence and never enter a statistic;
2. the four evaluation traces in a seeded order. For each trace there are fresh instances of
   C0, C1-PA(W), C1-PA(W/2), C2-base and C2-batch, and every request runs through all five
   back to back, in a seeded order per request (kvlab.m4_spec.config_order);
3. after each trace's timed replay (untimed): the strict check. Every configuration's
   first-token logits and token are compared with a ResidentReference replay of that
   configuration's own restore lengths and stored chunks (D12). The comparison is recorded as
   SHA-256 digests of the exact float32 logit bytes, plus the tokens. The validator recomputes
   the equality; no stored flag is trusted.

Timed region (D22): synchronize (CUDA); start; config.prefill(tokens) (restore, suffix prefill,
admission); argmax on the device; .item() = end. Logits are copied to the host only after the
end. Outside the region, per C2 request: the warm-disk preparation (below); memory readouts; records;
checks.

Warm-disk preparation, chosen by the spec (kvlab.m4_spec.warm_disk_protocol):
- "whole_tier" (M4_EVAL_V1, unchanged): `_residency`, a mincore check of every chunk file of that
  instance's disk tier, warming and re-measuring the files below 1.0 (as M2/M3 did). It runs before the
  C1 occupancy and allocator readouts.
- "request_candidates" (M4_EVAL_V2): kvlab.warm_disk.prepare. It plans the files this request's restore
  will read, observes them, prewarms each at most once, and observes them once more. It runs after the
  readouts, so that only its closing clock read and return, the forward-call count, the wall-clock read
  and the synchronize separate its final pass from the timer start. After the end (untimed),
  kvlab.warm_disk.complete adds the outcome, phase labels and prewarm count, the interval from the final
  pass to the timer start, and a second counter snapshot. A post-request page observation is not taken:
  it could not tell a cache hit from pages brought in by the read itself. The session report also gets
  counter snapshots at its start and end (the drop_pagecache rule; docs/M4_EVAL_V2_PROTOCOL.md).

Nothing is caught: any exception (including CUDA errors and OOM) propagates. The session then
exits nonzero, the batch stops (D31) and the evidence is incomplete (exit 3). Fault counters
are recorded cumulatively after every request; the validator rejects any unintended fault.
"""

import hashlib
import shutil
import time
from pathlib import Path

import torch

from kvlab import chunk_format as cf
from kvlab import m4_spec as ms
from kvlab import warm_disk as wd
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.prefix_aware_cache import PrefixAwareGPUPrefixCache
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import TieredKVCache
from kvlab.trace_replay import GPUOnly, NoCache, Tiered, cache_storage_bytes

SESSION_FORMAT = "m4-session-v1"


def logits_digest(logits: torch.Tensor) -> str:
    t = logits.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def resident_fraction(path) -> float:
    from kvlab.perf_harness import resident_fraction as rf
    return rf(path)


def warm(path) -> None:
    from kvlab.perf_harness import warm_page_cache
    warm_page_cache(path)


class Instances:
    """Fresh configuration instances for one trace, with their own disk directories."""

    def __init__(self, model, ns, spec, b: dict, disk_root: Path, pin: bool, disk_write_fn=None):
        self.ns, self.payload = ns, cf.payload_len(ns)
        self.file_size = cf.file_size(ns)
        self.dirs, self.c2 = {}, {}
        dev = next(model.parameters()).device
        self.cfg = {"C0": NoCache(model),
                    "C1PA_W": GPUOnly(model, PrefixAwareGPUPrefixCache(ns, b["C1PA_W"] * self.payload, dev)),
                    "C1PA_HALF": GPUOnly(model, PrefixAwareGPUPrefixCache(ns, b["C1PA_HALF"] * self.payload, dev))}
        for name, backend in (("C2_base", "baseline"), ("C2_batch", "batched")):
            d = Path(disk_root) / name
            if d.exists():
                raise FileExistsError(f"{d} exists; every trace starts with an empty disk tier")
            d.mkdir(parents=True)
            self.dirs[name] = d
            host = HostStore((b["host_slots"] + 1) * self.payload, self.payload, max_entries=b["host_slots"], pin=pin)
            kw = {} if disk_write_fn is None else {"write_fn": disk_write_fn}
            disk = DiskStore(d, b["disk_files"] * self.file_size, max_entries=b["disk_files"], **kw)
            t = TieredKVCache(ns, host, disk, restore_backend=backend,
                              restore_staging_chunks=spec["staging_chunks"])
            self.c2[name] = t
            self.cfg[name] = Tiered(AlwaysRestoreEngine(model, t))

    def c1_retained(self) -> dict:
        return {n: self.cfg[n].cache.retained_bytes() for n in ("C1PA_W", "C1PA_HALF")}

    def close(self) -> list:
        removed = []
        for name, t in self.c2.items():
            t.disk.clear()
            t.host.close()
            shutil.rmtree(self.dirs[name])
            removed.append(str(self.dirs[name]))
        return removed


def _c2_state(t: TieredKVCache) -> dict:
    d, h = t.disk, t.host
    c = t.counters.to_dict()
    return {"counters": c, "host_evictions": h.evictions, "host_entries": len(h), "host_slab_bytes": h.slab_bytes,
            "host_registered_bytes": h.registered_bytes, "host_resident_logical_bytes": h.resident_bytes,
            "disk_entries": len(d), "disk_evictions": d.evictions, "disk_used_bytes": d.used_bytes,
            "disk_peak_used_bytes": d.peak_bytes, "disk_tmp_bytes": d.tmp_bytes,
            "disk_undeletable_bytes": d.undeletable_bytes, "disk_unlink_failures": dict(d.unlink_failures),
            "disk_stale_removed": d.stale_removed, "last_restore": t.last_restore}


def _c1_state(cache) -> dict:
    return {"counters": dict(cache.counters), "entries": len(cache.entries), "retained_bytes": cache.retained_bytes(),
            "peak_retained_bytes": cache.peak_retained_bytes, "budget_bytes": cache.budget_bytes,
            "capacity_chunks": cache.capacity_chunks}


def _residency(t: TieredKVCache) -> dict:
    files = [t.disk._final_path(k) for k in t.disk.entries]
    before = [resident_fraction(f) for f in files]
    warmed = [i for i, fr in enumerate(before) if fr < 1.0]
    for i in warmed:
        warm(files[i])
    after = [resident_fraction(f) for f in files] if warmed else before
    return {"files": len(files), "min_fraction_before": min(before, default=1.0),
            "prewarmed_files": len(warmed), "resident_fraction_before": min(after, default=1.0)}


def _traffic(name: str, L: int, stored: list, C: int, payload: int, c2_delta: dict | None) -> dict:
    nr, ns_ = L // C, len(stored)
    if name == "C0":
        return {"d2d_gather": 0, "d2d_constructor": 0, "d2d_admission": 0, "h2d": 0, "d2d_placement": 0,
                "d2h_admission": 0, "disk_read": 0, "disk_written": 0}
    if name.startswith("C1"):
        return {"d2d_gather": nr * payload, "d2d_constructor": nr * payload, "d2d_admission": ns_ * payload, "h2d": 0,
                "d2d_placement": 0, "d2h_admission": 0, "disk_read": 0, "disk_written": 0}
    return {"d2d_gather": 0, "d2d_constructor": nr * payload, "d2d_admission": 0,
            "h2d": c2_delta["bytes_h2d"], "d2d_placement": nr * payload if name == "C2_batch" else 0,
            "d2h_admission": c2_delta["bytes_d2h"], "disk_read": c2_delta["disk_file_bytes_read"],
            "disk_written": c2_delta["disk_file_bytes_written"]}


def replay_trace(model, ns, spec, trace, name: str, role: str, seed: int, b: dict, disk_root: Path,
                 strict: bool, disk_write_fn=None) -> dict:
    """One trace through all five configurations, interleaved per request."""
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    request_scoped = ms.warm_disk_protocol(spec) == wd.PROTOCOL
    C, payload = ns.chunk_size, cf.payload_len(ns)
    inst = Instances(model, ns, spec, b, disk_root, pin=cuda, disk_write_fn=disk_write_fn)
    calls = []
    hook = model.register_forward_pre_hook(lambda *a: calls.append(1))
    records = []
    try:
        for idx, r in enumerate(trace.requests):
            tokens = list(r.tokens)
            for pos, cname in enumerate(ms.config_order(spec, seed, name, role, idx)):
                cfg = inst.cfg[cname]
                c2t = inst.c2.get(cname)
                c2_before = c2t.counters.to_dict() if c2t is not None else None
                residency = _residency(c2t) if c2t is not None and not request_scoped else None
                c1_before = inst.c1_retained()
                mem = None
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    mem = {"A0_allocated": torch.cuda.memory_allocated(), "persistent_c1_retained_at_A0": dict(c1_before)}
                if c2t is not None and request_scoped:
                    residency = wd.prepare(c2t, tokens)  # the last preparation step before the timer
                n0 = len(calls)
                t_wall = time.time()
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                p = cfg.prefill(tokens)
                token = int(torch.argmax(p.logits))  # .item(): the end of the timed region
                t1 = time.perf_counter_ns()
                if cuda:
                    mem["P_max_allocated"] = torch.cuda.max_memory_allocated()
                    mem["note"] = "process-global allocator figures (all co-resident configurations); not per configuration"
                if c2t is not None and request_scoped:  # untimed, after the end
                    wd.complete(residency, t0, wd.counters())
                fwd = len(calls) - n0
                digest = logits_digest(p.logits)
                request_kv = cache_storage_bytes(p.cache)
                p.cache = None
                c1_after = inst.c1_retained()
                rec = {"index": idx, "request_id": r.id, "config": cname, "pos": pos, "t_wall": t_wall,
                       "latency_ns": t1 - t0, "token": token, "logits_sha256": digest, "restore_len": p.restore_len,
                       "sources": list(p.sources), "stored": list(p.stored_chunks), "forward_calls": fwd,
                       "request_kv_bytes": request_kv, "restore_scratch_bytes": p.restore_scratch_bytes,
                       "c1_retained_before": c1_before, "c1_retained_after": c1_after, "memory": mem,
                       "residency": residency}
                if c2t is not None:
                    st = _c2_state(c2t)
                    delta = {k: st["counters"][k] - c2_before[k] for k in
                             ("bytes_h2d", "bytes_d2h", "disk_file_bytes_read", "disk_file_bytes_written")}
                    rec["c2"] = st
                    rec["traffic"] = _traffic(cname, p.restore_len, p.stored_chunks, C, payload, delta)
                elif cname.startswith("C1"):
                    rec["c1"] = _c1_state(cfg.cache)
                    rec["traffic"] = _traffic(cname, p.restore_len, p.stored_chunks, C, payload, None)
                else:
                    rec["traffic"] = _traffic(cname, 0, [], C, payload, None)
                records.append(rec)
    finally:
        hook.remove()
    if strict:
        by_cfg = {}
        for rec in records:
            by_cfg.setdefault(rec["config"], []).append(rec)
        for cname, recs in by_cfg.items():
            ref = ResidentReference(model, ns.digest(), C)
            for r, rec in zip(trace.requests, recs, strict=True):
                toks, steps = ref.run(list(r.tokens), rec["restore_len"], rec["stored"], spec["steps"])
                rec["ref_token"], rec["ref_logits_sha256"] = toks[0], logits_digest(steps[0])
            del ref
    removed = inst.close()
    return {"name": name, "role": role, "seed": spec["seeds"]["warmup" if role == "warmup" else role][name],
            "trace_sha256": trace.digest, "n_requests": len(trace.requests), "budgets": b,
            "strict_checked": strict, "disk_dirs_removed": removed, "records": records}


def run_session(model, ns, spec: dict, session: str, seed: int, disk_root: Path, *, source: dict,
                environment: dict, disk_write_fn=None) -> dict:
    """The whole session; returns the report (written by scripts/m4_measure.py)."""
    if spec["sessions"].get(session) != seed:
        raise ValueError(f"session {session!r} with seed {seed} is not in spec {spec['name']}")
    disk_root = Path(disk_root)
    if disk_root.exists():
        raise FileExistsError(f"{disk_root} exists; a session starts with no disk-tier state")
    request_scoped = ms.warm_disk_protocol(spec) == wd.PROTOCOL
    interference = {"start": wd.counters()} if request_scoped else None
    t_start = time.time()
    warm_traces = {t: ms.trace(spec, t, "warmup") for t in ms.TRACE_NAMES}
    warm_b = ms.budgets(spec, "warmup", warm_traces)
    warmup = [replay_trace(model, ns, spec, warm_traces[t], t, "warmup", seed, warm_b[t], disk_root / "warmup" / t,
                           strict=False, disk_write_fn=disk_write_fn) for t in ms.TRACE_NAMES]
    ev_traces = {t: ms.trace(spec, t, "evaluation") for t in ms.TRACE_NAMES}
    ev_b = ms.budgets(spec, "evaluation", ev_traces)
    order = ms.trace_order(spec, seed)
    evaluation = [replay_trace(model, ns, spec, ev_traces[t], t, "evaluation", seed, ev_b[t],
                               disk_root / "evaluation" / t, strict=True, disk_write_fn=disk_write_fn) for t in order]
    shutil.rmtree(disk_root)
    if request_scoped:
        interference["end"] = wd.counters()
    mism = sum(1 for blk in evaluation for r in blk["records"]
               if r["ref_logits_sha256"] != r["logits_sha256"] or r["ref_token"] != r["token"])
    return {"format": SESSION_FORMAT, "strict_mismatches": mism, "spec": spec["name"], "spec_sha256": spec["spec_sha256"], "session": session,
            "seed": seed, "trace_order": order, "device": str(next(model.parameters()).device),
            "namespace_sha256": ns.digest().hex(), "chunk_size": ns.chunk_size,
            "payload_bytes": cf.payload_len(ns), "file_bytes": cf.file_size(ns), "source": source,
            "environment": environment, "started_unix": t_start, "ended_unix": time.time(),
            "warmup": warmup, "evaluation": evaluation, "finished": True,
            **({"interference": interference} if request_scoped else {})}
