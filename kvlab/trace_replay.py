"""Replay one synthetic trace through one configuration (M4-A; docs/design.md §12).

Configurations (the same model, trace, first-token rule and decode length):
- **C0 `NoCache`:** full prefill of every prompt. No chain keys, lookups or admission
  (the `full_nocache` semantics of D28).
- **C1 `GPUOnly`:** the bounded device-resident prefix cache (`kvlab.gpu_prefix_cache`).
- **C2 `Tiered`:** the existing synchronous always-restore engine (`kvlab.always_restore`),
  unchanged. It is driven through its public `prefill_request()` followed by
  `greedy_continue()`, exactly as `AlwaysRestoreEngine.run()` does, so that the request's KV can
  be measured before it is dropped. The restore backend is whatever the given `TieredKVCache`
  was built with: `baseline` by default, `batched` only if selected explicitly.

Per request, the replay records the restore length and sources, the stored chunk indices
(their provenance for the strict check), the number of model forward calls, the first-token
logits and greedy tokens, and separate resource terms. The terms are never summed, and the
configurations do not use equal total resources:
- `persistent_device_cache_bytes`: KV kept on the model's device between requests (C1 only);
- `request_kv_bytes`: storage of the request's own prompt KV after prefill (all configurations);
- `restore_scratch_bytes`: temporary device bytes for assembling a restored prefix: C1's
  gather, or C2's destination + staging (`TieredKVCache.last_restore`). The cache built by the
  D4 constructor copy is the request's KV;
- `host_slab_bytes` / `host_resident_logical_bytes`: C2's host tier (allocated once; logical
  occupancy inside it);
- `disk_used_bytes` / `disk_peak_used_bytes`: C2's disk tier. The OS page cache serving warm
  reads is not counted anywhere.
Activation memory and allocator-rounded device memory are not measured here; they need the GPU
(deferred to M4-B).

On CPU, a replay is a correctness test. Its timings, which are not recorded, would not be GPU
performance evidence.
"""

import hashlib
from dataclasses import dataclass, field

import torch

from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.boundaries import max_restore_len
from kvlab.gpu_prefix_cache import GPUPrefixCache, same_device
from kvlab.identity import chain_keys
from kvlab.model import greedy_continue, prefill
from kvlab.resident_reference import ResidentReference


def cache_storage_bytes(cache) -> int:
    """Bytes of the distinct storages behind a DynamicCache's K/V tensors."""
    storages = {}
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            s = t.untyped_storage()
            storages[s.data_ptr()] = s.nbytes()
    return sum(storages.values())


@dataclass
class Prefilled:
    logits: torch.Tensor
    cache: object
    restore_len: int
    sources: list
    stored_chunks: list
    restore_scratch_bytes: int


class NoCache:
    name = "C0_no_cache"

    def __init__(self, model):
        self.model, self.device = model, next(model.parameters()).device

    def prefill(self, token_ids) -> Prefilled:
        ids = torch.tensor([list(token_ids)], device=self.device)
        logits, cache = prefill(self.model, ids, None, 0)
        return Prefilled(logits, cache, 0, [], [], 0)

    def resources(self) -> dict:
        return {"persistent_device_cache_bytes": 0, "host_slab_bytes": 0, "host_resident_logical_bytes": 0,
                "disk_used_bytes": 0, "disk_peak_used_bytes": 0}

    def describe(self) -> dict:
        return {"config": self.name}


class GPUOnly:
    name = "C1_gpu_only"

    def __init__(self, model, cache: GPUPrefixCache):
        self.model, self.cache, self.device = model, cache, next(model.parameters()).device
        if not same_device(cache.device, self.device):
            raise ValueError(f"cache device {cache.device} != model device {self.device}")

    def prefill(self, token_ids) -> Prefilled:
        token_ids = list(token_ids)
        c = self.cache
        ids = torch.tensor([token_ids], device=self.device)
        keys = chain_keys(c.ns_digest, token_ids, c.C)
        kv, L, sources = c.restore(token_ids, keys[:max_restore_len(len(token_ids), c.C) // c.C], self.model.config)
        if L:
            logits, kv = prefill(self.model, ids[:, L:], kv, L)
        else:
            logits, kv = prefill(self.model, ids, None, 0)
        stored = c.store(kv, token_ids, keys)  # the cache holds exactly the N prompt tokens here
        return Prefilled(logits, kv, L, sources, stored, c.last_restore["scratch_bytes"])

    def resources(self) -> dict:
        return {"persistent_device_cache_bytes": self.cache.retained_bytes(),
                "persistent_device_cache_peak_bytes": self.cache.peak_retained_bytes,
                "persistent_device_cache_budget_bytes": self.cache.budget_bytes,
                "host_slab_bytes": 0, "host_resident_logical_bytes": 0, "disk_used_bytes": 0, "disk_peak_used_bytes": 0}

    def describe(self) -> dict:
        c = self.cache
        return {"config": self.name, "budget_bytes": c.budget_bytes, "capacity_chunks": c.capacity_chunks,
                "chunk_bytes": c.chunk_bytes}


class Tiered:
    name = "C2_tiered_always_restore"

    def __init__(self, engine: AlwaysRestoreEngine):
        self.engine, self.model = engine, engine.model

    def prefill(self, token_ids) -> Prefilled:
        p = self.engine.prefill_request(token_ids)
        lr = self.engine.tiered.last_restore
        scratch = (lr["device_destination_bytes"] + lr["device_scratch_bytes"]) if lr else 0
        return Prefilled(p.logits, p.cache, p.restore_len, p.sources, p.stored_chunks, scratch)

    def resources(self) -> dict:
        b = self.engine.tiered.budgets()
        h, d = b["host"], b.get("disk") or {}
        return {"persistent_device_cache_bytes": 0, "host_slab_bytes": h["slab_allocated_bytes"],
                "host_registered_bytes": h["cuda_host_registered_bytes"],
                "host_resident_logical_bytes": h["resident_logical_bytes"],
                "disk_used_bytes": d.get("used_bytes", 0), "disk_peak_used_bytes": d.get("peak_used_bytes", 0)}

    def describe(self) -> dict:
        t = self.engine.tiered
        return {"config": self.name, **t.restore_config(),
                "host_budget_bytes": t.host.budget_bytes,
                "disk_budget_bytes": t.disk.budget_bytes if t.disk is not None else 0}


@dataclass
class ReplayResult:
    config: dict
    trace_sha256: str
    steps: int
    records: list = field(default_factory=list)


def replay(config, trace, steps: int) -> ReplayResult:
    """Replay every request of `trace` in order: prefill (with the configuration's restore and
    admission), then `steps` greedy tokens; the request's KV is then dropped. Exactly one
    prefill and steps − 1 decode forward calls per request are recorded and checked."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model = config.model
    out = ReplayResult(config.describe(), trace.digest, steps)
    calls = []
    hook = model.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        for r in trace.requests:
            n0 = len(calls)
            p = config.prefill(r.tokens)
            request_kv = cache_storage_bytes(p.cache)
            tokens, step_logits = greedy_continue(model, p.cache, p.logits, len(r.tokens), steps)
            p.cache = None
            forward_calls = len(calls) - n0
            if forward_calls != steps:
                raise RuntimeError(f"request {r.id}: {forward_calls} forward calls, expected {steps}")
            out.records.append({
                "id": r.id, "group": r.group, "prompt_len": len(r.tokens),
                "prompt_sha256": hashlib.sha256(repr(r.tokens).encode()).hexdigest(),
                "restore_len": p.restore_len, "sources": p.sources, "stored_chunks": p.stored_chunks,
                "forward_calls": forward_calls, "tokens": tokens, "step_logits": step_logits,
                "request_kv_bytes": request_kv, "restore_scratch_bytes": p.restore_scratch_bytes,
                **config.resources()})
    finally:
        hook.remove()
    return out


def summarize(result: ReplayResult) -> dict:
    """Totals and peaks per resource term; terms are reported separately, never summed."""
    recs = result.records
    peak = lambda k: max((r.get(k, 0) for r in recs), default=0)  # noqa: E731
    return {**result.config, "trace_sha256": result.trace_sha256, "requests": len(recs),
            "restored_requests": sum(1 for r in recs if r["restore_len"]),
            "restored_tokens": sum(r["restore_len"] for r in recs),
            "recomputed_tokens": sum(r["prompt_len"] - r["restore_len"] for r in recs),
            "stored_chunks": sum(len(r["stored_chunks"]) for r in recs),
            "peak": {k: peak(k) for k in ("persistent_device_cache_bytes", "request_kv_bytes", "restore_scratch_bytes",
                                          "host_slab_bytes", "host_resident_logical_bytes", "disk_used_bytes")},
            "note": "resource terms differ by configuration and are not summed; CPU replays are correctness tests only"}


def strict_mismatches(model, ns_digest: bytes, chunk_size: int, trace, result: ReplayResult) -> list:
    """Check a replay against an independent GPU-resident replay (`ResidentReference`) of the
    SAME configuration's decisions: identical restore lengths and stored chunks give identical
    computation provenance, so every step logit and token must be bitwise equal (D12). Requests
    of different configurations may use different splits and are never compared with each
    other bitwise (D10). Returns the ids of mismatching requests."""
    if result.trace_sha256 != trace.digest:
        raise ValueError("the replay was not made from this trace")
    ref = ResidentReference(model, ns_digest, chunk_size)
    bad = []
    for r, rec in zip(trace.requests, result.records, strict=True):
        tokens, steps = ref.run(r.tokens, rec["restore_len"], rec["stored_chunks"], result.steps)
        if tokens != rec["tokens"] or len(steps) != len(rec["step_logits"]) or \
                not all(torch.equal(a, b) for a, b in zip(steps, rec["step_logits"])):
            bad.append(r.id)
    return bad
