"""M5 correctness-gate bodies, shared by the GPU gate (tests/gpu/test_m5_policy.py: the real pinned model, cuda)
and its CPU variant (tests/test_m5_gate_cpu.py: a tiny random model).

Every body takes (model, ns, root, ...) and derives its prompt lengths from the chunk size, so the same checks run
at C = 256 on the GPU and at C = 4 on the CPU. The checks:
- forced lengths 0, intermediate and maximal on a mixed disk-then-host prefix, with a long suffix (S > C);
- CostAware wiring with a SYNTHETIC artifact (labelled correctness-only; never calibration or performance);
- truncation on corruption or a missing file inside the selected prefix, and no read beyond it;
- restored bytes and a matching-provenance ResidentReference replay of the actual length and stored chunks;
- exactly one prefill and steps - 1 decode calls;
- leases, staging and budgets; restored-tensor ownership;
- the M5 harness session itself (bank, state reset, page cache, strict checks), validated by the session
  validator.
"""

import copy
import time
from pathlib import Path

import torch

from kvlab import chunk_format as cf
from kvlab import m5_bench as mb
from kvlab import m5_fit as mf
from kvlab import m5_spec as sp
from kvlab import m5_validate as mv
from kvlab import restore_cost as rc
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import chain_keys
from kvlab.resident_reference import ResidentReference
from kvlab.restore_policy import CostAware, FixedChunks
from kvlab.tiered_cache import TieredKVCache
from m5_toy import artifact, coefficients, compat, domain, redigest

STEPS = 4
SYNTHETIC_PROVENANCE = {"synthetic": "GPU correctness gate only; never calibration or performance evidence"}


class GateRig:
    def __init__(self, model, ns, root, *, backend, host_slots, disk_files=32, group=2, pin):
        self.model, self.ns, self.C = model, ns, ns.chunk_size
        cb, self.file = cf.payload_len(ns), cf.file_size(ns)
        self.host = HostStore((host_slots + 1) * cb, cb, max_entries=64, pin=pin)
        self.disk = DiskStore(Path(root), disk_files * self.file, max_entries=64)
        self.tiered = TieredKVCache(ns, self.host, self.disk, restore_backend=backend, restore_staging_chunks=group)
        self.engine = AlwaysRestoreEngine(model, self.tiered)
        self.ref = ResidentReference(model, ns.digest(), self.C)

    def keys(self, prompt):
        return chain_keys(self.ns.digest(), prompt, self.C)

    def tiers(self, prompt):
        return [self.tiered.tier_of(k) for k in self.keys(prompt)]

    def path(self, prompt, i):
        return self.disk.root / f"{self.keys(prompt)[i].hex()}.kvc"

    def run(self, prompt, policy=None):
        self.engine.policy = policy
        calls, restored = [], []
        hook = self.model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = self.engine.run(prompt, STEPS, on_restored=lambda cache, L: restored.append(
                (self.ref.check_restored(prompt, cache, L), _owns_exactly(cache, L))))
        finally:
            hook.remove()
        assert len(calls) == STEPS, f"{len(calls)} forward calls: expected one prefill and {STEPS - 1} decodes"
        assert len(restored) == (r.restore_len > 0)
        assert all(chk["all_bytes_equal"] and owns for chk, owns in restored)
        ref_tokens, ref_steps = self.ref.run(prompt, r.restore_len, r.stored_chunks, STEPS)   # actual L, actual stores
        assert r.tokens == ref_tokens and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(r.step_logits, ref_steps))
        assert all(e.pins == 0 for e in self.host.entries.values()) and not self.host._staging_busy
        assert self.disk.tmp_bytes == 0 and self.disk.scan_dir_bytes()[0] <= self.disk.used_bytes <= self.disk.budget_bytes
        assert len(self.host.entries) <= self.host.slots
        return r

    def close(self):
        self.host.close()


def _owns_exactly(cache, L) -> bool:
    return all(x.shape[2] == L and x.untyped_storage().nbytes() == x.numel() * x.element_size()
               for layer in cache.layers for x in (layer.keys, layer.values))


def prompt(n_chunks, C, extra=3, start=10):
    return list(range(start, start + n_chunks * C + extra))


def check_forced_lengths(model, ns, root, backend, pin):
    C = ns.chunk_size
    rig = GateRig(model, ns, root, backend=backend, host_slots=3, pin=pin)
    P = prompt(6, C)
    seed = rig.run(P)
    assert seed.restore_len == 0 and seed.stored_chunks == list(range(6))
    assert rig.tiers(P) == ["disk"] * 3 + ["host"] * 3  # LRU spills the oldest chunks: disk then host
    for k in (0, 1, 3, 6):                      # zero, intermediate (k = 1: a suffix of 5C + 3 > C), maximal
        r = rig.run(P, FixedChunks(k))
        d = r.decision
        assert (d.selected_len, d.actual_len, r.restore_len, d.truncated) == (k * C, k * C, k * C, False)
        assert r.sources == (["disk"] * 3 + ["host"] * 3)[:k] and r.stored_chunks == []
    rig.close()


def synthetic_model(ns, backend, group, coef):
    art = artifact(coef, dom=domain(prompt=(1, 100_000), suffix=(1, 100_000)),
                   comp=compat(ns=ns.digest().hex(), C=ns.chunk_size, backend=backend,
                               group=group if backend == "batched" else None))
    art["provenance"] = dict(SYNTHETIC_PROVENANCE)
    return rc.RestoreCostModel(redigest(art))


def check_cost_aware(model, ns, root, backend, pin):
    C = ns.chunk_size
    rig = GateRig(model, ns, root, backend=backend, host_slots=1, pin=pin)
    P = prompt(3, C)
    rig.run(P)
    assert rig.tiers(P) == ["disk", "disk", "host"]
    disk_heavy = synthetic_model(ns, backend, 2, coefficients(full=(50.0, 0, 0), partial=(1.0, 0, 100.0, 0, 0)))
    r = rig.run(P, CostAware(disk_heavy))
    assert (r.decision.selected_len, r.restore_len, r.decision.reason) == (0, 0, "min_predicted_objective")
    flat = synthetic_model(ns, backend, 2, coefficients(full=(50.0, 0, 0), partial=(1.0, 0, 0, 0, 0)))
    r = rig.run(P, CostAware(flat))
    assert (r.decision.selected_len, r.restore_len) == (3 * C, 3 * C) and r.sources == ["disk", "disk", "host"]
    rig.close()


def check_truncation_and_no_read_beyond(model, ns, root, backend, pin):
    C = ns.chunk_size
    rig = GateRig(model, ns, Path(root) / "t", backend=backend, host_slots=1, pin=pin)
    P = prompt(3, C)
    rig.run(P)                                  # disk, disk, host
    raw = bytearray(rig.path(P, 1).read_bytes())
    raw[-6] ^= 0x40
    rig.path(P, 1).write_bytes(raw)
    reads, real = [], rig.disk.read_into

    def spy(key, *a):
        reads.append(key)
        return real(key, *a)
    rig.disk.read_into = spy
    r = rig.run(P, FixedChunks(1))               # the bad chunk 1 is beyond the selection: never read
    assert reads == [rig.keys(P)[0]] and r.restore_len == C and rig.path(P, 1).read_bytes() == bytes(raw)
    reads.clear()
    r = rig.run(P, FixedChunks(3))               # inside the selection: truncation to the verified prefix
    d = r.decision
    assert (d.selected_len, d.actual_len, d.truncated, d.restore_miss_reason) == (3 * C, C, True, "disk_payload_crc")
    assert r.stored_chunks == [1]
    rig.close()
    rig = GateRig(model, ns, Path(root) / "m", backend=backend, host_slots=1, pin=pin)
    rig.run(P)
    rig.path(P, 0).unlink()
    r = rig.run(P, FixedChunks(2))
    assert (r.decision.actual_len, r.decision.restore_miss_reason, r.stored_chunks) == (0, "disk_file_missing", [0])
    rig.close()


def mini_specs(C, vocab=sp.VOCAB, model=None):
    """A small calibration-split spec and an evaluation-split spec with the production structure, for the smoke
    sessions (never fitted: check_design=False)."""
    n1, n2, n3 = 2 * C + C // 2 + 1, 4 * C + 1, 3 * C + 3
    cal_cases = sp.grid_cases((("g-01", n1, "DH", 901), ("g-02", n2, "HDH", 902)), "calibration", C, vocab)
    ev_cases = sp.grid_cases((("g-e1", n3, "DH", 911),), "evaluation", C, vocab)
    sched = {"sessions": {"s01": 1}, "warmup_reps": 1, "reps": 1}
    an = {**sp.ANALYSIS, "partition": {"A": ["s01"], "B": []}}
    kw = dict(C_=C, backend="batched", group=2, deadline_s=900, model=model, analysis=an)
    cal = sp.build_spec("GATE_SMOKE_CAL", "calibration", cal_cases, sched, check_design=False, **kw)
    ev = sp.build_spec("GATE_SMOKE_EVAL", "evaluation", ev_cases, sched, calibration=cal, **kw)
    return cal, ev


def check_harness_smoke(model, ns, root, pin, *, vocab=sp.VOCAB, model_meta=None, require_cuda):
    """One calibration-split and one evaluation-split session of the real harness (default clock; timings are
    not evidence). Every recorded sample must pass the session validator."""
    C = ns.chunk_size
    cal, ev = mini_specs(C, vocab, model_meta)
    identity = {"source": {"git_head": "0" * 40, "git_status_porcelain": [], "git_diff_head_sha256": "0" * 64,
                           "files_sha256": {"gate": "0" * 64}},
                "packages": {"gate": "smoke"}, "model_blobs": {"gate": {"blob": "0" * 64, "bytes": 1}},
                "gpu": str(next(model.parameters()).device)}
    for spec in (cal, ev):
        cm = preds = None
        if spec["split"] == "evaluation":
            cm = synthetic_model(ns, "batched", 2, coefficients(full=(50.0, 0, 0), partial=(1.0, 0, 3.0, 0, 0)))
            preds = mf.make_predictions(spec, cm, created_unix=time.time())
        out = copy.deepcopy(identity)
        mb.run_session(model, ns, spec, "s01", Path(root) / spec["name"], out=out, pin=pin, cost_model=cm,
                       predictions=preds)
        problems, strict = [], []
        art = None if cm is None else copy.deepcopy(cm.artifact)
        mv._check_session("s01", out, spec, preds, art, require_cuda, problems, strict)
        assert problems == [] and strict == [], (problems[:5], strict[:5])
        assert out["status"] == "complete" and len(out["samples"]) == len(sp.schedule(spec, "s01"))
