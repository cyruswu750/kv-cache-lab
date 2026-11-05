"""M5a: restore planner, opt-in policies and their engine integration (CPU, tiny random Qwen2, temp stores).

- Planner: candidates at legal chunk boundaries, holes, mixed tiers, no staging buffer, and no
  side effects (LRU order, pins, staging, counters, files, last_restore, allocations, inference).
- policy=None: identical to the pre-M5a engine (an adapted copy of its prefill_request below) in
  results, logits, counters, phase marks, forward calls and restore calls; no planning at all.
- With a policy: forced 0 / intermediate / maximal lengths on both restore backends, invalid
  outputs rejected before any restore, validation failures inside the chosen prefix truncating to
  the verified prefix, chunks beyond the chosen prefix never read, exactly one prefill and
  steps - 1 decode calls (also with read and write failures), long suffixes, clean leases/staging/
  budgets.
- Strict checks: restored bytes against a ResidentReference seeded with the same requests, and
  outputs against that reference replaying the *actual* restore length and the *actual* newly
  stored chunks. No comparison across different restore lengths or against full prefill (D10).
"""

import errno
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import restore_cost as rc
from kvlab import tiered_cache as tc
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.boundaries import max_restore_len
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import chain_keys, namespace_for
from kvlab.model import greedy_continue, prefill
from kvlab.resident_reference import ResidentReference
from kvlab.restore_policy import (AlwaysRestore, CostAware, FixedChunks, PolicyDecision, PolicyError, Recompute,
                                  plan_restore, validate_selection)
from kvlab.tiered_cache import RESTORE_BACKENDS, TieredKVCache
from m5_toy import artifact, coefficients, compat, domain

C, STEPS = 4, 4
PROMPT = list(range(10, 23))      # N = 13: complete chunks 0..2, eligible L = 12
LONG = list(range(10, 37))        # N = 27: complete chunks 0..5, eligible L = 24


@pytest.fixture(scope="module")
def tiny_model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


class Rig:
    """One tiered store + engine + a ResidentReference fed the same requests (same provenance)."""

    def __init__(self, model, root, host_slots=8, disk_files=16, backend="baseline", group=2, policy=None,
                 write_fn=cf.default_write, readinto_fn=cf.default_readinto):
        self.model = model
        self.ns = namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)
        self.cb, self.file = cf.payload_len(self.ns), cf.file_size(self.ns)
        self.host = HostStore((host_slots + 1) * self.cb, self.cb, max_entries=64, pin=False)
        self.disk = DiskStore(root, disk_files * self.file, max_entries=64, write_fn=write_fn,
                              readinto_fn=readinto_fn)
        self.tiered = TieredKVCache(self.ns, self.host, self.disk, restore_backend=backend,
                                    restore_staging_chunks=group)
        self.engine = AlwaysRestoreEngine(model, self.tiered, policy=policy)
        self.ref = ResidentReference(model, self.ns.digest(), C)

    def keys(self, prompt):
        return chain_keys(self.ns.digest(), prompt, C)

    def tiers(self, prompt):
        return [self.tiered.tier_of(k) for k in self.keys(prompt)]

    def path(self, prompt, i):
        return self.disk.root / f"{self.keys(prompt)[i].hex()}.kvc"

    def run(self, prompt, policy="keep"):
        """engine.run with strict checks; returns the RequestResult."""
        if policy != "keep":
            self.engine.policy = policy
        calls, restored = [], []
        hook = self.model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = self.engine.run(prompt, STEPS, on_restored=lambda cache, L: restored.append(
                self.ref.check_restored(prompt, cache, L)))
        finally:
            hook.remove()
        assert len(calls) == STEPS, f"{len(calls)} forward calls: inference repeated or cut short"
        assert len(restored) == (r.restore_len > 0) and all(x["all_bytes_equal"] for x in restored)
        ref_tokens, ref_steps = self.ref.run(prompt, r.restore_len, r.stored_chunks, STEPS)  # actual L, actual stores
        assert r.tokens == ref_tokens and all(torch.equal(a, b) for a, b in zip(r.step_logits, ref_steps))
        assert_clean(self.tiered)
        return r


def assert_clean(t):
    h, d = t.host, t.disk
    assert all(e.pins == 0 for e in h.entries.values()) and not h._staging_busy
    assert len(h.entries) <= h.slots and h.slab_bytes <= h.budget_bytes
    assert d.tmp_bytes == 0 and d.scan_dir_bytes()[0] <= d.used_bytes <= d.budget_bytes


def corrupt(path):
    raw = bytearray(path.read_bytes())
    raw[-6] ^= 0x40  # payload byte before the payload CRC
    path.write_bytes(raw)
    return bytes(raw)


# ---- planner: pure logic on a duck-typed store -----------------------------------------------

class FakeTiered:
    def __init__(self, tiers, staging=True, chunk=C, backend="baseline"):
        self.C, self.tiers = chunk, tiers
        self.host = SimpleNamespace(staging_slots=1 if staging else 0)
        self.ns_digest, self.restore_backend, self.restore_staging_chunks = b"\x11" * 32, backend, 4

    def _check_backend(self):
        if self.restore_backend not in RESTORE_BACKENDS:
            raise ValueError("unknown restore backend")

    def tier_of(self, key):
        return self.tiers.get(key)


def fake(N, tiers, **kw):
    keys = [bytes([i]) * 32 for i in range(N // C)]
    return FakeTiered({k: t for k, t in zip(keys, tiers) if t is not None}, **kw), keys


@pytest.mark.parametrize("N", [0, 1, C - 1, C, C + 1, 2 * C, 2 * C + 1, 5 * C + 3])
def test_candidates_follow_the_first_token_rule_when_everything_is_indexed(N):
    t, keys = fake(N, ["host"] * (N // C))
    p = plan_restore(t, keys, N)
    assert p.eligible_len == max_restore_len(N, C)
    assert p.feasible_lengths == tuple(range(0, p.eligible_len + 1, C))
    assert all(L % C == 0 and (L == 0 or L <= N - 1) for L in p.feasible_lengths)
    assert p.limit_reason == ("no_eligible_chunk" if N <= C else "eligible_limit")
    if N == 2 * C:  # exact multiple: the last complete chunk is never a candidate
        assert p.feasible_lengths == (0, C) and p.indexed_tiers == ("host",)


@pytest.mark.parametrize("tiers, staging, indexed, feasible, reason", [
    ([None, "host", "host"], True, (), 0, "absent"),
    (["host", None, "host"], True, ("host",), 1, "absent"),
    (["disk", "disk", "host"], True, ("disk", "disk", "host"), 3, "eligible_limit"),
    (["host", "disk", "host"], False, ("host", "disk", "host"), 1, "no_staging_buffer"),
    (["disk", "host", "host"], False, ("disk", "host", "host"), 0, "no_staging_buffer"),
    (["host", "host", "host"], False, ("host", "host", "host"), 3, "eligible_limit"),
    (["host", None, "disk"], False, ("host",), 1, "absent"),
])
def test_holes_mixed_tiers_and_missing_staging(tiers, staging, indexed, feasible, reason):
    t, keys = fake(13, tiers, staging=staging)
    p = plan_restore(t, keys, 13)
    assert (p.indexed_tiers, p.max_feasible_len // C, p.limit_reason) == (indexed, feasible, reason)
    assert p.feasible_tiers == indexed[:feasible] and p.disk_staging_available is staging
    assert p.to_dict()["presence"] == "indexed_not_validated"
    json.dumps(p.to_dict())


def test_planner_rejects_inconsistent_inputs():
    t, keys = fake(13, ["host"] * 3)
    with pytest.raises(ValueError, match="chunk keys"):
        plan_restore(t, keys[:2], 13)
    with pytest.raises(ValueError, match="prompt_len"):
        plan_restore(t, keys, True)
    t.restore_backend = "fast"
    with pytest.raises(ValueError, match="unknown restore backend"):
        plan_restore(t, keys, 13)


# ---- planner: no side effects on a real store ----------------------------------------------

def snapshot(t, root):
    files = {}
    for name in sorted(os.listdir(root)):
        st = os.lstat(root / name)
        files[name] = (st.st_ino, st.st_size, st.st_mtime_ns, (root / name).read_bytes())
    return {"counters": t.counters.to_dict(), "host_lru": list(t.host.entries),
            "pins": [e.pins for e in t.host.entries.values()], "free": list(t.host._free),
            "staging_busy": t.host._staging_busy, "evictions": (t.host.evictions, t.disk.evictions),
            "disk_lru": list(t.disk.entries.items()), "undeletable": dict(t.disk.undeletable),
            "disk_bytes": (t.disk.committed_bytes, t.disk.tmp_bytes, t.disk.peak_bytes), "files": files,
            "slab": t.host.slab.clone()}


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_planning_has_no_side_effects(tiny_model, tmp_path, monkeypatch, backend):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=2, backend=backend)
    rig.run(LONG)  # 6 chunks stored: 0-3 spilled to disk, 4-5 in host
    corrupt(rig.path(LONG, 1))  # indexed but invalid: planning must still list it (presence, not validity)
    t, keys = rig.tiered, rig.keys(LONG)
    sentinel = {"marker": "previous restore"}
    t.last_restore = sentinel
    before = snapshot(t, rig.disk.root)
    calls = []
    hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))

    def forbidden(*a, **k):
        raise AssertionError("planning touched files, leases, staging, restore tensors or restore()")
    try:
        with monkeypatch.context() as m:
            for name in ("open", "stat", "lstat", "fstat", "scandir", "listdir", "unlink", "replace"):
                m.setattr(os, name, forbidden)
            m.setattr("builtins.open", forbidden)
            m.setattr(t.disk, "read_into", forbidden)
            m.setattr(t.host, "lease", forbidden)
            m.setattr(t.host, "staging", forbidden)
            m.setattr(t.host, "allocate", forbidden)
            m.setattr(t, "restore", forbidden)
            m.setattr(tc, "_BaselineDest", forbidden)
            m.setattr(tc, "_BatchedDest", forbidden)
            m.setattr(tc, "DynamicCache", forbidden)
            p = plan_restore(t, keys, len(LONG))
    finally:
        hook.remove()
    after = snapshot(t, rig.disk.root)
    slab_b, slab_a = before.pop("slab"), after.pop("slab")
    assert after == before and torch.equal(slab_a, slab_b) and t.last_restore is sentinel and not calls
    assert p.indexed_tiers == ("disk",) * 4 + ("host",) * 2 and p.max_feasible_len == 6 * C


def test_planner_agrees_with_the_m4_v2_warm_disk_planner(tiny_model, tmp_path):
    """Independent cross-check: the disk files of the maximal feasible prefix are exactly the files
    kvlab.warm_disk.plan_disk_reads (M4_EVAL_V2, unchanged) says the M1 restore would read."""
    from kvlab.warm_disk import plan_disk_reads
    stop = {"eligible_limit": "all_eligible_present", "no_eligible_chunk": "all_eligible_present",
            "absent": "first_absent", "no_staging_buffer": "no_staging_buffer"}
    rig = Rig(tiny_model, tmp_path / "d", host_slots=2)
    rig.run(LONG)  # disk x4, host x2
    host0 = HostStore(rig.cb - 1, rig.cb, max_entries=4, pin=False)
    stores = [rig.tiered, TieredKVCache(rig.ns, host0, rig.disk)]  # with and without a staging buffer
    prompts = [LONG, LONG[:13], LONG[:8], LONG[:3], list(range(300, 320))]
    rig.host.discard(next(k for k in rig.keys(LONG) if k in rig.host))  # a hole at chunk 4
    for t in stores:
        for prompt in prompts:
            p, w = plan_restore(t, chain_keys(rig.ns.digest(), prompt, C), len(prompt)), plan_disk_reads(t, prompt)
            disk_idx = [i for i, tier in enumerate(p.feasible_tiers) if tier == "disk"]
            assert (p.eligible_len // C, len(p.indexed_tiers), stop[p.limit_reason]) == \
                (w["eligible_chunks"], w["present_chunks"], w["stop"])
            assert disk_idx == [i for i, _ in w["candidates"]] and p.feasible_tiers.count("host") == w["host_chunks"]


# ---- policy outputs ------------------------------------------------------------------------

class Returns:
    name = "returns"

    def __init__(self, value):
        self.value = value

    def choose(self, plan):
        return self.value


@pytest.mark.parametrize("value, match", [
    (PolicyDecision(True, "x"), "integer"),
    (PolicyDecision(4.0, "x"), "integer"),
    (PolicyDecision(np.float64(4), "x"), "integer"),
    (PolicyDecision(-4, "x"), "negative"),
    (PolicyDecision(2, "x"), "multiple of the chunk size"),
    (PolicyDecision(16, "x"), "not a feasible candidate"),
    (PolicyDecision(4, ""), "reason"),
    (PolicyDecision(4, "x", objective_ms=[(4, 1.0)]), "objective_ms"),
    (4, "expected PolicyDecision"),
    (None, "expected PolicyDecision"),
])
def test_invalid_policy_outputs_are_rejected_before_any_restore(tiny_model, tmp_path, monkeypatch, value, match):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=2)
    rig.run(PROMPT)  # chunks 0 disk, 1-2 host
    t = rig.tiered
    before = snapshot(t, rig.disk.root)
    before.pop("slab")
    restores, calls = [], []
    monkeypatch.setattr(t, "restore", lambda *a, **k: restores.append(1))
    hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))
    rig.engine.policy = Returns(value)
    try:
        with pytest.raises(PolicyError, match=match):
            rig.engine.run(PROMPT, STEPS)
    finally:
        hook.remove()
    after = snapshot(t, rig.disk.root)
    after.pop("slab")
    assert not restores and not calls and after == before


def test_selection_beyond_a_hole_is_rejected_before_restore(tiny_model, tmp_path, monkeypatch):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8)
    rig.run(PROMPT)
    rig.host.discard(rig.keys(PROMPT)[1])  # chunk 2 stays indexed, but L = 8 or 12 would need chunk 1
    restores = []
    monkeypatch.setattr(rig.tiered, "restore", lambda *a, **k: restores.append(1))
    for L in (8, 12):
        rig.engine.policy = Returns(PolicyDecision(L, "x"))
        with pytest.raises(PolicyError, match="not a feasible candidate"):
            rig.engine.run(PROMPT, STEPS)
    assert not restores


def test_validate_selection_accepts_numpy_integers_and_fixed_chunks_never_substitutes():
    t, keys = fake(13, ["host", None, "host"])
    p = plan_restore(t, keys, 13)
    assert validate_selection(p, PolicyDecision(np.int64(4), "x")) == 4
    with pytest.raises(PolicyError, match="not a feasible candidate"):  # aligned and <= eligible, past the hole
        validate_selection(p, PolicyDecision(8, "x"))
    assert FixedChunks(1).choose(p).selected_len == C and FixedChunks(0).choose(p).selected_len == 0
    with pytest.raises(PolicyError, match="not feasible"):
        FixedChunks(2).choose(p)
    for bad in (True, -1, 1.0):
        with pytest.raises(ValueError):
            FixedChunks(bad)
    assert AlwaysRestore().choose(p) == PolicyDecision(C, "longest_feasible_prefix")
    assert Recompute().choose(p) == PolicyDecision(0, "policy_recompute")
    t, keys = fake(13, [None, "host", "host"])
    assert AlwaysRestore().choose(plan_restore(t, keys, 13)) == PolicyDecision(0, "no_feasible_restore")


# ---- forced lengths, both backends ---------------------------------------------------------

@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_forced_restore_lengths_are_executed_exactly(tiny_model, tmp_path, backend, k):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=2, backend=backend)
    seed = rig.run(PROMPT)
    assert (seed.restore_len, seed.stored_chunks, rig.tiers(PROMPT)) == (0, [0, 1, 2], ["disk", "host", "host"])
    keys, t = rig.keys(PROMPT), rig.tiered
    lru_before, c0 = list(t.host.entries), t.counters.to_dict()
    r = rig.run(PROMPT, FixedChunks(k))
    d = r.decision
    assert (r.restore_len, d.selected_len, d.actual_len, d.truncated) == (k * C, k * C, k * C, False)
    assert r.sources == ["disk", "host", "host"][:k] and list(d.actual_sources) == r.sources
    assert r.stored_chunks == [] and r.miss_reason is None and d.restore_miss_reason is None
    assert d.policy == "fixed_chunks" and d.plan.feasible_lengths == (0, 4, 8, 12)
    # only the restored host chunks were touched (LRU), in restore order
    touched = [keys[i] for i in range(k) if keys[i] in lru_before]
    assert list(t.host.entries) == [x for x in lru_before if x not in touched] + touched
    c = t.counters.to_dict()
    assert c["chunk_misses"] == c0["chunk_misses"] and c["miss_reasons"] == c0["miss_reasons"]
    assert c["restored_requests"] - c0["restored_requests"] == (k > 0)
    assert c["full_recompute_requests"] - c0["full_recompute_requests"] == (k == 0)
    assert c["recovered_prefix_tokens"] - c0["recovered_prefix_tokens"] == k * C
    if k == 0:
        assert t.last_restore is None and d.reason == "fixed_chunks"
    else:
        assert t.last_restore["chunks_planned"] == t.last_restore["chunks_loaded"] == k


def test_long_suffix_execution(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8, backend="batched", group=4)
    rig.run(LONG, Recompute())  # seeds every complete chunk; L = 0 is the policy's choice, not a miss
    for k in (1, 0, 6, 2):
        r = rig.run(LONG, FixedChunks(k))
        assert r.restore_len == k * C and len(LONG) - r.restore_len == {1: 23, 0: 27, 6: 3, 2: 19}[k]
    assert rig.tiered.counters.chunk_misses == 0


def test_policy_recompute_is_not_a_miss_and_clears_last_restore(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8)
    rig.run(PROMPT)
    rig.run(PROMPT)  # an always-restore request sets last_restore
    assert rig.tiered.last_restore is not None
    c0 = rig.tiered.counters.to_dict()
    r = rig.run(PROMPT, Recompute())
    c = rig.tiered.counters.to_dict()
    assert (r.restore_len, r.miss_reason, r.stored_chunks) == (0, None, [])
    assert r.decision.reason == "policy_recompute" and r.decision.restore_miss_reason is None
    assert rig.tiered.last_restore is None
    assert c["chunk_misses"] == c0["chunk_misses"] and c["full_recompute_requests"] == c0["full_recompute_requests"] + 1


def test_policy_marks_and_prefill_request_use_the_attached_policy(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8, policy=FixedChunks(1))
    rig.run(PROMPT, AlwaysRestore())
    rig.engine.policy = FixedChunks(1)
    marks = []
    p = rig.engine.prefill_request(PROMPT, mark=marks.append)
    assert marks == ["ingest", "decide", "restore", "prefill", "store"]
    assert (p.restore_len, p.decision.selected_len, p.decision.policy) == (C, C, "fixed_chunks")
    json.dumps(p.decision.to_dict())


def test_allow_restore_false_never_consults_the_policy(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8)
    rig.run(PROMPT)

    class Exploding:
        def choose(self, plan):
            raise AssertionError("policy consulted")
    rig.engine.policy = Exploding()
    marks, calls = [], []
    hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        p = rig.engine.prefill_request(PROMPT, allow_restore=False, mark=marks.append)
    finally:
        hook.remove()
    assert len(calls) == 1 and marks == ["ingest", "restore", "prefill", "store"]
    assert (p.restore_len, p.miss_reason, p.decision) == (0, "restore_not_attempted", None)


def test_always_restore_policy_does_not_fabricate_lookup_misses(tiny_model, tmp_path):
    """policy=None counts an 'absent' miss for a hole; the opt-in policy records it in the plan only."""
    rigs = {name: Rig(tiny_model, tmp_path / name, host_slots=8) for name in ("none", "policy")}
    out, misses = {}, {}
    for name, rig in rigs.items():
        rig.run(PROMPT)  # seed (policy=None on both: one "absent" miss each for the empty store)
        rig.host.discard(rig.keys(PROMPT)[1])
        c0 = rig.tiered.counters.to_dict()
        out[name] = rig.run(PROMPT, None if name == "none" else AlwaysRestore())
        c = rig.tiered.counters.to_dict()
        misses[name] = (c["chunk_misses"] - c0["chunk_misses"],
                        c["miss_reasons"].get("absent", 0) - c0["miss_reasons"].get("absent", 0))
    a, b = out["none"], out["policy"]
    assert (a.restore_len, a.miss_reason, a.stored_chunks) == (C, "absent", [1])
    assert (b.restore_len, b.miss_reason, b.stored_chunks) == (C, None, [1])
    assert b.decision.plan.limit_reason == "absent" and misses == {"none": (1, 1), "policy": (0, 0)}
    assert all(torch.equal(x, y) for x, y in zip(a.step_logits, b.step_logits))  # same split (L = C)


# ---- validation failures inside / outside the chosen prefix --------------------------------

@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
@pytest.mark.parametrize("fault, reason, actual", [
    ("crc_chunk1", "disk_payload_crc", 1),
    ("missing_chunk0", "disk_file_missing", 0),
    ("eio_chunk0", "disk_read_oserror:EIO", 0),
])
def test_failure_inside_the_chosen_prefix_truncates_to_the_verified_prefix(tiny_model, tmp_path, backend, fault,
                                                                         reason, actual):
    fail = {"on": False}

    def readinto(f, mv):
        if fail["on"]:
            raise OSError(errno.EIO, "injected read error")
        return cf.default_readinto(f, mv)
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1, backend=backend, readinto_fn=readinto)
    rig.run(PROMPT)
    assert rig.tiers(PROMPT) == ["disk", "disk", "host"]
    if fault == "crc_chunk1":
        corrupt(rig.path(PROMPT, 1))
    elif fault == "missing_chunk0":
        rig.path(PROMPT, 0).unlink()
    else:
        fail["on"] = True
    c0 = rig.tiered.counters.to_dict()
    r = rig.run(PROMPT, FixedChunks(3))
    d = r.decision
    assert (d.selected_len, d.actual_len, r.restore_len, d.truncated) == (3 * C, actual * C, actual * C, True)
    assert r.miss_reason == d.restore_miss_reason == reason and r.sources == ["disk"] * actual
    assert r.stored_chunks == [actual]  # the invalid entry was dropped; its chunk is recomputed and re-admitted
    c = rig.tiered.counters.to_dict()
    assert c["chunk_misses"] - c0["chunk_misses"] == 1 and c["miss_reasons"].get(reason, 0) == 1
    assert c["full_recompute_requests"] - c0["full_recompute_requests"] == (actual == 0)


def test_bad_chunk_beyond_the_chosen_prefix_is_never_read(tiny_model, tmp_path, monkeypatch):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1)
    rig.run(PROMPT)  # disk, disk, host
    keys = rig.keys(PROMPT)
    raw = corrupt(rig.path(PROMPT, 1))
    reads, real = [], rig.disk.read_into

    def spy(key, *a):
        reads.append(key)
        return real(key, *a)
    monkeypatch.setattr(rig.disk, "read_into", spy)
    misses = rig.tiered.counters.chunk_misses
    r = rig.run(PROMPT, FixedChunks(1))
    assert reads == [keys[0]] and (r.restore_len, r.miss_reason, r.decision.truncated) == (C, None, False)
    assert rig.path(PROMPT, 1).read_bytes() == raw and keys[1] in rig.disk and r.stored_chunks == []
    assert rig.tiered.counters.chunk_misses == misses


@pytest.mark.parametrize("policy, want_L", [(Recompute(), 0), (AlwaysRestore(), 6 * C), (FixedChunks(2), 2 * C)])
def test_write_failures_never_repeat_or_abort_inference(tiny_model, tmp_path, policy, want_L):
    fail = {"on": False}

    def write(f, mv):
        if fail["on"]:
            f.write(mv[:16])
            raise OSError(errno.ENOSPC, "No space left on device")
        return cf.default_write(f, mv)
    rig = Rig(tiny_model, tmp_path / "d", host_slots=2, write_fn=write)
    rig.run(LONG)  # chunks 0-3 on disk, 4-5 in host
    fail["on"] = True
    longer = LONG + list(range(37, 45))  # N = 35: complete chunks 0..7, 6 and 7 new
    r = rig.run(longer, policy)  # admitting 6 and 7 spills two host chunks; both spills fail (ENOSPC)
    assert r.restore_len == want_L and r.stored_chunks == [6, 7]
    assert rig.tiered.counters.spill_failures == {"write_oserror:ENOSPC": 2}
    assert rig.disk.undeletable == {} and rig.disk.tmp_bytes == 0


def test_no_staging_buffer_limits_candidates_without_a_counted_miss(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1)
    rig.run(PROMPT)  # disk, disk, host
    host0 = HostStore(rig.cb - 1, rig.cb, max_entries=4, pin=False)  # too small: no slot, no staging buffer
    assert host0.staging_slots == 0 and host0.slots == 0
    for policy, want_miss in ((None, {"absent": 1, "no_staging_buffer": 1}), (AlwaysRestore(), {})):
        t = TieredKVCache(rig.ns, host0, rig.disk)  # shares the disk store: chunks 0-1 indexed on disk
        eng = AlwaysRestoreEngine(tiny_model, t, policy=policy)
        calls = []
        hook = tiny_model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = eng.run(PROMPT, STEPS)
        finally:
            hook.remove()
        assert len(calls) == STEPS and r.restore_len == 0 and dict(t.counters.miss_reasons) == want_miss
        assert dict(t.counters.write_skips) == {"oversized_for_host": 1}  # chunk 2 only
        if policy is not None:
            assert r.decision.plan.limit_reason == "no_staging_buffer" and r.decision.reason == "no_feasible_restore"
            assert r.decision.plan.indexed_tiers == ("disk", "disk")  # chunk 2's host copy is in the other store


# ---- CostAware through the engine ----------------------------------------------------------

def tiny_model_artifact(rig, coef, backend="baseline", group=None):
    return artifact(coef, dom=domain(prompt=(1, 100), suffix=(1, 100), kmax=8, hmax=8, dmax=8, admitted=(0, 8)),
                    comp=compat(ns=rig.ns.digest().hex(), C=C, backend=backend, group=group))


def test_cost_aware_choice_is_executed_and_recorded(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=1, backend="batched", group=2)
    rig.run(PROMPT)  # disk, disk, host
    disk_heavy = coefficients(full=(50.0, 0.0, 0.0), partial=(1.0, 0.0, 100.0, 0.0, 0.0))
    model = rc.RestoreCostModel(tiny_model_artifact(rig, disk_heavy, "batched", 2))
    r = rig.run(PROMPT, CostAware(model))
    d = r.decision
    assert (d.selected_len, r.restore_len, d.reason) == (0, 0, "min_predicted_objective")
    assert d.objective_ms == ((0, 50.0), (4, 101.0), (8, 201.0), (12, 201.0)) and r.stored_chunks == []
    flat = coefficients(full=(50.0, 0.0, 0.0), partial=(1.0, 0.0, 0.0, 0.0, 0.0))  # positive candidates tie
    r = rig.run(PROMPT, CostAware(rc.RestoreCostModel(tiny_model_artifact(rig, flat, "batched", 2))))
    assert (r.decision.selected_len, r.restore_len, r.sources) == (3 * C, 3 * C, ["disk", "disk", "host"])


def test_cost_aware_incompatibility_raises_before_any_work(tiny_model, tmp_path, monkeypatch):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8, backend="baseline")
    rig.run(PROMPT)
    model = rc.RestoreCostModel(tiny_model_artifact(rig, coefficients(), "batched", 2))
    restores = []
    monkeypatch.setattr(rig.tiered, "restore", lambda *a, **k: restores.append(1))
    c0 = rig.tiered.counters.to_dict()
    rig.engine.policy = CostAware(model)
    with pytest.raises(rc.IncompatibleModelError, match="restore backend"):
        rig.engine.run(PROMPT, STEPS)
    assert not restores and rig.tiered.counters.to_dict() == c0


def test_cost_aware_out_of_domain_falls_back_to_always_restore(tiny_model, tmp_path):
    rig = Rig(tiny_model, tmp_path / "d", host_slots=8)
    rig.run(LONG)
    art = artifact(coefficients(full=(1.0, 0.0, 0.0), partial=(99.0, 0.0, 0.0, 0.0, 0.0)),  # would pick L = 0
                   dom=domain(prompt=(1, 20), suffix=(1, 100), kmax=8, hmax=8, dmax=8),
                   comp=compat(ns=rig.ns.digest().hex(), C=C))
    r = rig.run(LONG, CostAware(rc.RestoreCostModel(art)))  # N = 27 > 20
    assert (r.restore_len, r.decision.reason) == (6 * C, "out_of_domain_always_restore:prompt_len")
    r = rig.run(PROMPT)  # N = 13 is inside: the model's choice (L = 0) applies
    assert (r.restore_len, r.decision.reason) == (0, "min_predicted_objective")


# ---- policy=None: identical to the pre-M5a engine ------------------------------------------

def legacy_prefill_request(eng, token_ids, allow_restore=True, on_restored=None, mark=None):
    """The body of AlwaysRestoreEngine.prefill_request at ab39daa (before M5a), adapted only to run as a
    function: `self` -> `eng`, a no-op default mark, and a SimpleNamespace instead of Prefilled."""
    mark = mark or (lambda phase: None)
    token_ids = list(token_ids)
    n = len(token_ids)
    ids = torch.tensor([token_ids], device=eng.device)
    keys = chain_keys(eng.tiered.ns_digest, token_ids, eng.C)
    l_eligible = max_restore_len(n, eng.C)
    mark("ingest")
    if allow_restore:
        cache, L, sources, miss_reason = eng.tiered.restore(token_ids, keys[:l_eligible // eng.C],
                                                            eng.device, eng.model.config)
    else:
        cache, L, sources, miss_reason = None, 0, [], "restore_not_attempted"
    mark("restore")
    c = eng.tiered.counters
    c.requests += 1
    if L:
        c.restored_requests += 1
        if on_restored is not None:
            on_restored(cache, L)
        logits, cache = prefill(eng.model, ids[:, L:], cache, L)
    else:
        c.full_recompute_requests += 1
        logits, cache = prefill(eng.model, ids, None, 0)
    mark("prefill")
    stored = eng.tiered.store(cache, token_ids, keys)
    mark("store")
    return SimpleNamespace(logits=logits, cache=cache, prompt_len=n, eligible_restore_len=l_eligible,
                           restore_len=L, sources=sources, miss_reason=miss_reason, stored_chunks=stored)


def observe(rig, fn, restore_calls):
    marks, calls, restored = [], [], []
    hook = rig.model.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        out = fn(marks, lambda cache, L: restored.append(L))
    finally:
        hook.remove()
    t = rig.tiered
    obs = {"marks": marks, "calls": len(calls), "restored_cb": restored, "restore_calls": list(restore_calls),
           "counters": t.counters.to_dict(), "host_lru": list(t.host.entries), "disk_lru": list(t.disk.entries),
           "last_restore": t.last_restore,
           "fields": {k: getattr(out, k) for k in ("prompt_len", "eligible_restore_len", "restore_len", "sources",
                                                   "miss_reason", "stored_chunks", "tokens") if hasattr(out, k)}}
    return out, obs


@pytest.mark.parametrize("backend", RESTORE_BACKENDS)
def test_policy_none_matches_the_pre_m5a_engine(tiny_model, tmp_path, monkeypatch, backend):
    def boom(*a, **k):
        raise AssertionError("policy=None must not plan or validate")
    monkeypatch.setattr("kvlab.always_restore.plan_restore", boom)
    monkeypatch.setattr("kvlab.always_restore.validate_selection", boom)
    rigs = {"new": Rig(tiny_model, tmp_path / "new", host_slots=2, disk_files=6, backend=backend),
            "old": Rig(tiny_model, tmp_path / "old", host_slots=2, disk_files=6, backend=backend)}
    spies = {}
    for name, rig in rigs.items():
        spies[name], real = [], rig.tiered.restore
        monkeypatch.setattr(rig.tiered, "restore",
                            lambda toks, keys, *a, _s=spies[name], _r=real: (_s.append(len(keys)), _r(toks, keys, *a))[1])
    P2, P3 = list(range(10, 31)), list(range(200, 219))

    def step(kind, prompt, allow=True):
        res = {}
        for name, rig in rigs.items():
            eng = rig.engine
            if kind == "prefill":
                fn = (lambda m, cb, e=eng: e.prefill_request(prompt, allow, cb, m.append)) if name == "new" else \
                     (lambda m, cb, e=eng: legacy_prefill_request(e, prompt, allow, cb, m.append))
            elif name == "new":
                fn = lambda m, cb, e=eng: e.run(prompt, STEPS, on_restored=cb)
            else:
                def fn(m, cb, e=eng):
                    p = legacy_prefill_request(e, prompt, True, cb)
                    toks, steps = greedy_continue(e.model, p.cache, p.logits, p.prompt_len, STEPS)
                    return SimpleNamespace(**vars(p), tokens=toks, step_logits=steps)
            res[name] = observe(rig, fn, spies[name])
        (a, oa), (b, ob) = res["new"], res["old"]
        assert oa == ob
        assert a.decision is None
        if kind == "prefill":
            assert torch.equal(a.logits, b.logits)
            assert all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values)
                       for x, y in zip(a.cache.layers, b.cache.layers))
        else:
            assert all(torch.equal(x, y) for x, y in zip(a.step_logits, b.step_logits))
        return a

    assert step("prefill", PROMPT).stored_chunks == [0, 1, 2]
    assert step("prefill", PROMPT).restore_len == 3 * C
    step("prefill", P2)
    for rig in rigs.values():  # a hole: drop the first host-resident chunk of P2 beyond chunk 0
        k = next(k for k in rig.keys(P2)[1:] if k in rig.host)
        rig.host.discard(k)
    assert step("prefill", P2).miss_reason == "absent"
    for rig in rigs.values():
        corrupt(rig.disk.root / f"{next(k for k in rig.keys(P2) if k in rig.disk).hex()}.kvc")
    assert step("prefill", P2).miss_reason == "disk_payload_crc"
    assert step("prefill", PROMPT, allow=False).miss_reason == "restore_not_attempted"
    step("run", P3)
    step("run", P2)
    assert spies["new"] == spies["old"] and len(spies["new"]) == 7
