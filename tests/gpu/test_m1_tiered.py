"""M1 GPU integration: real model, pinned host tier, real files on disk.

Run: .venv/bin/python -m pytest -m "gpu and m1" -v

Every request is replayed by ResidentReference (GPU-only, same restore length,
same stored chunks). Storage correctness is asserted strictly: all step logits
bitwise equal, tokens equal, restored K/V byte-identical. The M0 tolerance is
not used here. Full-recompute comparisons are reported by scripts/m1_run.py.
Every request must also make exactly STEPS model forward calls (1 prefill +
STEPS-1 decode steps): cache failures never repeat or abort inference (M1.1).
M3: the `harness` fixture is parametrized over both restore backends, so every scenario
here also runs through the batched restore path (kvlab/tiered_cache.py, D30).
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import dataclasses
import errno
import os

import pytest
import torch

from kvlab import chunk_format as cf
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.fixtures import BOUNDARY_TEXT, CALIBRATION_TEXT, exact_length_ids
from kvlab.host_store import HostStore
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for
from kvlab.resident_reference import ResidentReference
from kvlab.tiered_cache import TieredKVCache

pytestmark = [pytest.mark.gpu, pytest.mark.m1]
C, STEPS = 256, 8


@pytest.fixture(scope="module")
def model_tok():
    if not torch.cuda.is_available():
        pytest.skip("BLOCKED: CUDA not available")
    return km.load_model()


@pytest.fixture(scope="module")
def prompts(model_tok):
    _, tok = model_tok
    shared = exact_length_ids(tok, BOUNDARY_TEXT, 768)
    calib = exact_length_ids(tok, CALIBRATION_TEXT, 900)
    tail_a = tok(" Summarize the logbook in one sentence.", add_special_tokens=False)["input_ids"]
    tail_b = tok(" What did the keeper do on foggy nights?", add_special_tokens=False)["input_ids"]
    return {
        "X": shared[:512] + calib[:100] + tail_a,        # 2 restorable chunks
        "Y": shared[:512] + calib[300:420] + tail_b,     # shares 2 chunks with X
        "Z": shared[:256] + calib[500:600] + tail_b,     # shares 1 chunk with X/Y
        "W": calib[:600],                                # unrelated
        "L3": shared[:768] + tail_a,                     # 3 restorable chunks
    }


class Harness:
    def __init__(self, model, root, host_chunks, disk_files, write_fn=cf.default_write, backend="baseline"):
        self.model = model
        self.ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, C)
        cb = cf.payload_len(self.ns)
        self.host = HostStore((host_chunks + 1) * cb, cb, max_entries=64, pin=True)
        self.disk = DiskStore(root, disk_files * cf.file_size(self.ns), max_entries=64, write_fn=write_fn)
        self.tiered = TieredKVCache(self.ns, self.host, self.disk, restore_backend=backend)
        self.engine = AlwaysRestoreEngine(model, self.tiered)
        self.ref = ResidentReference(model, self.ns.digest(), C)

    def run(self, token_ids):
        seen = {}

        def on_restored(cache, L):
            seen["bytes"] = self.ref.check_restored(token_ids, cache, L)
        calls = []
        hook = self.model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = self.engine.run(token_ids, STEPS, on_restored)
        finally:
            hook.remove()
        assert len(calls) == STEPS, f"{len(calls)} forward calls: inference repeated or cut short"
        ref_tokens, ref_steps = self.ref.run(token_ids, r.restore_len, r.stored_chunks, STEPS)
        assert all(bool(torch.isfinite(s).all()) for s in r.step_logits)
        assert r.tokens == ref_tokens
        assert all(torch.equal(a, b) for a, b in zip(r.step_logits, ref_steps)), "tiered != resident (bitwise)"
        if r.restore_len:
            assert seen["bytes"]["all_bytes_equal"], seen["bytes"]
        hs, d = self.host, self.disk
        assert hs.slab_bytes <= hs.budget_bytes and hs.resident_bytes + hs.staging_bytes <= hs.slab_bytes
        assert d.peak_bytes <= d.budget_bytes and d.tmp_bytes == 0
        assert d.scan_dir_bytes()[0] <= d.used_bytes <= d.budget_bytes
        return r

    def path(self, key):
        return self.disk.root / f"{key.hex()}.kvc"


@pytest.fixture(params=["baseline", "batched"])
def harness(model_tok, tmp_path, request):
    made = []

    def make(host_chunks, disk_files, **kw):
        h = Harness(model_tok[0], tmp_path / f"disk{len(made)}", host_chunks, disk_files, backend=request.param, **kw)
        made.append(h)
        return h
    yield make
    for h in made:
        h.host.close()


def test_namespace_reflects_loaded_model(model_tok):
    ns = namespace_for(model_tok[0], km.MODEL_ID, km.MODEL_REVISION, C)
    assert (ns.kv_dtype, ns.num_layers, ns.num_kv_heads, ns.head_dim, ns.attn_implementation) == \
        ("bfloat16", 28, 2, 128, "sdpa")
    assert cf.payload_len(ns) == 7_340_032


def test_host_hit_restoration(harness, prompts):
    h = harness(host_chunks=4, disk_files=4)
    r1 = h.run(prompts["X"])
    assert (r1.restore_len, r1.stored_chunks) == (0, [0, 1])
    r2 = h.run(prompts["Y"])
    assert (r2.restore_len, r2.sources, r2.miss_reason) == (512, ["host", "host"], None)
    assert h.tiered.counters.host_hits == 2 and h.tiered.counters.disk_hits == 0
    assert h.host.pinned and h.host.registered_bytes == h.host.slab_bytes == 5 * cf.payload_len(h.ns)
    assert (h.host.resident_bytes, h.host.peak_resident_bytes) == (2 * cf.payload_len(h.ns),) * 2


def test_forced_host_eviction_then_disk_hit(harness, prompts):
    h = harness(host_chunks=1, disk_files=8)
    h.run(prompts["X"])  # chunk 0 spilled to disk when chunk 1 arrived
    keys = chain_keys(h.ns.digest(), prompts["X"], C)
    assert h.tiered.tier_of(keys[0]) == "disk" and h.tiered.tier_of(keys[1]) == "host"
    assert h.path(keys[0]).stat().st_size == cf.file_size(h.ns)
    r = h.run(prompts["Y"])
    assert (r.restore_len, r.sources) == (512, ["disk", "host"])
    r = h.run(prompts["Z"])
    assert (r.restore_len, r.sources) == (256, ["disk"])
    c = h.tiered.counters
    assert c.disk_hits == 2 and c.host_spills_to_disk >= 1 and h.host.evictions >= 1


def test_partial_prefix_and_chunk_boundaries(harness, prompts):
    h = harness(host_chunks=4, disk_files=4)
    base = prompts["L3"]
    r = h.run(base[:513])
    assert (r.eligible_restore_len, r.restore_len, r.stored_chunks) == (512, 0, [0, 1])
    r = h.run(base[:512])   # exact multiple: chunk 1 is cached but must not be restored
    assert (r.eligible_restore_len, r.restore_len, r.stored_chunks) == (256, 256, [])
    r = h.run(base[:257])
    assert (r.restore_len, r.sources) == (256, ["host"])
    r = h.run(base[:256])   # no eligible boundary -> full recompute, chunk 0 already cached
    assert (r.eligible_restore_len, r.restore_len, r.restore_used) == (0, 0, False)
    r = h.run(base[:513])
    assert r.restore_len == 512
    r = h.run(base)         # 3rd chunk computed after a 2-chunk restore, then stored
    assert (r.restore_len, r.stored_chunks) == (512, [2])
    r = h.run(base[:769])
    assert r.restore_len == 768


def test_corrupt_middle_chunk_falls_back_to_verified_prefix(harness, prompts):
    h = harness(host_chunks=1, disk_files=8)
    p = prompts["L3"] + [11]  # 3 complete chunks, 3 restorable (N = 768 + len(tail) + 1)
    h.run(p)
    keys = chain_keys(h.ns.digest(), p, C)
    assert [h.tiered.tier_of(k) for k in keys] == ["disk", "disk", "host"]
    raw = bytearray(h.path(keys[1]).read_bytes())
    raw[len(raw) // 2] ^= 0x01  # one payload bit
    h.path(keys[1]).write_bytes(raw)
    r = h.run(p)
    assert (r.restore_len, r.sources, r.miss_reason) == (256, ["disk"], "disk_payload_crc")
    assert r.stored_chunks == [1]  # recomputed chunk stored again
    assert h.tiered.counters.miss_reasons["disk_payload_crc"] == 1
    r = h.run(p)
    assert r.restore_len == 768


def test_missing_file_falls_back_to_full_recompute(harness, prompts):
    h = harness(host_chunks=1, disk_files=8)
    h.run(prompts["X"])
    keys = chain_keys(h.ns.digest(), prompts["X"], C)
    h.path(keys[0]).unlink()
    r = h.run(prompts["Y"])
    assert (r.restore_len, r.restore_used, r.miss_reason) == (0, False, "disk_file_missing")


@pytest.mark.parametrize("kind", ["namespace", "layout"])
def test_incompatible_entry_is_a_miss(harness, prompts, kind):
    h = harness(host_chunks=1, disk_files=8)
    h.run(prompts["X"])
    keys = chain_keys(h.ns.digest(), prompts["X"], C)
    path = h.path(keys[0])
    raw = path.read_bytes()
    payload = raw[len(raw) - cf.payload_len(h.ns) - 4:-4]
    if kind == "namespace":   # same tokens and file name, written under another attention backend
        other = dataclasses.replace(h.ns, attn_implementation="eager")
        okeys = chain_keys(other.digest(), prompts["X"], C)
        oid = ChunkIdentity.for_chunk(other.digest(), prompts["X"], C, 0, okeys)
        with open(path, "wb") as f:
            cf.write_chunk(f, cf.encode_header(other, oid), payload)
        expected = "disk_namespace_mismatch:attn_implementation"
    else:                     # header claims 27 layers
        bad = bytearray(raw)
        bad[116:120] = (27).to_bytes(4, "little")
        path.write_bytes(bad)
        expected = "disk_layout_mismatch:num_layers"
    r = h.run(prompts["Y"])
    assert (r.restore_len, r.miss_reason) == (0, expected)


def test_multi_request_sequence_with_shared_prefixes_and_eviction(harness, prompts):
    h = harness(host_chunks=2, disk_files=2)  # 5 distinct chunks > 2 host + 2 disk -> both tiers evict
    order = ["X", "Y", "W", "Z", "X", "L3", "Y", "W", "X"]
    results = [h.run(prompts[name]) for name in order]
    c = h.tiered.counters
    assert c.host_hits > 0 and c.disk_hits > 0
    assert h.host.evictions > 0 and h.disk.evictions > 0
    assert c.restored_requests > 0 and c.full_recompute_requests > 0
    assert h.disk.scan_dir_bytes()[0] == h.disk.committed_bytes <= h.disk.budget_bytes
    assert any(r.sources and "disk" in r.sources for r in results)


def test_write_failure_does_not_repeat_inference(harness, prompts, model_tok):
    def enospc(f, mv):
        raise OSError(errno.ENOSPC, "No space left on device")
    h = harness(host_chunks=1, disk_files=8, write_fn=enospc)
    model = model_tok[0]
    calls = []
    hook = model.register_forward_pre_hook(lambda *a: calls.append(1))
    try:
        r = h.engine.run(prompts["L3"], STEPS)
    finally:
        hook.remove()
    assert len(calls) == STEPS  # 1 prefill + STEPS-1 decode steps; nothing recomputed
    assert r.stored_chunks == [0, 1, 2]
    assert h.tiered.counters.spill_failures == {"write_oserror:ENOSPC": 2}
    assert h.disk.scan_dir_bytes() == (0, [])


def _deny_unlink(monkeypatch, match):
    """os.unlink raises EACCES for matching file names (no real permissions are changed)."""
    real, attempts = os.unlink, []

    def fake_unlink(path, *args, **kwargs):
        name = os.path.basename(os.fspath(path))
        if match(name):
            attempts.append(name)
            raise OSError(errno.EACCES, os.strerror(errno.EACCES), os.fspath(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", fake_unlink)
    return attempts


@pytest.mark.parametrize("fault", ["evict_unlink", "invalid_cleanup"])
def test_disk_cleanup_failure_neither_aborts_nor_repeats_inference(harness, prompts, monkeypatch, fault):
    """M1.1: an undeletable disk file is a reported write skip (eviction) or a safe miss
    (invalid entry). Its bytes stay accounted, it is never read, and Harness.run checks
    exactly STEPS forward calls plus the strict resident replay on every request."""
    p = prompts["L3"] + [11]  # 3 complete chunks, all restorable
    if fault == "evict_unlink":
        h = harness(host_chunks=1, disk_files=1)
        keys = chain_keys(h.ns.digest(), p, C)
        attempts = _deny_unlink(monkeypatch, lambda n: n == h.path(keys[0]).name)
        r = h.run(p)  # chunk 1's spill must evict chunk 0's file, which cannot be deleted
        assert (r.restore_len, r.stored_chunks) == (0, [0, 1, 2])
        assert h.tiered.counters.spill_failures == {"evict_oserror:EACCES": 1}
        assert h.disk.undeletable == {h.path(keys[0]).name: cf.file_size(h.ns)}
        r = h.run(p)  # chunk 0 is not readable any more
        assert (r.restore_len, r.miss_reason) == (0, "absent") and len(attempts) == 1
    else:
        h = harness(host_chunks=1, disk_files=8)
        keys = chain_keys(h.ns.digest(), p, C)
        h.run(p)
        raw = bytearray(h.path(keys[1]).read_bytes())
        raw[len(raw) // 2] ^= 0x01  # one payload bit
        h.path(keys[1]).write_bytes(raw)
        attempts = _deny_unlink(monkeypatch, lambda n: n == h.path(keys[1]).name)
        r = h.run(p)
        assert (r.restore_len, r.sources, r.miss_reason, r.stored_chunks) == (256, ["disk"], "disk_payload_crc", [1])
        assert h.disk.undeletable == {h.path(keys[1]).name: cf.file_size(h.ns)}
        assert h.path(keys[1]).read_bytes() == bytes(raw)
        r = h.run(p)  # the recomputed chunk 1 comes from host; the invalid file is never read
        assert (r.restore_len, r.sources) == (768, ["disk", "host", "disk"]) and len(attempts) == 1
    assert h.disk.scan_dir_bytes()[0] == h.disk.used_bytes <= h.disk.budget_bytes  # only complete final files remain
