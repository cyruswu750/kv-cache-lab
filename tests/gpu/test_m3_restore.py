"""M3 GPU checks of the batched restore backend: real model, pinned slab, real files.

Run: .venv/bin/python -m pytest -m "gpu and m3" -v

- A/B on one store: the same restore under identical state is run once per backend
  (switching `restore_backend` between calls). K/V bytes, suffix-prefill logits, every greedy
  step's logits and the tokens must be bitwise equal, and both must equal the
  same-provenance resident replay (the M1 strict contract, D10/D12).
- Repeated disk reads through the single staging slot with distinct payloads.
- Ownership: a returned cache is not changed by later requests, buffer reuse or host-slot reuse.
- Every chunk transfer is a blocking copy (one synchronizing H2D per chunk), so a lease or the
  staging slot is never released while its transfer is still queued.
- Transient device memory is bounded as documented in docs/M3_DESIGN.md §5.
Full-prefill comparisons are not made here (report-only under D10).
"""

import kvlab.model as km  # first: sets HF_HOME before transformers import

import warnings

import pytest
import torch

from kvlab import chunk_format as cf
from kvlab.model import greedy_continue, prefill
from kvlab.identity import chain_keys
from test_m1_tiered import C, STEPS, Harness

pytestmark = [pytest.mark.gpu, pytest.mark.m3]


@pytest.fixture(scope="module")
def model_tok():
    if not torch.cuda.is_available():
        pytest.skip("BLOCKED: CUDA not available")
    return km.load_model()


@pytest.fixture(scope="module")
def long_ids(model_tok):
    from kvlab.fixtures import BOUNDARY_TEXT, CALIBRATION_TEXT, exact_length_ids
    return exact_length_ids(model_tok[1], BOUNDARY_TEXT + "\n\n" + CALIBRATION_TEXT, 8 * C + 40)


@pytest.fixture
def rig(model_tok, tmp_path):
    made = []

    def make(host_chunks, disk_files, backend="batched"):
        h = Harness(model_tok[0], tmp_path / f"disk{len(made)}", host_chunks, disk_files, backend=backend)
        made.append(h)
        return h
    yield make
    for h in made:
        h.host.close()


def kv_u8(cache):
    return [(layer.keys.view(torch.uint8).clone(), layer.values.view(torch.uint8).clone()) for layer in cache.layers]


def restore_with(h, backend, ids, n):
    h.tiered.restore_backend = backend
    keys = chain_keys(h.ns.digest(), ids, C)[:n]
    return h.tiered.restore(ids, keys, h.engine.device, h.model.config)


@pytest.mark.parametrize("layout", ["host", "disk", "mixed"])
@pytest.mark.parametrize("n", [1, 3, 5, 6])
def test_batched_matches_baseline_on_identical_state(rig, long_ids, layout, n):
    extra = 1 if layout == "disk" else 0             # disk: one more chunk, which stays in the 1-slot host
    ids = long_ids[:(n + extra) * C + 17]
    h = rig(host_chunks={"host": 8, "disk": 1, "mixed": 2}[layout], disk_files=8, backend="baseline")
    first = h.run(ids)                               # stores every chunk (spilling as the host fills)
    assert first.restore_len == 0 and first.stored_chunks == list(range(n + extra))
    keys = chain_keys(h.ns.digest(), ids, C)
    want = {"host": ["host"] * n, "disk": ["disk"] * n,
            "mixed": ["disk"] * max(0, n - 2) + ["host"] * min(n, 2)}[layout]
    assert [h.tiered.tier_of(k) for k in keys[:n]] == want
    n_plan = n
    out = {}
    for backend in ("baseline", "batched", "baseline"):  # identical state before every call
        cache, L, sources, why = restore_with(h, backend, ids, n_plan)
        assert (L, why) == (n_plan * C, None)
        assert h.ref.check_restored(ids, cache, L)["all_bytes_equal"]
        logits, cache = prefill(h.model, torch.tensor([ids[L:]], device=h.engine.device), cache, L)
        toks, steps = greedy_continue(h.model, cache, logits, len(ids), STEPS)
        out.setdefault(backend, []).append((sources, kv_u8(cache), logits.float().cpu(), toks, steps))
        del cache
    assert out["batched"][0][0] == want
    b0, x, b1 = out["baseline"][0], out["batched"][0], out["baseline"][1]
    for other in (x, b1):
        assert other[0] == b0[0]
        assert all(torch.equal(p, q) for a, b in zip(b0[1], other[1]) for p, q in zip(a, b))  # K/V after prefill
        assert torch.equal(b0[2], other[2]) and b0[3] == other[3]
        assert all(torch.equal(p, q) for p, q in zip(b0[4], other[4]))
    h.tiered.restore_backend = "batched"
    r = h.run(ids)  # resident replay: same restore length, same stored chunks, all steps bitwise
    assert r.restore_len == (n + extra) * C


def test_disk_reads_reuse_the_staging_slot_with_distinct_payloads(rig, long_ids):
    h = rig(host_chunks=1, disk_files=8)
    ids = long_ids[:6 * C + 1]
    h.run(ids)  # chunks 0-4 spilled to disk, chunk 5 in host
    keys = chain_keys(h.ns.digest(), ids, C)
    assert [h.tiered.tier_of(k) for k in keys[:6]] == ["disk"] * 5 + ["host"]
    firsts = {h.ref.chunks[keys[i]][0][0].view(torch.uint8).flatten()[:4096].cpu().numpy().tobytes() for i in range(5)}
    assert len(firsts) == 5  # the five staged payloads differ
    cache, L, sources, why = h.tiered.restore(ids, keys[:5], h.engine.device, h.model.config)
    assert (L, sources, why) == (5 * C, ["disk"] * 5, None)
    assert h.ref.check_restored(ids, cache, L)["all_bytes_equal"]
    assert h.tiered.last_restore["h2d_copy_calls"] == 5 and h.tiered.last_restore["placement_ops"] == 2


def test_every_chunk_transfer_is_one_synchronizing_copy(rig, long_ids):
    """Guard for the synchronization strategy: each chunk is one blocking H2D (memcpy + stream
    sync, reported by torch's sync debug mode), so leases and the staging slot are released only
    after their transfer has finished. A non_blocking copy here would make these counts drop."""
    h = rig(host_chunks=2, disk_files=8)
    ids = long_ids[:6 * C + 1]
    h.run(ids)  # disk x4, host x2
    keys = chain_keys(h.ns.digest(), ids, C)
    counts = {}
    for backend in ("baseline", "batched"):
        h.tiered.restore_backend = backend
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                cache, L, sources, _ = h.tiered.restore(ids, keys[:6], h.engine.device, h.model.config)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        assert sources == ["disk"] * 4 + ["host"] * 2
        counts[backend] = sum(str(x.message).startswith("called a synchronizing CUDA operation") for x in w)
        del cache
    assert counts == {"baseline": 6 * 2 * 28, "batched": 6}, counts


def test_returned_cache_is_not_changed_by_later_requests_or_buffer_reuse(rig, long_ids, model_tok):
    h = rig(host_chunks=2, disk_files=3)
    ids_a = long_ids[:3 * C + 5]
    h.run(ids_a)
    keys_a = chain_keys(h.ns.digest(), ids_a, C)
    cache_a, L, _, _ = h.tiered.restore(ids_a, keys_a[:3], h.engine.device, h.model.config)
    saved = kv_u8(cache_a)
    tok = model_tok[1]
    other = tok(" An unrelated request about tide tables and harbour pilots." * 120, add_special_tokens=False)["input_ids"]
    for ids in (other[:4 * C + 3], ids_a, other[:2 * C + 1], ids_a):  # evictions, disk reads, scratch reuse
        h.run(ids)
    h.host.slab.fill_(0x5A)  # overwrite every host slot and the staging slot
    torch.cuda.synchronize()
    assert all(torch.equal(p, q) for a, b in zip(saved, kv_u8(cache_a)) for p, q in zip(a, b))
    for layer in cache_a.layers:
        for x in (layer.keys, layer.values):
            assert x.untyped_storage().nbytes() == x.numel() * x.element_size()  # owns exactly [0, L)


def test_transient_device_memory_is_bounded(rig, long_ids):
    h = rig(host_chunks=8, disk_files=8)
    ids = long_ids[:6 * C + 1]
    h.run(ids)
    keys = chain_keys(h.ns.digest(), ids, C)
    cb, n = cf.payload_len(h.ns), 6
    peak = {}
    for backend in ("baseline", "batched", "baseline", "batched"):
        h.tiered.restore_backend = backend
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        cache, L, _, _ = h.tiered.restore(ids, keys[:n], h.engine.device, h.model.config)
        torch.cuda.synchronize()
        peak[backend] = torch.cuda.max_memory_allocated() - base
        assert torch.cuda.memory_allocated() - base == n * cb  # only the returned cache stays allocated
        del cache
    G = h.tiered.restore_staging_chunks
    assert peak["baseline"] <= 2 * n * cb + 2 * 2**20            # destination + cache (+ small temporaries)
    assert peak["batched"] <= 2 * n * cb + min(n, G) * cb + 2 * 2**20  # + chunk-major staging
    assert h.tiered.last_restore["device_scratch_bytes"] == min(n, G) * cb


def test_copy_failure_on_gpu_releases_leases_and_staging(rig, long_ids, monkeypatch):
    h = rig(host_chunks=2, disk_files=8)
    ids = long_ids[:4 * C + 1]
    h.run(ids)  # disk, disk, host, host
    keys = chain_keys(h.ns.digest(), ids, C)
    real, calls = h.tiered._typed, []

    def failing(flat):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated device copy failure")
        return real(flat)
    monkeypatch.setattr(h.tiered, "_typed", failing)
    with pytest.raises(RuntimeError, match="simulated"):
        h.tiered.restore(ids, keys[:4], h.engine.device, h.model.config)
    assert not h.host._staging_busy and all(e.pins == 0 for e in h.host.entries.values())
    monkeypatch.setattr(h.tiered, "_typed", real)
    r = h.run(ids)
    assert (r.restore_len, r.sources) == (4 * C, ["disk", "disk", "host", "host"])
