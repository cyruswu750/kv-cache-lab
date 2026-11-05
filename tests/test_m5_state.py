"""M5b benchmark-only state preparation and reset (tiny random Qwen2, CPU, real temporary files)."""

import pytest
import torch

import m5b_toy as t
from kvlab import m5_spec as sp
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.m5_state import CaseState, ChunkBank, StateError
from kvlab.restore_policy import AlwaysRestore, FixedChunks, Recompute


@pytest.fixture(scope="module")
def tiny():
    model = t.tiny_model()
    return model, t.tiny_ns(model)


def build(tiny, tmp_path, case_id="tc-04", split=0, backend="batched"):
    model, ns = tiny
    cal, ev = t.tiny_specs(backend=backend)
    spec = (cal, ev)[split]
    case = sp.case_by_id(spec)[case_id]
    tokens = sp.prompt_tokens(case["prompt"])
    bank = ChunkBank(model, ns, tokens, case["source"]["P"])
    return model, spec, case, tokens, bank, CaseState(ns, case, spec, bank, tmp_path / case_id, pin=False)


@pytest.mark.parametrize("case_id", ["tc-01", "tc-03", "tc-04", "tc-06", "tc-08", "tc-09", "tc-10"])
def test_prepared_state_is_exactly_the_declared_one(tiny, tmp_path, case_id):
    _m, _spec, case, _tok, bank, st = build(tiny, tmp_path, case_id)
    obs = st.observe()
    assert {k: v for k, v in obs.items() if k != "file_stats"} == st.expected
    assert [k for k in st.host.entries] == [bank.keys[i] for i in case["host_lru"]]
    assert [k for k in st.disk.entries] == [bank.keys[i] for i in case["disk_lru"]]
    assert st.host.slots == case["capacities"]["host_slots"] and st.disk.max_entries == case["capacities"]["disk_files"]
    assert st.content_check()["ok"] and st.check() == st.fingerprint
    st.close()


@pytest.mark.parametrize("backend", ["baseline", "batched"])
def test_reset_restores_state_across_candidates_and_policies_in_any_order(tiny, tmp_path, backend):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-03", backend=backend)   # H D H, then 1 absent
    eng = AlwaysRestoreEngine(model, st.tiered)
    C = spec["chunk_size"]
    orders = [FixedChunks(3), FixedChunks(0), FixedChunks(1), AlwaysRestore(), Recompute(), FixedChunks(2),
              FixedChunks(3), FixedChunks(1)]
    for pol in orders:
        before_counters = st.tiered.counters.to_dict()
        eng.policy = pol
        p = eng.prefill_request(tokens)
        assert p.stored_chunks == case["admitted"] == [3]                  # exactly the declared absent chunk
        assert p.decision.actual_len == p.decision.selected_len
        lru_now = list(st.host.entries)
        touched = [bank.keys[i] for i, t_ in enumerate(case["feasible_tiers"][:p.restore_len // C]) if t_ == "host"]
        assert lru_now[-1] == bank.keys[3] and all(k in lru_now for k in touched)
        st.reset(p.stored_chunks)                                             # raises if anything differs
        assert list(st.host.entries) == [bank.keys[i] for i in case["host_lru"]]
        assert list(st.host._free) == st.expected["host_free"]
        c = st.tiered.counters.to_dict()                                     # production counters keep counting
        assert c["requests"] == before_counters["requests"] + 1
    assert st.content_check()["ok"]
    st.close()


def test_admission_cleanup_alone_does_not_restore_lru_order(tiny, tmp_path):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-10")        # D H H H, then 1 absent
    eng = AlwaysRestoreEngine(model, st.tiered, policy=FixedChunks(2))          # leases host chunk 1 only
    p = eng.prefill_request(tokens)
    for i in reversed(p.stored_chunks):
        st.host.discard(bank.keys[i])
    with pytest.raises(StateError, match="host_lru"):
        st.check()
    st.reset([])                                                                  # the full reset fixes it
    st.close()


def test_disk_recency_is_reestablished_after_reads(tiny, tmp_path):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-07")        # D D D D D, then 1 absent
    eng = AlwaysRestoreEngine(model, st.tiered, policy=FixedChunks(2))
    p = eng.prefill_request(tokens)                                              # disk hits move 0, 1 to the end
    assert list(st.disk.entries)[-2:] == [bank.keys[0], bank.keys[1]]
    st.reset(p.stored_chunks)
    assert list(st.disk.entries) == [bank.keys[i] for i in case["disk_lru"]]
    st.close()


def test_checks_detect_tampering(tiny, tmp_path):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-03")
    st.host.entries.move_to_end(next(iter(st.host.entries)))
    with pytest.raises(StateError):
        st.check()
    st.host.entries.move_to_end(bank.keys[2])                                      # back to the declared order
    st.check()
    path = st.disk.root / f"{bank.keys[1].hex()}.kvc"
    raw = path.read_bytes()
    path.unlink()
    path.write_bytes(raw)                                                         # same bytes, a new inode/mtime
    with pytest.raises(StateError, match="file_stats"):
        st.check()
    st.setup = st.observe()                                                       # accept that file again
    slot = st.host.entries[bank.keys[0]].slot
    st.host.slot_view(slot)[5] ^= 1                                               # one payload bit
    assert st.content_check()["ok"] is False
    st.host.slot_view(slot)[5] ^= 1
    assert st.content_check()["ok"] is True
    with st.host.lease(bank.keys[0]):
        with pytest.raises(StateError, match="host_pins"):
            st.check()
    st.close()


def test_bank_is_the_single_provenance_of_stores_and_reference(tiny, tmp_path):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-04")        # D H
    ref = bank.reference(model)
    eng = AlwaysRestoreEngine(model, st.tiered, policy=FixedChunks(2))
    p = eng.prefill_request(tokens)
    chk = ref.check_restored(tokens, p.cache, p.restore_len)
    assert chk["all_bytes_equal"] and chk["tensors_checked"] == 2 * 2 * 2
    # the bank's bytes are those of ONE full prefill of prompt[:P]: a reference replay of the same split matches
    _tok, steps = ref.run(tokens, p.restore_len, [], steps=1)
    assert torch.equal(p.logits.float().cpu(), steps[0])
    assert len(bank.payload_sha256) == case["source"]["P"] // spec["chunk_size"]
    st.reset(p.stored_chunks)
    st.close()


def test_a_case_never_reuses_or_reopens_an_existing_directory(tiny, tmp_path):
    model, spec, case, tokens, bank, st = build(tiny, tmp_path, "tc-04")
    with pytest.raises(StateError, match="exists"):
        CaseState(tiny[1], case, spec, bank, st.root, pin=False)
    assert st.content_check()["ok"]                                                # the files are untouched
    st.close()
    assert not st.root.exists()
