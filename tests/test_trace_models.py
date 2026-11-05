"""Independent key-level models (kvlab/trace_models.py) against the real caches, on CPU."""

import random

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvlab import chunk_format as cf
from kvlab import trace_models as tm
from kvlab import trace_replay as rp
from kvlab import traces as tr
from kvlab.always_restore import AlwaysRestoreEngine
from kvlab.disk_store import DiskStore
from kvlab.host_store import HostStore
from kvlab.identity import namespace_for
from kvlab.tiered_cache import TieredKVCache

C = 4
V = (10, 500)
TRACES = [
    tr.no_reuse(1, n_requests=6, min_len=5, max_len=14, vocab=V),
    tr.shared_prefix(2, n_requests=6, prefix_len=9, suffix_min=1, suffix_max=6, vocab=V),
    tr.multi_turn(3, n_conversations=3, n_turns=3, opening_len=5, user_min=1, user_max=4, answer_min=1, answer_max=4,
                  vocab=V),
    tr.doc_reuse(4, n_requests=10, n_docs=3, instruction_len=3, doc_min=4, doc_max=9, question_min=1, question_max=3,
                 zipf_s=1.1, vocab=V),
]


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def model():
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ns(model):
    return namespace_for(model, "toy/tiny-qwen2", "0" * 40, C)


def prompts(t):
    return [list(r.tokens) for r in t.requests]


@pytest.mark.parametrize("ti", range(len(TRACES)))
@pytest.mark.parametrize("host_slots, disk_files", [(1, 2), (2, 3), (2, 100), (6, 100)])
@pytest.mark.parametrize("backend", ["baseline", "batched"])
def test_tiered_model_equals_the_real_tiered_cache(model, ns, tmp_path, ti, host_slots, disk_files, backend):
    t = TRACES[ti]
    cb, fs = cf.payload_len(ns), cf.file_size(ns)
    host = HostStore((host_slots + 1) * cb, cb, max_entries=host_slots, pin=False)
    disk = DiskStore(tmp_path / "d", disk_files * fs, max_entries=disk_files)
    tier = TieredKVCache(ns, host, disk, restore_backend=backend)
    eng = AlwaysRestoreEngine(model, tier)
    m = tm.tiered_model(prompts(t), C, host_slots, disk_files)
    spills = evictions = 0
    for r, want in zip(t.requests, m, strict=True):
        p = eng.prefill_request(list(r.tokens))
        spills += want["spills"]
        evictions += want["disk_evictions"]
        assert (p.restore_len, list(p.sources), list(p.stored_chunks)) == \
            (want["restore_len"], want["sources"], want["stored"])
        assert (tier.counters.host_spills_to_disk, disk.evictions, len(host), len(disk)) == \
            (spills, evictions, want["host_entries"], want["disk_entries"])
        assert not tier.counters.spill_failures and not tier.counters.write_skips
        assert set(tier.counters.miss_reasons) <= {"absent"}
    if disk_files >= 100:
        assert evictions == 0 and [x["restore_len"] for x in m] == tm.oracle_lengths(prompts(t), C)


def test_tiered_model_hand_worked():
    """C = 2, host 1 slot, disk 2 files. P = p0 p1 p2: p0 to host; p1 spills p0; p2 spills p1;
    host {p2}, disk {p0, p1}. Q = p0 p1 + 9: restores p0 (disk) and p1 (disk), stores q2 spilling
    p2, which evicts the disk LRU (p0 was read before p1, so p0 is evicted)."""
    P, Q = (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 9, 9, 8)
    m = tm.tiered_model([P, Q], 2, 1, 2)
    assert (m[0]["stored"], m[0]["spills"], m[0]["host_entries"], m[0]["disk_entries"]) == ([0, 1, 2], 2, 1, 2)
    assert (m[1]["restore_len"], m[1]["sources"], m[1]["stored"]) == (4, ["disk", "disk"], [2])
    assert (m[1]["spills"], m[1]["disk_evictions"]) == (1, 1)


def test_oracle_and_lru_models_agree_with_the_reviewed_statistics():
    rng = random.Random(3)
    for t in TRACES:
        assert tm.oracle_lengths(prompts(t), C) == tr.oracle_reuse(t, C)
        for cap in (0, 1, 2, 3, 7):
            assert [x["restore_len"] for x in tm.lru_model(prompts(t), C, cap)] == tr.lru_restore_lengths(t, C, cap)
    # prefix-aware never restores more than the oracle; at an ample capacity it equals it
    for t in TRACES:
        o = tr.oracle_reuse(t, C)
        for cap in (1, 2, 3, rng.randint(4, 9), 1000):
            got = [x["restore_len"] for x in tm.pa_model(prompts(t), C, cap)]
            assert all(g <= w for g, w in zip(got, o))
        assert [x["restore_len"] for x in tm.pa_model(prompts(t), C, 1000)] == o
