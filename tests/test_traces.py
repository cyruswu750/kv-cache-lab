"""M4-A synthetic traces and offline statistics (CPU only, no model).

The statistics are checked against independent computations (a brute-force longest-common-
prefix oracle and hand-worked LRU sequences), not against a copy of their own code.
"""

import json

import pytest

from kvlab import traces as tr
from kvlab.boundaries import max_restore_len

V = (10, 500)
SMALL = {
    "no_reuse": dict(n_requests=12, min_len=1, max_len=30, vocab=V),
    "shared_prefix": dict(n_requests=10, prefix_len=11, suffix_min=1, suffix_max=9, vocab=V),
    "multi_turn": dict(n_conversations=3, n_turns=4, opening_len=5, user_min=1, user_max=6, answer_min=0,
                       answer_max=6, vocab=V),
    "doc_reuse": dict(n_requests=15, n_docs=4, instruction_len=3, doc_min=2, doc_max=14, question_min=1,
                      question_max=5, zipf_s=1.1, vocab=V),
}


def make(kind, seed=7):
    return tr.GENERATORS[kind](seed, **SMALL[kind])


def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def brute_force_oracle(trace, C):
    """Unbounded cache, every earlier request stores all its complete chunks: request j can restore
    min(eligible, max over earlier q of the complete chunks q stored that match j's prefix)."""
    out = []
    for j, r in enumerate(trace.requests):
        best = max((min(lcp(r.tokens, q.tokens) // C, len(q.tokens) // C) * C for q in trace.requests[:j]), default=0)
        out.append(min(best, max_restore_len(len(r.tokens), C)))
    return out


@pytest.mark.parametrize("kind", list(SMALL))
def test_generation_is_deterministic_and_seed_sensitive(kind):
    a, b, c = make(kind, 7), make(kind, 7), make(kind, 8)
    assert a == b and a.digest == b.digest
    assert c.digest != a.digest and c.requests != a.requests
    assert a.header["synthetic"] is True and a.header["generator"] == kind and a.header["seed"] == 7
    assert all(V[0] <= t < V[1] for r in a.requests for t in r.tokens)


@pytest.mark.parametrize("kind", list(SMALL))
def test_dump_load_round_trip_refuses_overwrite_and_detects_tampering(kind, tmp_path):
    t = make(kind)
    p = tmp_path / "t.jsonl"
    assert tr.dump(t, p) == t.digest
    assert tr.load(p) == t
    with pytest.raises(FileExistsError):
        tr.dump(t, p)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["tokens"][0] += 1
    (tmp_path / "bad.jsonl").write_text("\n".join([lines[0], json.dumps(rec), *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="digest"):
        tr.load(tmp_path / "bad.jsonl")
    (tmp_path / "short.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="requests"):
        tr.load(tmp_path / "short.jsonl")


def test_generator_structures():
    t0 = make("no_reuse")
    assert len({r.tokens[0] for r in t0.requests}) == len(t0.requests)          # no shared prefix at all
    assert all(1 <= len(r.tokens) <= 30 for r in t0.requests)
    t1 = make("shared_prefix")
    rs = t1.requests
    assert all(lcp(a.tokens, b.tokens) == 11 for i, a in enumerate(rs) for b in rs[i + 1:])  # exactly the prefix
    t2 = make("multi_turn")
    assert [r.id for r in t2.requests[:4]] == ["c0t0", "c1t0", "c2t0", "c0t1"]   # turn-major interleaving
    for c in range(3):
        conv = [r.tokens for r in t2.requests if r.group == f"c{c}"]
        assert all(b[:len(a)] == a and len(b) > len(a) for a, b in zip(conv, conv[1:]))  # each turn extends the last
    assert all(lcp(a.tokens, b.tokens) == 0 for a in t2.requests for b in t2.requests if a.group != b.group)
    t3 = make("doc_reuse")
    for a in t3.requests:
        for b in t3.requests:
            if a.id < b.id:
                n = lcp(a.tokens, b.tokens)
                assert (n > 3) if a.group == b.group else (n == 3)  # same document vs only the instruction


@pytest.mark.parametrize("kind", list(SMALL))
@pytest.mark.parametrize("C", [1, 2, 4, 8])
def test_oracle_reuse_equals_a_brute_force_prefix_oracle(kind, C):
    t = make(kind)
    assert tr.oracle_reuse(t, C) == brute_force_oracle(t, C)


def test_no_reuse_control_has_zero_reuse_and_reuse_traces_have_some():
    for C in (1, 4):
        s = tr.trace_stats(make("no_reuse"), C)
        assert s["oracle_reusable_tokens"] == 0 and s["requests_with_reuse"] == 0
    for kind in ("shared_prefix", "multi_turn", "doc_reuse"):
        assert tr.trace_stats(make(kind), 2)["oracle_reusable_tokens"] > 0, kind


def _hand_trace(prompts):
    return tr.Trace(tr._header("hand", 0, {}), tuple(tr.TraceRequest(f"r{i}", "g", tuple(p)) for i, p in enumerate(prompts)))


def test_first_token_rule_at_exact_chunk_multiples():
    # C = 2; the stored prompt has 2 complete chunks, [1,2] and [3,4]
    t = _hand_trace([(1, 2, 3, 4, 5), (1, 2, 3, 4), (1, 2, 3, 4, 9), (1, 2), (1, 2, 7)])
    # (1,2,3,4) is an exact multiple: L <= N-1 = 3 allows only 2 tokens, although both chunks are cached
    assert tr.oracle_reuse(t, 2) == [0, 2, 4, 0, 2]


def test_lru_restore_lengths_on_hand_worked_sequences():
    # C = 2. P has chunks p0 p1 p2 (all eligible); Q shares p0 p1; R shares p0 only.
    P, Q, R = (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 9), (1, 2, 8, 8, 8)
    t = _hand_trace([P, Q, R, P])
    assert tr.lru_restore_lengths(t, 2, 100) == tr.oracle_reuse(t, 2) == [0, 4, 2, 6]
    # capacity 3: P stores p0 p1 p2; Q hits p0 p1; R hits p0, stores r1 evicting p2; P hits p0 p1 only
    assert tr.lru_restore_lengths(t, 2, 3) == [0, 4, 2, 4]
    # capacity 2: P's own p2 evicts p0, so Q finds no leading chunk; Q re-stores p0 p1; R hits p0 and
    # stores r1 evicting p1; P hits p0 only
    assert tr.lru_restore_lengths(t, 2, 2) == [0, 0, 2, 2]
    assert tr.lru_restore_lengths(t, 2, 0) == [0, 0, 0, 0]
    # storing skips a present chunk without refreshing it: Y = X[:4] makes only x0 eligible (touched),
    # x1 stays least recent and is the one Z evicts, so X still restores x0
    X, Y, Z = (1, 2, 3, 4, 5), (1, 2, 3, 4), (7, 8, 9)
    assert tr.lru_restore_lengths(_hand_trace([X, Y, Z, X]), 2, 2) == [0, 2, 0, 2]


def test_stats_working_set_budgets_and_rho():
    t = _hand_trace([(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 9), (1, 2, 8, 8, 8)])
    s = tr.trace_stats(t, 2, chunk_bytes=100, budgets_bytes=(0, 199, 200, 1000))
    assert s["unique_chunks"] == 4 and s["working_set_bytes"] == 400  # p0 p1 p2 r1
    assert s["prompt_tokens"] == 17 and s["oracle_reusable_tokens"] == 6 and s["prompt_len"]["median"] == 5
    assert [(b["capacity_chunks"], b["rho"]) for b in s["budgets"]] == [(0, None), (1, 400 / 199), (2, 2.0), (10, 0.4)]
    assert s["synthetic"] is True


def test_generators_reject_inconsistent_parameters():
    with pytest.raises(ValueError, match="marker"):
        tr.no_reuse(1, n_requests=20, min_len=1, max_len=3, vocab=(0, 10))
    with pytest.raises(ValueError, match="suffix_min"):
        tr.shared_prefix(1, n_requests=2, prefix_len=4, suffix_min=5, suffix_max=3, vocab=V)
    with pytest.raises(ValueError, match="non-negative"):
        tr.doc_reuse(1, n_requests=2, n_docs=2, instruction_len=-1, doc_min=1, doc_max=2, question_min=1,
                     question_max=2, zipf_s=1.0, vocab=V)
