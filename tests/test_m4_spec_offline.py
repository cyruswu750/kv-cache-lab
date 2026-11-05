"""M4_EVAL_V1 offline properties (CPU, key-level only; no model): the trace statistics and budgets in
docs/M4B_PROTOCOL.md §3-§5, and the warmup-coverage proof required by §6 before any freeze."""

import pytest

from kvlab import m4_spec as ms
from kvlab import trace_models as tm
from kvlab import traces as tr

SPEC = ms.M4_EVAL_V1
C = SPEC["chunk_size"]


@pytest.fixture(scope="module")
def ev():
    return {t: ms.trace(SPEC, t, "evaluation") for t in ms.TRACE_NAMES}


def test_evaluation_traces_and_budgets_match_the_protocol_tables(ev):
    want = {"T0": ("f04ff29d4817", 48, 63538, 0, 222), "T1": ("5f9da2bf6661", 64, 105096, 96768, 6),
            "T2": ("05191908301a", 48, 57337, 36096, 56), "T3": ("1c30ff4fbfcb", 64, 90692, 69120, 58)}
    for t, (sha, n, tok, oracle, W) in want.items():
        s = tr.trace_stats(ev[t], C)
        assert (ev[t].digest[:12], s["n_requests"], s["prompt_tokens"], s["oracle_reusable_tokens"],
                s["unique_chunks"]) == (sha, n, tok, oracle, W), t
    b = ms.budgets(SPEC, "evaluation", ev)
    assert {t: (b[t]["C1PA_W"], b[t]["C1PA_HALF"], b[t]["host_slots"], b[t]["disk_files"]) for t in b} == \
        {"T0": (58, 29, 29, 60), "T1": (6, 3, 3, 8), "T2": (56, 28, 28, 58), "T3": (58, 29, 29, 60)}


def test_offline_restored_tokens_at_half_capacity_match_the_protocol(ev):
    b = ms.budgets(SPEC, "evaluation", ev)
    want = {"T1": (0, 48384), "T2": (8192, 8448), "T3": (63744, 65536), "T0": (0, 0)}
    for t, (lru, pa) in want.items():
        p = [list(r.tokens) for r in ev[t].requests]
        assert sum(x["restore_len"] for x in tm.lru_model(p, C, b[t]["C1PA_HALF"])) == lru
        assert sum(x["restore_len"] for x in tm.pa_model(p, C, b[t]["C1PA_HALF"])) == pa
    p = [list(r.tokens) for r in ev["T0"].requests]
    assert sum(x["evictions"] for x in tm.pa_model(p, C, 58)) == 222 - 58
    for t in ("T1", "T2", "T3"):  # C2 with D >= W loses nothing: restore lengths equal the oracle
        p = [list(r.tokens) for r in ev[t].requests]
        m = tm.tiered_model(p, C, b[t]["host_slots"], b[t]["disk_files"])
        assert [x["restore_len"] for x in m] == tm.oracle_lengths(p, C) and sum(x["disk_evictions"] for x in m) == 0


def test_warmup_coverage_proof():
    w = {t: ms.trace(SPEC, t, "warmup") for t in ms.TRACE_NAMES}
    assert [len(w[t].requests) for t in ms.TRACE_NAMES] == [4, 4, 6, 6]
    b = ms.budgets(SPEC, "warmup", w)
    tot = {"C1PA_W": [0, 0, 0], "C1PA_HALF": [0, 0, 0], "C2": [0, 0, 0, 0, 0]}
    for t in ms.TRACE_NAMES:
        p = [list(r.tokens) for r in w[t].requests]
        for c in ("C1PA_W", "C1PA_HALF"):
            for x in tm.pa_model(p, C, b[t][c]):
                tot[c][0] += bool(x["restore_len"])
                tot[c][1] += bool(x["stored"])
                tot[c][2] += x["evictions"] + x["stopped"]
        for x in tm.tiered_model(p, C, b[t]["host_slots"], b[t]["disk_files"]):
            tot["C2"][0] += bool(x["restore_len"])
            tot["C2"][1] += bool(x["stored"])
            tot["C2"][2] += x["sources"].count("host")
            tot["C2"][3] += x["sources"].count("disk")
            tot["C2"][4] += x["spills"]
    assert tot["C1PA_W"][0] and tot["C1PA_W"][1]
    assert tot["C1PA_HALF"][0] and tot["C1PA_HALF"][1] and tot["C1PA_HALF"][2]
    assert all(tot["C2"]), tot
