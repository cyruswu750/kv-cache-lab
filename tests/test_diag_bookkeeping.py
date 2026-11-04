"""M2.1 diagnostic sample validation (kvlab/diag_harness.py), CPU only."""

import pytest

from kvlab import diag_harness as dh
from kvlab import perf


def sample(path, mode, L=512, S=1, **over):
    s = {"path": path, "mode": mode, "L": L, "S": S, "e2e_ns": 30_000_000, **dh.expected(path, L, S)}
    if path == "disk":
        s["resident_fraction_before"] = 1.0
    if mode == "synced":
        s["phases_ns"] = ({p: 1 for p in ("ingest", "prefill", "select")} if path == "full_nocache"
                          else {p: 1 for p in perf.PHASES})
    if mode == "events":
        phases = dh.NOCACHE_PHASES if path == "full_nocache" else dh.ENGINE_PHASES
        s["events"] = {"marks": [{"phase": p, "gpu_ms": 0.5 * i} for i, p in enumerate(phases)]}
    s.update(over)
    return s


@pytest.mark.parametrize("path", dh.DIAG_PATHS)
@pytest.mark.parametrize("mode", dh.MODES)
def test_well_formed_samples_of_every_path_and_mode_are_valid(path, mode):
    assert dh.sample_problems(sample(path, mode)) == []


def test_nocache_baseline_restores_and_admits_nothing_even_when_full_would_admit():
    assert dh.expected("full_nocache", 512, 256) == {"restore_len": 0, "sources": [], "stored_chunks": []}
    assert perf.expected("full", 512, 256)["stored_chunks"] == [2]  # the M2 baseline admits the new chunk
    bad = sample("full_nocache", "plain", stored_chunks=[2])
    assert any("stored_chunks" in p for p in dh.sample_problems(bad))


def test_wrong_tier_missing_marks_and_bad_event_times_are_excluded():
    assert any("sources" in p for p in dh.sample_problems(sample("disk", "events", sources=["host", "host"])))
    ev = sample("host", "events")
    ev["events"]["marks"].pop(2)  # the "prefill" mark is missing
    assert any("event marks" in p for p in dh.sample_problems(ev))
    neg = sample("full", "events")
    neg["events"]["marks"][1]["gpu_ms"] = -1.0
    assert dh.sample_problems(neg) == ["negative or missing event time"]
    assert any("phases_ns" in p for p in dh.sample_problems(sample("full_nocache", "synced", phases_ns={"ingest": 1})))
    assert any("page cache" in p for p in dh.sample_problems(sample("disk", "plain", resident_fraction_before=0.5)))


def test_probe_samples_need_a_positive_time_and_a_passed_work_check():
    ok = {"path": "probe_launch", "mode": "plain", "L": 512, "S": None, "e2e_ns": 5_000_000, "probe_check": True}
    assert dh.sample_problems(ok) == []
    assert dh.sample_problems({**ok, "probe_check": False})        # e.g. fewer than 500 launches observed
    assert dh.sample_problems({**ok, "e2e_ns": 0})
    assert set(dh.PROBES) & set(dh.DIAG_PATHS) == set()            # probes are never latency paths
