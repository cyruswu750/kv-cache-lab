"""M3 statistics (kvlab/m3_stats.py) and M3 sample validation (kvlab/m3_bench.py), CPU only."""

import numpy as np
import pytest

from kvlab import m3_bench as mb
from kvlab import m3_stats as ms
from kvlab import perf


def test_stratified_indices_never_join_two_strata():
    lengths = [5, 7, 3]
    idx = ms.stratified_indices(lengths, block_len=4, n_boot=500, rng=np.random.default_rng(0))
    assert idx.shape == (500, 15)
    bounds = [(0, 5), (5, 12), (12, 15)]
    for (lo, hi), cols in zip(bounds, (slice(0, 5), slice(5, 12), slice(12, 15))):
        part = idx[:, cols]
        assert part.min() >= lo and part.max() < hi  # resampled within its own stratum only


def test_boot_ci_point_estimates_and_definitions():
    a = [np.array([10.0, 12.0, 11.0, 13.0]), np.array([20.0, 22.0])]
    b = [np.array([8.0, 9.0, 9.0, 10.0]), np.array([10.0, 11.0])]
    aa, bb = np.concatenate(a), np.concatenate(b)
    assert ms.boot_ci(a, b, "median_d", 2)["estimate"] == pytest.approx(np.median(bb - aa))
    assert ms.boot_ci(a, b, "mean_d", 2)["estimate"] == pytest.approx(np.mean(bb - aa))
    assert ms.boot_ci(a, b, "marginal_median_ratio", 2)["estimate"] == pytest.approx(np.median(aa) / np.median(bb))
    assert ms.boot_ci(a, b, "marginal_mean_ratio", 2)["estimate"] == pytest.approx(aa.mean() / bb.mean())
    assert ms.boot_ci(a, b, "median_paired_ratio", 2)["estimate"] == pytest.approx(np.median(aa / bb))
    # the two ratio definitions differ on this data: a marginal ratio is not a paired ratio
    assert ms.boot_ci(a, b, "marginal_median_ratio", 2)["estimate"] != pytest.approx(
        ms.boot_ci(a, b, "median_paired_ratio", 2)["estimate"])
    with pytest.raises(ValueError):
        ms.boot_ci(a, [b[0]], "median_d", 2)


def _strata(shift, n_strata=10, n=12, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    a = [10 + rng.normal(0, noise, n) for _ in range(n_strata)]
    b = [x + shift + rng.normal(0, noise, n) for x in a]
    return a, b


def test_classify_case_rules():
    a, b = _strata(-3.0)
    sess = [f"s{i // 2}" for i in range(10)]
    r = ms.classify_case(a, b, sess, span_s=300, longest_regime_s=2.0)
    assert r["verdict"] == "batched_faster" and r["R1_all_block_lengths_agree"] and r["R2_all_sessions_agree"]
    assert ms.classify_case(a, b, sess, span_s=15, longest_regime_s=2.0)["verdict"] == "inconclusive"  # R3: too short
    a0, b0 = _strata(0.0)
    assert ms.classify_case(a0, b0, sess, 300, 2.0)["verdict"] == "inconclusive"                        # R1
    b2 = [x.copy() for x in b]
    b2[0] += 10.0  # session s0 now slower with the candidate
    b2[1] += 10.0
    r2 = ms.classify_case(a, b2, sess, 300, 2.0)
    assert not r2["R2_all_sessions_agree"] and r2["verdict"] == "inconclusive"
    up = ms.classify_case(*_strata(+3.0), sess, 300, 2.0)
    assert up["verdict"] == "batched_slower"
    # strongly autocorrelated differences -> small effective n -> inconclusive (R3)
    a3 = [np.full(12, 10.0) for _ in range(4)]
    b3 = [x - 3 + np.linspace(-1, 1, 12) for x in a3]  # lag-1 rho ~0.75 -> n_eff ~1.7 per stratum
    r3 = ms.classify_case(a3, b3, ["s0", "s0", "s1", "s1"], 300, 2.0)
    assert r3["R3"]["ar1_effective_n"] < 10 and r3["verdict"] == "inconclusive"


def test_regime_durations():
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert ms.regimes(t, [False, True, True, False, True, False, True]) == [1.0, 0.0, 0.0]
    assert ms.regimes(t, [True] * 7) == [6.0]
    assert ms.regimes(t, [False] * 7) == []


def _sample(path, backend, mode="plain", L=1024, S=1, **over):
    n = L // perf.C
    s = {"path": path, "backend": backend, "mode": mode, "L": L, "S": S, "e2e_ns": 30_000_000,
         **perf.expected(path, L, S)}
    if path == "disk":
        s["resident_fraction_before"] = 1.0
    if mode == "synced":
        s["phases_ns"] = {p: 1 for p in perf.PHASES}
    s["last_restore"] = None if path == "full" else {
        "backend": backend, "chunks_planned": n, "chunks_loaded": n, "h2d_bytes": n * 7_340_032,
        "device_destination_bytes": n * 7_340_032, "placement_ops": 0 if backend == "baseline" else 1,
        "h2d_copy_calls": mb.expected_copy_calls(backend, n, 28), "device_scratch_bytes": 0}
    s.update(over)
    return s


@pytest.mark.parametrize("combo", mb.COMBOS)
def test_well_formed_m3_samples_are_valid(combo):
    assert mb.sample_problems(_sample(*combo)) == []


def test_m3_samples_that_did_not_run_the_requested_backend_are_excluded():
    ok = _sample("host", "batched")
    assert any("backend" in p for p in mb.sample_problems({**ok, "last_restore": {**ok["last_restore"], "backend": "baseline"}}))
    assert any("h2d_copy_calls" in p for p in mb.sample_problems(
        {**ok, "last_restore": {**ok["last_restore"], "h2d_copy_calls": 4 * 56}}))
    assert any("restore record" in p for p in mb.sample_problems(
        {**ok, "last_restore": {**ok["last_restore"], "chunks_loaded": 3}}))
    assert mb.sample_problems({**ok, "last_restore": None}) == ["no restore record"]
    assert any("full path ran a restore" in p for p in mb.sample_problems(
        _sample("full", "none", last_restore=ok["last_restore"])))
    assert any("phases_ns" in p for p in mb.sample_problems(_sample("disk", "baseline", "synced", phases_ns={})))
    assert any("sources" in p for p in mb.sample_problems(_sample("disk", "batched", sources=["host"] * 4)))
