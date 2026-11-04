"""M2.1 dependence-aware statistics (kvlab/perf_stats.py), CPU only."""

import json
from pathlib import Path

import numpy as np
import pytest

from kvlab import perf_stats as ps


def test_paired_series_keeps_only_common_reps_in_chronological_order():
    reps, d = ps.paired_series({3: 5.0, 1: 4.0, 2: 9.0}, {2: 1.0, 1: 1.0, 7: 0.0})
    assert reps == [1, 2] and d.tolist() == [3.0, 8.0]


def test_block_indices_are_circular_runs_of_consecutive_reps():
    rng = np.random.default_rng(0)
    idx = ps.block_bootstrap_indices(10, 4, 50, rng)
    assert idx.shape == (50, 10) and idx.min() >= 0 and idx.max() <= 9
    for row in idx:
        for start in (0, 4):  # full blocks at positions 0..3 and 4..7
            blk = row[start:start + 4]
            assert all((b - a) % 10 == 1 for a, b in zip(blk, blk[1:]))
    assert ps.block_bootstrap_indices(5, 99, 3, rng).shape == (3, 5)  # block longer than series is clipped
    with pytest.raises(ValueError):
        ps.block_bootstrap_indices(0, 1, 3, rng)


def test_block_bootstrap_widens_the_interval_for_a_clustered_series():
    rng = np.random.default_rng(1)
    # 40 reps: alternating regimes of 10 reps (fast -10 ms, slow +40 ms), small noise.
    x = np.concatenate([np.full(10, -10.0), np.full(10, 40.0)] * 2) + rng.normal(scale=0.5, size=40)
    iid = ps.block_bootstrap_ci(x, 1, stat="mean", seed=3)
    blk = ps.block_bootstrap_ci(x, 8, stat="mean", seed=3)
    assert iid["estimate"] == blk["estimate"] == pytest.approx(x.mean())
    assert (blk["ci"][1] - blk["ci"][0]) > 1.3 * (iid["ci"][1] - iid["ci"][0])
    assert iid["reliable_block_count"] and blk["reliable_block_count"]  # 40 / 8 = 5 blocks
    assert not ps.block_bootstrap_ci(x[:20], 8)["reliable_block_count"]  # 2.5 blocks


def test_too_few_pairs_give_no_interval():
    assert ps.block_bootstrap_ci([1.0] * (ps.MIN_PAIRS - 1), 2)["status"] == "insufficient_samples"
    s = ps.sensitivity([1.0] * (ps.MIN_PAIRS - 1))
    assert all(r["status"] == "insufficient_samples" for r in s["mean"].values())
    assert s["mean_class_stable_across_blocks"] is False


def test_classification_never_reads_an_interval_containing_zero_as_a_winner():
    assert ps.classify([-3.0, -0.1]) == "restore_faster"
    assert ps.classify([0.2, 9.0]) == "recompute_faster"
    assert ps.classify([-3.0, 0.0]) == "inconclusive"
    assert ps.classify([-3.0, 40.0]) == "inconclusive"


def test_lag1_autocorrelation_and_effective_sample_size():
    assert ps.lag1_autocorr([1.0, -1.0] * 10) == pytest.approx(-0.95, abs=0.01)
    assert ps.lag1_autocorr([2.0] * 10) is None
    assert ps.lag1_autocorr([0.0] * 10 + [1.0] * 10) > 0.8
    assert ps.ar1_effective_n(40, 0.6) == pytest.approx(10.0)
    assert ps.ar1_effective_n(40, -0.3) == 40.0 and ps.ar1_effective_n(40, None) == 40.0


def test_selection_penalty_metric_and_status():
    d = np.array([5.0] * 10)  # restore is 5 ms slower than recompute on every pair
    r = ps.selection_penalty("restore", d, ci_mean=[4.0, 6.0])
    assert r["point_penalty_ms"] == 5.0 and r["penalty_status"] == "established"
    f = ps.selection_penalty("full", d, ci_mean=[4.0, 6.0])
    assert f["point_penalty_ms"] == 0.0 and f["mean_chosen_minus_other_ms"] == -5.0
    assert f["chosen_minus_other_ci_ms"] == [-6.0, -4.0] and f["penalty_status"] == "established_absent"
    amb = ps.selection_penalty("restore", d, ci_mean=[-1.0, 6.0])
    assert amb["point_penalty_ms"] == 5.0 and amb["penalty_status"] == "inconclusive"  # positive point, not established


def test_policy_totals_and_oracle_regret():
    cases = {"a": {"restore": 10.0, "full": 30.0}, "b": {"restore": 50.0, "full": 40.0}}
    t = ps.policy_totals(cases, {"r": {"a": "restore", "b": "restore"}, "f": {"a": "full", "b": "full"}})
    assert t["oracle_sum_ms"] == 50.0 and t["n_cases"] == 2
    assert t["r"] == {"sum_ms": 60.0, "regret_vs_oracle_ms": 10.0}
    assert t["f"] == {"sum_ms": 70.0, "regret_vs_oracle_ms": 20.0}


# ---- regression on the preserved M2 evidence (VAL-005): read only

M2_RUN = Path(__file__).resolve().parent.parent / "results" / "m2" / "run_20261103T131543"


@pytest.mark.skipif(not (M2_RUN / "evaluation_samples.json").exists(), reason="M2 evidence not present")
def test_m2_iid_interval_was_overconfident_for_a_clustered_held_out_series():
    from kvlab import perf
    ev = json.loads((M2_RUN / "evaluation_samples.json").read_text())
    valid, _ = perf.partition(ev["samples"])
    by = {}
    for s in valid:
        if s.get("stage") == "e2e" and s["L"] == 1024 and s["S"] == 64:
            by.setdefault(s["path"], {})[s["rep"]] = perf.ns_to_ms(s["e2e_ns"])
    reps, d = ps.paired_series(by["host"], by["full"])
    assert len(reps) == 40
    assert ps.lag1_autocorr(d) > 0.5                                  # strongly clustered
    old = perf.paired_difference(by["host"], by["full"], seed=1024 * 7 + 64)  # the M2 scoring call
    assert perf.measured_winner(old) == "restore"                    # M2 called it established
    blk = ps.block_bootstrap_ci(d, 8, stat="median", seed=0)
    assert ps.classify(blk["ci"]) == "inconclusive"                  # blocks of 8 reps do not support it


@pytest.mark.skipif(not (M2_RUN / "results.json").exists(), reason="M2 evidence not present")
def test_m2_zero_penalty_excluded_cases_with_a_positive_point_penalty():
    res = json.loads((M2_RUN / "results.json").read_text())
    assert res["held_out_decisions"]["total_penalty_ms"] == 0.0      # what M2 reported
    tie = [d for d in res["held_out"]["decisions"] if (d["tier"], d["L"], d["S"]) == ("host", 512, 1)][0]
    assert tie["measured_winner"] == "tie" and tie["penalty_ms"] is None  # excluded from the total
    assert tie["predicted_winner"] == "restore" and tie["measured_paired_diff"]["median_ms"] > 30  # restore slower
