#!/usr/bin/env python3
"""Standalone assert-script tests for harness/stats.py (no pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness import stats


def main() -> None:
    # --- percentile/median/iqr/mad --------------------------------------------
    values = [1.0, 2.0, 3.0, 4.0, 100.0]  # outlier-tolerant stats
    assert stats.median(values) == 3.0
    assert stats.iqr(values) == 2.0  # q1=2.0, q3=4.0 (linear interpolation lands on points)
    assert stats.mad(values) == 1.0
    assert stats.percentile([], 0.5) is None

    # --- paired improvements ----------------------------------------------------
    imp = stats.paired_improvements([10.0, 10.0], [5.0, 4.0], higher_is_better=False)
    assert imp == [100.0, 150.0]
    imp = stats.paired_improvements([10.0, 10.0], [11.0, 12.0], higher_is_better=True)
    assert abs(imp[0] - 10.0) < 1e-9 and abs(imp[1] - 20.0) < 1e-9
    assert stats.paired_improvements([0.0], [1.0], True) == []

    # --- deterministic bootstrap --------------------------------------------------
    sample = [5.0, 6.0, 7.0, 8.0, 9.0]
    ci1 = stats.bootstrap_ci(sample)
    ci2 = stats.bootstrap_ci(sample)
    assert ci1 == ci2, "bootstrap must be deterministic (random.Random(0))"
    assert ci1[0] <= stats.median(sample) <= ci1[1]
    assert stats.bootstrap_ci([], 0.95, 2000) == (None, None)

    # --- verdict logic --------------------------------------------------------------
    # Clear 50% improvement, tight distribution -> verified.
    verdict = stats.robust_speedup_verdict(
        baseline_runs=[10.0, 10.1, 9.9, 10.0, 10.05],
        candidate_runs=[6.6, 6.7, 6.6, 6.65, 6.7],
        higher_is_better=False,
        min_improvement_percent=5.0,
        noise_floor_percent=2.0,
    )
    assert verdict["verified"] is True and verdict["inconclusive"] is False, verdict
    assert 1.4 < verdict["median_speedup"] < 1.6
    assert verdict["ci_low"] <= verdict["median_speedup"] <= verdict["ci_high"]

    # No difference -> inconclusive (never zero-speedup).
    same = [10.0, 10.1, 9.9, 10.0, 10.05]
    verdict = stats.robust_speedup_verdict(same, list(reversed(same)), False, 5.0, 2.0)
    assert verdict["verified"] is False and verdict["inconclusive"] is True
    assert "required margin" in verdict["reason"]

    # Unpaired / empty inputs -> inconclusive with reasons.
    verdict = stats.robust_speedup_verdict([], [1.0], False, 5.0, 2.0)
    assert verdict["inconclusive"] and verdict["reason"] == "no measurement runs"
    verdict = stats.robust_speedup_verdict([1.0, 2.0], [1.0], False, 5.0, 2.0)
    assert verdict["inconclusive"] and "not paired" in verdict["reason"]

    # --- noise floor ------------------------------------------------------------------
    floor = stats.estimate_noise_floor(same, list(reversed(same)), higher_is_better=False)
    assert floor["noise_floor_percent_observed"] is not None
    assert floor["noise_floor_percent_observed"] < 5.0, floor
    assert stats.estimate_noise_floor([], [], False)["noise_floor_percent_observed"] is None

    # A huge control-vs-control spread raises the effective floor and can flip verdicts.
    verdict = stats.robust_speedup_verdict(
        baseline_runs=[10.0, 10.0, 10.0],
        candidate_runs=[9.0, 9.0, 9.0],
        higher_is_better=False,
        min_improvement_percent=5.0,
        noise_floor_percent=2.0,
        control_a_runs=[10.0, 5.0, 20.0],
        control_b_runs=[5.0, 20.0, 10.0],
    )
    assert verdict["inconclusive"] is True, verdict

    print("test_stats: OK")


if __name__ == "__main__":
    main()
