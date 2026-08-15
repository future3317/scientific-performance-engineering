#!/usr/bin/env python3
"""Scoring keeps correctness-valid but unverified positives in the denominator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.harness.scoring import aggregate_track


def _task(speedup: float, verified: bool) -> dict:
    return {
        "gates_passed": True,
        "kind": "positive",
        "tripwired": False,
        "verified_speedup": {"median_speedup": speedup, "verified": verified},
        "diagnosis_correct": None,
        "cost": {},
        "inconclusive": not verified,
    }


def main() -> None:
    result = aggregate_track([_task(1.20, True), _task(0.98, False)])
    assert result["verified_optimization_rate"] == 0.5
    assert result["raw"]["speedups_in_all_valid_geomean"] == 2
    assert abs(result["geomean_speedup_all_valid"] - (1.20 * 0.98) ** 0.5) < 1e-6
    print("test_scoring: OK")


if __name__ == "__main__":
    main()
