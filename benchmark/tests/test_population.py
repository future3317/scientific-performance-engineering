#!/usr/bin/env python3
"""Population-validity and empirical-calibration report contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.taskgen.validate_population import build_report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report, errors = build_report(repo_root / "benchmark" / "tasks")
    assert errors == [], errors
    assert report["num_tasks"] == 20
    assert report["track_counts"] == {"spe_core": 11, "sciml": 7, "evolution": 2}
    assert report["empirical_calibration"]["status"] == "pending"
    assert report["empirical_calibration"]["calibration_gate"] == "blocked"
    assert report["formal_50_task_results"] == "not_claimed"
    assert set(report["empirical_rejection_flags"]) == {
        "oracle_effect_too_small", "noise_too_high", "oracle_effect_unstable",
        "baseline_already_optimal", "semantic_gate_too_weak", "repair_pattern_duplicate",
        "difficulty_ceiling", "difficulty_floor", "platform_direction_flip",
        "agent_shortcut_detected",
    }
    print("test_population: OK")


if __name__ == "__main__":
    main()
