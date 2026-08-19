#!/usr/bin/env python3
"""Population-validity and empirical-calibration report contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.taskgen.validate_population import _empirical_flags, build_report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report, errors = build_report(repo_root / "benchmark" / "tasks")
    assert errors == [], errors
    assert report["num_tasks"] == 30
    assert report["track_counts"] == {"spe_core": 16, "sciml": 11, "evolution": 3}
    assert report["empirical_calibration"]["status"] == "pending"
    assert report["empirical_calibration"]["calibration_gate"] == "blocked"
    assert report["task_calibration"]["CORE-COMPILE-DYNAMIC-11"]["status"] == "eligible"
    assert report["formal_50_task_results"] == "not_claimed"
    assert set(report["empirical_rejection_flags"]) == {
        "oracle_effect_too_small", "noise_too_high", "oracle_effect_unstable",
        "baseline_already_optimal", "semantic_gate_too_weak", "repair_pattern_duplicate",
        "difficulty_ceiling", "difficulty_floor", "platform_direction_flip",
        "agent_shortcut_detected",
    }
    print("test_population: OK")


def test_empirical_floor_uses_observed_control_noise() -> None:
    import json
    import tempfile

    specs = [{
        "task_id": "DYNAMIC",
        "track": "spe_core",
        "scientific_gates": ["finite_loss"],
        "measurement": {"noise_floor_percent": 2.0, "min_improvement_percent": 5.0},
    }]
    payload = {
        "tasks": [{
            "task_id": "DYNAMIC",
            "oracle_ci_low_percent": 61.2165,
            "oracle_ci_high_percent": 233.8057,
            "control_noise_percent": [51.8665, 61.2165, 58.4659],
            "baseline_speedups": [1.20, 1.30],
        }]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name
    flags, calibration = _empirical_flags(specs, __import__("pathlib").Path(path), [])
    assert flags["oracle_effect_too_small"] == []
    assert flags["oracle_effect_unstable"] == ["DYNAMIC"]
    assert flags["noise_too_high"] == ["DYNAMIC"]
    assert flags["baseline_already_optimal"] == ["DYNAMIC"]
    assert calibration["calibration_gate"] == "blocked"


def test_empirical_floor_can_clear_high_declared_noise() -> None:
    import json
    import tempfile

    specs = [{
        "task_id": "DYNAMIC",
        "track": "spe_core",
        "scientific_gates": ["finite_loss"],
        "measurement": {"noise_floor_percent": 2.0, "min_improvement_percent": 5.0},
    }]
    payload = {"tasks": [{
        "task_id": "DYNAMIC",
        "oracle_ci_low_percent": 156.0413,
        "oracle_ci_high_percent": 233.8057,
        "control_noise_percent": [51.8665, 61.2165, 58.4659],
    }]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle)
        path = handle.name
    flags, calibration = _empirical_flags(specs, __import__("pathlib").Path(path), [])
    assert flags["oracle_effect_too_small"] == []
    assert flags["oracle_effect_unstable"] == []
    assert flags["noise_too_high"] == []
    assert calibration["calibration_gate"] == "ready_for_review"


if __name__ == "__main__":
    main()
