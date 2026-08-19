from __future__ import annotations

from benchmark.harness import scoring


def test_grader_never_converts_failed_science_to_verified_speedup():
    result = scoring.score_task({
        "verdict": "pass",
        "correctness_pass": False,
        "scientific_gates": {"finite_loss": False},
        "verified_speedup": {"median_speedup": 9.0, "verified": True},
    })
    assert result["gates_passed"] is False
    assert result["task_score"] == 0.0


def test_grader_preserves_protocol_valid_failure_as_zero_score():
    result = scoring.score_task({
        "verdict": "error",
        "correctness_pass": False,
        "scientific_gates": {"finite_loss": False},
        "protocol_failure": False,
        "execution_validity": "valid",
    })
    assert result.get("task_score") == 0.0
