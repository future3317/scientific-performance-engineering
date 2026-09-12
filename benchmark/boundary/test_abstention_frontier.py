from __future__ import annotations

from benchmark.boundary.policy_eval import Decision, evaluate_policy, first_blocking_reason


def test_abstention_decomposition_separates_missed_support_and_correct_abstention() -> None:
    evaluation = [
        {"id": "supported", "family": "f", "seed": 1, "context": {"x": 1}},
        {"id": "unsupported", "family": "f", "seed": 1, "context": {"x": 2}},
    ]
    scores = {
        "supported": {"tau": 0.5, "scientific_gate": True, "ci_width": 0.1},
        "unsupported": {"tau": -0.2, "scientific_gate": True, "ci_width": 0.8},
    }
    report = evaluate_policy(
        lambda context: Decision("abstain", 0.0, 0.2 if context["x"] == 1 else 0.9, "A"),
        calibration=[], evaluation=evaluation, score_records=scores, delta=0.0,
    )
    assert report["missed_supported_deployment"] == 1
    assert report["correct_abstention"] == 1
    assert report["abstention_reason_counts"] == {"A": 2}
    assert report["ci_width_summary"]["missed_supported_mean"] == 0.1
    assert report["ci_width_summary"]["correct_abstention_mean"] == 0.8
    assert report["uncertainty_interpretation"] == "fact_only_until_registered_comparison"


def test_blocking_reason_has_registered_precedence() -> None:
    assert first_blocking_reason(["R", "S", "A"]) == "A"
    assert first_blocking_reason(["R", "S"]) == "S"
    assert first_blocking_reason([]) is None
