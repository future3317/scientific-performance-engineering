from __future__ import annotations

from benchmark.external.evaluator import evaluate_paired_rows, summarize_policy


def _rows() -> list[dict]:
    return [
        {
            "task_id": "a",
            "baseline_score": 0.2,
            "skill_score": 0.8,
            "policy_decisions": {"p": {"reuse": True, "reason": "support"}},
        },
        {
            "task_id": "b",
            "baseline_score": 0.8,
            "skill_score": 0.3,
            "policy_decisions": {"p": {"reuse": True, "reason": "support"}},
        },
        {
            "task_id": "c",
            "baseline_score": 0.4,
            "skill_score": 0.9,
            "policy_decisions": {"p": {"reuse": False, "reason": "abstain"}},
        },
    ]


def test_summary_separates_deployed_and_all_denominators() -> None:
    summary = summarize_policy(_rows(), "p")
    assert summary["deployed"] == 2
    assert summary["coverage"] == 2 / 3
    assert summary["unsupported_deployed"] == 1
    assert summary["unsupported_deployed_rate"] == 0.5
    assert summary["unsupported_all"] == 1
    assert summary["unsupported_all_rate"] == 1 / 3
    assert summary["repair_recall"] == 1 / 2


def test_evaluator_reports_task_metrics_and_deterministic_bootstrap() -> None:
    first = evaluate_paired_rows(_rows(), ["p"], bootstrap_seed=7, bootstrap_reps=50)
    second = evaluate_paired_rows(_rows(), ["p"], bootstrap_seed=7, bootstrap_reps=50)
    assert first == second
    assert first["policies"]["p"]["support_decision_regret"] == 1 / 3
    assert "bootstrap" in first["policies"]["p"]


def test_macro_average_is_cell_macro_not_pooled() -> None:
    result = evaluate_paired_rows(
        [
            {**_rows()[0], "benchmark": "a", "model": "m1"},
            {**_rows()[1], "benchmark": "b", "model": "m1"},
        ],
        ["p"],
        bootstrap_seed=1,
        bootstrap_reps=10,
    )
    assert result["macro_cells"] == 2
    assert result["macro"]["p"]["coverage"] == 1.0
