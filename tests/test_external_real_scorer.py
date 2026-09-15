from __future__ import annotations

from experiments.external_real.models import PairedOutcome, PolicyDecision
from experiments.external_real.scorer import paired_task_bootstrap, score_policies


def _outcome(task_id: str, seed: int, baseline: bool, skill: bool) -> PairedOutcome:
    return PairedOutcome(
        task_id=task_id,
        repeat_seed=seed,
        baseline_score=1.0 if baseline else 0.0,
        skill_score=1.0 if skill else 0.0,
        baseline_success=baseline,
        skill_success=skill,
        native_metrics={"official_score": 1.0 if skill else 0.0},
        execution_valid=True,
        failure_reason=None,
    )


def test_score_uses_shared_potential_outcomes_and_explicit_denominators() -> None:
    decisions = {
        "acre": [
            PolicyDecision.create("task-1", "acre", True, "support", "2026-09-15T12:00:00+00:00"),
            PolicyDecision.create("task-2", "acre", False, "abstain", "2026-09-15T12:00:00+00:00"),
        ]
    }
    outcomes = [_outcome("task-1", 0, False, True), _outcome("task-2", 0, True, False)]

    report = score_policies(decisions, outcomes, lambda outcome, deploy: {
        "score": outcome.skill_score if deploy else outcome.baseline_score,
        "success": outcome.skill_success if deploy else outcome.baseline_success,
    })

    metrics = report["acre"]
    assert metrics["coverage"] == 0.5
    assert metrics["repairs_all"] == 1
    assert metrics["repairs_deployed"] == 1
    assert metrics["regressions_deployed"] == 0
    assert metrics["repair_recall"] == 1.0
    assert metrics["paired_score_gain_mean"] == 0.0


def test_bootstrap_samples_whole_tasks_not_individual_repeats() -> None:
    rows = [
        {"task_id": "task-1", "repeat_seed": 0, "deployed": True, "repair": True, "regression": False},
        {"task_id": "task-1", "repeat_seed": 1, "deployed": True, "repair": True, "regression": False},
        {"task_id": "task-2", "repeat_seed": 0, "deployed": False, "repair": False, "regression": False},
    ]

    report = paired_task_bootstrap(rows, reps=20, seed=4)

    assert report["cluster_unit"] == "task_id"
    assert report["replicates"] == 20
    assert len(report["coverage"]) == 20
