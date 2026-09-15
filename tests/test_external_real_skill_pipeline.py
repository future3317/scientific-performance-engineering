from __future__ import annotations

import hashlib

from experiments.external_real.models import PairedOutcome
from experiments.external_real.skill_pipeline import FrozenSkill, freeze_skill, verify_skill
from experiments.external_real.support import (
    FALLBACK_TO_BASE_AGENT,
    SupportEvidence,
    count_evidence_sources,
    decide_policies,
    fit_support_model,
    support_evidence_from_outcome,
)


def _row(task_id: str, family: str, effect: float, seed: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "repeat_seed": seed,
        "context": {"task_family": family},
        "effect": effect,
        "execution_valid": True,
        "scientific_gate": True,
    }


def test_frozen_skill_hash_binds_body_scripts_resources_and_semantic_abstract() -> None:
    skill = freeze_skill(
        body="Use the task grammar before acting.",
        scripts=("helper.py",),
        resources=("scienceworld-actions.md",),
        semantic_abstract="scienceworld action planning",
        benchmark="scienceworld",
        generation_seed=2,
    )

    expected = hashlib.sha256(
        b'{"benchmark":"scienceworld","body":"Use the task grammar before acting.","generation_seed":2,"resources":["scienceworld-actions.md"],"scripts":["helper.py"],"semantic_abstract":"scienceworld action planning"}'
    ).hexdigest()
    assert isinstance(skill, FrozenSkill)
    assert skill.skill_hash == expected


def test_support_model_uses_only_train_rows_and_abstains_when_unresolved() -> None:
    rows = [_row("positive", "navigation", 1.0, seed) for seed in range(10)]
    rows += [_row("negative", "measurement", -1.0, seed) for seed in range(10)]
    contexts = {
        "positive": {"task_family": "navigation"},
        "negative": {"task_family": "measurement"},
        "unseen": {"task_family": "chemistry"},
    }

    model = fit_support_model(rows, contexts, delta=0.05)

    assert model.training_task_ids == ("negative", "positive")
    assert model.decide(contexts["unseen"]).deploy is False
    assert model.decide(contexts["unseen"]).reason == FALLBACK_TO_BASE_AGENT


def test_upstream_and_acre_evidence_are_counted_separately() -> None:
    evidence = [
        SupportEvidence("task-1", 0, 1.0, 1.5, True, True, 0.5, "upstream_verification"),
        SupportEvidence("task-2", 0, 1.0, 0.0, True, False, -1.0, "acre_additional_query"),
    ]

    assert count_evidence_sources(evidence) == {
        "upstream_verification": 1,
        "acre_additional_query": 1,
    }


def test_valid_upstream_pair_is_converted_without_rerunning_it() -> None:
    outcome = PairedOutcome("task-1", 3, 1.0, 1.5, True, True, {"official_score": 1.5}, True, None)

    evidence = support_evidence_from_outcome(outcome, "upstream_verification")

    assert evidence == SupportEvidence("task-1", 3, 1.0, 1.5, True, True, 0.5, "upstream_verification")


def test_global_verification_withholds_invalid_skill_before_acquisition() -> None:
    skill = freeze_skill("body", (), (), "abstract", "scienceworld", 0)

    class Adapter:
        def run_pair(self, task, frozen_skill, seed):
            return PairedOutcome(task.task_id, seed, 1.0, 0.0, True, False, {}, True, "skill_failed")

    task = type("Task", (), {"task_id": "task-1"})()
    report = verify_skill(Adapter(), skill, [task])

    assert report.outcome == "WITHHOLD_SKILL"


def test_policy_decision_falls_back_when_support_is_unresolved() -> None:
    rows = [_row("positive", "navigation", 1.0, seed) for seed in range(4)]
    model = fit_support_model(rows, {"positive": {"task_family": "navigation"}}, delta=0.05)
    task = type("Task", (), {"task_id": "unseen", "public_context": {"task_family": "measurement"}})()

    decisions = decide_policies(model, [task], policy_name="acre", decision_timestamp="2026-09-15T12:00:00+00:00")

    assert decisions[0].deploy is False
    assert decisions[0].reason == FALLBACK_TO_BASE_AGENT
