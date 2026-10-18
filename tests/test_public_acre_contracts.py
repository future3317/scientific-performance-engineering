from __future__ import annotations

import pytest

from benchmark.external.contracts import (
    DecisionRecord,
    validate_no_outcome_fields,
    validate_public_context,
    validate_task_split,
)
from benchmark.external.manifest import PublicTreatmentManifest


def test_decision_record_rejects_outcome_fields() -> None:
    record = DecisionRecord(
        run_id="run-1",
        condition="D",
        task_id="task-1",
        task_index=0,
        stage="acquisition",
        policy="acre",
        decision_id="decision-1",
        reuse=True,
        score=0.8,
        uncertainty=0.1,
        reason="certificate",
        certificate_id="cert-1",
        skill_version="skill-v0",
        public_context_digest="ctx-digest",
        available_evidence_ids=("ev-1",),
    )
    validate_no_outcome_fields(record.to_dict())
    with pytest.raises(ValueError, match="outcome fields"):
        validate_no_outcome_fields({**record.to_dict(), "judge_score": 1.0})


def test_public_context_rejects_scoring_only_fields() -> None:
    context = {"benchmark": "scienceworld", "task_id": "task-1", "category": "physics"}
    validate_public_context(context, {"gold_actions", "hidden_score"})
    with pytest.raises(ValueError, match="scoring-only"):
        validate_public_context({**context, "gold_actions": ["open"]}, {"gold_actions"})


def test_task_split_requires_disjoint_task_ids() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_task_split(
            {"synthesis": ["a"], "calibration": ["b"], "evaluation": ["b"]}
        )
    validate_task_split(
        {"synthesis": ["a"], "calibration": ["b"], "evaluation": ["c"]}
    )


def test_manifest_contains_protocol_identity_and_deterministic_digest() -> None:
    first = PublicTreatmentManifest.create(
        revision="public-acre-treatment-20260915-v1",
        implementation_commit="abc123",
        benchmark_revisions={"scienceworld": "sw-1"},
        task_split={"synthesis": ["a"], "calibration": ["b"], "evaluation": ["c"]},
        model_config={"provider": "deepseek", "model": "deepseek-v4-flash"},
        budgets={"max_tokens": 768, "max_tool_calls": 8},
        seeds=[0, 1, 2],
        policies=["global", "semantic_router", "acre"],
    )
    second = PublicTreatmentManifest.create(
        revision="public-acre-treatment-20260915-v1",
        implementation_commit="abc123",
        benchmark_revisions={"scienceworld": "sw-1"},
        task_split={"synthesis": ["a"], "calibration": ["b"], "evaluation": ["c"]},
        model_config={"provider": "deepseek", "model": "deepseek-v4-flash"},
        budgets={"max_tokens": 768, "max_tool_calls": 8},
        seeds=[0, 1, 2],
        policies=["global", "semantic_router", "acre"],
    )
    assert first["manifest_digest"] == second["manifest_digest"]
    assert first["results_claimed"] is False
    assert first["hosted_efficacy_status"] == "UNSUPPORTED"
