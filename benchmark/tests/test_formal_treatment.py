from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.formal.run_campaign import _read_agent_extensions, materialize_agent_task
from benchmark.harness import conditions
from core.acre.engine import AcreEngine
from core.models import RelationSpec, RelationState, RuleSpec, RuleState


def _rule(rule_id: str = "R1") -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        version=1,
        parent=None,
        applicability={"all": []},
        intervention={"action": "reuse_cache"},
        expected_mechanism="compile",
        evidence_requirements=["paired"],
        scientific_invariants=[],
        abstain_conditions={},
        relations={},
        runtime_cost={"tokens": 1},
        provenance_policy={"required": True},
    )


def test_engine_from_store_requires_matching_materialized_state(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "R1.json").write_text(json.dumps(_rule().to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="missing materialized state"):
        AcreEngine.from_store(tmp_path)
    (rules / "R1.state.json").write_text(json.dumps(RuleState("R1", 1, "canonical").to_dict()), encoding="utf-8")
    engine = AcreEngine.from_store(tmp_path)
    assert [spec.rule_id for spec in engine.rule_specs] == ["R1"]


def test_engine_from_store_reads_relation_state_pipeline_dir(tmp_path: Path) -> None:
    relations = tmp_path / "relations"
    relations.mkdir()
    relation = RelationSpec(
        relation_id="REL-A-B",
        version=1,
        parent=None,
        endpoints={"left": "A", "right": "B"},
        orientation="symmetric",
        kind="synergy",
        applicability={"all": []},
        contrast_definition={"quantity": "gamma"},
        practical_margin=0.05,
        scientific_invariants=[],
        provenance_policy={"required": True},
    )
    (relations / "REL-A-B.json").write_text(json.dumps(relation.to_dict()), encoding="utf-8")
    relation_states = tmp_path / "relation_states"
    relation_states.mkdir()
    (relation_states / "REL-A-B.json").write_text(
        json.dumps(RelationState("REL-A-B", 1, 0.2, status="canonical").to_dict()), encoding="utf-8"
    )
    engine = AcreEngine.from_store(tmp_path)
    assert [spec.relation_id for spec in engine.relation_specs] == ["REL-A-B"]


def test_condition_adapter_reads_canonical_store_without_fake_state(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    from scripts.render_skill_view import render_skill_view

    render_skill_view(bundle, tmp_path / "rendered")
    store = tmp_path / "store"
    conditions.materialize_condition("D", tmp_path / "rendered", store)
    context = FormalConditionAdapter("D", store).retrieved_context({"workload": {"mechanism": "compile"}})
    assert context["condition"] == "D"
    assert context["proposed_interventions"] == []


def test_task_experience_is_not_causal_intervention(tmp_path: Path) -> None:
    from benchmark.harness.experience_retrieval import RawExperienceRetriever

    inbox = tmp_path / "experience" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "task.json").write_text(json.dumps({
        "record_type": "task_experience",
        "lesson": {"proposed_interventions": ["not-a-causal-action"]},
    }), encoding="utf-8")
    assert RawExperienceRetriever(tmp_path).propose_interventions() == ["not-a-causal-action"]
    assert "assignment" not in json.loads((inbox / "task.json").read_text(encoding="utf-8"))


def test_public_task_projection_does_not_copy_hidden_metadata(tmp_path: Path) -> None:
    task = tmp_path / "task"
    (task / "workspace").mkdir(parents=True)
    (task / "public_tests").mkdir()
    (task / "task.yaml").write_text(
        "task_id: T\ntitle: public\nrequires_cuda: false\ntime_budget_s: 1\nworkspace:\n  entrypoint: solution.py\n  api: train_loop_v1\n",
        encoding="utf-8",
    )
    (task / "workspace" / "solution.py").write_text("x = 1\n", encoding="utf-8")
    public = tmp_path / "public"
    materialize_agent_task(task, public)
    assert (public / "public_task.json").is_file()
    assert not (public / "task.yaml").exists()


def test_worker_extensions_cannot_override_scored_result(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "verdict": "pass",
        "score": 999,
        "lesson": {"type": "boundary"},
        "causal_evidence_events": [],
        "acre_candidates": [],
    }), encoding="utf-8")
    extensions = _read_agent_extensions(result)
    assert set(extensions) == {"lesson", "causal_evidence_events", "acre_candidates"}
    assert "verdict" not in extensions and "score" not in extensions
