from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.formal.run_campaign import _read_agent_extensions, _read_executor_receipt, materialize_agent_task
from benchmark.harness import conditions
from core.acre.engine import AcreEngine
from core.models import RelationSpec, RelationState, RuleSpec, RuleState
from core.models import validate_identifier


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


def test_relation_promotion_round_trips_as_spec_state_and_record(tmp_path: Path) -> None:
    from core.governance import apply_promotion

    candidate = RelationSpec(
        relation_id="REL-X-Y", version=1, parent=None,
        endpoints={"left": "X", "right": "Y"}, orientation="symmetric", kind="synergy",
        applicability={"all": []}, contrast_definition={"gamma": "cs"}, practical_margin=0.05,
        scientific_invariants=[], provenance_policy={"required": True},
    ).to_dict()
    from core.models import identifier_digest
    for endpoint in ("X", "Y"):
        rules = tmp_path / "rules" / identifier_digest(endpoint)
        rules.mkdir(parents=True)
        (rules / "v0001.json").write_text(json.dumps(_rule(endpoint).to_dict()), encoding="utf-8")
        (rules / "v0001.state.json").write_text(json.dumps(RuleState(endpoint, 1, "canonical").to_dict()), encoding="utf-8")
    replay = {
        "evidence_type": "factorial_contrast", "outcome": "passed",
        "result": {"mean_effect": 0.2, "utility_effect_lcb": 0.1, "utility_effect_ucb": 0.3,
                   "promotion_probability_lower_bound": 0.9, "p_min": 0.8},
        "promotion_record": {
            "representative_groups": ["g1", "g2"], "heldout_regression_digest": "h",
            "poison_gate": {"passed": True}, "promotion_probability_lcb": 0.9,
            "utility_effect_cs": {"lcb": 0.1, "ucb": 0.3}, "replay_manifest_digest": "m",
        },
        "relation_evidence_certificate": {
            "contrast_cs": {"gamma": {"lcb": 0.1, "ucb": 0.3}}, "alpha_budget": 0.05,
            "look_schedule": [8, 16], "scientific_arm_gates": {"00": True, "01": True, "10": True, "11": True},
            "applicability_provenance": {"source": "test"}, "endpoint_versions": {"X": 1, "Y": 1},
        },
    }
    decision = apply_promotion(tmp_path, candidate, replay, replay_path="evolution/contrast.json")
    assert decision.allowed
    engine = AcreEngine.from_store(tmp_path)
    assert engine.relation_states["REL-X-Y"].status == "canonical"
    assert (tmp_path / "evolution" / "promotions").is_dir()


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
        "causal_evidence_events": [{"event_id": "forged"}],
        "acre_candidates": [{"rule_id": "forged", "cases": [{"utility_on": 1.0}]}],
        "acre_proposals": [{"rule_id": "R1", "applicability": {"all": []}}],
    }), encoding="utf-8")
    extensions = _read_agent_extensions(result)
    assert set(extensions) == {"lesson", "acre_proposals"}
    assert extensions["acre_proposals"][0]["rule_id"] == "R1"
    assert "verdict" not in extensions and "score" not in extensions


def test_external_ids_reject_path_traversal() -> None:
    with pytest.raises(ValueError):
        validate_identifier("../../rules/escape", "rule_id")
    with pytest.raises(ValueError):
        RuleState("../../rules/escape", 1)
    with pytest.raises(ValueError):
        RelationState("..\\relations\\escape", 1, 0.0)


def test_public_routing_context_is_explicit_and_hidden_labels_are_not_projected(tmp_path: Path) -> None:
    task = tmp_path / "task"
    (task / "workspace").mkdir(parents=True)
    (task / "public_tests").mkdir()
    (task / "task.yaml").write_text(
        "task_id: T\ntitle: public\nmechanism: hidden\nfamily_id: hidden-family\npublic_context:\n  workload:\n    device: cpu\nworkspace:\n  entrypoint: solution.py\n",
        encoding="utf-8",
    )
    (task / "workspace" / "solution.py").write_text("x = 1\n", encoding="utf-8")
    public = tmp_path / "public"
    materialize_agent_task(task, public)
    payload = json.loads((public / "public_task.json").read_text(encoding="utf-8"))
    assert payload["routing_context"] == {"workload": {"device": "cpu"}}
    assert payload["routing_context"].get("mechanism") is None


def test_b_c_d_share_the_same_rendered_skill_snapshot(tmp_path: Path) -> None:
    from scripts.render_skill_view import render_skill_view

    source = tmp_path / "skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    render_skill_view(source, bundle)
    manifests = {}
    for condition in ("B", "C", "D"):
        store = tmp_path / condition
        manifests[condition] = conditions.materialize_condition(condition, bundle, store)
        assert (store / "SKILL.md").read_text(encoding="utf-8") == (bundle / "SKILL.md").read_text(encoding="utf-8")
    assert manifests["B"]["files"]["SKILL.md"] == manifests["C"]["files"]["SKILL.md"] == manifests["D"]["files"]["SKILL.md"]


def test_executor_receipt_requires_network_and_mount_attestation(tmp_path: Path) -> None:
    receipt = tmp_path / "executor_receipt.json"
    receipt.write_text(json.dumps({"mode": "external_namespace_executor", "network_mode": "bridge"}), encoding="utf-8")
    _, errors = _read_executor_receipt(receipt, "a" * 64)
    assert "external executor must declare network_mode=none" in errors
    assert any("mount_allowlist" in error for error in errors)


def test_condition_a_receipt_rejects_skill_mount(tmp_path: Path) -> None:
    receipt = tmp_path / "executor_receipt.json"
    receipt.write_text(json.dumps({
        "mode": "external_namespace_executor", "network_mode": "none",
        "mount_allowlist": ["task", "solution", "skill_view", "retrieved_context", "result", "executor_receipt"],
        "executor_digest": "x", "worker_uid": "u", "usage": {}, "skill_view_digest": "x",
    }), encoding="utf-8")
    _, errors = _read_executor_receipt(receipt, None)
    assert "condition A must not mount skill_view" in errors


def test_replay_rejects_zero_control_arm() -> None:
    from scripts import run_rule_replay
    with pytest.raises(ValueError, match="non-zero control"):
        run_rule_replay.evaluate_cases([{
            "case_id": "C-1", "paired_replay": True, "same_fixture_id": "F-1",
            "utility_on": 0.5, "utility_off": 0.0, "scientific_ok": True,
        }], 0.05, 0.8, 0.05)


def test_missing_promotion_record_gate_is_rejected() -> None:
    from core.governance import evaluate_candidate
    decision = evaluate_candidate(_rule("R-GATE").to_dict(), {"outcome": "passed", "result": {"p_min": 0.8}})
    assert not decision.allowed and "promotion record" in decision.reason


def test_evidence_utility_is_bounded_for_confidence_accounting() -> None:
    from core.models import EvidenceEvent

    payload = {
        "schema_version": 2,
        "event_id": "E-1",
        "context": {"domain": "runtime"},
        "assignment": {"interventions": {"R1": 1}, "propensity": 0.5, "design_id": "D-1"},
        "outcome_vector": {"utility": 1.1},
        "scientific_gates": {"ok": True},
        "artifacts": {}, "versions": {}, "source_id": "S-1", "independence_group": "G-1",
        "timestamp": "2026-01-01T00:00:00Z", "evidence_stream": "representative", "query_id": "Q-1",
        "trust_zone": "local", "attacker_controlled_fields": [],
    }
    with pytest.raises(ValueError, match="bounded"):
        EvidenceEvent.from_dict(payload)
