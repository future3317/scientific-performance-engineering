from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.formal.condition_adapter import FormalConditionAdapter
from benchmark.formal.run_campaign import (
    InterventionRealizer,
    _read_agent_extensions,
    _read_executor_receipt,
    hydrate_candidate_cases,
    materialize_agent_task,
    persist_collecting_proposals,
    sanitize_submission,
    synthesize_applicability,
)
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
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "rules.json").write_text(json.dumps({
        "schema_version": 1,
        "rules": [{
            "rule_id": endpoint,
            "path": f"rules/{identifier_digest(endpoint)}/v0001.json",
            "status": "canonical",
            "version": 1,
        } for endpoint in ("X", "Y")],
    }), encoding="utf-8")
    validation = {
        "synthesis_case_ids": ["CASE-REL"],
        "promotion_case_ids": ["CASE-REL"],
        "heldout_regression_cases": [{"case_id": "HELDOUT-REL", "executed": True, "execution_source": "verifier", "scientific_ok": True, "effect_lcb": 0.1}],
        "poison_probe_cases": [{"case_id": "POISON-REL", "executed": True, "execution_source": "environment", "accepted": False}],
    }
    validation_path = tmp_path / "evolution" / "validation.json"
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    validation_digest = __import__("hashlib").sha256(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    replay = {
        "evidence_type": "factorial_contrast", "outcome": "passed",
        "result": {"mean_effect": 0.2, "utility_effect_lcb": 0.1, "utility_effect_ucb": 0.3,
                   "promotion_probability_lower_bound": 0.9, "p_min": 0.8},
            "promotion_record": {
                "representative_groups": ["g1", "g2"], "promotion_case_ids": ["CASE-REL"], "heldout_regression_digest": "h",
                "poison_gate": {"passed": True}, "promotion_probability_lcb": 0.9,
                "utility_effect_cs": {"lcb": 0.1, "ucb": 0.3}, "replay_manifest_digest": "m",
                "validation_artifact_path": "evolution/validation.json", "validation_artifact_digest": validation_digest,
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


def test_relation_certificate_must_support_declared_kind() -> None:
    from core.acre.factorial import RelationEvidenceCertificate

    relation = RelationSpec(
        relation_id="REL-KIND", version=1, parent=None,
        endpoints={"left": "X", "right": "Y"}, orientation="symmetric", kind="synergy",
        applicability={"all": []}, contrast_definition={"quantity": "gamma"}, practical_margin=0.05,
        scientific_invariants=[], provenance_policy={"required": True},
    )
    certificate = RelationEvidenceCertificate(
        contrast_cs={"gamma": {"lcb": -0.02, "ucb": 0.02}}, alpha_budget=0.05,
        look_schedule=(8,), scientific_arm_gates={arm: True for arm in ("00", "10", "01", "11")},
        applicability_provenance={"source": "test"}, endpoint_versions={"X": 1, "Y": 1},
    )
    with pytest.raises(ValueError, match="declared relation kind"):
        certificate.validate_for(relation, {"X": {"version": 1}, "Y": {"version": 1}})


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


def test_raw_retrieval_uses_nearest_public_context(tmp_path: Path) -> None:
    from benchmark.harness.experience_retrieval import RawExperienceRetriever

    inbox = tmp_path / "experience" / "inbox"
    inbox.mkdir(parents=True)
    for name, rate, lesson in (("near", 0.2, "near-action"), ("far", 0.9, "far-action")):
        (inbox / f"{name}.json").write_text(json.dumps({
            "record_type": "task_experience",
            "public_context": {"workload": {"dynamic_shape_rate": rate}},
            "lesson": {"proposed_interventions": [lesson]},
        }), encoding="utf-8")
    records = RawExperienceRetriever(tmp_path).retrieve({"workload": {"dynamic_shape_rate": 0.25}})
    assert records[0]["lesson"]["proposed_interventions"] == ["near-action"]


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
    assert "applicability" not in extensions["acre_proposals"][0]
    assert "verdict" not in extensions and "score" not in extensions


def test_replay_counts_one_trial_per_independence_group() -> None:
    from scripts import run_rule_replay

    cases = []
    for group in ("G-1", "G-2"):
        cases.append({
            "case_id": group,
            "paired_replay": True,
            "same_fixture_id": group,
            "independence_group": group,
            "intervention_measurements": [1.2] * 12,
            "baseline_measurements": [1.0] * 12,
            "control_measured": True,
            "scientific_ok": True,
            "quality_ok": True,
        })
    result = run_rule_replay.evaluate_cases(cases, epsilon=0.05, p_min=0.0, delta=0.05)
    assert result["n"] == 2
    assert result["successes"] == 2
    assert result["failures"] == 0


def test_intervention_realizer_materializes_only_the_proposed_patch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    realized = tmp_path / "realized"
    proposal = {
        "rule_id": "RULE-1",
        "intervention": {
            "file": "solution.py",
            "replacements": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
        },
    }
    artifact = InterventionRealizer.realize(baseline, realized, proposal)
    assert artifact == realized
    assert (realized / "solution.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_poison_probe_requires_a_realized_patch(tmp_path: Path) -> None:
    from benchmark.formal.run_campaign import execute_poison_probe

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    realized = tmp_path / "realized"
    InterventionRealizer.realize(
        baseline,
        realized,
        {"intervention": {"file": "solution.py", "replacements": [{"old": "VALUE = 1", "new": "VALUE = 2"}]}},
    )
    result = execute_poison_probe(
        {"task_id": "T", "family_id": "compile"},
        {"workload": {"logical_steps": 128}},
        {"intervention": {"file": "solution.py", "replacements": [{"old": "VALUE = 1", "new": "VALUE = 2"}]}},
        realized,
        baseline,
    )
    assert result["executed"] is True
    assert result["realized_changed"] is True
    assert result["intervention_id"] == "reuse_compile_cache"


def test_applicability_is_synthesized_from_public_evidence() -> None:
    from benchmark.formal.run_campaign import synthesize_applicability

    synthesized = synthesize_applicability([
        {"case_id": "POS", "context": {"workload": {"dynamic_shape_rate": 0.2}}, "intervention_measurements": [0.8] * 256, "baseline_measurements": [0.5] * 256, "higher_is_better": True, "utility_scale": 0.5, "control_measured": True, "scientific_ok": True, "quality_ok": True},
        {"case_id": "NEG", "context": {"workload": {"dynamic_shape_rate": 0.8}}, "intervention_measurements": [0.4] * 256, "baseline_measurements": [0.5] * 256, "higher_is_better": True, "utility_scale": 0.5, "control_measured": True, "scientific_ok": True, "quality_ok": True},
    ])
    assert synthesized is not None
    predicate, provenance = synthesized
    assert "compare" in predicate
    assert provenance["source"] == "harness-cegis"


def test_applicability_waits_for_certified_repeated_effects() -> None:
    cases = [
        {
            "case_id": "POS",
            "context": {"workload": {"dynamic_shape_rate": 0.2}},
            "intervention_measurements": [0.8, 0.8],
            "baseline_measurements": [0.5, 0.5],
            "scientific_ok": True,
            "quality_ok": True,
            "control_measured": True,
        },
        {
            "case_id": "NEG",
            "context": {"workload": {"dynamic_shape_rate": 0.8}},
            "intervention_measurements": [0.51, 0.51],
            "baseline_measurements": [0.5, 0.5],
            "scientific_ok": True,
            "quality_ok": True,
            "control_measured": True,
        },
    ]
    assert synthesize_applicability(cases, family_id="compile") is None


def test_applicability_uses_preregistered_family_lattice() -> None:
    cases = [
        {
            "case_id": "POS",
            "context": {"workload": {"logical_steps": 256, "dynamic_shape_rate": 0.2}},
            "intervention_measurements": [0.8] * 256,
            "baseline_measurements": [0.5] * 256,
            "scientific_ok": True,
            "quality_ok": True,
            "control_measured": True,
        },
        {
            "case_id": "NEG",
            "context": {"workload": {"logical_steps": 64, "dynamic_shape_rate": 0.8}},
            "intervention_measurements": [0.4] * 256,
            "baseline_measurements": [0.5] * 256,
            "scientific_ok": True,
            "quality_ok": True,
            "control_measured": True,
        },
    ]
    synthesized = synthesize_applicability(cases, family_id="compile")
    assert synthesized is not None
    assert synthesized[1]["decision_context_count"] > len(cases)


def test_formal_promotion_round_trip_routes_and_abstains_by_cegis_boundary(tmp_path: Path) -> None:
    from core.governance import apply_promotion
    from core.acre.engine import AcreEngine
    from core.models import TaskContext

    synthesized = synthesize_applicability([
        {"case_id": "POS", "context": {"domain": "runtime", "workload": {"dynamic_shape_rate": 0.2}}, "intervention_measurements": [0.8] * 256, "baseline_measurements": [0.5] * 256, "higher_is_better": True, "utility_scale": 0.5, "control_measured": True, "scientific_ok": True, "quality_ok": True},
        {"case_id": "NEG", "context": {"domain": "runtime", "workload": {"dynamic_shape_rate": 0.8}}, "intervention_measurements": [0.4] * 256, "baseline_measurements": [0.5] * 256, "higher_is_better": True, "utility_scale": 0.5, "control_measured": True, "scientific_ok": True, "quality_ok": True},
    ])
    assert synthesized is not None
    predicate, provenance = synthesized
    candidate = _rule("RULE-CEGIS")
    candidate = candidate.__class__(**{**candidate.__dict__, "applicability": predicate}).to_dict()
    validation = {
        "synthesis_case_ids": ["CASE-CEGIS-1", "CASE-CEGIS-2"],
        "promotion_case_ids": ["CASE-CEGIS-1", "CASE-CEGIS-2"],
        "heldout_regression_cases": [{"case_id": "HELDOUT-CEGIS", "executed": True, "execution_source": "verifier", "scientific_ok": True, "effect_lcb": 0.1}],
        "poison_probe_cases": [{"case_id": "POISON-CEGIS", "executed": True, "execution_source": "environment", "accepted": False}],
    }
    validation_path = tmp_path / "evolution" / "validation.json"
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    import hashlib
    validation_digest = hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    replay = {
        "outcome": "passed",
        "result": {"mean_effect": 0.3, "utility_effect_lcb": 0.2, "utility_effect_ucb": 0.4, "promotion_probability_lower_bound": 0.9, "p_min": 0.8},
        "promotion_record": {
            "representative_groups": ["G1", "G2"], "promotion_case_ids": validation["promotion_case_ids"],
            "heldout_regression_digest": "heldout", "poison_gate": {"passed": True},
            "promotion_probability_lcb": 0.9, "utility_effect_cs": {"lcb": 0.2, "ucb": 0.4},
            "replay_manifest_digest": "replay", "validation_artifact_path": "evolution/validation.json", "validation_artifact_digest": validation_digest,
        },
    }
    assert apply_promotion(tmp_path, candidate, replay, replay_path="evolution/replay.json").allowed
    engine = AcreEngine.from_store(tmp_path)
    positive = engine.route(TaskContext("runtime", {"dynamic_shape_rate": 0.2}, {}, {}, {}, 4096))
    negative = engine.route(TaskContext("runtime", {"dynamic_shape_rate": 0.8}, {}, {}, {}, 4096))
    assert positive.selected_rule_ids == ("RULE-CEGIS",)
    assert negative.selected_rule_ids == ()


def test_relation_promotion_requires_harness_endpoint_states(tmp_path: Path) -> None:
    from core.governance import evaluate_candidate

    candidate = RelationSpec(
        relation_id="REL-X-Y", version=1, parent=None,
        endpoints={"left": "X", "right": "Y"}, orientation="symmetric", kind="synergy",
        applicability={"all": []}, contrast_definition={"gamma": "cs"}, practical_margin=0.05,
        scientific_invariants=[], provenance_policy={"required": True},
    ).to_dict()
    replay = {
        "outcome": "passed",
        "evidence_type": "factorial_contrast",
        "result": {"mean_effect": 0.3, "utility_effect_lcb": 0.2, "utility_effect_ucb": 0.4, "promotion_probability_lower_bound": 0.9, "p_min": 0.8},
        "promotion_record": {
            "representative_groups": ["G1", "G2"], "promotion_case_ids": ["CASE-1"],
            "heldout_regression_digest": "heldout", "poison_gate": {"passed": True},
            "promotion_probability_lcb": 0.9, "utility_effect_cs": {"lcb": 0.2, "ucb": 0.4},
            "replay_manifest_digest": "replay",
            "validation_artifact_path": "evolution/validation.json", "validation_artifact_digest": "digest",
        },
        "relation_evidence_certificate": {
            "contrast_cs": {"gamma": {"lcb": 0.1, "ucb": 0.2}}, "alpha_budget": 0.05,
            "look_schedule": [8], "scientific_arm_gates": {"00": True, "10": True, "01": True, "11": True},
            "applicability_provenance": {"source": "test"}, "endpoint_versions": {"X": 1, "Y": 1},
        },
    }
    decision = evaluate_candidate(candidate, replay)
    assert not decision.allowed and "endpoint states" in decision.reason


def test_submission_sanitizer_rejects_symlink_and_unallowlisted_files(tmp_path: Path) -> None:
    source = tmp_path / "worker"
    source.mkdir()
    (source / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "extra.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlisted"):
        sanitize_submission(source, tmp_path / "submitted", {"solution.py"})
    (source / "extra.py").unlink()
    try:
        (source / "solution.py.link").symlink_to(source / "solution.py")
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        sanitize_submission(source, tmp_path / "submitted", {"solution.py", "solution.py.link"})


def test_submission_sanitizer_rejects_changed_helper_outside_public_allowlist(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    source = tmp_path / "worker"
    baseline.mkdir()
    source.mkdir()
    (baseline / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    (baseline / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "solution.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public change surface"):
        sanitize_submission(source, tmp_path / "submitted", {"solution.py"}, baseline=baseline)


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


def test_validation_artifact_requires_executed_disjoint_probes() -> None:
    from core.governance import validate_validation_artifact

    artifact = {
        "synthesis_case_ids": ["CASE-1"],
        "promotion_case_ids": ["CASE-1"],
        "heldout_regression_cases": [{"case_id": "CASE-1", "executed": True}],
        "poison_probe_cases": [{"case_id": "POISON-1", "executed": False, "accepted": False}],
    }
    errors = validate_validation_artifact(artifact, {"CASE-1"})
    assert any("disjoint" in error for error in errors)
    assert any("executed" in error for error in errors)


def test_validation_artifact_requires_heldout_gate() -> None:
    from core.governance import validate_validation_artifact

    artifact = {
        "synthesis_case_ids": ["CASE-1"],
        "promotion_case_ids": ["CASE-1"],
        "heldout_regression_cases": [{
            "case_id": "HELDOUT-1", "executed": True, "execution_source": "verifier",
            "scientific_ok": False, "effect_lcb": -0.1,
        }],
        "poison_probe_cases": [{
            "case_id": "POISON-1", "executed": True, "execution_source": "environment", "accepted": False,
        }],
    }
    errors = validate_validation_artifact(artifact, {"CASE-1"})
    assert any("held-out" in error and "scientific" in error for error in errors)


def test_candidate_evidence_hydrates_all_ledger_members(tmp_path: Path) -> None:
    from benchmark.harness.evolution_ledger import CandidateEvidenceLedger

    cases_dir = tmp_path / "experience" / "cases"
    cases_dir.mkdir(parents=True)
    ledger = CandidateEvidenceLedger(tmp_path / "evidence.jsonl")
    first = {"case_id": "CASE-A", "context": {"workload": {"x": 1}}, "utility_on": 1.0, "utility_off": 0.5}
    second = {"case_id": "CASE-B", "context": {"workload": {"x": 2}}, "utility_on": 1.0, "utility_off": 0.5}
    for case in (first, second):
        path = cases_dir / f"{case['case_id']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        ledger.append("RULE-A", 1, {**case, "case_path": str(path.relative_to(tmp_path)).replace("\\", "/")})
    hydrated = hydrate_candidate_cases(tmp_path, {"rule_id": "RULE-A", "version": 1, "cases": ["CASE-A"]}, ledger)
    assert [case["case_id"] for case in hydrated] == ["CASE-A", "CASE-B"]


def test_proposal_is_persisted_while_collecting_evidence(tmp_path: Path) -> None:
    persist_collecting_proposals(tmp_path, [{
        "rule_id": "RULE-COLLECTING",
        "intervention": {"file": "solution.py", "replacements": [{"old": "x", "new": "y"}]},
    }], [])
    card = json.loads(next((tmp_path / "evolution" / "candidates").glob("*.json")).read_text(encoding="utf-8"))
    assert card["status"] == "collecting_evidence"
    assert card["cases"] == []


def test_utility_transform_is_dimensionless_and_versioned() -> None:
    from core.utility import UTILITY_POLICY_ID, utility_effect

    assert UTILITY_POLICY_ID == "bounded_log_speedup_v1"
    assert utility_effect(95.0, 100.0, higher_is_better=False) > 0.0
    assert utility_effect(2.0, 1.0, higher_is_better=True) > utility_effect(1.1, 1.0, higher_is_better=True)


def test_replay_events_use_group_effects_not_task_scores() -> None:
    from scripts.run_rule_replay import build_evidence_events, evaluate_cases

    case = {
        "case_id": "CASE-MEASURED",
        "independence_group": "GROUP-1",
        "paired_replay": True,
        "same_fixture_id": "FIXTURE-1",
        "utility_on": 0.0,
        "utility_off": 0.0,
        "intervention_measurements": [95.0, 96.0],
        "baseline_measurements": [100.0, 100.0],
        "control_measured": True,
        "higher_is_better": False,
        "scientific_ok": True,
        "quality_ok": True,
    }
    result = evaluate_cases([case], epsilon=0.0, p_min=0.0, delta=0.05)
    assert result["successes"] == 1
    events = build_evidence_events({"rule_id": "RULE-MEASURED", "cases": [case]})
    assert len(events) == 2
    assert events[0]["outcome_vector"]["paired_effect"] > 0.0


def test_formal_identification_requires_a_nonempty_decision_lattice() -> None:
    assert synthesize_applicability(
        [{
            "case_id": "POS",
            "context": {"workload": {"dynamic_shape_rate": 0.2}},
            "intervention_measurements": [0.8] * 16,
            "baseline_measurements": [0.5] * 16,
            "control_measured": True,
            "scientific_ok": True,
        }],
        family_id="missing-family",
        require_identified=True,
    ) is None


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
