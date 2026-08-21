from __future__ import annotations

import inspect
import json

from core.acre.acquisition import (
    AcquisitionPolicy,
    AcquisitionQuery,
    evaluate_trajectory,
    run_acquisition,
)
from core.acre.predicates import PredicateGrammar, predicate_complexity
from core.acre.router import ConservativeCausalRouter
from core.acre.engine import AcreEngine
from core.acre.evidence import assess, representative_events
from core.governance import EvolutionDecision
from core.models import RelationSpec, RelationState, RuleSpec, RuleState, TaskContext
from scripts.run_rule_replay import build_evidence_events


def test_acquisition_stopping_uses_posterior_not_hidden_truth() -> None:
    queries = [
        AcquisitionQuery("a", "edge-a", 1.0),
        AcquisitionQuery("b", "edge-b", 1.0),
    ]
    labels = {"a": True, "b": False}
    trajectory = run_acquisition(queries, labels, AcquisitionPolicy.DECISION_AWARE, confidence_target=0.9)
    assert "edge_truths" not in inspect.signature(run_acquisition).parameters
    assert set(trajectory.selected_query_ids) == {"a", "b"}
    offline = evaluate_trajectory(trajectory, queries, labels, {"edge-a": True, "edge-b": False}, target_error=0.0)
    assert offline.cost_to_target == 2.0


def test_predicate_complexity_gates_parent_and_boolean_specializations() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "x", "type": "numeric"}, {"path": "kind", "type": "categorical"}],
        "max_depth": 3,
        "max_literals": 3,
    })
    parent = {"equals": {"kind": "graph"}}
    candidates = grammar.candidates([{"x": 1, "kind": "graph"}, {"x": 3, "kind": "graph"}], parent)
    assert any("all" in value for value in candidates)
    # Parent-safe boolean forms are nested under ``all`` after equivalent
    # candidate pruning; inspect the predicate tree rather than only its root.
    assert any("not" in json.dumps(value) or "any" in json.dumps(value) for value in candidates)
    assert all(predicate_complexity(value)["depth"] <= 3 for value in candidates)
    assert all(predicate_complexity(value)["literals"] <= 3 for value in candidates)


def test_router_consumes_canonical_rule_and_relation_models() -> None:
    def rule(rule_id: str) -> RuleSpec:
        return RuleSpec(rule_id, 1, None, {"equals": {"workload": "graph"}}, {"action": rule_id}, "mechanism", ["evidence"], [], {}, {}, {"tokens": 1.0}, {"required": True})

    specs = [rule("a"), rule("b")]
    states = {rule_id: RuleState(rule_id, 1, status="canonical", effect={"lower_utility": 0.5}) for rule_id in ("a", "b")}
    relation = RelationSpec("a-b", 1, None, {"left": "a", "right": "b"}, "symmetric", "synergy", {"equals": {"workload": "graph"}}, {"contrast": "gamma"}, 0.05, [], {"required": True})
    relation_state = RelationState("a-b", 1, 0.2, {"lcb": 0.2}, status="canonical")
    context = TaskContext("runtime", {"workload": "graph"}, {}, {}, {}, token_budget=4)
    decision = ConservativeCausalRouter(token_budget=4).route(specs, states, [relation], {"a-b": relation_state}, context)
    assert decision.selected_rule_ids == ("a", "b")


def test_replay_writer_emits_canonical_evidence_v2() -> None:
    events = build_evidence_events({
        "rule_id": "RULE-A",
        "rule_version": 2,
        "cases": [{"case_id": "case-1", "utility_on": 1.0, "utility_off": 0.5, "scientific_ok": True}],
    })
    assert events
    for event in events:
        assert event["schema_version"] == 2
        assert "rule_id" not in event
        assert "arm" not in event["assignment"]
        assert event["assignment"]["interventions"] == {"RULE-A": int(event["event_id"].endswith("-on"))}
        assert event["context"]["rule_versions"] == {"RULE-A": 2}


def test_acre_engine_is_the_single_public_orchestration_facade() -> None:
    spec = RuleSpec("RULE-A", 1, None, {"equals": {}}, {"action": "a"}, "mechanism", ["evidence"], [], {}, {}, {"tokens": 1.0}, {"required": True})
    state = RuleState("RULE-A", 1, status="canonical", effect={"lower_utility": 0.5})
    engine = AcreEngine(rule_specs=[spec], rule_states={"RULE-A": state}, query_proposer=lambda context, events: "experiment")
    event = {
        "schema_version": 2, "event_id": "e1", "context": {"rule_versions": {"RULE-A": 1}},
        "assignment": {"interventions": {"RULE-A": 1}, "propensity": 0.5, "design_id": "test"},
        "outcome_vector": {"utility": 0.5}, "scientific_gates": {"scientific_ok": True}, "artifacts": {}, "versions": {},
        "source_id": "test", "independence_group": "g1", "timestamp": "2026-01-01T00:00:00Z", "evidence_stream": "representative", "query_id": "q1", "trust_zone": "local", "attacker_controlled_fields": [],
    }
    engine.observe(event)
    decision = engine.evolve("RULE-A")
    assert isinstance(decision, EvolutionDecision)
    assert decision.subject_type == "rule"
    assert decision.operation == "NO_OP"
    assert engine.propose_experiment({}) == "experiment"
    assert engine.route({}) .selected_rule_ids == ("RULE-A",)


def test_adversarial_evidence_cannot_change_promotion_lcb() -> None:
    def event(event_id: str, stream: str, *, utility: float, scientific_ok: bool = True) -> dict[str, object]:
        return {
            "schema_version": 2,
            "event_id": event_id,
            "context": {"rule_versions": {"RULE-A": 1}},
            "assignment": {"interventions": {"RULE-A": 1}, "propensity": 0.5, "design_id": "test"},
            "outcome_vector": {"utility": utility, "counterexample": not scientific_ok},
            "scientific_gates": {"scientific_ok": scientific_ok},
            "artifacts": {}, "versions": {}, "source_id": "test", "independence_group": event_id,
            "timestamp": "2026-01-01T00:00:00Z", "evidence_stream": stream, "query_id": event_id,
            "trust_zone": "local", "attacker_controlled_fields": [],
        }

    representative = [event("rep-1", "representative", utility=0.8)]
    baseline = assess(representative)
    adversarial = [event(f"adv-{index}", "adversarial", utility=-1.0, scientific_ok=False) for index in range(10_000)]
    combined = assess(representative + adversarial)
    assert combined.promotion_lcb == baseline.promotion_lcb
    assert combined.representative_count == baseline.representative_count == 1
    assert len(combined.specialization_event_ids) == 10_000
    assert len(representative_events(representative + adversarial)) == 1
