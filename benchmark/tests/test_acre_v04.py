from __future__ import annotations

from core.acre.factorial import FactorialBlock, FactorialEngine, ThreeWayBlock, estimate_higher_order, canonical_relation_label
from core.acre.predicates import PredicateGrammar
from core.acre.relation import RelationIdentifier, relational_cegis
from core.acre.router import ConservativeCausalRouter
from core.models import RelationSpec, RelationState, RuleSpec, RuleState, TaskContext
from benchmark.families.environment import EpisodeEnvironmentState, FamilyEnvironment
from benchmark.interaction.factorial_bench import build_three_way_oracle, run_interaction_power_curve


def _rule(rule_id: str, action: str) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id, version=1, parent=None,
        applicability={"all": []}, intervention={"action": action},
        expected_mechanism="test", evidence_requirements=["paired"],
        scientific_invariants=[], abstain_conditions={}, relations={},
        runtime_cost={"tokens": 1}, provenance_policy={"required": True},
    )


def _state(rule_id: str) -> RuleState:
    return RuleState(rule_id=rule_id, version=1, status="canonical", effect={"utility": 0.2}, confidence_sequence={"lcb": 0.2})


def _factorial(outcomes: dict[str, float], blocks: int = 128) -> FactorialEngine:
    engine = FactorialEngine(delta=0.05, practical_margin=0.1, look_count=1)
    for i in range(blocks):
        engine.add_block(FactorialBlock(str(i), outcomes))
    return engine


def test_relational_cegis_returns_typed_child_predicates_and_router_selects_one():
    low = _factorial({"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}, blocks=4096).estimate()
    high = _factorial({"00": 0.0, "10": 0.8, "01": 0.8, "11": 0.2}, blocks=4096).estimate()
    identifier = RelationIdentifier(practical_margin=0.1)
    identified = identifier.identify({"low": low, "high": high})
    grammar = PredicateGrammar.from_dict({"schema_version": 1, "features": [{"path": "dynamic_shape_rate", "type": "numeric"}]})
    children = relational_cegis(
        identifier, {"low": {"dynamic_shape_rate": 0.2}, "high": {"dynamic_shape_rate": 0.8}}, identified, grammar,
    )
    assert len(children) == 2
    assert all("compare" in child.applicability for child in children)
    specs = [_rule("a", "a"), _rule("b", "b")]
    states = {item.rule_id: _state(item.rule_id) for item in specs}
    relation_specs = list(children)
    relation_states = {item.relation_id: RelationState(item.relation_id, 1, 0.2, {"lcb": 0.2}, "canonical") for item in relation_specs}
    router = ConservativeCausalRouter(token_budget=4)
    low_decision = router.route(specs, states, relation_specs, relation_states, TaskContext("x", {"dynamic_shape_rate": 0.2}, {}, {}, {}))
    high_decision = router.route(specs, states, relation_specs, relation_states, TaskContext("x", {"dynamic_shape_rate": 0.8}, {}, {}, {}))
    assert low_decision.bundle_certificate is not None
    assert "relation_overlap_conflict" not in sum((list(v) for v in low_decision.rejected_reasons.values()), [])
    assert low_decision.selected_rule_ids == high_decision.selected_rule_ids


def test_router_rejects_overlapping_semantic_relations():
    specs = [_rule("a", "a"), _rule("b", "b")]
    states = {item.rule_id: _state(item.rule_id) for item in specs}
    relation_specs = [
        RelationSpec("r1", 1, None, {"left": "a", "right": "b"}, "symmetric", "synergy", {"all": []}, {"source": "test"}, 0.1, [], {"required": True}),
        RelationSpec("r2", 1, None, {"left": "a", "right": "b"}, "symmetric", "antagonism", {"all": []}, {"source": "test"}, 0.1, [], {"required": True}),
    ]
    relation_states = {item.relation_id: RelationState(item.relation_id, 1, 0.2, {"lcb": 0.2}, "canonical") for item in relation_specs}
    decision = ConservativeCausalRouter(token_budget=4).route(specs, states, relation_specs, relation_states, {"x": 1})
    assert any("relation_overlap_conflict" in reasons for reasons in decision.rejected_reasons.values())


def test_factorial_estimate_exposes_independent_contrast_intervals():
    estimate = _factorial({"00": 0.0, "10": 0.2, "01": 0.1, "11": 0.8}).estimate()
    assert set(estimate.contrast_intervals) >= {"gamma", "delta_a_given_b0", "delta_a_given_b1", "delta_b_given_a0", "delta_b_given_a1", "redundancy"}


def test_higher_order_estimate_reports_normalized_residual():
    blocks = [ThreeWayBlock(str(i), {arm: 0.0 for arm in ("000", "001", "010", "011", "100", "101", "110", "111")}, scientific_gates={arm: True for arm in ("000", "001", "010", "011", "100", "101", "110", "111")}) for i in range(8)]
    estimate = estimate_higher_order(blocks)
    assert estimate.raw_residual == 0.0
    assert estimate.normalized_residual == 0.0
    assert estimate.status == "unresolved"


def test_relation_report_labels_share_one_canonical_namespace():
    assert canonical_relation_label("synergy") == "confirmed_synergy"
    assert canonical_relation_label("context_dependent_interaction") == "context_dependent_relation"


def test_relation_identifier_does_not_depend_on_context_insertion_order():
    unresolved = _factorial({"00": 0.0, "10": 0.0, "01": 0.0, "11": 0.0}, blocks=1).estimate()
    strong = _factorial({"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}, blocks=4096).estimate()
    identifier = RelationIdentifier(practical_margin=0.1)
    first = identifier.identify({"unresolved": unresolved, "strong": strong})
    second = identifier.identify({"strong": strong, "unresolved": unresolved})
    assert first.decision == second.decision == "underidentified_context_relation"


def test_three_way_oracle_hits_requested_residual_exactly():
    cells = build_three_way_oracle(residual=0.24)
    raw = cells["111"] - cells["110"] - cells["101"] - cells["011"] + cells["100"] + cells["010"] + cells["001"] - cells["000"]
    assert abs(raw - 0.24) < 1e-12


def test_environment_transformations_persist_until_revalidated():
    state = EpisodeEnvironmentState()
    environment = FamilyEnvironment("compile")
    state = state.apply({"kind": "software", "parameters": {"to_runtime": "B"}})
    drifted = environment.evaluate({}, (), state)
    assert drifted.oracle_bundle == ()
    recovered = state.apply({"kind": "revalidation", "parameters": {}})
    stable = environment.evaluate({}, (), recovered)
    assert stable.oracle_bundle == ()


def test_interaction_power_curve_crosses_effect_strength_and_noise():
    report = run_interaction_power_curve(blocks=(8, 16), repetitions=3)
    assert len(report["results"]) == 12
    assert {row["effect_strength"] for row in report["results"]} == {"near-null", "near-margin", "moderate", "strong"}
