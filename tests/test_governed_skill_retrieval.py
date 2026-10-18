from core.acre.engine import AcreEngine
from core.models import RuleSpec, RuleState
from core.skill_retrieval import bm25_candidates, retrieve_and_route


def spec(rule_id, text):
    return RuleSpec(rule_id=rule_id, version=1, parent=None, applicability={"task": "clean"},
                    intervention={"action": "guide"}, expected_mechanism="procedural guide",
                    evidence_requirements=["paired"], scientific_invariants=[],
                    abstain_conditions={}, relations={}, runtime_cost={"tokens": 10},
                    provenance_policy={"required": True}, text=text)


def test_body_ranking_uses_body_without_changing_permission():
    specs = [spec("unrelated", "heat an object"), spec("cleaning", "clean wash sink")]
    assert bm25_candidates(specs, "clean wash", include_body=True)[0] == "cleaning"
    engine = AcreEngine(rule_specs=specs, rule_states={s.rule_id: RuleState(s.rule_id, 1) for s in specs})
    result = retrieve_and_route(engine, "clean wash", {"task": "clean"})
    assert result["body_ranked_ids"][0] == "cleaning"
    assert result["selected_ids"] == []  # Candidate retrieval cannot promote a rule.
    assert result["activation"] == "not_observed"
