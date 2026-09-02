from __future__ import annotations

from dataclasses import dataclass


def test_execution_reserve_rejects_evolution_without_consuming_future_task_reserve():
    from core.acre.resource_budget import ExecutionReserveBudget

    ledger = ExecutionReserveBudget(total_budget=10.0, scheduled_minimums=(3.0, 3.0, 3.0))
    assert ledger.execution_reserve == 9.0
    assert ledger.evolution_available == 1.0

    next_ledger, decision = ledger.admit_evolution(2.0, query_id="q-expensive")
    assert decision.accepted is False
    assert decision.reason == "execution_reserve"
    assert next_ledger.evolution_spent == 0.0
    assert next_ledger.execution_reserve == 9.0


def test_execution_reserve_accounting_is_conserved_and_replayable():
    from core.acre.resource_budget import ExecutionReserveBudget

    initial = ExecutionReserveBudget(total_budget=20.0, scheduled_minimums=(4.0, 4.0))
    after_query, accepted = initial.admit_evolution(3.0, query_id="q1")
    after_task = after_query.advance_task(4.0)
    replay_query, replay_accepted = initial.admit_evolution(3.0, query_id="q1")
    replay_task = replay_query.advance_task(4.0)

    assert accepted.accepted and replay_accepted.accepted
    assert after_task == replay_task
    assert after_task.evolution_spent + after_task.execution_reserve + after_task.evolution_available == 20.0


def test_budget_exhaustion_freezes_evolution_but_not_scheduled_execution():
    from core.acre.resource_budget import ExecutionReserveBudget

    ledger = ExecutionReserveBudget(total_budget=5.0, scheduled_minimums=(1.0, 1.0))
    frozen, decision = ledger.admit_evolution(5.0, query_id="q-over-budget")
    assert not decision.accepted
    assert frozen.evolution_frozen is True
    assert frozen.scheduled_execution_allowed is True


def test_certificate_critical_acquisition_ignores_queries_that_cannot_change_decision():
    from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, run_acquisition

    queries = [
        AcquisitionQuery("q-irrelevant", "edge", 1.0),
        AcquisitionQuery("q-critical", "edge", 1.0, can_change_decisions=("deploy",)),
    ]
    result = run_acquisition(
        queries,
        {"q-irrelevant": True, "q-critical": True},
        AcquisitionPolicy.CERTIFICATE_CRITICAL,
        confidence_target=0.5,
    )
    assert result.selected_query_ids == ("q-critical",)
    assert result.stop_reason == "no_critical_query"


def test_lazy_relation_returns_underidentified_without_running_factorial_engine(monkeypatch):
    from core.acre import maintainer as maintainer_module
    from core.acre.engine import AcreEngine
    from core.acre.relation import RelationExperimentEligibility

    @dataclass
    class ExplodingEngine:
        def __init__(self, **_kwargs):
            raise AssertionError("FactorialEngine must not run for an ineligible relation")

    monkeypatch.setattr("core.acre.factorial.FactorialEngine", ExplodingEngine)
    result = AcreEngine().maintainer.execute_relation_experiment(
        {"ctx": []},
        eligibility=RelationExperimentEligibility(False, "single_rule_gate_missing"),
    )
    assert result.decision == "underidentified_context_relation"
    assert result.confidence["reason"] == "single_rule_gate_missing"


def test_certificate_cache_reuses_only_unchanged_dependencies():
    from core.acre.certificate_cache import compare_certificate_dependencies

    old = {
        "context_fields": ["workload.shape"],
        "model_id": "m1",
        "software": "s1",
        "hardware": "h1",
        "scientific_policy": "strict-v1",
    }
    same = dict(old)
    changed = {**old, "software": "s2"}

    reused = compare_certificate_dependencies(old, same, evidence_id="e1")
    invalidated = compare_certificate_dependencies(old, changed, evidence_id="e1")
    assert reused.reusable is True
    assert reused.invalidated_clauses == ()
    assert reused.receipt["source_evidence_id"] == "e1"
    assert invalidated.reusable is False
    assert "E" in invalidated.invalidated_clauses
    assert "software" in invalidated.receipt["changed_dependencies"]
