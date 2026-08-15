from __future__ import annotations

from core.acre.acquisition import (
    AcquisitionQuery,
    AcquisitionPolicy,
    evaluate_trajectory,
    run_acquisition,
)


def pool() -> tuple[list[AcquisitionQuery], dict[str, bool], dict[str, bool]]:
    queries = [
        AcquisitionQuery("q-a1", "edge-a", cost=2.0, experiment_type="factorial", risk=0.2, provenance_novelty=0.8),
        AcquisitionQuery("q-a2", "edge-a", cost=1.0, experiment_type="factorial", risk=0.8, provenance_novelty=0.5),
        AcquisitionQuery("q-b1", "edge-b", cost=1.0, experiment_type="factorial", risk=0.8, provenance_novelty=0.5),
        AcquisitionQuery("q-b2", "edge-b", cost=1.0, experiment_type="factorial", risk=0.5, provenance_novelty=0.4),
        AcquisitionQuery("q-c1", "edge-c", cost=2.0, experiment_type="factorial", risk=0.7, provenance_novelty=0.4),
        AcquisitionQuery("q-c2", "edge-c", cost=2.0, experiment_type="factorial", risk=0.4, provenance_novelty=0.3),
    ]
    labels = {"q-a1": True, "q-a2": True, "q-b1": False, "q-b2": False, "q-c1": True, "q-c2": True}
    truths = {"edge-a": True, "edge-b": False, "edge-c": True}
    return queries, labels, truths


def test_decision_aware_acquisition_reaches_target_with_cost_report() -> None:
    queries, labels, truths = pool()
    result = run_acquisition(queries, labels, AcquisitionPolicy.DECISION_AWARE, confidence_target=0.9)
    evaluation = evaluate_trajectory(result, queries, labels, truths, target_error=0.0)
    assert evaluation.target_reached
    assert evaluation.cost_to_target == 4.0
    assert evaluation.final_error == 0.0


def test_policies_share_pool_and_selection_has_no_sealed_labels() -> None:
    queries, labels, truths = pool()
    for policy in AcquisitionPolicy:
        result = run_acquisition(queries, labels, policy, confidence_target=0.9)
        assert set(result.selected_query_ids) <= {query.query_id for query in queries}
        assert result.cumulative_cost[-1] == result.total_cost
        assert all("label" not in selection for selection in result.selection_trace)


def test_decision_value_is_recomputed_from_revealed_posterior() -> None:
    queries, labels, _ = pool()
    states: list[dict[str, tuple[bool, ...]]] = []

    def decision_value(query: AcquisitionQuery, observations: dict[str, list[bool]]) -> float:
        states.append({edge: tuple(values) for edge, values in observations.items()})
        return 1.0 if query.edge_id not in observations else 0.1

    result = run_acquisition(
        queries,
        labels,
        AcquisitionPolicy.DECISION_AWARE,
        confidence_target=0.9,
        decision_sensitivity_fn=decision_value,
    )
    assert states and states[0] == {}
    assert any(state for state in states[1:])
    assert all("decision_sensitivity" in item for item in result.selection_trace)


def test_query_contract_rejects_invalid_cost() -> None:
    try:
        AcquisitionQuery("bad", "edge", 0.0)
    except ValueError as exc:
        assert "cost" in str(exc)
    else:
        raise AssertionError("invalid query cost was accepted")
