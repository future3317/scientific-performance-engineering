from __future__ import annotations

from core.acre.acquisition import (
    AcquisitionQuery,
    AcquisitionPolicy,
    run_acquisition,
)


def pool() -> tuple[list[AcquisitionQuery], dict[str, bool], dict[str, bool]]:
    queries = [
        AcquisitionQuery("q-a1", "edge-a", uncertainty=0.9, decision_value=0.1, cost=2.0),
        AcquisitionQuery("q-a2", "edge-a", uncertainty=0.5, decision_value=1.0, cost=1.0),
        AcquisitionQuery("q-b1", "edge-b", uncertainty=0.7, decision_value=1.0, cost=1.0),
        AcquisitionQuery("q-b2", "edge-b", uncertainty=0.4, decision_value=1.0, cost=1.0),
        AcquisitionQuery("q-c1", "edge-c", uncertainty=0.6, decision_value=0.8, cost=2.0),
        AcquisitionQuery("q-c2", "edge-c", uncertainty=0.3, decision_value=0.8, cost=2.0),
    ]
    labels = {"q-a1": True, "q-a2": True, "q-b1": False, "q-b2": False, "q-c1": True, "q-c2": True}
    truths = {"edge-a": True, "edge-b": False, "edge-c": True}
    return queries, labels, truths


def test_decision_aware_acquisition_reaches_target_with_cost_report() -> None:
    queries, labels, truths = pool()
    result = run_acquisition(queries, labels, truths, AcquisitionPolicy.DECISION_AWARE, target_error=0.0)
    assert result.target_reached
    assert result.cost_to_target == 4.0
    assert result.final_error == 0.0


def test_policies_share_pool_and_selection_has_no_sealed_labels() -> None:
    queries, labels, truths = pool()
    for policy in AcquisitionPolicy:
        result = run_acquisition(queries, labels, truths, policy, target_error=0.0)
        assert set(result.selected_query_ids) <= {query.query_id for query in queries}
        assert result.cumulative_cost[-1] == result.total_cost
        assert all("label" not in selection for selection in result.selection_trace)


def test_query_contract_rejects_invalid_cost() -> None:
    try:
        AcquisitionQuery("bad", "edge", 0.5, 0.5, 0.0)
    except ValueError as exc:
        assert "cost" in str(exc)
    else:
        raise AssertionError("invalid query cost was accepted")
