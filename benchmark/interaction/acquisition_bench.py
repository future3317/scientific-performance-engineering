"""Finite-pool acquisition comparison for the ACRE pilot."""

from __future__ import annotations

from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, run_acquisition


def _fixture() -> tuple[list[AcquisitionQuery], dict[str, bool], dict[str, bool]]:
    queries = [
        AcquisitionQuery("q-a1", "edge-a", 0.9, 0.1, 2.0),
        AcquisitionQuery("q-a2", "edge-a", 0.5, 1.0, 1.0),
        AcquisitionQuery("q-b1", "edge-b", 0.7, 1.0, 1.0),
        AcquisitionQuery("q-b2", "edge-b", 0.4, 1.0, 1.0),
        AcquisitionQuery("q-c1", "edge-c", 0.6, 0.8, 2.0),
        AcquisitionQuery("q-c2", "edge-c", 0.3, 0.8, 2.0),
    ]
    labels = {"q-a1": True, "q-a2": True, "q-b1": False, "q-b2": False, "q-c1": True, "q-c2": True}
    truths = {"edge-a": True, "edge-b": False, "edge-c": True}
    return queries, labels, truths


def run_acquisition_benchmark(*, target_error: float = 0.0, seed: int = 7) -> dict[str, object]:
    queries, labels, truths = _fixture()
    results = {
        policy.value: run_acquisition(queries, labels, truths, policy, target_error=target_error, seed=seed)
        for policy in AcquisitionPolicy
    }
    return {
        "target_error": target_error,
        "cost_to_target": {name: result.cost_to_target for name, result in results.items()},
        "final_error": {name: result.final_error for name, result in results.items()},
        "selected_query_ids": {name: list(result.selected_query_ids) for name, result in results.items()},
    }
