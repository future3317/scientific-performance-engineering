"""Finite-pool acquisition comparison for the ACRE pilot."""

from __future__ import annotations

from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, evaluate_trajectory, run_acquisition


def _fixture() -> tuple[list[AcquisitionQuery], dict[str, bool], dict[str, bool]]:
    queries = [
        AcquisitionQuery("q-a1", "edge-a", 2.0, experiment_type="factorial", risk=0.2, provenance_novelty=0.8),
        AcquisitionQuery("q-a2", "edge-a", 1.0, experiment_type="factorial", risk=0.8, provenance_novelty=0.5),
        AcquisitionQuery("q-b1", "edge-b", 1.0, experiment_type="factorial", risk=0.8, provenance_novelty=0.5),
        AcquisitionQuery("q-b2", "edge-b", 1.0, experiment_type="factorial", risk=0.5, provenance_novelty=0.4),
        AcquisitionQuery("q-c1", "edge-c", 2.0, experiment_type="factorial", risk=0.7, provenance_novelty=0.4),
        AcquisitionQuery("q-c2", "edge-c", 2.0, experiment_type="factorial", risk=0.4, provenance_novelty=0.3),
    ]
    labels = {"q-a1": True, "q-a2": True, "q-b1": False, "q-b2": False, "q-c1": True, "q-c2": True}
    truths = {"edge-a": True, "edge-b": False, "edge-c": True}
    return queries, labels, truths


def run_acquisition_benchmark(*, target_error: float = 0.0, seed: int = 7) -> dict[str, object]:
    queries, labels, truths = _fixture()
    results = {
        policy.value: (
            run_acquisition(queries, labels, policy, confidence_target=0.9, seed=seed),
            truths,
        )
        for policy in AcquisitionPolicy
    }
    evaluations = {
        name: evaluate_trajectory(trajectory, queries, labels, truths, target_error=target_error)
        for name, (trajectory, truths) in results.items()
    }
    return {
        "target_error": target_error,
        "cost_to_target": {name: evaluation.cost_to_target for name, evaluation in evaluations.items()},
        "final_error": {name: evaluation.final_error for name, evaluation in evaluations.items()},
        "selected_query_ids": {name: list(trajectory.selected_query_ids) for name, (trajectory, _) in results.items()},
    }
