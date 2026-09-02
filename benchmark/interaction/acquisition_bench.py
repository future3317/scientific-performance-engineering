"""Finite-pool acquisition comparison for the ACRE pilot."""

from __future__ import annotations

from dataclasses import replace

from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, evaluate_trajectory, run_acquisition
from core.acre.resource_budget import ExecutionReserveBudget


def _fixture() -> tuple[list[AcquisitionQuery], dict[str, bool], dict[str, bool]]:
    queries = [
        AcquisitionQuery("q-a1", "edge-a", 2.0, context={"decision_relevant": False}, experiment_type="factorial", risk=0.2, provenance_novelty=0.8),
        AcquisitionQuery("q-a2", "edge-a", 1.0, context={"decision_relevant": True}, experiment_type="factorial", risk=0.8, provenance_novelty=0.5, can_change_decisions=("deploy",)),
        AcquisitionQuery("q-b1", "edge-b", 1.0, context={"decision_relevant": True}, experiment_type="factorial", risk=0.8, provenance_novelty=0.5, can_change_decisions=("abstain",)),
        AcquisitionQuery("q-b2", "edge-b", 1.0, context={"decision_relevant": False}, experiment_type="factorial", risk=0.5, provenance_novelty=0.4),
        AcquisitionQuery("q-c1", "edge-c", 2.0, context={"decision_relevant": False}, experiment_type="factorial", risk=0.7, provenance_novelty=0.4),
        AcquisitionQuery("q-c2", "edge-c", 2.0, context={"decision_relevant": True}, experiment_type="factorial", risk=0.4, provenance_novelty=0.3, can_change_decisions=("specialize",)),
    ]
    labels = {"q-a1": True, "q-a2": True, "q-b1": False, "q-b2": False, "q-c1": True, "q-c2": True}
    truths = {"edge-a": True, "edge-b": False, "edge-c": True}
    return queries, labels, truths


def run_acquisition_benchmark(*, target_error: float = 0.0, seed: int = 7) -> dict[str, object]:
    queries, labels, truths = _fixture()
    def decision_sensitivity(query: AcquisitionQuery, _observations: dict[str, list[bool]]) -> float:
        # The formal benchmark adapter replaces this fixture callback with a
        # hypothetical CEGIS+router simulation.  The pilot fixture exposes the
        # same decision-relevance contract without consulting sealed labels.
        return 1.0 if bool(query.context.get("decision_relevant", False)) else 0.0
    results = {
        policy.value: (
            run_acquisition(
                queries, labels, policy, confidence_target=0.9, seed=seed,
                decision_sensitivity_fn=decision_sensitivity if policy is AcquisitionPolicy.DECISION_AWARE else None,
            ),
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


def run_budget_aware_acquisition_benchmark(*, seed: int = 7) -> dict[str, object]:
    """Compare legacy and certificate-critical acquisition without a provider."""
    queries, labels, truths = _fixture()
    reserve = ExecutionReserveBudget(total_budget=12.0, scheduled_minimums=(1.0, 1.0, 1.0))
    configurations = {
        "legacy_acre": (queries, AcquisitionPolicy.DECISION_AWARE, None),
        "execution_reserve": (queries, AcquisitionPolicy.DECISION_AWARE, reserve),
        "certificate_critical": (queries, AcquisitionPolicy.CERTIFICATE_CRITICAL, None),
        "full_budget_aware_acre": (queries, AcquisitionPolicy.CERTIFICATE_CRITICAL, reserve),
    }
    report: dict[str, object] = {"provider_calls": 0, "variants": {}}
    for name, (pool, policy, ledger) in configurations.items():
        trajectory = run_acquisition(
            pool, labels, policy, confidence_target=0.9, seed=seed,
            resource_budget=ledger,
        )
        evaluation = evaluate_trajectory(trajectory, pool, labels, truths, target_error=0.0)
        report["variants"][name] = {
            "selected_queries": len(trajectory.selected_query_ids),
            "total_cost": trajectory.total_cost,
            "stop_reason": trajectory.stop_reason,
            "target_reached": evaluation.target_reached,
            "final_error": evaluation.final_error,
            "evolution_frozen": trajectory.stop_reason in {"evolution_frozen", "execution_reserve"},
            "provider_calls": 0,
        }
    return report
