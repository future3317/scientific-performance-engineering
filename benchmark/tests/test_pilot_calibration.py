from __future__ import annotations

from pathlib import Path

from benchmark.interaction.factorial_bench import run_family_factorial_benchmark
from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, run_acquisition
from scripts.run_pilot_surface_experiments import run_drift_poison


def test_acquisition_certificate_is_observable_and_requires_repeated_evidence():
    queries = [
        AcquisitionQuery(f"q-{index}", "edge", 1.0)
        for index in range(4)
    ]
    labels = {query.query_id: True for query in queries}
    result = run_acquisition(queries, labels, AcquisitionPolicy.UNCERTAINTY_ONLY, confidence_target=0.5)
    # Four observations are intentionally insufficient for the formal
    # time-uniform CS; no posterior heuristic may certify this trajectory.
    assert not result.identification_certificate
    assert result.selection_trace[-1]["confidence_sequence"]["edge"][0] < 0.5
    assert len(result.selected_query_ids) == 4
    assert all("certificate_min_confidence" in item for item in result.selection_trace)


def test_sequential_interaction_records_surface_level_evidence():
    report = run_family_factorial_benchmark(count=16, seed=3)
    assert set(report["block_schedule"]) == {8, 16, 32, 64, 128}
    assert len(report["surface_results"]) == 16
    row = report["surface_results"][0]
    assert {"hidden_relation", "predicted_relation", "gamma_lcb", "gamma_ucb", "stopping_blocks"} <= set(row)
    assert report["confusion_matrix"]


def test_evolution_episode_has_transfer_and_regret_evidence():
    result = run_drift_poison(root=Path(__file__).resolve().parents[2])
    assert result["D"]["status"] == "complete"
    assert result["D"]["metrics"]["library_growth"]["canonical_rule_count"] >= 1
    assert result["D"]["metrics"]["transfer_gain"] >= 0.0
    assert result["D"]["metrics"]["evolution_regret"]["total"] is not None
