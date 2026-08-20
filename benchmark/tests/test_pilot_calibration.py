from __future__ import annotations

from pathlib import Path

from benchmark.interaction.factorial_bench import run_family_factorial_benchmark
from benchmark.boundary.families import family_cases
from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, run_acquisition
from core.acre.engine import AcreEngine
from core.models import RuleSpec, RuleState, TaskContext
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


def test_interaction_diagnostic_does_not_stop_on_underidentified_context():
    report = run_family_factorial_benchmark(count=1, seed=3)
    row = report["surface_results"][0]
    assert row["stopping_blocks"] is None or row["stopping_blocks"] >= 128
    assert row["predicted_relation"] in {
        "underidentified_context_relation",
        "unresolved",
    } or row["stopping_blocks"] >= 128


def test_interaction_failure_rates_exclude_underidentified_from_resolved():
    report = run_family_factorial_benchmark(count=16, seed=3)
    assert report["wrong_relation_rate_among_resolved"] == 0.0
    assert report["unresolved_rate"] == report["total_identification_failure_rate"]


def test_evolution_episode_has_transfer_and_regret_evidence():
    result = run_drift_poison(root=Path(__file__).resolve().parents[2])
    assert result["D"]["status"] == "complete"
    assert result["D"]["metrics"]["library_growth"]["canonical_rule_count"] >= 1
    assert result["C"]["promoted_rules"] == []
    assert result["D"]["promoted_rules"]
    assert any(
        rule.get("intervention", {}).get("action") == "stabilize_dynamic_guards"
        for rule in result["D"]["canonical_rules"]
    )
    promotion_cases = {
        case_id
        for record in result["D"]["promotion_records"]
        for case_id in record.get("record", {}).get("promotion_case_ids", [])
    }
    assert "REG-COMPILE-RECOMPILE-04" not in promotion_cases
    # The episode must demonstrate governed reuse on the matched dynamic
    # context; oracle speedup alone is not an evolution result.
    assert result["D"]["metrics"]["transfer_gain"] > 0.0
    assert result["D"]["metrics"]["rule_reuse_utility"] > 0.0
    assert result["D"]["metrics"]["negative_transfer_rate"] == 0.0
    assert result["D"]["metrics"]["evolution_regret"]["total"] is not None


def test_router_keeps_realistic_positive_rule_with_prompt_cost() -> None:
    spec = RuleSpec(
        rule_id="REALISTIC-COST-RULE", version=1, parent=None,
        applicability={"all": [{"compare": {"workload.dynamic_shape_rate": {"gt": 0.0}}}]},
        intervention={"action": "stabilize_dynamic_guards"},
        expected_mechanism="compile_dynamic_shapes", evidence_requirements=["paired_replay"],
        scientific_invariants=["compile_correctness"], abstain_conditions={}, relations={},
        runtime_cost={}, provenance_policy={"required": True}, domain="compiler",
    )
    state = RuleState(
        rule_id=spec.rule_id, version=1, status="canonical", drift_state="stable",
        effect={"utility": 0.2, "lower_utility": 0.05},
        confidence_sequence={"utility_effect_lcb": 0.05, "utility_effect_ucb": 0.8},
    )
    decision = AcreEngine(rule_specs=[spec], rule_states={spec.rule_id: state}).route(
        TaskContext("compiler", {"dynamic_shape_rate": 0.3}, {}, {}, {}, 4096)
    )
    assert decision.selected_rule_ids == (spec.rule_id,)


def test_boundary_pool_contains_threshold_neighbors():
    pools = family_cases("compile", surface_count=24, seed=0)
    values = [
        float(item.context["workload"]["dynamic_shape_rate"])
        for name in ("sealed_boundary_pool", "representative_pool", "active_query_pool")
        for item in pools[name]
    ]
    assert any(abs(value - 0.0) < 1e-12 for value in values)
    assert any(abs(value - 0.4) < 1e-12 for value in values)


def test_boundary_diagnostic_honors_surface_count_for_sciml_family():
    from benchmark.boundary.families import run_boundary_family

    report = run_boundary_family("equivariant_head", surface_count=100, seed=7)
    assert report["pool_sizes"]["representative_pool"] > 8
    assert report["status"] == "identified"


def test_boundary_diagnostic_keeps_large_compile_slice_consistent():
    from benchmark.boundary.families import run_boundary_family

    report = run_boundary_family("compile", surface_count=100, seed=7)
    assert report["status"] == "identified"
    assert report["sealed_errors"] == 0
