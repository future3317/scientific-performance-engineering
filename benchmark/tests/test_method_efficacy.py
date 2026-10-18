from __future__ import annotations

import math
from pathlib import Path

from core.acre.cegis import BoundaryObservation


def _observation(observation_id: str, x: int, *, positive: bool) -> BoundaryObservation:
    return BoundaryObservation(
        observation_id=observation_id,
        context={"workload": {"x": x}},
        effect=1.0 if positive else 0.0,
        gate_passed=positive,
        effect_lower=1.0 if positive else 0.0,
        effect_upper=1.0 if positive else 0.0,
    )


def test_applicability_metrics_report_confusion_and_unsafe_rate() -> None:
    from benchmark.boundary.evaluator import evaluate_applicability

    result = evaluate_applicability(
        {"p": True, "n": True, "u": False},
        {"p": True, "n": False, "u": True},
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 0
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["unsafe_generalization_rate"] == 1 / 3
    assert result["abstention_rate"] == 1 / 3


def test_applicability_metrics_return_none_for_undefined_precision() -> None:
    from benchmark.boundary.evaluator import evaluate_applicability

    result = evaluate_applicability({"n": False}, {"n": False})

    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["unsafe_generalization_rate"] == 0.0
    assert result["abstention_rate"] == 1.0


def test_nearest_experience_uses_visible_context_only() -> None:
    from benchmark.boundary.baselines import predict_applicability

    visible = [_observation("p", 2, positive=True), _observation("n", 10, positive=False)]
    heldout = [_observation("near", 3, positive=True), _observation("far", 20, positive=False)]

    predictions = predict_applicability("nearest_experience", visible, heldout)

    assert set(predictions) == {"near", "far"}
    assert predictions["near"] is True
    assert predictions["far"] is False


def test_positive_only_baseline_does_not_read_hidden_labels() -> None:
    from benchmark.boundary.baselines import predict_applicability

    visible = [_observation("p", 2, positive=True)]
    heldout = [_observation("same", 2, positive=False)]

    predictions = predict_applicability("observed_positive_only", visible, heldout)

    assert predictions == {"same": True}


def test_acre_policy_api_uses_observed_fields_not_hidden_applicability() -> None:
    from benchmark.boundary.baselines import predict_applicability

    visible = [
        _observation("p", 2, positive=True),
        _observation("n", 10, positive=False),
    ]
    heldout = [_observation("u", 3, positive=True)]

    predictions = predict_applicability("acre_full", visible, heldout, family="compile")

    assert set(predictions) == {"u"}
    assert all(not hasattr(item, "expected_applicable") for item in visible + heldout)


def test_oracle_separation_audit_blocks_hidden_labels_from_policy_api() -> None:
    from scripts.audit_oracle_separation import run_oracle_separation_audit

    report = run_oracle_separation_audit()

    assert report["status"] == "PASS"
    assert report["provider_calls"] == 0
    assert all(check["passed"] for check in report["checks"].values())


def test_acre_full_requires_a_certified_counterexample() -> None:
    from benchmark.boundary.baselines import predict_applicability

    visible = [_observation("p", 2, positive=True)]
    heldout = [_observation("same", 2, positive=True)]

    predictions = predict_applicability("acre_full", visible, heldout, family="compile")

    assert predictions == {"same": False}


def test_method_efficacy_runner_covers_all_families_and_prefixes() -> None:
    from scripts.run_pilot_surface_experiments import run_method_efficacy

    report = run_method_efficacy(surface_count=12, seed=7)

    assert report["provider_calls"] == 0
    assert report["family_count"] == 11
    assert report["policies"] == [
        "always_apply",
        "observed_positive_only",
        "nearest_experience",
        "untyped_positive_summary",
        "shortest_without_counterexamples",
        "acre_full",
    ]
    assert report["evidence_prefixes"] == [1, 2, 4, 8, 16, "all"]
    assert len(report["family_results"]) == 11
    assert set(report["aggregate"]) == set(report["policies"])
    assert "pairwise_acre" in report
    assert "nearest_experience" in report["pairwise_acre"]
    assert "unsafe_generalization_rate" in report["pairwise_acre"]["nearest_experience"]["all"]


def test_provider_free_ablation_reports_common_contract_and_components() -> None:
    from scripts.run_pilot_surface_experiments import run_provider_free_ablation

    report = run_provider_free_ablation(
        root=Path(__file__).resolve().parents[2],
        surface_count=12,
        seed=7,
    )

    assert report["provider_calls"] == 0
    assert set(report["applicability"]) == {
        "acre_full",
        "minus_counterexample",
        "minus_typed",
    }
    assert "router" in report
    assert "lifecycle" in report
    assert len(report["lifecycle"]["episodes"]) == 3


def test_support_boundary_package_keeps_calibration_and_evaluation_separate() -> None:
    from scripts.run_pilot_surface_experiments import run_support_boundary_package

    report = run_support_boundary_package(surface_count=12, seed=7, subset_seeds=2)

    assert report["provider_calls"] == 0
    assert report["family_count"] == 11
    assert report["layers"][-1] == "full_acre"
    assert all(
        row["calibration_count"] > 0 and row["evaluation_count"] > 0
        for row in report["family_rows"]
    )
    assert report["counterexample_dose"]["subset_seeds"] == 2
    assert set(report["counterexample_dose"]["summary"]) == {"0.0", "0.25", "0.5", "1.0"}
    assert "pooled_unsupported_upper_95_if_zero" not in report["counterexample_dose"]["summary"]["1.0"]
    assert "subset_seed_summary" in report["family_rows"][0]["counterexample_dose"]["1.0"]


def test_matched_coverage_can_select_zero_deployment_with_unit_similarity(monkeypatch) -> None:
    from types import SimpleNamespace
    from scripts import run_pilot_surface_experiments as pilot

    observation = SimpleNamespace(observation_id="cal", positive_anchor=lambda: True)
    monkeypatch.setattr(pilot, "predict_applicability", lambda *args, **kwargs: {"cal": False})
    monkeypatch.setattr(pilot, "_positive_similarity", lambda *args: 1.0)
    threshold, coverage, _ = pilot._matched_positive_threshold([observation], [observation], family="compile")
    assert coverage == 0.0
    assert threshold > 1.0
    assert pilot._matched_positive_predictions([observation], [observation], threshold) == {"cal": False}


def test_support_boundary_replication_uses_frozen_policy_artifacts() -> None:
    from scripts.run_pilot_surface_experiments import (
        run_support_boundary_package,
        run_support_boundary_replication,
    )

    frozen = run_support_boundary_package(surface_count=24, seed=7, subset_seeds=1)
    report = run_support_boundary_replication(
        frozen,
        surface_count=24,
        seeds=(8,),
    )

    assert report["provider_calls"] == 0
    assert report["policy_source"] == "frozen_support_boundary_package"
    assert report["fit_seeds"] == []
    assert report["replications"][0]["seed"] == 8
    assert set(report["replications"][0]["policies"]) >= {"matched_coverage", "full_acre"}
    full_counts = report["replications"][0]["policies"]["full_acre"]["raw_counts"]
    assert report["replications"][0]["evaluation_context_count"] == sum(full_counts.values())
    assert set(report["aggregate_over_seeds"]) >= {"matched_coverage", "full_acre"}


def test_support_boundary_replication_rejects_source_training_seed() -> None:
    import pytest
    from scripts.run_pilot_surface_experiments import run_support_boundary_package, run_support_boundary_replication

    frozen = run_support_boundary_package(surface_count=12, seed=7, subset_seeds=1)
    with pytest.raises(ValueError, match="source seed"):
        run_support_boundary_replication(frozen, surface_count=12, seeds=(7,))


def test_counterexample_dose_reanalysis_reports_family_conditional_risk() -> None:
    from scripts.reanalyze_counterexample_dose import reanalyze_counterexample_dose

    def row(dose: str, tp: int, fp: int) -> dict[str, object]:
        return {
            "dose": float(dose), "subset_seed": 0, "counterexample_count": 1,
            "tp": tp, "fp": fp, "fn": 1, "tn": 1,
            "unsafe_generalization_rate": fp / 12,
            "description_length": 3,
        }

    source = {
        "seed": 7,
        "counterexample_dose": {"subset_seeds": 1},
        "family_rows": [{
            "family": "fixture",
            "counterexample_dose": {
                dose: {"rows": [row(dose, 10 if dose != "0.25" else 9, 0 if dose != "0.25" else 1)]}
                for dose in ("0.0", "0.25", "0.5", "1.0")
            },
        }],
    }

    report = reanalyze_counterexample_dose(source)

    assert report["family_count"] == 1
    assert report["risk_25_minus_risk_100"]["non_increasing_family_count"] == 1
    assert report["families"][0]["doses"]["0.25"]["mean_coverage"] == 10 / 12
    assert report["family_cluster_unsafe_rate_all_contexts"]["0.25"]["family_count"] == 1


def test_relation_metrics_separate_coverage_from_resolved_accuracy() -> None:
    from benchmark.interaction.factorial_bench import relation_recovery_metrics

    metrics = relation_recovery_metrics([
        {"hidden_relation": "synergy", "predicted_relation": "synergy"},
        {"hidden_relation": "antagonism", "predicted_relation": "unresolved"},
        {"hidden_relation": "independence", "predicted_relation": "synergy"},
    ])

    assert metrics["coverage"] == 2 / 3
    assert metrics["overall_accuracy"] == 1 / 3
    assert metrics["resolved_accuracy"] == 0.5
    assert metrics["abstention_accuracy"] is None


def test_observability_bounds_collapse_without_censoring() -> None:
    from benchmark.formal.observability_simulation import partial_identification_bounds

    result = partial_identification_bounds(
        [0.2, 0.6, 0.8],
        [0.1, 0.3, 0.4],
        [True, True, True],
    )

    assert math.isclose(result["true_effect"], 0.26666666666666666)
    assert math.isclose(result["lower"], result["upper"])
    assert math.isclose(result["lower"], result["true_effect"])
    assert math.isclose(result["complete_case_effect"], result["true_effect"])
    assert result["interval_width"] >= 0
    assert math.isclose(result["interval_width"], 0.0)


def test_observability_bounds_cover_all_censored_treatment_outcomes() -> None:
    from benchmark.formal.observability_simulation import partial_identification_bounds

    result = partial_identification_bounds(
        [0.2, 0.6, 0.8],
        [0.1, 0.3, 0.4],
        [False, False, False],
    )

    assert result["complete_case_effect"] is None
    assert math.isclose(result["lower"], -0.26666666666666666)
    assert math.isclose(result["upper"], 0.7333333333333334)
    assert result["lower"] <= result["true_effect"] <= result["upper"]


def test_observability_simulation_distinguishes_independent_and_coupled_censoring() -> None:
    from benchmark.formal.observability_simulation import run_observability_simulation

    report = run_observability_simulation(
        censoring_rates=(0.0, 0.5),
        population_size=40,
        repetitions=20,
        seed=7,
    )

    assert report["provider_calls"] == 0
    assert report["censoring_rates"] == [0.0, 0.5]
    assert {row["mechanism"] for row in report["results"]} == {
        "independent",
        "treatment_coupled",
    }
    assert all(row["coverage"] == 1.0 for row in report["results"])
    independent = next(row for row in report["results"] if row["mechanism"] == "independent" and row["censoring_rate"] == 0.5)
    coupled = next(row for row in report["results"] if row["mechanism"] == "treatment_coupled" and row["censoring_rate"] == 0.5)
    assert coupled["mean_complete_case_bias"] < independent["mean_complete_case_bias"]
