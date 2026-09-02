from __future__ import annotations

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
