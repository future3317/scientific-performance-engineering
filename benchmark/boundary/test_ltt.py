from __future__ import annotations

from benchmark.boundary.ltt import LTTConfig, LTTLogisticPolicy, select_ltt_threshold
from benchmark.boundary.typed_features import TypedFeatureSchema
from benchmark.boundary.policy_eval import Decision


def _schema() -> TypedFeatureSchema:
    return TypedFeatureSchema(
        family="fixture",
        paths=("workload.x",),
        features=({"path": "workload.x", "type": "numeric", "mean": 0.0, "std": 1.0},),
        output_width=2,
    )


def test_ltt_threshold_is_selected_from_calibration_only() -> None:
    policy = LTTLogisticPolicy.fit(
        synthesis_rows=[
            {"context": {"workload": {"x": -2}}, "label": False},
            {"context": {"workload": {"x": 2}}, "label": True},
        ],
        calibration_rows=[{"context": {"workload": {"x": 1}}, "unsupported": False} for _ in range(100)],
        schema=_schema(),
        config=LTTConfig(threshold_grid=(0.2, 0.5, 0.8), target_risk=0.05, confidence=0.95),
    )
    assert policy.threshold in {0.2, 0.5, 0.8}
    assert policy.fit_source == "synthesis_then_calibration"
    assert isinstance(policy.decide({"workload": {"x": 1}}), Decision)
    assert policy.calibration_deployed == 100
    assert policy.calibration_risk_upper <= 0.05


def test_ltt_uses_all_abstention_when_calibration_cannot_certify_risk() -> None:
    policy = LTTLogisticPolicy.fit(
        synthesis_rows=[{"context": {"workload": {"x": 1}}, "label": True}],
        calibration_rows=[{"context": {"workload": {"x": 1}}, "unsupported": True}],
        schema=_schema(),
        config=LTTConfig(threshold_grid=(0.2,), target_risk=0.05, confidence=0.95),
    )
    decision = policy.decide({"workload": {"x": 1}})
    assert decision.action == "abstain"
    assert decision.reason == "risk_not_certified"


def test_ltt_fit_is_deterministic_and_does_not_store_evaluation_rows() -> None:
    kwargs = dict(
        synthesis_rows=[
            {"context": {"workload": {"x": -1}}, "label": False},
            {"context": {"workload": {"x": 1}}, "label": True},
        ],
        calibration_rows=[{"context": {"workload": {"x": 0}}, "unsupported": False}],
        schema=_schema(),
        config=LTTConfig.default(),
    )
    first = LTTLogisticPolicy.fit(**kwargs)
    second = LTTLogisticPolicy.fit(**kwargs)
    assert first.to_dict() == second.to_dict()
    assert "evaluation" not in first.to_dict()


def test_empty_calibration_cannot_certify_a_deployment_threshold() -> None:
    policy = LTTLogisticPolicy.fit(
        synthesis_rows=[{"context": {"workload": {"x": 1}}, "label": True}],
        calibration_rows=[], schema=_schema(), config=LTTConfig.default(),
    )
    assert policy.threshold is None
    assert policy.calibration_risk_upper is None
    assert policy.decide({"workload": {"x": 1}}).action == "abstain"


def test_threshold_selection_uses_simultaneous_not_pointwise_bounds() -> None:
    policy = LTTLogisticPolicy.fit(
        synthesis_rows=[{"context": {"workload": {"x": 1}}, "label": True}],
        calibration_rows=[{"context": {"workload": {"x": 1}}, "unsupported": False} for _ in range(60)],
        schema=_schema(), config=LTTConfig(threshold_grid=(0.0, 0.1), target_risk=0.05, confidence=0.95),
    )
    # 60 error-free observations pass a pointwise 95% bound, but not either
    # threshold's 97.5% bound required for selecting from two candidates.
    assert policy.threshold is None


def test_shared_ltt_threshold_selection_does_not_certify_zero_deployments() -> None:
    selected = select_ltt_threshold(
        [(0.1, False)],
        LTTConfig(threshold_grid=(0.5, 0.8), target_risk=0.05, confidence=0.95),
    )
    assert selected == (None, None, 0, 0)


def test_shared_ltt_threshold_selection_applies_bonferroni_correction() -> None:
    # 60 error-free observations do not pass the per-threshold 97.5% upper bound.
    selected = select_ltt_threshold(
        [(0.9, False)] * 60,
        LTTConfig(threshold_grid=(0.0, 0.1), target_risk=0.05, confidence=0.95),
    )
    assert selected == (None, None, 0, 0)
