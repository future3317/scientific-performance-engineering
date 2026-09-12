from __future__ import annotations

from benchmark.boundary.ltt import LTTConfig, LTTLogisticPolicy
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
        calibration_rows=[
            {"context": {"workload": {"x": -1}}, "unsupported": False},
            {"context": {"workload": {"x": 1}}, "unsupported": False},
        ],
        schema=_schema(),
        config=LTTConfig(threshold_grid=(0.2, 0.5, 0.8), target_risk=0.05, confidence=0.95),
    )
    assert policy.threshold in {0.2, 0.5, 0.8}
    assert policy.fit_source == "synthesis_then_calibration"
    assert isinstance(policy.decide({"workload": {"x": 1}}), Decision)


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
