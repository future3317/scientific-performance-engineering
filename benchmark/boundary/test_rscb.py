from __future__ import annotations

import math

from benchmark.boundary.rscb import RSCBConfig, RSCBModel, matched_threshold
from benchmark.boundary.typed_features import TypedFeatureSchema


def _fixture() -> tuple[TypedFeatureSchema, list[dict]]:
    rows = [
        {"context": {"workload": {"graph_size": 10, "dynamic_shape_rate": 0.0, "logical_steps": 64}}, "label": False},
        {"context": {"workload": {"graph_size": 20, "dynamic_shape_rate": 0.1, "logical_steps": 64}}, "label": False},
        {"context": {"workload": {"graph_size": 100, "dynamic_shape_rate": 0.5, "logical_steps": 192}}, "label": True},
        {"context": {"workload": {"graph_size": 120, "dynamic_shape_rate": 0.6, "logical_steps": 192}}, "label": True},
    ]
    return TypedFeatureSchema.from_calibration("compile", [row["context"] for row in rows]), rows


def test_default_config_is_frozen_and_explicit() -> None:
    config = RSCBConfig.default()
    assert config.k == 1.64485362695147
    assert config.theta == 0.90
    assert config.max_iterations == 100
    assert config.tolerance == 1e-9
    assert config.threshold_grid == (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def test_fit_is_deterministic_and_produces_finite_laplace_posterior() -> None:
    schema, rows = _fixture()
    first = RSCBModel.fit(rows, schema, RSCBConfig.default())
    second = RSCBModel.fit(rows, schema, RSCBConfig.default())
    assert first.to_dict() == second.to_dict()
    assert all(math.isfinite(value) for value in first.coefficients)
    assert all(math.isfinite(value) for row in first.covariance for value in row)


def test_lcb_decreases_or_stays_equal_as_penalty_increases() -> None:
    schema, rows = _fixture()
    model = RSCBModel.fit(rows, schema, RSCBConfig.default())
    context = rows[-1]["context"]
    values = [model.lcb(context, k) for k in (0.0, 0.67448975, 1.64485362695147, 2.32634787)]
    assert all(left + 1e-12 >= right for left, right in zip(values, values[1:]))
    assert all(0.0 <= value <= 1.0 for value in values)


def test_matched_threshold_uses_only_calibration_coverage_and_tie_breaks_high() -> None:
    schema, rows = _fixture()
    model = RSCBModel.fit(rows, schema, RSCBConfig.default())
    threshold = matched_threshold(model, [row["context"] for row in rows], target_coverage=0.5)
    assert threshold in RSCBConfig.default().threshold_grid
    assert model.config.theta == 0.90
    assert matched_threshold(model, [row["context"] for row in rows], target_coverage=0.0) == 0.95
