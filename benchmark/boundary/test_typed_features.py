from __future__ import annotations

import pytest

from benchmark.boundary.typed_features import TypedFeatureSchema


def test_schema_uses_sorted_registered_paths_and_calibration_statistics() -> None:
    contexts = [
        {"workload": {"graph_size": 10, "dynamic_shape_rate": 0.0, "kind": "b"}},
        {"workload": {"graph_size": 20, "dynamic_shape_rate": 1.0, "kind": "a"}},
    ]
    schema = TypedFeatureSchema.from_calibration("compile", contexts)
    assert schema.paths == tuple(sorted(schema.paths))
    assert "workload.graph_size" in schema.paths
    assert schema.to_dict()["fit_source"] == "calibration"
    transformed = schema.transform({"workload": {"graph_size": 15, "dynamic_shape_rate": 0.5}})
    assert len(transformed) == schema.output_width
    assert all(value == value for value in transformed)


def test_categorical_vocabulary_is_calibration_only_and_unknown_is_deterministic() -> None:
    contexts = [
        {"workload": {"runtime_version": "v1", "context_width": 2, "drift_rate": 0.1}},
        {"workload": {"runtime_version": "v2", "context_width": 4, "drift_rate": 0.2}},
    ]
    schema = TypedFeatureSchema.from_calibration("episode", contexts)
    payload = schema.to_dict()
    categorical = next(item for item in payload["features"] if item["path"] == "workload.runtime_version")
    assert categorical["vocabulary"] == ["v1", "v2", "<UNK>"]
    unknown = schema.transform({"workload": {"runtime_version": "evaluation-only", "context_width": 4, "drift_rate": 0.2}})
    unknown_again = schema.transform({"workload": {"runtime_version": "evaluation-only", "context_width": 4, "drift_rate": 0.2}})
    assert unknown == unknown_again


def test_missing_public_feature_has_missing_indicator_without_using_hidden_fields() -> None:
    contexts = [{"workload": {"graph_size": 10, "dynamic_shape_rate": 0.2, "logical_steps": 64}}]
    schema = TypedFeatureSchema.from_calibration("compile", contexts)
    clean = schema.transform({"workload": {"graph_size": 10, "dynamic_shape_rate": 0.2, "logical_steps": 64}})
    missing = schema.transform({
        "workload": {"graph_size": 10, "dynamic_shape_rate": 0.2, "logical_steps": 64},
        "expected_applicable": True,
        "tau": 100.0,
        "scientific_gate": False,
        "oracle_label": True,
        "split": "evaluation",
    })
    assert clean == missing
    partially_missing = schema.transform({"workload": {"graph_size": 10, "dynamic_shape_rate": 0.2}})
    assert partially_missing != clean


def test_schema_rejects_unknown_family_and_non_public_calibration_record() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        TypedFeatureSchema.from_calibration("does-not-exist", [])
    with pytest.raises(ValueError, match="public context"):
        TypedFeatureSchema.from_calibration("compile", [{"context": {"workload": {}}, "tau": 1.0}])
