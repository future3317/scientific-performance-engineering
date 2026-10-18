import copy

import pytest

from experiments.real_intervention_v2.contracts import digest, preexecution_context, validate_manifest


def manifest(contexts=None):
    return {
        "schema_version": 1,
        "run_id": "unit-test",
        "provider_calls": 0,
        "resource_caps": {"cpu_seconds": 1, "gpu_seconds": 1, "wall_seconds": 1,
                          "storage_bytes": 1, "download_bytes": 0},
        "device": "cpu",
        "seed": 7,
        "contexts": contexts if contexts is not None else [{
            "context_id": "ctx-a", "lineage_id": "lineage-a", "split": "development",
            "adapter": "torchvision", "model": "resnet18", "input_shape": [1, 3, 32, 32],
            "input_shapes": [[1, 3, 32, 32]],
            "planned_horizon": 8,
            "sessions": [{"session_id": "s1", "order": "eager_first"},
                         {"session_id": "s2", "order": "compile_first"}],
        }],
    }


def test_manifest_is_deterministically_digestable_and_preexec_only():
    value = manifest()
    validate_manifest(value)
    assert digest(value) == digest(copy.deepcopy(value))
    row = preexecution_context(value, value["contexts"][0])
    assert row["planned_horizon"] == 8
    assert "speedup" not in row


@pytest.mark.parametrize("field", ["speedup", "compile_seconds", "correctness", "benefit"])
def test_manifest_rejects_nested_post_execution_information(field):
    row = manifest()["contexts"][0] | {"metadata": {field: 1}}
    with pytest.raises(ValueError, match="outcome field"):
        validate_manifest(manifest([row]))


def test_model_lineage_cannot_cross_splits():
    rows = manifest()["contexts"] * 2
    rows[1] = rows[1] | {"context_id": "ctx-b", "split": "final_untouched"}
    with pytest.raises(ValueError, match="lineage crosses splits"):
        validate_manifest(manifest(rows))


def test_manifest_rejects_duplicate_contexts():
    with pytest.raises(ValueError, match="duplicate context_id"):
        validate_manifest(manifest(manifest()["contexts"] * 2))
