import pytest

from experiments.real_intervention_v3.arm_worker import _validate_spec


def _spec():
    return {"source": "torchvision", "model": "resnet18", "family": "resnet", "batch": 1,
            "shape": [3, 224, 224], "seed": 31, "horizon": 16, "arm": "S1",
            "intervention": "compile", "reference_path": "reference.pt",
            "correctness": {"rtol": 1e-4, "atol": 1e-5}}


def test_worker_accepts_requested_stable_spec():
    _validate_spec(_spec())


def test_worker_rejects_missing_frozen_baseline_reference():
    spec = _spec()
    spec.pop("reference_path")
    with pytest.raises(ValueError, match="reference_path"):
        _validate_spec(spec)


def test_worker_requires_pinned_torchbench_checkout():
    spec = _spec()
    spec.update(source="torchbench", model="resnet50")
    with pytest.raises(ValueError, match="torchbench_root"):
        _validate_spec(spec)
