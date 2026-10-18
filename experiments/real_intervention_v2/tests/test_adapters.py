import sys
import types

import pytest

torch = pytest.importorskip("torch")

from experiments.real_intervention_v2.adapters import load_model


def test_torchbench_get_module_returns_model_and_inputs(monkeypatch):
    model = torch.nn.Linear(3, 2)
    inputs = (torch.ones(1, 3),)
    benchmark_module = types.ModuleType("torchbenchmark.models.fake")

    class Model:
        def __init__(self, **kwargs):
            assert kwargs == {"test": "eval", "device": "cpu", "batch_size": 1}

        def get_module(self):
            return model, inputs

        def get_example_inputs(self):
            raise AssertionError("adapter must use the paired get_module() contract")

    benchmark_module.Model = Model
    package = types.ModuleType("torchbenchmark")
    package.__path__ = []
    models = types.ModuleType("torchbenchmark.models")
    models.__path__ = []
    monkeypatch.setitem(sys.modules, "torchbenchmark", package)
    monkeypatch.setitem(sys.modules, "torchbenchmark.models", models)
    monkeypatch.setitem(sys.modules, "torchbenchmark.models.fake", benchmark_module)
    loaded, loaded_inputs = load_model({"adapter": "torchbench", "model": "fake", "batch_size": 1}, "cpu")
    assert loaded is model
    assert loaded_inputs == inputs
    assert not loaded.training
