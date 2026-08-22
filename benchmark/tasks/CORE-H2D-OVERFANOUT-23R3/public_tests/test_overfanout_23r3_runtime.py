from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark():
    path = Path(__file__).parents[1] / "benchmark.py"
    spec = importlib.util.spec_from_file_location("overfanout_23r3_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixture_declares_executable_counterexample_settings():
    fixtures = _load_benchmark().make_fixtures(seed=0, device="cpu")
    assert fixtures["data_config"]["batch_size"] == 32
    assert fixtures["worker_count"] == 5
    assert fixtures["prefetch_factor"] == 4
    assert fixtures["pin_memory"] is True


def test_runtime_loader_consumes_declared_counterexample_settings():
    module = _load_benchmark()
    fixtures = module.make_fixtures(seed=0, device="cpu")
    loader = module._make_runtime_dataloader(fixtures)
    assert loader.batch_size == 32
    assert loader.num_workers == 5
    assert loader.prefetch_factor == 4
    assert loader.pin_memory is True
    del loader
