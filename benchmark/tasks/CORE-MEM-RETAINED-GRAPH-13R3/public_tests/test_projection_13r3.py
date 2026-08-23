from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_projection_uses_declared_long_horizon_instance():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_13r3", root / "benchmark.py")
    solution = _load("solution_13r3", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["memory_pressure"] == 0.82
    assert fixtures["segment_count"] == 12
    assert fixtures["recompute_ratio"] == 0.5
    assert fixtures["checkpointed_segments"] == 6
    assert fixtures["logical_batch_size"] == 512
    model = solution.build_model(fixtures)
    assert len(model.segments) == fixtures["segment_count"]
    assert model.checkpointed_segments == fixtures["checkpointed_segments"]
