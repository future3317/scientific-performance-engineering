from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_projection_reaches_model_checkpoint_path():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_14r2", root / "benchmark.py")
    solution = _load("solution_14r2", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["segment_count"] == 3
    assert fixtures["recompute_ratio"] == 0.2
    model = solution.build_model(fixtures)
    assert len(model.segments) == fixtures["segment_count"]
    assert model.checkpoint_policy == "checkpoint"


