from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_projection_uses_real_segment_checkpoint_path():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_13r2", root / "benchmark.py")
    solution = _load("solution_13r2", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["segment_count"] == 4
    assert fixtures["recompute_ratio"] == 0.2
    model = solution.build_model(fixtures)
    assert len(model.segments) == fixtures["segment_count"]
    assert model.checkpointed_segments == 1



