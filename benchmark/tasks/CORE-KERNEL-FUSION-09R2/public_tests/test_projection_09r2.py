from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_projection_drives_real_compile_workload():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_09r2", root / "benchmark.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["compile_profile"]["backend"] == "inductor"
    assert fixtures["logical_steps"] == 192
    assert fixtures["graph_size"] == 320
    assert fixtures["dynamic_shape_rate"] == 0.2
    shapes = benchmark.workload_shapes(fixtures)
    assert len(shapes) == fixtures["logical_steps"]
    assert len(set(shapes)) > 1



