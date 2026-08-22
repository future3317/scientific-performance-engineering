from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_vjp_density_selects_real_jacobian_rows():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_25r2", root / "benchmark.py")
    solution = _load("solution_25r2", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["jacobian_density"] == 0.5
    assert fixtures["jacobian_output_mask"].tolist() == [True, False]
    model = solution.build_model(fixtures)
    jacobian = solution.jacobian_features(model, fixtures["batch"][0])
    assert jacobian.shape[0] == int(fixtures["jacobian_output_mask"].sum())



