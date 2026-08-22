from pathlib import Path
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_vjp_projection_builds_a_real_jacobian_objective():
    root = Path(__file__).parents[1]
    benchmark = _load("benchmark_15r2", root / "benchmark.py")
    solution = _load("solution_15r2", root / "workspace" / "solution.py")
    fixtures = benchmark.make_fixtures(0)
    assert fixtures["output_count"] == 8
    assert fixtures["input_dim"] == 128
    assert fixtures["jacobian_density"] == 0.5
    assert fixtures["vjp_output_mask"].shape == (8,)
    assert int(fixtures["vjp_output_mask"].sum()) == 4
    model = solution.build_model(fixtures)
    optimizer = __import__("torch").optim.SGD(model.parameters(), lr=fixtures["lr"])
    result = solution.train_step(model, fixtures["batch"], optimizer)
    assert result["work_units"]["vjp"] == 4


