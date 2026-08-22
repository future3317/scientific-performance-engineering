from pathlib import Path
import importlib.util
import torch

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("smoke_solution", root / "workspace" / "solution.py")
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
spec = importlib.util.spec_from_file_location("smoke_benchmark", root / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)
fixtures = benchmark.make_fixtures(0)
model = solution.build_model(fixtures)
optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])
out = solution.train_step(model, fixtures["batch"], optimizer)
assert torch.isfinite(out["loss"])
assert out["work_units"]["vjp"] == 4
print("smoke_test: OK")
