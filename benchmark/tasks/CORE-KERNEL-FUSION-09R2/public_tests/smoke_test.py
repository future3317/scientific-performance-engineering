from pathlib import Path
import importlib.util
import torch
import sys

root = Path(__file__).parents[1]
sys.path.insert(0, str(root.parents[2]))
spec = importlib.util.spec_from_file_location("smoke_solution", root / "workspace" / "solution.py")
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
spec = importlib.util.spec_from_file_location("smoke_benchmark", root / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)
fixtures = benchmark.make_fixtures(0)
ctx = solution.init(fixtures)
x = torch.randn(fixtures["shape"])
residual = torch.randn(fixtures["shape"])
out = solution.forward(ctx, x, residual)
assert out.shape == x.shape and torch.isfinite(out).all()
assert ctx["compile_profile"]["backend"] == "inductor"
print("smoke_test: OK")
