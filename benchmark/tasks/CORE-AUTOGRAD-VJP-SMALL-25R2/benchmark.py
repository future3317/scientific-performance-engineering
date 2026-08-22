from __future__ import annotations
import hashlib
import importlib.util
import statistics
import time
from pathlib import Path
import torch

_TASK_DIR = Path(__file__).resolve().parent


def _load(path, prefix):
    spec = importlib.util.spec_from_file_location(f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


_checks = _load(_TASK_DIR / "hidden_verifier" / "checks.py", "vjp25_checks")
_science = _load(_TASK_DIR / "scientific_contract.py", "vjp25_science")


def load_solution(path, device=None):
    path = Path(path); module = _load(path / "solution.py" if path.is_dir() else path, "vjp25_solution")
    missing = [name for name in ("build_model", "jacobian_features", "train_step", "run_training") if not callable(getattr(module, name, None))]
    if missing: raise RuntimeError("API violations: missing " + ", ".join(missing))
    return module


def make_fixtures(seed, device="cpu"):
    generator = torch.Generator().manual_seed(seed); input_dim, output_count, density = 64, 2, 0.5
    mask = torch.tensor([True, False]); probe = _load(_TASK_DIR / "workspace" / "solution.py", "vjp25_probe"); model = probe.Model(input_dim, output_count, mask)
    for parameter in model.parameters(): parameter.data.normal_(0, 0.02, generator=generator)
    return {"device": device, "input_dim": input_dim, "output_count": output_count, "jacobian_density": density, "jacobian_output_mask": mask, "batch": (torch.randn(8, input_dim, generator=generator),), "logical_batch_size": 8, "lr": 0.001, "init_state": model.state_dict()}


def run_correctness(solution, fixtures): return _checks.check_vjp(solution, fixtures)
def run_scientific_gates(solution, fixtures): return {"gradient_equivalence": _science.gradient_equivalence(solution, fixtures)}


def run_performance(solution, fixtures, warmup=2, iterations=12, device="cpu"):
    model = solution.build_model(fixtures); optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])
    for _ in range(warmup): solution.train_step(model, fixtures["batch"], optimizer)
    times = []; last = None
    for _ in range(iterations):
        start = time.perf_counter(); last = solution.train_step(model, fixtures["batch"], optimizer); times.append((time.perf_counter() - start) * 1000.0)
    value = statistics.median(times)
    return {"value": value, "work_units": {"samples": fixtures["logical_batch_size"] * iterations, "vjp": int(fixtures["jacobian_output_mask"].sum()) * iterations, "optimizer": iterations}, "output_checksums": {"loss": _checks.checksum_tensor(last["loss"].reshape(1))}, "timing": {"metric": "step_ms_p50", "step_times_ms": times, "median_ms": value}}



