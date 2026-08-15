"""Harness-side entry for CORE-DATALOADER-FANOUT-16.

Exposes the functions the harness verifier/runner expect:
  load_solution, make_fixtures, run_correctness,
  run_scientific_gates, run_performance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from benchmark.harness.api import validate_solution_api

_TASK_DIR = Path(__file__).resolve().parent


def _import_module_by_path(path: str | Path, prefix: str) -> Any:
    p = Path(path)
    name = f"{prefix}_{hashlib.sha1(str(p).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {p}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(p.parent))
        except ValueError:
            pass
    return module


_checks_mod = _import_module_by_path(
    _TASK_DIR / "hidden_verifier" / "checks.py", "spe_compile_checks"
)
check_training_correctness = _checks_mod.check_training_correctness
checksum_tensor = _checks_mod.checksum_tensor

_sci_mod = _import_module_by_path(
    _TASK_DIR / "scientific_contract.py", "spe_compile_sci"
)
run_gates = _sci_mod.run_gates


def _import_solution(path: str) -> Any:
    return _import_module_by_path(path, "spe_compile_solution")


class _TinyResMLP(torch.nn.Module):
    """Local copy of the workspace model architecture for fixture generation."""

    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        )
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.relu(self.fc1(x))
        for lin in self.blocks:
            out = torch.nn.functional.relu(lin(h))
            h = h + out
        return self.fc2(h)


def main() -> None:
    """No-op CLI entry; the harness drives this module by its functions."""
    pass


def load_solution(path: str, device: str | None = None) -> Any:
    """Import the workspace solution and verify the train_loop_v1 API."""
    module = _import_solution(path)
    violations = validate_solution_api(module, "train_loop_v1")
    if violations:
        raise RuntimeError("API violations: " + "; ".join(violations))
    return module


def make_fixtures(seed: int, device: str = "cpu") -> dict[str, Any]:
    """Build a deterministic fixture with 8 cyclic batch sizes."""
    rng = torch.Generator().manual_seed(seed)
    data_config = {"num_samples": 2048, "in_dim": 256}
    model_config = {"in_dim": data_config["in_dim"], "hidden_dim": 2048}
    batch_sizes = [16, 20, 24, 28, 32, 36, 40, 44]

    inputs = torch.randn(
        data_config["num_samples"], data_config["in_dim"], generator=rng, dtype=torch.float32
    )
    teacher = torch.nn.Linear(data_config["in_dim"], 1)
    with torch.no_grad():
        teacher.weight.normal_(0.0, 0.1, generator=rng)
        teacher.bias.normal_(0.0, 0.1, generator=rng)
        targets = teacher(inputs).squeeze(-1)

    init_model = _TinyResMLP(model_config["in_dim"], model_config["hidden_dim"], num_blocks=4)
    with torch.no_grad():
        init_model.fc1.weight.normal_(0.0, 0.01, generator=rng)
        init_model.fc1.bias.normal_(0.0, 0.01, generator=rng)
        for lin in init_model.blocks:
            lin.weight.normal_(0.0, 0.01, generator=rng)
            lin.bias.normal_(0.0, 0.01, generator=rng)
        init_model.fc2.weight.normal_(0.0, 0.01, generator=rng)
        init_model.fc2.bias.normal_(0.0, 0.01, generator=rng)
    init_state = init_model.state_dict()

    eval_inputs = torch.randn(
        128, data_config["in_dim"], generator=rng, dtype=torch.float32
    )

    return {
        "device": device,
        "data_config": data_config,
        "model_config": model_config,
        "optimizer_config": {"lr": 0.001},
        "batch_sizes": batch_sizes,
        "inputs": inputs,
        "targets": targets,
        "init_state": init_state,
        "eval_inputs": eval_inputs,
    }


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S2 correctness gate: candidate vs fp64 live-recomputed reference."""
    return check_training_correctness(
        solution,
        fixtures,
        rtol=1.0e-4,
        atol=1.0e-5,
        train_steps=5,
    )


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """S3 scientific gates."""
    return run_gates(solution, fixtures)


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 5,
    iterations: int = 24,
    device: str = "cpu",
) -> dict[str, Any]:
    """S5 performance: median step time while cycling through batch sizes."""
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )

    final_loss = None
    step_times_ms = []
    use_cuda = device.startswith("cuda")
    batch_at = getattr(solution, "_batch_at", None)

    for i in range(warmup):
        if batch_at is not None:
            batch = batch_at(fixtures, i)
        else:
            size = fixtures["batch_sizes"][i % len(fixtures["batch_sizes"])]
            offset = (i * size) % (fixtures["inputs"].shape[0] - size)
            batch = (
                fixtures["inputs"][offset : offset + size],
                fixtures["targets"][offset : offset + size],
            )
        solution.train_step(model, batch, optimizer)

    for i in range(iterations):
        if batch_at is not None:
            batch = batch_at(fixtures, i)
        else:
            size = fixtures["batch_sizes"][i % len(fixtures["batch_sizes"])]
            offset = (i * size) % (fixtures["inputs"].shape[0] - size)
            batch = (
                fixtures["inputs"][offset : offset + size],
                fixtures["targets"][offset : offset + size],
            )
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            out = solution.train_step(model, batch, optimizer)
            end_event.record()
            torch.cuda.synchronize()
            step_times_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            out = solution.train_step(model, batch, optimizer)
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        final_loss = out["loss"]

    median_step_ms = statistics.median(step_times_ms)
    return {
        "value": median_step_ms,
        "work_units": {
            "forward": iterations,
            "backward": iterations,
            "optimizer": iterations,
        },
        "output_checksums": {
            "final_loss": checksum_tensor(final_loss.unsqueeze(0)) if final_loss is not None else None,
        },
        "timing": {"step_times_ms": step_times_ms, "median_ms": median_step_ms},
    }
