"""Harness-side entry for CORE-H2D-OVERFANOUT-23R3."""

from __future__ import annotations

import hashlib
import importlib.util
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


_TASK_DIR = Path(__file__).resolve().parent


def _load(path: str | Path, prefix: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHECKS = _load(_TASK_DIR / "hidden_verifier" / "checks.py", "candidate_checks")
_SCIENCE = _load(_TASK_DIR / "scientific_contract.py", "candidate_science")
REQUIRED_API = ("build_model", "train_step", "run_training")


def load_solution(path: str | Path, device: str | None = None) -> Any:
    path = Path(path)
    if path.is_dir():
        path = path / "solution.py"
    module = _load(path, "candidate_solution")
    missing = [name for name in REQUIRED_API if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError("API violations: missing " + ", ".join(missing))
    return module


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_fixtures(seed: int, device: str = "cuda") -> dict[str, Any]:
    if not torch.cuda.is_available():
        device = "cpu"
    generator = torch.Generator().manual_seed(seed)
    data_config = {"num_samples": 32, "in_dim": 64, "batch_size": 32}
    worker_count = 5
    prefetch_factor = 4
    pin_memory = True
    inputs = torch.randn(data_config["num_samples"], data_config["in_dim"], generator=generator)
    targets = torch.randn(data_config["num_samples"], generator=generator)
    probe = _load(_TASK_DIR / "workspace" / "solution.py", "probe")
    model = probe.Model(data_config["in_dim"])
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.02, generator=generator)
    return {
        "device": device,
        "in_dim": data_config["in_dim"],
        "data_config": data_config,
        "worker_count": worker_count,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "batch": (inputs, targets),
        "logical_batch_size": data_config["batch_size"],
        "lr": 0.01,
        "init_state": model.state_dict(),
    }


def clone_fixtures(fixtures: dict[str, Any]) -> dict[str, Any]:
    """Keep one generated H2D fixture per repetition while isolating each arm."""
    cloned = dict(fixtures)
    cloned["batch"] = tuple(value.clone() for value in fixtures["batch"])
    cloned["data_config"] = dict(fixtures["data_config"])
    cloned["init_state"] = {name: value.clone() for name, value in fixtures["init_state"].items()}
    return cloned


def run_correctness(solution, fixtures):
    return _CHECKS.check_batch(solution, fixtures)


def run_scientific_gates(solution, fixtures):
    return {"batch_semantics_preserved": _SCIENCE.batch_semantics_preserved(solution, fixtures)}


def _has_pin(module):
    text = Path(module.__file__).read_text(encoding="utf-8")
    return "pin_memory" in text and "non_blocking" in text


def run_activation_evidence(solution, baseline_solution, fixtures):
    return {
        "candidate_metrics": {"pin_nonblocking_overlap": _has_pin(solution)},
        "baseline_metrics": {"pin_nonblocking_overlap": _has_pin(baseline_solution)},
    }


def _make_runtime_dataloader(fixtures: dict[str, Any]):
    """Build the tiny high-fanout loader declared by the counterexample."""
    config = fixtures["data_config"]
    return DataLoader(
        TensorDataset(fixtures["batch"][0], fixtures["batch"][1]),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=fixtures["worker_count"],
        prefetch_factor=fixtures["prefetch_factor"] if fixtures["worker_count"] else None,
        pin_memory=fixtures["pin_memory"],
        drop_last=True,
    )


def _next_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(iter(loader))


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 3,
    iterations: int = 20,
    device: str = "cpu",
) -> dict[str, Any]:
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(fixtures.get("lr", 0.01)))
    loader = _make_runtime_dataloader(fixtures)
    loader_iter = iter(loader)
    for _ in range(warmup):
        solution.train_step(model, _next_batch(loader_iter, loader), optimizer)
    _sync(str(fixtures.get("device", device)))
    times = []
    last = None
    for _ in range(iterations):
        start = time.perf_counter()
        last = solution.train_step(model, _next_batch(loader_iter, loader), optimizer)
        _sync(str(fixtures.get("device", device)))
        times.append((time.perf_counter() - start) * 1000.0)
    value = statistics.median(times)
    loss = last.get("loss") if isinstance(last, dict) else None
    checksum = None
    if isinstance(loss, torch.Tensor):
        checksum = hashlib.sha256(loss.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "value": value,
        "work_units": {"samples": int(fixtures["data_config"]["batch_size"]) * iterations, "optimizer": iterations},
        "output_checksums": {"loss": checksum},
        "timing": {"metric": "step_ms_p50", "step_times_ms": times, "median_ms": value},
    }
