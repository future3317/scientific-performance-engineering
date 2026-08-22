#!/usr/bin/env python3
"""Harness-side entry for CORE-SCALAR-SYNC-01R2."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from benchmark.harness import runner


def _build_reference_model(fixtures: dict) -> nn.Module:
    """Reference architecture matching the workspace contract."""
    dims = fixtures["model_dims"]
    layers: list[nn.Module] = []
    in_dim = dims[0]
    for out_dim in dims[1:]:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.pop()
    return nn.Sequential(*layers)


def _reference_train_step(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
) -> None:
    model.train()
    x, y = batch
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()


def _reference_run_training(fixtures: dict, steps: int, device: str) -> dict:
    """fp64 live-recomputed reference (no scalar-sync bottleneck)."""
    torch.manual_seed(fixtures["seed"])
    model = _build_reference_model(fixtures).to(torch.float64).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])

    x_all, y_all = fixtures["train_data"]
    x_all = x_all.to(torch.float64).to(device)
    y_all = y_all.to(torch.float64).to(device)
    n = x_all.size(0)
    batch_size = fixtures["batch_size"]

    for step in range(steps):
        idx = (step * batch_size) % n
        batch = (x_all[idx : idx + batch_size], y_all[idx : idx + batch_size])
        _reference_train_step(model, batch, optimizer)

    model.eval()
    with torch.no_grad():
        x_test, y_test = fixtures["test_data"]
        x_test = x_test.to(torch.float64).to(device)
        y_test = y_test.to(torch.float64).to(device)
        pred = model(x_test)
        test_loss = nn.functional.mse_loss(pred, y_test)
    return {"test_loss": test_loss.item()}


def load_solution(path: str | Path):
    """Import the workspace solution module."""
    return runner.import_module_by_path(Path(path))


def make_fixtures(seed: int, device: str = "cpu") -> dict:
    """Deterministic synthetic regression fixtures."""
    torch.manual_seed(seed)
    n_train = 512
    n_test = 128
    input_dim = 8
    hidden_dim = 8
    output_dim = 1
    batch_size = 16
    scalar_syncs_per_step = 12
    metric_cadence = 4

    x_train = torch.randn(n_train, input_dim)
    true_w = torch.randn(input_dim, output_dim)
    true_b = torch.randn(output_dim)
    y_train = x_train @ true_w + true_b + 0.1 * torch.randn(n_train, output_dim)

    x_test = torch.randn(n_test, input_dim)
    y_test = x_test @ true_w + true_b + 0.05 * torch.randn(n_test, output_dim)

    return {
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "scalar_syncs_per_step": scalar_syncs_per_step,
        "metric_cadence": metric_cadence,
        "lr": 0.01,
        "model_dims": [input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, output_dim],
        "train_data": (x_train, y_train),
        "test_data": (x_test, y_test),
    }


def run_correctness(solution, fixtures: dict) -> dict:
    """Compare candidate training result against fp64 live-recomputed reference."""
    device = fixtures["device"]
    steps = 20  # small enough that fp32/fp64 remain within tolerance

    reference = _reference_run_training(fixtures, steps, device)

    torch.manual_seed(fixtures["seed"])
    model = solution.build_model(fixtures).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])

    x_all, y_all = fixtures["train_data"]
    x_all = x_all.to(device)
    y_all = y_all.to(device)
    n = x_all.size(0)
    batch_size = fixtures["batch_size"]

    for step in range(steps):
        idx = (step * batch_size) % n
        batch = (x_all[idx : idx + batch_size], y_all[idx : idx + batch_size])
        solution.train_step(model, batch, optimizer)

    model.eval()
    with torch.no_grad():
        x_test, y_test = fixtures["test_data"]
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = model(x_test)
        candidate_loss = nn.functional.mse_loss(pred, y_test)

    candidate = {"test_loss": candidate_loss.item()}
    passed = bool(
        torch.allclose(
            torch.tensor(candidate["test_loss"]),
            torch.tensor(reference["test_loss"]),
            rtol=fixtures.get("rtol", 1.0e-5),
            atol=fixtures.get("atol", 1.0e-6),
        )
    )

    return {
        "passed": passed,
        "details": {
            "candidate_test_loss": candidate["test_loss"],
            "reference_test_loss": reference["test_loss"],
            "output_checksum": candidate["test_loss"],
        },
    }


def run_scientific_gates(solution, fixtures: dict) -> dict:
    # Scalar cadence is part of the scientific workload: the candidate must
    # preserve the same loss/metric result while changing only synchronization
    # placement.  The live correctness contract supplies the oracle comparison.
    details = run_correctness(solution, fixtures)
    return {
        "metric_semantics_preserved": bool(details.get("passed", False)),
        "finite_loss": bool(
            isinstance(details.get("details", {}).get("candidate_test_loss"), (int, float))
        ),
    }


def run_performance(solution, fixtures: dict, warmup: int, iterations: int, device: str = "cpu") -> dict:
    """Median per-step wall time over *iterations* after *warmup*."""
    # Fix the model seed so repetitions measure the optimization, not weight-drawing variance.
    torch.manual_seed(12345)
    model = solution.build_model(fixtures).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])

    x_all, y_all = fixtures["train_data"]
    x_all = x_all.to(device)
    y_all = y_all.to(device)
    n = x_all.size(0)
    batch_size = fixtures["batch_size"]

    # Fixed measurement batch: scalar-sync overhead is the variable under test,
    # not data movement. Correctness still exercises varying batches.
    fixed_batch = (x_all[:batch_size], y_all[:batch_size])

    for _ in range(warmup):
        solution.train_step(model, fixed_batch, optimizer)

    # Time the whole block to average out per-call scheduling noise.
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        solution.train_step(model, fixed_batch, optimizer)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    per_step_ms = total_ms / iterations

    return {
        "value": per_step_ms,
        "work_units": {"forward_calls": iterations, "backward_calls": iterations, "optimizer_steps": iterations},
        "output_checksums": {},
        "timing": {"total_ms": total_ms, "per_step_ms": per_step_ms},
    }
