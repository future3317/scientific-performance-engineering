"""Scientific gates for CORE-H2D-PIPELINE-03R2 (train_loop_v1).

Gates are pure numerical validity checks; they do not encode the performance
optimization itself.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def finite_loss_gate(solution: Any, fixtures: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Run a short training loop and assert all recorded losses are finite."""
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    dataset = TensorDataset(fixtures["inputs"], fixtures["targets"])
    loader = DataLoader(
        dataset,
        batch_size=fixtures["data_config"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    it = iter(loader)
    losses = []
    for _ in range(5):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        out = solution.train_step(model, batch, optimizer)
        losses.append(out["loss"])
    loss_tensor = torch.stack(losses)
    passed = bool(torch.isfinite(loss_tensor).all())
    return passed, {"losses": loss_tensor.tolist(), "finite": passed}


def run_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, tuple[bool, dict[str, Any]]]:
    """Return all scientific gate results for the harness."""
    return {
        "finite_loss": finite_loss_gate(solution, fixtures),
    }
