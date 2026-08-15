"""Scientific gates for CORE-COMPILE-TINY-12 (train_loop_v1)."""

from __future__ import annotations

from typing import Any

import torch


def finite_loss_gate(solution: Any, fixtures: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Run a short training loop and assert all recorded losses are finite."""
    model = solution.build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    losses = []
    for i in range(5):
        batch = (
            fixtures["inputs"][i * 32 : (i + 1) * 32],
            fixtures["targets"][i * 32 : (i + 1) * 32],
        )
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
