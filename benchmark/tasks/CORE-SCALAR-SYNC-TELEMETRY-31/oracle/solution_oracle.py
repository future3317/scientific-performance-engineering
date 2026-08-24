"""Oracle: one device-to-host transfer for the complete telemetry vector."""

from __future__ import annotations

import torch
import torch.nn as nn


METRICS_PER_EVENT = 16


def build_model(fixtures: dict) -> nn.Module:
    dims = fixtures["model_dims"]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
    layers.pop()
    return nn.Sequential(*layers)


def _metric_vector(model: nn.Module) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for layer in model:
        if not isinstance(layer, nn.Linear):
            continue
        gradient = layer.weight.grad.detach()
        values.extend(
            (
                gradient.mean(),
                gradient.std(unbiased=False),
                gradient.abs().max(),
                gradient.square().mean().sqrt(),
            )
        )
    metrics = torch.stack(values)
    if metrics.numel() != METRICS_PER_EVENT:
        raise RuntimeError(f"expected {METRICS_PER_EVENT} telemetry metrics, got {metrics.numel()}")
    return metrics


def train_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer) -> dict:
    model.train()
    x, y = batch
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    metrics = _metric_vector(model)
    # One host transfer emits the same metric vector without per-scalar reads.
    telemetry = metrics.detach().cpu().tolist()
    return {
        "loss": loss,
        "telemetry": telemetry,
        "telemetry_event_count": 1,
        "work_units": {"forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1},
    }


def run_training(fixtures: dict, steps: int) -> dict:
    model = build_model(fixtures).to(fixtures["device"])
    optimizer = torch.optim.SGD(model.parameters(), lr=float(fixtures["lr"]))
    x_all, y_all = (value.to(fixtures["device"]) for value in fixtures["train_data"])
    history: list[float] = []
    for step in range(steps):
        start = (step * fixtures["batch_size"]) % x_all.shape[0]
        result = train_step(model, (x_all[start : start + fixtures["batch_size"]], y_all[start : start + fixtures["batch_size"]]), optimizer)
        history.append(float(result["loss"].detach().cpu()))
    return {"final_loss": history[-1] if history else float("nan"), "loss_trajectory": history, "steps": steps}
