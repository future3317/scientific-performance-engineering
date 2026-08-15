#!/usr/bin/env python3
"""Baseline solution for CORE-SCALAR-SYNC-01.

A tiny MLP regression loop. The injected bottleneck is per-step scalar
synchronization: every train_step calls .item() on the loss AND on every
per-parameter-group gradient norm, appends the Python floats to lists, and
computes running Python-float statistics. The loop semantics and final metrics
are correct but the loop pays many small host-read overheads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_model(fixtures: dict) -> nn.Module:
    """Build the MLP described by fixtures['model_dims']."""
    dims = fixtures["model_dims"]
    layers: list[nn.Module] = []
    in_dim = dims[0]
    for out_dim in dims[1:]:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.pop()  # remove trailing ReLU
    return nn.Sequential(*layers)


def train_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer) -> dict:
    """One training step with per-step scalar synchronization."""
    model.train()
    x, y = batch
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()

    optimizer.step()

    # Heavy scalar synchronization: pull many per-parameter statistics to host every step.
    scalar_stats: list[float] = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            scalar_stats.append(p.grad.detach().mean().item())
            if p.grad.numel() > 1:
                scalar_stats.append(p.grad.detach().std().item())
            else:
                scalar_stats.append(0.0)
            scalar_stats.append(p.detach().abs().max().item())
    # Also synchronize the loss scalar.
    scalar_stats.append(loss.detach().item())

    return {
        "loss": loss,
        "loss_scalar": scalar_stats[-1],
        "grad_norms": scalar_stats,
        "work_units": {"forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1},
    }


def run_training(fixtures: dict, steps: int) -> dict:
    """Run *steps* SGD updates and return final statistics."""
    device = fixtures["device"]
    model = build_model(fixtures).to(device)

    # One parameter group per tensor to maximize per-step scalar-sync calls.
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])

    x_all, y_all = fixtures["train_data"]
    x_all = x_all.to(device)
    y_all = y_all.to(device)
    n = x_all.size(0)
    batch_size = fixtures["batch_size"]

    loss_history: list[float] = []
    grad_norm_history: list[list[float]] = [[] for _ in optimizer.param_groups]

    for step in range(steps):
        idx = (step * batch_size) % n
        batch = (x_all[idx : idx + batch_size], y_all[idx : idx + batch_size])
        out = train_step(model, batch, optimizer)
        loss_history.append(out["loss_scalar"])
        for i, gn in enumerate(out["grad_norms"]):
            grad_norm_history[i].append(gn)

    mean_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
    mean_grad_norms = [sum(g) / len(g) if g else 0.0 for g in grad_norm_history]

    return {
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "mean_loss": mean_loss,
        "mean_grad_norms": mean_grad_norms,
        "steps": steps,
    }
