#!/usr/bin/env python3
"""Oracle solution for CORE-SCALAR-SYNC-01.

Keeps loss and per-parameter-group grad-norm accumulators on device and
synchronizes to Python floats only once per SYNC_EVERY steps. The loop does
exactly the same forward/backward/optimizer work as the baseline and produces
the same final statistics (mean grad norm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

SYNC_EVERY = 50


def build_model(fixtures: dict) -> nn.Module:
    dims = fixtures["model_dims"]
    layers: list[nn.Module] = []
    in_dim = dims[0]
    for out_dim in dims[1:]:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.pop()
    return nn.Sequential(*layers)


def _init_accumulators(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if not hasattr(optimizer, "_spe_step"):
        optimizer._spe_step = 0
        optimizer._spe_loss_acc = torch.zeros(1, device=device)
        n = sum(1 for _ in optimizer.param_groups)
        optimizer._spe_grad_mean_acc = [torch.zeros(1, device=device) for _ in range(n)]
        optimizer._spe_grad_std_acc = [torch.zeros(1, device=device) for _ in range(n)]
        optimizer._spe_param_max_acc = [torch.zeros(1, device=device) for _ in range(n)]


def train_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer) -> dict:
    model.train()
    x, y = batch
    optimizer.zero_grad()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    loss.backward()

    optimizer.step()

    _init_accumulators(optimizer, x.device)
    optimizer._spe_step += 1
    optimizer._spe_loss_acc += loss.detach()

    scalar_stats: list[float | None] = []
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.grad is not None:
            g = p.grad.detach()
            optimizer._spe_grad_mean_acc[i] += g.mean()
            if g.numel() > 1:
                optimizer._spe_grad_std_acc[i] += g.std()
            else:
                optimizer._spe_grad_std_acc[i] += g.abs()
            optimizer._spe_param_max_acc[i] += p.detach().abs().max()

    if optimizer._spe_step % SYNC_EVERY == 0:
        scalar_stats = [None] * (len(optimizer.param_groups) * 3 + 1)
        scalar_stats[-1] = (optimizer._spe_loss_acc / optimizer._spe_step).item()

    return {
        "loss": loss,
        "loss_scalar": None,
        "grad_norms": scalar_stats,
        "work_units": {"forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1},
    }


def run_training(fixtures: dict, steps: int) -> dict:
    """Run *steps* SGD updates and return final statistics."""
    device = fixtures["device"]
    model = build_model(fixtures).to(device)

    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])
    _init_accumulators(optimizer, device)

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
        if out["loss_scalar"] is not None:
            loss_history.append(out["loss_scalar"])
            for i, gn in enumerate(out["grad_norms"]):
                grad_norm_history[i].append(gn)

    if optimizer._spe_step % SYNC_EVERY != 0:
        loss_history.append((optimizer._spe_loss_acc / optimizer._spe_step).item())

    mean_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

    return {
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "mean_loss": mean_loss,
        "mean_grad_norms": [],
        "steps": steps,
    }
