"""Baseline solution for CORE-H2D-PIPELINE-03R2 (train_loop_v1 API).

This baseline is CORRECT but INEFFICIENT: its DataLoader uses the declared
worker/prefetch/pinning workload, while the hot training step performs
blocking host-to-device copies plus per-step CUDA synchronization. See
README.md for the task.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_model(fixtures):
    """build_model(fixtures) -> torch.nn.Module (train_loop_v1)."""
    config = fixtures["model_config"]
    model = torch.nn.Sequential(
        torch.nn.Linear(config["in_dim"], config["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_dim"], 1),
    )
    model.load_state_dict(fixtures["init_state"])
    model.to(fixtures["device"])
    return model


def _make_dataloader(fixtures):
    """Build the training DataLoader used by run_training."""
    cfg = fixtures["data_config"]
    dataset = TensorDataset(fixtures["inputs"], fixtures["targets"])
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=fixtures["worker_count"],
        pin_memory=fixtures["pin_memory"],
        prefetch_factor=fixtures["prefetch_factor"] if fixtures["worker_count"] else None,
        drop_last=True,
    )


def train_step(model, batch, optimizer):
    """One blocking train step (forward + backward + optimizer).

    Returns {"loss": tensor, "work_units": {...}}.
    """
    device = next(model.parameters()).device
    inputs, targets = batch

    # Inefficient baseline: blocking H2D copy + explicit per-step sync.
    inputs = inputs.to(device, non_blocking=False)
    targets = targets.to(device, non_blocking=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    optimizer.zero_grad()
    preds = model(inputs).squeeze(-1)
    loss = torch.nn.functional.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "loss": loss.detach(),
        "work_units": {"forward": 1, "backward": 1, "optimizer": 1},
    }


def run_training(fixtures, steps):
    """Run *steps* train steps and return final metrics."""
    model = build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    loader = _make_dataloader(fixtures)
    it = iter(loader)
    losses = []
    work_units = {"forward": 0, "backward": 0, "optimizer": 0}
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        out = train_step(model, batch, optimizer)
        losses.append(out["loss"])
        for k in work_units:
            work_units[k] += out["work_units"][k]
    return {
        "losses": torch.stack(losses),
        "work_units": work_units,
        "final_loss": losses[-1].item(),
    }
