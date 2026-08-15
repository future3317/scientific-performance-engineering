"""Oracle (reference) solution for CORE-H2D-PIPELINE-03 (train_loop_v1 API).

Optimized pipeline: pinned-memory DataLoader with prefetch and non-blocking
host-to-device copies; no per-step synchronization.
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
    """Build a pinned, prefetched DataLoader."""
    cfg = fixtures["data_config"]
    dataset = TensorDataset(fixtures["inputs"], fixtures["targets"])
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )


def train_step(model, batch, optimizer):
    """One overlapped train step."""
    device = next(model.parameters()).device
    inputs, targets = batch

    # Pinned source tensors allow non-blocking H2D copies.
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad()
    preds = model(inputs).squeeze(-1)
    loss = torch.nn.functional.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()

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
