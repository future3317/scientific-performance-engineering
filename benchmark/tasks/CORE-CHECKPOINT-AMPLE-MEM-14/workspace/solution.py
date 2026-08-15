"""Baseline solution for CORE-CHECKPOINT-AMPLE-MEM-14 (train_loop_v1 API).

This baseline is CORRECT but INEFFICIENT: the model is wrapped in torch.compile,
but its forward method contains an `.item()`-dependent Python control-flow graph
break inside a residual block. The measurement loop cycles through 8 different
batch sizes, each forcing a dynamo recompilation. See README.md for the task.
"""

from __future__ import annotations

TASK_VARIANT = "CORE-CHECKPOINT-AMPLE-MEM-14"

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyResMLP(nn.Module):
    """Small residual MLP with a graph-breaking no-op branch in forward."""

    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        for lin in self.blocks:
            out = F.relu(lin(h))
            # Graph break + host synchronization: the branch is a no-op, but
            # dynamo cannot compile across the `.item()` call.
            if out.mean().item() > 1e9:
                out = out * 1.0
            h = h + out
        return self.fc2(h)


def build_model(fixtures):
    """build_model(fixtures) -> torch.nn.Module (train_loop_v1)."""
    config = fixtures["model_config"]
    model = TinyResMLP(config["in_dim"], config["hidden_dim"], num_blocks=4)
    model.load_state_dict(fixtures["init_state"])
    model.to(fixtures["device"])
    # Compiled with dynamic shapes disabled: every new batch size recompiles.
    model = torch.compile(model, dynamic=False)
    return model


def _batch_at(fixtures, index):
    """Return the CPU batch for the given cyclic index (variable shape)."""
    sizes = fixtures["batch_sizes"]
    size = sizes[index % len(sizes)]
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    return (
        fixtures["inputs"][offset : offset + size],
        fixtures["targets"][offset : offset + size],
    )


def train_step(model, batch, optimizer):
    """One train step with a graph-breaking model forward."""
    if len(batch) == 3:
        inputs, targets, mask = batch
    else:
        inputs, targets = batch
        mask = None

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    preds = model(inputs).squeeze(-1)
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
    loss = F.mse_loss(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "work_units": {"forward": 1, "backward": 1, "optimizer": 1},
    }


def run_training(fixtures, steps):
    """Run *steps* train steps, cycling through the fixture's batch sizes."""
    model = build_model(fixtures)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=fixtures["optimizer_config"]["lr"]
    )
    losses = []
    work_units = {"forward": 0, "backward": 0, "optimizer": 0}
    for i in range(steps):
        batch = _batch_at(fixtures, i)
        out = train_step(model, batch, optimizer)
        losses.append(out["loss"])
        for k in work_units:
            work_units[k] += out["work_units"][k]
    return {
        "losses": torch.stack(losses),
        "work_units": work_units,
        "final_loss": losses[-1].item(),
    }
