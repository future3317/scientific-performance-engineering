"""Baseline for the tiny short-lived compile counterexample.

The correct policy for this workload is to remain eager: compile startup is
larger than the short schedule being measured.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_VARIANT = "CORE-COMPILE-TINY-12"


class TinyResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks))
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        for lin in self.blocks:
            h = h + F.relu(lin(h))
        return self.fc2(h)


def build_model(fixtures):
    config = fixtures["model_config"]
    model = TinyResMLP(config["in_dim"], config["hidden_dim"], int(config.get("num_blocks", 4)))
    model.load_state_dict(fixtures["init_state"])
    return model.to(fixtures["device"])


def _batch_at(fixtures, index):
    sizes = fixtures["batch_sizes"]
    size = sizes[index % len(sizes)]
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    return fixtures["inputs"][offset : offset + size], fixtures["targets"][offset : offset + size]


def train_step(model, batch, optimizer):
    inputs, targets = batch[:2]
    preds = model(inputs.to(next(model.parameters()).device)).squeeze(-1)
    loss = F.mse_loss(preds, targets.to(preds.device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.detach(), "work_units": {"forward": 1, "backward": 1, "optimizer": 1}}


def run_training(fixtures, steps):
    model = build_model(fixtures)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["optimizer_config"]["lr"])
    losses = []
    for index in range(steps):
        losses.append(train_step(model, _batch_at(fixtures, index), optimizer)["loss"])
    return {"losses": torch.stack(losses), "final_loss": losses[-1].item()}
