"""Oracle: remove the graph break and bucket all shapes into one fixed graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_blocks: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks))
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        for lin in self.blocks:
            h = h + F.relu(lin(h))
        return self.fc2(h)


def build_model(fixtures):
    config = fixtures["model_config"]
    model = TinyResMLP(config["in_dim"], config["hidden_dim"], int(config.get("num_blocks", 4)))
    model.load_state_dict(fixtures["init_state"])
    return torch.compile(model.to(fixtures["device"]))


def _batch_at(fixtures, index):
    sizes = fixtures["batch_sizes"]
    size = sizes[index % len(sizes)]
    max_size = max(sizes)
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    inputs = fixtures["inputs"][offset : offset + size]
    targets = fixtures["targets"][offset : offset + size]
    padded_inputs = torch.zeros(max_size, inputs.shape[1], dtype=inputs.dtype)
    padded_targets = torch.zeros(max_size, dtype=targets.dtype)
    mask = torch.zeros(max_size, dtype=torch.bool)
    padded_inputs[:size], padded_targets[:size], mask[:size] = inputs, targets, True
    return padded_inputs, padded_targets, mask


def train_step(model, batch, optimizer):
    inputs, targets, mask = batch
    device = next(model.parameters()).device
    preds = model(inputs.to(device)).squeeze(-1)
    diff = preds - targets.to(device)
    mask = mask.to(device)
    loss = (diff * diff * mask.to(diff.dtype)).sum() / mask.sum()
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
