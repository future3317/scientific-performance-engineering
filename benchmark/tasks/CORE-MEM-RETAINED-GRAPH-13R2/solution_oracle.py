"""Oracle (reference) solution for CORE-MEM-RETAINED-GRAPH-13R2 (train_loop_v1 API).

Optimized loop: pad every batch to the maximum shape, remove the graph-breaking
.item() branch inside the model forward, and compile a single fixed-shape graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyResMLP(nn.Module):
    """Small residual MLP with a pure-tensor forward (no graph break)."""

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
            h = h + out
        return self.fc2(h)


def build_model(fixtures):
    """build_model(fixtures) -> torch.nn.Module (train_loop_v1)."""
    config = fixtures["model_config"]
    model = TinyResMLP(config["in_dim"], config["hidden_dim"], num_blocks=4)
    model.load_state_dict(fixtures["init_state"])
    model.to(fixtures["device"])
    # One fixed max batch size is used in train_step, so a single graph suffices.
    model = torch.compile(model)
    return model


def _batch_at(fixtures, index):
    """Return a batch padded to the maximum size so shapes are constant."""
    sizes = fixtures["batch_sizes"]
    size = sizes[index % len(sizes)]
    max_size = max(sizes)
    offset = (index * size) % (fixtures["inputs"].shape[0] - size)
    inputs = fixtures["inputs"][offset : offset + size]
    targets = fixtures["targets"][offset : offset + size]

    padded_inputs = torch.zeros(max_size, inputs.shape[1], dtype=inputs.dtype)
    padded_targets = torch.zeros(max_size, dtype=targets.dtype)
    mask = torch.zeros(max_size, dtype=torch.bool)
    padded_inputs[:size] = inputs
    padded_targets[:size] = targets
    mask[:size] = True
    return padded_inputs, padded_targets, mask


def train_step(model, batch, optimizer):
    """One compiled train step with no graph break and constant-shape input."""
    if len(batch) == 3:
        inputs, targets, mask = batch
    else:
        inputs, targets = batch
        mask = torch.ones(inputs.shape[0], dtype=torch.bool)
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    preds = model(inputs).squeeze(-1)
    # Ignore padded dummy samples using only fixed-shape ops.
    diff = preds - targets
    mask_f = mask.to(diff.dtype)
    loss = (diff * diff * mask_f).sum() / mask.sum()

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


