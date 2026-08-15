#!/usr/bin/env python3
"""Baseline solution for CORE-REPEATED-BACKBONE-02.

A multi-head regression model with a shared backbone. The injected bottleneck
is repeated computation: the baseline recomputes the embedding projection of a
fixed input batch for every head and recomputes the backbone output separately
for every head each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadModel(nn.Module):
    def __init__(self, step_dim: int, fixed_dim: int, emb_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed = nn.Sequential(
            nn.Linear(fixed_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(step_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])

    def project_fixed(self, fixed_input: torch.Tensor) -> torch.Tensor:
        """Project a fixed batch to a single context vector."""
        e = self.embed(fixed_input)  # [B_fixed, emb_dim]
        return e.mean(dim=0)  # [emb_dim]

    def forward_head(self, step_input: torch.Tensor, context: torch.Tensor, head_idx: int) -> torch.Tensor:
        ctx = context.unsqueeze(0).expand(step_input.size(0), -1)
        x = torch.cat([step_input, ctx], dim=-1)
        h = self.backbone(x)
        return self.heads[head_idx](h)


def build_model(fixtures: dict) -> nn.Module:
    return MultiHeadModel(
        fixtures["step_dim"],
        fixtures["fixed_dim"],
        fixtures["emb_dim"],
        fixtures["hidden_dim"],
        fixtures["num_heads"],
    )


def train_step(model: nn.Module, batch: dict, optimizer: torch.optim.Optimizer) -> dict:
    model.train()
    step_input = batch["step"]
    fixed_input = batch["fixed"]
    targets = batch["targets"]

    total_loss = torch.zeros(1, device=step_input.device)
    for head_idx in range(model.num_heads):
        context = model.project_fixed(fixed_input)
        pred = model.forward_head(step_input, context, head_idx)
        total_loss += nn.functional.mse_loss(pred, targets[head_idx])

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss,
        "work_units": {"forward_calls": model.num_heads, "backward_calls": 1, "optimizer_steps": 1},
    }


def run_training(fixtures: dict, steps: int) -> dict:
    device = fixtures["device"]
    model = build_model(fixtures).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=fixtures["lr"])

    step_data = fixtures["step_data"].to(device)
    fixed_data = fixtures["fixed_data"].to(device)
    targets = [t.to(device) for t in fixtures["targets"]]
    n = step_data.size(0)
    batch_size = fixtures["batch_size"]
    regime = fixtures.get("regime", "fixed")

    losses: list[float] = []
    for step in range(steps):
        sidx = (step * batch_size) % n
        step_batch = step_data[sidx : sidx + batch_size]
        step_targets = [t[sidx : sidx + batch_size] for t in targets]
        if regime == "fixed":
            fixed = fixed_data
        else:
            idx = step % fixtures["fixed_data_changing"].size(0)
            fixed = fixtures["fixed_data_changing"][idx].to(device)
        batch = {"step": step_batch, "fixed": fixed, "targets": step_targets}
        out = train_step(model, batch, optimizer)
        losses.append(out["loss"].item())

    return {"final_loss": losses[-1], "mean_loss": sum(losses) / len(losses), "steps": steps}
