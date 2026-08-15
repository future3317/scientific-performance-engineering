#!/usr/bin/env python3
"""Public smoke test for CORE-REPEATED-BACKBONE-02.

Agents may run this to check that the workspace loads and runs without errors.
It does NOT contain hidden-verifier logic or gold outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace"))

import torch
import solution


def _make_local_fixtures(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    return {
        "seed": seed,
        "device": device,
        "batch_size": 64,
        "lr": 0.001,
        "step_dim": 32,
        "fixed_dim": 64,
        "emb_dim": 16,
        "hidden_dim": 64,
        "num_heads": 4,
        "step_data": torch.randn(512, 32),
        "fixed_data": torch.randn(128, 64),
        "fixed_data_changing": torch.randn(20, 128, 64),
        "targets": [torch.randn(512, 1) for _ in range(4)],
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixtures = _make_local_fixtures(42, device)
    model = solution.build_model(fixtures).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fixtures["lr"])
    step = fixtures["step_data"][: fixtures["batch_size"]].to(device)
    fixed = fixtures["fixed_data"].to(device)
    targets = [t[: fixtures["batch_size"]].to(device) for t in fixtures["targets"]]
    batch = {"step": step, "fixed": fixed, "targets": targets}
    out = solution.train_step(model, batch, optimizer)
    assert "loss" in out and "work_units" in out
    print("smoke_test: train_step OK", out["work_units"])

    metrics = solution.run_training(fixtures, steps=10)
    assert "final_loss" in metrics and "mean_loss" in metrics
    print("smoke_test: run_training OK final_loss=", metrics["final_loss"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
