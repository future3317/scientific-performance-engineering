#!/usr/bin/env python3
"""Public smoke test for CORE-SCALAR-SYNC-01R2.

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
        "batch_size": 16,
        "lr": 0.01,
        "model_dims": [8, 8, 8, 8, 8, 1],
        "train_data": (torch.randn(512, 8), torch.randn(512, 1)),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixtures = _make_local_fixtures(42, device)

    model = solution.build_model(fixtures).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{"params": [p], "lr": fixtures["lr"]} for p in params])
    x, y = fixtures["train_data"]
    x = x[: fixtures["batch_size"]].to(device)
    y = y[: fixtures["batch_size"]].to(device)
    out = solution.train_step(model, (x, y), optimizer)
    assert "loss" in out and "work_units" in out
    print("smoke_test: train_step OK", out["work_units"])

    metrics = solution.run_training(fixtures, steps=10)
    assert "final_loss" in metrics and "mean_loss" in metrics
    print("smoke_test: run_training OK final_loss=", metrics["final_loss"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
