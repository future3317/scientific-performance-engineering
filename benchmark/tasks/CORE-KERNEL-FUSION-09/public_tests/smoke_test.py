#!/usr/bin/env python3
"""Public smoke test for CORE-KERNEL-FUSION-09.

Agent-visible sanity check: import the workspace solution, run a tiny fixture,
and assert the output shape/finiteness and approximate agreement with a plain
eager recomputation. Contains no hidden-verifier logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def eager_reference(x, residual, params):
    z = params["a1"] * x + params["b1"]
    h = z * torch.sigmoid(z)
    y = h + residual
    yc = torch.clamp(y, params["clamp_min"], params["clamp_max"])
    return params["a2"] * yc + params["b2"]


def main() -> int:
    workspace = Path(__file__).resolve().parents[1] / "workspace"
    sys.path.insert(0, str(workspace))
    try:
        import solution
    finally:
        sys.path.pop(0)

    fixtures = {
        "shape": (64, 128),
        "a1": 1.2,
        "b1": -0.3,
        "a2": 0.8,
        "b2": 0.1,
        "clamp_min": -2.0,
        "clamp_max": 2.0,
        "device": "cpu",
        "seed": 0,
    }
    torch.manual_seed(0)
    x = torch.randn(fixtures["shape"])
    residual = torch.randn(fixtures["shape"])

    ctx = solution.init(fixtures)
    out = solution.forward(ctx, x, residual)

    assert isinstance(out, torch.Tensor), "forward must return a tensor"
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "output contains non-finite values"

    ref = eager_reference(x, residual, fixtures)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5), "output disagrees with eager reference"

    print("smoke_test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
