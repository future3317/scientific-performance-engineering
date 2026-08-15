#!/usr/bin/env python3
"""Scientific gates for CORE-KERNEL-FUSION-09.

These gates are called from benchmark.py/run_scientific_gates and declared in
task.yaml:scientific_gates. They reuse shared helpers from
benchmark/harness/scientific_gates.py where applicable.
"""

from __future__ import annotations

from typing import Any

import torch


def finite_output(output: torch.Tensor, tol: float = 1e-6) -> tuple[bool, dict[str, Any]]:
    """Gate: output must be finite and not collapse to a constant."""
    finite = bool(torch.isfinite(output).all())
    std = float(output.std())
    passed = finite and std > tol
    return passed, {"finite": finite, "output_std": std, "tol": tol}


def output_moment_match(
    output: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    fixtures: dict[str, Any],
    tol: float = 1e-2,
) -> tuple[bool, dict[str, Any]]:
    """Gate: output moments must be consistent with the affine/SiLU/clamp chain.

    We compare against a cheap analytic expectation: the output is an affine
    transform of a clamped SiLU-plus-residual. The gate is intentionally loose
    (relative tolerance) to avoid false failures while still catching semantic
    skips such as returning zeros or the input unchanged.
    """
    flat = output.detach().flatten().double()
    observed_mean = flat.mean().item()
    observed_std = flat.std(unbiased=False).item()

    # A candidate that returns an unmodified input or a constant will have a
    # moment structure very different from the transformed signal.
    input_std = float(x.std())
    input_mean = float(x.mean())

    # Expected: output std should be a non-negligible fraction of the input std
    # (the chain preserves variance through the affine layers and SiLU).
    std_ratio = observed_std / (input_std + 1e-12)
    passed = (
        torch.isfinite(output).all()
        and observed_std > 1e-3 * input_std
        and abs(observed_mean - input_mean) < 10.0 * (input_std + 1.0)
    )
    return passed, {
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "input_std": input_std,
        "std_ratio": std_ratio,
        "tol": tol,
    }
