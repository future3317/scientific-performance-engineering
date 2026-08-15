"""Baseline solution: launch-fragmented pointwise chain (kernel_module_v1 API).

Computes, over a moderate fp32 tensor:

    z   = a1 * x + b1            (affine scale/shift)
    h   = SiLU(z) = z * sigmoid(z)
    y   = h + residual           (residual add)
    yc  = clamp(y, clamp_min, clamp_max)
    out = a2 * yc + b2           (second affine)

This baseline executes the chain as a sequence of separate eager torch
operations. It is CORRECT but not fast. See README.md for the task.
"""

from __future__ import annotations

import torch


def init(fixtures):
    """Build the opaque context from the fixtures (kernel_module_v1 API)."""
    return {
        "a1": float(fixtures["a1"]),
        "b1": float(fixtures["b1"]),
        "a2": float(fixtures["a2"]),
        "b2": float(fixtures["b2"]),
        "clamp_min": float(fixtures["clamp_min"]),
        "clamp_max": float(fixtures["clamp_max"]),
    }


def forward(context, x, residual):
    """Run the pointwise chain; returns the output tensor."""
    a1 = context["a1"]
    b1 = context["b1"]
    a2 = context["a2"]
    b2 = context["b2"]
    clamp_min = context["clamp_min"]
    clamp_max = context["clamp_max"]

    t1 = x * a1                              # kernel 1: scale
    t2 = t1 + b1                             # kernel 2: shift
    s = torch.sigmoid(t2)                    # kernel 3: sigmoid
    t3 = t2 * s                              # kernel 4: SiLU = z * sigmoid(z)
    t4 = t3 + residual                       # kernel 5: residual add
    t5 = torch.clamp(t4, clamp_min, clamp_max)  # kernel 6: clamp
    t6 = t5 * a2                             # kernel 7: second scale
    t7 = t6 + b2                             # kernel 8: second shift
    return t7
