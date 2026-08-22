"""Scientific gates for SCIML-EQUIV-RECOMPUTE-06R2.

Reuses the shared gate library in ``benchmark/harness/scientific_gates.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from benchmark.harness import scientific_gates


def equivariance_rank3_gate(solution: Any, model: torch.nn.Module, graph: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check rank-3 equivariance under a random rotation of positions."""
    graph_kwargs = {k: v for k, v in graph.items() if k != "positions"}

    def pred_fn(positions: torch.Tensor):
        return solution.energy_fn(model, positions, **graph_kwargs)

    return scientific_gates.equivariance_rank3(pred_fn, graph["positions"], tol=1e-3)


def translation_invariance_gate(solution: Any, model: torch.nn.Module, graph: dict[str, Any], seed: int = 123) -> tuple[bool, dict[str, Any]]:
    """The rank-3 tensor must not change under a global translation."""
    graph_kwargs = {k: v for k, v in graph.items() if k != "positions"}
    positions = graph["positions"]
    generator = torch.Generator(device=positions.device).manual_seed(seed)
    shift = torch.randn(3, generator=generator, device=positions.device, dtype=positions.dtype)

    tensor0 = solution.energy_fn(model, positions, **graph_kwargs)
    tensor1 = solution.energy_fn(model, positions + shift, **graph_kwargs)
    delta = (tensor0 - tensor1).abs().max().item()
    passed = bool(delta <= 1e-5)
    return passed, {"max_tensor_delta": delta, "tol": 1e-5}


def run_all_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Run every scientific gate declared in task.yaml."""
    model = solution.build_model(fixtures)
    first = fixtures["trajectory"][0]

    equi_pass, equi_details = equivariance_rank3_gate(solution, model, first)
    trans_pass, trans_details = translation_invariance_gate(solution, model, first)

    return {
        "equivariance_rank3": (equi_pass, equi_details),
        "translation_invariance": (trans_pass, trans_details),
    }
