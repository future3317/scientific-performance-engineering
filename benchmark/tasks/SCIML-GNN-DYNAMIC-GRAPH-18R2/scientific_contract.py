"""Scientific gates for SCIML-GNN-DYNAMIC-GRAPH-18R2.

Implemented as harness-side callable functions. They reuse the shared gate
library in ``benchmark/harness/scientific_gates.py`` wherever possible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from benchmark.harness import scientific_gates


def _import_reference(task_dir: Path):
    """Load the hidden fp64 reference module by path (no relative import)."""
    import importlib.util

    ref_path = task_dir / "hidden_verifier" / "reference.py"
    spec = importlib.util.spec_from_file_location("sciml_gnn_ragged_05_ref", str(ref_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_and_config(fixtures: dict[str, Any], ref_module: Any):
    return ref_module._fp64_state(fixtures), fixtures["config"]


def energy_force_consistency_gate(solution: Any, model: torch.nn.Module, graph: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check the workspace's F == -dE/dx for one graph."""
    graph_kwargs = {k: v for k, v in graph.items() if k != "positions"}

    def energy_and_forces(positions: torch.Tensor):
        return (
            solution.energy_fn(model, positions, **graph_kwargs),
            solution.forces_fn(model, positions, **graph_kwargs),
        )

    positions = graph["positions"]
    passed, details = scientific_gates.energy_force_consistency(energy_and_forces, positions, tol=1e-4)
    return passed, details


def gradient_agreement_gate(
    solution: Any,
    model: torch.nn.Module,
    graph: dict[str, Any],
    ref_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    ref_module: Any,
) -> tuple[bool, dict[str, Any]]:
    """Check the candidate energy gradient agrees with the fp64 live reference."""
    graph_kwargs = {k: v for k, v in graph.items() if k != "positions"}

    def candidate_fn(positions: torch.Tensor):
        return solution.energy_fn(model, positions, **graph_kwargs)

    def reference_fn(positions: torch.Tensor):
        return ref_module.graph_energy_fp64(
            ref_state, config, positions,
            graph_kwargs["x0"], graph_kwargs["edge_index"],
            graph_kwargs["cell_offsets"], graph_kwargs["cell_a"],
        )

    inputs = graph["positions"]
    return scientific_gates.gradient_agreement(candidate_fn, reference_fn, inputs, tol=1e-4)


def translation_invariance_gate(solution: Any, model: torch.nn.Module, graph: dict[str, Any], seed: int = 123) -> tuple[bool, dict[str, Any]]:
    """Energy must not change under a global translation of all atoms."""
    graph_kwargs = {k: v for k, v in graph.items() if k != "positions"}
    positions = graph["positions"]
    generator = torch.Generator(device=positions.device).manual_seed(seed)
    shift = torch.randn(3, generator=generator, device=positions.device, dtype=positions.dtype)

    energy0 = solution.energy_fn(model, positions, **graph_kwargs)
    energy1 = solution.energy_fn(model, positions + shift, **graph_kwargs)
    delta = (energy0 - energy1).abs().max().item()
    passed = bool(delta <= 1e-5)
    return passed, {"max_energy_delta": delta, "tol": 1e-5}


def run_all_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Run every scientific gate declared in task.yaml."""
    task_dir = Path(__file__).resolve().parent
    ref_module = _import_reference(task_dir)
    ref_state, config = _state_and_config(fixtures, ref_module)

    model = solution.build_model(fixtures)
    graphs = fixtures["graphs"]

    # Per-gate booleans over all graphs, with representative details from the first graph.
    first = graphs[0]
    efc_pass = all(energy_force_consistency_gate(solution, model, graph)[0] for graph in graphs)
    _, efc_details = energy_force_consistency_gate(solution, model, first)
    grad_pass = all(gradient_agreement_gate(solution, model, graph, ref_state, config, ref_module)[0] for graph in graphs)
    _, grad_details = gradient_agreement_gate(solution, model, first, ref_state, config, ref_module)
    trans_pass = all(translation_invariance_gate(solution, model, graph)[0] for graph in graphs)
    _, trans_details = translation_invariance_gate(solution, model, first)

    return {
        "energy_force_consistency": (efc_pass, efc_details),
        "gradient_agreement": (grad_pass, grad_details),
        "translation_invariance": (trans_pass, trans_details),
    }
