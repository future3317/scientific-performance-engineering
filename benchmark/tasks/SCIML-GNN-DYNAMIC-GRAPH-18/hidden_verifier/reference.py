"""Hidden fp64 reference for SCIML-GNN-DYNAMIC-GRAPH-18 (harness-only, never sandboxed).

Independent live recompute of the energy/force math in float64. Stored goldens
are not permitted (BENCHMARK_DESIGN.md section 4.1); everything here recomputes
from the fixtures on every call.
"""

from __future__ import annotations

import torch


def _fp64_state(fixtures):
    return {k: v.detach().cpu().double() for k, v in fixtures["init_state"].items()}


def _graph_energy_fp64(state, config, positions, x0, edge_index, cell_offsets, cell_a):
    gamma = float(config["gamma"])
    centers = state["rbf_centers"]
    num_layers = int(config["num_layers"])

    pos = positions.double()
    h = torch.tanh(x0.detach().cpu().double() @ state["lin0.weight"].T + state["lin0.bias"])
    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    vec = pos[col] - pos[row] + cell_offsets.cpu().double() * float(cell_a)
    dist = torch.linalg.vector_norm(vec, dim=-1)
    rbf = torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)
    for layer in range(num_layers):
        msg = torch.cat([h[row], h[col], rbf], dim=-1)
        msg = torch.tanh(msg @ state[f"msg1.{layer}.weight"].T + state[f"msg1.{layer}.bias"])
        msg = msg @ state[f"msg2.{layer}.weight"].T + state[f"msg2.{layer}.bias"]
        agg = torch.zeros_like(h).index_add_(0, row, msg)
        h = h + torch.tanh(agg @ state[f"res.{layer}.weight"].T + state[f"res.{layer}.bias"])
    return (h @ state["out.weight"].T + state["out.bias"]).sum()


def energy_forces_fp64(fixtures):
    """Return (energies [G], forces [N_total, 3]) in float64."""
    state = _fp64_state(fixtures)
    config = fixtures["config"]
    energies = []
    forces = []
    for graph in fixtures["graphs"]:
        pos = graph["positions"].detach().cpu().double().requires_grad_(True)
        energy = _graph_energy_fp64(
            state, config, pos, graph["x0"], graph["edge_index"],
            graph["cell_offsets"], graph["cell_a"],
        )
        force = -torch.autograd.grad(energy, pos)[0]
        energies.append(energy.detach())
        forces.append(force)
    return torch.stack(energies), torch.cat(forces, dim=0)


def total_energy_fn_fp64(fixtures):
    """Closure mapping concatenated positions [N_total, 3] -> total energy (fp64)."""
    state = _fp64_state(fixtures)
    config = fixtures["config"]
    graphs = fixtures["graphs"]

    def fn(pos_cat):
        pos64 = pos_cat.double()
        energies = []
        start = 0
        for graph in graphs:
            n = graph["positions"].shape[0]
            p = pos64[start : start + n]
            energies.append(
                _graph_energy_fp64(
                    state, config, p, graph["x0"], graph["edge_index"],
                    graph["cell_offsets"], graph["cell_a"],
                )
            )
            start += n
        return torch.stack(energies).sum()

    return fn


def graph_energy_fp64(state, config, positions, x0, edge_index, cell_offsets, cell_a):
    """Single-graph fp64 energy, accepting any device and returning energy on *positions* device.

    *positions* must stay differentiable; it is moved to CPU for the fp64
    reference computation and the result is moved back so gradients flow.
    """
    target_device = positions.device
    energy = _graph_energy_fp64(
        state, config,
        positions.cpu().double(),
        x0.detach().cpu().double(),
        edge_index,
        cell_offsets,
        cell_a,
    )
    return energy.to(target_device)
