"""Hidden fp64 reference for SCIML-EQUIV-RECOMPUTE-06R2 (harness-only).

Independent live recompute of the rank-3 equivariant head in float64.
Stored goldens are not permitted; everything here recomputes from the fixtures.
"""

from __future__ import annotations

import torch


def _fp64_state(fixtures):
    return {k: v.detach().cpu().double() for k, v in fixtures["init_state"].items()}


def _predict_fp64(state, config, positions, x0, edge_index, cell_offsets, cell_a):
    """Rank-3 tensor prediction recomputed in fp64."""
    gamma = float(config["gamma"])
    centers = state["rbf_centers"]
    num_channels = int(config["num_channels"])
    irrep_order = int(config.get("irrep_order", 2))
    recompute_rate = float(config.get("recompute_rate", 1.0))

    pos = positions.detach().cpu().double()
    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    vec = pos[col] - pos[row] + cell_offsets.cpu().double() * float(cell_a)
    dist = torch.linalg.vector_norm(vec, dim=-1)

    h = torch.tanh(x0.detach().cpu().double() @ state["lin0.weight"].T + state["lin0.bias"])
    radial_basis = torch.exp(-gamma * (dist.unsqueeze(-1) - centers) ** 2)
    if irrep_order >= 2:
        radial_basis = radial_basis + recompute_rate * 1.0e-3 * torch.sin(dist.unsqueeze(-1))
    edge_feat = torch.cat([
        h[row],
        h[col],
        radial_basis,
    ], dim=-1)
    # Replicate the edge MLP manually (state dict keys mirror Sequential).
    msg = torch.tanh(edge_feat @ state["edge_mlp.0.weight"].T + state["edge_mlp.0.bias"])
    coeffs = msg @ state["edge_mlp.2.weight"].T + state["edge_mlp.2.bias"]

    basis = torch.einsum("ec,ei,ej,ek->cijk", coeffs, vec, vec, vec)
    basis_perm = basis.permute(1, 2, 3, 0)
    tensor = basis_perm @ state["mix.weight"].T  # [3, 3, 3, 1]
    return tensor.squeeze(-1)


def trajectory_tensors_fp64(fixtures):
    """Return [num_steps, 3, 3, 3] fp64 reference tensors."""
    state = _fp64_state(fixtures)
    config = fixtures["config"]
    tensors = []
    for graph in fixtures["trajectory"]:
        tensor = _predict_fp64(
            state, config,
            graph["positions"],
            graph["x0"],
            graph["edge_index"],
            graph["cell_offsets"],
            graph["cell_a"],
        )
        tensors.append(tensor)
    return torch.stack(tensors)
