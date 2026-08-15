"""SCIML-EQUIV-RECOMPUTE-06 baseline solution (energy_force_v1 API).

A rank-3 equivariant head predicting a cartesian tensor property from edge
vectors of a small periodic graph. The fixture is a structure-relaxation
trajectory: positions change at every step.

This baseline is CORRECT but unremarkable: it recomputes the edge vectors and
the rank-3 symmetric basis every step. The tempting optimization — caching the
basis across steps — is wrong here because the inputs change.
"""

import torch


class Rank3EquivariantHead(torch.nn.Module):
    """Predict a rank-3 cartesian tensor from edge vectors.

    Each edge vector ``v`` contributes a symmetric rank-3 basis element
    ``v \otimes v \otimes v``. A learned radial coefficient per edge and
    channel accumulates ``num_channels`` such bases; a final linear mix maps
    channels to a single ``[3, 3, 3]`` tensor. The whole construction is
    equivariant under rotations of the input positions.
    """

    def __init__(self, config):
        super().__init__()
        hidden = int(config["hidden"])
        in_feat = int(config["in_feat"])
        num_rbf = int(config["num_rbf"])
        num_channels = int(config["num_channels"])
        self.gamma = float(config["gamma"])
        self.num_channels = num_channels

        self.lin0 = torch.nn.Linear(in_feat, hidden)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden + num_rbf, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, num_channels),
        )
        self.mix = torch.nn.Linear(num_channels, 1, bias=False)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(float(config["rbf_min"]), float(config["rbf_max"]), num_rbf),
        )

    def _rbf(self, dist):
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.rbf_centers) ** 2)

    def predict(self, positions, x0, edge_index, cell_offsets, cell_a, step_index=None):
        """Return the rank-3 tensor [3, 3, 3] for the given graph."""
        row, col = edge_index[0], edge_index[1]
        vec = positions[col] - positions[row] + cell_offsets.to(positions.dtype) * float(cell_a)
        dist = torch.linalg.vector_norm(vec, dim=-1)

        h = torch.tanh(self.lin0(x0))
        edge_feat = torch.cat([h[row], h[col], self._rbf(dist)], dim=-1)
        coeffs = self.edge_mlp(edge_feat)  # [E, num_channels]

        # Build num_channels symmetric rank-3 basis tensors.
        basis = torch.einsum("ec,ei,ej,ek->cijk", coeffs, vec, vec, vec)  # [C, 3, 3, 3]
        # Mix channels into a single rank-3 tensor (no bias: must stay equivariant).
        basis_perm = basis.permute(1, 2, 3, 0)  # [3, 3, 3, C]
        tensor = self.mix(basis_perm).squeeze(-1)  # [3, 3, 3]
        return tensor


def build_model(fixtures):
    """build_model(fixtures) -> torch.nn.Module (energy_force_v1 API)."""
    model = Rank3EquivariantHead(fixtures["config"])
    model.load_state_dict(fixtures["init_state"])
    device = fixtures["trajectory"][0]["positions"].device
    model.to(device)
    model.eval()
    return model


def energy_fn(model, positions, **graph):
    """energy_fn(model, positions, **graph) -> rank-3 tensor [3, 3, 3]."""
    return model.predict(
        positions,
        graph["x0"],
        graph["edge_index"],
        graph["cell_offsets"],
        graph["cell_a"],
        step_index=graph.get("step_index"),
    )


def forces_fn(model, positions, **graph):
    """forces_fn(model, positions, **graph) -> forces tensor.

    The API requires a -dE/dx-like callable. We use a scalar proxy energy
    equal to the squared norm of the predicted rank-3 tensor so the gradient
    is well-defined and inexpensive.
    """
    pos = positions.detach().clone().requires_grad_(True)
    tensor = energy_fn(model, pos, **graph)
    proxy = tensor.pow(2).sum()
    return -torch.autograd.grad(proxy, pos)[0]


def eval_trajectory(model, trajectory):
    """Evaluate the rank-3 tensor at every step of the trajectory.

    Returns (tensors [num_steps, 3, 3, 3], work_units dict).
    """
    tensors = []
    total_edges = 0
    for step, graph in enumerate(trajectory):
        # The rank-3 basis depends on the changing positions, so we recompute it
        # every step rather than caching it from a previous step.
        tensor = model.predict(
            graph["positions"],
            graph["x0"],
            graph["edge_index"],
            graph["cell_offsets"],
            graph["cell_a"],
            step_index=graph.get("step_index"),
        )
        tensors.append(tensor)
        total_edges += graph["edge_index"].shape[1]
    work_units = {"steps": len(trajectory), "edges": total_edges}
    return torch.stack(tensors), work_units
