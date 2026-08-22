"""SCIML-GNN-RAGGED-05R2 baseline solution (energy_force_v1 API).

Evaluates per-graph energies and forces (F = -dE/dx) for a batch of small
periodic crystals with an invariant message-passing model (scatter via
index_add). Plain torch only — no PyG.

This baseline is CORRECT but INEFFICIENT: see README.md.
"""

import torch


class InvariantEnergyModel(torch.nn.Module):
    """Small invariant message-passing energy model.

    Messages depend on interatomic *distances* only (via an RBF expansion), so
    the per-graph energy is invariant to global translations and rotations.
    """

    def __init__(self, config):
        super().__init__()
        hidden = int(config["hidden"])
        in_feat = int(config["in_feat"])
        num_rbf = int(config["num_rbf"])
        self.num_layers = int(config["num_layers"])
        self.gamma = float(config["gamma"])
        self.lin0 = torch.nn.Linear(in_feat, hidden)
        self.msg1 = torch.nn.ModuleList(
            torch.nn.Linear(2 * hidden + num_rbf, hidden) for _ in range(self.num_layers)
        )
        self.msg2 = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in range(self.num_layers))
        self.res = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in range(self.num_layers))
        self.out = torch.nn.Linear(hidden, 1)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(float(config["rbf_min"]), float(config["rbf_max"]), num_rbf),
        )

    def _rbf(self, dist):
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.rbf_centers) ** 2)

    def node_features(self, x0):
        return torch.tanh(self.lin0(x0))

    def propagate(self, h, dist, row, col):
        for layer in range(self.num_layers):
            msg = torch.cat([h[row], h[col], self._rbf(dist)], dim=-1)
            msg = self.msg2[layer](torch.tanh(self.msg1[layer](msg)))
            agg = torch.zeros_like(h).index_add_(0, row, msg)
            h = h + torch.tanh(self.res[layer](agg))
        return h

    def graph_energy(self, positions, x0, edge_index, cell_offsets, cell_a):
        """Energy of a single graph. Differentiable w.r.t. positions."""
        row, col = edge_index[0], edge_index[1]
        vec = positions[col] - positions[row] + cell_offsets.to(positions.dtype) * float(cell_a)
        dist = torch.linalg.vector_norm(vec, dim=-1)
        h = self.propagate(self.node_features(x0), dist, row, col)
        return self.out(h).sum()


def build_model(fixtures):
    """build_model(fixtures) -> torch.nn.Module (energy_force_v1)."""
    model = InvariantEnergyModel(fixtures["config"])
    model.load_state_dict(fixtures["init_state"])
    device = fixtures["graphs"][0]["positions"].device
    model.to(device)
    model.eval()
    return model


def energy_fn(model, positions, **graph):
    """energy_fn(model, positions, **graph) -> energy tensor.

    `graph` carries x0, edge_index, cell_offsets, cell_a for ONE graph.
    `positions` must stay differentiable (autograd w.r.t. positions).
    """
    return model.graph_energy(
        positions, graph["x0"], graph["edge_index"], graph["cell_offsets"], graph["cell_a"]
    )


def forces_fn(model, positions, **graph):
    """forces_fn(model, positions, **graph) -> forces equal to -dE/dx."""
    pos = positions.detach().clone().requires_grad_(True)
    energy = energy_fn(model, pos, **graph)
    return -torch.autograd.grad(energy, pos)[0]


def eval_batch(model, graphs):
    """Evaluate energies and forces for a list of graphs.

    Returns (energies [G], forces [N_total, 3], work_units dict).
    """
    energies = []
    forces = []
    atoms = 0
    edges = 0
    for graph in graphs:
        pos = graph["positions"].detach().clone().requires_grad_(True)
        energy = model.graph_energy(
            pos, graph["x0"], graph["edge_index"], graph["cell_offsets"], graph["cell_a"]
        )
        force = -torch.autograd.grad(energy, pos)[0]
        energies.append(energy.detach())
        forces.append(force)
        atoms += pos.shape[0]
        edges += graph["edge_index"].shape[1]
    work_units = {"graphs": len(graphs), "atoms": atoms, "edges": edges}
    return torch.stack(energies), torch.cat(forces, dim=0), work_units
