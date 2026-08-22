"""Harness-side entry for SCIML-GNN-STATIC-GRAPH-CACHE-17R2.

Exposes the five functions the verifier/runner expect:
  load_solution, make_fixtures, run_correctness, run_scientific_gates, run_performance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_hidden_reference(task_dir: Path):
    """Load hidden_verifier/reference.py by absolute path."""
    ref_path = task_dir / "hidden_verifier" / "reference.py"
    spec = importlib.util.spec_from_file_location("sciml_gnn_ragged_05_ref", str(ref_path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ref_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(ref_path.parent))
        except ValueError:
            pass
    return module


def _import_scientific_contract(task_dir: Path):
    contract_path = task_dir / "scientific_contract.py"
    spec = importlib.util.spec_from_file_location("sciml_gnn_ragged_05_contract", str(contract_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Solution loading
# ---------------------------------------------------------------------------


def load_solution(path: str, device: str = "cpu") -> Any:
    """Import the workspace solution.py at *path*."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"solution entrypoint not found: {path}")
    name = f"sciml_gnn_ragged_05_solution_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, str(path_obj))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path_obj.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(path_obj.parent))
        except ValueError:
            pass
    return module


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------


def _radius_graph_pbc(positions: torch.Tensor, cell_a: float, r_cut: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All-pairs radius graph with cubic periodic boundary conditions.

    Returns (edge_index [2, E], cell_offsets [E, 3]). Offsets are integer
    lattice-image shifts so that the real displacement is
    ``positions[col] - positions[row] + offset * cell_a``.
    """
    n = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    offsets = torch.tensor(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=torch.long,
        device=device,
    )
    rows_all, cols_all, offs_all = [], [], []
    for offset in offsets:
        # displacement vectors for every ordered pair under this image
        displacement = positions.unsqueeze(0) - positions.unsqueeze(1) + offset.to(dtype).unsqueeze(0).unsqueeze(0) * cell_a
        dist = torch.linalg.vector_norm(displacement, dim=-1)
        mask = (dist < r_cut) & (dist > 1e-8)
        rows, cols = mask.nonzero(as_tuple=True)
        rows_all.append(rows)
        cols_all.append(cols)
        offs_all.append(offset.unsqueeze(0).expand(rows.shape[0], -1))
    row = torch.cat(rows_all).long()
    col = torch.cat(cols_all).long()
    edge_index = torch.stack([row, col], dim=0)
    cell_offsets = torch.cat(offs_all, dim=0).long()
    return edge_index, cell_offsets


def make_fixtures(seed: int = 0, device: str = "cpu") -> dict[str, Any]:
    """Generate a seeded, deterministic ragged batch of small crystals."""
    generator = torch.Generator(device=device).manual_seed(seed)
    graph_cache_config = {
        "geometry_displacement": 0.03,
        "skin": 0.4,
        "graph_size": 96,
        "dynamic_rate": 0.1,
    }
    config = {
        "in_feat": 4,
        "hidden": 16,
        "num_layers": 2,
        "num_rbf": 8,
        "gamma": 4.0,
        "rbf_min": 0.4,
        "rbf_max": 2.4,
        "r_cut": 2.4,
    }
    r_cut = config["r_cut"]

    # The family graph_size is the total node count across eight ragged graphs.
    graph_size = int(graph_cache_config["graph_size"])
    sizes = [graph_size // 8 + int(i < graph_size % 8) for i in range(8)]
    schedule_period = max(1, round(1.0 / max(float(graph_cache_config["dynamic_rate"]), 1.0e-6)))
    graphs: list[dict[str, Any]] = []
    for graph_index, n_atoms in enumerate(sizes):
        # Cubic cell large enough to keep atoms from piling up at the cutoff.
        cell_a = float(max(5.5, n_atoms ** (1.0 / 3.0) * 2.0))
        positions = torch.rand(n_atoms, 3, generator=generator, device=device) * cell_a
        construction_step = graph_index % schedule_period
        phase = construction_step * float(graph_cache_config["dynamic_rate"])
        positions = positions + float(graph_cache_config["geometry_displacement"]) * torch.sin(positions * 0.17 + phase)
        graph_cutoff = r_cut + float(graph_cache_config["skin"])
        edge_index, cell_offsets = _radius_graph_pbc(positions, cell_a, graph_cutoff)
        x0 = torch.randn(n_atoms, config["in_feat"], generator=generator, device=device)
        graphs.append({
            "positions": positions,
            "x0": x0,
            "edge_index": edge_index,
            "cell_offsets": cell_offsets,
            "cell_a": cell_a,
            "construction_step": construction_step,
        })

    # Deterministic model init: build on CPU, then optionally move state to device.
    torch.manual_seed(seed)
    task_dir = Path(__file__).resolve().parent
    sol_module = load_solution(str(task_dir / "workspace" / "solution.py"))
    model = sol_module.InvariantEnergyModel(config)
    init_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}

    return {
        "config": config,
        "graph_cache_config": graph_cache_config,
        "init_state": init_state,
        "graphs": graphs,
        "tolerance": {"rtol": 1.0e-4, "atol": 1.0e-5},
    }


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def _checksum_tensors(*tensors: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous()
        hasher.update(arr.numpy().tobytes())
        hasher.update(str(tuple(arr.shape)).encode())
    return hasher.hexdigest()


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Compare the candidate batch evaluation to the fp64 live reference."""
    task_dir = Path(__file__).resolve().parent
    ref_module = _import_hidden_reference(task_dir)
    ref_energies, ref_forces = ref_module.energy_forces_fp64(fixtures)

    model = solution.build_model(fixtures)
    energies, forces, _work = solution.eval_batch(model, fixtures["graphs"])

    rtol = fixtures["tolerance"]["rtol"]
    atol = fixtures["tolerance"]["atol"]
    energy_ok = torch.allclose(energies.detach().cpu(), ref_energies.float(), rtol=rtol, atol=atol)
    force_ok = torch.allclose(forces.detach().cpu(), ref_forces.float(), rtol=rtol, atol=atol)
    max_energy_err = (energies.detach().cpu() - ref_energies.float()).abs().max().item()
    max_force_err = (forces.detach().cpu() - ref_forces.float()).abs().max().item()

    passed = bool(
        torch.isfinite(energies).all()
        and torch.isfinite(forces).all()
        and energy_ok
        and force_ok
    )
    return {
        "passed": passed,
        "details": {
            "output_checksum": _checksum_tensors(energies, forces),
            "max_energy_error": max_energy_err,
            "max_force_error": max_force_err,
            "rtol": rtol,
            "atol": atol,
        },
    }


# ---------------------------------------------------------------------------
# Scientific gates
# ---------------------------------------------------------------------------


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Run the task-declared scientific gates."""
    task_dir = Path(__file__).resolve().parent
    contract = _import_scientific_contract(task_dir)
    return contract.run_all_gates(solution, fixtures)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 5,
    iterations: int = 25,
    device: str = "cpu",
) -> dict[str, Any]:
    """Time eval_batch over *iterations* calls and return a representative step time."""
    model = solution.build_model(fixtures)
    graphs = fixtures["graphs"]

    # Warmup
    for _ in range(warmup):
        solution.eval_batch(model, graphs)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iterations):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        energies, forces, work_units = solution.eval_batch(model, graphs)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]

    return {
        "value": median_ms,
        "work_units": work_units,
        "output_checksums": {"energies_forces": _checksum_tensors(energies, forces)},
        "timing": {"raw_times_ms": times_ms, "median_ms": median_ms},
    }
