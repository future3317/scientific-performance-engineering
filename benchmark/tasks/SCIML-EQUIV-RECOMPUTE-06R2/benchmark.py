"""Harness-side entry for SCIML-EQUIV-RECOMPUTE-06R2.

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


def _import_hidden_reference(task_dir: Path):
    """Load hidden_verifier/reference.py by absolute path."""
    ref_path = task_dir / "hidden_verifier" / "reference.py"
    spec = importlib.util.spec_from_file_location("sciml_equiv_recompute_06_ref", str(ref_path))
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
    spec = importlib.util.spec_from_file_location("sciml_equiv_recompute_06_contract", str(contract_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution(path: str, device: str = "cpu") -> Any:
    """Import the workspace solution.py at *path*."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"solution entrypoint not found: {path}")
    name = f"sciml_equiv_recompute_06_solution_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
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


def _radius_graph(positions: torch.Tensor, r_cut: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All-pairs radius graph (no periodic boundary conditions)."""
    n = positions.shape[0]
    device = positions.device
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.linalg.vector_norm(diff, dim=-1)
    mask = (dist < r_cut) & (dist > 1e-8)
    row, col = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0).long()
    cell_offsets = torch.zeros(edge_index.shape[1], 3, dtype=torch.long, device=device)
    return edge_index, cell_offsets


def make_fixtures(seed: int = 0, device: str = "cpu") -> dict[str, Any]:
    """Generate a seeded, deterministic relaxation trajectory."""
    generator = torch.Generator(device=device).manual_seed(seed)
    config = {
        "in_feat": 4,
        "hidden": 16,
        "num_channels": 8,
        "num_rbf": 8,
        "gamma": 4.0,
        "rbf_min": 0.4,
        "rbf_max": 2.4,
        "r_cut": 2.4,
        "node_count": 64,
        "irrep_order": 2,
        "recompute_rate": 0.5,
    }
    r_cut = config["r_cut"]
    n_atoms = int(config["node_count"])
    cell_a = 6.0
    num_steps = 12

    # Initial positions, then a small fake relaxation walk.
    positions = torch.rand(n_atoms, 3, generator=generator, device=device) * cell_a
    trajectory: list[dict[str, Any]] = []
    for step in range(num_steps):
        edge_index, cell_offsets = _radius_graph(positions, r_cut)
        x0 = torch.randn(n_atoms, config["in_feat"], generator=generator, device=device)
        trajectory.append({
            "positions": positions.detach().clone(),
            "x0": x0,
            "edge_index": edge_index,
            "cell_offsets": cell_offsets,
            "cell_a": cell_a,
            "step_index": step,
        })
        # Small random displacement (finite cluster, no PBC wrapping).
        step_size = 0.08
        positions = positions + torch.randn(n_atoms, 3, generator=generator, device=device) * step_size

    torch.manual_seed(seed)
    task_dir = Path(__file__).resolve().parent
    sol_module = load_solution(str(task_dir / "workspace" / "solution.py"))
    model = sol_module.Rank3EquivariantHead(config)
    init_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}

    return {
        "config": config,
        "irrep_order": int(config["irrep_order"]),
        "recompute_rate": float(config["recompute_rate"]),
        "init_state": init_state,
        "trajectory": trajectory,
        "eval_positions": trajectory[0]["positions"],
        "tolerance": {"rtol": 1.0e-4, "atol": 1.0e-5},
    }


def _checksum_tensors(*tensors: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous()
        hasher.update(arr.numpy().tobytes())
        hasher.update(str(tuple(arr.shape)).encode())
    return hasher.hexdigest()


def run_correctness(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate per-step rank-3 tensors to the fp64 live reference."""
    task_dir = Path(__file__).resolve().parent
    ref_module = _import_hidden_reference(task_dir)
    ref_tensors = ref_module.trajectory_tensors_fp64(fixtures)

    model = solution.build_model(fixtures)
    tensors, _work = solution.eval_trajectory(model, fixtures["trajectory"])

    rtol = fixtures["tolerance"]["rtol"]
    atol = fixtures["tolerance"]["atol"]
    max_err = (tensors.detach().cpu().double() - ref_tensors).abs().max().item()
    passed = bool(torch.isfinite(tensors).all() and torch.allclose(tensors.detach().cpu().double(), ref_tensors, rtol=rtol, atol=atol))

    return {
        "passed": passed,
        "details": {
            "output_checksum": _checksum_tensors(tensors),
            "max_tensor_error": max_err,
            "rtol": rtol,
            "atol": atol,
        },
    }


def run_scientific_gates(solution: Any, fixtures: dict[str, Any]) -> dict[str, Any]:
    """Run the task-declared scientific gates."""
    task_dir = Path(__file__).resolve().parent
    contract = _import_scientific_contract(task_dir)
    return contract.run_all_gates(solution, fixtures)


def run_performance(
    solution: Any,
    fixtures: dict[str, Any],
    warmup: int = 5,
    iterations: int = 25,
    device: str = "cpu",
) -> dict[str, Any]:
    """Time eval_trajectory over *iterations* calls and return a representative step time."""
    model = solution.build_model(fixtures)
    trajectory = fixtures["trajectory"]

    for _ in range(warmup):
        solution.eval_trajectory(model, trajectory)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iterations):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        tensors, work_units = solution.eval_trajectory(model, trajectory)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    return {
        "value": median_ms,
        "work_units": work_units,
        "output_checksums": {"trajectory_tensors": _checksum_tensors(tensors)},
        "timing": {"raw_times_ms": times_ms, "median_ms": median_ms},
    }
