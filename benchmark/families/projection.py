"""Audit declared FamilySpec parameters against executable task fixtures.

The audit reads workload values from ``make_fixtures(seed=0)`` and never trusts
task metadata returned by the benchmark. Each task is loaded in a fresh Python
process because task packages intentionally use common sibling module names.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from benchmark.harness import miniyaml


_WORKER = r'''
import json
import sys
from pathlib import Path

from benchmark.harness import runner, miniyaml
from benchmark.families.projection import project_fixture

task_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
device = sys.argv[3]
spec = miniyaml.load(str(task_dir / "task.yaml"))
module = runner.import_module_by_path(task_dir / "benchmark.py", module_name="projection_task")
fixtures = runner.call_benchmark_fn(module.make_fixtures, seed=seed, device=device)
print(json.dumps(project_fixture(spec.get("family_id", spec.get("family")), fixtures), sort_keys=True))
'''


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(int(item) for item in shape) if shape is not None else None


def _graph_signature(fixtures: Mapping[str, Any]) -> dict[str, Any]:
    graphs = fixtures.get("graphs")
    if not isinstance(graphs, (list, tuple)):
        return {}
    node_counts: list[int] = []
    for graph in graphs:
        positions = graph.get("positions") if isinstance(graph, Mapping) else None
        shape = _shape(positions)
        if shape:
            node_counts.append(shape[0])
    return {"graph_count": len(graphs), "graph_node_counts": node_counts}


def project_fixture(family_id: str, fixtures: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only workload values observable in the executable fixture."""
    if family_id == "compile" and isinstance(fixtures.get("compile_profile"), Mapping):
        profile = fixtures["compile_profile"]
        return {key: profile.get(key) for key in ("logical_steps", "graph_size", "dynamic_shape_rate")}
    if family_id == "h2d_pipeline":
        config = fixtures.get("data_config")
        return {
            "batch_size": config.get("batch_size") if isinstance(config, Mapping) else fixtures.get("logical_batch_size"),
            "worker_count": fixtures.get("worker_count"),
            "prefetch_factor": fixtures.get("prefetch_factor"),
            "pin_memory": fixtures.get("pin_memory"),
        }
    if family_id == "scalar_sync":
        return {}
    if family_id == "repeated_compute":
        return {
            "repeat_count": fixtures.get("num_heads"),
            "backbone_width": fixtures.get("hidden_dim", fixtures.get("width")),
            "batch_size": fixtures.get("batch_size", fixtures.get("logical_batch_size")),
        }
    if family_id == "checkpoint":
        return {
            "segment_count": fixtures.get("blocks"),
            "logical_batch_size": fixtures.get("logical_batch_size"),
            "memory_pressure": fixtures.get("memory_pressure"),
            "recompute_ratio": fixtures.get("recompute_ratio"),
        }
    if family_id == "autograd":
        data = fixtures.get("data_config")
        return {
            "input_dim": fixtures.get("input_dim") if fixtures.get("input_dim") is not None else data.get("in_dim") if isinstance(data, Mapping) else None,
            "output_count": fixtures.get("output_count"),
            "jacobian_density": fixtures.get("jacobian_density"),
        }
    if family_id == "graph_cache":
        return _graph_signature(fixtures)
    if family_id == "crystal_generation":
        target_shape = _shape(fixtures.get("target"))
        return {
            "atom_count": target_shape[0] if target_shape else None,
            "diffusion_steps": fixtures.get("num_steps"),
            "guidance_scale": fixtures.get("guidance_scale"),
        }
    if family_id == "crystal_sampling":
        initial_shape = _shape(fixtures.get("initial"))
        return {
            "sample_count": initial_shape[0] if initial_shape else None,
            "neighbor_count": None,
            "geometry_variation": None,
        }
    if family_id == "equivariant_head":
        batch = fixtures.get("batch")
        positions = batch[0] if isinstance(batch, (list, tuple)) and batch else fixtures.get("eval_positions")
        position_shape = _shape(positions)
        return {
            "node_count": position_shape[0] if position_shape else None,
            "irrep_order": fixtures.get("irrep_order"),
            "recompute_rate": fixtures.get("recompute_rate"),
        }
    if family_id == "episode":
        public_context = fixtures.get("public_context")
        workload = public_context.get("workload") if isinstance(public_context, Mapping) else None
        return dict(workload) if isinstance(workload, Mapping) else {}
    return {}


def _compare(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: dict[str, dict[str, Any]] = {}
    matches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual or actual[key] is None:
            missing.append(key)
        elif actual[key] != expected_value:
            mismatches[key] = {"declared": expected_value, "actual": actual[key]}
        else:
            matches.append(key)
    return {"status": "pass" if not missing and not mismatches else "drift", "matches": matches, "missing": missing, "mismatches": mismatches}


def audit_task(task_dir: Path, *, seed: int = 0, device: str = "cpu", timeout_s: float = 120.0) -> dict[str, Any]:
    spec = miniyaml.load(str(task_dir / "task.yaml"))
    expected = dict(spec.get("family_parameters") or {})
    family_id = str(spec.get("family_id", spec.get("family")))
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER, str(task_dir), str(seed), device],
            cwd=str(task_dir.parents[2]), text=True, capture_output=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"task_id": spec.get("task_id", task_dir.name), "family_id": family_id, "status": "fixture_timeout"}
    if completed.returncode != 0:
        return {"task_id": spec.get("task_id", task_dir.name), "family_id": family_id, "status": "fixture_error", "stderr": completed.stderr[-2000:]}
    try:
        actual = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"task_id": spec.get("task_id", task_dir.name), "family_id": family_id, "status": "fixture_error", "stderr": "worker did not return JSON"}
    return {"task_id": spec.get("task_id", task_dir.name), "family_id": family_id, "declared": expected, "actual": actual, **_compare(expected, actual)}


def audit_population(tasks_root: str | Path, *, seed: int = 0, device: str = "cpu") -> dict[str, Any]:
    root = Path(tasks_root)
    tasks = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "task.yaml").is_file()]
    records = [audit_task(path, seed=seed, device=device) for path in tasks]
    return {
        "schema_version": 1,
        "seed": seed,
        "device": device,
        "num_tasks": len(records),
        "num_pass": sum(record.get("status") == "pass" for record in records),
        "num_drift": sum(record.get("status") == "drift" for record in records),
        "num_errors": sum(record.get("status") in {"fixture_error", "fixture_timeout"} for record in records),
        "tasks": records,
    }
