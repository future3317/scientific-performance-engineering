"""Frozen calibration protocol and task execution-class policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.formal import attest
from benchmark.harness.api import execution_class_for_task


def load_calibration_protocol(repo_root: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(repo_root) / "benchmark" / "calibration" / "calibration_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or int(protocol.get("atomic_outer_trials", 0)) < 1:
        raise ValueError(f"invalid calibration protocol: {path}")
    required_topology = {
        "omp_num_threads", "mkl_num_threads", "openblas_num_threads", "numexpr_num_threads",
        "torch_num_threads", "torch_num_interop_threads", "compiler_threads",
    }
    topology = protocol.get("thread_topology")
    if not isinstance(topology, dict) or set(topology) != required_topology:
        raise ValueError("calibration protocol must declare the complete thread_topology")
    return protocol, attest.file_digest(path)


def outer_trial_count(spec: dict[str, Any], protocol: dict[str, Any]) -> int:
    """Resolve repetition policy through the registered API execution class."""
    if execution_class_for_task(spec) == "episode":
        return int(spec.get("measurement", {}).get("repetitions", 0))
    return int(protocol["atomic_outer_trials"])

