"""Frozen calibration protocol and task execution-class policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.provenance import file_digest
from benchmark.harness.api import execution_class_for_task


def load_calibration_protocol(repo_root: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(repo_root) / "benchmark" / "calibration" / "calibration_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or int(protocol.get("atomic_outer_trials", 0)) < 1:
        raise ValueError(f"invalid calibration protocol: {path}")
    if int(protocol.get("measurement_protocol_version", 0)) != 2:
        raise ValueError("calibration protocol measurement_protocol_version must be 2")
    measurement = protocol.get("measurement_protocol")
    if not isinstance(measurement, dict):
        raise ValueError("calibration protocol must declare measurement_protocol")
    if measurement.get("noise_estimator") != "log_ratio_median_mad_v1":
        raise ValueError("calibration protocol noise estimator mismatch")
    if float(measurement.get("device_conditioning_prelude_ms", -1.0)) < 0:
        raise ValueError("calibration protocol conditioning prelude must be non-negative")
    phase_policy = measurement.get("phase_policy")
    if phase_policy != {"compiler": "cold_start_inclusive", "default": "steady_state"}:
        raise ValueError("calibration protocol phase_policy mismatch")
    required_topology = {
        "omp_num_threads", "mkl_num_threads", "openblas_num_threads", "numexpr_num_threads",
        "torch_num_threads", "torch_num_interop_threads", "compiler_threads",
    }
    topology = protocol.get("thread_topology")
    if not isinstance(topology, dict) or set(topology) != required_topology:
        raise ValueError("calibration protocol must declare the complete thread_topology")
    return protocol, file_digest(path)


def outer_trial_count(spec: dict[str, Any], protocol: dict[str, Any]) -> int:
    """Resolve repetition policy through the registered API execution class."""
    if execution_class_for_task(spec) == "episode":
        return int(spec.get("measurement", {}).get("repetitions", 0))
    return int(protocol["atomic_outer_trials"])
