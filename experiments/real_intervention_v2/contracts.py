from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


OUTCOME_KEYS = {"outcome", "result", "latency", "speedup", "score", "label", "utility"}
SPLITS = {"development", "gate_calibration", "final_untouched"}
CAP_KEYS = {"cpu_seconds", "gpu_seconds", "wall_seconds", "storage_bytes", "download_bytes"}


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(value)
    return value


def validate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != 1 or not value.get("run_id"):
        raise ValueError("manifest requires schema_version=1 and run_id")
    if value.get("provider_calls") != 0:
        raise ValueError("real_intervention_v2 is provider-free")
    caps = value.get("resource_caps")
    if not isinstance(caps, Mapping) or set(caps) != CAP_KEYS:
        raise ValueError(f"resource_caps must contain exactly {sorted(CAP_KEYS)}")
    if any(type(caps[key]) is not int or caps[key] < 0 for key in CAP_KEYS):
        raise ValueError("resource caps must be non-negative integers")
    if not caps["wall_seconds"] or not caps["storage_bytes"]:
        raise ValueError("wall_seconds and storage_bytes must be positive")
    contexts = value.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("manifest contexts must be a non-empty list")
    ids: set[str] = set()
    lineage_split: dict[str, str] = {}

    def reject_outcomes(item: Any, path: str = "context") -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                name = str(key).lower()
                if name in OUTCOME_KEYS or name.endswith("_outcome") or name in {
                    "correctness", "latency_ms", "compile_seconds", "measured_seconds",
                    "oom", "timeout", "policy_decision", "harm", "benefit", "regret",
                }:
                    raise ValueError(f"pre-execution context contains outcome field: {path}.{key}")
                reject_outcomes(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                reject_outcomes(nested, f"{path}[{index}]")

    for row in contexts:
        if not isinstance(row, Mapping):
            raise ValueError("each context must be an object")
        reject_outcomes(row)
        context_id, lineage, split = row.get("context_id"), row.get("lineage_id"), row.get("split")
        if not all(isinstance(item, str) and item for item in (context_id, lineage)) or split not in SPLITS:
            raise ValueError("contexts require context_id, lineage_id, and development/gate_calibration/final_untouched split")
        if context_id in ids:
            raise ValueError(f"duplicate context_id: {context_id}")
        ids.add(context_id)
        if lineage in lineage_split and lineage_split[lineage] != split:
            raise ValueError(f"lineage crosses splits: {lineage}")
        lineage_split[lineage] = split
        if row.get("adapter") not in {"torchbench", "torchvision"} or not row.get("model"):
            raise ValueError(f"context {context_id} needs adapter and model")
        if type(row.get("planned_horizon")) is not int or row["planned_horizon"] < 1:
            raise ValueError(f"context {context_id} needs a positive planned_horizon")
        sessions = row.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            raise ValueError(f"context {context_id} needs a frozen session schedule")
        session_ids = set()
        for session in sessions:
            if (not isinstance(session, Mapping) or not isinstance(session.get("session_id"), str)
                    or not session["session_id"] or session.get("order") not in {"eager_first", "compile_first"}
                    or session["session_id"] in session_ids):
                raise ValueError(f"context {context_id} has an invalid or duplicate session schedule")
            session_ids.add(session["session_id"])
        if len(sessions) != 2 or {session["order"] for session in sessions} != {"eager_first", "compile_first"}:
            raise ValueError(f"context {context_id} must have one frozen eager-first and one compile-first session")
        shape = row.get("input_shape")
        if row["adapter"] == "torchvision" and (not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError(f"torchvision context {context_id} requires positive input_shape")
        shapes = row.get("input_shapes")
        if not isinstance(shapes, list) or not shapes or any(
                not isinstance(item, list) or any(type(size) is not int or size <= 0 for size in item)
                for item in shapes):
            raise ValueError(f"context {context_id} requires frozen positive input_shapes")
    if value.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if value.get("device") == "cuda" and not value.get("gpu_uuid"):
        raise ValueError("CUDA runs require one frozen GPU UUID")
    if value.get("device") == "cuda" and any(row.get("gpu_uuid") != value["gpu_uuid"] for row in contexts):
        raise ValueError("all contexts must use the frozen GPU UUID")
    if type(value.get("seed", 0)) is not int:
        raise ValueError("seed must be an integer")


def preexecution_context(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Only frozen, execution-visible covariates; never reads a receipt/outcome."""
    fields = (
        "context_id", "lineage_id", "split", "benchmark_source", "model_id", "task_mode",
        "planned_horizon", "batch_size", "input_shapes", "input_shape", "dtype", "shape_regime",
        "model_parameter_count", "device_type", "gpu_uuid", "torch_version",
        "compiler_backend", "model_source_digest", "input_manifest_digest", "declared_cold_cache",
    )
    return {key: row[key] for key in fields if key in row}
