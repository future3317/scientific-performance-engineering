#!/usr/bin/env python3
"""Validate the structural and evidence contracts of a benchmark record."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "assets" / "benchmark_record.schema.json"
EVIDENCE_LEVELS = ("static", "micro", "module", "logical_update", "amortized_job", "time_to_quality")
EVIDENCE_RANK = {name: index for index, name in enumerate(EVIDENCE_LEVELS)}
FEATURE_KEYS = {
    "data_loader", "h2d", "backward", "higher_order_autograd", "optimizer", "cache",
    "compiler", "custom_op", "distributed", "checkpoint", "auxiliary_tasks", "sampling",
}
LOGICAL_UPDATE_STAGES = {
    "fetch", "cpu_preprocess", "h2d", "gpu_preprocess", "forward", "loss",
    "autograd_aux", "backward", "grad_transform", "clipping", "communication",
    "optimizer", "scheduler", "ema_swa", "metrics", "checkpoint", "validation",
}
LIFECYCLE_STAGES = {
    "startup", "precompute", "logical_update", "evaluation_sampling",
    "checkpoint_resume", "teardown", "failure_retry",
}


def get_path(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def require_object(record: dict[str, Any], path: str, errors: list[str]) -> dict[str, Any] | None:
    value = get_path(record, path)
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    return value


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != schema["properties"]["schema_version"]["const"]:
        errors.append("schema_version must be 4")

    comparison_class = record.get("comparison_class")
    if comparison_class not in {"systems", "scaling", "algorithmic"}:
        errors.append("comparison_class must be systems, scaling, or algorithmic")
    evidence_level = record.get("evidence_level")
    if evidence_level not in EVIDENCE_RANK:
        errors.append("evidence_level must be static, micro, module, logical_update, amortized_job, or time_to_quality")
    features = record.get("features")
    if not isinstance(features, dict) or set(features) != FEATURE_KEYS or any(not isinstance(features[key], bool) for key in FEATURE_KEYS):
        errors.append("features must declare every supported feature as a boolean")
        features = {}

    level_rank = EVIDENCE_RANK.get(evidence_level, 0)
    requires_logical_update = level_rank >= EVIDENCE_RANK["logical_update"]
    requires_lifecycle = evidence_level in {"amortized_job", "time_to_quality"}
    requires_preflight = requires_lifecycle or features.get("compiler", False) or features.get("distributed", False) or features.get("checkpoint", False)

    for path in schema["required"]:
        if path not in record:
            errors.append(f"missing top-level field {path}")
    for section, contract in schema["properties"].items():
        if section not in record or not isinstance(contract, dict):
            continue
        if contract.get("type") != "object" and "required" not in contract:
            continue
        required = contract.get("required", [])
        if not isinstance(required, list):
            continue
        value = record[section]
        if not isinstance(value, dict):
            errors.append(f"{section} must be an object")
            continue
        for key in required:
            if key not in value:
                errors.append(f"missing field {section}.{key}")

    preflight_value = get_path(record, "preflight")
    if requires_preflight or preflight_value is not None:
        preflight = require_object(record, "preflight", errors)
    else:
        preflight = None
    if preflight is not None:
        if preflight.get("compatibility_status") not in {"pass", "fail", "inconclusive"}:
            errors.append("preflight.compatibility_status must be pass, fail, or inconclusive")
        topology = preflight.get("runtime_topology")
        if not isinstance(topology, dict):
            errors.append("preflight.runtime_topology must be an object")
        unsupported = preflight.get("unsupported_combinations")
        if not isinstance(unsupported, list):
            errors.append("preflight.unsupported_combinations must be a list")

    work = require_object(record, "work", errors)
    if work is not None:
        dag = work.get("logical_update_dag")
        if requires_logical_update or dag is not None:
            if not isinstance(dag, list) or not LOGICAL_UPDATE_STAGES.issubset({item.get("stage") for item in dag if isinstance(item, dict)}):
                errors.append("work.logical_update_dag must enumerate the required logical-update stages for logical-update evidence")

        lifecycle = work.get("campaign_lifecycle")
        if requires_lifecycle or lifecycle is not None:
            if not isinstance(lifecycle, dict) or not LIFECYCLE_STAGES.issubset(lifecycle):
                errors.append("work.campaign_lifecycle must declare all stages for amortized-job evidence")
            else:
                lifecycle_stages = LIFECYCLE_STAGES
                for stage in lifecycle_stages:
                    entry = lifecycle[stage]
                    if not isinstance(entry, dict) or not {"included", "seconds", "evidence"}.issubset(entry):
                        errors.append(f"work.campaign_lifecycle.{stage} must include included, seconds, and evidence")
                    elif not isinstance(entry["included"], bool):
                        errors.append(f"work.campaign_lifecycle.{stage}.included must be boolean")
                    elif not isinstance(entry["evidence"], str):
                        errors.append(f"work.campaign_lifecycle.{stage}.evidence must be a string")
                    elif entry["seconds"] is not None and (
                        isinstance(entry["seconds"], bool)
                        or not isinstance(entry["seconds"], (int, float))
                        or not math.isfinite(float(entry["seconds"]))
                        or entry["seconds"] < 0
                    ):
                        errors.append(f"work.campaign_lifecycle.{stage}.seconds must be null or a non-negative finite number")

        census = work.get("sync_census")
        if requires_logical_update or features.get("backward", False) or features.get("optimizer", False) or census is not None:
            if not isinstance(census, list) or any(
                not isinstance(item, dict) or item.get("disposition") not in {"required", "removable", "amortizable", "overlappable"}
                for item in census
            ):
                errors.append("work.sync_census entries need a valid disposition for update evidence")

        cache = work.get("cache_contract")
        cache_keys = {"dataset_identity", "sample_identity", "cutoff", "pbc_convention", "augmentation", "species_mapping", "graph_builder_version", "dtype_layout", "basis_version"}
        if features.get("cache", False) or cache is not None:
            if not isinstance(cache, dict) or not cache_keys.issubset(cache):
                errors.append("work.cache_contract must declare all cache-key components when cache is in scope")
            elif cache.get("cache_state") not in {"cold", "warm", "disabled", "mixed"}:
                errors.append("work.cache_contract.cache_state must be cold, warm, disabled, or mixed")

        h2d = work.get("h2d_proof")
        h2d_keys = {"is_pinned", "non_blocking", "copy_stream", "source_lifetime", "consumer_dependency", "overlap_evidence"}
        if features.get("h2d", False) or h2d is not None:
            if not isinstance(h2d, dict) or not h2d_keys.issubset(h2d):
                errors.append("work.h2d_proof must include pinned/non-blocking/stream/lifetime/dependency/overlap evidence when H2D is in scope")

    metrics = require_object(record, "metrics", errors)
    if metrics is not None:
        if evidence_level in {"amortized_job", "time_to_quality"}:
            for key in ("amortized_training_throughput", "time_to_quality_seconds"):
                if key not in metrics:
                    errors.append(f"missing metrics.{key} for {evidence_level} evidence")

    compiler = record.get("compiler")
    if features.get("compiler", False) or compiler is not None:
        compiler = require_object(record, "compiler", errors)
        if compiler is not None and compiler.get("compile_cache_state") not in {"cold", "warm", "disabled", "mixed"}:
            errors.append("compiler.compile_cache_state must be cold, warm, disabled, or mixed")

    resume = record.get("contract", {}).get("checkpoint_state_contract") if isinstance(record.get("contract"), dict) else None
    if features.get("checkpoint", False) or resume is not None:
        resume = require_object(record, "contract.checkpoint_state_contract", errors)
        if resume is not None:
            for key in ("dataloader_cursor", "ema_swa_scheduler", "preserve_rng"):
                if key not in resume:
                    errors.append(f"missing contract.checkpoint_state_contract.{key}")

    return errors


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "benchmark_record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise SystemExit("benchmark record must be a JSON object")
    errors = validate_record(record, schema)
    if errors:
        for error in errors:
            print(f"invalid: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"valid benchmark record: {path}")


if __name__ == "__main__":
    main()
