"""Experiment and artifact attestation for formal evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from benchmark.harness import anticheat, miniyaml
from benchmark.harness.fingerprint import fingerprints_compatible
from benchmark.harness.skill_view import validate_skill_view_bundle
from benchmark.provenance import benchmark_revision, canonical_json, digest_mapping, file_digest
from benchmark.calibration.bundle import calibration_envelope, validate_calibration_envelope
from benchmark.calibration.execution import executor_digest
from benchmark.calibration.identity import task_package_digest, taskset_digest


def skill_view_digest(bundle: str | Path) -> str:
    bundle = Path(bundle)
    errors = validate_skill_view_bundle(bundle)
    if errors:
        raise ValueError("invalid skill-view bundle: " + "; ".join(errors))
    files = anticheat.hash_tree(bundle)
    files.pop("skill_view_manifest.json", None)
    return digest_mapping(files)


def harness_digest(repo_root: str | Path) -> str:
    """Formal-facing name for the calibration executor digest."""
    return executor_digest(repo_root)


task_manifest_digest = taskset_digest


REQUIRED_FIELDS = (
    "schema_version", "experiment_id", "benchmark_revision", "skill_view_digest",
    "task_manifest_digest", "agent_model_id", "agent_config", "condition",
    "context_mode", "worker_isolation", "task_order", "outer_trial_id", "budgets",
    "hardware_fingerprint", "software_fingerprint", "torch_version", "cuda_version",
)


def validate_experiment(manifest: dict[str, Any]) -> list[str]:
    errors = [f"missing {key}" for key in REQUIRED_FIELDS if key not in manifest]
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("condition") not in {"A", "A_CTX", "B", "C", "C_STRESS", "D"}:
        errors.append("condition must be A, A_CTX, B, C, C_STRESS, or D")
    if manifest.get("context_mode") not in {"reset", "carry"}:
        errors.append("context_mode must be reset or carry")
    if not isinstance(manifest.get("task_order"), list) or not manifest.get("task_order"):
        errors.append("task_order must be a non-empty list")
    if manifest.get("population_id") == "SPE-EvoBench-v1.0-50":
        if not manifest.get("slot_id"):
            errors.append("formal manifest requires slot_id")
        if manifest.get("visibility") not in {"sealed", "public_dev"}:
            errors.append("formal manifest requires explicit visibility")
        config = manifest.get("agent_config")
        required_config = (
            "provider", "model_snapshot", "temperature", "top_p", "max_output_tokens",
            "system_prompt_digest", "agent_code_commit", "tool_versions", "tool_allowlist",
            "retry_policy", "timeout_s", "container_digest", "pricing_revision",
        )
        if not isinstance(config, dict):
            errors.append("formal agent_config must be an object")
        else:
            errors.extend(f"formal agent_config missing {key}" for key in required_config if key not in config)
            for key in ("provider", "model_snapshot", "system_prompt_digest", "agent_code_commit", "container_digest", "pricing_revision"):
                if key in config and (not isinstance(config[key], str) or not config[key].strip() or config[key] in {"unknown", "unidentified-agent", "pending"}):
                    errors.append(f"formal agent_config {key} must be a non-placeholder string")
            for key in ("system_prompt_digest", "container_digest"):
                if key in config and isinstance(config[key], str) and not re.fullmatch(r"[a-f0-9]{64}", config[key]):
                    errors.append(f"formal agent_config {key} must be a 64-hex digest")
            if "agent_code_commit" in config and isinstance(config["agent_code_commit"], str) and not re.fullmatch(r"[0-9a-f]{7,64}", config["agent_code_commit"]):
                errors.append("formal agent_config agent_code_commit must be a commit digest")
            for key in ("tool_versions", "retry_policy"):
                if key in config and not isinstance(config[key], dict):
                    errors.append(f"formal agent_config {key} must be an object")
            if "tool_allowlist" in config and (not isinstance(config["tool_allowlist"], list) or not config["tool_allowlist"]):
                errors.append("formal agent_config tool_allowlist must be a non-empty array")
    isolation = manifest.get("worker_isolation")
    if not isinstance(isolation, dict):
        errors.append("worker_isolation must be an object")
    else:
        if isolation.get("mode") != "external_namespace_executor":
            errors.append("worker_isolation.mode must be external_namespace_executor")
        if isolation.get("network_mode") != "none":
            errors.append("worker_isolation.network_mode must be none")
        if not isinstance(isolation.get("mount_allowlist"), list) or not isolation.get("mount_allowlist"):
            errors.append("worker_isolation.mount_allowlist must be non-empty")
        receipt = isolation.get("executor_receipt")
        if receipt is not None and (not isinstance(receipt, dict) or receipt.get("network_mode") != "none"):
            errors.append("worker_isolation.executor_receipt must attest network_mode=none")
    return errors


def write_experiment(path: str | Path, manifest: dict[str, Any]) -> None:
    errors = validate_experiment(manifest)
    if errors:
        raise ValueError("invalid experiment manifest: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
