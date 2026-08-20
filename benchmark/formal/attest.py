"""Experiment and artifact attestation for formal evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import re
from pathlib import Path
from typing import Any

from benchmark.harness import anticheat, miniyaml
from benchmark.harness.fingerprint import fingerprints_compatible
from scripts.render_skill_view import validate_skill_view_bundle


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_mapping(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def skill_view_digest(bundle: str | Path) -> str:
    bundle = Path(bundle)
    errors = validate_skill_view_bundle(bundle)
    if errors:
        raise ValueError("invalid skill-view bundle: " + "; ".join(errors))
    files = anticheat.hash_tree(bundle)
    files.pop("skill_view_manifest.json", None)
    return digest_mapping(files)


PACKAGE_DIRS = ("workspace", "public_tests", "hidden_verifier", "oracle")
PACKAGE_FILES = ("task.yaml", "metadata.json", "benchmark.py", "scientific_contract.py")


def file_digest(path: str | Path) -> str:
    """Digest one persisted calibration artifact by its exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_digest(repo_root: str | Path) -> str:
    """Digest the executable calibration/verifier surface."""
    root = Path(repo_root)
    files: dict[str, str] = {}
    roots = [root / "benchmark" / "harness", root / "benchmark" / "formal"]
    paths = [
        root / "scripts" / "run_active30_calibration.py",
        root / "benchmark" / "schema" / "task.schema.json",
        root / "benchmark" / "schema" / "result.schema.json",
    ]
    for base in roots:
        if base.is_dir():
            paths.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    for path in sorted(set(paths)):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return digest_mapping(files)


def calibration_envelope(
    *,
    producer_revision: str,
    task_package_digest: str,
    population_manifest_digest: str,
    harness_digest_value: str,
    calibration_runner_digest: str,
    noise_digest: str,
    raw_result_digest: str,
    fingerprint: dict[str, Any],
    task_id: str,
    outer_trial_id: str,
    seed: int,
    measurement_class: str,
) -> dict[str, Any]:
    """Return the immutable identity envelope for one calibration cell."""
    envelope = {
        "schema_version": 1,
        "task_id": str(task_id),
        "outer_trial_id": str(outer_trial_id),
        "seed": int(seed),
        "measurement_class": str(measurement_class),
        "producer_revision": str(producer_revision),
        "task_package_digest": str(task_package_digest),
        "population_manifest_digest": str(population_manifest_digest),
        "harness_digest": str(harness_digest_value),
        "calibration_runner_digest": str(calibration_runner_digest),
        "noise_digest": str(noise_digest),
        "raw_result_digest": str(raw_result_digest),
        "fingerprint": dict(fingerprint),
    }
    envelope["envelope_digest"] = digest_mapping(envelope)
    return envelope


def validate_calibration_envelope(payload: dict[str, Any], expected: dict[str, Any] | None = None) -> list[str]:
    """Validate the cell identity and self-digest before reuse."""
    errors: list[str] = []
    required = {
        "schema_version", "task_id", "outer_trial_id", "seed", "measurement_class",
        "producer_revision", "task_package_digest", "population_manifest_digest",
        "harness_digest", "calibration_runner_digest", "noise_digest", "raw_result_digest",
        "fingerprint", "envelope_digest",
    }
    errors.extend(f"missing {key}" for key in sorted(required - set(payload)))
    if payload.get("schema_version") != 1:
        errors.append("schema_version mismatch")
    if payload.get("envelope_digest") != digest_mapping({key: value for key, value in payload.items() if key != "envelope_digest"}):
        errors.append("envelope_digest mismatch")
    for key, value in (expected or {}).items():
        if key == "fingerprint":
            actual_fingerprint = payload.get("fingerprint")
            if not isinstance(actual_fingerprint, dict) or not isinstance(value, dict):
                errors.append("fingerprint missing or invalid")
            else:
                compatible, reasons = fingerprints_compatible(actual_fingerprint, value)
                if not compatible:
                    errors.append("fingerprint mismatch: " + "; ".join(reasons))
            continue
        if key in payload and payload.get(key) != value:
            errors.append(f"{key} mismatch")
    return errors


def task_package_digest(task_dir: str | Path) -> str:
    """Digest the executable task package, not just its declarative manifest."""
    root = Path(task_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"task package not found: {root}")
    files: dict[str, str] = {}
    paths = [root / name for name in PACKAGE_FILES]
    for directory in PACKAGE_DIRS:
        base = root / directory
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"))
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"task package member missing: {path}")
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest_mapping(files)


def task_manifest_digest(tasks_root: str | Path, task_ids: list[str]) -> str:
    root = Path(tasks_root)
    payload: dict[str, Any] = {}
    for task_id in task_ids:
        task_dir = root / task_id
        task_path = task_dir / "task.yaml"
        if not task_path.is_file():
            raise FileNotFoundError(f"task manifest not found: {task_path}")
        payload[task_id] = {
            "package_digest": task_package_digest(task_dir),
            "task_id": task_id,
        }
    return digest_mapping(payload)


def benchmark_revision(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
