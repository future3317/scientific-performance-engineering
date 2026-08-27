"""Baseline-only resource-domain eligibility for the registered population."""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.calibration.identity import task_package_digest
from benchmark.harness.fingerprint import fingerprints_compatible
from benchmark.provenance import benchmark_revision, file_digest, json_digest


SCHEMA_VERSION = 1
MANIFEST_ID = "resource-domain-eligibility-v1"
STATUSES = {"in_domain", "out_of_domain_capacity", "not_applicable", "unresolved"}
_FORBIDDEN_KEY_PARTS = {
    "agent", "oracle", "candidate", "d_b", "d_c", "b_a", "condition_results",
    "verified_speedup", "efficacy", "utility", "score",
}
_TASK_FIELDS = {
    "task_id", "task_package_digest", "execution_class", "resource_domain_status",
    "baseline_probe_session_id", "clean_admission", "foreign_processes",
    "total_device_memory", "preflight_free_memory", "peak_allocated", "peak_reserved",
    "reserved_unallocated", "failed_allocation_bytes", "exception_class",
    "failure_signature", "confirmation_attempt_count", "structured_reason",
    "evidence_digest", "protocol_valid", "baseline_completed", "cleanup_quiescent",
}
_MANIFEST_FIELDS = {
    "schema_version", "manifest_id", "status", "benchmark_revision",
    "population_manifest_digest", "resource_protocol_digest", "runtime_contract",
    "runtime_contract_digest", "device_policy", "selected_gpu_uuid", "total_vram_bytes",
    "admission_contract", "task_ids", "in_domain_task_ids", "out_of_domain_task_ids",
    "not_applicable_task_ids", "unresolved_task_ids", "tasks", "artifact_digest",
}


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(normalized == part or normalized.startswith(part + "_") for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError("resource-domain evidence contains agent/outcome fields")
            _reject_forbidden_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_keys(child)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _capacity_evidence(probe: Mapping[str, Any]) -> bool:
    total = _number(probe.get("total_device_memory"))
    peak = _number(probe.get("peak_allocated"))
    free = _number(probe.get("preflight_free_memory"))
    failed = _number(probe.get("failed_allocation_bytes"))
    return bool(
        total and peak and free is not None and failed
        and free < failed
        and str(probe.get("exception_class")) in {"torch.OutOfMemoryError", "cuda.OutOfMemoryError"}
    )


def classify_baseline_probe(probe: Mapping[str, Any]) -> str:
    """Classify resource feasibility without consulting any performance outcome."""
    if not isinstance(probe, Mapping):
        raise ValueError("resource-domain probe must be an object")
    _reject_forbidden_keys(probe)
    if probe.get("baseline_only") is not True:
        return "unresolved"
    if probe.get("clean_admission") is not True or probe.get("foreign_processes"):
        return "unresolved"
    if probe.get("protocol_valid") is not True or probe.get("cleanup_quiescent") is not True:
        return "unresolved"
    if probe.get("resource_exhausted"):
        confirmed = int(probe.get("confirmation_attempt_count", 0)) >= 2
        if confirmed and probe.get("capacity_failure_reproduced") is True and _capacity_evidence(probe):
            return "out_of_domain_capacity"
        return "unresolved"
    if probe.get("not_applicable") is True:
        return "not_applicable"
    if probe.get("baseline_completed") is True:
        return "in_domain"
    return "unresolved"


def _active_manifest(repo_root: Path) -> tuple[dict[str, Any], Path]:
    path = repo_root / "benchmark" / "pilot_population.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _task_entry(probe: Mapping[str, Any], status: str) -> dict[str, Any]:
    entry = {
        "task_id": str(probe["task_id"]),
        "task_package_digest": str(probe["task_package_digest"]),
        "execution_class": str(probe.get("execution_class", "atomic_performance")),
        "resource_domain_status": status,
        "baseline_probe_session_id": str(probe.get("baseline_probe_session_id", "")),
        "clean_admission": bool(probe.get("clean_admission")),
        "foreign_processes": list(probe.get("foreign_processes", [])),
        "total_device_memory": probe.get("total_device_memory"),
        "preflight_free_memory": probe.get("preflight_free_memory"),
        "peak_allocated": probe.get("peak_allocated"),
        "peak_reserved": probe.get("peak_reserved"),
        "reserved_unallocated": probe.get("reserved_unallocated"),
        "failed_allocation_bytes": probe.get("failed_allocation_bytes"),
        "exception_class": probe.get("exception_class"),
        "failure_signature": probe.get("failure_signature"),
        "confirmation_attempt_count": int(probe.get("confirmation_attempt_count", 0)),
        "structured_reason": (
            str(probe.get("structured_reason") or "HOST_CAPACITY_LIMIT") if status == "out_of_domain_capacity"
            else "RESOURCE_BASELINE_COMPLETED" if status == "in_domain"
            else "RESOURCE_DOMAIN_UNRESOLVED"
        ),
        "evidence_digest": json_digest(dict(probe)),
        "protocol_valid": bool(probe.get("protocol_valid")),
        "baseline_completed": bool(probe.get("baseline_completed")),
        "cleanup_quiescent": bool(probe.get("cleanup_quiescent")),
    }
    return entry


def build_resource_domain_manifest(
    *, repo_root: Path, task_probes: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any], out_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic frozen manifest from baseline-only probe records."""
    _reject_forbidden_keys(task_probes)
    _reject_forbidden_keys(runtime_contract)
    active, active_path = _active_manifest(repo_root)
    task_ids = [str(item) for item in active["task_ids"]]
    by_id = {str(probe.get("task_id")): probe for probe in task_probes}
    if set(by_id) != set(task_ids) or len(by_id) != len(task_ids):
        raise ValueError("resource census must account for exactly the active population")
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        probe = by_id[task_id]
        task_dir = repo_root / "benchmark" / "tasks" / task_id
        expected_digest = task_package_digest(task_dir)
        if probe.get("task_package_digest") != expected_digest:
            raise ValueError(f"task package digest mismatch for {task_id}")
        if probe.get("baseline_only") is not True:
            raise ValueError(f"resource probe is not baseline-only for {task_id}")
        tasks.append(_task_entry(probe, classify_baseline_probe(probe)))
    runtime = dict(runtime_contract)
    statuses = {item["task_id"]: item["resource_domain_status"] for item in tasks}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "status": "frozen",
        "benchmark_revision": benchmark_revision(repo_root),
        "population_manifest_digest": file_digest(active_path),
        "resource_protocol_digest": file_digest(Path(__file__)),
        "runtime_contract": runtime,
        "runtime_contract_digest": json_digest(runtime),
        "device_policy": {
            "mode": "fixed_gpu_uuid",
            "cuda_visible_devices": runtime.get("cuda_visible_devices"),
            "no_shared_compute": True,
        },
        "selected_gpu_uuid": runtime.get("gpu_uuid"),
        "total_vram_bytes": runtime.get("total_device_memory"),
        "admission_contract": {
            "clean_required": True,
            "foreign_processes_allowed": False,
            "baseline_only": True,
            "capacity_confirmation_attempts": 2,
        },
        "task_ids": task_ids,
        "in_domain_task_ids": [task_id for task_id in task_ids if statuses[task_id] == "in_domain"],
        "out_of_domain_task_ids": [task_id for task_id in task_ids if statuses[task_id] == "out_of_domain_capacity"],
        "not_applicable_task_ids": [task_id for task_id in task_ids if statuses[task_id] == "not_applicable"],
        "unresolved_task_ids": [task_id for task_id in task_ids if statuses[task_id] == "unresolved"],
        "tasks": tasks,
    }
    manifest["artifact_digest"] = json_digest(manifest)
    errors = validate_resource_domain_manifest(manifest, repo_root=repo_root)
    if errors:
        raise ValueError("invalid resource-domain manifest: " + "; ".join(errors))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def validate_resource_domain_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path,
    expected_runtime_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        _reject_forbidden_keys(manifest)
    except ValueError as exc:
        return [str(exc)]
    if set(manifest) != _MANIFEST_FIELDS:
        errors.append("resource-domain manifest contains unsupported or missing fields")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_id") != MANIFEST_ID:
        errors.append("resource-domain manifest identity mismatch")
    if manifest.get("status") != "frozen":
        errors.append("resource-domain manifest is not frozen")
    expected_artifact = json_digest({key: value for key, value in manifest.items() if key != "artifact_digest"})
    if manifest.get("artifact_digest") != expected_artifact:
        errors.append("resource-domain artifact digest mismatch")
    active, active_path = _active_manifest(repo_root)
    active_ids = [str(item) for item in active["task_ids"]]
    if manifest.get("benchmark_revision") != benchmark_revision(repo_root):
        errors.append("resource-domain benchmark revision mismatch")
    if manifest.get("population_manifest_digest") != file_digest(active_path):
        errors.append("resource-domain population digest mismatch")
    if manifest.get("resource_protocol_digest") != file_digest(Path(__file__)):
        errors.append("resource-domain protocol digest mismatch")
    runtime = manifest.get("runtime_contract")
    if not isinstance(runtime, Mapping) or manifest.get("runtime_contract_digest") != json_digest(dict(runtime)):
        errors.append("resource-domain runtime digest mismatch")
    if expected_runtime_contract is not None and dict(runtime or {}) != dict(expected_runtime_contract):
        errors.append("resource-domain runtime contract mismatch")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or [str(item.get("task_id")) for item in tasks if isinstance(item, Mapping)] != active_ids:
        errors.append("resource-domain task coverage or ordering mismatch")
        return errors
    statuses: dict[str, str] = {}
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != _TASK_FIELDS:
            errors.append("resource-domain task entry contains unsupported or missing fields")
            continue
        task_id = str(task["task_id"])
        statuses[task_id] = str(task.get("resource_domain_status"))
        if statuses[task_id] not in STATUSES:
            errors.append(f"{task_id}: invalid resource-domain status")
        try:
            if task.get("task_package_digest") != task_package_digest(repo_root / "benchmark" / "tasks" / task_id):
                errors.append(f"{task_id}: task package digest mismatch")
        except FileNotFoundError:
            errors.append(f"{task_id}: task package missing")
        if not task.get("baseline_probe_session_id"):
            errors.append(f"{task_id}: baseline probe session is missing")
    if manifest.get("in_domain_task_ids") != [task_id for task_id in active_ids if statuses.get(task_id) == "in_domain"]:
        errors.append("resource-domain in-domain projection mismatch")
    if manifest.get("out_of_domain_task_ids") != [task_id for task_id in active_ids if statuses.get(task_id) == "out_of_domain_capacity"]:
        errors.append("resource-domain out-of-domain projection mismatch")
    if manifest.get("not_applicable_task_ids") != [task_id for task_id in active_ids if statuses.get(task_id) == "not_applicable"]:
        errors.append("resource-domain not-applicable projection mismatch")
    if manifest.get("unresolved_task_ids") != [task_id for task_id in active_ids if statuses.get(task_id) == "unresolved"]:
        errors.append("resource-domain unresolved projection mismatch")
    return errors


def load_resource_domain_manifest(
    path: str | Path, *, repo_root: Path,
    expected_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resource-domain manifest unavailable: {path}") from exc
    errors = validate_resource_domain_manifest(manifest, repo_root=repo_root, expected_runtime_contract=expected_runtime_contract)
    if errors:
        raise ValueError("invalid resource-domain manifest: " + "; ".join(errors))
    return dict(manifest)


def validate_runtime_contract(
    manifest: Mapping[str, Any], fingerprint: Mapping[str, Any],
) -> list[str]:
    """Compare a frozen resource runtime with the current execution fingerprint."""
    runtime = manifest.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        return ["resource-domain runtime contract is missing"]
    compatible, reasons = fingerprints_compatible(dict(runtime), dict(fingerprint))
    return [] if compatible else ["; ".join(reasons)]


def run_baseline_probe(
    *, task_dir: str | Path, solution_dir: str | Path, out_path: str | Path, seed: int = 0,
) -> dict[str, Any]:
    """Execute one registered baseline performance call, never an oracle/candidate."""
    from benchmark.harness import miniyaml, runner
    from benchmark.harness.api import execution_class_for_task

    task_root = Path(task_dir)
    spec = miniyaml.load(str(task_root / "task.yaml"))
    if execution_class_for_task(spec) == "episode":
        raise ValueError("episode resource probes are not applicable")
    device, usable = runner.select_device(bool(spec.get("requires_cuda")))
    probe: dict[str, Any] = {
        "task_id": str(spec["task_id"]),
        "execution_class": "atomic_performance",
        "baseline_only": True,
        "baseline_completed": False,
        "resource_exhausted": False,
        "exception_class": None,
        "failure_signature": None,
        "total_device_memory": None,
        "preflight_free_memory": None,
        "peak_allocated": None,
        "peak_reserved": None,
        "reserved_unallocated": None,
        "failed_allocation_bytes": None,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not usable:
        probe.update({"protocol_valid": True, "cleanup_quiescent": True, "failure_signature": "CUDA unavailable"})
        Path(out_path).write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        return probe
    try:
        import torch

        module = runner.import_module_by_path(task_root / "benchmark.py")
        solution = runner.call_benchmark_fn(module.load_solution, path=str(solution_dir), device=device)
        fixtures = runner.call_benchmark_fn(module.make_fixtures, seed=seed, device=device)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            properties = torch.cuda.get_device_properties(0)
            probe["total_device_memory"] = int(properties.total_memory)
            probe["preflight_free_memory"] = int(torch.cuda.mem_get_info()[0])
        runner.call_benchmark_fn(
            module.run_performance, solution=solution, fixtures=fixtures,
            warmup=0, iterations=1, device=device,
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            probe["peak_allocated"] = int(torch.cuda.max_memory_allocated())
            probe["peak_reserved"] = int(torch.cuda.max_memory_reserved())
            probe["reserved_unallocated"] = max(0, probe["peak_reserved"] - probe["peak_allocated"])
        probe["baseline_completed"] = True
        probe["protocol_valid"] = True
        probe["cleanup_quiescent"] = True
    except Exception as exc:  # the parent classifies this evidence; no outcome is inferred
        probe["protocol_valid"] = True
        probe["cleanup_quiescent"] = True
        probe["exception_class"] = f"{type(exc).__module__}.{type(exc).__name__}"
        probe["failure_signature"] = str(exc)[0:4000]
        if "outofmemory" in probe["exception_class"].lower() or "out of memory" in str(exc).lower():
            probe["resource_exhausted"] = True
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                probe["peak_allocated"] = int(torch.cuda.max_memory_allocated())
                probe["peak_reserved"] = int(torch.cuda.max_memory_reserved())
                probe["reserved_unallocated"] = max(0, probe["peak_reserved"] - probe["peak_allocated"])
                probe["total_device_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass
        match = re.search(r"Tried to allocate\s+([0-9.]+)\s+MiB", str(exc))
        if match:
            probe["failed_allocation_bytes"] = int(float(match.group(1)) * 1024**2)
        free_match = re.search(r"of which\s+([0-9.]+)\s+MiB is free", str(exc))
        if free_match:
            probe["preflight_free_memory"] = int(float(free_match.group(1)) * 1024**2)
    Path(out_path).write_text(json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-task", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    probe = run_baseline_probe(
        task_dir=args.probe_task, solution_dir=args.solution, out_path=args.out, seed=args.seed,
    )
    print(json.dumps({"task_id": probe["task_id"], "baseline_completed": probe["baseline_completed"], "resource_exhausted": probe["resource_exhausted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
