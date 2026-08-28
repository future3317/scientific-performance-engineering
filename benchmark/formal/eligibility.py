"""Agent-blind measurement eligibility for formal efficacy analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from benchmark.calibration.identity import task_package_digest
from benchmark.harness import miniyaml
from benchmark.harness.fingerprint import fingerprints_compatible
from benchmark.provenance import json_digest


SCHEMA_VERSION = 1
MANIFEST_ID = "agent-blind-efficacy-eligibility-v1"
_RUNTIME_FIELDS = (
    "python_version", "platform", "torch_version", "cuda_version", "gpu_name",
    "gpu_count", "torch_geometric_version", "cpu_affinity", "driver_version",
    "cuda_visible_devices", "gpu_uuid", "thread_topology", "cuda_available",
)
_FORBIDDEN_TASK_FIELDS = {
    "agent_outcome", "agent_result", "condition_results", "d_b_effect",
    "d_c_effect", "b_a_effect", "verified_speedup", "oracle_ci",
    "oracle_ci_low_percent", "oracle_ci_high_percent", "baseline_speedups",
}
_TASK_FIELDS = {
    "task_id", "track", "kind", "family_id", "role", "revision",
    "task_package_digest", "measurement_sessions", "runtime_contract",
    "runtime_contract_digest", "declared_noise_floor_percent",
    "observed_typical_noise_percent", "effective_noise_floor_percent",
    "baseline_stability", "protocol_valid", "correctness_scientific_valid",
    "reason", "resource_domain_status",
}
_MANIFEST_FIELDS = {
    "schema_version", "manifest_id", "status", "benchmark_revision",
    "population_manifest_digest", "calibration_protocol_digest", "task_ids",
    "resource_domain_manifest_digest", "efficacy_task_ids", "diagnostic_task_ids", "tasks", "artifact_digest",
}
_ELIGIBILITY_INPUT_FIELDS = {
    "schema_version", "benchmark_revision", "population_manifest_digest",
    "calibration_protocol_digest", "active_task_ids", "tasks",
}
_ELIGIBILITY_INPUT_TASK_FIELDS = {
    "task_id", "task_package_digest", "revision", "environment", "outer_trials",
    "noise_control", "baseline_stability", "calibration_eligible",
    "semantic_gates", "anti_cheat",
}
_RESOURCE_ELIGIBILITY_INPUT_TASK_FIELDS = _ELIGIBILITY_INPUT_TASK_FIELDS | {"resource_domain_status"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _runtime_contract(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise ValueError("measurement environment fingerprint is missing")
    return {field: fingerprint.get(field) for field in _RUNTIME_FIELDS}


def _session_map(
    task: Mapping[str, Any], *, task_id: str, benchmark_revision: str,
    population_manifest_digest: str, calibration_protocol_digest: str,
    task_package_digest_value: str, require_noise: bool,
) -> tuple[dict[str, str], list[str]]:
    reasons: list[str] = []
    trials = task.get("outer_trials")
    if not isinstance(trials, list) or not trials:
        return {}, ["outer_trials are missing"]
    sessions: dict[str, str] = {}
    for trial in trials:
        if not isinstance(trial, Mapping):
            reasons.append("outer trial is not an object")
            continue
        outer_id = str(trial.get("outer_trial_id", ""))
        session_id = trial.get("measurement_session_id")
        if not outer_id or not isinstance(session_id, str) or not session_id:
            reasons.append(f"missing measurement_session_id for {outer_id or 'outer trial'}")
            continue
        if session_id in sessions.values():
            reasons.append(f"duplicate measurement_session_id {session_id}")
        sessions[outer_id] = session_id
    if require_noise:
        noise = task.get("noise_control")
        artifacts = noise.get("artifacts") if isinstance(noise, Mapping) else None
        if not isinstance(artifacts, list) or len(artifacts) != len(trials):
            reasons.append("baseline noise artifacts do not cover every performance cell")
        else:
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    reasons.append("baseline noise artifact is not an object")
                    continue
                outer_id = str(artifact.get("outer_trial_id", ""))
                if artifact.get("measurement_session_id") != sessions.get(outer_id):
                    reasons.append(f"baseline/cell session mismatch for {outer_id or 'unknown outer trial'}")
                for key, expected in {
                    "task_id": task_id,
                    "benchmark_revision": benchmark_revision,
                    "task_package_digest": task_package_digest_value,
                    "population_manifest_digest": population_manifest_digest,
                }.items():
                    if artifact.get(key) != expected:
                        reasons.append(f"baseline noise {key} mismatch")
    return sessions, reasons


def _task_validity(
    task: Mapping[str, Any], *, task_id: str, benchmark_revision: str,
    population_manifest_digest: str, calibration_protocol_digest: str,
    task_package_digest_value: str, require_noise: bool,
    resource_domain_status: str | None = None,
) -> tuple[bool, bool, bool, dict[str, str], list[str], list[float]]:
    reasons: list[str] = []
    protocol_valid = True
    correctness_scientific_valid = True
    if resource_domain_status in {
        "out_of_domain_capacity", "out_of_domain_walltime", "unresolved",
    }:
        reasons.append(
            "RESOURCE_OUT_OF_TARGET_RUNTIME_DOMAIN"
            if resource_domain_status in {"out_of_domain_capacity", "out_of_domain_walltime"}
            else "RESOURCE_DOMAIN_UNRESOLVED"
        )
        return False, True, True, {}, reasons, []
    trials = task.get("outer_trials")
    if not isinstance(trials, list) or not trials:
        return False, False, False, {}, ["outer_trials are missing"]
    for trial in trials:
        if not isinstance(trial, Mapping):
            protocol_valid = False
            reasons.append("outer trial is not an object")
            continue
        if trial.get("execution_validity", "valid") != "valid" or trial.get("protocol_failure") or trial.get("timeout"):
            protocol_valid = False
            reasons.append("protocol or resource validity failed")
        if trial.get("correctness_pass") is not True:
            correctness_scientific_valid = False
            reasons.append("correctness failed")
        gates = trial.get("scientific_gates")
        if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
            correctness_scientific_valid = False
            reasons.append("scientific gate failed")
    semantic = task.get("semantic_gates")
    if not isinstance(semantic, Mapping) or not semantic or not all(value is True for value in semantic.values()):
        correctness_scientific_valid = False
        reasons.append("aggregate semantic gate failed")
    anti_cheat = task.get("anti_cheat")
    if not isinstance(anti_cheat, Mapping):
        correctness_scientific_valid = False
        reasons.append("anti-cheat contract failed")
    else:
        anti_findings = anti_cheat.get("findings", [])
        hard_fail = (
            anti_cheat.get("hard_fail") is True
            or any(
                isinstance(finding, Mapping) and finding.get("severity") == "hard_fail"
                for finding in (anti_findings if isinstance(anti_findings, list) else [])
            )
        )
        if hard_fail:
            correctness_scientific_valid = False
            reasons.append("anti-cheat hard failure")
        elif anti_cheat.get("status") not in {"pass", "passed", "clean"}:
            correctness_scientific_valid = False
            reasons.append("anti-cheat audit warning")
    sessions, session_reasons = _session_map(
        task, task_id=task_id, benchmark_revision=benchmark_revision,
        population_manifest_digest=population_manifest_digest,
        calibration_protocol_digest=calibration_protocol_digest,
        task_package_digest_value=task_package_digest_value,
        require_noise=require_noise,
    )
    if session_reasons:
        protocol_valid = False
        reasons.extend(session_reasons)
    noise = task.get("noise_control")
    artifacts = noise.get("artifacts") if isinstance(noise, Mapping) else None
    observed: list[float] = []
    if require_noise:
        if not isinstance(artifacts, list):
            protocol_valid = False
        else:
            for artifact in artifacts:
                value = artifact.get("typical_noise_percent", artifact.get("observed_noise_floor_percent")) if isinstance(artifact, Mapping) else None
                if not _finite(value) or float(value) < 0:
                    protocol_valid = False
                    reasons.append("baseline typical noise is missing or invalid")
                else:
                    observed.append(float(value))
    return protocol_valid and correctness_scientific_valid, protocol_valid, correctness_scientific_valid, sessions, reasons, observed


def build_manifest(
    *, pilot_calibration: Mapping[str, Any], tasks_root: str | Path,
    benchmark_revision: str, population_manifest_digest: str,
    calibration_protocol_digest: str,
) -> dict[str, Any]:
    """Project calibration-only evidence into a frozen, agent-blind manifest."""
    root = Path(tasks_root)
    allowed_input_fields = {
        frozenset(_ELIGIBILITY_INPUT_FIELDS),
        frozenset(_ELIGIBILITY_INPUT_FIELDS | {"resource_domain_manifest_digest"}),
    }
    if frozenset(pilot_calibration) not in allowed_input_fields:
        raise ValueError("eligibility input contains unsupported or missing fields")
    if _FORBIDDEN_TASK_FIELDS.intersection(pilot_calibration):
        raise ValueError("eligibility input contains agent/outcome fields")
    for key, expected in {
        "benchmark_revision": benchmark_revision,
        "population_manifest_digest": population_manifest_digest,
        "calibration_protocol_digest": calibration_protocol_digest,
    }.items():
        if pilot_calibration.get(key) != expected:
            label = {
                "benchmark_revision": "benchmark revision",
                "population_manifest_digest": "population digest",
                "calibration_protocol_digest": "protocol digest",
            }[key]
            raise ValueError(f"eligibility {label} mismatch")
    resource_manifest_digest = pilot_calibration.get("resource_domain_manifest_digest")
    resource_bound = resource_manifest_digest is not None
    if resource_bound and (not isinstance(resource_manifest_digest, str) or not resource_manifest_digest):
        raise ValueError("resource domain manifest digest is invalid")
    task_ids = [str(item) for item in pilot_calibration.get("active_task_ids", [])]
    by_id = {str(item.get("task_id")): item for item in pilot_calibration.get("tasks", []) if isinstance(item, Mapping)}
    if not task_ids or set(task_ids) != set(by_id):
        raise ValueError("pilot calibration must cover exactly its active task IDs")
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        pilot_task = by_id[task_id]
        task_dir = root / task_id
        spec_path = task_dir / "task.yaml"
        if not spec_path.is_file():
            raise ValueError(f"task package is missing: {task_id}")
        spec = miniyaml.load(str(spec_path))
        forbidden = _FORBIDDEN_TASK_FIELDS.intersection(pilot_task)
        if forbidden:
            raise ValueError(f"eligibility input contains agent/outcome fields for {task_id}: {sorted(forbidden)}")
        expected_task_fields = _RESOURCE_ELIGIBILITY_INPUT_TASK_FIELDS if resource_bound else _ELIGIBILITY_INPUT_TASK_FIELDS
        if set(pilot_task) != expected_task_fields:
            if not resource_bound and "resource_domain_status" in pilot_task:
                raise ValueError("resource domain status requires a frozen resource domain manifest")
            raise ValueError(f"eligibility input contains unsupported or missing fields for {task_id}")
        package_digest = pilot_task.get("task_package_digest")
        if package_digest != task_package_digest(task_dir):
            raise ValueError(f"task package digest mismatch for {task_id}")
        task_digest = str(package_digest)
        revision = pilot_task.get("revision")
        if revision != benchmark_revision:
            raise ValueError(f"benchmark revision mismatch for {task_id}")
        fingerprint = pilot_task.get("environment")
        contract = _runtime_contract(fingerprint)
        require_noise = spec.get("track") != "evolution"
        resource_status = pilot_task.get("resource_domain_status")
        if resource_bound and resource_status not in {
            "in_domain", "out_of_domain_capacity", "out_of_domain_walltime",
            "not_applicable", "unresolved",
        }:
            raise ValueError(f"{task_id}: invalid resource domain status")
        valid, protocol_valid, scientific_valid, sessions, reasons, observed = _task_validity(
            pilot_task, task_id=task_id, benchmark_revision=benchmark_revision,
            population_manifest_digest=population_manifest_digest,
            calibration_protocol_digest=calibration_protocol_digest,
            task_package_digest_value=task_digest, require_noise=require_noise,
            resource_domain_status=resource_status,
        )
        if resource_bound and resource_status != "in_domain" and resource_status not in {
            "out_of_domain_capacity", "out_of_domain_walltime", "unresolved",
        }:
            reasons.append("RESOURCE_DOMAIN_UNRESOLVED")
        role = "efficacy" if valid and require_noise and str(spec.get("kind")) == "positive" and (not resource_bound or resource_status == "in_domain") else "diagnostic"
        if role == "efficacy" and pilot_task.get("calibration_eligible", True) is not True:
            role = "diagnostic"
            reasons.append("CALIBRATION_ATTRITION")
        declared = float((spec.get("measurement") or {}).get("noise_floor_percent", 2.0))
        effective = max([declared, *observed])
        tasks.append({
            "task_id": task_id,
            "track": str(spec.get("track", "")),
            "kind": str(spec.get("kind", "")),
            "family_id": str(spec.get("family_id", spec.get("family", ""))),
            "role": role,
            "revision": str(revision),
            "task_package_digest": task_digest,
            "measurement_sessions": dict(sorted(sessions.items())),
            "runtime_contract": contract,
            "runtime_contract_digest": json_digest(contract),
            "declared_noise_floor_percent": declared,
            "observed_typical_noise_percent": max(observed) if observed else None,
            "effective_noise_floor_percent": effective,
            "baseline_stability": (
                "not_applicable" if not require_noise
                else "pass" if pilot_task.get("baseline_stability") is True
                else "descriptive_only"
            ),
            "protocol_valid": protocol_valid,
            "correctness_scientific_valid": scientific_valid,
            "reason": sorted(set(reasons)),
            "resource_domain_status": resource_status,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "status": "frozen",
        "benchmark_revision": str(benchmark_revision),
        "population_manifest_digest": str(population_manifest_digest),
        "calibration_protocol_digest": str(calibration_protocol_digest),
        "resource_domain_manifest_digest": resource_manifest_digest,
        "task_ids": task_ids,
        "efficacy_task_ids": [item["task_id"] for item in tasks if item["role"] == "efficacy"],
        "diagnostic_task_ids": [item["task_id"] for item in tasks if item["role"] == "diagnostic"],
        "tasks": tasks,
    }
    manifest["artifact_digest"] = json_digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(manifest) != _MANIFEST_FIELDS:
        errors.append("eligibility manifest contains unsupported or missing fields")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_id") != MANIFEST_ID:
        errors.append("eligibility manifest identity mismatch")
    if manifest.get("status") != "frozen":
        errors.append("eligibility manifest is not frozen")
    expected_digest = json_digest({key: value for key, value in manifest.items() if key != "artifact_digest"})
    if manifest.get("artifact_digest") != expected_digest:
        errors.append("eligibility manifest digest mismatch")
    tasks = manifest.get("tasks")
    task_ids = manifest.get("task_ids")
    if not isinstance(tasks, list) or not isinstance(task_ids, list) or len(tasks) != len(task_ids):
        errors.append("eligibility manifest task coverage is invalid")
        return errors
    seen_sessions: set[str] = set()
    observed_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != _TASK_FIELDS:
            errors.append("eligibility task contains unsupported or missing fields")
            continue
        task_id = str(task.get("task_id"))
        observed_ids.append(task_id)
        sessions = task.get("measurement_sessions")
        if not isinstance(sessions, Mapping):
            errors.append(f"{task_id}: measurement_sessions must be an object")
        else:
            for session_id in sessions.values():
                if not isinstance(session_id, str) or not session_id or session_id in seen_sessions:
                    errors.append(f"{task_id}: measurement_session_id is missing or duplicated")
                seen_sessions.add(str(session_id))
        if task.get("role") not in {"efficacy", "diagnostic"}:
            errors.append(f"{task_id}: invalid eligibility role")
    if observed_ids != [str(item) for item in task_ids]:
        errors.append("eligibility task ordering or IDs mismatch")
    expected_efficacy = [str(item.get("task_id")) for item in tasks if item.get("role") == "efficacy"]
    expected_diagnostic = [str(item.get("task_id")) for item in tasks if item.get("role") == "diagnostic"]
    if manifest.get("efficacy_task_ids") != expected_efficacy or manifest.get("diagnostic_task_ids") != expected_diagnostic:
        errors.append("eligibility role projections mismatch")
    return errors


def load_manifest(path: str | Path, *, benchmark_revision: str | None = None, population_manifest_digest: str | None = None, calibration_protocol_digest: str | None = None) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"eligibility manifest unavailable: {path}") from exc
    errors = validate_manifest(manifest)
    if benchmark_revision is not None and manifest.get("benchmark_revision") != benchmark_revision:
        errors.append("eligibility benchmark revision mismatch")
    if population_manifest_digest is not None and manifest.get("population_manifest_digest") != population_manifest_digest:
        errors.append("eligibility population digest mismatch")
    if calibration_protocol_digest is not None and manifest.get("calibration_protocol_digest") != calibration_protocol_digest:
        errors.append("eligibility protocol digest mismatch")
    if errors:
        raise ValueError("invalid eligibility manifest: " + "; ".join(errors))
    return dict(manifest)


def validate_runtime_contract(manifest: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for task in manifest.get("tasks", []):
        if task.get("role") != "efficacy":
            continue
        compatible, reasons = fingerprints_compatible(dict(task.get("runtime_contract", {})), dict(fingerprint))
        if not compatible:
            errors.append(f"{task.get('task_id')}: runtime contract mismatch: {'; '.join(reasons)}")
    return errors


def task_entry(manifest: Mapping[str, Any], task_id: str) -> dict[str, Any] | None:
    return next((dict(item) for item in manifest.get("tasks", []) if str(item.get("task_id")) == str(task_id)), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-calibration", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--population-manifest-digest", required=True)
    parser.add_argument("--calibration-protocol-digest", required=True)
    parser.add_argument("--resource-domain-manifest", type=Path, default=None, help="frozen baseline-only resource-domain manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from benchmark.calibration.report import build_efficacy_eligibility_input

    resource_domain_manifest = None
    if args.resource_domain_manifest is not None:
        from benchmark.formal.resource_domain import load_resource_domain_manifest

        resource_domain_manifest = load_resource_domain_manifest(
            args.resource_domain_manifest,
            repo_root=args.tasks_root.resolve().parents[1],
        )
    pilot = build_efficacy_eligibility_input(
        json.loads(args.pilot_calibration.read_text(encoding="utf-8")),
        resource_domain_manifest=resource_domain_manifest,
    )
    manifest = build_manifest(
        pilot_calibration=pilot, tasks_root=args.tasks_root,
        benchmark_revision=args.benchmark_revision,
        population_manifest_digest=args.population_manifest_digest,
        calibration_protocol_digest=args.calibration_protocol_digest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "efficacy_tasks": len(manifest["efficacy_task_ids"]), "diagnostic_tasks": len(manifest["diagnostic_task_ids"]), "artifact_digest": manifest["artifact_digest"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
