#!/usr/bin/env python3
"""Validate and summarize the v1.0-30 population without running long campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml
from benchmark.harness.split import check_leakage
from benchmark.taskgen.generate import ast_skeleton_hash
from benchmark.families import resolve_family_id
from benchmark.families.catalog import FAMILY_SPECS, family_instance_digest, reconstruct_anchor_instance
from benchmark.formal.attest import task_package_digest


ATOMIC_REQUIRED = (
    "generator_family_id", "oracle_fix_pattern_id", "scientific_contract_id",
    "workspace_ast_skeleton_hash", "difficulty_tier",
)
EXPECTED_COUNTS = {"spe_core": 16, "sciml": 11, "evolution": 3}
EMPIRICAL_FLAGS = (
    "oracle_effect_too_small",
    "noise_too_high",
    "oracle_effect_unstable",
    "baseline_already_optimal",
    "semantic_gate_too_weak",
    "repair_pattern_duplicate",
    "difficulty_ceiling",
    "difficulty_floor",
    "platform_direction_flip",
    "agent_shortcut_detected",
)
CALIBRATION_FIELDS = (
    "task_digest", "revision", "environment", "outer_trials", "noise_control",
    "oracle_ci", "semantic_gates", "anti_cheat",
)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def load_active_manifest(tasks_root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Load the explicit active population; never infer formal population from a directory listing."""
    root = Path(tasks_root)
    path = Path(manifest_path) if manifest_path is not None else root.parent / "pilot_population.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"active population manifest is unreadable: {path}: {exc}") from exc
    task_ids = payload.get("task_ids") if isinstance(payload, dict) else None
    if payload.get("status") != "active_manifest" or not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise ValueError("active population manifest must be status=active_manifest with unique task_ids")
    return payload


def _artifact_findings(task_dir: Path, spec: dict[str, Any]) -> list[str]:
    oracle = task_dir / "oracle"
    required = ["bottleneck.json", "expected_mechanism.json", "reference_patch.diff", "tempting_wrong_patch.md", "noise_floor.json", "validation.json"]
    errors = [f"{task_dir.name}: missing oracle/{name}" for name in required if not (oracle / name).is_file()]
    if not (task_dir / "workspace" / str(spec["workspace"]["entrypoint"])).is_file():
        errors.append(f"{task_dir.name}: baseline workspace entrypoint missing")
    if not (task_dir / "benchmark.py").is_file():
        errors.append(f"{task_dir.name}: benchmark.py missing")
    validation_path = oracle / "validation.json"
    if validation_path.is_file():
        try:
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{task_dir.name}: validation.json is invalid: {exc}")
        else:
            for key in ("baseline_validation", "oracle_validation", "anti_cheat", "deterministic_fixture"):
                if key not in payload:
                    errors.append(f"{task_dir.name}: validation.json lacks {key}")
    noise_path = oracle / "noise_floor.json"
    if noise_path.is_file():
        try:
            noise = json.loads(noise_path.read_text(encoding="utf-8"))
            if not isinstance(noise.get("declared_percent"), (int, float)):
                errors.append(f"{task_dir.name}: noise_floor.json lacks numeric declared_percent")
        except json.JSONDecodeError as exc:
            errors.append(f"{task_dir.name}: noise_floor.json is invalid: {exc}")
    if not (task_dir / "hidden_verifier").is_dir():
        errors.append(f"{task_dir.name}: hidden_verifier directory missing")
    return errors


def _metadata_findings(task_dir: Path, spec: dict[str, Any]) -> list[str]:
    path = task_dir / "metadata.json"
    if not path.is_file():
        return [f"{task_dir.name}: metadata.json missing"]
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{task_dir.name}: metadata.json is invalid: {exc}"]
    lineage = spec.get("lineage", {})
    errors: list[str] = []
    if metadata.get("task_id") != spec.get("task_id"):
        errors.append(f"{task_dir.name}: metadata task_id does not match task.yaml")
    if metadata.get("track") != spec.get("track") or metadata.get("family") != spec.get("family"):
        errors.append(f"{task_dir.name}: metadata track/family does not match task.yaml")
    metadata_lineage = metadata.get("lineage", {})
    for key in ("source", "mutation_template_id"):
        if metadata_lineage.get(key) != lineage.get(key):
            errors.append(f"{task_dir.name}: metadata lineage.{key} does not match task.yaml")
    if metadata.get("difficulty") != spec.get("difficulty_tier"):
        errors.append(f"{task_dir.name}: metadata difficulty does not match difficulty_tier")
    for key in ("family_id", "anchor_instance_id"):
        if spec.get(key) and metadata.get(key) != spec.get(key):
            errors.append(f"{task_dir.name}: metadata {key} does not match task.yaml")
    return errors


def _isolated_validate_task(task_dir: Path) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "benchmark.harness.cli", "validate-task", str(task_dir)],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return []
    output = (completed.stderr or completed.stdout).strip()
    return [output or "isolated validate-task failed"]


def _compile_projection_findings(task_dir: Path, spec: dict[str, Any]) -> list[str]:
    """Check that compile task metadata reaches the executable benchmark."""
    if str(spec.get("task_id")) not in {
        "CORE-COMPILE-RECOMPILE-04",
        "CORE-COMPILE-DYNAMIC-11",
        "CORE-COMPILE-TINY-12",
    }:
        return []
    errors: list[str] = []
    params = spec.get("family_parameters", {})
    measurement = spec.get("measurement", {})
    if int(measurement.get("measured_iterations", -1)) != int(params.get("logical_steps", -2)):
        errors.append(f"{task_dir.name}: measured_iterations must equal family logical_steps")
    if str(measurement.get("primary_metric")) != "schedule_wall_ms":
        errors.append(f"{task_dir.name}: compile primary metric must be schedule_wall_ms")
    if str(spec.get("kind")) == "positive" and int(measurement.get("repetitions", 0)) < 5:
        errors.append(f"{task_dir.name}: positive compile anchor requires at least five repetitions")
    try:
        import importlib.util
        module_spec = importlib.util.spec_from_file_location(f"compile_profile_{task_dir.name}", task_dir / "benchmark.py")
        if module_spec is None or module_spec.loader is None:
            raise ImportError("benchmark module unavailable")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        profile = dict(getattr(module, "compile_profile"))
        fixtures = module.make_fixtures(0)
        projected = fixtures["model_config"]
        if int(profile.get("logical_steps", -1)) != int(params.get("logical_steps", -2)):
            errors.append(f"{task_dir.name}: benchmark logical_steps projection disagrees with FamilySpec")
        if int(profile.get("graph_size", -1)) != int(params.get("graph_size", -2)):
            errors.append(f"{task_dir.name}: benchmark graph_size projection disagrees with FamilySpec")
        expected_graph_size = int(projected["hidden_dim"]) * (int(projected["num_blocks"]) + 1)
        if expected_graph_size != int(params.get("graph_size", -2)):
            errors.append(f"{task_dir.name}: executable model graph size does not equal declared graph_size")
    except Exception as exc:
        errors.append(f"{task_dir.name}: compile executable projection unavailable: {exc}")
    return errors


def _empirical_flags(
    specs: list[dict[str, Any]], empirical_path: Path | None, duplicate_findings: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Translate measured pilot evidence into replacement/review flags.

    The validator remains structural when no measurements are supplied.  In
    that case the report explicitly stays ``pending`` rather than treating
    declared oracle ranges as empirical evidence.
    """
    flags = {name: [] for name in EMPIRICAL_FLAGS}
    for finding in duplicate_findings:
        if finding.get("type") in {"same_fix_pattern", "exact"}:
            flags["repair_pattern_duplicate"].extend(str(item) for item in finding.get("task_ids", []))
    for spec in specs:
        if spec.get("track") != "evolution" and not spec.get("scientific_gates"):
            flags["semantic_gate_too_weak"].append(str(spec.get("task_id")))
    for name in flags:
        flags[name] = sorted(set(flags[name]))
    if empirical_path is None:
        return flags, {"status": "pending", "source": None, "hard_flags": [], "calibration_gate": "blocked"}

    try:
        payload = json.loads(empirical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return flags, {
            "status": "invalid",
            "source": str(empirical_path),
            "hard_flags": [f"empirical input unreadable: {exc}"],
            "calibration_gate": "blocked",
        }
    records = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return flags, {
            "status": "invalid",
            "source": str(empirical_path),
            "hard_flags": ["empirical input must contain a tasks list"],
            "calibration_gate": "blocked",
        }
    specs_by_id = {str(spec.get("task_id")): spec for spec in specs}
    seen: set[str] = set()
    records_by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id", ""))
        if task_id not in specs_by_id:
            continue
        seen.add(task_id)
        records_by_task[task_id] = dict(record)
        spec = specs_by_id[task_id]
        if "_task_dir" in spec:
            missing_fields = [key for key in CALIBRATION_FIELDS if key not in record]
            if missing_fields:
                flags["semantic_gate_too_weak"].append(task_id)
                continue
            if not isinstance(record.get("environment"), dict) or not record.get("environment"):
                flags["semantic_gate_too_weak"].append(task_id)
            if not isinstance(record.get("outer_trials"), list) or not record.get("outer_trials"):
                flags["semantic_gate_too_weak"].append(task_id)
            if not isinstance(record.get("noise_control"), dict) or not record.get("noise_control"):
                flags["noise_too_high"].append(task_id)
            if not isinstance(record.get("oracle_ci"), dict) or not record.get("oracle_ci"):
                flags["oracle_effect_unstable"].append(task_id)
            if not isinstance(record.get("semantic_gates"), dict) or not record.get("semantic_gates"):
                flags["semantic_gate_too_weak"].append(task_id)
            if not isinstance(record.get("anti_cheat"), dict) or record.get("anti_cheat", {}).get("status") not in {"pass", "passed", "clean"}:
                flags["agent_shortcut_detected"].append(task_id)
        if "_task_dir" in spec:
            task_dir = Path(spec.get("_task_dir", ""))
            if record.get("task_digest") != task_package_digest(task_dir):
                flags["agent_shortcut_detected"].append(task_id)
            if not isinstance(record.get("revision"), str) or record.get("revision") in {"", "unknown", "pending"}:
                flags["agent_shortcut_detected"].append(task_id)
        measurement = spec.get("measurement", {})
        noise_limit = float(measurement.get("noise_floor_percent", 2.0))
        min_improvement = float(measurement.get("min_improvement_percent", 5.0))
        low = record.get("oracle_ci_low_percent")
        high = record.get("oracle_ci_high_percent")
        control_noise = record.get("control_noise_percent")
        observed_noise = [
            float(value)
            for value in control_noise
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0
        ] if isinstance(control_noise, list) else []
        empirical_floor = max(min_improvement, max(observed_noise, default=noise_limit))
        if isinstance(high, (int, float)) and float(high) < empirical_floor:
            flags["oracle_effect_too_small"].append(task_id)
        elif isinstance(low, (int, float)) and float(low) <= empirical_floor:
            flags["oracle_effect_unstable"].append(task_id)
        baseline = record.get("baseline_speedups")
        if isinstance(baseline, list) and baseline and all(
            isinstance(value, (int, float)) and float(value) <= 1.0 + empirical_floor / 100.0 for value in baseline
        ):
            flags["baseline_already_optimal"].append(task_id)
        if observed_noise and isinstance(low, (int, float)) and max(observed_noise) >= float(low):
            flags["noise_too_high"].append(task_id)
        gate_rate = record.get("semantic_gate_pass_rate")
        if isinstance(gate_rate, (int, float)) and float(gate_rate) < 1.0:
            flags["semantic_gate_too_weak"].append(task_id)
        elif not spec.get("scientific_gates"):
            flags["semantic_gate_too_weak"].append(task_id)
        effects = record.get("platform_effects")
        if isinstance(effects, dict):
            numeric = [float(value) for value in effects.values() if isinstance(value, (int, float))]
            if numeric and min(numeric) < 0 < max(numeric):
                flags["platform_direction_flip"].append(task_id)
        if record.get("agent_shortcut_detected") is True:
            flags["agent_shortcut_detected"].append(task_id)
        difficulty = record.get("difficulty_score")
        if isinstance(difficulty, (int, float)):
            if float(difficulty) >= 0.8:
                flags["difficulty_ceiling"].append(task_id)
            elif float(difficulty) <= 0.2:
                flags["difficulty_floor"].append(task_id)
    for name in flags:
        flags[name] = sorted(set(flags[name]))
    hard_flags = [name for name, task_ids in flags.items() if task_ids]
    missing = sorted(set(specs_by_id) - seen)
    if missing:
        hard_flags.append("missing_empirical_task_records")
    return flags, {
        "status": "observed",
        "source": str(empirical_path),
        "records": len(records),
        "missing_task_ids": missing,
        "hard_flags": sorted(set(hard_flags)),
        "calibration_gate": "blocked" if hard_flags else "ready_for_review",
        "records_by_task": records_by_task,
    }


def build_report(tasks_root: str | Path, empirical_path: str | Path | None = None, manifest_path: str | Path | None = None) -> tuple[dict[str, Any], list[str]]:
    tasks_root = Path(tasks_root)
    errors: list[str] = []
    try:
        active_manifest = load_active_manifest(tasks_root, manifest_path)
    except ValueError as exc:
        return {"population_id": "SPE-EvoBench-v1.0-30-pilot", "num_tasks": 0}, [str(exc)]
    active_task_ids = [str(item) for item in active_manifest["task_ids"]]
    active_set = set(active_task_ids)
    specs: list[dict[str, Any]] = []
    for task_id in active_task_ids:
        task_dir = tasks_root / task_id
        if not task_dir.is_dir() or not (task_dir / "task.yaml").is_file():
            errors.append(f"active task package missing: {task_id}")
            continue
        try:
            spec = miniyaml.load(str(task_dir / "task.yaml"))
        except Exception as exc:
            errors.append(f"{task_dir.name}: task.yaml parse failed: {exc}")
            continue
        spec["_task_dir"] = task_dir
        specs.append(spec)
        errors.extend(_metadata_findings(task_dir, spec))
        errors.extend(_compile_projection_findings(task_dir, spec))
        public_context = spec.get("public_context")
        if not isinstance(public_context, dict) or not isinstance(public_context.get("workload"), dict) or not public_context.get("workload"):
            errors.append(f"{task_dir.name}: missing explicit public routing context")
        elif isinstance(spec.get("family_parameters"), dict) and public_context.get("workload") != spec.get("family_parameters"):
            errors.append(f"{task_dir.name}: public_context.workload must equal family_parameters")
        if not spec.get("family_id"):
            errors.append(f"{task_dir.name}: missing family_id anchor projection")
        elif not spec.get("anchor_instance_id"):
            errors.append(f"{task_dir.name}: missing anchor_instance_id")
        if spec.get("track") != "evolution":
            for key in ATOMIC_REQUIRED:
                if not spec.get(key):
                    errors.append(f"{task_dir.name}: missing atomic metadata {key}")
            if spec.get("difficulty_tier") not in {"easy", "medium", "hard"}:
                errors.append(f"{task_dir.name}: invalid difficulty_tier")
            if spec.get("family_id"):
                try:
                    family_id = resolve_family_id(str(spec["family_id"]))
                    anchor_id = str(spec.get("anchor_instance_id") or spec.get("task_id"))
                    if anchor_id not in FAMILY_SPECS[family_id].anchors:
                        errors.append(f"{task_dir.name}: task is not a declared anchor of family {family_id}")
                    else:
                        expected_contract = FAMILY_SPECS[family_id].policy_spec().policy_id
                        if str(spec.get("scientific_contract_id", "")) != expected_contract:
                            errors.append(f"{task_dir.name}: scientific_contract_id must match FamilySpec policy {expected_contract}")
                        instance = reconstruct_anchor_instance(anchor_id, family_id)
                        declared_parameters = spec.get("family_parameters")
                        if not isinstance(declared_parameters, dict) or dict(declared_parameters) != dict(instance.parameters):
                            errors.append(f"{task_dir.name}: family_parameters do not match FamilySpec anchor")
                        expected_digest = family_instance_digest(family_id, instance.parameters)
                        if str(spec.get("family_instance_digest", "")) != expected_digest:
                            errors.append(f"{task_dir.name}: family_instance_digest does not match FamilySpec anchor")
                        declared_kind = str(spec.get("kind", ""))
                        expected_kind = "positive" if instance.applicable else "counterexample"
                        if declared_kind in {"positive", "counterexample"} and declared_kind != expected_kind:
                            errors.append(f"{task_dir.name}: task polarity disagrees with FamilySpec applicability")
                except KeyError:
                    errors.append(f"{task_dir.name}: unknown family_id {spec.get('family_id')}")
            errors.extend(_artifact_findings(task_dir, spec))
            actual_hash = ast_skeleton_hash(task_dir)
            if spec.get("workspace_ast_skeleton_hash") != actual_hash:
                errors.append(f"{task_dir.name}: workspace_ast_skeleton_hash is stale")
            if int(spec.get("ast_skeleton_version", 0)) != 2:
                errors.append(f"{task_dir.name}: ast_skeleton_version must be 2")
        errors.extend(f"{task_dir.name}: {item}" for item in _isolated_validate_task(task_dir))

    counts = Counter(str(spec.get("track")) for spec in specs)
    if len(specs) != len(active_task_ids):
        errors.append(f"active population package count mismatch: expected {len(active_task_ids)}, found {len(specs)}")
    for track, expected in EXPECTED_COUNTS.items():
        if counts[track] != expected:
            errors.append(f"{track} count must be {expected}, found {counts[track]}")

    exact_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    pattern_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    lineage_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for spec in specs:
        task_id = str(spec["task_id"])
        exact_groups[(
            str(spec.get("generator_family_id")), str(spec.get("oracle_fix_pattern_id")),
            str(spec.get("scientific_contract_id")), str(spec.get("workspace_ast_skeleton_hash")),
        )].append(task_id)
        pattern_groups[(str(spec.get("oracle_fix_pattern_id")), str(spec.get("scientific_contract_id")))].append(task_id)
        lineage = spec.get("lineage", {})
        lineage_groups[(str(lineage.get("source")), str(lineage.get("mutation_template_id")))].append(task_id)
    duplicate_findings: list[dict[str, Any]] = []
    for key, ids in exact_groups.items():
        if len(ids) > 1:
            duplicate_findings.append({"type": "exact", "key": key, "task_ids": ids})
            errors.append(f"exact duplicate population lineage: {ids}")
    for key, ids in pattern_groups.items():
        if len(ids) > 1:
            duplicate_findings.append({"type": "same_fix_pattern", "key": key, "task_ids": ids})
            errors.append(f"same oracle fix pattern across tasks: {ids}")
    for key, ids in lineage_groups.items():
        if len(ids) > 1:
            duplicate_findings.append({"type": "lineage", "key": key, "task_ids": ids})
            errors.append(f"explicit lineage leak: {ids}")

    empirical_flags, empirical_calibration = _empirical_flags(
        specs, Path(empirical_path) if empirical_path is not None else None, duplicate_findings
    )
    task_calibration: dict[str, Any] = {}
    retired_tasks: list[dict[str, str]] = []
    for spec in specs:
        metadata_path = Path(spec["_task_dir"]) / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        calibration = metadata.get("calibration")
        if isinstance(calibration, dict):
            task_calibration[str(spec.get("task_id"))] = calibration
        if metadata.get("retired_for_formal"):
            retired_tasks.append({
                "task_id": str(spec.get("task_id")),
                "replacement_task_id": str(metadata.get("replacement_task_id", "")),
                "reason": str(metadata.get("retirement_reason", "")),
            })

    report = {
        "schema_version": 1,
        "population_id": str(active_manifest.get("population_id", "SPE-EvoBench-v1.0-30-pilot")),
        "active_manifest": str((Path(manifest_path) if manifest_path else tasks_root.parent / "pilot_population.json").as_posix()),
        "active_task_ids": active_task_ids,
        "num_tasks": len(specs),
        "track_counts": dict(counts),
        "mechanism_distribution": dict(Counter(str(spec.get("mechanism")) for spec in specs)),
        "family_distribution": dict(Counter(str(spec.get("family")) for spec in specs)),
        "family_id_distribution": dict(Counter(str(spec.get("family_id", spec.get("family"))) for spec in specs)),
        "anchor_projection": {
            str(spec.get("family_id")): sorted(
                [str(item.get("task_id")) for item in specs if item.get("anchor_instance_id") and item.get("family_id") == spec.get("family_id")]
            )
            for spec in specs if spec.get("family_id") and spec.get("anchor_instance_id")
        },
        "polarity_distribution": dict(Counter(str(spec.get("kind")) for spec in specs)),
        "difficulty_distribution": dict(Counter(str(spec.get("difficulty_tier")) for spec in specs)),
        "duplicate_near_duplicate_findings": duplicate_findings,
        "oracle_effect_distribution": [spec.get("oracle", {}).get("expected_speedup_range") for spec in specs],
        "baseline_noise_distribution_percent": [spec.get("measurement", {}).get("noise_floor_percent") for spec in specs],
        "runtime_cost_distribution": [
            {"task_id": spec.get("task_id"), "time_budget_s": spec.get("time_budget_s"), "measured_iterations": spec.get("measurement", {}).get("measured_iterations"), "repetitions": spec.get("measurement", {}).get("repetitions")}
            for spec in specs
        ],
        "semantic_gate_failures": [spec.get("task_id") for spec in specs if not spec.get("scientific_gates")],
        "positive_counterexample_balance": {
            "positive": sum(spec.get("kind") == "positive" for spec in specs if spec.get("track") != "evolution"),
            "counterexample_or_do_not_apply": sum(spec.get("kind") in {"counterexample", "do_not_apply"} for spec in specs if spec.get("track") != "evolution"),
        },
        "lineage_leakage_checked": True,
        "empirical_calibration": empirical_calibration,
        "task_calibration": task_calibration,
        "retired_for_formal": sorted(retired_tasks, key=lambda item: item["task_id"]),
        "public_context": {
            str(spec.get("task_id")): spec.get("public_context")
            for spec in specs
        },
        "empirical_rejection_flags": empirical_flags,
        "formal_50_task_results": "not_claimed",
    }
    return report, errors


def build_pilot_calibration(report: dict[str, Any], tasks_root: str | Path) -> dict[str, Any]:
    """Materialize one auditable calibration view for every pilot task.

    This is deliberately a projection of task-local evidence; it never turns
    pending or blocked measurements into eligibility.  The artifact is the
    single input to human approval before any sealed population is generated.
    """
    root = Path(tasks_root)
    tasks: list[dict[str, Any]] = []
    active_manifest = load_active_manifest(root)
    task_ids = [str(item) for item in active_manifest["task_ids"]]
    for task_id in task_ids:
        task_dir = root / task_id
        task_path = task_dir / "task.yaml"
        try:
            spec = miniyaml.load(str(task_path))
        except Exception:
            spec = {}
        metadata: dict[str, Any] = {}
        try:
            metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        local = report.get("task_calibration", {}).get(task_id, {})
        empirical = report.get("empirical_calibration", {}).get("records_by_task", {}).get(task_id, {})
        # Metadata is descriptive only.  Eligibility comes from a complete,
        # digest-attested empirical record for the active task.
        status = "eligible" if isinstance(empirical, dict) and empirical and not any(
            task_id in ids for ids in report.get("empirical_rejection_flags", {}).values()
        ) and report.get("empirical_calibration", {}).get("status") == "observed" else "pending"
        if metadata.get("retired_for_formal"):
            status = "blocked"
        if task_id in set(report.get("empirical_calibration", {}).get("missing_task_ids", [])):
            status = "pending"
        flags = sorted({flag for flag, ids in report.get("empirical_rejection_flags", {}).items() if task_id in ids})
        if metadata.get("retired_for_formal"):
            flags.append("retired_for_formal")
        if flags or task_id in set(report.get("semantic_gate_failures", [])):
            status = "blocked"
        tasks.append({
            "task_id": task_id,
            "task_digest": task_package_digest(task_dir) if task_path.is_file() else None,
            "revision": empirical.get("revision", "unknown") if isinstance(empirical, dict) else "unknown",
            "environment": empirical.get("environment", {}) if isinstance(empirical, dict) else {},
            "outer_trials": empirical.get("outer_trials", []) if isinstance(empirical, dict) else [],
            "noise_control": empirical.get("noise_control", {}) if isinstance(empirical, dict) else {},
            "oracle_ci": empirical.get("oracle_ci", {}) if isinstance(empirical, dict) else {},
            "semantic_gates": empirical.get("semantic_gates", {"declared": list(spec.get("scientific_gates", [])), "failures": []}) if isinstance(empirical, dict) else {"declared": list(spec.get("scientific_gates", [])), "failures": []},
            "anti_cheat": empirical.get("anti_cheat", {}) if isinstance(empirical, dict) else {},
            "empirical_evidence": {key: value for key, value in empirical.items() if not str(key).startswith("_")} if isinstance(empirical, dict) else {},
            "eligibility": status == "eligible",
            "status": status,
            "block_reason": flags or ([] if status == "eligible" else ["calibration_pending"]),
        })
    artifact = {
        "schema_version": 1,
        "population_id": report.get("population_id"),
        "active_task_ids": task_ids,
        "population_report_digest": _json_digest(report),
        "calibration_gate": report.get("empirical_calibration", {}).get("calibration_gate", "blocked"),
        "tasks": tasks,
        "formal_50_generation": "withheld",
    }
    artifact["artifact_digest"] = _json_digest(artifact)
    return artifact


def validate_formal_readiness(report: dict[str, Any], calibration: dict[str, Any] | None, approval: dict[str, Any] | None) -> list[str]:
    """Fail closed for formal generation/campaign entry points."""
    from benchmark.formal.approval import validate_calibration_approval
    errors: list[str] = []
    if report.get("empirical_calibration", {}).get("calibration_gate") != "ready_for_review":
        errors.append("empirical calibration gate is not ready_for_review")
    if report.get("semantic_gate_failures"):
        errors.append("semantic_gate_failures are present")
    if any(report.get("empirical_rejection_flags", {}).values()):
        errors.append("empirical hard flags are present")
    if calibration is None or calibration.get("calibration_gate") != "ready_for_review":
        errors.append("pilot_calibration artifact is missing or blocked")
    errors.extend(validate_calibration_approval(report, calibration, approval))
    calibration_tasks = list((calibration or {}).get("tasks", []))
    active_ids = set(str(item) for item in report.get("active_task_ids", []))
    observed_ids = {str(item.get("task_id")) for item in calibration_tasks}
    if observed_ids != active_ids:
        errors.append("pilot calibration does not cover exactly the active population")
    if any(item.get("status") != "eligible" or item.get("eligibility") is not True for item in calibration_tasks):
        errors.append("every active pilot task must be eligible")
    if report.get("retired_for_formal") and any(str(item.get("task_id")) in active_ids for item in report.get("retired_for_formal", [])):
        errors.append("retired task remains in active population")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=Path(__file__).resolve().parents[1] / "tasks")
    parser.add_argument("--population-manifest", type=Path, default=None, help="explicit active population manifest")
    parser.add_argument("--split", type=Path, default=Path(__file__).resolve().parents[1] / "split" / "sequential.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--empirical", type=Path, default=None, help="measured pilot JSON; omitted means calibration pending")
    parser.add_argument("--strict-formal", action="store_true", help="fail closed unless pilot calibration and approval are ready")
    parser.add_argument("--pilot-calibration", type=Path, default=None, help="write unified pilot_calibration.json")
    parser.add_argument("--approval", type=Path, default=None, help="calibration_approval.json to validate in --strict-formal mode")
    args = parser.parse_args()
    if args.strict_formal:
        if args.empirical is None:
            errors = ["--strict-formal requires an explicit --empirical artifact"]
            print(json.dumps({"errors": errors}, ensure_ascii=False))
            return 1
        report_path = args.out or (args.tasks_root.parent / "population_report.json")
        calibration_path = args.pilot_calibration or report_path.with_name("pilot_calibration.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            approval_path = args.approval or report_path.with_name("calibration_approval.json")
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report, calibration, approval = {}, None, None
            errors = [f"read-only formal readiness inputs unavailable: {exc}"]
        errors.extend(validate_formal_readiness(report, calibration, approval))
    else:
        report, errors = build_report(args.tasks_root, args.empirical, args.population_manifest)
        split_errors = check_leakage(args.split, args.tasks_root)
        errors.extend(split_errors)
        report["split_leakage_findings"] = split_errors
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        calibration_path = args.pilot_calibration
        if calibration_path is None and args.out:
            calibration_path = args.out.with_name("pilot_calibration.json")
        calibration = build_pilot_calibration(report, args.tasks_root)
        if calibration_path:
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"population": report.get("population_id"), "num_tasks": report.get("num_tasks"), "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
