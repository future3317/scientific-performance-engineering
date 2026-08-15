#!/usr/bin/env python3
"""Validate and summarize the v1.0-20 population without running long campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml
from benchmark.harness.split import check_leakage
from benchmark.taskgen.generate import ast_skeleton_hash
from benchmark.families import resolve_family_id
from benchmark.families.catalog import FAMILY_SPECS


ATOMIC_REQUIRED = (
    "generator_family_id", "oracle_fix_pattern_id", "scientific_contract_id",
    "workspace_ast_skeleton_hash", "difficulty_tier",
)
EXPECTED_COUNTS = {"spe_core": 11, "sciml": 7, "evolution": 2}
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
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id", ""))
        if task_id not in specs_by_id:
            continue
        seen.add(task_id)
        spec = specs_by_id[task_id]
        measurement = spec.get("measurement", {})
        noise_limit = float(measurement.get("noise_floor_percent", 2.0))
        min_improvement = float(measurement.get("min_improvement_percent", 5.0))
        threshold = max(noise_limit, min_improvement)
        low = record.get("oracle_ci_low_percent")
        high = record.get("oracle_ci_high_percent")
        if isinstance(high, (int, float)) and high < threshold:
            flags["oracle_effect_too_small"].append(task_id)
        elif isinstance(low, (int, float)) and low < threshold:
            flags["oracle_effect_unstable"].append(task_id)
        baseline = record.get("baseline_speedups")
        if isinstance(baseline, list) and baseline and all(
            isinstance(value, (int, float)) and float(value) <= 1.0 + noise_limit / 100.0 for value in baseline
        ):
            flags["baseline_already_optimal"].append(task_id)
        control_noise = record.get("control_noise_percent")
        if isinstance(control_noise, list) and control_noise and any(
            isinstance(value, (int, float)) and float(value) > noise_limit for value in control_noise
        ):
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
    }


def build_report(tasks_root: str | Path, empirical_path: str | Path | None = None) -> tuple[dict[str, Any], list[str]]:
    tasks_root = Path(tasks_root)
    errors: list[str] = []
    specs: list[dict[str, Any]] = []
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.yaml").is_file():
            continue
        try:
            spec = miniyaml.load(str(task_dir / "task.yaml"))
        except Exception as exc:
            errors.append(f"{task_dir.name}: task.yaml parse failed: {exc}")
            continue
        spec["_task_dir"] = task_dir
        specs.append(spec)
        errors.extend(_metadata_findings(task_dir, spec))
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
                    if spec.get("anchor_instance_id") and spec["anchor_instance_id"] == spec.get("task_id") and spec["task_id"] not in FAMILY_SPECS[family_id].anchors:
                        errors.append(f"{task_dir.name}: task is not a declared anchor of family {family_id}")
                except KeyError:
                    errors.append(f"{task_dir.name}: unknown family_id {spec.get('family_id')}")
            errors.extend(_artifact_findings(task_dir, spec))
            actual_hash = ast_skeleton_hash(task_dir)
            if spec.get("workspace_ast_skeleton_hash") != actual_hash:
                errors.append(f"{task_dir.name}: workspace_ast_skeleton_hash is stale")
        errors.extend(f"{task_dir.name}: {item}" for item in _isolated_validate_task(task_dir))

    counts = Counter(str(spec.get("track")) for spec in specs)
    if len(specs) != 20:
        errors.append(f"population must contain exactly 20 tasks, found {len(specs)}")
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

    report = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-20-pilot",
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
        "empirical_rejection_flags": empirical_flags,
        "formal_50_task_results": "not_claimed",
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=Path(__file__).resolve().parents[1] / "tasks")
    parser.add_argument("--split", type=Path, default=Path(__file__).resolve().parents[1] / "split" / "sequential.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--empirical", type=Path, default=None, help="measured pilot JSON; omitted means calibration pending")
    args = parser.parse_args()
    report, errors = build_report(args.tasks_root, args.empirical)
    split_errors = check_leakage(args.split, args.tasks_root)
    errors.extend(split_errors)
    report["split_leakage_findings"] = split_errors
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"population": report["population_id"], "num_tasks": report["num_tasks"], "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
