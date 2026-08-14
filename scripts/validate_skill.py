#!/usr/bin/env python3
"""Validate this skill's structure, metadata, links, assets, and Python syntax."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FRONTMATTER_KEYS = {"name", "description"}
REQUIRED_SCRIPTS = {
    "collect_env.py",
    "compare_benchmarks.py",
    "run_with_gpu_monitor.py",
    "validate_skill.py",
}
REQUIRED_REFERENCES = {
    "CRYSTAL_GENERATION.md",
    "EQUIVARIANT_OPERATOR_DESIGN.md",
    "GNN_PREDICTION_WORKLOADS.md",
    "MEASUREMENT_CONTRACT.md",
    "PATCH_PATTERNS.md",
    "PERFORMANCE_PLAYBOOK.md",
    "REPOSITORY_NOTES.md",
    "SOURCES.md",
    "TECHNOLOGY_MATRIX.md",
}
REQUIRED_ASSETS = {
    "benchmark_record.json",
    "materials_gnn_checks.py",
    "performance_report.md",
}
REQUIRED_MAIN_ROUTES = {
    "PERFORMANCE_PLAYBOOK.md",
    "GNN_PREDICTION_WORKLOADS.md",
    "CRYSTAL_GENERATION.md",
    "EQUIVARIANT_OPERATOR_DESIGN.md",
}
REQUIRED_CORE_MARKERS = {
    "five explicit phases",
    "hypothesis card",
    "three levels",
    "Amdahl ceiling",
    "forward and gradient agreement",
    "experiment",
    "unsupported hypothetical cases are non-blocking",
    "timing bucket audit",
    "data_ready",
    "task census",
    "Stochastic thinning",
    "logical update",
    "host contention",
    "Do not use for CUDA/runtime correctness bugs",
    "one independently attributable intervention",
    "Reachable correctness, deadlock, OOM, fallback, reproducibility, API, or scientific-quality risks",
}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"duplicate frontmatter key: {key}")
        values[key] = value.strip().strip('"')
    return values, text[end + 5 :]


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_openai_yaml(root: Path, name: str) -> None:
    path = root / "agents" / "openai.yaml"
    text = path.read_text(encoding="utf-8")
    if not text.startswith("interface:\n"):
        raise ValueError("agents/openai.yaml must start with interface:")

    values: dict[str, str] = {}
    for key in ("display_name", "short_description", "default_prompt"):
        match = re.search(rf'^  {key}: "([^"]+)"\s*$', text, re.MULTILINE)
        if not match:
            raise ValueError(f"agents/openai.yaml needs a quoted interface.{key}")
        values[key] = match.group(1)
    if not 25 <= len(values["short_description"]) <= 64:
        raise ValueError("interface.short_description must contain 25-64 characters")
    if f"${name}" not in values["default_prompt"]:
        raise ValueError(f"interface.default_prompt must mention ${name}")


def validate_benchmark_asset(root: Path) -> None:
    path = root / "assets" / "benchmark_record.json"
    record = require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    if record.get("schema_version") != 3:
        raise ValueError("benchmark_record.json schema_version must be 3")
    for key in ("identity", "hardware", "software", "contract", "candidate", "work", "metrics", "quality"):
        require_mapping(record.get(key), f"benchmark_record.json.{key}")
    contract = record["contract"]
    for key in (
        "scientific_contract_id",
        "data_manifest_or_hash",
        "seeds",
        "precision_policy",
        "stochastic_thinning",
        "checkpoint_state_contract",
        "gradient_clipping_contract",
        "quality_gates",
        "numerical_tolerances",
    ):
        if key not in contract:
            raise ValueError(f"benchmark_record.json.contract.{key} is required")
    identity = record["identity"]
    for key in ("base_revision", "benchmark_harness_hash", "candidate_patch_hash", "declared_change_set"):
        if key not in identity:
            raise ValueError(f"benchmark_record.json.identity.{key} is required")
    hardware = record["hardware"]
    if "gpu_uuid" not in hardware:
        raise ValueError("benchmark_record.json.hardware.gpu_uuid is required")
    candidate = record["candidate"]
    for key in (
        "hypothesis",
        "measured_bottleneck_share",
        "changed_levers",
        "expected_metric_movement",
        "semantic_risk",
        "falsification_test",
        "reference_output",
        "amdahl_ceiling",
        "explicitly_authorized_algorithmic_changes",
        "active_path_evidence",
    ):
        if key not in candidate:
            raise ValueError(f"benchmark_record.json.candidate.{key} is required")
    work = record["work"]
    for key in (
        "optimization_objective",
        "benchmark_levels",
        "logical_update_definition",
        "task_composition",
        "timing_bucket_definition",
    ):
        if key not in work:
            raise ValueError(f"benchmark_record.json.work.{key} is required")
    for key in ("cuda_timing_proof", "timing_buckets", "unaccounted_ratio"):
        if key not in work:
            raise ValueError(f"benchmark_record.json.work.{key} is required")
    step_audit = record.get("step_audit")
    require_mapping(step_audit, "benchmark_record.json.step_audit")
    for key in (
        "auxiliary_forward_calls",
        "autograd_grad_calls",
        "skipped_task_calls",
        "unaccounted_step_ms_p50",
        "host_load_average",
        "host_available_memory_mb",
        "host_swap_percent",
        "worker_rss_mb",
    ):
        if key not in step_audit:
            raise ValueError(f"benchmark_record.json.step_audit.{key} is required")
    acceptance = require_mapping(record.get("acceptance"), "benchmark_record.json.acceptance")
    for key in (
        "primary_metric",
        "higher_is_better",
        "minimum_improvement_percent",
        "noise_floor_percent",
        "confidence_level",
        "bootstrap_samples",
        "minimum_runs",
        "required_quality_gates",
        "max_unaccounted_ratio",
    ):
        if key not in acceptance:
            raise ValueError(f"benchmark_record.json.acceptance.{key} is required")
    measurements = require_mapping(record.get("measurements"), "benchmark_record.json.measurements")
    for key in ("run_order", "runs"):
        if key not in measurements:
            raise ValueError(f"benchmark_record.json.measurements.{key} is required")


def validate_resources(root: Path) -> None:
    references = root / "references"
    assets = root / "assets"
    missing_references = sorted(name for name in REQUIRED_REFERENCES if not (references / name).is_file())
    if missing_references:
        raise ValueError(f"missing required references: {', '.join(missing_references)}")
    missing_assets = sorted(name for name in REQUIRED_ASSETS if not (assets / name).is_file())
    if missing_assets:
        raise ValueError(f"missing required assets: {', '.join(missing_assets)}")
    compile((assets / "materials_gnn_checks.py").read_text(encoding="utf-8"), "materials_gnn_checks.py", "exec")


def validate_main_routes(body: str) -> None:
    missing = sorted(name for name in REQUIRED_MAIN_ROUTES if name not in body)
    if missing:
        raise ValueError(f"SKILL.md must route required modules: {', '.join(missing)}")
    folded_body = body.casefold()
    missing_markers = sorted(marker for marker in REQUIRED_CORE_MARKERS if marker.casefold() not in folded_body)
    if missing_markers:
        raise ValueError(f"SKILL.md missing core workflow markers: {', '.join(missing_markers)}")


def validate_links(root: Path) -> None:
    for markdown in root.rglob("*.md"):
        contents = markdown.read_text(encoding="utf-8")
        if len(contents.splitlines()) > 100 and "## Contents" not in contents:
            raise ValueError(f"long reference needs a Contents section: {markdown}")
        for target in LINK_RE.findall(contents):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative = target.split("#", 1)[0].strip("<>")
            resolved = (markdown.parent / relative).resolve()
            if not resolved.exists():
                raise ValueError(f"broken local link in {markdown}: {target}")


def validate_python(root: Path) -> None:
    scripts = root / "scripts"
    present = {path.name for path in scripts.glob("*.py")}
    missing = sorted(REQUIRED_SCRIPTS - present)
    if missing:
        raise ValueError(f"missing required scripts: {', '.join(missing)}")
    for script in scripts.glob("*.py"):
        compile(script.read_text(encoding="utf-8"), str(script), "exec")


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    skill_path = root / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(text)
    if set(metadata) != FRONTMATTER_KEYS:
        raise ValueError(f"frontmatter keys must be exactly {sorted(FRONTMATTER_KEYS)}")
    name = metadata["name"]
    description = metadata["description"]
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid skill name: {name!r}")
    if root.name != name:
        raise ValueError(f"skill folder {root.name!r} must match name {name!r}")
    if not description or len(description) > 1024:
        raise ValueError("description must contain 1-1024 characters")
    if len(body.splitlines()) > 500:
        raise ValueError("SKILL.md body must not exceed 500 lines")

    validate_main_routes(body)
    validate_openai_yaml(root, name)
    validate_benchmark_asset(root)
    validate_resources(root)
    validate_links(root)
    validate_python(root)
    print(f"valid: {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        raise SystemExit(1)
