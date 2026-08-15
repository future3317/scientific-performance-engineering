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
    "capture_experience.py",
    "capture_rule_usage.py",
    "experience_contract_tests.py",
    "evolution_contract_tests.py",
    "run_with_gpu_monitor.py",
    "run_rule_replay.py",
    "generate_rule_schemas.py",
    "rule_engine_tests.py",
    "evolution_statistics_tests.py",
    "assess_rule_drift.py",
    "validate_rule_os.py",
    "score_rule_library.py",
    "validate_benchmark.py",
    "validate_evolution.py",
    "validate_experience.py",
    "validate_rule_usage.py",
    "validate_skill.py",
}
REQUIRED_REFERENCES = {
    "CRYSTAL_GENERATION.md",
    "EQUIVARIANT_OPERATOR_DESIGN.md",
    "GNN_PREDICTION_WORKLOADS.md",
    "MEASUREMENT_CONTRACT.md",
    "CODE_AND_RUNTIME_AUDIT.md",
    "DATA_AND_TRAINING_LIFECYCLE.md",
    "EXPERIENCE_EVOLUTION.md",
    "MEMORY_COMPILER_DISTRIBUTED.md",
    "PATCH_PATTERNS.md",
    "PERFORMANCE_PLAYBOOK.md",
    "REPOSITORY_NOTES.md",
    "SOURCES.md",
    "TECHNOLOGY_MATRIX.md",
}
REQUIRED_ASSETS = {
    "benchmark_record.json",
    "benchmark_record.schema.json",
    "experience_record.json",
    "experience_record.schema.json",
    "rule_candidate.json",
    "rule_candidate.schema.json",
    "rule_regression_case.json",
    "rule_regression_case.schema.json",
    "rule_usage_record.json",
    "rule_usage_record.schema.json",
    "materials_gnn_checks.py",
    "performance_report.md",
    "rule_spec.schema.json",
    "evidence_event.schema.json",
    "rule_state.schema.json",
    "rule_spec.json",
    "evidence_event.json",
    "rule_state.json",
    "rule_card.json",
    "rule_card.schema.json",
}
REQUIRED_REGISTRY = {"rules.json"}
REQUIRED_MAIN_ROUTES = {
    "CODE_AND_RUNTIME_AUDIT.md",
    "DATA_AND_TRAINING_LIFECYCLE.md",
    "EXPERIENCE_EVOLUTION.md",
    "MEMORY_COMPILER_DISTRIBUTED.md",
    "PERFORMANCE_PLAYBOOK.md",
    "GNN_PREDICTION_WORKLOADS.md",
    "CRYSTAL_GENERATION.md",
    "EQUIVARIANT_OPERATOR_DESIGN.md",
}
REQUIRED_CORE_MARKERS = {
    "Preflight",
    "Contract Freeze",
    "Lifecycle Census",
    "Amortized Job",
    "Amdahl ceiling",
    "Statistical Gate",
    "active-path",
    "logical-update DAG",
    "synchronization census",
    "cache/H2D",
    "precompute",
    "campaign resource",
    "comparison_class=algorithmic",
    "paired-intervention",
    "Bayesian",
    "rate-distortion",
    "shortest useful horizon",
    "record is the executable contract",
    "inconclusive",
    "opcheck",
    "Do not use this skill for CUDA/runtime correctness bugs",
    "Unsupported hypothetical cases remain non-blocking",
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
    schema = require_mapping(json.loads((root / "assets" / "benchmark_record.schema.json").read_text(encoding="utf-8")), "benchmark_record.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("benchmark_record.schema.json must declare draft 2020-12")
    if record.get("schema_version") != schema.get("properties", {}).get("schema_version", {}).get("const"):
        raise ValueError("benchmark_record.json schema version does not match benchmark_record.schema.json")


def validate_resources(root: Path) -> None:
    references = root / "references"
    assets = root / "assets"
    registry = root / "registry"
    rules = root / "rules"
    regression_cases = root / "tests" / "rule_cases"
    missing_references = sorted(name for name in REQUIRED_REFERENCES if not (references / name).is_file())
    if missing_references:
        raise ValueError(f"missing required references: {', '.join(missing_references)}")
    missing_assets = sorted(name for name in REQUIRED_ASSETS if not (assets / name).is_file())
    if missing_assets:
        raise ValueError(f"missing required assets: {', '.join(missing_assets)}")
    missing_registry = sorted(name for name in REQUIRED_REGISTRY if not (registry / name).is_file())
    if missing_registry:
        raise ValueError(f"missing required registry files: {', '.join(missing_registry)}")
    if not rules.is_dir():
        raise ValueError("missing rules directory for canonical rule cards")
    if not regression_cases.is_dir():
        raise ValueError("missing tests/rule_cases directory for replay evidence")
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
    for module in (root / "core").glob("*.py"):
        compile(module.read_text(encoding="utf-8"), str(module), "exec")


def validate_benchmark_with_tool(root: Path) -> None:
    source = (root / "scripts" / "validate_benchmark.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "validate_benchmark", "__file__": str(root / "scripts" / "validate_benchmark.py")}
    exec(compile(source, "validate_benchmark.py", "exec"), namespace)
    record = json.loads((root / "assets" / "benchmark_record.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "assets" / "benchmark_record.schema.json").read_text(encoding="utf-8"))
    errors = namespace["validate_record"](record, schema)
    if errors:
        raise ValueError(f"benchmark_record.json failed lifecycle validation: {'; '.join(errors)}")


def validate_canonical_model_schemas(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    namespace: dict[str, Any] = {"__name__": "schema", "__file__": str(root / "core" / "schema.py")}
    exec(compile((root / "core" / "schema.py").read_text(encoding="utf-8"), "core/schema.py", "exec"), namespace)
    for name, expected in namespace["schemas"]().items():
        actual = json.loads((root / "assets" / name).read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{name} is stale; run scripts/generate_rule_schemas.py")


def validate_canonical_model_assets(root: Path) -> None:
    import importlib
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    models = importlib.import_module("core.models")
    for name, cls in (("rule_spec.json", models.RuleSpec), ("evidence_event.json", models.EvidenceEvent), ("rule_state.json", models.RuleState)):
        value = json.loads((root / "assets" / name).read_text(encoding="utf-8"))
        cls.from_dict(value)
    validator_source = (root / "scripts" / "validate_rule_os.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "validate_rule_os", "__file__": str(root / "scripts" / "validate_rule_os.py")}
    exec(compile(validator_source, "validate_rule_os.py", "exec"), namespace)
    errors = namespace["validate"](json.loads((root / "assets" / "rule_card.json").read_text(encoding="utf-8")))
    if errors:
        raise ValueError("rule_card.json failed typed validation: " + "; ".join(errors))


def validate_experience_with_tool(root: Path) -> None:
    source = (root / "scripts" / "validate_experience.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "validate_experience", "__file__": str(root / "scripts" / "validate_experience.py")}
    exec(compile(source, "validate_experience.py", "exec"), namespace)
    schema = namespace["load_schema"](root / "assets" / "experience_record.schema.json")
    record = json.loads((root / "assets" / "experience_record.json").read_text(encoding="utf-8"))
    errors = namespace["validate_record"](record, schema)
    if errors:
        raise ValueError(f"experience_record.json failed validation: {'; '.join(errors)}")


def validate_evolution_with_tool(root: Path) -> None:
    source = (root / "scripts" / "validate_evolution.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "validate_evolution", "__file__": str(root / "scripts" / "validate_evolution.py")}
    exec(compile(source, "validate_evolution.py", "exec"), namespace)
    schema = namespace["load_schema"](root / "assets" / "rule_candidate.schema.json")
    card = json.loads((root / "assets" / "rule_candidate.json").read_text(encoding="utf-8"))
    errors = namespace["validate_rule"](card, schema)
    if errors:
        raise ValueError(f"rule_candidate.json failed validation: {'; '.join(errors)}")
    regression_schema = namespace["load_schema"](root / "assets" / "rule_regression_case.schema.json")
    regression = json.loads((root / "assets" / "rule_regression_case.json").read_text(encoding="utf-8"))
    errors = namespace["validate_regression_case"](regression, regression_schema)
    if errors:
        raise ValueError(f"rule_regression_case.json failed validation: {'; '.join(errors)}")
    errors = namespace["audit"](root)
    if errors:
        raise ValueError(f"evolution audit failed: {'; '.join(errors)}")


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
    validate_benchmark_with_tool(root)
    validate_canonical_model_schemas(root)
    validate_canonical_model_assets(root)
    validate_experience_with_tool(root)
    validate_evolution_with_tool(root)
    print(f"valid: {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        raise SystemExit(1)
