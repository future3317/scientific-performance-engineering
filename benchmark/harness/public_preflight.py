"""Cheap public contract checks before the hidden verifier pipeline."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from . import runner
from .api import validate_solution_api


def _seed_policy_violations(path: Path, public_contract: Mapping[str, Any]) -> list[str]:
    policy = public_contract.get("seed_policy")
    if not isinstance(policy, Mapping) or policy.get("owner") != "harness":
        return []
    forbidden = {str(item) for item in policy.get("forbidden_calls", []) if isinstance(item, str)}
    if not forbidden:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        value: ast.AST = node.func
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        call_name = ".".join(reversed(parts))
        if call_name in forbidden:
            violations.append(
                f"harness-owned seed policy forbids candidate call {call_name} at line {node.lineno}"
            )
    return violations


def validate_public_solution(
    solution_path: str | Path,
    task_spec: Mapping[str, Any],
    *,
    check_import: bool = True,
) -> dict[str, Any]:
    """Validate only the task-declared public API and import boundary."""
    path = Path(solution_path).resolve()
    workspace = task_spec.get("workspace") if isinstance(task_spec, Mapping) else None
    if not isinstance(workspace, Mapping):
        return {"status": "fail", "violations": ["task workspace contract is missing"]}
    entrypoint = str(workspace.get("entrypoint", ""))
    api_name = str(workspace.get("api", ""))
    violations: list[str] = []
    if not entrypoint or path.name != Path(entrypoint).name:
        violations.append("candidate entrypoint does not match the public task entrypoint")
    if not api_name:
        violations.append("task workspace.api is missing")
    if not path.is_file():
        violations.append("candidate public entrypoint is missing")
    if path.is_file() and isinstance(task_spec.get("public_contract"), Mapping):
        violations.extend(_seed_policy_violations(path, task_spec["public_contract"]))
    if not violations and check_import:
        try:
            module = runner.import_module_by_path(path)
            violations.extend(validate_solution_api(module, api_name))
        except (OSError, TypeError, ValueError, ImportError, SyntaxError) as exc:
            violations.append(f"public import failed: {type(exc).__name__}: {exc}")
    return {
        "status": "pass" if not violations else "fail",
        "entrypoint": entrypoint,
        "api": api_name,
        "violations": violations,
        "scope": "public_contract_only",
    }


__all__ = ["validate_public_solution"]
