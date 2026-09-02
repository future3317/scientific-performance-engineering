"""Cheap public contract checks before the hidden verifier pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .api import validate_solution_api
from .runner import import_module_by_path


def validate_public_solution(solution_path: str | Path, task_spec: Mapping[str, Any]) -> dict[str, Any]:
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
    if not violations:
        try:
            module = import_module_by_path(path, "spe_public_preflight")
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
