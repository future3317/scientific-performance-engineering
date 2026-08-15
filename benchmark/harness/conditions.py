#!/usr/bin/env python3
"""Experimental conditions A-D materialization (BENCHMARK_DESIGN.md section 9).

Each condition builds an isolated skill copy from a pinned snapshot with the
appropriate read-only/writable bits and an injection policy, then hash-attests
the result so a run can prove which skill bits were visible:

- A (no-skill):        empty directory.
- B (frozen-skill):    read-only copy; experience/evolution machinery disabled.
- C (append-only):     copy with ``experience/inbox/`` writable; injection
                       policy = any inbox record is eligible for injection.
- D (governed):        copy with the full pipeline dirs; injection policy =
                       only canonical rules in ``rules/`` + ``registry/rules.json``.

The core skill has no runtime retrieval interface (INTEGRATION_REQUIREMENTS.md
R1), so the harness assembles and attests the condition store itself.
"""

from __future__ import annotations

import json
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import anticheat

CONDITIONS = ("A", "B", "C", "D")

# Pipeline dirs that must exist in the governed (D) store, per the core layout.
PIPELINE_DIRS = (
    "experience/inbox",
    "experience/cases",
    "experience/archive",
    "experience/usage",
    "evolution/candidates",
    "evolution/conflicts",
    "evolution/maintenance_reports",
    "evolution/retired",
    "rules",
    "registry",
    "tests/rule_cases",
)

INJECTION_POLICIES = {
    "A": {"mode": "none", "description": "no skill visible"},
    "B": {"mode": "frozen", "description": "initial snapshot only; no experience or rules injected"},
    "C": {"mode": "inbox_any", "description": "any record in experience/inbox/ is eligible for injection"},
    "D": {"mode": "canonical_only", "description": "only canonical rules in rules/ + registry/rules.json are eligible"},
}


def _make_read_only(path: Path) -> None:
    for entry in sorted(path.rglob("*")):
        if entry.is_dir():
            entry.chmod(stat.S_IRUSR | stat.S_IXUSR)
        elif entry.is_file() and not entry.is_symlink():
            entry.chmod(stat.S_IRUSR)
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)


def _attest(out_dir: Path, condition: str, policy: dict[str, str], snapshot_dir: Path | None) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "condition": condition,
        "injection_policy": policy,
        "source_snapshot": str(snapshot_dir) if snapshot_dir else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": anticheat.hash_tree(out_dir),
    }
    (out_dir / "condition_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def materialize_condition(
    condition: str, snapshot_dir: str | Path | None, out_dir: str | Path
) -> dict[str, Any]:
    """Build the condition store; returns the attestation manifest dict.

    *snapshot_dir* is required for B/C/D and ignored for A. Existing *out_dir*
    content is replaced.
    """
    condition = condition.upper()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    policy = dict(INJECTION_POLICIES[condition])
    if condition == "A":
        return _attest(out_dir, condition, policy, None)

    if snapshot_dir is None:
        raise ValueError(f"condition {condition} requires --snapshot DIR")
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {snapshot_dir}")
    anticheat.assert_no_vcs(snapshot_dir)
    shutil.copytree(snapshot_dir, out_dir, dirs_exist_ok=True)

    if condition == "B":
        # Attest first (writes condition_manifest.json), then lock the tree read-only.
        manifest = _attest(out_dir, condition, policy, snapshot_dir)
        _make_read_only(out_dir)
        return manifest

    if condition == "C":
        inbox = out_dir / "experience" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
    elif condition == "D":
        for rel in PIPELINE_DIRS:
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
        registry = out_dir / "registry" / "rules.json"
        if not registry.is_file():
            registry.write_text(json.dumps({"schema_version": 1, "rules": []}, indent=2) + "\n", encoding="utf-8")

    policy_file = out_dir / "injection_policy.json"
    policy_file.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return _attest(out_dir, condition, policy, snapshot_dir)


def verify_attestation(condition_dir: str | Path) -> tuple[bool, list[str]]:
    """Re-hash a condition store and compare against its attestation manifest."""
    condition_dir = Path(condition_dir)
    manifest_path = condition_dir / "condition_manifest.json"
    if not manifest_path.is_file():
        return False, ["condition_manifest.json missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = dict(manifest.get("files", {}))
    current = anticheat.hash_tree(condition_dir)
    current.pop("condition_manifest.json", None)
    return anticheat.manifests_equal(recorded, current)
