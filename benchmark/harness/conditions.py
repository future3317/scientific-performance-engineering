#!/usr/bin/env python3
"""Experimental conditions A-D materialization (BENCHMARK_DESIGN.md section 9).

Each condition builds an isolated skill copy from a pinned, rendered skill-view
bundle with the
appropriate read-only/writable bits and an injection policy, then hash-attests
the result so a run can prove which skill bits were visible:

- A (no-skill):        empty directory.
- B (frozen-skill):    read-only copy; experience/evolution machinery disabled.
- C (raw retrieval):   copy with ``experience/inbox/`` writable; raw records
                       are retrieved under a matched token budget, without
                       RuleSpec abstraction or governance transitions.
- C_STRESS (ablation): old append-only/inject-everything control.
- D (governed):        copy with the full pipeline dirs; injection policy =
                       only canonical rules in ``rules/`` + ``registry/rules.json``.

The core skill's ``scripts/render_skill_view.py`` is the allowlisted view
boundary; the harness adds only condition-specific writable pipeline state and
attests the resulting store.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import anticheat

CONDITIONS = ("A", "B", "C", "C_STRESS", "D")
_RAW_EXPERIENCE_FORBIDDEN_KEYS = {
    "rulespec", "rule_spec", "promotion", "replay_manifest", "specialization", "retirement"
}

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
    "relations",
    "relation_states",
    "registry",
    "tests/rule_cases",
)

INJECTION_POLICIES = {
    "A": {"mode": "none", "description": "no skill visible"},
    "B": {"mode": "frozen", "description": "initial snapshot only; no experience or rules injected"},
    "C": {"mode": "raw_experience_retrieval", "retrieval_budget_tokens": 4096, "description": "raw records retrieved under a matched token budget; no rule abstraction"},
    "C_STRESS": {"mode": "inbox_any", "retrieval_budget_tokens": 4096, "description": "append-only stress ablation; any inbox record is eligible"},
    "D": {"mode": "canonical_only", "retrieval_budget_tokens": 4096, "description": "only canonical rules in rules/ + registry/rules.json are eligible"},
}


def _make_read_only(path: Path) -> None:
    for entry in sorted(path.rglob("*")):
        if entry.is_dir():
            entry.chmod(stat.S_IRUSR | stat.S_IXUSR)
        elif entry.is_file() and not entry.is_symlink():
            entry.chmod(stat.S_IRUSR)
    path.chmod(stat.S_IRUSR | stat.S_IXUSR)


def _attest(out_dir: Path, condition: str, policy: dict[str, Any], snapshot_dir: Path | None, context_mode: str) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "condition": condition,
        "context_mode": context_mode,
        "injection_policy": policy,
        # The worker-visible manifest records the contract, never a host path.
        # The source bundle is already identified by the attested file set.
        "source_snapshot": "redacted" if snapshot_dir else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": anticheat.hash_tree(out_dir),
    }
    manifest["files"].pop("condition_manifest.json", None)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (out_dir / "condition_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _raw_experience_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in _RAW_EXPERIENCE_FORBIDDEN_KEYS or _raw_experience_forbidden(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_raw_experience_forbidden(item) for item in value)
    return False


def store_digest(condition_dir: str | Path) -> str:
    """Digest the attested store contents, excluding its mutable manifest."""
    files = anticheat.hash_tree(Path(condition_dir))
    files.pop("condition_manifest.json", None)
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def refresh_attestation(condition_dir: str | Path) -> dict[str, Any]:
    """Advance the store attestation after an allowed maintenance transition."""
    condition_dir = Path(condition_dir)
    policy_ok, policy_errors = verify_condition_policy(condition_dir)
    if not policy_ok:
        raise ValueError("cannot refresh attestation for an unauthorized transition: " + "; ".join(policy_errors))
    manifest_path = condition_dir / "condition_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid condition manifest: {exc}") from exc
    if str(previous.get("condition", "")).upper() == "D":
        from core.mutation_journal import MutationJournal
        MutationJournal(condition_dir / "evolution" / "mutation_journal.jsonl").append(
            "update_state", "condition-store", artifact_path="condition_manifest.json"
        )
    return _attest(
        condition_dir,
        str(previous["condition"]),
        dict(previous["injection_policy"]),
        previous.get("source_snapshot"),
        str(previous.get("context_mode", "reset")),
    )


def materialize_condition(
    condition: str, snapshot_dir: str | Path | None, out_dir: str | Path, context_mode: str = "reset"
) -> dict[str, Any]:
    """Build the condition store; returns the attestation manifest dict.

    *snapshot_dir* is required for B/C/D and must be a
    ``render_skill_view.py`` bundle; it is ignored for A. Existing *out_dir*
    content is replaced.
    """
    condition = condition.upper()
    if context_mode not in {"reset", "carry"}:
        raise ValueError("context_mode must be reset or carry")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    policy = dict(INJECTION_POLICIES[condition])
    if condition == "A":
        return _attest(out_dir, condition, policy, None, context_mode)

    if snapshot_dir is None:
        raise ValueError(f"condition {condition} requires --snapshot DIR")
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {snapshot_dir}")
    if not (snapshot_dir / "skill_view_manifest.json").is_file():
        raise ValueError("snapshot must be a render_skill_view.py bundle, not a repository root")
    from scripts.render_skill_view import validate_skill_view_bundle

    bundle_errors = validate_skill_view_bundle(snapshot_dir)
    if bundle_errors:
        raise ValueError("invalid skill-view bundle: " + "; ".join(bundle_errors))
    shutil.copytree(snapshot_dir, out_dir, dirs_exist_ok=True)

    if condition == "B":
        # Attest first (writes condition_manifest.json), then lock the tree read-only.
        manifest = _attest(out_dir, condition, policy, snapshot_dir, context_mode)
        _make_read_only(out_dir)
        return manifest

    if condition in {"C", "C_STRESS"}:
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
    return _attest(out_dir, condition, policy, snapshot_dir, context_mode)


def verify_attestation(condition_dir: str | Path) -> tuple[bool, list[str]]:
    """Re-hash a condition store and compare against its attestation manifest."""
    condition_dir = Path(condition_dir)
    manifest_path = condition_dir / "condition_manifest.json"
    if not manifest_path.is_file():
        return False, ["condition_manifest.json missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"invalid condition manifest: {exc}"]
    digest = manifest.get("manifest_digest")
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    expected_digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if digest != expected_digest:
        return False, ["condition manifest digest mismatch"]
    recorded = dict(manifest.get("files", {}))
    current = anticheat.hash_tree(condition_dir)
    current.pop("condition_manifest.json", None)
    return anticheat.manifests_equal(recorded, current)


def verify_condition_policy(condition_dir: str | Path) -> tuple[bool, list[str]]:
    """Check mutable-store boundaries for B/C/C_STRESS without hiding writes."""
    condition_dir = Path(condition_dir)
    manifest_path = condition_dir / "condition_manifest.json"
    if not manifest_path.is_file():
        return False, ["condition_manifest.json missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"invalid condition manifest: {exc}"]
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    expected_digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if manifest.get("manifest_digest") != expected_digest:
        return False, ["condition manifest digest mismatch"]
    condition = manifest.get("condition")
    baseline_files = dict(manifest.get("files", {}))
    current_files = anticheat.hash_tree(condition_dir)
    current_files.pop("condition_manifest.json", None)
    baseline = set(baseline_files)
    current = set(current_files)
    additions = sorted(current - baseline)
    changed = sorted(path for path in baseline & current if baseline_files[path] != current_files[path])
    deleted = sorted(baseline - current)
    if condition in {"C", "C_STRESS"}:
        if changed or deleted:
            return False, [
                *[f"{condition} store modified attested file: {path}" for path in changed],
                *[f"{condition} store deleted attested file: {path}" for path in deleted],
            ]
        forbidden = [path for path in additions if not path.startswith("experience/inbox/")]
        if forbidden:
            return False, [f"{condition} store wrote outside experience/inbox: {path}" for path in forbidden]
        if condition == "C":
            for relative in additions:
                path = condition_dir / relative
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if _raw_experience_forbidden(record):
                    return False, [f"C raw experience contains governed-rule structure: {relative}"]
    elif condition == "D":
        mutable_prefixes = (
            "experience/", "evolution/", "rules/", "relations/", "relation_states/",
            "registry/", "tests/rule_cases/"
        )
        forbidden_additions = [path for path in additions if not path.startswith(mutable_prefixes)]
        forbidden_changes = [path for path in changed if not path.startswith(mutable_prefixes)]
        forbidden_deletions = [path for path in deleted if not path.startswith(mutable_prefixes)]
        if forbidden_additions or forbidden_changes or forbidden_deletions:
            return False, [
                *[f"D store wrote outside governance state: {path}" for path in forbidden_additions],
                *[f"D store modified immutable skill state: {path}" for path in forbidden_changes],
                *[f"D store deleted immutable skill state: {path}" for path in forbidden_deletions],
            ]
    elif condition == "B" and (additions or changed or deleted):
        return False, [
            *[f"frozen B store changed: {path}" for path in additions],
            *[f"frozen B store modified: {path}" for path in changed],
            *[f"frozen B store deleted: {path}" for path in deleted],
        ]
    return True, []
