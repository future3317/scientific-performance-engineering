#!/usr/bin/env python3
"""Render the agent-visible subset of a skill snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


ALLOWED_FILES = ("SKILL.md",)
ALLOWED_DIRS = ("agents", "assets", "core", "references", "registry", "rules")
ALLOWED_SCRIPT_SUFFIXES = {".py", ".md", ".json"}
DENIED_NAMES = {".git", "benchmark", "oracle", "hidden_verifier", "__pycache__"}


def _manifest_digest(manifest: dict[str, object]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_entries(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in DENIED_NAMES or part.endswith(".pyc") for part in relative.parts):
            continue
        yield path


def _copy_tree(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    if not source.is_dir():
        return copied
    for path in _safe_entries(source):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append((destination / relative).relative_to(destination.parent).as_posix())
    return copied


def render_skill_view(source: str | Path, output: str | Path) -> dict[str, object]:
    source = Path(source).resolve()
    output = Path(output)
    if not source.is_dir():
        raise FileNotFoundError(f"skill snapshot directory not found: {source}")
    if source == output.resolve():
        raise ValueError("skill view output must differ from the source snapshot")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied: list[str] = []
    for name in ALLOWED_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
            copied.append(name)
    for name in ALLOWED_DIRS:
        copied.extend(_copy_tree(source / name, output / name))
    scripts = source / "scripts"
    if scripts.is_dir():
        for path in _safe_entries(scripts):
            if path.is_file() and path.suffix.lower() in ALLOWED_SCRIPT_SUFFIXES:
                target = output / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(path.relative_to(source).as_posix())

    manifest = {
        "schema_version": 1,
        "kind": "skill-view",
        "source_snapshot": str(source),
        "files": sorted(set(copied)),
        "excluded_top_level": ["benchmark", ".git"],
    }
    manifest["manifest_digest"] = _manifest_digest(manifest)
    (output / "skill_view_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_skill_view_bundle(bundle: str | Path) -> list[str]:
    """Validate the generated bundle boundary before a condition can consume it."""
    bundle = Path(bundle)
    manifest_path = bundle / "skill_view_manifest.json"
    if not manifest_path.is_file():
        return ["skill_view_manifest.json missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid skill-view manifest: {exc}"]
    errors: list[str] = []
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "skill-view":
        errors.append("skill-view manifest kind/schema mismatch")
    if not isinstance(manifest.get("files"), list) or not all(isinstance(item, str) for item in manifest["files"]):
        errors.append("skill-view manifest files must be a list of paths")
    if isinstance(manifest.get("manifest_digest"), str) and manifest["manifest_digest"] != _manifest_digest(manifest):
        errors.append("skill-view manifest digest mismatch")
    elif "manifest_digest" not in manifest:
        errors.append("skill-view manifest_digest missing")
    actual = sorted(
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "skill_view_manifest.json"
    )
    declared = sorted(set(manifest.get("files", []))) if isinstance(manifest.get("files"), list) else []
    if actual != declared:
        errors.append("skill-view manifest file list does not match bundle contents")
    for relative in declared:
        path = Path(relative)
        if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] in DENIED_NAMES:
            errors.append(f"skill-view manifest contains forbidden path: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(render_skill_view(args.source, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
