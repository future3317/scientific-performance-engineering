"""Fail-closed preflight probes for the external worker boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def run_preflight(worker_root: str | Path, *, forbidden: list[str] | None = None) -> dict[str, Any]:
    root = Path(worker_root).resolve()
    forbidden = [str(item) for item in (forbidden or [])]
    findings: list[str] = []
    for item in forbidden:
        candidate = Path(item)
        if candidate.exists():
            findings.append(f"forbidden path visible: {item}")
    for path in root.rglob("*"):
        if path.is_symlink():
            findings.append(f"symlink present in worker root: {path.relative_to(root)}")
        if path.name == ".git" or ".git" in path.parts:
            findings.append(f"VCS metadata visible: {path.relative_to(root)}")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "SPE_CONDITION", "SPE_FUTURE_SCHEDULE"):
        if key in os.environ:
            findings.append(f"forbidden worker environment variable present: {key}")
    return {"status": "pass" if not findings else "fail", "findings": findings}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_root", type=Path)
    parser.add_argument("--forbidden", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.worker_root, forbidden=args.forbidden)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
