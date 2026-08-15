"""Feasibility-only verifier sketch; it is not wired into the pilot harness."""

from pathlib import Path


def verify(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root)
    paths = [root / "torchbenchmark/util/model.py", root / "torchbenchmark/util/env_check.py"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {"passed": False, "reason": "pinned checkout missing paths", "missing": missing}
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    return {"passed": ".item()" not in text, "reason": "scalar synchronization probe"}
