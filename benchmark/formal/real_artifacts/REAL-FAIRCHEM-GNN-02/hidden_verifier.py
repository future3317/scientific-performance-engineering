"""Feasibility-only verifier sketch for a pinned FAIRChem checkout."""

from pathlib import Path


def verify(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root)
    paths = [
        root / "src/fairchem/core/models/gemnet/gemnet.py",
        root / "src/fairchem/core/models/gemnet/layers/atom_update_block.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {"passed": False, "reason": "pinned checkout missing paths", "missing": missing}
    text = paths[0].read_text(encoding="utf-8")
    return {"passed": "_cached_graph" in text, "reason": "static-geometry cache activation probe"}
