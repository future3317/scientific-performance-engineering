#!/usr/bin/env python3
"""Validate the ACRE-v0 grammar and deterministic BoundaryBench pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.boundary.families import family_cases, run_boundary_family
from core.acre.predicate_synthesis import PredicateGrammar, SYNTHESIZER_VERSION


def validate(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        grammar = PredicateGrammar.from_dict(json.loads((root / "assets" / "predicate_grammar.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"invalid predicate grammar: {exc}"]
    if grammar.max_depth > 3 or grammar.max_literals > 4:
        errors.append("grammar bounds exceed ACRE-v0 limits")
    for family in ("graph_cache_geometry_motion", "compile_horizon"):
        pools = family_cases(family)
        seen: set[str] = set()
        for pool_name, pool in pools.items():
            for item in pool:
                if item.observation_id in seen:
                    errors.append(f"{family}: duplicate observation id {item.observation_id}")
                seen.add(item.observation_id)
            if not pool:
                errors.append(f"{family}: empty {pool_name}")
        first = run_boundary_family(family)
        second = run_boundary_family(family)
        if first != second:
            errors.append(f"{family}: synthesis is not deterministic")
        if first["status"] != "accepted" or first["sealed_errors"]:
            errors.append(f"{family}: boundary synthesis failed: {first}")
        if first["result"]["synthesizer_version"] != SYNTHESIZER_VERSION:
            errors.append(f"{family}: missing synthesizer provenance")
    return errors


def main() -> int:
    errors = validate()
    print(json.dumps({"acre_version": "v0", "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
