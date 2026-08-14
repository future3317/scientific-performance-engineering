#!/usr/bin/env python3
"""Score rule-library description length, utility distortion, and conflict cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_library(cards: list[dict[str, Any]], utility: dict[str, dict[str, float]], lambda_length: float = 0.001, gamma_conflict: float = 1.0) -> dict[str, Any]:
    active = [card for card in cards if card.get("status") == "canonical"]
    # Rate counts what retrieval/selection pays for, not provenance digests or
    # review metadata.  Distortion is counterfactual: callers may provide
    # leave-one-out utility as ``without_<rule_id>``.
    description_length = sum(len(json.dumps(card.get("spec", card.get("rule", card)), sort_keys=True, ensure_ascii=False)) for card in active)
    distortion = sum(max(0.0, float(values.get("reference_utility", 0.0)) - float(values.get("library_utility", 0.0))) for values in utility.values())
    leave_one_out = {
        rule_id: max(0.0, float(values.get("reference_utility", 0.0)) - float(values.get(f"without_{rule_id}", values.get("library_utility", 0.0))))
        for rule_id, values in utility.items()
    }
    conflict_edges = sum(len(card.get("conflicts_with", [])) for card in active)
    objective = distortion + lambda_length * description_length + gamma_conflict * conflict_edges
    recommendations: list[dict[str, Any]] = []
    if distortion == 0.0 and description_length:
        recommendations.append({"action": "retain_or_compress", "reason": "zero measured distortion is evidence that the current library preserves utility; do not infer retirement from zero distortion"})
    for card in active:
        if card.get("conflicts_with"):
            recommendations.append({"action": "specialize_or_resolve", "rule_id": card.get("rule_id"), "reason": "canonical conflict cost is non-zero"})
    return {
        "active_rules": len(active),
        "description_length": description_length,
        "distortion": round(distortion, 12),
        "leave_one_out_distortion": leave_one_out,
        "conflict_cost": conflict_edges,
        "objective": objective,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cards", type=Path, help="JSON file containing a list of rule cards")
    parser.add_argument("--utility", type=Path, required=True, help="reference/library utility JSON mapping")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_library(json.loads(args.cards.read_text(encoding="utf-8")), json.loads(args.utility.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
