#!/usr/bin/env python3
"""Audit experience cases, candidate rule cards, and canonical registry references."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


SEVERITIES = {"P0", "P1", "P2", "P3", "P4"}
STATUSES = {"candidate", "canonical", "retired"}


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"invalid rule-card schema: {path}")
    return value


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def validate_rule(card: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(card, dict):
        return ["rule card must be a JSON object"]
    if card.get("schema_version") != schema.get("properties", {}).get("schema_version", {}).get("const"):
        errors.append("schema_version must be 1")
    if not isinstance(card.get("rule_id"), str) or not card["rule_id"]:
        errors.append("rule_id must be non-empty")
    if card.get("status") not in STATUSES:
        errors.append("status must be candidate, canonical, or retired")
    if card.get("severity") not in SEVERITIES:
        errors.append("severity must be P0-P4")
    if not isinstance(card.get("domain"), str) or not card["domain"]:
        errors.append("domain must be non-empty")
    trigger = card.get("trigger")
    if not isinstance(trigger, dict) or not isinstance(trigger.get("all"), list) or not trigger["all"]:
        errors.append("trigger.all must be a non-empty list")
    if not isinstance(card.get("requires_evidence"), list) or not card["requires_evidence"]:
        errors.append("requires_evidence must be a non-empty list")
    rule = card.get("rule")
    if not isinstance(rule, dict) or _missing(rule.get("text")):
        errors.append("rule.text must be non-empty")
    if not isinstance(card.get("do_not_apply_when"), list):
        errors.append("do_not_apply_when must be a list")
    if card.get("risk") not in {"low", "medium", "high"}:
        errors.append("risk must be low, medium, or high")
    for key, prefix in (("source_cases", "EXP-"), ("validated_cases", "REG-"), ("regression_cases", "REG-")):
        values = card.get(key)
        if not isinstance(values, list) or any(not isinstance(item, str) or not item.startswith(prefix) for item in values):
            errors.append(f"{key} must contain {prefix} identifiers")
    if not isinstance(card.get("conflicts_with"), list) or not isinstance(card.get("supersedes"), list):
        errors.append("conflicts_with and supersedes must be lists")
    verified = card.get("last_verified")
    if not isinstance(verified, dict) or _missing(verified.get("pytorch")) or _missing(verified.get("date")):
        errors.append("last_verified needs pytorch and date")
    if not isinstance(card.get("owner"), str) or not card["owner"]:
        errors.append("owner must be non-empty")
    promotion = card.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("replay_status") not in {"pending", "passed", "failed"} or not isinstance(promotion.get("human_review"), bool):
        errors.append("promotion needs replay_status pending/passed/failed and boolean human_review")
    elif card.get("status") == "canonical":
        if promotion["replay_status"] != "passed":
            errors.append("canonical rule requires replay_status=passed")
        if not promotion["human_review"]:
            errors.append("canonical rule requires human_review=true")
        if not card.get("validated_cases") or not card.get("regression_cases"):
            errors.append("canonical rule requires validated_cases and regression_cases")
        if _missing(promotion.get("replay_evidence")):
            errors.append("canonical rule requires replay_evidence")
    elif card.get("status") == "retired" and _missing(card.get("retirement_reason")):
        errors.append("retired rule requires retirement_reason")
    return errors


def validate_regression_case(case: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(case, dict):
        return ["regression case must be a JSON object"]
    if case.get("schema_version") != schema.get("properties", {}).get("schema_version", {}).get("const"):
        errors.append("regression case schema_version must be 1")
    if not isinstance(case.get("case_id"), str) or not case["case_id"].startswith("REG-"):
        errors.append("regression case case_id must start with REG-")
    if not isinstance(case.get("rule_id"), str) or not case["rule_id"]:
        errors.append("regression case rule_id must be non-empty")
    if case.get("kind") not in {"positive", "counterexample"}:
        errors.append("regression case kind must be positive or counterexample")
    if case.get("status") not in {"pending", "pass", "fail"}:
        errors.append("regression case status must be pending, pass, or fail")
    scope = case.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("requires"), list) or not isinstance(scope.get("excludes"), list):
        errors.append("regression case scope requires and excludes must be lists")
    for key in ("expected", "observed", "evidence"):
        if _missing(case.get(key)):
            errors.append(f"regression case {key} must be non-empty")
    return errors


def validate_registry(registry: Any, cards: dict[str, dict[str, Any]]) -> list[str]:
    if not isinstance(registry, dict) or registry.get("schema_version") != 1 or not isinstance(registry.get("rules"), list):
        return ["registry needs schema_version=1 and a rules list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(registry["rules"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("rule_id"), str) or not isinstance(entry.get("path"), str):
            errors.append(f"registry.rules[{index}] needs rule_id and path")
            continue
        rule_id = entry["rule_id"]
        if rule_id in seen:
            errors.append(f"duplicate registry rule_id: {rule_id}")
        seen.add(rule_id)
        card = cards.get(rule_id)
        if card is None:
            errors.append(f"registry rule has no card: {rule_id}")
        elif card.get("status") != "canonical" or entry.get("status") != "canonical":
            errors.append(f"registry rule must reference a canonical card: {rule_id}")
    return errors


def audit(root: Path) -> list[str]:
    schema = load_schema(root / "assets" / "rule_candidate.schema.json")
    regression_schema = load_schema(root / "assets" / "rule_regression_case.schema.json")
    errors: list[str] = []
    cards: dict[str, dict[str, Any]] = {}
    regression_cases: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "tests" / "rule_cases").glob("*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        errors.extend(f"{path}: {error}" for error in validate_regression_case(case, regression_schema))
        if isinstance(case, dict) and isinstance(case.get("case_id"), str):
            if case["case_id"] in regression_cases:
                errors.append(f"duplicate regression case_id: {case['case_id']}")
            regression_cases[case["case_id"]] = case
    for directory, expected_status in (
        (root / "evolution" / "candidates", "candidate"),
        (root / "rules", "canonical"),
        (root / "evolution" / "retired", "retired"),
    ):
        for path in sorted(directory.glob("*.json")):
            try:
                card = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            errors.extend(f"{path}: {error}" for error in validate_rule(card, schema))
            if isinstance(card, dict) and card.get("status") != expected_status:
                errors.append(f"{path}: expected status {expected_status}")
            if isinstance(card, dict) and isinstance(card.get("rule_id"), str):
                if card["rule_id"] in cards:
                    errors.append(f"duplicate rule_id across card directories: {card['rule_id']}")
                cards[card["rule_id"]] = card
                if expected_status == "canonical":
                    for case_id in card.get("regression_cases", []):
                        case = regression_cases.get(case_id)
                        if case is None or case.get("status") != "pass":
                            errors.append(f"{path}: canonical rule regression case is not passing: {case_id}")
    registry_path = root / "registry" / "rules.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            errors.extend(validate_registry(registry, cards))
            for entry in registry.get("rules", []) if isinstance(registry, dict) else []:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str) and not (root / entry["path"]).is_file():
                    errors.append(f"registry path does not exist: {entry['path']}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{registry_path}: {exc}")
    return errors


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    errors = audit(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("evolution audit: ok")


if __name__ == "__main__":
    main()
