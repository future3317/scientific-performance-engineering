#!/usr/bin/env python3
"""Audit experience provenance, replay-gated rule cards, and rule-graph integrity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.models import RelationSpec, RelationState, RuleSpec, RuleState, identifier_digest


SEVERITIES = {"P0", "P1", "P2", "P3", "P4"}
STATUSES = {"collecting_evidence", "candidate", "canonical", "retired"}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"invalid rule-card schema: {path}")
    return value


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def validate_rule(card: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(card, dict):
        return ["rule card must be a JSON object"]
    if "applicability" in card and "trigger" not in card:
        try:
            RuleSpec.from_dict(card)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"canonical RuleSpec invalid: {exc}")
        return errors
    try:
        RuleSpec.from_dict(card)
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"canonical RuleSpec invalid: {exc}")
    if isinstance(card.get("state"), dict):
        try:
            RuleState.from_dict(card["state"])
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"canonical RuleState invalid: {exc}")
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
    for key, prefix in (("source_cases", "EXP-"), ("admission_cases", "REG-"), ("regression_cases", "REG-")):
        values = card.get(key)
        if not isinstance(values, list) or (key != "regression_cases" and not values) or any(not isinstance(item, str) or not item.startswith(prefix) for item in values):
            errors.append(f"{key} must contain non-empty {prefix} identifiers")
    if not isinstance(card.get("conflicts_with"), list) or not isinstance(card.get("supersedes"), list):
        errors.append("conflicts_with and supersedes must be lists")
    verified = card.get("last_verified")
    if not isinstance(verified, dict) or _missing(verified.get("pytorch")) or _missing(verified.get("date")):
        errors.append("last_verified needs pytorch and date")
    if not isinstance(card.get("owner"), str) or not card["owner"]:
        errors.append("owner must be non-empty")
    if card.get("collector_confidence") not in {"low", "medium", "high"}:
        errors.append("collector_confidence must be low, medium, or high")
    confidence = card.get("confidence")
    if not isinstance(confidence, dict) or confidence.get("method") not in {"anytime-hoeffding-union-bound", "beta-binomial", "beta-binomial-mixture-cs", "beta-binomial-mixture-e-process"}:
        errors.append("confidence.method must be beta-binomial-mixture-e-process or a legacy method")
    else:
        for key in ("prior_alpha", "prior_beta", "successes", "failures"):
            if not isinstance(confidence.get(key), int) or confidence[key] < 0:
                errors.append(f"confidence.{key} must be a non-negative integer")
        for key in ("p_min", "delta", "posterior_probability", "effective_samples"):
            if not isinstance(confidence.get(key), (int, float)):
                errors.append(f"confidence.{key} must be numeric")
        if confidence.get("method") in {"anytime-hoeffding-union-bound", "beta-binomial-mixture-cs", "beta-binomial-mixture-e-process"} and not isinstance(confidence.get("promotion_probability_lower_bound"), (int, float)):
            errors.append("anytime confidence requires promotion_probability_lower_bound")
    promotion = card.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("replay_status") not in {"pending", "passed", "failed"} or not isinstance(promotion.get("human_review"), bool):
        errors.append("promotion needs replay_status pending/passed/failed and boolean human_review")
    elif card.get("status") == "canonical":
        if promotion["replay_status"] != "passed":
            errors.append("canonical rule requires replay_status=passed")
        if not promotion["human_review"] and promotion.get("mode") != "bounded-auto":
            errors.append("canonical rule requires human review unless mode=bounded-auto")
        if _missing(promotion.get("replay_manifest")):
            errors.append("canonical rule requires replay_manifest")
        for key in ("review_commit", "reviewer", "reviewed_at", "review_diff_hash"):
            if _missing(promotion.get(key)):
                errors.append(f"canonical rule requires promotion.{key}")
        if promotion.get("review_commit") and (not isinstance(promotion["review_commit"], str) or not 7 <= len(promotion["review_commit"]) <= 64 or any(char not in "0123456789abcdefABCDEF" for char in promotion["review_commit"])):
            errors.append("promotion.review_commit must be a Git revision")
        if promotion.get("review_diff_hash") and not _is_digest(promotion["review_diff_hash"]):
            errors.append("promotion.review_diff_hash must be a SHA-256 digest")
        if not card.get("admission_cases") or not card.get("regression_cases"):
            errors.append("canonical rule requires admission_cases and regression_cases")
        if not isinstance(promotion.get("replay_manifest"), dict):
            errors.append("canonical replay_manifest must be an object")
        else:
            manifest = promotion["replay_manifest"]
            for key in ("path", "command", "case_bundle_path", "case_bundle_sha256", "harness_revision", "result_digest", "outcome"):
                if _missing(manifest.get(key)):
                    errors.append(f"canonical replay_manifest requires {key}")
            if not _is_digest(manifest.get("case_bundle_sha256")) or not _is_digest(manifest.get("result_digest")):
                errors.append("canonical replay_manifest requires 64-character digests")
    elif card.get("status") == "retired" and _missing(card.get("retirement_reason")):
        errors.append("retired rule requires retirement_reason")
    return errors


def validate_candidate_projection(card: Any) -> list[str]:
    """Validate the immutable identity of a collecting candidate projection."""
    if not isinstance(card, dict):
        return ["candidate projection must be an object"]
    errors: list[str] = []
    identity = card.get("candidate_identity")
    if not isinstance(identity, str) or not identity:
        errors.append("collecting candidate requires candidate_identity")
    if not isinstance(card.get("rule_id") or card.get("relation_id"), str):
        errors.append("collecting candidate requires a subject id")
    if int(card.get("version", 0)) < 1:
        errors.append("collecting candidate version must be positive")
    state = card.get("synthesis_state")
    if not isinstance(state, dict) or state.get("status") not in {"collecting_evidence", "identified"}:
        errors.append("collecting candidate requires synthesis_state")
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
    if case.get("kind") not in {"admission", "positive", "counterexample"}:
        errors.append("regression case kind must be admission, positive, or counterexample")
    if case.get("status") not in {"pending", "pass", "fail"}:
        errors.append("regression case status must be pending, pass, or fail")
    scope = case.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("requires"), list) or not isinstance(scope.get("excludes"), list):
        errors.append("regression case scope requires and excludes must be lists")
    for key in ("expected", "observed", "evidence"):
        if _missing(case.get(key)):
            errors.append(f"regression case {key} must be non-empty")
    lineage = case.get("lineage")
    if not isinstance(lineage, dict) or not isinstance(lineage.get("derived_from_experience_ids"), list) or _missing(lineage.get("repository_revision")) or _missing(lineage.get("task_family")):
        errors.append("regression case lineage needs derived_from_experience_ids, repository_revision, and task_family")
    return errors


def validate_card_links(card: dict[str, Any], experiences: dict[str, dict[str, Any]], regression_cases: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    source_ids = set(card.get("source_cases", []))
    admission_ids = set(card.get("admission_cases", []))
    regression_ids = set(card.get("regression_cases", []))
    for case_id in source_ids:
        if case_id not in experiences or experiences[case_id].get("status") != "case":
            errors.append(f"source case is missing or not reviewed: {case_id}")
    for case_id in admission_ids | regression_ids:
        if case_id not in regression_cases:
            errors.append(f"referenced regression case is missing: {case_id}")
    overlap = admission_ids & regression_ids
    if overlap:
        errors.append(f"admission and regression cases must be disjoint: {sorted(overlap)}")
    for case_id in admission_ids | regression_ids:
        case = regression_cases.get(case_id)
        if not case:
            continue
        if case_id in admission_ids and case.get("kind") != "admission":
            errors.append(f"admission case must have kind=admission: {case_id}")
        if case_id in regression_ids and case.get("kind") == "admission":
            errors.append(f"regression case cannot have kind=admission: {case_id}")
        derived = set(case.get("lineage", {}).get("derived_from_experience_ids", []))
        if source_ids & derived:
            errors.append(f"evidence leakage: {case_id} lineage overlaps source cases")
    return errors


def validate_provenance_diversity(card: dict[str, Any], experiences: dict[str, dict[str, Any]]) -> list[str]:
    """Gate bounded auto-promotion on independent evidence groups."""
    if card.get("status") != "canonical":
        return []
    promotion = card.get("promotion", {})
    if card.get("severity") in {"P0", "P1"} and promotion.get("human_review") is not True:
        return [f"{card.get('rule_id')}: P0/P1 rules require human review"]
    if promotion.get("mode") != "bounded-auto":
        return []
    groups = {experiences[item].get("independence_group") for item in card.get("source_cases", []) if item in experiences}
    if len(groups) < 2:
        return [f"{card.get('rule_id')}: bounded-auto promotion requires two independent provenance groups"]
    return []


def validate_rule_graph(cards: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for rule_id, card in cards.items():
        for field in ("conflicts_with", "supersedes"):
            for target in card.get(field, []):
                if target not in cards:
                    errors.append(f"{rule_id} has dangling {field} edge to {target}")
                if target == rule_id:
                    errors.append(f"{rule_id} has self {field} edge")
        for target in card.get("conflicts_with", []):
            if target in cards and cards[target].get("status") == "canonical" and card.get("status") == "canonical":
                errors.append(f"canonical rules conflict: {rule_id} and {target}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(rule_id: str) -> None:
        if rule_id in visiting:
            errors.append(f"supersedes cycle detected at {rule_id}")
            return
        if rule_id in visited:
            return
        visiting.add(rule_id)
        for target in cards.get(rule_id, {}).get("supersedes", []):
            if target in cards:
                visit(target)
        visiting.remove(rule_id)
        visited.add(rule_id)

    for rule_id in cards:
        visit(rule_id)
    return errors


def validate_replay_manifest(card: dict[str, Any], root: Path) -> list[str]:
    if card.get("status") != "canonical":
        return []
    manifest_ref = card.get("promotion", {}).get("replay_manifest", {})
    if not isinstance(manifest_ref, dict):
        return [f"{card.get('rule_id')}: replay_manifest must be an object"]
    path = (root / str(manifest_ref.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return [f"{card.get('rule_id')}: replay manifest escapes repository root"]
    if not path.is_file():
        return [f"{card.get('rule_id')}: replay manifest does not exist: {manifest_ref.get('path')}" ]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: invalid replay manifest: {exc}"]
    errors: list[str] = []
    case_bundle_path = (root / str(manifest.get("case_bundle_path", ""))).resolve()
    try:
        case_bundle_path.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{path}: case bundle escapes repository root")
    if not case_bundle_path.is_file():
        errors.append(f"{path}: case bundle does not exist: {manifest.get('case_bundle_path')}")
    elif case_bundle_path.is_file():
        try:
            case_bundle = json.loads(case_bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid case bundle: {exc}")
        else:
            if hashlib.sha256(canonical_json(case_bundle)).hexdigest() != manifest.get("case_bundle_sha256"):
                errors.append(f"{path}: case_bundle_sha256 mismatch")
    if manifest.get("rule_id") != card.get("rule_id") or manifest.get("outcome") != "passed":
        errors.append(f"{path}: replay manifest rule_id/outcome does not prove the card")
    result = manifest.get("result")
    if not isinstance(result, dict):
        errors.append(f"{path}: replay manifest needs a machine-readable result")
    else:
        digest = hashlib.sha256(canonical_json(result)).hexdigest()
        if digest != manifest.get("result_digest") or digest != manifest_ref.get("result_digest"):
            errors.append(f"{path}: result_digest mismatch")
        confidence = card.get("confidence", {})
        for key in ("successes", "failures", "prior_alpha", "prior_beta"):
            if result.get(key) != confidence.get(key):
                errors.append(f"{path}: replay result differs from card confidence: {key}")
        for key in ("p_min", "delta", "posterior_probability"):
            if not isinstance(result.get(key), (int, float)) or abs(float(result[key]) - float(confidence.get(key, -1))) > 1e-12:
                errors.append(f"{path}: replay result differs from card confidence: {key}")
        if result.get("mean_effect", float("-inf")) <= result.get("epsilon", float("inf")) or not result.get("scientific_gates_passed", False):
            errors.append(f"{path}: replay result does not clear paired utility/scientific gates")
        if result.get("confidence_method") in {"anytime-hoeffding-union-bound", "beta-binomial-mixture-cs", "beta-binomial-mixture-e-process"} and result.get("promotion_probability_lower_bound", 0.0) < result.get("p_min", 1.0):
            errors.append(f"{path}: replay result does not clear the anytime-valid promotion gate")
        if result.get("utility_policy_id") != "bounded_log_speedup_v1":
            errors.append(f"{path}: replay result must declare utility_policy_id=bounded_log_speedup_v1")
        if not -1.0 <= float(result.get("mean_effect", 2.0)) <= 1.0:
            errors.append(f"{path}: normalized mean_effect must be in [-1, 1]")
    attestation = manifest.get("attestation")
    body = {key: value for key, value in manifest.items() if key != "attestation"}
    if not isinstance(attestation, dict) or attestation.get("algorithm") != "sha256" or attestation.get("manifest_digest") != hashlib.sha256(canonical_json(body)).hexdigest():
        errors.append(f"{path}: manifest attestation mismatch")
    for key in ("command", "case_bundle_path", "case_bundle_sha256", "harness_revision", "result_digest", "outcome"):
        if manifest.get(key) != manifest_ref.get(key):
            errors.append(f"{path}: replay manifest field differs from card: {key}")
    return errors


def validate_experience_provenance(record: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    for name, artifact in (record.get("artifacts") or {}).items():
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            errors.append(f"artifact provenance is malformed: {name}")
            continue
        path = (root / artifact["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"artifact path escapes repository root: {name}")
            continue
        if not path.is_file():
            errors.append(f"artifact path does not exist: {artifact['path']}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest().lower() != str(artifact.get("sha256", "")).lower():
            errors.append(f"artifact digest mismatch: {artifact['path']}")
    return errors


def validate_registry(registry: Any, cards: dict[str, dict[str, Any]], root: Path | None = None) -> list[str]:
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
        if root is not None:
            path = root / str(entry.get("path", ""))
            try:
                active = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                active = None
            if not isinstance(active, dict) or active.get("rule_id") != rule_id:
                errors.append(f"registry rule active path is not the declared subject: {rule_id}")
            elif int(active.get("version", 0)) != int(entry.get("version", 0)):
                errors.append(f"registry rule active version mismatch: {rule_id}")
    return errors


def validate_relation_artifact(card: Any, state: Any, expected_status: str) -> list[str]:
    errors: list[str] = []
    try:
        spec = RelationSpec.from_dict(card)
        relation_state = RelationState.from_dict(state)
    except (TypeError, ValueError, KeyError) as exc:
        return [f"relation spec/state invalid: {exc}"]
    if spec.relation_id != relation_state.relation_id or spec.version != relation_state.version:
        errors.append(f"relation spec/state version mismatch: {spec.relation_id}")
    if relation_state.status != expected_status:
        errors.append(f"relation state expected {expected_status}")
    promotion_lcb = relation_state.confidence_sequence.get("promotion_probability_lcb")
    if expected_status == "canonical" and not isinstance(promotion_lcb, (int, float)):
        errors.append("canonical relation requires promotion_probability_lcb")
    return errors


def audit(root: Path, schema_root: Path | None = None) -> list[str]:
    schema_root = schema_root or root
    schema = load_schema(schema_root / "assets" / "rule_candidate.schema.json")
    regression_schema = load_schema(schema_root / "assets" / "rule_regression_case.schema.json")
    errors: list[str] = []
    cards: dict[str, dict[str, Any]] = {}
    history: dict[tuple[str, int], dict[str, Any]] = {}
    regression_cases: dict[str, dict[str, Any]] = {}
    experiences: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "experience" / "cases").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        case_id = record.get("case_id") if isinstance(record, dict) else None
        if isinstance(case_id, str):
            if case_id in experiences:
                errors.append(f"duplicate experience case_id: {case_id}")
            experiences[case_id] = record
            if record.get("status") == "case":
                errors.extend(f"{path}: {error}" for error in validate_experience_provenance(record, root))
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
    candidate_identities: set[str] = set()
    for directory, expected_status in ((root / "evolution" / "candidates", "candidate"), (root / "rules", "canonical"), (root / "evolution" / "retired", "retired")):
        for path in sorted(directory.rglob("*.json")):
            if path.name.endswith(".state.json"):
                continue
            try:
                card = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            if directory == root / "evolution" / "candidates" and card.get("status") in {"collecting_evidence", "candidate"}:
                errors.extend(f"{path}: {error}" for error in validate_candidate_projection(card))
                identity = card.get("candidate_identity")
                if isinstance(identity, str):
                    if identity in candidate_identities:
                        errors.append(f"duplicate candidate_identity: {identity}")
                    candidate_identities.add(identity)
                continue
            errors.extend(f"{path}: {error}" for error in validate_rule(card, schema))
            card_status = card.get("status") if isinstance(card, dict) else None
            if card_status is None and isinstance(card, dict):
                state_path = path.with_name(f"{path.stem}.state.json")
                try:
                    state_value = json.loads(state_path.read_text(encoding="utf-8"))
                    card_status = state_value.get("status")
                except (OSError, json.JSONDecodeError):
                    card_status = None
            if card_status != expected_status:
                errors.append(f"{path}: expected status {expected_status}")
            if isinstance(card, dict) and "status" not in card and card_status is not None:
                card = dict(card)
                card["status"] = card_status
            if isinstance(card, dict) and isinstance(card.get("rule_id"), str):
                key = (str(card["rule_id"]), int(card.get("version", 1)))
                if key in history:
                    errors.append(f"duplicate rule revision: {key[0]} v{key[1]}")
                history[key] = card
                previous = cards.get(card["rule_id"])
                if previous is None or int(card.get("version", 1)) >= int(previous.get("version", 1)):
                    cards[card["rule_id"]] = card
                if "trigger" in card:
                    errors.extend(f"{path}: {error}" for error in validate_card_links(card, experiences, regression_cases))
                    errors.extend(f"{path}: {error}" for error in validate_provenance_diversity(card, experiences))
                    errors.extend(f"{path}: {error}" for error in validate_replay_manifest(card, root))
                else:
                    promotion_dir = root / "evolution" / "promotions" / identifier_digest(card['rule_id'])
                    promotion_path = promotion_dir / f"v{int(card.get('version', 1)):04d}.json"
                    if card_status == "canonical" and not promotion_path.is_file():
                        errors.append(f"{path}: canonical RuleSpec missing promotion record")
                    elif card_status == "canonical" and promotion_path.is_file():
                        try:
                            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            promotion = {}
                        record = promotion.get("record") if isinstance(promotion, dict) else None
                        if not isinstance(record, dict):
                            errors.append(f"{path}: canonical RuleSpec missing PromotionRecord")
    errors.extend(validate_rule_graph(cards))
    registry_path = root / "registry" / "rules.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            errors.extend(validate_registry(registry, cards, root))
            for entry in registry.get("rules", []) if isinstance(registry, dict) else []:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str) and not (root / entry["path"]).is_file():
                    errors.append(f"registry path does not exist: {entry['path']}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{registry_path}: {exc}")
    relation_registry_path = root / "registry" / "relations.json"
    relation_dir = root / "relations"
    relation_cards: dict[str, dict[str, Any]] = {}
    relation_history: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(relation_dir.rglob("*.json")) if relation_dir.is_dir() else []:
        if path.name.endswith(".state.json"):
            continue
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
            state_path = path.with_name(f"{path.stem}.state.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        errors.extend(f"{path}: {error}" for error in validate_relation_artifact(card, state, "canonical"))
        try:
            relation_id = str(card.get("relation_id"))
            promotion_dir = root / "evolution" / "promotions" / identifier_digest(relation_id)
            promotion_path = promotion_dir / f"v{int(card.get('version', 1)):04d}.json"
            if not promotion_path.is_file():
                errors.append(f"{path}: canonical RelationSpec missing promotion record")
            elif isinstance(card, dict):
                try:
                    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    promotion = {}
                if not isinstance(promotion, dict) or not isinstance(promotion.get("record"), dict):
                    errors.append(f"{path}: canonical RelationSpec missing PromotionRecord")
        except (TypeError, ValueError):
            pass
        if isinstance(card, dict) and isinstance(card.get("relation_id"), str):
            key = (str(card["relation_id"]), int(card.get("version", 1)))
            if key in relation_history:
                errors.append(f"duplicate relation revision: {key[0]} v{key[1]}")
            relation_history[key] = card
            previous = relation_cards.get(card["relation_id"])
            if previous is None or int(card.get("version", 1)) >= int(previous.get("version", 1)):
                relation_cards[card["relation_id"]] = card
    if relation_registry_path.exists():
        try:
            relation_registry = json.loads(relation_registry_path.read_text(encoding="utf-8"))
            entries = relation_registry.get("relations", []) if isinstance(relation_registry, dict) else []
            if not isinstance(entries, list):
                errors.append("relation registry needs a relations list")
            else:
                for entry in entries:
                    if not isinstance(entry, dict) or entry.get("relation_id") not in relation_cards:
                        errors.append(f"registry relation has no card: {entry}")
                    elif entry.get("status") != "canonical":
                        errors.append(f"registry relation must be canonical: {entry.get('relation_id')}")
                    else:
                        card = relation_cards[str(entry["relation_id"])]
                        path_value = entry.get("path")
                        if not isinstance(path_value, str) or not (root / path_value).is_file():
                            errors.append(f"registry relation path does not exist: {path_value}")
                        else:
                            active = json.loads((root / path_value).read_text(encoding="utf-8"))
                            if int(active.get("version", 0)) != int(entry.get("version", 0)):
                                errors.append(f"registry relation active version mismatch: {entry.get('relation_id')}")
                            path = root / path_value
                            if entry.get("spec_digest") and hashlib.sha256(path.read_bytes()).hexdigest() != str(entry["spec_digest"]):
                                errors.append(f"registry relation spec digest mismatch: {entry['relation_id']}")
                            if int(entry.get("version", 0)) != int(card.get("version", 0)):
                                errors.append(f"registry relation version mismatch: {entry['relation_id']}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relation_registry_path}: {exc}")
    return errors


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    errors = audit(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("evolution audit: ok")


if __name__ == "__main__":
    main()
