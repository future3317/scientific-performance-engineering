"""Single promotion decision boundary for benchmark and maintenance callers."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from enum import StrEnum

from .models import RelationSpec, RelationState, RuleSpec, RuleState, identifier_digest, validate_identifier


@dataclass(frozen=True)
class EvolutionDecision:
    subject_type: str
    subject_id: str
    operation: str
    status: str
    mode: str
    reason: str
    evidence_ids: tuple[str, ...] = ()
    policy_version: str = "acre-governance-1"

    def __post_init__(self) -> None:
        if self.subject_type not in {"rule", "relation"}:
            raise ValueError("subject_type must be rule or relation")
        if self.operation not in {"NO_OP", "PROMOTE", "SPECIALIZE", "SPLIT", "QUARANTINE", "REVALIDATE", "RETIRE"}:
            raise ValueError("invalid evolution operation")

    @property
    def allowed(self) -> bool:
        return self.status == "approved"

    @property
    def rule_id(self) -> str:
        return self.subject_id if self.subject_type == "rule" else ""


class ValidationKind(StrEnum):
    REPLICATION = "replication"
    TRANSFER = "transfer"
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class ValidationCertificate:
    kind: ValidationKind
    case_id: str
    executed: bool
    execution_source: str
    scientific_ok: bool | None = None
    effect_lcb: float | None = None
    abstained: bool | None = None
    accepted: bool | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidationCertificate":
        raw_kind = value.get("holdout_class", value.get("validation_kind", "adversarial"))
        aliases = {"heldout": "replication", "transfer_holdout": "transfer", "boundary_challenge": "boundary", "poison": "adversarial"}
        kind = ValidationKind(aliases.get(str(raw_kind), str(raw_kind)))
        return cls(kind, str(value["case_id"]), value.get("executed") is True, str(value.get("execution_source", "")), value.get("scientific_ok"), float(value["effect_lcb"]) if value.get("effect_lcb") is not None else None, value.get("abstained"), value.get("accepted"))

    def validate(self, tolerance: float = 0.0) -> str | None:
        if not self.executed or not self.execution_source:
            return "validation must be executed by a named source"
        if self.kind in {ValidationKind.REPLICATION, ValidationKind.TRANSFER}:
            if self.scientific_ok is not True or self.effect_lcb is None or self.effect_lcb < tolerance:
                return "positive holdout did not clear scientific/regression gates"
        elif self.kind is ValidationKind.BOUNDARY:
            if self.abstained is not True and self.effect_lcb is not None and self.effect_lcb > tolerance:
                return "boundary challenge was neither abstained nor non-positive"
        elif self.kind is ValidationKind.ADVERSARIAL and self.accepted is not False:
            return "adversarial validation was accepted"
        return None


def _promotion_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    value = manifest.get("promotion_record")
    return value if isinstance(value, dict) else None


def _validate_promotion_record(manifest: dict[str, Any]) -> str | None:
    record = _promotion_record(manifest)
    if record is None:
        return "promotion record is required"
    required = ("representative_groups", "promotion_case_ids", "heldout_regression_digest", "poison_gate", "promotion_probability_lcb", "utility_effect_cs", "replay_manifest_digest")
    missing = [key for key in required if key not in record]
    if missing:
        return "promotion record missing: " + ", ".join(missing)
    if not isinstance(record["representative_groups"], list) or len(set(record["representative_groups"])) < 2:
        return "promotion requires at least two independent representative groups"
    if not isinstance(record["promotion_case_ids"], list) or not record["promotion_case_ids"] or any(not isinstance(item, str) or not item for item in record["promotion_case_ids"]):
        return "promotion case ids are required"
    if not isinstance(record["heldout_regression_digest"], str) or not record["heldout_regression_digest"]:
        return "held-out regression digest is required"
    empty_digest = hashlib.sha256(b"[]").hexdigest()
    if record["heldout_regression_digest"] == empty_digest:
        return "held-out regression evidence cannot be empty"
    if not isinstance(record["poison_gate"], dict) or record["poison_gate"].get("passed") is not True:
        return "poisoning gate did not pass"
    if not isinstance(record["utility_effect_cs"], dict) or not {"lcb", "ucb"}.issubset(record["utility_effect_cs"]):
        return "utility effect confidence sequence is required"
    if not isinstance(record["replay_manifest_digest"], str) or not record["replay_manifest_digest"]:
        return "replay manifest digest is required"
    if "validation_artifact_digest" in record and not isinstance(record["validation_artifact_digest"], str):
        return "validation artifact digest must be a string"
    if "validation_artifact_path" in record and not isinstance(record["validation_artifact_path"], str):
        return "validation artifact path must be a string"
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    if float(record["promotion_probability_lcb"]) < float(result.get("p_min", 1.0)):
        return "promotion probability gate is below p_min"
    utility_cs = record["utility_effect_cs"]
    if "utility_effect_lcb" in result and abs(float(utility_cs["lcb"]) - float(result["utility_effect_lcb"])) > 1e-12:
        return "promotion utility confidence sequence does not match replay result"
    if "utility_effect_ucb" in result and abs(float(utility_cs["ucb"]) - float(result["utility_effect_ucb"])) > 1e-12:
        return "promotion utility confidence sequence does not match replay result"
    return None


def validate_validation_artifact(value: dict[str, Any], promotion_case_ids: set[str]) -> list[str]:
    """Validate executed, promotion-disjoint held-out and poison probes."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["validation artifact must be an object"]
    declared = {str(item) for item in value.get("promotion_case_ids", []) if isinstance(item, str)}
    if declared != {str(item) for item in promotion_case_ids}:
        errors.append("validation artifact promotion case membership mismatch")
    synthesis = {str(item) for item in value.get("synthesis_case_ids", []) if isinstance(item, str)}
    if not synthesis:
        errors.append("validation artifact synthesis case membership is required")
    if not declared.issubset(synthesis):
        errors.append("promotion cases must be drawn from synthesis cases")
    heldout = value.get("heldout_regression_cases")
    poison = value.get("poison_probe_cases")
    try:
        regression_tolerance = float(value.get("regression_tolerance", 0.0))
    except (TypeError, ValueError):
        regression_tolerance = 0.0
        errors.append("regression_tolerance must be numeric")
    if not isinstance(heldout, list) or not heldout:
        errors.append("held-out validation cases are required")
        heldout = []
    if not isinstance(poison, list) or not poison:
        errors.append("poison validation cases are required")
        poison = []
    seen: set[str] = set()
    for label, entries in (("held-out", heldout), ("poison", poison)):
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or not entry["case_id"]:
                errors.append(f"{label} validation case needs case_id")
                continue
            case_id = str(entry["case_id"])
            if case_id in promotion_case_ids:
                errors.append(f"validation evidence must be disjoint from promotion cases: {case_id}")
            if case_id in seen:
                errors.append(f"validation case ids must be unique: {case_id}")
            seen.add(case_id)
            if entry.get("executed") is not True:
                errors.append(f"{label} validation case was not executed: {case_id}")
            if not isinstance(entry.get("execution_source"), str) or not entry["execution_source"]:
                errors.append(f"{label} validation case needs execution_source: {case_id}")
            try:
                typed = ValidationCertificate.from_dict({
                    **entry,
                    "holdout_class": entry.get("holdout_class", "adversarial" if label == "poison" else "replication"),
                })
                typed_error = typed.validate(regression_tolerance)
                if typed_error:
                    errors.append(f"{label} validation case {case_id}: {typed_error}")
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(f"{label} validation case {case_id}: invalid typed certificate ({exc})")
            if label == "held-out":
                if entry.get("scientific_ok") is not True:
                    errors.append(f"held-out validation case failed scientific gates: {case_id}")
                try:
                    effect_lcb = float(entry["effect_lcb"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"held-out validation case needs effect_lcb: {case_id}")
                else:
                    if effect_lcb < regression_tolerance:
                        errors.append(f"held-out validation case regressed below tolerance: {case_id}")
            if label == "poison" and entry.get("accepted") is not False:
                errors.append(f"poison validation case must be rejected by execution: {case_id}")
            if label == "poison" and value.get("scope") == "formal" and entry.get("validation_class") == "synthetic_validation_only":
                errors.append(f"synthetic poison validation cannot authorize promotion: {case_id}")
    return errors


def _versioned_path(store: Path, directory: str, subject_id: str, version: int) -> Path:
    return store / directory / identifier_digest(subject_id) / f"v{int(version):04d}.json"


def _existing_version(store: Path, directory: str, subject_id: str, version: int) -> Path | None:
    path = _versioned_path(store, directory, subject_id, version)
    if path.is_file():
        return path
    # Read-only support for pre-versioned stores; new writes never use this path.
    legacy = store / directory / f"{identifier_digest(subject_id)}.json"
    return legacy if legacy.is_file() else None


def _has_any_version(store: Path, directory: str, subject_id: str) -> bool:
    directory_path = store / directory / identifier_digest(subject_id)
    return bool(list(directory_path.glob("v*.json"))) or (store / directory / f"{identifier_digest(subject_id)}.json").is_file()


def _canonical_rule_states(store: Path, endpoint_ids: set[str]) -> dict[str, dict[str, Any]]:
    registry_path = store / "registry" / "rules.json"
    if not registry_path.is_file():
        raise ValueError("canonical rule registry is required for relation promotion")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("canonical rule registry is invalid") from exc
    entries = registry.get("rules") if isinstance(registry, dict) else None
    if not isinstance(entries, list):
        raise ValueError("canonical rule registry is invalid")
    states: dict[str, dict[str, Any]] = {}
    for endpoint_id in endpoint_ids:
        matching = [entry for entry in entries if isinstance(entry, dict) and entry.get("rule_id") == endpoint_id and entry.get("status") == "canonical"]
        if len(matching) != 1:
            raise ValueError(f"canonical endpoint version is unavailable: {endpoint_id}")
        relative = Path(str(matching[0].get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"canonical endpoint path is invalid: {endpoint_id}")
        state_path = (store / relative).with_name((store / relative).stem + ".state.json")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"canonical endpoint state is unavailable: {endpoint_id}") from exc
        try:
            version_matches = int(state.get("version", 0)) == int(matching[0].get("version", 0))
        except (AttributeError, TypeError, ValueError):
            version_matches = False
        if (
            not isinstance(state, dict)
            or state.get("rule_id") != endpoint_id
            or state.get("status") != "canonical"
            or not version_matches
        ):
            raise ValueError(f"canonical endpoint state is invalid: {endpoint_id}")
        states[endpoint_id] = state
    return states


def evaluate_candidate(candidate: dict[str, Any], replay_manifest: dict[str, Any]) -> EvolutionDecision:
    """Evaluate replay and policy gates without mutating a store."""
    subject_type = "relation" if candidate.get("relation_id") else "rule"
    subject_id = str(candidate.get("relation_id") if subject_type == "relation" else candidate.get("rule_id") or candidate.get("id") or "")
    if not subject_id:
        return EvolutionDecision(subject_type, "", "PROMOTE", "rejected", "none", "candidate has no subject id")
    try:
        validate_identifier(subject_id, "subject_id")
    except ValueError as exc:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", str(exc))
    try:
        if subject_type == "relation":
            if replay_manifest.get("evidence_type") != "factorial_contrast":
                return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "relations require factorial contrast evidence")
            certificate = candidate.get("relation_evidence_certificate") or replay_manifest.get("relation_evidence_certificate")
            if not isinstance(certificate, dict):
                return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "typed relation evidence certificate is required")
            required_certificate = {"contrast_cs", "alpha_budget", "look_schedule", "scientific_arm_gates", "applicability_provenance", "endpoint_versions"}
            if not required_certificate.issubset(certificate):
                return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "relation evidence certificate is incomplete")
            relation_fields = RelationSpec.__dataclass_fields__
            relation_spec = RelationSpec.from_dict({key: candidate[key] for key in relation_fields if key in candidate})
            from core.acre.factorial import RelationEvidenceCertificate
            cert_obj = RelationEvidenceCertificate(
                contrast_cs=certificate["contrast_cs"],
                alpha_budget=float(certificate["alpha_budget"]),
                look_schedule=tuple(int(item) for item in certificate["look_schedule"]),
                scientific_arm_gates=certificate["scientific_arm_gates"],
                applicability_provenance=certificate["applicability_provenance"],
                endpoint_versions={str(key): int(value) for key, value in certificate["endpoint_versions"].items()},
            )
            endpoint_states = replay_manifest.get("endpoint_states")
            if not isinstance(endpoint_states, dict):
                return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "relation endpoint states must be harness supplied")
            cert_obj.validate_for(relation_spec, endpoint_states)
        else:
            RuleSpec.from_dict({key: value for key, value in candidate.items() if key in RuleSpec.__dataclass_fields__ or key in {"trigger", "requires_evidence", "do_not_apply_when", "conflicts_with", "supersedes", "rule"}})
    except (TypeError, ValueError, KeyError) as exc:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", f"typed {subject_type} invalid: {exc}")
    if replay_manifest.get("outcome") != "passed":
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "replay did not pass")
    gate_error = _validate_promotion_record(replay_manifest)
    if gate_error:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", gate_error)
    severity = str(candidate.get("severity", "P2"))
    requires_review = severity in {"P0", "P1"} or (subject_type == "relation" and candidate.get("kind") in {"prerequisite", "semantic_conflict"})
    if requires_review:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "review_required", "human-review", "scientific-sensitive subject requires human review")
    if severity not in {"P2", "P3"}:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", f"unsupported auto-promotion severity: {severity}")
    return EvolutionDecision(subject_type, subject_id, "PROMOTE", "approved", "bounded-auto", "replay and typed policy gates passed")


def apply_promotion(
    store: str | Path,
    candidate: dict[str, Any],
    replay_manifest: dict[str, Any],
    *,
    replay_path: str,
    candidate_storage_key: str | None = None,
) -> EvolutionDecision:
    """Apply only an approved P2/P3 promotion and update the registry."""
    manifest_for_evaluation = dict(replay_manifest)
    if candidate.get("relation_id"):
        try:
            relation_spec = RelationSpec.from_dict({key: candidate[key] for key in RelationSpec.__dataclass_fields__ if key in candidate})
            manifest_for_evaluation["endpoint_states"] = _canonical_rule_states(store=Path(store), endpoint_ids=set(relation_spec.endpoints.values()))
        except (TypeError, ValueError, KeyError) as exc:
            return EvolutionDecision("relation", str(candidate.get("relation_id", "")), "PROMOTE", "rejected", "none", str(exc))
    decision = evaluate_candidate(candidate, manifest_for_evaluation)
    if not decision.allowed:
        return decision
    store = Path(store)
    record = _promotion_record(replay_manifest) or {}
    validation_path_value = record.get("validation_artifact_path")
    validation_digest_value = record.get("validation_artifact_digest")
    if not isinstance(validation_path_value, str) or not isinstance(validation_digest_value, str):
        return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "validation artifact reference is required")
    else:
        relative_validation = Path(validation_path_value)
        if relative_validation.is_absolute() or ".." in relative_validation.parts:
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "validation artifact path must be store-relative")
        validation_path = store / validation_path_value
        try:
            validation_value = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "validation artifact is missing or invalid")
        actual_digest = hashlib.sha256(json.dumps(validation_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        if actual_digest != validation_digest_value:
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "validation artifact digest mismatch")
        promotion_case_ids = {str(item) for item in record.get("promotion_case_ids", []) if isinstance(item, str)}
        validation_errors = validate_validation_artifact(validation_value, promotion_case_ids)
        if validation_errors:
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "; ".join(validation_errors))
        if not promotion_case_ids:
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "validation artifact needs promotion case ids")
    if decision.subject_type == "relation":
        spec = RelationSpec.from_dict({key: value for key, value in candidate.items() if key in RelationSpec.__dataclass_fields__})
        for endpoint in spec.endpoints.values():
            if not _has_any_version(store, "rules", endpoint):
                return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", f"dangling relation endpoint: {endpoint}")
        card = spec.to_dict()
    else:
        spec = RuleSpec.from_dict({key: value for key, value in candidate.items() if key in RuleSpec.__dataclass_fields__ or key in {"trigger", "requires_evidence", "do_not_apply_when", "conflicts_with", "supersedes", "rule"}})
        card = spec.to_dict()
    replay_result = replay_manifest.get("result", {})
    promotion = {
        "subject_type": decision.subject_type,
        "subject_id": decision.subject_id,
        "mode": decision.mode,
        "replay_status": "passed",
        "human_review": False,
        "replay_manifest": dict(replay_manifest, path=replay_path),
        "review_commit": "0000000",
        "reviewer": "bounded-auto",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "review_diff_hash": hashlib.sha256(json.dumps(card, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "record": dict(_promotion_record(replay_manifest) or {}),
    }
    promotion_path = store / "evolution" / "promotions" / identifier_digest(decision.subject_id) / f"v{int(card.get('version', 1)):04d}.json"
    if decision.subject_type == "relation":
        target = _versioned_path(store, "relations", decision.subject_id, int(card.get("version", 1)))
        certificate = replay_manifest.get("relation_evidence_certificate")
        contrast_bounds = {}
        semantic_certificate = {}
        if isinstance(certificate, dict):
            contrast_bounds = {
                str(name): {
                    "lcb": float(interval["lcb"]),
                    "ucb": float(interval["ucb"]),
                }
                for name, interval in certificate.get("contrast_cs", {}).items()
                if isinstance(interval, dict) and {"lcb", "ucb"}.issubset(interval)
            }
            semantic_certificate = {
                "decision": spec.kind,
                "orientation": spec.orientation,
                "certificate": certificate,
            }
        state = RelationState(
            relation_id=decision.subject_id,
            version=int(card.get("version", 1)),
            estimate=float(replay_result.get("mean_effect", 0.0)),
            confidence_sequence={
                "utility_effect_lcb": max(-1.0, min(1.0, float(replay_result.get("utility_effect_lcb", replay_result.get("mean_effect", 0.0))))),
                "utility_effect_ucb": max(-1.0, min(1.0, float(replay_result.get("utility_effect_ucb", replay_result.get("mean_effect", 0.0))))),
                "promotion_probability_lcb": max(0.0, min(1.0, float(replay_result.get("promotion_probability_lower_bound", 0.0)))),
            },
            contrast_bounds=contrast_bounds,
            semantic_certificate=semantic_certificate,
            status="canonical",
        )
    else:
        target = _versioned_path(store, "rules", decision.subject_id, int(card.get("version", 1)))
        mean_effect = float(replay_result.get("mean_effect", 0.0))
        utility_lcb = float(replay_result.get("utility_effect_lcb", mean_effect))
        state = RuleState(
            rule_id=decision.subject_id,
            version=int(card.get("version", 1)),
            status="canonical",
            effect={"utility": mean_effect, "lower_utility": utility_lcb},
            confidence_sequence={
                "utility_effect_lcb": max(-1.0, min(1.0, utility_lcb)),
                "utility_effect_ucb": max(-1.0, min(1.0, float(replay_result.get("utility_effect_ucb", mean_effect)))),
                "promotion_probability_lcb": max(0.0, min(1.0, float(replay_result.get("promotion_probability_lower_bound", 0.0)))),
            },
            retrieval_utility=mean_effect,
            provenance_diversity=0,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(card, indent=2, ensure_ascii=False) + "\n"
    existing = _existing_version(store, "relations" if decision.subject_type == "relation" else "rules", decision.subject_id, int(card.get("version", 1)))
    if existing is not None:
        try:
            existing_card = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_card = None
        if existing_card != card:
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "immutable_spec_violation")
        target = existing if existing.parent == target.parent else target
    else:
        target.write_text(serialized, encoding="utf-8")
    state_path = target.with_name(f"{target.stem}.state.json")
    state_serialized = json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if state_path.exists():
        try:
            existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "immutable_state_violation")
        if existing_state != state.to_dict():
            return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "immutable_state_violation")
    else:
        state_path.write_text(state_serialized, encoding="utf-8")
    from core.mutation_journal import MutationJournal
    journal = MutationJournal(store / "evolution" / "mutation_journal.jsonl")
    journal.append(
        "add_v2_spec",
        decision.subject_id,
        version=int(card.get("version", 1)),
        artifact_path=str(target.relative_to(store)).replace("\\", "/"),
        digest=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    journal.append(
        "update_state",
        decision.subject_id,
        version=int(card.get("version", 1)),
        artifact_path=str(state_path.relative_to(store)).replace("\\", "/"),
        digest=hashlib.sha256(state_path.read_bytes()).hexdigest(),
    )
    # Promotion is a state transition, not a second copy of the candidate.
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_path.write_text(json.dumps(promotion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    storage_key = candidate_storage_key or str(candidate.get("candidate_identity") or decision.subject_id)
    try:
        candidate_path = store / "evolution" / "candidates" / f"{identifier_digest(storage_key)}.json"
    except ValueError:
        return EvolutionDecision(decision.subject_type, decision.subject_id, "PROMOTE", "rejected", "none", "candidate storage key is invalid")
    if candidate_path.is_file():
        candidate_path.unlink()

    registry_name = "relations.json" if decision.subject_type == "relation" else "rules.json"
    registry_path = store / "registry" / registry_name
    default_key = "relations" if decision.subject_type == "relation" else "rules"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"schema_version": 1, default_key: []}
    key = "relations" if decision.subject_type == "relation" else "rules"
    id_key = "relation_id" if decision.subject_type == "relation" else "rule_id"
    directory = "relations" if decision.subject_type == "relation" else "rules"
    entries = [entry for entry in registry.get(key, []) if entry.get(id_key) != decision.subject_id]
    entries.append({
        id_key: decision.subject_id,
        "path": str(target.relative_to(store)).replace("\\", "/"),
        "status": "canonical",
        "version": int(card.get("version", 1)),
        "spec_digest": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
    registry[key] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    journal.append(
        "activate_registry",
        decision.subject_id,
        version=int(card.get("version", 1)),
        artifact_path=str(registry_path.relative_to(store)).replace("\\", "/"),
        digest=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    )
    return decision
