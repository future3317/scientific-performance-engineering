"""Single promotion decision boundary for benchmark and maintenance callers."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _promotion_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    value = manifest.get("promotion_record")
    return value if isinstance(value, dict) else None


def _validate_promotion_record(manifest: dict[str, Any]) -> str | None:
    record = _promotion_record(manifest)
    if record is None:
        return "promotion record is required"
    required = ("representative_groups", "heldout_regression_digest", "poison_gate", "promotion_probability_lcb", "utility_effect_cs", "replay_manifest_digest")
    missing = [key for key in required if key not in record]
    if missing:
        return "promotion record missing: " + ", ".join(missing)
    if not isinstance(record["representative_groups"], list) or len(set(record["representative_groups"])) < 2:
        return "promotion requires at least two independent representative groups"
    if not isinstance(record["heldout_regression_digest"], str) or not record["heldout_regression_digest"]:
        return "held-out regression digest is required"
    if not isinstance(record["poison_gate"], dict) or record["poison_gate"].get("passed") is not True:
        return "poisoning gate did not pass"
    if not isinstance(record["utility_effect_cs"], dict) or not {"lcb", "ucb"}.issubset(record["utility_effect_cs"]):
        return "utility effect confidence sequence is required"
    if not isinstance(record["replay_manifest_digest"], str) or not record["replay_manifest_digest"]:
        return "replay manifest digest is required"
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    if float(record["promotion_probability_lcb"]) < float(result.get("p_min", 1.0)):
        return "promotion probability gate is below p_min"
    utility_cs = record["utility_effect_cs"]
    if "utility_effect_lcb" in result and abs(float(utility_cs["lcb"]) - float(result["utility_effect_lcb"])) > 1e-12:
        return "promotion utility confidence sequence does not match replay result"
    if "utility_effect_ucb" in result and abs(float(utility_cs["ucb"]) - float(result["utility_effect_ucb"])) > 1e-12:
        return "promotion utility confidence sequence does not match replay result"
    return None


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
            RelationSpec.from_dict({key: candidate[key] for key in relation_fields if key in candidate})
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
) -> EvolutionDecision:
    """Apply only an approved P2/P3 promotion and update the registry."""
    decision = evaluate_candidate(candidate, replay_manifest)
    if not decision.allowed:
        return decision
    store = Path(store)
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
    promotion_path = store / "evolution" / "promotions" / f"{identifier_digest(decision.subject_id)}.json"
    if decision.subject_type == "relation":
        target = _versioned_path(store, "relations", decision.subject_id, int(card.get("version", 1)))
        state = RelationState(
            relation_id=decision.subject_id,
            version=int(card.get("version", 1)),
            estimate=float(replay_result.get("mean_effect", 0.0)),
            confidence_sequence={
                "utility_effect_lcb": max(-1.0, min(1.0, float(replay_result.get("utility_effect_lcb", replay_result.get("mean_effect", 0.0))))),
                "utility_effect_ucb": max(-1.0, min(1.0, float(replay_result.get("utility_effect_ucb", replay_result.get("mean_effect", 0.0))))),
                "promotion_probability_lcb": max(0.0, min(1.0, float(replay_result.get("promotion_probability_lower_bound", 0.0)))),
            },
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
    state_path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Promotion is a state transition, not a second copy of the candidate.
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    promotion_path.write_text(json.dumps(promotion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate_path = store / "evolution" / "candidates" / f"{identifier_digest(decision.subject_id)}.json"
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
    entries.append({id_key: decision.subject_id, "path": str(target.relative_to(store)).replace("\\", "/"), "status": "canonical", "version": int(card.get("version", 1))})
    registry[key] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision
