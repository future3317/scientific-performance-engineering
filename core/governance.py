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
            relation_fields = RelationSpec.__dataclass_fields__
            RelationSpec.from_dict({key: candidate[key] for key in relation_fields if key in candidate})
        else:
            RuleSpec.from_dict({key: value for key, value in candidate.items() if key in RuleSpec.__dataclass_fields__ or key in {"trigger", "requires_evidence", "do_not_apply_when", "conflicts_with", "supersedes", "rule"}})
    except (TypeError, ValueError, KeyError) as exc:
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", f"typed {subject_type} invalid: {exc}")
    if replay_manifest.get("outcome") != "passed":
        return EvolutionDecision(subject_type, subject_id, "PROMOTE", "rejected", "none", "replay did not pass")
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
    }
    promotion_path = store / "evolution" / "promotions" / f"{identifier_digest(decision.subject_id)}.json"
    if decision.subject_type == "relation":
        target = store / "relations" / f"{identifier_digest(decision.subject_id)}.json"
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
        target = store / "rules" / f"{identifier_digest(decision.subject_id)}.json"
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
    target.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    entries.append({id_key: decision.subject_id, "path": f"{directory}/{identifier_digest(decision.subject_id)}.json", "status": "canonical"})
    registry[key] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision
