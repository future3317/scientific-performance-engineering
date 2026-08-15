"""Single promotion decision boundary for benchmark and maintenance callers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RelationSpec, RuleSpec


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
        if subject_type == "relation":
            relation_fields = RelationSpec.__dataclass_fields__
            RelationSpec.from_dict({key: candidate[key] for key in relation_fields if key in candidate})
        else:
            RuleSpec.from_dict(candidate)
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
    card = dict(candidate)
    card.pop("cases", None)
    card.pop("epsilon", None)
    card.pop("p_min", None)
    card.pop("delta", None)
    if decision.subject_type == "relation":
        card["relation_id"] = decision.subject_id
    else:
        card["rule_id"] = decision.subject_id
    card["status"] = "canonical"
    replay_result = replay_manifest.get("result", {})
    confidence = dict(card.get("confidence") or {})
    for key in (
        "method", "prior_alpha", "prior_beta", "successes", "failures", "p_min",
        "delta", "posterior_probability", "effective_samples", "promotion_probability_lower_bound",
    ):
        source_key = "confidence_method" if key == "method" else key
        if source_key in replay_result:
            confidence[key] = replay_result[source_key]
    card["confidence"] = confidence
    promotion = dict(card.get("promotion") or {})
    promotion.update({
        "mode": decision.mode,
        "replay_status": "passed",
        "human_review": False,
        "replay_manifest": dict(replay_manifest, path=replay_path),
    })
    card["promotion"] = promotion
    if decision.subject_type == "relation":
        target = store / "relations" / f"{decision.subject_id}.json"
    else:
        target = store / "rules" / f"{decision.subject_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    registry_name = "relations.json" if decision.subject_type == "relation" else "rules.json"
    registry_path = store / "registry" / registry_name
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"schema_version": 1, "rules": []}
    key = "relations" if decision.subject_type == "relation" else "rules"
    id_key = "relation_id" if decision.subject_type == "relation" else "rule_id"
    directory = "relations" if decision.subject_type == "relation" else "rules"
    entries = [entry for entry in registry.get(key, []) if entry.get(id_key) != decision.subject_id]
    entries.append({id_key: decision.subject_id, "path": f"{directory}/{decision.subject_id}.json", "status": "canonical"})
    registry[key] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision
