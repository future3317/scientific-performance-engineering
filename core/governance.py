"""Single promotion decision boundary for benchmark and maintenance callers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RuleSpec


@dataclass(frozen=True)
class PromotionDecision:
    rule_id: str
    status: str
    mode: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.status == "approved"


def evaluate_candidate(candidate: dict[str, Any], replay_manifest: dict[str, Any]) -> PromotionDecision:
    """Evaluate replay and policy gates without mutating a store."""
    rule_id = str(candidate.get("rule_id") or candidate.get("id") or "")
    if not rule_id:
        return PromotionDecision("", "rejected", "none", "candidate has no rule_id")
    try:
        RuleSpec.from_dict(candidate)
    except (TypeError, ValueError, KeyError) as exc:
        return PromotionDecision(rule_id, "rejected", "none", f"typed RuleSpec invalid: {exc}")
    if replay_manifest.get("outcome") != "passed":
        return PromotionDecision(rule_id, "rejected", "none", "replay did not pass")
    severity = str(candidate.get("severity", "P2"))
    if severity in {"P0", "P1"}:
        return PromotionDecision(rule_id, "review_required", "human-review", f"{severity} rules require human review")
    if severity not in {"P2", "P3"}:
        return PromotionDecision(rule_id, "rejected", "none", f"unsupported auto-promotion severity: {severity}")
    return PromotionDecision(rule_id, "approved", "bounded-auto", "replay and typed policy gates passed")


def apply_promotion(
    store: str | Path,
    candidate: dict[str, Any],
    replay_manifest: dict[str, Any],
    *,
    replay_path: str,
) -> PromotionDecision:
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
    card["rule_id"] = decision.rule_id
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
    target = store / "rules" / f"{decision.rule_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    registry_path = store / "registry" / "rules.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"schema_version": 1, "rules": []}
    entries = [entry for entry in registry.get("rules", []) if entry.get("rule_id") != decision.rule_id]
    entries.append({"rule_id": decision.rule_id, "path": f"rules/{decision.rule_id}.json", "status": "canonical"})
    registry["rules"] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision
