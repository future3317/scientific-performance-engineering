from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.acre.engine import AcreEngine
from core.acre.factorial import RelationEvidenceCertificate
from core.governance import apply_promotion
from core.models import RelationSpec, RuleSpec, identifier_digest


def _validation(store: Path, prefix: str) -> tuple[str, str]:
    value = {
        "scope": "calibration",
        "promotion_case_ids": [f"{prefix}-promotion"],
        "synthesis_case_ids": [f"{prefix}-promotion"],
        "heldout_regression_cases": [{
            "case_id": f"{prefix}-heldout", "executed": True,
            "execution_source": "verifier", "scientific_ok": True,
            "effect_lcb": 0.1, "effect_ucb": 0.4,
        }],
        "poison_probe_cases": [{
            "case_id": f"{prefix}-poison", "executed": True,
            "execution_source": "verifier", "accepted": False,
        }],
        "regression_tolerance": 0.0,
    }
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    path = store / "evolution" / "validation" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(store)).replace("\\", "/"), digest


def _rule(rule_id: str, action: str) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id, version=1, parent=None,
        applicability={"all": []}, intervention={"action": action},
        expected_mechanism="measured", evidence_requirements=["factorial"],
        scientific_invariants=[], abstain_conditions={}, relations={},
        runtime_cost={"tokens": 1}, provenance_policy={"required": True},
    )


def test_relation_promotion_reload_and_route_round_trip(tmp_path: Path):
    left, right = _rule("RULE-LEFT", "left-action"), _rule("RULE-RIGHT", "right-action")
    rules = tmp_path / "rules"
    for spec in (left, right):
        directory = rules / identifier_digest(spec.rule_id)
        directory.mkdir(parents=True)
        (directory / "v0001.json").write_text(json.dumps(spec.to_dict()) + "\n", encoding="utf-8")
        (directory / "v0001.state.json").write_text(json.dumps({
            "rule_id": spec.rule_id, "version": 1, "status": "canonical", "drift_state": "stable",
            "effect": {"lower_utility": 0.2}, "confidence_sequence": {"utility_effect_lcb": 0.2},
            "applicability_calibration": {}, "retrieval_utility": 0.2, "override_rate": 0.0,
            "provenance_diversity": 1,
        }) + "\n", encoding="utf-8")
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "rules.json").write_text(json.dumps({"rules": [
        {"rule_id": spec.rule_id, "path": f"rules/{identifier_digest(spec.rule_id)}/v0001.json", "status": "canonical", "version": 1,
         "spec_digest": hashlib.sha256((rules / identifier_digest(spec.rule_id) / "v0001.json").read_bytes()).hexdigest()}
        for spec in (left, right)
    ]}) + "\n", encoding="utf-8")
    validation_path, validation_digest = _validation(tmp_path, "REL")
    relation = RelationSpec(
        relation_id="REL-LEFT-RIGHT", version=1, parent=None,
        endpoints={"left": left.rule_id, "right": right.rule_id}, orientation="symmetric",
        kind="synergy", applicability={"all": []}, contrast_definition={"contrast": "gamma"},
        practical_margin=0.05, scientific_invariants=[], provenance_policy={"required": True},
    )
    certificate = RelationEvidenceCertificate(
        contrast_cs={"gamma": {"lcb": 0.2, "ucb": 0.4}}, alpha_budget=0.05,
        look_schedule=(32,), scientific_arm_gates={arm: True for arm in ("00", "10", "01", "11")},
        applicability_provenance={"source": "core-factorial"}, endpoint_versions={left.rule_id: 1, right.rule_id: 1},
    )
    manifest = {
        "outcome": "passed", "evidence_type": "factorial_contrast", "execution_source": "external_executor",
        "result": {"p_min": 0.8, "mean_effect": 0.2, "utility_effect_lcb": 0.1, "utility_effect_ucb": 0.4, "promotion_probability_lower_bound": 0.9},
        "relation_evidence_certificate": certificate.to_dict(),
        "promotion_record": {
            "representative_groups": ["g1", "g2"], "promotion_case_ids": ["REL-promotion"],
            "heldout_regression_digest": validation_digest, "validation_artifact_digest": validation_digest,
            "validation_artifact_path": validation_path, "poison_gate": {"passed": True},
            "promotion_probability_lcb": 0.9, "utility_effect_cs": {"lcb": 0.1, "ucb": 0.4},
            "replay_manifest_digest": "external-replay",
        },
    }
    decision = apply_promotion(tmp_path, {**relation.to_dict(), "severity": "P2"}, manifest, replay_path="external.json")
    assert decision.allowed
    restarted = AcreEngine.from_store(tmp_path)
    assert restarted.relation_states[relation.relation_id].status == "canonical"
    routed = restarted.route({"domain": "runtime", "workload": {}})
    assert set(routed.selected_rule_ids) == {left.rule_id, right.rule_id}
