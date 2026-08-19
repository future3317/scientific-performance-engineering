from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.acre.engine import AcreEngine
from core.acre.factorial import FactorialBlock, FactorialEngine, RelationEvidenceCertificate
from core.acre.cegis import BoundaryObservation, StatisticalCEGIS
from core.acre.experiments import ExperimentPlan, execute_paired_plan
from core.acre.predicates import PredicateGrammar
from core.governance import apply_promotion
from core.models import RelationSpec, RuleSpec, RuleState, identifier_digest
from benchmark.formal.schedule import execute_required_experiments


def _validation(store: Path, prefix: str) -> tuple[str, str]:
    value = {
        "scope": "calibration",
        "promotion_case_ids": [f"{prefix}-promotion"],
        "synthesis_case_ids": [f"{prefix}-synthesis"],
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


def _write_rule_store(store: Path, specs: list[RuleSpec], *, relations: list[RelationSpec] = ()) -> None:
    rules = store / "rules"
    entries = []
    for spec in specs:
        directory = rules / identifier_digest(spec.rule_id)
        directory.mkdir(parents=True, exist_ok=True)
        card = directory / "v0001.json"
        card.write_text(json.dumps(spec.to_dict()) + "\n", encoding="utf-8")
        (directory / "v0001.state.json").write_text(json.dumps({
            "rule_id": spec.rule_id, "version": 1, "status": "canonical", "drift_state": "stable",
            "effect": {"lower_utility": 0.3}, "confidence_sequence": {"utility_effect_lcb": 0.3},
            "applicability_calibration": {}, "retrieval_utility": 0.3, "override_rate": 0.0,
            "provenance_diversity": 2,
        }) + "\n", encoding="utf-8")
        entries.append({"rule_id": spec.rule_id, "path": str(card.relative_to(store)).replace("\\", "/"),
                        "status": "canonical", "version": 1,
                        "spec_digest": hashlib.sha256(card.read_bytes()).hexdigest()})
    registry = store / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "rules.json").write_text(json.dumps({"rules": entries}) + "\n", encoding="utf-8")
    if relations:
        relation_entries = []
        for spec in relations:
            directory = store / "relations" / identifier_digest(spec.relation_id)
            directory.mkdir(parents=True, exist_ok=True)
            card = directory / "v0001.json"
            card.write_text(json.dumps(spec.to_dict()) + "\n", encoding="utf-8")
            (directory / "v0001.state.json").write_text(json.dumps({
                "relation_id": spec.relation_id, "version": 1, "estimate": 0.2,
                "confidence_sequence": {"utility_effect_lcb": 0.1}, "status": "canonical",
                "drift_state": "stable", "counterexample_count": 0, "contrast_bounds": {"gamma": {"lcb": 0.1, "ucb": 0.3}},
                "semantic_certificate": {},
            }) + "\n", encoding="utf-8")
            relation_entries.append({"relation_id": spec.relation_id, "path": str(card.relative_to(store)).replace("\\", "/"),
                                     "status": "canonical", "version": 1,
                                     "spec_digest": hashlib.sha256(card.read_bytes()).hexdigest()})
        (registry / "relations.json").write_text(json.dumps({"relations": relation_entries}) + "\n", encoding="utf-8")


_NODE_EXECUTOR = r'''
import json, sys
from pathlib import Path
request = json.loads(sys.argv[1])
arm = sys.argv[2]
context = request.get("context", request)
workload = context.get("workload", context)
positive = arm == "on" and float(workload.get("x", -1)) >= 0.0
value = 1.1 if positive else 1.0
payload = {"measurements": [value, value, value, value], "scientific_ok": True,
           "higher_is_better": True, "utility_scale": 0.5,
           "activation": {"status": "verified", "matched_actions": ["RULE-E2E"]}}
if Path('/worker/result').is_dir():
    Path('/worker/solution/worker-output.json').write_text(json.dumps(payload))
    Path('/worker/result/result.json').write_text(json.dumps(payload))
print(json.dumps(payload))
'''


def _external_node(context: dict[str, object], *, arm: str) -> dict[str, object]:
    if shutil.which("bwrap") is not None:
        from benchmark.formal.reference_executor import ReferenceExecutor
        with tempfile.TemporaryDirectory(prefix="acre-formal-worker-", dir=str(Path.home())) as raw_root:
            root = Path(raw_root)
            for name in ("task", "skill_view", "retrieved_context", "context_state", "solution", "result", "executor_receipt"):
                (root / name).mkdir()
            (root / "task" / "public_task.json").write_text("{}\n", encoding="utf-8")
            receipt_path = root / "executor_receipt" / "receipt.json"
            completed = ReferenceExecutor().execute(
                [sys.executable, "-c", _NODE_EXECUTOR, json.dumps(context), arm], root,
                receipt_path=receipt_path, worker_uid="formal-smoke",
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert receipt["isolation_canary"] is True
            return json.loads((root / "result" / "result.json").read_text(encoding="utf-8"))
    completed = subprocess.run(
        [sys.executable, "-c", _NODE_EXECUTOR, json.dumps(context), arm],
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def test_external_node_router_to_restart_lifecycle(tmp_path: Path):
    """A real subprocess closes proposal, evidence, CEGIS, promotion, and reuse."""
    contexts = tuple({"context_id": name, "context": {"domain": "runtime", "workload": {"x": value}},
                      "independence_group": name, "version": 1}
                     for name, value in (("pos-a", 1), ("pos-b", 2)))
    cases: list[dict[str, object]] = []
    execution = execute_paired_plan(
        ExperimentPlan("RULE-E2E", contexts, max_groups=2),
        type("Executor", (), {"execute": lambda self, context, arm="on": _external_node(context, arm=arm)})(),
        record_case=lambda case: cases.append(dict(case)),
        update_certificate=lambda _cases: {"status": "collecting"},
    )
    assert execution.groups_executed == 2
    assert all(case["on"]["activation"]["status"] == "verified" for case in cases)

    grammar = PredicateGrammar.from_dict({"schema_version": 1, "features": [{"path": "workload.x", "type": "numeric"}],
                                          "threshold_universe": {"workload.x": [0.0]}})
    observations = [
        BoundaryObservation("pos-a", {"workload": {"x": 1}}, 0.1, True, 0.08, 0.12),
        BoundaryObservation("pos-b", {"workload": {"x": 2}}, 0.1, True, 0.08, 0.12),
        BoundaryObservation("neg", {"workload": {"x": -1}}, 0.0, False, -0.01, 0.01),
    ]
    synthesis = StatisticalCEGIS(grammar).synthesize(
        positive=observations[:2], counterexamples=observations[2:], parent_predicate=None,
        decision_contexts=[item.context for item in observations],
    )
    assert synthesis.status == "identified"
    candidate = replace(_rule("RULE-E2E", "e2e-action"), applicability=synthesis.predicate).to_dict()
    candidate["severity"] = "P3"
    heldout_on = _external_node({"context": {"domain": "runtime", "workload": {"x": 3}}}, arm="on")
    heldout_off = _external_node({"context": {"domain": "runtime", "workload": {"x": 3}}}, arm="off")
    poison_on = _external_node({"context": {"domain": "runtime", "workload": {"x": -2}}}, arm="on")
    poison_off = _external_node({"context": {"domain": "runtime", "workload": {"x": -2}}}, arm="off")
    heldout_effect = float(heldout_on["measurements"][0]) - float(heldout_off["measurements"][0])
    poison_accepted = float(poison_on["measurements"][0]) > float(poison_off["measurements"][0])
    validation = {
        "scope": "calibration", "promotion_case_ids": [case["case_id"] for case in cases],
        "synthesis_case_ids": ["RULE-E2E-synthesis"],
        "heldout_regression_cases": [{"case_id": "RULE-E2E-heldout", "executed": True, "execution_source": "external_executor",
                                       "scientific_ok": bool(heldout_on["scientific_ok"] and heldout_off["scientific_ok"]),
                                       "effect_lcb": heldout_effect, "effect_ucb": heldout_effect}],
        "poison_probe_cases": [{"case_id": "RULE-E2E-poison", "executed": True, "execution_source": "external_executor",
                                 "accepted": poison_accepted, "abstained": not poison_accepted}],
        "regression_tolerance": 0.0,
    }
    validation_path = tmp_path / "evolution" / "validation" / "node.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    validation_digest = hashlib.sha256(json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"outcome": "passed", "execution_source": "external_executor", "result": {
        "p_min": 0.8, "mean_effect": 0.1, "utility_effect_lcb": 0.05, "utility_effect_ucb": 0.15,
        "promotion_probability_lower_bound": 0.95,
    }, "promotion_record": {
        "representative_groups": ["pos-a", "pos-b"], "promotion_case_ids": [case["case_id"] for case in cases],
        "heldout_regression_digest": validation_digest, "validation_artifact_digest": validation_digest,
        "validation_artifact_path": "evolution/validation/node.json", "poison_gate": {"passed": True},
        "promotion_probability_lcb": 0.95, "utility_effect_cs": {"lcb": 0.05, "ucb": 0.15},
        "replay_manifest_digest": "external-node-replay",
    }}
    decision = apply_promotion(tmp_path, candidate, manifest, replay_path="external-node.json")
    assert decision.allowed
    restarted = AcreEngine.from_store(tmp_path)
    assert restarted.route({"domain": "runtime", "workload": {"x": 1}}).selected_rule_ids == ("RULE-E2E",)
    assert restarted.route({"domain": "runtime", "workload": {"x": -1}}).selected_rule_ids == ()


_PAIR_EXECUTOR = r'''
import json, sys
payload = json.loads(sys.argv[1])
arms = ["00", "10", "01", "11"]
print(json.dumps({"status": "executed", "execution_source": "external_executor",
  "arm_evidence": {arm: {"utility": {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}[arm], "scientific_ok": True} for arm in arms},
  "scientific_gates": {arm: True for arm in arms},
  "blocks": [{arm: {"utility": {"00": 0.0, "10": 0.1, "01": 0.1, "11": 0.9}[arm], "scientific_ok": True} for arm in arms} for _ in range(1024)]}))
'''


def test_external_pair_request_and_promotion_reuse(tmp_path: Path):
    left, right = _rule("RULE-PAIR-A", "a"), _rule("RULE-PAIR-B", "b")
    left = replace(left, scientific_invariants=["finite_loss"])
    right = replace(right, scientific_invariants=["finite_loss"])
    states = {spec.rule_id: RuleState(spec.rule_id, 1, status="canonical", effect={"lower_utility": 0.3}, retrieval_utility=0.3) for spec in (left, right)}
    engine = AcreEngine(rule_specs=[left, right], rule_states=states)
    initial = engine.route({"domain": "runtime", "workload": {}})
    request = next(item for item in initial.required_experiments if item["experiment_type"] == "pair_factorial")

    def external(payload: dict[str, object]) -> dict[str, object]:
        completed = subprocess.run([sys.executable, "-c", _PAIR_EXECUTOR, json.dumps(payload)], check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    executed = execute_required_experiments([request], executor=external)[0]
    assert executed["status"] == "executed" and executed["execution_source"] == "external_executor"
    factorial = FactorialEngine(delta=0.2, practical_margin=0.05, look_count=1)
    for index, block in enumerate(executed["blocks"]):
        factorial.add_block(FactorialBlock(str(index), {arm: float(block[arm]["utility"]) for arm in ("00", "10", "01", "11")}))
    estimate = factorial.estimate()
    assert estimate.decision == "confirmed_synergy"
    relation = RelationSpec("REL-PAIR", 1, None, {"left": left.rule_id, "right": right.rule_id}, "symmetric", "synergy", {"all": []}, {"contrast": "gamma"}, 0.05, [], {"required": True})
    cert = RelationEvidenceCertificate({name: {"lcb": bounds[0], "ucb": bounds[1]} for name, bounds in estimate.contrast_intervals.items()}, 0.2, (1024,), {arm: True for arm in ("00", "10", "01", "11")}, {"source": "external-executor"}, {left.rule_id: 1, right.rule_id: 1})
    _write_rule_store(tmp_path, [left, right])
    validation_path, validation_digest = _validation(tmp_path, "PAIR")
    manifest = {"outcome": "passed", "evidence_type": "factorial_contrast", "execution_source": "external_executor",
                "result": {"p_min": 0.8, "mean_effect": estimate.gamma, "utility_effect_lcb": estimate.gamma_lcb, "utility_effect_ucb": estimate.gamma_ucb, "promotion_probability_lower_bound": 0.95},
                "relation_evidence_certificate": cert.to_dict(), "promotion_record": {
                    "representative_groups": ["g1", "g2"], "promotion_case_ids": ["PAIR-promotion"],
                    "heldout_regression_digest": validation_digest, "validation_artifact_digest": validation_digest,
                    "validation_artifact_path": validation_path, "poison_gate": {"passed": True},
                    "promotion_probability_lcb": 0.95, "utility_effect_cs": {"lcb": estimate.gamma_lcb, "ucb": estimate.gamma_ucb}, "replay_manifest_digest": "external-pair"}}
    assert apply_promotion(tmp_path, {**relation.to_dict(), "severity": "P3"}, manifest, replay_path="external-pair.json").allowed
    assert set(AcreEngine.from_store(tmp_path).route({"domain": "runtime", "workload": {}}).selected_rule_ids) == {left.rule_id, right.rule_id}


_THREE_EXECUTOR = r'''
import json
arms = [f"{a}{b}{c}" for a in (0,1) for b in (0,1) for c in (0,1)]
print(json.dumps({"execution_source": "external_executor", "outcomes": {arm: 0.1 for arm in arms},
                  "scientific_gates": {arm: True for arm in arms}}))
'''


def test_external_higher_order_request_persists_and_unblocks_bundle(tmp_path: Path):
    rules = [_rule(f"RULE-{name}", name.lower()) for name in ("A", "B", "C")]
    rules = [replace(spec, scientific_invariants=["finite_loss"]) for spec in rules]
    relations = [
        RelationSpec(f"REL-{left}{right}", 1, None, {"left": f"RULE-{left}", "right": f"RULE-{right}"},
                     "symmetric", "synergy", {"all": []}, {"contrast": "gamma"}, 0.05, [], {"required": True})
        for left, right in (("A", "B"), ("A", "C"), ("B", "C"))
    ]
    _write_rule_store(tmp_path, rules, relations=relations)
    engine = AcreEngine.from_store(tmp_path)
    context = {"domain": "runtime", "workload": {}}
    initial = engine.route(context)
    request = next(item for item in initial.required_experiments if item["experiment_type"] == "three_way_factorial")
    assert set(request["required_arms"]) == {f"{a}{b}{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)}

    completed = subprocess.run([sys.executable, "-c", _THREE_EXECUTOR], check=True, capture_output=True, text=True)
    external_result = json.loads(completed.stdout)
    assert external_result["execution_source"] == "external_executor"
    cached = dict(external_result)
    def external_higher(_context: dict[str, object]) -> dict[str, object]:
        return cached

    contexts = [{"context_id": f"higher-{index}", "context": {"rule_versions": {spec.rule_id: 1 for spec in rules}},
                 "context_predicate": {"all": []}, "regime_digest": "smoke"} for index in range(1024)]
    certificate = engine.maintainer.execute_higher_order_experiment(contexts, external_higher, practical_margin=0.2)
    assert certificate["status"] == "pairwise_certified"
    restarted = AcreEngine.from_store(tmp_path)
    routed = restarted.route(context)
    assert set(routed.selected_rule_ids) == {spec.rule_id for spec in rules}
    assert not any(item["experiment_type"] == "three_way_factorial" for item in routed.required_experiments)


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
