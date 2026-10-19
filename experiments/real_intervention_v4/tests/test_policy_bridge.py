from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from benchmark.families import family_predicate_grammar
from core.acre import AcreEngine
from core.acre.cegis import SynthesisCertificate, SynthesisResult, synthesis_certificate_valid
from core.acre.regimes import SupportScope
from core.models import RevisionRef, RuleSpec, RuleState, TaskContext, identifier_digest
from core.utility import utility_effect
import core.acre.regimes as regimes
import core.acre.typed_gate as typed_gate
import experiments.real_intervention_v4.policy_bridge as bridge
from experiments.real_intervention_v4.policy_bridge import (
    build_development_candidate,
    validate_calibration_and_route,
)


FEATURES = (
    "batch", "horizon", "conv2d_module_count", "input_channels",
    "input_height", "input_width",
)
RUNTIME = {
    "torch": "2.6.0", "gpu_uuid": "GPU-test", "source_commit": "source-v4",
    "torch_cuda_version": "12.6", "driver_version": "535.247.01",
}
SCOPE = SupportScope(
    "v4-calibration",
    {"equals": {"domain": "runtime"}},
    {**RUNTIME, "skill_digest": "layout-v1"},
)


def _spec() -> RuleSpec:
    return RuleSpec(
        rule_id="CHANNELS_LAST",
        version=1,
        parent=None,
        applicability={"all": []},
        intervention={"layout": "channels_last", "contract_digest": "layout-v1"},
        expected_mechanism="channels-last memory layout",
        evidence_requirements=["independent paired receipts"],
        scientific_invariants=["numerical correctness"],
        abstain_conditions={},
        relations={},
        runtime_cost={"tokens": 1.0},
        provenance_policy={"required": True},
    )


def _engine(*, status: str = "candidate") -> AcreEngine:
    spec = _spec()
    state = RuleState(spec.rule_id, spec.version, status=status)
    return AcreEngine(rule_specs=[spec], rule_states={spec.rule_id: state})


def _stored_engine(root):
    spec = _spec()
    state = RuleState(spec.rule_id, spec.version, status="canonical",
                      effect={"utility": 0.1, "lower_utility": 0.01},
                      retrieval_utility=0.1)
    key = identifier_digest(spec.rule_id)
    directory = root / "rules" / key
    directory.mkdir(parents=True)
    spec_path = directory / "v0001.json"
    state_path = directory / "v0001.state.json"
    spec_payload = spec.to_dict()
    state_payload = state.to_dict()
    spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    semantic_digest = hashlib.sha256(json.dumps(
        spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    artifact_digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    registry = root / "registry" / "rules.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"rules": [{
        "rule_id": spec.rule_id,
        "path": str(spec_path.relative_to(root)).replace("\\", "/"),
        "status": "canonical",
        "version": spec.version,
        "semantic_spec_digest": semantic_digest,
        "artifact_digest": artifact_digest,
    }]}), encoding="utf-8")
    return AcreEngine.from_store(root)


def _identified_candidate(engine):
    seed = _development(engine, [_pair("dev-1", "family-dev", "development")])
    predicate = {"equals": {"workload.batch": 1}}
    certificate = SynthesisCertificate(
        status="identified", predicate=predicate,
        positive_anchor_ids=("dev-1",), counterexample_ids=("dev-2",),
        unresolved_ids=(), grammar_digest="test", decision_lattice_digest="test",
        version_space_digest="test", alpha_budget=0.01, practical_threshold=0.0,
        support_scope_digest="scope",
        evidence_contexts=({"domain": "runtime"},),
        decision_features=("workload.batch",),
        decision_contexts=({"workload.batch": 1},),
    )
    synthesis = SynthesisResult(
        "identified", predicate, ("dev-2",), ("dev-1",),
        certificate=certificate,
    )
    digest = hashlib.sha256(json.dumps(
        seed.spec.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    candidate_spec = replace(
        seed.spec, version=seed.spec.version + 1,
        parent=RevisionRef(seed.spec.rule_id, seed.spec.version, digest),
        applicability={"all": [seed.spec.applicability, predicate]},
    )
    return replace(
        seed, status="CANDIDATE_ONLY", candidate_spec=candidate_spec,
        state=RuleState(seed.spec.rule_id, candidate_spec.version, status="candidate"),
        synthesis=synthesis,
    )


def _calibration_pairs(family_count: int):
    pairs = []
    for index in range(family_count):
        family = f"family-cal-{index}"
        for batch, inside in ((1, True), (8, False)):
            for horizon in (1, 16, 128, 1024):
                pair = _pair(
                    f"cal-{index}-batch-{batch}-horizon-{horizon}",
                    family, "calibration",
                    s0=2.0 if inside else 1.0,
                    s1=1.0 if inside else 2.0,
                )
                pair["features"]["batch"] = batch
                pair["features"]["horizon"] = horizon
                pairs.append(pair)
    return pairs


def _pair(pair_id: str, family: str, split: str, *, status: str = "VALID",
          correctness: str = "PASS", s0: float = 2.0, s1: float = 1.5):
    identity = {
        "runtime_gpu_uuid": "GPU-test",
        "inputs_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "source_commit": "source-v4",
        "config_digest": "c" * 64,
        "context_digest": "d" * 64,
    }
    runtime_identity = {
        "device_uuid": "GPU-test",
        "torch_version": "2.6.0",
        "torch_cuda_version": "12.6",
        "driver": {"driver_version": "535.247.01"},
        "packages": {},
    }
    return {
        "pair_id": pair_id,
        "family_id": family,
        "split": split,
        "effect_status": status.lower(),
        "features": {
            "batch": 1,
            "horizon": 16,
            "conv2d_module_count": 4,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
        },
        "classification": {
            "arm_execution": "COMPLETE",
            "scientific_correctness": correctness,
            "paired_comparability": "MATCHED",
            "policy_evidence": "ELIGIBLE",
        },
        "S0": {**identity, "raw_receipt_sha256": "e" * 64,
               "runtime_identity": runtime_identity,
               "scientific_correctness": correctness,
               "timing": {"cold_seconds": s0}},
        "S1": {**identity, "raw_receipt_sha256": "f" * 64,
               "runtime_identity": runtime_identity,
               "scientific_correctness": correctness,
               "timing": {"cold_seconds": s1}},
    }


def _development(engine: AcreEngine, pairs):
    return build_development_candidate(
        engine=engine,
        rule_id="CHANNELS_LAST",
        development_pairs=pairs,
        candidate_features={
            "batch": 1,
            "horizon": 16,
            "conv2d_module_count": 4,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
        },
        pre_execution_features=FEATURES,
        family_id="channels_last",
        support_scope=SCOPE,
        runtime_versions=RUNTIME,
        hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"},
        source_id="v4-development",
        timestamp="2026-10-19T00:00:00Z",
    )


def test_development_receipts_use_the_canonical_layout_grammar():
    layout_grammar = family_predicate_grammar("channels_last")
    assert {
        feature["path"].rsplit(".", 1)[-1]
        for feature in layout_grammar["features"]
    } == set(FEATURES)

    engine = _engine()
    candidate = _development(engine, [
        _pair("dev-1", "family-dev", "development"),
        _pair("dev-2", "family-dev", "development", s0=2.1, s1=1.6),
    ])

    assert candidate.status == "CANDIDATE_UNAVAILABLE"
    assert candidate.candidate_spec is None
    assert candidate.synthesis.provenance.get("reason") != "no_canonical_layout_applicability_grammar"
    assert candidate.state.status == "candidate"
    assert all(event.evidence_role == "synthesis" for event in candidate.events)
    assert {event.independence_group for event in candidate.events} == {"family-dev"}
    assert engine.rule_states["CHANNELS_LAST"].status == "candidate"
    assert not synthesis_certificate_valid(candidate.synthesis)


def test_development_multi_context_evidence_is_retained_but_clustered_by_family():
    first = _pair("dev-context-1", "family-dev", "development")
    second = _pair("dev-context-2", "family-dev", "development", s0=2.0, s1=1.4)
    second["features"]["batch"] = 8
    candidate = _development(_engine(), [first, second])

    assert candidate.development_families == {"family-dev"}
    assert candidate.development_pair_ids == {"dev-context-1", "dev-context-2"}
    assert len(candidate.events) == 4
    assert {event.query_id for event in candidate.events} == {
        "dev-context-1", "dev-context-2",
    }
    assert {event.independence_group for event in candidate.events} == {"family-dev"}
    for event in candidate.events:
        refs = event.artifacts["paired_receipt_refs"]
        assert refs["S0"] == {"pair_id": event.query_id, "receipt_sha256": "e" * 64}
        assert refs["S1"] == {"pair_id": event.query_id, "receipt_sha256": "f" * 64}
    assert candidate.excluded_pairs == ()


def test_development_synthesis_uses_each_context_stratum_and_one_value_per_family(monkeypatch):
    captured = {}

    def synthesize(cases, **kwargs):
        captured["cases"] = cases
        return SynthesisResult("insufficient_evidence", None, (), ())

    monkeypatch.setattr(bridge, "synthesize_applicability", synthesize)
    pairs = []
    for family in ("family-a", "family-b"):
        for batch in (1, 8):
            for session in range(2):
                pair = _pair(
                    f"{family}-batch-{batch}-session-{session}",
                    family, "development", s0=2.0, s1=1.5 + session * 0.1,
                )
                pair["features"]["batch"] = batch
                pairs.append(pair)

    candidate = _development(_engine(), pairs)

    cases = captured["cases"]
    assert len(cases) == 2
    assert {case["context"]["workload"]["batch"] for case in cases} == {1, 8}
    assert all(case["cluster_families"] == ["family-a", "family-b"] for case in cases)
    assert all(len(case["intervention_measurements"]) == 2 for case in cases)
    assert all(len(case["baseline_measurements"]) == 2 for case in cases)
    for case in cases:
        batch = case["context"]["workload"]["batch"]
        family_effects = [
            sum(bridge._pair_utility(pair) for pair in pairs
                if pair["family_id"] == family and pair["features"]["batch"] == batch) / 2
            for family in ("family-a", "family-b")
        ]
        encoded_effects = [
            utility_effect(1.0, baseline, higher_is_better=False, log_scale=math.log(2.0))
            for baseline in case["baseline_measurements"]
        ]
        assert encoded_effects == pytest.approx(family_effects)
    assert candidate.development_families == {"family-a", "family-b"}


def test_invalid_development_receipt_does_not_add_its_family_to_the_denominator():
    valid = _pair("dev-valid", "family-valid", "development")
    invalid = _pair(
        "dev-invalid", "family-invalid", "development",
        status="NUMERIC_MISMATCH", correctness="FAIL",
    )
    candidate = _development(_engine(), [valid, invalid])

    assert candidate.development_families == {"family-valid"}
    assert candidate.development_pair_ids == {"dev-valid", "dev-invalid"}
    assert {event.independence_group for event in candidate.events} == {
        "family-valid", "family-invalid",
    }


def test_post_execution_features_are_rejected_even_if_declared():
    with pytest.raises(ValueError, match="post-execution"):
        build_development_candidate(
            engine=_engine(), rule_id="CHANNELS_LAST", development_pairs=[],
            candidate_features={"observed_latency": 1.5},
            pre_execution_features=("observed_latency",), family_id="compile",
            support_scope=SCOPE, runtime_versions=RUNTIME,
            hardware={"gpu_uuid": "GPU-test"}, software={"torch": "2.6.0"},
            source_id="v4-development", timestamp="2026-10-19T00:00:00Z",
        )


def test_development_rejects_calibration_and_final_rows():
    for split in ("calibration", "final"):
        with pytest.raises(ValueError, match="development"):
            _development(_engine(), [_pair(f"{split}-1", "family-x", split)])


def test_known_correctness_failure_is_adversarial_not_a_positive_candidate():
    candidate = _development(_engine(), [
        _pair("dev-bad", "family-dev", "development",
              status="NUMERIC_MISMATCH", correctness="FAIL"),
    ])
    assert candidate.candidate_spec is None
    assert candidate.excluded_pairs == (("dev-bad", "NUMERIC_MISMATCH"),)
    assert len(candidate.events) == 2
    assert all(event.evidence_role == "adversarial" for event in candidate.events)


def test_baseline_correctness_failure_is_excluded_without_candidate_adversarial_evidence():
    pair = _pair("dev-baseline-bad", "family-dev", "development")
    pair["S0"]["scientific_correctness"] = "FAIL"
    pair["classification"]["scientific_correctness"] = "FAIL"

    candidate = _development(_engine(), [pair])

    assert candidate.excluded_pairs == (("dev-baseline-bad", "VALID"),)
    assert candidate.events == ()


def test_calibration_is_family_disjoint_and_uses_validation_evidence_only():
    engine = _engine()
    candidate = _development(engine, [_pair("dev-1", "family-dev", "development")])
    calibration = [
        _pair(f"cal-{index}", f"family-cal-{index}", "calibration")
        for index in range(5)
    ]
    result = validate_calibration_and_route(
        engine=engine,
        candidate=candidate,
        calibration_pairs=calibration,
        candidate_features={
            "batch": 1,
            "horizon": 16,
            "conv2d_module_count": 4,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
        },
        pre_execution_features=FEATURES,
        support_scope=SCOPE,
        runtime_versions=RUNTIME,
        hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"},
        source_id="v4-calibration",
        timestamp="2026-10-19T00:01:00Z",
        min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in calibration],
    )

    assert result.status == "NOT_FITTED"
    assert result.reason.startswith(
        "candidate, independent calibration, canonical governance, scope, or route gate did not pass:"
    )
    assert result.route.selected_rule_ids == ()
    assert result.calibration_families == {f"family-cal-{index}" for index in range(5)}
    assert all(event.evidence_role == "validation" for event in result.events)
    assert result.assessment["representative_count"] == 0
    assert result.validation_certificates == ()
    assert result.governance_errors == ()


def test_calibration_rejects_development_family_reuse_and_final_rows():
    candidate = _development(
        _engine(), [_pair("dev-1", "family-dev", "development")],
    )
    args = dict(
        engine=_engine(), candidate=candidate,
        candidate_features={
            "batch": 1,
            "horizon": 16,
            "conv2d_module_count": 4,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
        },
        pre_execution_features=FEATURES, support_scope=SCOPE,
        runtime_versions=RUNTIME, hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"}, source_id="v4-calibration",
        timestamp="2026-10-19T00:01:00Z", min_calibration_families=5,
        planned_calibration_pair_ids=["cal-1"],
    )
    with pytest.raises(ValueError, match="family crosses"):
        validate_calibration_and_route(
            **args,
            calibration_pairs=[_pair("cal-1", "family-dev", "calibration")],
        )
    with pytest.raises(ValueError, match="calibration"):
        validate_calibration_and_route(
            **args,
            calibration_pairs=[_pair("final-1", "family-final", "final")],
        )


def test_calibration_infrastructure_missingness_cannot_become_harm_or_fit():
    engine = _engine()
    candidate = _development(engine, [_pair("dev-1", "family-dev", "development")])
    failed = [
        _pair(f"cal-{index}", f"family-cal-{index}", "calibration",
              status="INFRA", s0=0.0, s1=0.0)
        for index in range(5)
    ]
    result = validate_calibration_and_route(
        engine=engine,
        candidate=candidate,
        calibration_pairs=failed,
        candidate_features={
            "batch": 1,
            "horizon": 16,
            "conv2d_module_count": 4,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
        },
        pre_execution_features=FEATURES,
        support_scope=SCOPE,
        runtime_versions=RUNTIME,
        hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"},
        source_id="v4-calibration",
        timestamp="2026-10-19T00:01:00Z",
        min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in failed],
    )
    assert result.status == "NOT_FITTED"
    assert result.events == ()
    assert result.regime.action == "abstain"
    assert len(result.excluded_pairs) == 5


def test_calibration_promotes_reloads_and_routes_only_with_disjoint_family_pools(
    tmp_path, monkeypatch,
):
    engine = _stored_engine(tmp_path)
    candidate = _identified_candidate(engine)
    monkeypatch.setattr(bridge, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(regimes, "_identified_synthesis", lambda result: result is not None)
    monkeypatch.setattr(regimes, "_scope_digest", lambda scope: "scope")
    monkeypatch.setattr(regimes, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(typed_gate, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(typed_gate, "context_in_decision_lattice", lambda *args: True)
    context_deltas = []

    def family_cluster_interval(values, delta):
        context_deltas.append(delta)
        mean = sum(values) / len(values)
        return mean, mean

    monkeypatch.setattr(bridge, "bounded_mean_interval", family_cluster_interval)
    # If the implementation routes against its pre-promotion object, this fails.
    engine.route = lambda *args, **kwargs: pytest.fail("route must use the reloaded store")
    pairs = _calibration_pairs(46)
    result = validate_calibration_and_route(
        engine=engine,
        candidate=candidate,
        calibration_pairs=pairs,
        candidate_features={
            # FITTED certifies promotion, not this arbitrary query's route.
            "batch": 8, "horizon": 16, "conv2d_module_count": 4,
            "input_channels": 3, "input_height": 224, "input_width": 224,
        },
        pre_execution_features=FEATURES,
        support_scope=SCOPE,
        runtime_versions=RUNTIME,
        hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"},
        source_id="synthetic-unit-fixture",
        timestamp="2026-10-19T00:01:00Z",
        min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in pairs],
    )

    reloaded = AcreEngine.from_store(tmp_path)
    assert result.status == "FITTED"
    assert result.route.selected_rule_ids == ()
    assert reloaded.rule_states["CHANNELS_LAST"].status == "canonical"
    assert reloaded.rule_states["CHANNELS_LAST"].version == 2
    assert result.assessment["representative_count"] == 0
    assert len({event.independence_group for event in result.events}) == len(result.calibration_families)
    assert len(result.validation_certificates) == len(result.calibration_families)
    assert context_deltas[:8] == pytest.approx([0.05 / 4 / 8] * 8)
    validation = json.loads(
        (tmp_path / "evolution" / "validation" / "CHANNELS_LAST-v2.json").read_text()
    )
    context_evidence = validation["context_strata_validation"]
    assert len(context_evidence) == 8
    assert all(len(cell["independence_groups"]) == 46 for cell in context_evidence)
    assert all(len({ref["family_id"] for ref in cell["paired_receipt_refs"]}) == 46
               for cell in context_evidence)
    assert result.reason == "scope-bound canonical promotion established; route each target context separately"
    boundary_context = result.task_context.to_dict()
    boundary_context["workload"]["batch"] = 8
    boundary_route = reloaded.route(
        TaskContext(**boundary_context),
        support_regimes={"CHANNELS_LAST": result.regime},
        runtime_versions=RUNTIME,
        candidate_ids=["CHANNELS_LAST"],
    )
    assert boundary_route.selected_rule_ids == ()
    support_context = result.task_context.to_dict()
    support_context["workload"]["batch"] = 1
    support_route = reloaded.route(
        TaskContext(**support_context),
        support_regimes={"CHANNELS_LAST": result.regime},
        runtime_versions=RUNTIME,
        candidate_ids=["CHANNELS_LAST"],
    )
    assert support_route.selected_rule_ids == ("CHANNELS_LAST",)


def test_calibration_insufficient_promotion_families_abstains_without_mutation(
    tmp_path, monkeypatch,
):
    engine = _stored_engine(tmp_path)
    candidate = _identified_candidate(engine)
    monkeypatch.setattr(bridge, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(regimes, "_identified_synthesis", lambda result: result is not None)
    monkeypatch.setattr(regimes, "_scope_digest", lambda scope: "scope")
    monkeypatch.setattr(regimes, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(typed_gate, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(typed_gate, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(
        bridge, "bounded_mean_interval",
        lambda values, delta: (sum(values) / len(values), sum(values) / len(values)),
    )
    pairs = _calibration_pairs(10)
    result = validate_calibration_and_route(
        engine=engine, candidate=candidate, calibration_pairs=pairs,
        candidate_features={"batch": 1, "horizon": 16, "conv2d_module_count": 4,
                           "input_channels": 3, "input_height": 224, "input_width": 224},
        pre_execution_features=FEATURES, support_scope=SCOPE,
        runtime_versions=RUNTIME, hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"}, source_id="synthetic-unit-fixture",
        timestamp="2026-10-19T00:01:00Z", min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in pairs],
    )

    assert result.status == "NOT_FITTED"
    assert result.regime.action == "reuse"
    assert result.route.selected_rule_ids == ()
    assert engine.rule_states["CHANNELS_LAST"].status == "canonical"
    assert len([path for path in (tmp_path / "evolution" / "promotions").rglob("*.json")]) == 0


def test_calibration_does_not_certify_a_family_missing_a_frozen_context_cell(
    tmp_path, monkeypatch,
):
    engine = _stored_engine(tmp_path)
    candidate = _identified_candidate(engine)
    monkeypatch.setattr(bridge, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(regimes, "_identified_synthesis", lambda result: result is not None)
    monkeypatch.setattr(regimes, "_scope_digest", lambda scope: "scope")
    monkeypatch.setattr(regimes, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(typed_gate, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(typed_gate, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(
        bridge, "bounded_mean_interval",
        lambda values, delta: (sum(values) / len(values), sum(values) / len(values)),
    )
    pairs = _calibration_pairs(10)
    missing = next(
        index for index, pair in enumerate(pairs)
        if pair["family_id"] == "family-cal-0"
        and pair["features"]["batch"] == 8
        and pair["features"]["horizon"] == 1024
    )
    pairs.pop(missing)
    result = validate_calibration_and_route(
        engine=engine, candidate=candidate, calibration_pairs=pairs,
        candidate_features={"batch": 1, "horizon": 16, "conv2d_module_count": 4,
                            "input_channels": 3, "input_height": 224, "input_width": 224},
        pre_execution_features=FEATURES, support_scope=SCOPE,
        runtime_versions=RUNTIME, hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"}, source_id="synthetic-unit-fixture",
        timestamp="2026-10-19T00:01:00Z", min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in pairs],
    )

    assert result.status == "NOT_FITTED"
    assert result.regime.action == "abstain"
    assert result.route.selected_rule_ids == ()


def test_calibration_family_contributes_across_contexts_but_counts_once(
    tmp_path, monkeypatch,
):
    engine = _stored_engine(tmp_path)
    candidate = _identified_candidate(engine)
    monkeypatch.setattr(bridge, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(regimes, "_identified_synthesis", lambda result: result is not None)
    monkeypatch.setattr(regimes, "_scope_digest", lambda scope: "scope")
    monkeypatch.setattr(regimes, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(typed_gate, "synthesis_certificate_valid", lambda result: True)
    monkeypatch.setattr(typed_gate, "context_in_decision_lattice", lambda *args: True)
    monkeypatch.setattr(
        bridge, "bounded_mean_interval",
        lambda values, delta: (sum(values) / len(values), sum(values) / len(values)),
    )
    pairs = _calibration_pairs(10)
    mixed = _pair("cal-mixed-2", "family-cal-0", "calibration", s0=1.0, s1=2.0)
    mixed["features"]["batch"] = 8
    pairs.append(mixed)
    result = validate_calibration_and_route(
        engine=engine, candidate=candidate, calibration_pairs=pairs,
        candidate_features={"batch": 1, "horizon": 16, "conv2d_module_count": 4,
                           "input_channels": 3, "input_height": 224, "input_width": 224},
        pre_execution_features=FEATURES, support_scope=SCOPE,
        runtime_versions=RUNTIME, hardware={"gpu_uuid": "GPU-test"},
        software={"torch": "2.6.0"}, source_id="synthetic-unit-fixture",
        timestamp="2026-10-19T00:01:00Z", min_calibration_families=5,
        planned_calibration_pair_ids=[pair["pair_id"] for pair in pairs],
    )

    assert result.status == "NOT_FITTED"
    assert result.regime.action == "reuse"
    assert result.route.selected_rule_ids == ()
    assert len(result.calibration_families) == 10
    assert len(result.validation_certificates) == 10
