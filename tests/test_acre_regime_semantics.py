from __future__ import annotations

from dataclasses import replace

from core.acre.cegis import BoundaryObservation, StatisticalCEGIS, SynthesisResult
from core.acre.predicates import PredicateGrammar
from core.acre.regimes import (
    SupportGroup,
    SupportRegime,
    SupportScope,
    classify_support_regime,
)
from core.acre.typed_gate import decision_from_regime, decision_from_synthesis
from core.acre.engine import AcreEngine
from core.models import RuleSpec, RuleState
from core.predicates import match_predicate


VERSIONS = {"environment": "env-1", "model": "model-1"}
SCOPE = SupportScope(
    population_id="compile-v1",
    population_predicate={"equals": {"domain": "compile"}},
    versions=VERSIONS,
    required_strata=("small", "large"),
)


def _group(group_id: str, effect: float, *, stratum: str = "small") -> SupportGroup:
    return SupportGroup(
        group_id,
        {"domain": "compile"},
        effect,
        effect,
        True,
        stratum,
        VERSIONS,
    )


def _identified_result(scope: SupportScope = SCOPE) -> SynthesisResult:
    domain = scope.population_predicate.get("equals", {}).get("domain", "runtime")
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "workload.x", "type": "numeric"}],
        "threshold_universe": {"workload.x": [0.0]},
    })
    positive = BoundaryObservation("p", {"domain": domain, "workload": {"x": 1}}, 0.5, True, 0.2, 0.8)
    negative = BoundaryObservation("n", {"domain": domain, "workload": {"x": -1}}, -0.5, True, -0.8, -0.2)
    return StatisticalCEGIS(grammar).synthesize(
        positive=[positive],
        counterexamples=[negative],
        parent_predicate=None,
        decision_contexts=[positive.context, negative.context],
        certificate_scope=scope.to_dict(),
    )


def test_positive_evidence_alone_never_certifies_global_reuse() -> None:
    groups = [_group(f"g-{index}", 0.9, stratum="small" if index % 2 else "large") for index in range(8)]
    result = classify_support_regime(
        groups,
        scope=SCOPE,
        target_context={"domain": "compile"},
        runtime_versions=VERSIONS,
        fixed_time_complete=True,
        min_groups=2,
        max_harm_probability=1.0,
    )
    assert result.regime == SupportRegime.UNCERTAIN
    assert result.action == "abstain"


def test_global_certificate_is_fixed_time_scoped_and_versioned() -> None:
    groups = [
        _group(f"g-{index}", 1.0, stratum="small" if index % 2 else "large")
        for index in range(400)
    ]
    result = classify_support_regime(
        groups,
        scope=SCOPE,
        target_context={"domain": "compile"},
        runtime_versions=VERSIONS,
        fixed_time_complete=True,
        min_groups=20,
        min_net_benefit=0.0,
        max_harm_probability=0.1,
    )
    assert result.regime == SupportRegime.GLOBAL_VERIFIED
    assert result.allows_reuse({"domain": "compile"}, VERSIONS)
    assert not result.allows_reuse({"domain": "other"}, VERSIONS)
    assert not result.allows_reuse({"domain": "compile"}, {**VERSIONS, "model": "model-2"})

    incomplete = classify_support_regime(
        groups,
        scope=SCOPE,
        target_context={"domain": "compile"},
        runtime_versions=VERSIONS,
        fixed_time_complete=False,
        min_groups=20,
        max_harm_probability=0.1,
    )
    assert incomplete.regime == SupportRegime.UNCERTAIN

    stratum_harm = [
        _group(f"s-{index}", -0.5 if index % 2 else 1.0, stratum="small" if index % 2 else "large")
        for index in range(400)
    ]
    blocked = classify_support_regime(
        stratum_harm,
        scope=SCOPE,
        target_context={"domain": "compile"},
        runtime_versions=VERSIONS,
        fixed_time_complete=True,
        min_groups=20,
        min_groups_per_stratum=20,
        min_net_benefit=-0.1,
        max_harm_probability=0.1,
    )
    assert blocked.regime == SupportRegime.UNCERTAIN


def test_hard_gate_failure_rejects_and_selective_support_stays_local() -> None:
    hard_failure = SupportGroup(
        "bad", {"domain": "compile"}, 0.5, 0.8, False,
        "small", VERSIONS,
    )
    rejected = classify_support_regime(
        [hard_failure], scope=SCOPE,
        target_context={"domain": "compile"}, runtime_versions=VERSIONS,
    )
    assert rejected.regime == SupportRegime.REJECTED

    synthesis = _identified_result()
    assert synthesis.status == "identified"
    selected = classify_support_regime(
        [], scope=SCOPE, target_context={"domain": "compile"},
        runtime_versions=VERSIONS, selective=synthesis,
    )
    assert selected.regime == SupportRegime.SELECTIVE_IDENTIFIED
    assert decision_from_regime(
        selected, {"domain": "compile", "workload": {"x": 1}}, VERSIONS,
    ).action == "reuse"
    assert decision_from_regime(
        selected, {"domain": "compile", "workload": {"x": -1}}, VERSIONS,
    ).action == "abstain"
    tampered = replace(selected, predicate={"all": []})
    assert not tampered.allows_reuse({"domain": "compile", "workload": {"x": 1}}, VERSIONS)

    changed_scope = SupportScope(
        "compile-v2", SCOPE.population_predicate,
        {**VERSIONS, "skill": "skill-v2"}, SCOPE.required_strata,
    )
    transplanted = classify_support_regime(
        [], scope=changed_scope, target_context={"domain": "compile"},
        runtime_versions=changed_scope.versions, selective=synthesis,
    )
    assert transplanted.regime == SupportRegime.UNCERTAIN


def test_candidate_search_exhaustion_cannot_be_identified_or_deployed() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [
            {"path": f"x{i}", "type": "categorical"} for i in range(4)
        ],
    })
    observations = [
        BoundaryObservation("p", {f"x{i}": "yes" for i in range(4)}, 0.5, True, 0.2, 0.8),
        BoundaryObservation("n", {f"x{i}": "no" for i in range(4)}, -0.5, True, -0.8, -0.2),
    ]
    result = StatisticalCEGIS(grammar, max_candidates=1).synthesize(
        positive=observations[:1], counterexamples=observations[1:],
        parent_predicate=None,
        decision_contexts=[item.context for item in observations],
    )
    assert result.status == "search_exhausted"
    assert result.search_exhausted
    assert result.candidates_examined == 1
    assert decision_from_synthesis(result, observations[0].context).action == "abstain"


def test_incremental_threshold_vocabulary_is_reenumerated_without_cache_staleness() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "x", "type": "numeric"}],
    })
    synthesizer = StatisticalCEGIS(grammar)
    positive = BoundaryObservation("p", {"x": 2.0}, 0.5, True, 0.2, 0.8)
    first_negative = BoundaryObservation("n1", {"x": 0.0}, -0.5, True, -0.8, -0.2)
    first = synthesizer.synthesize(
        positive=[positive], counterexamples=[first_negative], parent_predicate=None,
    )
    assert first.status == "underidentified"

    new_negative = BoundaryObservation("n2", {"x": 1.5}, -0.5, True, -0.8, -0.2)
    incremental = synthesizer.synthesize(
        positive=[positive], counterexamples=[first_negative, new_negative], parent_predicate=None,
    )
    fresh = StatisticalCEGIS(grammar).synthesize(
        positive=[positive], counterexamples=[first_negative, new_negative], parent_predicate=None,
    )
    assert incremental.status == "underidentified"
    assert incremental.predicate == fresh.predicate
    assert incremental.version_space == fresh.version_space


def test_decision_lattice_features_do_not_leak_into_inductive_vocabulary() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "x", "type": "numeric"}],
    })
    positive = BoundaryObservation("p", {"x": 10.0}, 0.5, True, 0.2, 0.8)
    negative = BoundaryObservation("n", {"x": 0.0}, -0.5, True, -0.8, -0.2)

    def synthesize(lattice: list[dict[str, float]]) -> SynthesisResult:
        return StatisticalCEGIS(grammar).synthesize(
            positive=[positive], counterexamples=[negative], parent_predicate=None,
            decision_contexts=lattice,
        )

    first = synthesize([{"x": 2.0}, {"x": 8.0}])
    perturbed = synthesize([{"x": -100.0}, {"x": 500.0}])
    assert first.provenance["hypothesis_vocabulary_digest"] == perturbed.provenance["hypothesis_vocabulary_digest"]
    assert first.provenance["decision_contexts_in_vocabulary"] is False

    transductive_a = StatisticalCEGIS(grammar).synthesize(
        positive=[positive], counterexamples=[negative], parent_predicate=None,
        decision_contexts=[{"x": 2.0}, {"x": 8.0}],
        vocabulary_contexts=[{"x": 2.0}, {"x": 8.0}],
    )
    transductive_b = StatisticalCEGIS(grammar).synthesize(
        positive=[positive], counterexamples=[negative], parent_predicate=None,
        decision_contexts=[{"x": -100.0}, {"x": 500.0}],
        vocabulary_contexts=[{"x": -100.0}, {"x": 500.0}],
    )
    assert transductive_a.provenance["hypothesis_vocabulary_digest"] != transductive_b.provenance["hypothesis_vocabulary_digest"]


def test_training_equivalent_predicates_are_not_collapsed_before_lattice_check() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "x", "type": "numeric"}],
        "threshold_universe": {"x": [-1.0, 0.0]},
    })
    positive = BoundaryObservation("p", {"x": 1.0}, 0.5, True, 0.2, 0.8)
    negative = BoundaryObservation("n", {"x": -3.0}, -0.5, True, -0.8, -0.2)
    result = StatisticalCEGIS(grammar).synthesize(
        positive=[positive], counterexamples=[negative], parent_predicate=None,
        decision_contexts=[{"x": -0.5}, positive.context, negative.context],
    )
    assert result.status == "underidentified"
    assert len(result.version_space) > 1
    assert any(
        left != right and any(
            bool(match_predicate(left, context)) != bool(match_predicate(right, context))
            for context in ({"x": -0.5},)
        )
        for left in result.version_space for right in result.version_space
    )


def test_engine_keeps_canonical_gate_and_rejects_changed_skill_digest() -> None:
    skill_digest = "contract-v1"
    scoped_versions = {**VERSIONS, "skill_digest": skill_digest}
    scope = SupportScope(
        "runtime-skill-v1", {"equals": {"domain": "runtime"}}, scoped_versions,
    )
    synthesis = _identified_result(scope)
    regime = classify_support_regime(
        [], scope=scope,
        target_context={"domain": "runtime", "workload": {"x": 1}},
        runtime_versions=scoped_versions,
        selective=synthesis,
    )
    rule = RuleSpec(
        rule_id="RULE-REGIME", version=1, parent=None,
        applicability={"all": []},
        intervention={"action": "reuse", "contract_digest": skill_digest},
        expected_mechanism="measured", evidence_requirements=["paired replay"],
        scientific_invariants=[], abstain_conditions={}, relations={},
        runtime_cost={"tokens": 1}, provenance_policy={"required": True},
    )
    context = {"domain": "runtime", "workload": {"x": 1}}

    candidate_engine = AcreEngine(
        rule_specs=[rule],
        rule_states={"RULE-REGIME": RuleState("RULE-REGIME", 1, status="candidate")},
    )
    candidate = candidate_engine.route(
        context, support_regimes={"RULE-REGIME": regime}, runtime_versions=VERSIONS,
    )
    assert candidate.selected_rule_ids == ()
    assert "rule_state_ineligible" in candidate.rejected_reasons["RULE-REGIME"]

    changed_rule = RuleSpec(
        **{**rule.__dict__, "intervention": {"action": "reuse", "contract_digest": "contract-v2"}}
    )
    canonical_engine = AcreEngine(
        rule_specs=[changed_rule],
        rule_states={"RULE-REGIME": RuleState("RULE-REGIME", 1, status="canonical", effect={"lower_utility": 1.0}, retrieval_utility=1.0)},
    )
    changed = canonical_engine.route(
        context, support_regimes={"RULE-REGIME": regime}, runtime_versions=VERSIONS,
    )
    assert changed.selected_rule_ids == ()
    assert changed.rejected_reasons["RULE-REGIME"] == ("support_regime_not_deployable",)


def test_diagnostic_underidentified_match_is_never_operational_reuse() -> None:
    grammar = PredicateGrammar.from_dict({
        "schema_version": 1,
        "features": [{"path": "x", "type": "numeric"}],
        "threshold_universe": {"x": [0.0]},
    })
    underidentified = StatisticalCEGIS(grammar).synthesize(
        positive=[BoundaryObservation("p", {"x": 1}, 0.5, True, 0.2, 0.8)],
        counterexamples=[BoundaryObservation("n", {"x": -1}, -0.5, True, -0.8, -0.2)],
        parent_predicate=None,
    )
    assert underidentified.status == "underidentified"
    candidate_context = {"x": 1}
    assert decision_from_synthesis(underidentified, candidate_context, mode="diagnostic").action == "hypothesis_match"
    assert decision_from_synthesis(underidentified, candidate_context, mode="diagnostic").deployable is False
    assert decision_from_synthesis(underidentified, candidate_context, mode="operational").action == "abstain"


def test_synthesis_serialization_is_compact_and_reads_legacy_full_space() -> None:
    result = _identified_result()
    compact = result.to_dict()
    assert compact["search_exhausted"] is False
    assert compact["candidates_examined"] == result.candidates_examined
    assert compact["version_space_size"] == len(result.version_space)
    assert compact["version_space_digest"]
    assert "version_space" not in compact
    assert "version_space" not in (compact["provenance"] or {})

    debug = result.to_dict(include_version_space=True)
    assert debug["version_space"] == list(result.version_space)
    assert "version_space" not in (debug["provenance"] or {})

    legacy = {
        "status": result.status,
        "predicate": result.predicate,
        "certified_counterexamples": list(result.certified_counterexamples),
        "positive_anchors": list(result.positive_anchors),
        "synthesizer_version": "acre-predicate-synth-v2",
        "provenance": {"version_space": list(result.version_space)},
        "version_space_size": len(result.version_space),
        "version_space": list(result.version_space),
        "certificate": None,
    }
    restored = SynthesisResult.from_dict(legacy)
    assert restored.version_space == result.version_space
    assert restored.synthesizer_version == "acre-predicate-synth-v2"
