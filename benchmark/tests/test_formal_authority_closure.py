from __future__ import annotations

import pytest

from benchmark.families import family_decision_lattice, family_views
from benchmark.families.activation import classify_activation
from core.acre.budget import StatisticalBudget
from core.acre.cegis import synthesize_applicability
from core.acre.experiments import ExperimentPlan, execute_paired_plan
from core.models import EvidenceEvent
from core.mutation_journal import MutationJournal
from benchmark.formal.schedule import execute_required_experiments
from core.acre.actions import action_from_proposal
from core.acre.maintainer import AcreMaintainer
from core.acre.engine import AcreEngine


def test_family_surface_membership_is_frozen_and_disjoint() -> None:
    small = family_views("compile", count=24, seed=4)
    large = family_views("compile", count=264, seed=4)
    assert {item.instance_id for item in small["active_query_pool"]}.isdisjoint(
        {item.instance_id for item in large["representative_pool"]}
    )
    assert len(family_decision_lattice("compile", seed=4)) == 264


def test_cegis_returns_underidentified_result_with_version_space() -> None:
    result = synthesize_applicability(
        [{
            "case_id": "POS",
            "context": {"workload": {"logical_steps": 256, "dynamic_shape_rate": 0.2, "graph_size": 128}},
            "intervention_measurements": [0.8] * 64,
            "baseline_measurements": [0.5] * 64,
            "higher_is_better": True,
            "scientific_ok": True,
        }],
        family_id="compile",
    )
    assert result.status == "insufficient_evidence"
    assert result.version_space == ()


def test_statistical_budget_spends_group_alpha() -> None:
    budget = StatisticalBudget()
    assert budget.group_delta(1) > budget.group_delta(2)
    assert budget.group_delta(1) + budget.group_delta(2) < budget.group


def test_activation_requires_exactly_one_action() -> None:
    specs = {
        "reuse_compile_cache": {"activation_validator": "compile_cache_guard_hit"},
        "stabilize_dynamic_guards": {"activation_validator": "compile_dynamic_guard_stability"},
    }
    assert classify_activation("compile", specs, {"compile_cache_hit": True}, {"compile_cache_hit": False})["status"] == "passed"
    assert classify_activation("compile", specs, {"compile_cache_hit": True, "dynamic_guard_stable": True}, {"compile_cache_hit": False, "dynamic_guard_stable": False})["status"] == "rejected"


def test_active_query_is_synthesis_role_not_promotion_role() -> None:
    class Executor:
        def execute(self, context, *, arm="on"):
            value = 0.7 if arm == "on" else 0.5
            return {"measurements": [value, value], "scientific_ok": True}

    events = execute_paired_plan(
        ExperimentPlan("RULE-ROLE", ({"context_id": "Q", "query_type": "active_query", "context": {"workload": {"x": 1}}},), max_groups=1),
        Executor(), record_case=lambda _: None, update_certificate=lambda _: {"status": "collecting"},
    ).evidence_events
    assert events[0].evidence_role == "synthesis"


def test_evidence_role_rejects_stream_mismatch() -> None:
    payload = {
        "schema_version": 2, "event_id": "E-ROLE", "context": {"domain": "runtime"},
        "assignment": {"interventions": {"R": 1}, "propensity": 0.5, "design_id": "D"},
        "outcome_vector": {"utility": 0.1}, "scientific_gates": {"ok": True}, "artifacts": {}, "versions": {},
        "source_id": "S", "independence_group": "G", "timestamp": "2026-01-01T00:00:00Z",
        "evidence_stream": "representative", "evidence_role": "adversarial", "query_id": "Q",
        "trust_zone": "harness", "attacker_controlled_fields": [],
    }
    with pytest.raises(ValueError, match="adversarial evidence_role"):
        EvidenceEvent.from_dict(payload)


def test_required_experiment_never_accepts_unverified_or_partial_execution() -> None:
    request = {
        "experiment_id": "pair:R1:R2",
        "experiment_type": "pair_factorial",
        "bundle_ids": ["R1", "R2"],
        "required_arms": ["00", "10", "01", "11"],
    }
    blocked = execute_required_experiments([request], executor=lambda _: {
        "status": "executed", "execution_source": "synthetic_family",
    })
    assert blocked[0]["status"] == "blocked"
    partial = execute_required_experiments([request], executor=lambda _: {
        "status": "executed", "execution_source": "external_executor",
        "arm_evidence": {"00": {}}, "scientific_gates": {"00": True},
    })
    assert partial[0]["status"] == "blocked"


def test_mutation_journal_rejects_unjournaled_governed_artifact(tmp_path) -> None:
    journal = MutationJournal(tmp_path / "journal.jsonl")
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "rules.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unjournaled governed artifact"):
        journal.verify_against_store_diff(tmp_path)


def test_source_proposal_cannot_inherit_family_default_action() -> None:
    with pytest.raises(ValueError, match="explicit action_spec or source intervention"):
        action_from_proposal("compile", {"rule_id": "R-DEFAULT"})


def test_higher_order_execution_requires_all_arm_gates() -> None:
    maintainer = AcreMaintainer(AcreEngine())
    context = {"context_id": "three-way", "rule_versions": {"A": 1, "B": 1, "C": 1}}
    with pytest.raises(ValueError, match="scientific gates for all eight arms"):
        maintainer.execute_higher_order_experiment(
            [context],
            lambda _ctx: {
                "outcomes": {arm: 0.0 for arm in ("000", "001", "010", "011", "100", "101", "110", "111")},
                "scientific_gates": {"000": True},
            },
        )
