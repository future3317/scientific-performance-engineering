from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.acre.experiments import ExperimentPlan, ReplaySequentialCertificate, execute_paired_plan
from core.acre.evidence import assess
from core.models import EvidenceEvent, RuleState, identifier_digest
from core.mutation_journal import MutationJournal
from core.public_context import build_public_context
from core.state_store import StateStore
from benchmark.formal.schedule import FamilyReplayExecutor, PromotionReplayScheduler


class _Executor:
    def execute(self, context, *, arm="on"):
        base = 0.8 if arm == "on" else 0.2
        return {"measurements": [base, base], "scientific_ok": True}


def test_paired_execution_exposes_one_group_contrast_envelope():
    cases = []
    execution = execute_paired_plan(
        ExperimentPlan("RULE-C", ({"context_id": "ctx", "independence_group": "g", "context": {"workload": {"x": 1}}},), max_groups=1),
        _Executor(), record_case=cases.append, update_certificate=lambda _: {"status": "collecting"},
    )
    assert len(execution.evidence_events) == 1
    event = execution.evidence_events[0]
    assert event.artifacts["paired_contrast"]["effect"] > 0
    assert event.artifacts["paired_contrast"]["source_on_event_ids"]
    assert event.artifacts["paired_contrast"]["source_off_event_ids"]


def test_replay_scheduler_can_recover_after_failures():
    scheduler = PromotionReplayScheduler(p_min=0.8, delta=0.05)
    assert scheduler.max_groups >= scheduler.minimum_groups * 3
    contexts = scheduler.pending_contexts("compile", seed=3)
    assert len(contexts) >= scheduler.max_groups


def test_group_seed_is_independent_but_arms_are_paired():
    executor = FamilyReplayExecutor("compile", "reuse_compile_cache", repetitions=4)
    first = executor.execute({"context_id": "c1", "independence_group": "g1", "context": {"workload": {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.2}}}, arm="on")
    second = executor.execute({"context_id": "c1", "independence_group": "g1", "context": {"workload": {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.2}}}, arm="off")
    other = executor.execute({"context_id": "c2", "independence_group": "g2", "context": {"workload": {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.2}}}, arm="on")
    assert first["measurements"] != other["measurements"]
    assert first["measurements"] != second["measurements"]


def test_public_context_flattens_family_parameters():
    params = {"logical_steps": 128, "dynamic_shape_rate": 0.2}
    public = build_public_context({"family_parameters": params})
    assert public == {"workload": params}


def test_mutable_journal_validates_transition_chain(tmp_path: Path):
    store = StateStore(tmp_path)
    journal = MutationJournal(tmp_path / "journal.jsonl")
    old = RuleState("RULE-C", 1, "canonical", "stable")
    mid = RuleState("RULE-C", 1, "canonical", "revalidating")
    new = RuleState("RULE-C", 1, "canonical", "suspected_drift")
    from core.governance import EvolutionDecision
    store.apply_transition(old, mid, decision=EvolutionDecision("rule", "RULE-C", "REVALIDATE", "review_required", "human-review", "drift"), journal=journal)
    store.apply_transition(mid, new, decision=EvolutionDecision("rule", "RULE-C", "REVALIDATE", "review_required", "drift", "second"), journal=journal)
    journal.verify_against_store_diff(tmp_path)


def test_state_transition_uses_compare_and_swap(tmp_path: Path):
    store = StateStore(tmp_path)
    from core.governance import EvolutionDecision
    first = RuleState("RULE-CAS", 1, "canonical", "stable")
    next_state = RuleState("RULE-CAS", 1, "canonical", "revalidating")
    decision = EvolutionDecision("rule", "RULE-CAS", "REVALIDATE", "review_required", "human-review", "drift")
    store.apply_transition(first, next_state, decision=decision)
    with pytest.raises(ValueError, match="compare-and-swap"):
        store.apply_transition(first, RuleState("RULE-CAS", 1, "canonical", "stale"), decision=decision)


def test_higher_order_certificate_is_persisted_for_restart(tmp_path: Path):
    from core.acre.engine import AcreEngine
    from core.acre.factorial import HigherOrderCertificate
    engine = AcreEngine(state_store=StateStore(tmp_path))
    certificate = HigherOrderCertificate(
        bundle_versions={"A": 1, "B": 1, "C": 1},
        context_predicate={"all": []}, regime_digest="r",
            residual_lcb=-0.01, residual_ucb=0.01, normalized_residual=0.0,
            raw_residual=0.0, status="pairwise_certified",
            scientific_arm_gates={arm: True for arm in ("000", "001", "010", "011", "100", "101", "110", "111")},
    )
    key = engine.register_higher_order_certificate(certificate.to_dict())
    assert (tmp_path / "evolution" / "certificates" / f"{identifier_digest(key)}.json").is_file()


def test_raw_realization_precedes_action_classification(tmp_path: Path):
    from benchmark.formal.run_campaign import InterventionRealizer
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "solution.py").write_text("VALUE = 1\n", encoding="utf-8")
    destination = tmp_path / "realized"
    proposal = {"intervention": {"file": "solution.py", "replacements": [{"old": "VALUE = 1", "new": "VALUE = 2"}]}}
    with pytest.raises(ValueError, match="action_spec must be supplied"):
        InterventionRealizer.realize_action(baseline, destination, proposal, family_id="compile", task_id="TASK-1", context_id="CTX-1")
    assert (destination / "solution.py").read_text(encoding="utf-8") == "VALUE = 2\n"
