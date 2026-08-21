from __future__ import annotations

import json
from pathlib import Path

from core.acre.engine import AcreEngine
from core.acre.experiments import ExperimentPlan, ReplaySequentialCertificate, execute_paired_plan
from core.models import RuleSpec, RuleState, identifier_digest
from core.mutation_journal import MutationJournal
from core.state_store import StateStore


class _Executor:
    def execute(self, context, *, arm="on"):
        value = 0.2 if arm == "on" else 0.0
        return {"measurements": [value, value + 0.001], "scientific_ok": True}


def test_paired_execution_emits_canonical_evidence_events():
    cases = []
    execution = execute_paired_plan(
        ExperimentPlan("RULE-1", ({"context_id": "CTX-1", "independence_group": "G-1", "context": {"workload": {"x": 1}}},), max_groups=1),
        _Executor(), record_case=cases.append,
        update_certificate=lambda current: {"status": "collecting"},
    )
    assert execution.evidence_events
    assert all(event.schema_version == 2 for event in execution.evidence_events)
    assert all(event.assignment["interventions"] == {"RULE-1": arm} for event in execution.evidence_events for arm in (0, 1) if list(event.assignment["interventions"].values())[0] == arm)


def test_lifecycle_transition_is_persisted_and_reloadable(tmp_path: Path):
    store = StateStore(tmp_path)
    journal = MutationJournal(tmp_path / "evolution" / "mutation_journal.jsonl")
    old = RuleState("RULE-1", 1, "canonical", "suspected_drift")
    new = RuleState("RULE-1", 1, "canonical", "revalidating")
    from core.governance import EvolutionDecision
    decision = EvolutionDecision("rule", "RULE-1", "REVALIDATE", "review_required", "human-review", "drift")
    path, old_digest, new_digest = store.apply_transition(old, new, decision=decision, journal=journal)
    assert json.loads(path.read_text(encoding="utf-8"))["drift_state"] == "revalidating"
    assert old_digest != new_digest
    journal.verify()
    assert store.load("rule", "RULE-1", 1, spec_path=path).drift_state == "revalidating"


def test_engine_restart_reads_persisted_lifecycle_state(tmp_path: Path):
    rule = RuleSpec(
        rule_id="RULE-1", version=1, parent=None, applicability={"all": []},
        intervention={"action": "reuse"}, expected_mechanism="mechanism",
        evidence_requirements=["paired_replay"], scientific_invariants=[],
        abstain_conditions={}, relations={}, runtime_cost={"tokens": 1},
        provenance_policy={"required": True},
    )
    rule_dir = tmp_path / "rules" / identifier_digest("RULE-1")
    rule_dir.mkdir(parents=True)
    (rule_dir / "v0001.json").write_text(json.dumps(rule.to_dict()), encoding="utf-8")
    (rule_dir / "v0001.state.json").write_text(json.dumps(RuleState("RULE-1", 1, "canonical", "suspected_drift").to_dict()), encoding="utf-8")
    first = AcreEngine.from_store(tmp_path)
    assert first.evolve("RULE-1").operation == "REVALIDATE"
    restarted = AcreEngine.from_store(tmp_path)
    assert restarted.rule_states["RULE-1"].drift_state == "revalidating"


def test_replay_certificate_uses_group_mixture_not_all_success_count():
    certificate = ReplaySequentialCertificate(minimum_groups=2, p_min=0.5, delta=0.05)
    cases = [
        {"case_id": "A", "independence_group": "G1", "query_type": "representative", "intervention_measurements": [0.9] * 64, "baseline_measurements": [0.1] * 64, "higher_is_better": True},
        {"case_id": "B", "independence_group": "G2", "query_type": "representative", "intervention_measurements": [0.9] * 64, "baseline_measurements": [0.1] * 64, "higher_is_better": True},
        {"case_id": "B-duplicate", "independence_group": "G2", "query_type": "representative", "intervention_measurements": [0.9] * 64, "baseline_measurements": [0.1] * 64, "higher_is_better": True},
    ]
    result = certificate.update(cases)
    assert result["groups"] == 2
    assert result["successes"] == 2


def test_evolution_episode_loads_replay_builder_once(tmp_path: Path, monkeypatch):
    from benchmark.harness import evolution

    repo_root = Path(__file__).resolve().parents[2]
    episode = repo_root / "benchmark" / "tasks" / "EVOL-COMPILER-DRIFT-20" / "episodes" / "compiler_drift_episode.yaml"
    calls = []
    original_import = evolution._import_core_replay_build_manifest

    def counted_import(*args, **kwargs):
        calls.append(str(args[0]))
        return original_import(*args, **kwargs)

    monkeypatch.setattr(evolution, "_import_core_replay_build_manifest", counted_import)
    monkeypatch.setattr(evolution, "EPISODE_REPLAY_REPETITIONS", 1)
    evolution.run_episode(
        episode,
        "D",
        tmp_path / "candidate",
        core_repo=repo_root,
        snapshot_dir=repo_root,
        seed=0,
        max_wall_time_s=600,
    )

    assert len(calls) == 1


def test_higher_order_execution_registers_typed_router_certificate():
    from core.acre.engine import AcreEngine
    engine = AcreEngine()
    contexts = [{"context_id": f"CTX-3-{index}", "bundle_ids": ["A", "B", "C"], "rule_versions": {"A": 1, "B": 1, "C": 1}} for index in range(1024)]
    cert = engine.maintainer.execute_higher_order_experiment(
        contexts,
        lambda _context: {
            "outcomes": {arm: 0.0 for arm in ("000", "001", "010", "011", "100", "101", "110", "111")},
            "scientific_gates": {arm: True for arm in ("000", "001", "010", "011", "100", "101", "110", "111")},
        },
        practical_margin=0.3,
    )
    assert cert["status"] == "pairwise_certified"
    assert cert["bundle_versions"] == {"A": 1, "B": 1, "C": 1}
    assert any(key.startswith("A:B:C:") for key in engine.higher_order_certificates)
