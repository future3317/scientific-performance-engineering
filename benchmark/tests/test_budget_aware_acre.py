from __future__ import annotations

from dataclasses import dataclass
import json


def test_execution_reserve_rejects_evolution_without_consuming_future_task_reserve():
    from core.acre.resource_budget import ExecutionReserveBudget

    ledger = ExecutionReserveBudget(total_budget=10.0, scheduled_minimums=(3.0, 3.0, 3.0))
    assert ledger.execution_reserve == 9.0
    assert ledger.evolution_available == 1.0

    next_ledger, decision = ledger.admit_evolution(2.0, query_id="q-expensive")
    assert decision.accepted is False
    assert decision.reason == "execution_reserve"
    assert next_ledger.evolution_spent == 0.0
    assert next_ledger.execution_reserve == 9.0


def test_execution_reserve_accounting_is_conserved_and_replayable():
    from core.acre.resource_budget import ExecutionReserveBudget

    initial = ExecutionReserveBudget(total_budget=20.0, scheduled_minimums=(4.0, 4.0))
    after_query, accepted = initial.admit_evolution(3.0, query_id="q1")
    after_task = after_query.advance_task(4.0)
    replay_query, replay_accepted = initial.admit_evolution(3.0, query_id="q1")
    replay_task = replay_query.advance_task(4.0)

    assert accepted.accepted and replay_accepted.accepted
    assert after_task == replay_task
    assert after_task.evolution_spent + after_task.execution_reserve + after_task.evolution_available == 20.0


def test_budget_exhaustion_freezes_evolution_but_not_scheduled_execution():
    from core.acre.resource_budget import ExecutionReserveBudget

    ledger = ExecutionReserveBudget(total_budget=5.0, scheduled_minimums=(1.0, 1.0))
    frozen, decision = ledger.admit_evolution(5.0, query_id="q-over-budget")
    assert not decision.accepted
    assert frozen.evolution_frozen is True
    assert frozen.scheduled_execution_allowed is True


def test_formal_evolution_exhaustion_policy_keeps_scheduled_cell_observed():
    from benchmark.formal.budget import evolution_exhaustion_policy

    policy = evolution_exhaustion_policy(exhausted=True)
    assert policy["status"] == "evolution_frozen"
    assert policy["scheduled_execution"] == "continue"
    assert policy["execution_validity"] == "valid"
    assert policy["rerunnable"] is False


def test_certificate_critical_acquisition_ignores_queries_that_cannot_change_decision():
    from core.acre.acquisition import AcquisitionPolicy, AcquisitionQuery, run_acquisition

    queries = [
        AcquisitionQuery("q-irrelevant", "edge", 1.0),
        AcquisitionQuery("q-critical", "edge", 1.0, can_change_decisions=("deploy",)),
    ]
    result = run_acquisition(
        queries,
        {"q-irrelevant": True, "q-critical": True},
        AcquisitionPolicy.CERTIFICATE_CRITICAL,
        confidence_target=0.5,
    )
    assert result.selected_query_ids == ("q-critical",)
    assert result.stop_reason == "no_critical_query"


def test_lazy_relation_returns_underidentified_without_running_factorial_engine(monkeypatch):
    from core.acre import maintainer as maintainer_module
    from core.acre.engine import AcreEngine
    from core.acre.relation import RelationExperimentEligibility

    @dataclass
    class ExplodingEngine:
        def __init__(self, **_kwargs):
            raise AssertionError("FactorialEngine must not run for an ineligible relation")

    monkeypatch.setattr("core.acre.factorial.FactorialEngine", ExplodingEngine)
    result = AcreEngine().maintainer.execute_relation_experiment(
        {"ctx": []},
        eligibility=RelationExperimentEligibility(False, "single_rule_gate_missing"),
    )
    assert result.decision == "underidentified_context_relation"
    assert result.confidence["reason"] == "single_rule_gate_missing"


def test_relation_scheduler_skips_pair_when_bundle_decision_cannot_change():
    from benchmark.formal.schedule import RelationExperimentScheduler

    def unexpected_blocks(_context, *, context_id):
        raise AssertionError(f"relation blocks unexpectedly executed: {context_id}")

    result = RelationExperimentScheduler().execute(
        {
            "relation_id": "REL-LAZY",
            "individual_gates": {"left": True, "right": True},
            "router_may_bundle": True,
            "changes_bundle_decision": False,
        },
        "compile",
        block_executor=unexpected_blocks,
        maintainer=object(),
    )
    assert result["status"] == "underidentified"
    assert result["reason"] == "relation_cannot_change_bundle"


def test_certificate_cache_reuses_only_unchanged_dependencies():
    from core.acre.certificate_cache import compare_certificate_dependencies

    old = {
        "context_fields": ["workload.shape"],
        "model_id": "m1",
        "software": "s1",
        "hardware": "h1",
        "scientific_policy": "strict-v1",
    }
    same = dict(old)
    changed = {**old, "software": "s2"}

    reused = compare_certificate_dependencies(old, same, evidence_id="e1")
    invalidated = compare_certificate_dependencies(old, changed, evidence_id="e1")
    assert reused.reusable is True
    assert reused.invalidated_clauses == ()
    assert reused.receipt["source_evidence_id"] == "e1"
    assert invalidated.reusable is False
    assert "E" in invalidated.invalidated_clauses
    assert "software" in invalidated.receipt["changed_dependencies"]


def test_provider_free_budget_ablation_keeps_legacy_reference_and_reports_savings():
    from scripts.run_pilot_surface_experiments import run_budget_aware_ablation

    report = run_budget_aware_ablation(seed=7)
    assert report["provider_calls"] == 0
    assert set(report["variants"]) == {
        "legacy_acre",
        "execution_reserve",
        "certificate_critical",
        "lazy_relation",
        "full_budget_aware_acre",
    }
    assert report["acquisition"]["legacy_acre"]["provider_calls"] == 0
    assert report["relation"]["lazy_relation"]["factorial_invocations"] <= report["relation"]["eager_reference"]["factorial_invocations"]
    assert report["calibration_split"]["seed_disjoint"] is True
    assert report["calibration_split"]["promotion_disjoint"] is True
    assert report["calibration_split"]["evidence_disjoint"] is True


def test_provider_free_ablation_reports_reserve_limited_execution_completion():
    from scripts.run_pilot_surface_experiments import run_budget_aware_ablation

    report = run_budget_aware_ablation(seed=7)
    reserve = report["execution_reserve"]
    assert reserve["evolution_frozen"] is True
    assert reserve["evolution_queries_admitted"] == 0
    assert reserve["scheduled_cells"] == reserve["scheduled_cells_completed"] == 2
    assert reserve["missing_scheduled_cells"] == 0


def test_provider_free_reserve_stress_has_no_missing_scheduled_cells():
    from scripts.run_pilot_surface_experiments import run_budget_aware_reserve_stress

    report = run_budget_aware_reserve_stress(seed=17, schedule_count=128)

    assert report["provider_calls"] == 0
    assert report["schedule_count"] == 128
    assert report["scheduled_completion_rate"] == 1.0
    assert report["missing_scheduled_cells"] == 0
    assert report["pressure_levels"]["tight"]["evolution_spent"] <= report["pressure_levels"]["slack"]["evolution_spent"]
    assert report["pressure_levels"]["tight"]["abstentions"] >= report["pressure_levels"]["slack"]["abstentions"]


def test_condition_blind_calibration_split_has_disjoint_ids():
    from benchmark.calibration.split import build_calibration_split

    split = build_calibration_split(("compile", "episode"), calibration_seed=101, final_seed=7, count=4)
    assert split["seed_disjoint"] is True
    assert split["promotion_disjoint"] is True
    assert split["evidence_disjoint"] is True
    assert not set(split["calibration_instance_ids"]) & set(split["final_instance_ids"])


def test_public_preflight_checks_only_the_declared_solution_contract(tmp_path):
    from benchmark.harness.public_preflight import validate_public_solution

    solution = tmp_path / "solution.py"
    solution.write_text("def build_model(fixtures): return None\n", encoding="utf-8")
    result = validate_public_solution(
        solution,
        {"workspace": {"entrypoint": "solution.py", "api": "train_loop_v1"}},
    )
    assert result["status"] == "fail"
    assert any("train_step" in item for item in result["violations"])
    assert "oracle" not in result
    assert "hidden_verifier" not in result


def test_public_task_projection_exposes_task_specific_execution_contract(tmp_path):
    from benchmark.formal.run_campaign import materialize_agent_task

    task = tmp_path / "task"
    (task / "workspace").mkdir(parents=True)
    (task / "public_tests").mkdir()
    (task / "task.yaml").write_text(
        "task_id: T\ntitle: public\nrequires_cuda: false\ntime_budget_s: 1\n"
        "workspace:\n  entrypoint: solution.py\n  api: train_loop_v1\n"
        "public_contract:\n"
        "  batch:\n"
        "    type: mapping\n"
        "    required_keys: [step, fixed, targets]\n"
        "  model_methods:\n"
        "    project_fixed: project_fixed(fixed_input) -> tensor\n"
        "    forward_head: forward_head(step_input, context, head_idx) -> tensor\n"
        "  seed_policy:\n"
        "    owner: harness\n"
        "    forbidden_calls: [random.seed, torch.manual_seed]\n",
        encoding="utf-8",
    )
    (task / "workspace" / "solution.py").write_text("x = 1\n", encoding="utf-8")

    public = tmp_path / "public"
    materialize_agent_task(task, public)
    payload = json.loads((public / "public_task.json").read_text(encoding="utf-8"))

    assert payload["public_contract"]["batch"]["required_keys"] == ["step", "fixed", "targets"]
    assert payload["public_contract"]["model_methods"]["project_fixed"].startswith("project_fixed")
    assert payload["public_contract"]["seed_policy"]["owner"] == "harness"


def test_public_preflight_reports_harness_owned_seed_violation(tmp_path):
    from benchmark.harness.public_preflight import validate_public_solution

    solution = tmp_path / "solution.py"
    solution.write_text(
        "import torch\n"
        "def build_model(fixtures):\n    torch.manual_seed(1)\n    return None\n"
        "def train_step(model, batch, optimizer): return {'loss': 0, 'work_units': {}}\n"
        "def run_training(fixtures, steps): return {}\n",
        encoding="utf-8",
    )
    result = validate_public_solution(
        solution,
        {
            "workspace": {"entrypoint": "solution.py", "api": "train_loop_v1"},
            "public_contract": {
                "seed_policy": {
                    "owner": "harness",
                    "forbidden_calls": ["random.seed", "torch.manual_seed"],
                }
            },
        },
    )

    assert result["status"] == "fail"
    assert any("harness-owned seed policy" in item for item in result["violations"])


def test_near_budget_finalize_uses_estimated_next_request_cost():
    from benchmark.formal.host_agent import should_finalize_before_next_turn

    assert should_finalize_before_next_turn(
        turn_index=2,
        max_turns=16,
        input_tokens=50_000,
        output_tokens=0,
        input_limit=65_536,
        output_limit=8_192,
        estimated_next_input_tokens=1_000,
        reserve_tokens=4_096,
    ) is False
    assert should_finalize_before_next_turn(
        turn_index=2,
        max_turns=16,
        input_tokens=60_000,
        output_tokens=0,
        input_limit=65_536,
        output_limit=8_192,
        estimated_next_input_tokens=2_000,
        reserve_tokens=4_096,
    ) is True


def test_partial_controller_telemetry_preserves_trace_and_candidate_state(tmp_path):
    from benchmark.formal.host_agent import persist_partial_controller_telemetry

    worker_root = tmp_path / "worker"
    (worker_root / "solution").mkdir(parents=True)
    (worker_root / "solution" / "solution.py").write_text("candidate = True\n", encoding="utf-8")
    output = tmp_path / "host_controller_partial.json"

    persist_partial_controller_telemetry(
        output,
        worker_root,
        outcome="input_token_limit",
        error="context window exceeded",
        turn_trace=[{
            "turn_index": 3,
            "tool_calls": [{"name": "write_file", "result_bytes": 42}],
            "cumulative_input_tokens": 70_458,
        }],
        input_tokens=70_458,
        output_tokens=15_237,
        tool_calls=16,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "aborted"
    assert payload["outcome"] == "input_token_limit"
    assert payload["turn_trace"][0]["tool_calls"][0]["result_bytes"] == 42
    assert payload["usage"]["input_tokens"] == 70_458
    assert payload["candidate_state"]["files"]["solution.py"]["exists"] is True


def test_host_controller_near_budget_policy_and_thinking_mode_are_explicit():
    from benchmark.formal.host_agent import build_agent_attestation, should_finalize_before_next_turn

    assert should_finalize_before_next_turn(
        turn_index=15, max_turns=16, input_tokens=0, output_tokens=0,
        input_limit=12000, output_limit=8192,
    ) is True
    attestation = build_agent_attestation(
        model_list_digest="m", agent_code_commit="c", container_digest="d", pricing_revision="p",
    )
    assert attestation["thinking"] == "disabled"
    assert attestation["reasoning_effort"] == "none"
