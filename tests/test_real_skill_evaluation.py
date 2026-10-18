from copy import deepcopy
import json

import pytest

from experiments.skill_evolution.evaluation import (
    POLICIES,
    dry_run_report,
    evaluate_fixed_skill,
    evaluate_policy_value_on_real_paired_rollouts,
    assess_success_only_efficiency,
    freeze_acre_route_decisions,
    freeze_policy_decisions,
    main,
)


def _plan(*, evidence_kind="SYNTHETIC_FIXTURE", selection_count=1, count=12):
    tasks = []
    for index in range(count):
        task_id = f"task-{index:02d}"
        tasks.append({
            "task_id": task_id,
            "template_id": f"template-{index:02d}",
            "cluster_id": f"cluster-{index:02d}",
            "split": "train_outer",
            "seed": index,
            "trial_ids": {"off": f"{task_id}-off", "on": f"{task_id}-on"},
        })
    return {
        "plan_type": "fixed_skill_evaluation_plan",
        "evidence_kind": evidence_kind,
        "study_id": "offline-eval-test",
        "run_id": "run-1",
        "source_commit": "abc123",
        "split_manifest_sha256": "split-digest",
        "skill_sha": "skill-digest",
        "skill_rule_ids": ["skill-rule"],
        "agent_sha": "agent-digest",
        "model_identity": "model-snapshot",
        "configuration_sha": "config-digest",
        "benchmark_revision": "benchmark-digest",
        "candidate_selection_count": selection_count,
        "minimum_independent_clusters_for_interval": 10,
        "policies": list(POLICIES),
        "tasks": tasks,
    }


def _decisions(plan, *, deploy_original=None, deploy_regime=None):
    decisions = []
    for task in plan["tasks"]:
        for policy in POLICIES:
            deployed = True
            if policy == "original_acre_fixed_skill" and deploy_original is not None:
                deployed = deploy_original(task)
            if policy == "regime_acre_fixed_skill" and deploy_regime is not None:
                deployed = deploy_regime(task)
            row = {
                "task_id": task["task_id"], "policy": policy, "deployed": deployed,
                "skill_sha": plan["skill_sha"], "agent_sha": plan["agent_sha"],
                "configuration_sha": plan["configuration_sha"],
            }
            if policy != "global_fixed_skill":
                row["decision_source"] = "AcreEngine.route"
                row["selected_rule_ids"] = ["skill-rule"] if deployed else []
            decisions.append(row)
    return {
        "record_type": "frozen_policy_decisions",
        "frozen_before_outcome_join": True,
        "study_id": plan["study_id"],
        "run_id": plan["run_id"],
        "skill_sha": plan["skill_sha"],
        "agent_sha": plan["agent_sha"],
        "decisions": decisions,
    }


def _receipt(plan, task, arm, outcome, *, status=None, trial=None):
    if status is None:
        status = "VERIFIED_SUCCESS" if outcome is True else "VERIFIED_FAILURE" if outcome is False else "INCONCLUSIVE"
    receipt = {
        "schema_version": 2,
        "receipt_type": "fixture_episode_receipt" if plan["evidence_kind"] == "SYNTHETIC_FIXTURE" else None,
        "study_id": plan["study_id"], "run_id": plan["run_id"], "source_commit": plan["source_commit"],
        "trial_id": trial or task["trial_ids"][arm], "task_id": task["task_id"], "arm": arm,
        "template_id": task["template_id"], "cluster_id": task["cluster_id"], "split": task["split"],
        "skill_sha": plan["skill_sha"] if arm == "on" else None,
        "agent_revision_sha256": plan["agent_sha"], "model_request_name": plan["model_identity"],
        "model_response_identity": None,
        "prompt_template_sha256": plan["configuration_sha"],
        "benchmark_revision": plan["benchmark_revision"], "seed": task["seed"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "status": status,
        "success": outcome,
        "environment_won": outcome,
        "environment_done": status in {"VERIFIED_SUCCESS", "VERIFIED_FAILURE"},
        "trajectory": [{
            "input_tokens": 11, "output_tokens": 3, "reasoning_tokens": 1,
            "cache_hit_tokens": 0, "cache_miss_tokens": 11,
        }],
        "provider_calls": 999, "input_tokens": 999999,
        "provider_budget": {"provider_calls": 999, "input_tokens": 999999},
    }
    if plan["evidence_kind"] == "SYNTHETIC_FIXTURE":
        receipt["evidence_origin"] = "SYNTHETIC_FIXTURE"
    else:
        receipt["evidence_origin"] = "REAL_AGENT"
        receipt["receipt_type"] = "real_agent_episode_receipt"
    return receipt


def _paired_receipts(plan, pairs):
    return [
        _receipt(plan, task, arm, pairs[index][0 if arm == "off" else 1])
        for index, task in enumerate(plan["tasks"])
        for arm in ("off", "on")
    ]


def test_content_and_gate_effects_are_separate_and_reuse_physical_receipts():
    plan = _plan()
    pairs = [(False, True), (True, False), (True, True), (False, False)] + [(False, True)] * 8
    report, rows = evaluate_fixed_skill(plan, _decisions(
        plan,
        deploy_original=lambda task: int(task["task_id"].split("-")[-1]) % 2 == 0,
        deploy_regime=lambda task: int(task["task_id"].split("-")[-1]) < 2,
    ), _paired_receipts(plan, pairs))

    assert report["status"] == "SYNTHETIC_FIXTURE_ONLY"
    assert {key: report["content_contrast"]["paired_counts"][key] for key in ("00", "01", "10", "11", "missing")} == {
        "00": 1, "01": 9, "10": 1, "11": 1, "missing": 0,
    }
    assert report["content_contrast"]["paired_counts"]["regression"] == 1
    assert report["content_contrast"]["paired_delta_on_minus_off"]["complete_case"]["value"] == pytest.approx(8 / 12)
    assert report["gate_contrasts"]["original_acre_fixed_skill"]["coverage"] == pytest.approx(0.5)
    assert report["gate_contrasts"]["regime_acre_fixed_skill"]["coverage"] == pytest.approx(2 / 12)
    assert report["counts"]["n_completed"] == 12
    assert len(rows) == 24
    assert rows[0]["arm"] == "off" and rows[1]["arm"] == "on"
    assert report["cost"]["task_only"]["provider_calls"]["value"] == 24
    assert report["cost"]["task_only"]["input_tokens"]["value"] == 264
    assert report["cost"]["task_only"]["provider_calls"]["value"] != 999 * 24


def test_mock_fixture_cannot_be_joined_as_real_and_real_requires_explicit_origin():
    real_plan = _plan(evidence_kind="REAL_AGENT", count=1)
    decisions = _decisions(real_plan)
    fixture_plan = _plan(count=1)
    fixture_receipt = _paired_receipts(fixture_plan, [(False, True)])[0]
    with pytest.raises(ValueError, match="type/origin"):
        evaluate_fixed_skill(real_plan, decisions, [fixture_receipt])

    real_receipt = _receipt(real_plan, real_plan["tasks"][0], "off", False)
    real_receipt.pop("evidence_origin")
    with pytest.raises(ValueError, match="type/origin"):
        evaluate_fixed_skill(real_plan, decisions, [real_receipt])


def test_missing_pairs_get_itt_bounds_and_zero_deployment_harm_is_undefined():
    plan = _plan(count=2)
    decisions = _decisions(plan, deploy_original=lambda _task: False, deploy_regime=lambda _task: False)
    receipts = _paired_receipts(plan, [(False, True), (True, False)])[:3]
    report, _ = evaluate_fixed_skill(plan, decisions, receipts)
    delta = report["content_contrast"]["paired_delta_on_minus_off"]
    assert delta["complete_case"]["value"] == 1.0
    assert delta["planned_itt_bounds"] == {
        "lower": 0.0, "upper": 1.0, "n_planned": 2, "n_missing": 1,
        "status": "BOUNDED_WITH_MISSING",
    }
    gate = report["gate_contrasts"]["original_acre_fixed_skill"]
    assert gate["n_deployed"] == 0
    assert gate["harm_deployed"] is None
    assert gate["harm_deployed_status"] == "UNDEFINED_ZERO_OBSERVED_DEPLOYMENT"
    assert gate["zero_harm_upper_95"]["status"] == "UNDEFINED_ZERO_DEPLOYMENT"


def test_shared_arm_policy_identity_is_zero_even_when_no_outcome_was_observed():
    plan = _plan(count=2)
    decisions = _decisions(plan, deploy_original=lambda _task: False, deploy_regime=lambda _task: True)
    report, _ = evaluate_fixed_skill(plan, decisions, [])
    abstain = report["gate_contrasts"]["original_acre_fixed_skill"]
    assert abstain["incremental_success_over_base"]["complete_case"]["value"] == 0
    assert abstain["incremental_success_over_base"]["planned_itt_bounds"]["upper"] == 0
    assert abstain["harm_all"]["planned_itt_bounds"] == {"lower": 0, "upper": 0}
    assert abstain["harm_deployed"] is None
    reuse = report["gate_contrasts"]["regime_acre_fixed_skill"]
    assert reuse["paired_delta_vs_global"]["complete_case"]["value"] == 0
    assert report["content_contrast"]["paired_counts"]["missing"] == 2


def test_agent_policy_loop_stop_is_observed_failure_not_missing():
    plan = _plan(count=1)
    receipts = _paired_receipts(plan, [(False, True)])
    receipts[0].update(status="AGENT_POLICY_FAILURE", success=False,
                       environment_won=False, done=False,
                       termination_reason="agent_policy_failure")
    receipts[0].pop("environment_done")

    report, _ = evaluate_fixed_skill(plan, _decisions(plan), receipts)

    assert report["counts"]["n_success_false"] == 1
    assert report["counts"]["n_success_unknown"] == 0
    assert report["content_contrast"]["paired_counts"]["repair"] == 1


def test_decisions_must_be_frozen_complete_and_outcome_free():
    plan = _plan(count=1)
    decisions = _decisions(plan)
    bad = deepcopy(decisions)
    bad["decisions"][0]["environment_won"] = True
    with pytest.raises(ValueError, match="cannot contain outcomes"):
        evaluate_fixed_skill(plan, bad, [])
    bad = deepcopy(decisions)
    bad["frozen_before_outcome_join"] = False
    with pytest.raises(ValueError, match="frozen before outcome join"):
        evaluate_fixed_skill(plan, bad, [])
    bad = deepcopy(decisions)
    bad["decisions"].pop()
    with pytest.raises(ValueError, match="one frozen decision"):
        evaluate_fixed_skill(plan, bad, [])


def test_route_freezer_calls_canonical_engine_for_each_preexecution_context():
    class Engine:
        def __init__(self):
            self.calls = []

        def route(self, context, **kwargs):
            self.calls.append((context, kwargs))
            return type("Decision", (), {"selected_rule_ids": ("skill-rule",) if context["eligible"] else ()})()

    engine = Engine()
    frozen = freeze_acre_route_decisions(
        engine,
        [{"task_id": "t1", "context": {"eligible": True}}, {"task_id": "t2", "context": {"eligible": False}}],
        policy="regime_acre_fixed_skill", skill_rule_ids=["skill-rule"],
        skill_sha="skill-digest", agent_sha="agent-digest", support_regimes={"skill-rule": "frozen"},
        study_id="study", run_id="run",
    )
    assert len(engine.calls) == 2
    assert all(call[1]["support_regimes"] == {"skill-rule": "frozen"} for call in engine.calls)
    assert [item["deployed"] for item in frozen["decisions"]] == [True, False]
    assert all(item["decision_source"] == "AcreEngine.route" for item in frozen["decisions"])


def test_route_freezer_rejects_future_outcome_features():
    class Engine:
        def route(self, context, **kwargs):
            return type("Decision", (), {"selected_rule_ids": ()})()

    with pytest.raises(ValueError, match="cannot contain outcomes or labels"):
        freeze_acre_route_decisions(
            Engine(), [{"task_id": "t1", "context": {"future_observation": "won"}}],
            policy="regime_acre_fixed_skill", skill_rule_ids=["skill-rule"],
            skill_sha="skill-digest", agent_sha="agent-digest", study_id="study", run_id="run",
        )


def test_combined_freeze_emits_same_fixed_skill_policy_decisions_before_join():
    class Engine:
        def route(self, context, **kwargs):
            return type("Decision", (), {"selected_rule_ids": ("skill-rule",) if context["eligible"] else ()})()

    frozen = freeze_policy_decisions(
        Engine(), Engine(),
        [{"task_id": "t1", "context": {"eligible": True}}],
        skill_rule_ids=["skill-rule"], skill_sha="skill-digest", agent_sha="agent-digest",
        study_id="study", run_id="run", configuration_sha="config-digest",
        support_regimes={"skill-rule": "frozen"},
    )
    assert frozen["frozen_before_outcome_join"] is True
    assert {row["policy"] for row in frozen["decisions"]} == set(POLICIES)
    assert all(row["skill_sha"] == "skill-digest" and row["agent_sha"] == "agent-digest"
               for row in frozen["decisions"])
    assert all(row["decision_source"] == "AcreEngine.route"
               for row in frozen["decisions"] if row["policy"] != "global_fixed_skill")
    assert frozen["policy_value_contract"]["feature_projection"] == "PREEXECUTION_CONTEXT_ONLY"
    assert frozen["configuration_sha"] == "config-digest"


def test_cluster_interval_is_suppressed_after_candidate_selection():
    plan = _plan(selection_count=3)
    report, _ = evaluate_fixed_skill(plan, _decisions(plan), _paired_receipts(plan, [(False, True)] * 12))
    assert report["content_contrast"]["semantic_cluster_interval"]["status"] == "NOT_REPORTED_SELECTED_CANDIDATE_MULTIPLICITY"
    assert report["independent_claims"] is False


def test_multiple_template_directories_share_semantic_cluster_and_interval_is_task_weighted():
    plan = _plan(count=4)
    plan["minimum_independent_clusters_for_interval"] = 2
    for index, task in enumerate(plan["tasks"]):
        task["template_id"] = f"directory-template-{index}"
        task["cluster_id"] = "semantic-cluster-a" if index < 3 else "semantic-cluster-b"
    receipts = _paired_receipts(plan, [(False, True)] * 3 + [(True, False)])

    report, _ = evaluate_fixed_skill(plan, _decisions(plan), receipts)

    interval = report["content_contrast"]["semantic_cluster_interval"]
    delta = report["content_contrast"]["paired_delta_on_minus_off"]["complete_case"]
    assert interval["status"] == "EXPLORATORY_SEMANTIC_CLUSTER_NORMAL_95"
    assert interval["independent_clusters"] == 2
    assert interval["complete_case_tasks"] == 4
    assert interval["estimand"].startswith("task-weighted complete-case")
    assert interval["value"] == delta["value"] == pytest.approx(0.5)


def test_shared_costs_are_counted_once_and_missing_usage_stays_unknown():
    plan = _plan(count=1)
    receipts = _paired_receipts(plan, [(False, True)])
    for receipt in receipts:
        receipt["trajectory"] = [{"input_tokens": 5, "output_tokens": 2}]
    shared = [{"cost_id": "proposal-call-1", "cost": {
        "provider_calls": 1, "input_tokens": 7, "completion_tokens": 4,
        "reasoning_tokens": None, "cache_hit_tokens": None, "cache_miss_tokens": None,
        "provider_usd_estimate": None, "wall_seconds": 1.0,
        "gpu_seconds": None, "verification_seconds": None,
    }}]
    report, _ = evaluate_fixed_skill(plan, _decisions(plan), receipts, shared_cost_items=shared)
    assert report["cost"]["task_only"]["provider_calls"]["value"] == 2
    assert report["cost"]["all_in"]["provider_calls"]["value"] == 3
    assert report["cost"]["all_in"]["input_tokens"]["value"] == 17
    assert report["cost"]["all_in"]["provider_usd_estimate"]["value"] is None
    assert report["cost"]["all_in"]["provider_usd_estimate"]["status"] == "UNKNOWN"
    with pytest.raises(ValueError, match="unique cost_id"):
        evaluate_fixed_skill(plan, _decisions(plan), receipts, shared_cost_items=shared * 2)


def test_taskbook_done_and_per_request_usd_fields_are_consumed():
    plan = _plan(count=1)
    receipts = _paired_receipts(plan, [(True, True)])
    for receipt in receipts:
        receipt["done"] = receipt.pop("environment_done")
        receipt["trajectory"] = [{
            "raw_output": "look", "input_tokens": 4, "output_tokens": 2,
            "estimated_provider_usd": "0.025",
        }]

    report, _ = evaluate_fixed_skill(plan, _decisions(plan), receipts)

    assert report["content_contrast"]["paired_counts"]["both_correct"] == 1
    assert report["cost"]["task_only"]["provider_calls"]["value"] == 2
    assert report["cost"]["task_only"]["provider_usd_estimate"]["value"] == pytest.approx(0.05)


def _priced_success_report(*, count=3, input_off=100, completion_off=100,
                           input_on=50, completion_on=50, usd_off="0.10", usd_on="0.08",
                           outcomes=None, prefix="", split="train_outer"):
    plan = _plan(evidence_kind="REAL_AGENT", count=count)
    for task in plan["tasks"]:
        for key in ("task_id", "template_id", "cluster_id"):
            task[key] = prefix + task[key]
        task["trial_ids"] = {arm: prefix + value for arm, value in task["trial_ids"].items()}
        task["split"] = split
        if split == "train_outer":
            task["child_split"] = "outer_skill_validation"
    plan["deployment_iid_assumption"] = True
    plan["deployment_scope_frozen"] = True
    pairs = outcomes or [(True, True)] * count
    receipts = _paired_receipts(plan, pairs)
    for receipt in receipts:
        off = receipt["arm"] == "off"
        receipt["trajectory"] = [{
            "input_tokens": input_off if off else input_on,
            "output_tokens": completion_off if off else completion_on,
            "reasoning_tokens": 20 if off else 10,
            "provider_request_made": True,
            "estimated_provider_usd": usd_off if off else usd_on,
        }]
        receipt["actions"] = 4 if off else 3
    return plan, _decisions(plan), receipts


def test_inner_favorable_outer_regression_is_rejected_without_promotion():
    inner_plan, inner_decisions, inner_receipts = _priced_success_report(prefix="inner-", split="train_inner")
    inner, _ = evaluate_fixed_skill(inner_plan, inner_decisions, inner_receipts)
    good_plan, good_decisions, good_receipts = _priced_success_report(prefix="outer-")
    good_outer, _ = evaluate_fixed_skill(good_plan, good_decisions, good_receipts)
    favorable = assess_success_only_efficiency(inner, good_outer)
    assert favorable["outer_status"] == "FAVORABLE"
    assert favorable["canonical_regime"] == "UNCHANGED_NO_PROMOTION"
    outer_plan, outer_decisions, outer_receipts = _priced_success_report(
        outcomes=[(True, False), (True, True), (True, True)], prefix="outer-")
    outer, _ = evaluate_fixed_skill(outer_plan, outer_decisions, outer_receipts)

    result = assess_success_only_efficiency(inner, outer)

    assert result["inner_status"] == "FAVORABLE"
    assert result["outer_status"] == "REJECTED"
    assert result["outer_reason"] == "OBSERVED_REGRESSION"
    assert result["promotion"] == "NO_REAL_PROMOTION"
    assert result["canonical_regime"] == "UNCHANGED_NO_PROMOTION"


def test_inner_cannot_be_renamed_as_independent_outer():
    plan, decisions, receipts = _priced_success_report(split="train_inner")
    report, _ = evaluate_fixed_skill(plan, decisions, receipts)
    with pytest.raises(ValueError, match="reserved outer cohorts"):
        assess_success_only_efficiency(report, report)


def test_break_even_requires_measured_build_not_total_evaluation_cost():
    plan, decisions, receipts = _priced_success_report()
    report, _ = evaluate_fixed_skill(plan, decisions, receipts)
    assert report["cost"]["break_even_deployments"] is None
    plan["measured_build_costs_usd"] = {key: 1.0 for key in (
        "diagnostic", "proposal", "patch", "inner", "outer", "gate_calibration")}
    report, _ = evaluate_fixed_skill(plan, decisions, receipts)
    assert report["cost"]["measured_build_cost_usd"] == 6.0
    assert report["cost"]["break_even_deployments"] == pytest.approx(300, abs=1)


def test_both_success_but_higher_cost_and_tokens_is_no_net_efficiency_gain():
    plan, decisions, receipts = _priced_success_report(
        input_off=50, completion_off=50, input_on=100, completion_on=100,
        usd_off="0.05", usd_on="0.10")

    report, _ = evaluate_fixed_skill(plan, decisions, receipts)

    assert report["content_contrast"]["paired_counts"]["11"] == 3
    assert report["success_only_gate"]["status"] == "NO_NET_EFFICIENCY_GAIN"
    assert report["success_only_gate"]["median_both_success_total_token_saving_fraction"] == pytest.approx(-1.0)
    assert report["cost"]["break_even_deployments"] is None


def test_failed_rollouts_with_lower_usage_do_not_count_as_efficiency_savings():
    plan, decisions, receipts = _priced_success_report(
        input_off=100, completion_off=100, input_on=1, completion_on=1,
        outcomes=[(False, False)] * 3)

    report, _ = evaluate_fixed_skill(plan, decisions, receipts)

    assert report["success_only_efficiency"]["n_both_success"] == 0
    assert report["success_only_efficiency"]["metrics"]["total_tokens"]["saving_fraction"]["n"] == 0
    assert report["success_only_gate"]["status"] == "NO_NET_EFFICIENCY_GAIN"


def test_missing_on_is_not_imputed_as_zero_regression_or_zero_retention():
    plan = _plan(count=1)
    receipts = _paired_receipts(plan, [(True, True)])[:1]
    report, _ = evaluate_fixed_skill(plan, _decisions(plan), receipts)
    screen = report["success_only_gate"]
    assert screen["off_success_denominator"] == 1
    assert screen["regression_rate_given_off_success"] is None
    assert screen["on_success_given_off_success"] is None
    assert screen["regression_rate_bounds"] == {"lower": 0.0, "upper": 1.0}
    assert screen["on_success_given_off_success_bounds"] == {"lower": 0.0, "upper": 1.0}


def test_static_policy_value_requires_real_fixed_predecision_rollouts_and_unique_clusters():
    plan, _, receipts = _priced_success_report(count=3)
    for task in plan["tasks"][1:]:
        task["cluster_id"] = "shared-cluster"
    plan["tasks"][0]["cluster_id"] = "shared-cluster"
    for receipt in receipts:
        receipt["cluster_id"] = "shared-cluster"
    decisions = {
        "record_type": "frozen_policy_decisions", "frozen_before_outcome_join": True,
        "study_id": plan["study_id"], "run_id": plan["run_id"],
        "skill_sha": plan["skill_sha"], "agent_sha": plan["agent_sha"],
        "configuration_sha": plan["configuration_sha"],
        "policy_value_contract": {
            "decision_timing": "BEFORE_EPISODE",
            "feature_projection": "PREEXECUTION_CONTEXT_ONLY",
            "router_behavior": "INJECT_OR_ABSTAIN_ONLY",
        },
        "decisions": [],
    }
    for task in plan["tasks"]:
        for policy in POLICIES:
            deployed = policy == "global_fixed_skill"
            decisions["decisions"].append({
                "task_id": task["task_id"], "policy": policy, "deployed": deployed,
                "skill_sha": plan["skill_sha"], "agent_sha": plan["agent_sha"],
                "configuration_sha": plan["configuration_sha"],
                **({"decision_source": "AcreEngine.route", "selected_rule_ids": ["skill-rule"] if deployed else []}
                   if policy != "global_fixed_skill" else {}),
            })
    semantics = {
        "off": "fixed_agent_config_without_skill",
        "on": "same_fixed_agent_config_with_frozen_skill",
        "router": "skill_injection_or_abstention_only",
    }

    report, _ = evaluate_policy_value_on_real_paired_rollouts(
        plan, decisions, receipts, arm_semantics=semantics)

    assert report["policy_value"]["status"] == "POLICY_VALUE_ON_REAL_PAIRED_ROLLOUTS"
    assert report["policy_value"]["independent_semantic_clusters"] == 1
    assert report["pair_denominators"] == {
        "planned": 3, "started": 3, "completed": 3, "paired_valid": 3,
        "incomplete": 0, "not_started": 0,
    }
    with pytest.raises(ValueError, match="dynamic routers"):
        evaluate_policy_value_on_real_paired_rollouts(
            plan, decisions, receipts, arm_semantics=semantics, dynamic_router=True)


def test_zero_harm_bound_needs_declared_iid_fixed_unselected_design():
    plan, decisions, receipts = _priced_success_report(count=3)

    report, _ = evaluate_fixed_skill(plan, decisions, receipts)

    bound = report["gate_contrasts"]["original_acre_fixed_skill"]["zero_harm_upper_95"]
    assert bound["status"] == "EXACT_ONE_SIDED_BINOMIAL"
    assert bound["n_independent_deployments"] == 3
    assert bound["value"] == pytest.approx(1 - 0.05 ** (1 / 3))
    del plan["deployment_iid_assumption"]
    uncertified, _ = evaluate_fixed_skill(plan, decisions, receipts)
    assert uncertified["gate_contrasts"]["original_acre_fixed_skill"]["zero_harm_upper_95"]["status"] == "UNCERTIFIED"


def test_dry_run_is_explicitly_not_an_outcome_analysis():
    report = dry_run_report(_plan())
    assert report["status"] == "DRY_RUN_NOT_EXECUTED"
    assert report["provider_calls"] == 0
    assert report["outcomes"] is None


def test_cli_defaults_to_offline_plan_only_and_renders_all_formats(tmp_path):
    plan_path = tmp_path / "plan.json"
    output_dir = tmp_path / "report"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    assert main(["--plan", str(plan_path), "--output-dir", str(output_dir)]) == 0
    report = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert report["status"] == "DRY_RUN_NOT_EXECUTED"
    assert (output_dir / "REPORT.md").is_file()
    csv_path = output_dir / "paired_task_index.csv"
    assert csv_path.is_file()
    assert "arm,trial_id,baseline_trial_id" in csv_path.read_text(encoding="utf-8").splitlines()[0]
