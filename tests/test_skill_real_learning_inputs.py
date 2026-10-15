from dataclasses import replace

import pytest

from core.acre.resource_budget import ProviderBudgetExceeded, manifest_digest
from core.skills import EvidenceBundle
from experiments.skill_evolution.real_learning import validate_learning_inputs


def learning_fixture():
    membership = []
    receipts = []
    for index, (outcome, status) in enumerate(((True, "VERIFIED_SUCCESS"),
                                                (True, "VERIFIED_SUCCESS"),
                                                (False, "AGENT_POLICY_FAILURE"))):
        task_id = f"source-template-{index}/trial_{index}"
        row = {
            "trial_id": f"source-run:{index}",
            "task_id": task_id,
            "template_id": f"source-template-{index}",
            "cluster_id": f"source-cluster-{index}",
            "split": "train_inner",
            "status": status,
            "termination_reason": "success" if outcome else "agent_policy_failure",
            "environment_won": outcome,
            "environment_done": outcome,
            "success": outcome,
            "input_tokens": 17,
            "completion_tokens": 13,
            "provider_budget": {"tokens_unknown": False},
            "evidence_origin": "REAL_AGENT",
            "task_family": "clean_place",
            "agent_revision_sha256": "a" * 64,
            "configuration_sha": "e" * 64, "model_request_name": "deepseek-flash",
        }
        row["receipt_sha256"] = manifest_digest(row)
        receipts.append(row)
        membership.append({
            "task_id": task_id, "template_id": row["template_id"], "cluster_id": row["cluster_id"],
            "split": "train_inner", "task_type": "pick_clean_then_place_in_recep",
            "identity_status": "IDENTIFIED", "confirmatory_eligible": True,
        })

    pairs = []
    for split, suffix in (("train_inner", "inner"), ("train_outer", "outer")):
        task_id = f"{suffix}-template/trial_1"
        gamefile = f"/data/json_2.1.1/train/{task_id}/game.tw-pddl"
        row = {"task_id": task_id, "template_id": f"{suffix}-template", "gamefile": gamefile,
               "cluster_id": f"{suffix}-cluster", "split": split,
               "identity_status": "IDENTIFIED", "confirmatory_eligible": True}
        membership.append(row)
        pairs.append({
            "pair_id": f"pair-{suffix}", "task_id": task_id, "template_id": row["template_id"],
            "cluster_id": row["cluster_id"], "split": split, "task_type_id": 3,
            "gamefile": gamefile, "seed": 19, "arm_order": ["off", "on"],
        })

    split_manifest = {"task_membership": membership}
    plan = {"pairs": pairs[:1], "outer_pairs": pairs[1:]}
    manifest = {
        "stage": "skill_evolution", "split": "train_inner", "thinking": "enabled",
        "max_retries": 0, "split_manifest_sha256": manifest_digest(split_manifest),
        "plan": plan, "plan_sha256": manifest_digest(plan),
        "viability_receipts_sha256": "b" * 64, "source_receipts_sha256": "c" * 64,
        "max_steps": 2, "max_input_per_request": 100, "max_completion_per_request": 100,
        "proposal_completion_calls": 1, "max_calls": 9, "agent_sha": "a" * 64,
        "configuration_sha": "e" * 64, "model_request_name": "deepseek-flash",
        "source_commit": "controller-commit",
    }
    evidence = EvidenceBundle(
        source_split="train_inner",
        source_task_ids=tuple(row["task_id"] for row in receipts),
        source_template_ids=tuple(row["template_id"] for row in receipts),
        source_cluster_ids=tuple(row["cluster_id"] for row in receipts),
        source_trace_ids=tuple(row["trial_id"] for row in receipts),
        source_trace_hashes=tuple(row["receipt_sha256"] for row in receipts),
        train_split_manifest_sha256=manifest["split_manifest_sha256"],
        task_family="clean_place", agent_hash=manifest["agent_sha"],
        successful_trajectory_summaries=("Verified visible completion.",),
        failed_trajectory_summaries=("Frozen agent reached a policy loop; environment did not report a win.",),
        redaction_report_sha256="d" * 64,
    )
    return manifest, split_manifest, [dict(row) for row in receipts], [dict(row) for row in receipts], evidence


def test_c_inputs_use_canonical_membership_semantic_clusters_and_prebudget_outer_pairs():
    manifest, split, viability, sources, evidence = learning_fixture()
    plan = validate_learning_inputs(manifest, split, viability, sources, evidence)
    assert plan["status"] == "READY_FOR_APPROVAL"
    assert plan["B_independent_wins"] == 2
    assert plan["B_verified_failures"] == 1
    assert plan["planned_provider_calls"] == 9
    assert plan["paired_clusters"] == {"train_inner": ["inner-cluster"], "train_outer": ["outer-cluster"]}


def test_b_gate_counts_semantic_clusters_not_directory_templates():
    manifest, split, viability, sources, evidence = learning_fixture()
    viability[1]["cluster_id"] = viability[0]["cluster_id"]
    with pytest.raises(ProviderBudgetExceeded, match="B_viability_gate_not_satisfied"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_source_receipt_hash_is_recomputed_and_tampering_is_rejected():
    manifest, split, viability, sources, evidence = learning_fixture()
    sources[0]["tampered"] = True
    with pytest.raises(ValueError, match="receipt hashes"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_inner_outer_semantic_cluster_overlap_is_rejected_even_with_distinct_templates():
    manifest, split, viability, sources, evidence = learning_fixture()
    outer_task = manifest["plan"]["outer_pairs"][0]
    membership = next(row for row in split["task_membership"] if row["task_id"] == outer_task["task_id"])
    membership["cluster_id"] = "inner-cluster"
    outer_task["cluster_id"] = "inner-cluster"
    manifest["plan_sha256"] = manifest_digest(manifest["plan"])
    manifest["split_manifest_sha256"] = manifest_digest(split)
    evidence = replace(evidence, train_split_manifest_sha256=manifest["split_manifest_sha256"])
    with pytest.raises(ValueError, match="inner and outer"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_wrong_native_gamefile_cannot_be_labeled_as_a_different_planned_task():
    manifest, split, viability, sources, evidence = learning_fixture()
    manifest["plan"]["pairs"][0]["gamefile"] = "/data/json_2.1.1/train/other/trial_1/game.tw-pddl"
    manifest["plan_sha256"] = manifest_digest(manifest["plan"])
    with pytest.raises(ValueError, match="gamefile"):
        validate_learning_inputs(manifest, split, viability, sources, evidence)


def test_old_agent_source_evidence_cannot_be_used_as_current_agent_evidence():
    manifest, split, viability, sources, evidence = learning_fixture()
    with pytest.raises(ValueError, match="different frozen agent"):
        validate_learning_inputs(manifest, split, viability, sources, replace(evidence, agent_hash="old"))


def test_real_pair_adapter_uses_production_episode_and_persists_both_arms(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from experiments.skill_evolution.real_learning import _run_one_pair
    manifest, _, _, _, _ = learning_fixture()
    pair = manifest["plan"]["pairs"][0]
    closed, persisted, insertions = [], [], []

    class Environment:
        def seed(self, seed):
            assert seed == pair["seed"]

        def reset(self):
            return ["Your task is to: inspect the room."], {
                "extra.gamefile": [pair["gamefile"]], "extra.expert_plan": [["EXPERT_SECRET"]],
                "admissible_commands": [["look", "inventory"]]}

        def step(self, actions):
            assert type(actions) is list and len(actions) == 1
            return ["Visible result."], [0], [True], {"won": [actions[0] == "look"]}

        def close(self):
            closed.append(True)

    class Owner:
        def __init__(self, config, train_eval):
            assert train_eval == "train"

        def init_env(self, batch_size):
            assert batch_size == 1
            return Environment()

    class Client:
        budget = None
        def choose_action(self, state, admissible, skill, progress, reason):
            assert "EXPERT_SECRET" not in repr(state) + progress
            insertions.append(skill)
            return {"raw_output": "Action: look" if skill else "Action: inventory",
                    "input_tokens": 10, "output_tokens": 3}

    monkeypatch.setitem(sys.modules, "alfworld.agents.environment.alfred_tw_env", SimpleNamespace(AlfredTWEnv=Owner))
    candidate = SimpleNamespace(body="Verify visible outcomes.", digest="f" * 64, version=1)
    off, on = _run_one_pair(tmp_path, Client(), pair, candidate, split="train_inner", max_steps=2,
                           run_id="fixture", agent_revision="agent", benchmark_revision="benchmark",
                           provenance=manifest, on_arm=lambda arm, row: persisted.append((arm, row)))
    assert off["success"] is False and on["success"] is True
    assert [arm for arm, _ in persisted] == ["off", "on"]
    assert insertions == [None, candidate.body]
    assert len(closed) == 2
    assert on["skill_version_sha256"] == candidate.digest
    assert on["split"] == "train_inner" and on["trajectory"][0]["selected_action"] == "look"
