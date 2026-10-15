"""Approved, train_inner model patching and paired ALFWorld evaluation.

Dry-run is the default. Live calls require a C-scoped ApprovedProviderBudget,
eligible B viability receipts, disjoint frozen templates, and a complete pair
budget before the provider client or environment is created.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from collections.abc import Callable

from core.acre.resource_budget import ApprovedProviderBudget, ProviderBudgetExceeded, manifest_digest
from .evaluation import receipt_outcome
from core.skills import (
    EvidenceBundle,
    ModelSkillProposer,
    PairReceipt,
    SkillContract,
    SkillEvolution,
    SkillEvolutionStore,
    SkillPatchOptimizer,
    approve_outer_validation,
    export_skill_pack,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_bundle(value: Mapping[str, Any]) -> EvidenceBundle:
    tuple_fields = {
        "source_task_ids", "source_template_ids", "source_cluster_ids", "source_trace_ids", "source_trace_hashes",
        "successful_trajectory_summaries", "failed_trajectory_summaries", "first_divergence_events",
        "verified_positive_patterns", "verified_failure_patterns", "subgoal_compliance_events",
        "action_rejections", "loop_or_stall_events", "budget_failures",
    }
    return EvidenceBundle(**{
        field.name: tuple(value[field.name]) if field.name in tuple_fields else value[field.name]
        for field in fields(EvidenceBundle) if field.name in value
    })


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    rows = payload.get(key) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} must be a JSON list of receipt objects")
    return rows


def validate_learning_inputs(
    manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    viability_receipts: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]],
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    """Validate C-stage scope, the B gate, and template-disjoint frozen inputs."""
    if manifest.get("stage") != "skill_evolution" or manifest.get("split") != "train_inner":
        raise ProviderBudgetExceeded("unsupported_skill_learning_scope")
    if manifest.get("thinking") != "enabled" or manifest.get("max_retries") != 0:
        raise ProviderBudgetExceeded("unsupported_provider_configuration")
    agent_sha = manifest.get("agent_revision_sha256") or manifest.get("agent_sha")
    prompt_identity = manifest.get("configuration_sha")
    if not all(isinstance(value, str) and value.strip() for value in (
        agent_sha, prompt_identity, manifest.get("model_request_name"), manifest.get("source_commit"),
    )):
        raise ProviderBudgetExceeded("frozen_agent_prompt_model_and_source_identities_required")
    if manifest.get("split_manifest_sha256") != manifest_digest(split_manifest):
        raise ProviderBudgetExceeded("split_manifest_digest_mismatch")
    if evidence.train_split_manifest_sha256 != manifest["split_manifest_sha256"]:
        raise ProviderBudgetExceeded("evidence_split_manifest_mismatch")
    if manifest.get("viability_receipts_sha256") is None or manifest.get("source_receipts_sha256") is None:
        raise ProviderBudgetExceeded("frozen_source_receipt_hashes_required")
    evidence.validate()
    if evidence.agent_hash != agent_sha:
        raise ValueError("source evidence belongs to a different frozen agent")

    from .real_study import viability_gate
    b_gate = viability_gate(list(viability_receipts))
    if b_gate["status"] != "B-VIABLE-TRAIN-ONLY":
        raise ProviderBudgetExceeded("B_viability_gate_not_satisfied")
    b_successes = [row for row in viability_receipts if (
        receipt_outcome(row) is True
        and row.get("evidence_origin") == "REAL_AGENT" and not row.get("teacher_only_source", False)
    )]
    b_failures = [row for row in viability_receipts if (
        receipt_outcome(row) is False
        and row.get("evidence_origin") == "REAL_AGENT" and not row.get("teacher_only_source", False)
    )]
    win_clusters = {row.get("cluster_id") for row in b_successes}
    failure_clusters = {row.get("cluster_id") for row in b_failures}
    if None in win_clusters or None in failure_clusters:
        raise ProviderBudgetExceeded("B_viability_receipts_require_semantic_clusters")
    b_clusters = win_clusters | failure_clusters
    b_templates = {row.get("template_id") for row in (*b_successes, *b_failures)}
    if None in b_templates:
        raise ProviderBudgetExceeded("B_viability_requires_template_identity")

    source_by_id = {row.get("trial_id"): row for row in source_receipts}
    if len(source_by_id) != len(source_receipts) or None in source_by_id:
        raise ValueError("train_inner source receipt IDs must be unique")
    if set(evidence.source_trace_ids) != set(source_by_id):
        raise ValueError("evidence trace IDs do not match the frozen source receipt set")
    if any(row.get("split") != "train_inner" or receipt_outcome(row) is None
           or row.get("evidence_origin") != "REAL_AGENT" or row.get("teacher_only_source", False)
           for row in source_receipts):
        raise ValueError("proposal sources must be autonomous, verifier-resolved train_inner receipts")
    if any(row.get("agent_revision_sha256", row.get("agent_sha")) != agent_sha for row in source_receipts):
        raise ValueError("source receipts belong to a different frozen agent")
    if any(row.get("configuration_sha", row.get("prompt_template_sha256")) != prompt_identity
           or row.get("model_request_name") != manifest["model_request_name"] for row in source_receipts):
        raise ValueError("source receipts belong to a different frozen provider configuration")
    if set(evidence.source_task_ids) != {row.get("task_id") for row in source_receipts}:
        raise ValueError("source task IDs do not match the frozen receipt set")
    if set(evidence.source_template_ids) != {row.get("template_id") for row in source_receipts}:
        raise ValueError("source template IDs do not match the frozen receipt set")
    if set(evidence.source_cluster_ids) != {row.get("cluster_id") for row in source_receipts}:
        raise ValueError("source semantic cluster IDs do not match the frozen receipt set")
    source_hashes = tuple(sorted(
        row["receipt_sha256"] for row in source_receipts
        if row.get("receipt_sha256") == manifest_digest({k: v for k, v in row.items() if k != "receipt_sha256"})
    ))
    if len(source_hashes) != len(source_receipts) or source_hashes != tuple(sorted(evidence.source_trace_hashes)):
        raise ValueError("source receipt hashes do not match the frozen evidence bundle")
    if any(receipt_outcome(row) is None for row in source_receipts):
        raise ValueError("source verifier labels conflict with environment_won")
    frozen_tasks = split_manifest.get("task_membership")
    if not isinstance(frozen_tasks, list):
        raise ValueError("split manifest requires the canonical task_membership list")
    split_by_task = {row.get("task_id"): row for row in frozen_tasks if isinstance(row, dict)}
    if len(split_by_task) != len(frozen_tasks):
        raise ValueError("split manifest task IDs must be unique")
    if manifest.get("plan_sha256") != manifest_digest(manifest.get("plan", {})):
        raise ProviderBudgetExceeded("pair_plan_digest_mismatch")
    source_ids, source_templates = set(evidence.source_task_ids), set(evidence.source_template_ids)
    source_clusters = set(evidence.source_cluster_ids)
    inner_pairs = manifest.get("plan", {}).get("pairs")
    outer_pairs = manifest.get("plan", {}).get("outer_pairs")
    if not isinstance(inner_pairs, list) or not inner_pairs:
        raise ValueError("C manifest requires a frozen non-empty inner pair list")
    if not isinstance(outer_pairs, list) or not outer_pairs:
        raise ValueError("C manifest requires a prebudgeted, frozen non-empty train_outer pair list")
    pair_templates: dict[str, set[str]] = {"train_inner": set(), "train_outer": set()}
    pair_clusters: dict[str, set[str]] = {"train_inner": set(), "train_outer": set()}
    pair_task_ids: dict[str, set[str]] = {"train_inner": set(), "train_outer": set()}
    for split, pairs in (("train_inner", inner_pairs), ("train_outer", outer_pairs)):
        for pair in pairs:
            if not isinstance(pair, dict) or pair.get("split") != split:
                raise ValueError(f"every paired task must be explicitly assigned to {split}")
            task_id, template_id = pair.get("task_id"), pair.get("template_id")
            frozen = split_by_task.get(task_id)
            if (not isinstance(task_id, str) or not isinstance(template_id, str) or frozen is None
                    or frozen.get("split") != split or frozen.get("template_id") != template_id):
                raise ValueError(f"paired task does not match the frozen {split} split manifest")
            from .readiness import _train_identity
            if pair.get("gamefile") != frozen.get("gamefile") or _train_identity(str(pair.get("gamefile")))[1] != task_id:
                raise ValueError("paired gamefile does not match its frozen native task identity")
            if frozen.get("identity_status") != "IDENTIFIED" or frozen.get("confirmatory_eligible") is not True:
                raise ValueError(f"{split} pair is not confirmatory-eligible in the frozen split manifest")
            cluster_id = frozen.get("cluster_id")
            if (task_id in source_ids or template_id in source_templates or template_id in b_templates
                    or cluster_id in source_clusters or cluster_id in b_clusters
                    or type(pair.get("task_type_id")) is not int or not pair.get("gamefile")):
                raise ValueError("proposal, B viability, and paired validation templates/clusters must be disjoint")
            if pair.get("cluster_id") != cluster_id:
                raise ValueError("pair plan semantic cluster differs from canonical task membership")
            if (task_id in pair_task_ids[split] or template_id in pair_templates[split]
                    or cluster_id in pair_clusters[split]):
                raise ValueError(f"{split} validation requires unique task, template, and semantic cluster")
            if pair.get("arm_order") not in (["off", "on"], ["on", "off"]):
                raise ValueError("each pair must freeze its off/on execution order")
            pair_task_ids[split].add(task_id)
            pair_templates[split].add(template_id)
            pair_clusters[split].add(cluster_id)
    if (pair_templates["train_inner"] & pair_templates["train_outer"]
            or pair_clusters["train_inner"] & pair_clusters["train_outer"]):
        raise ValueError("inner and outer pair template and semantic cluster groups must be disjoint")
    if not set(evidence.source_task_ids).issubset(split_by_task):
        raise ValueError("source task missing from split manifest")
    if source_clusters != {split_by_task[task_id].get("cluster_id") for task_id in source_ids}:
        raise ValueError("source semantic clusters do not match the frozen task membership")
    for row in source_receipts:
        task = split_by_task.get(row["task_id"])
        if (task is None or task.get("split") != "train_inner"
                or task.get("template_id") != row.get("template_id")
                or task.get("cluster_id") != row.get("cluster_id")
                or task.get("identity_status") != "IDENTIFIED"
                or task.get("confirmatory_eligible") is not True):
            raise ValueError("source receipt does not match the frozen train_inner manifest")
    from .readiness import _FAMILIES
    family_by_type = {task_type: family for family, task_type in _FAMILIES.items()}
    source_families = {
        row.get("task_family") or family_by_type.get(split_by_task[row["task_id"]].get("task_type"))
        for row in source_receipts
    }
    if source_families != {evidence.task_family} or {receipt_outcome(row) for row in source_receipts} != {True, False}:
        raise ValueError("source receipts do not provide a same-family verified success/failure contrast")

    max_steps = manifest.get("max_steps")
    input_cap = manifest.get("max_input_per_request")
    completion_cap = manifest.get("max_completion_per_request")
    if any(type(value) is not int or value <= 0 for value in (max_steps, input_cap, completion_cap)):
        raise ProviderBudgetExceeded("finite step and per-request token caps are required")
    proposal_calls = manifest.get("proposal_completion_calls")
    if proposal_calls != 1:
        raise ProviderBudgetExceeded("one bounded proposal completion is required for this patch stage")
    required_calls = proposal_calls + (len(inner_pairs) + len(outer_pairs)) * 2 * max_steps
    if manifest.get("max_calls") != required_calls:
        raise ProviderBudgetExceeded("manifest call cap must equal complete proposal and paired schedule")
    return {
        "status": "READY_FOR_APPROVAL",
        "provider_calls": 0,
        "planned_provider_calls": required_calls,
        "inner_pairs": len(inner_pairs),
        "outer_pairs": len(outer_pairs),
        "source_tasks": len(source_receipts),
        "B_independent_wins": len(win_clusters),
        "B_verified_failures": len(b_failures),
        "source_templates": sorted(source_templates),
        "source_clusters": sorted(source_clusters),
        "paired_templates": {split: sorted(values) for split, values in pair_templates.items()},
        "paired_clusters": {split: sorted(values) for split, values in pair_clusters.items()},
        "results_claimed": False,
    }


def prepare_skill_learning(
    manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    viability_receipts: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]],
    evidence: EvidenceBundle,
    approval: Mapping[str, Any],
    pricing: Mapping[str, Any],
    *,
    revision: str,
    base_url: str,
) -> ApprovedProviderBudget:
    if manifest.get("controller_commit") != revision:
        raise ProviderBudgetExceeded("run_revision_mismatch")
    if manifest.get("max_steps", 0) > approval.get("per_episode_max_steps", 0):
        raise ProviderBudgetExceeded("episode_horizon_outside_approval")
    plan = validate_learning_inputs(manifest, split_manifest, viability_receipts, source_receipts, evidence)
    budget = ApprovedProviderBudget(
        manifest, approval, pricing,
        controller_commit=revision,
        stage="skill_evolution",
        model=manifest["model"],
        base_url=base_url,
    )
    calls = plan["planned_provider_calls"]
    max_input = manifest["max_input_per_request"]
    max_output = manifest["max_completion_per_request"]
    budget.ensure_capacity(calls, calls * max_input, calls * max_output)
    return budget


def _run_one_pair(data_root: Path, client: Any, pair: Mapping[str, Any], candidate: SkillContract,
                  *, split: str, max_steps: int, run_id: str, agent_revision: str,
                  benchmark_revision: str, provenance: Mapping[str, Any],
                  on_arm: Callable[[str, dict[str, Any]], None]) -> tuple[dict[str, Any], dict[str, Any]]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from experiments.external_real.alfworld_stateful_runner import _alfworld_config, run_episode

    gamefile = pair["gamefile"]
    rows = {}
    for arm in pair["arm_order"]:
        arm_id = f"{run_id}:{pair['pair_id']}:{arm}"
        receipt_metadata = {
            "study_id": "skill_evolution_real_v2", "run_id": run_id,
            "pair_id": pair["pair_id"], "trial_id": arm_id,
            "task_id": pair["task_id"], "template_id": pair["template_id"],
            "cluster_id": pair["cluster_id"],
            "arm": arm, "split": split, "evidence_origin": "REAL_AGENT",
            "skill_digest": candidate.digest if arm == "on" else None,
            "skill_version_sha256": candidate.digest if arm == "on" else None,
            "skill_sha": candidate.digest if arm == "on" else None,
            "agent_revision": agent_revision, "benchmark_revision": benchmark_revision,
            "agent_revision_sha256": provenance.get("agent_revision_sha256") or provenance.get("agent_sha"),
            "agent_sha": provenance.get("agent_sha") or provenance.get("agent_revision_sha256"),
            "prompt_template_sha256": provenance.get("prompt_template_sha256"),
            "configuration_sha": provenance.get("configuration_sha"),
            "model_request_name": provenance["model_request_name"],
            "source_commit": provenance["source_commit"],
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "seed": pair["seed"],
        }
        if isinstance(client.budget, ApprovedProviderBudget):
            steps = max_steps * (2 - len(rows))
            client.budget.ensure_capacity(
                steps, steps * client.max_input_per_request,
                steps * client.max_completion_tokens,
            )
        owner = AlfredTWEnv(_alfworld_config(data_root, [pair["task_type_id"]], max_steps), train_eval="train")
        owner.game_files = [gamefile]
        owner.num_games = 1
        environment = owner.init_env(batch_size=1)
        environment.seed(int(pair["seed"]))
        try:
            row = run_episode(
                environment, client,
                skill=candidate.body if arm == "on" else None,
                skill_version=str(candidate.version) if arm == "on" else None,
                max_steps=max_steps,
                receipt_metadata=receipt_metadata,
                expected_gamefiles=[gamefile],
            )
        finally:
            environment.close()
        row["receipt_sha256"] = manifest_digest(row)
        rows[arm] = row
        on_arm(arm, row)
    return rows["off"], rows["on"]


def run_live_learning(
    manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], viability_receipts: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]], evidence: EvidenceBundle, parent: SkillContract,
    budget: ApprovedProviderBudget, *, data_root: Path, output: Path, api_key: str, base_url: str,
) -> dict[str, Any]:
    """Run one approved patch proposal, then frozen inner and outer pairs."""
    if parent.task_family != evidence.task_family:
        raise ValueError("same-family source contrast does not match the parent skill")
    from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps({
        "manifest": manifest, "split_manifest_sha256": manifest["split_manifest_sha256"],
        "plan_sha256": manifest["plan_sha256"], "parent": parent.to_dict(),
        "parent_digest": parent.digest, "candidate_status": "proposal_pending",
        "results_claimed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "arm_trajectories.jsonl").open("x", encoding="utf-8"):
        pass
    proposal_record: dict[str, Any] = {"status": "pending"}
    proposal_path = output / "proposal.json"

    def persist_proposal(record: Mapping[str, Any]) -> None:
        proposal_record.update(record)
        with proposal_path.open("w", encoding="utf-8") as stream:
            json.dump(proposal_record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    persist_proposal(proposal_record)
    def persist(snapshot: dict[str, Any]) -> None:
        with (output / "provider_journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    client = DeepSeekStatefulClient(
        api_key, base_url, manifest["model"], budget=budget,
        max_completion_tokens=manifest["max_completion_per_request"], persist_budget=persist,
    )
    client.max_input_per_request = manifest["max_input_per_request"]
    client.receipt_metadata = {"study_id": manifest["study_id"], "run_id": manifest["run_id"],
                               "approval_reference": budget.approval["approval_reference"]}
    proposer = ModelSkillProposer(
        client, max_completion_calls=1,
        max_completion_tokens=manifest["max_completion_per_request"],
        max_skill_characters=manifest.get("max_skill_characters", 1800),
        on_completion=persist_proposal,
    )
    try:
        patch_proposal = proposer.propose_patch(parent, evidence)
    except (ValueError, ProviderBudgetExceeded) as exc:
        status = "REJECTED_PROPOSAL" if isinstance(exc, ValueError) else "BUDGET_EXHAUSTED"
        persist_proposal({"status": status, "reason": str(exc)})
        report = {"status": status, "reason": str(exc), "provider_calls": budget.calls_reserved,
                  "provider_budget": budget.as_dict(), "canonical_promotion": False, "results_claimed": False}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    persist_proposal({
        "status": "validated",
        "patch": {"operation": patch_proposal.patch.operation, "index": patch_proposal.patch.index,
                  "text": patch_proposal.patch.text},
        "failure_hypothesis": patch_proposal.failure_hypothesis,
        "expected_benefit": patch_proposal.expected_benefit,
        "possible_regressions": list(patch_proposal.possible_regressions),
        "evidence_trace_ids": list(patch_proposal.evidence_trace_ids),
        "prompt_sha256": patch_proposal.prompt_sha256, "output_sha256": patch_proposal.output_sha256,
        "model_identity": patch_proposal.model_identity,
    })
    iteration = manifest.get("iteration", 1)
    optimizer = SkillPatchOptimizer(
        max_edit_ops_per_round=1, max_iterations=manifest.get("max_iterations", 1),
        max_candidate_count=1, max_skill_characters=manifest.get("max_skill_characters", 1800),
    )
    candidate, errors = optimizer.apply(parent, (patch_proposal.patch,), iteration=iteration)
    if candidate is None:
        raise ValueError("model patch failed deterministic optimizer: " + ",".join(errors))
    candidate = SkillContract.from_dict({
        **candidate.to_dict(),
        "source_split": "train_inner",
        "source_trial_ids": sorted(set(parent.source_trial_ids) | set(evidence.source_trace_ids)),
        "source_task_ids": sorted(set(parent.source_task_ids) | set(evidence.source_task_ids)),
        "source_template_ids": sorted(set(parent.source_template_ids) | set(evidence.source_template_ids)),
        "source_cluster_ids": sorted(set(parent.source_cluster_ids) | set(evidence.source_cluster_ids)),
        "source_trace_hashes": sorted(set(parent.source_trace_hashes) | set(evidence.source_trace_hashes)),
        "train_split_manifest_sha256": evidence.train_split_manifest_sha256,
        "generator_provider": "deepseek",
        "model_identity": patch_proposal.model_identity,
        "proposal_prompt_sha256": patch_proposal.prompt_sha256,
        "proposal_output_sha256": patch_proposal.output_sha256,
        "runtime_agent_sha": evidence.agent_hash,
    })
    if candidate.parent_digest != parent.digest:
        raise ValueError("candidate parent digest changed during proposal binding")
    (output / "candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    receipt_rows: list[dict[str, Any]] = []
    def persist_arm(split: str, pair_id: str, arm: str, row: dict[str, Any]) -> None:
        with (output / "arm_trajectories.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"split": split, "pair_id": pair_id, "arm": arm,
                                     "episode": row}, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def run_pairs(pairs: Sequence[Mapping[str, Any]], split: str) -> list[PairReceipt] | None:
        split_receipts: list[PairReceipt] = []
        for pair in pairs:
            off, on = _run_one_pair(
                data_root, client, pair, candidate, split=split, max_steps=manifest["max_steps"],
                run_id=manifest["run_id"], agent_revision=manifest["agent_revision"],
                benchmark_revision=manifest["benchmark_revision"],
                provenance=manifest,
                on_arm=lambda arm, row: persist_arm(split, pair["pair_id"], arm, row),
            )
            compact = []
            for arm, row in (("off", off), ("on", on)):
                summary = {
                    "trial_id": row["trial_id"], "pair_id": pair["pair_id"], "task_id": pair["task_id"],
                    "template_id": pair["template_id"], "cluster_id": pair["cluster_id"],
                    "seed": pair["seed"], "split": split,
                    "arm": arm, "skill_digest": candidate.digest if arm == "on" else None,
                    "agent_revision": manifest["agent_revision"],
                    "benchmark_revision": manifest["benchmark_revision"],
                    "success": receipt_outcome(row),
                    "status": row["status"], "steps": row["steps"],
                    "provider_budget": row["provider_budget"], "evidence_origin": "REAL_AGENT",
                }
                summary["receipt_sha256"] = row["receipt_sha256"]
                compact.append(summary)
                receipt_rows.append(summary)
            if any(row["success"] is None for row in compact):
                (output / "paired_receipts.json").write_text(
                    json.dumps(receipt_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
                return None
            split_receipts.extend(PairReceipt(
                trial_id=summary["trial_id"], pair_id=summary["pair_id"], task_id=summary["task_id"],
                seed=summary["seed"], arm=summary["arm"], split=split,
                skill_digest=summary["skill_digest"], agent_revision=summary["agent_revision"],
                benchmark_revision=summary["benchmark_revision"], success=summary["success"],
                utility=float(summary["success"]), verifier_status="verified", hard_gates_passed=True,
                template_id=summary["template_id"], cluster_id=summary["cluster_id"],
                receipt_sha256=summary["receipt_sha256"],
            ) for summary in compact)
            (output / "paired_receipts.json").write_text(
                json.dumps(receipt_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
        return split_receipts

    evolution = SkillEvolution(SkillPatchOptimizer(max_candidate_count=1), SkillEvolutionStore(output / "evolution"))
    preview, preview_errors = evolution.optimizer.apply(parent, (patch_proposal.patch,), iteration=iteration)
    if preview is None:
        raise ValueError("evolution preview rejected deterministic patch: " + ",".join(preview_errors))
    preview = SkillContract.from_dict({
        **preview.to_dict(),
        "source_split": "train_inner",
        "source_trial_ids": sorted(set(parent.source_trial_ids) | set(evidence.source_trace_ids)),
        "source_task_ids": sorted(set(parent.source_task_ids) | set(evidence.source_task_ids)),
        "source_template_ids": sorted(set(parent.source_template_ids) | set(evidence.source_template_ids)),
        "source_cluster_ids": sorted(set(parent.source_cluster_ids) | set(evidence.source_cluster_ids)),
        "source_trace_hashes": sorted(set(parent.source_trace_hashes) | set(evidence.source_trace_hashes)),
        "train_split_manifest_sha256": evidence.train_split_manifest_sha256,
        "generator_provider": "deepseek", "model_identity": patch_proposal.model_identity,
        "proposal_prompt_sha256": patch_proposal.prompt_sha256,
        "proposal_output_sha256": patch_proposal.output_sha256, "runtime_agent_sha": evidence.agent_hash,
    })
    if preview.digest != candidate.digest:
        raise ValueError("evolution candidate preview differs from the candidate frozen for rollouts")
    inner_receipts = run_pairs(manifest["plan"]["pairs"], "train_inner")
    if inner_receipts is None:
        report = {"status": "INCOMPLETE_PAIR_UNKNOWN", "provider_calls": budget.calls_reserved,
                  "paired_receipts": receipt_rows, "provider_budget": budget.as_dict(), "results_claimed": False}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    result = evolution.run_round(
        parent, (patch_proposal.patch,), inner_receipts, iteration=iteration, split="train_inner",
        source_trial_ids=evidence.source_trace_ids, source_task_ids=evidence.source_task_ids,
        source_template_ids=evidence.source_template_ids, source_cluster_ids=evidence.source_cluster_ids,
        source_trace_hashes=evidence.source_trace_hashes,
        train_split_manifest_sha256=evidence.train_split_manifest_sha256,
        generator_provider="deepseek", model_identity=patch_proposal.model_identity,
        proposal_prompt_sha256=patch_proposal.prompt_sha256,
        proposal_output_sha256=patch_proposal.output_sha256, runtime_agent_sha=evidence.agent_hash,
    )
    if result.candidate is None or result.evidence is None or result.assessment_status != "CANDIDATE_ACCEPTED":
        report = {"status": result.assessment_status, "candidate_digest": result.candidate.digest if result.candidate else None,
                  "inner_status": result.assessment_status, "outer_status": "NOT_RUN_INNER_NOT_PROMISING",
                  "canonical_promotion": False, "provider_calls": budget.calls_reserved,
                  "paired_receipts": receipt_rows, "provider_budget": budget.as_dict(), "results_claimed": False}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    if result.candidate.digest != candidate.digest:
        raise ValueError("evolution result differs from the candidate frozen for rollouts")
    outer_receipts = run_pairs(manifest["plan"]["outer_pairs"], "train_outer")
    if outer_receipts is None:
        report = {"status": "INCOMPLETE_OUTER_PAIR_UNKNOWN", "candidate_digest": result.candidate.digest,
                  "inner_status": result.assessment_status, "provider_calls": budget.calls_reserved,
                  "paired_receipts": receipt_rows, "provider_budget": budget.as_dict(), "results_claimed": False}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    outer_evidence = approve_outer_validation(
        result.candidate, result.candidate.digest, result.evidence, outer_receipts,
        split="train_outer", alpha=manifest.get("alpha", 0.05),
    )
    outer_accepted = evolution.store.record_outer_assessment(result.candidate, parent, outer_evidence)
    report = {
        "status": outer_evidence.assessment_status,
        "inner_status": result.assessment_status,
        "outer_status": outer_evidence.assessment_status,
        "accepted_for_further_review": outer_accepted,
        "canonical_promotion": False,
        "candidate_digest": result.candidate.digest if result.candidate else None,
        "proposal_prompt_sha256": patch_proposal.prompt_sha256,
        "proposal_output_sha256": patch_proposal.output_sha256,
        "source_trace_hashes": list(evidence.source_trace_hashes),
        "paired_receipts": receipt_rows,
        "inner_pair_count": len(result.evidence.pairs) if result.evidence else 0,
        "outer_pair_count": len(outer_evidence.pairs),
        "outer_evidence_ids": list(outer_evidence.evidence_trial_ids),
        "inner_repairs": result.evidence.repairs if result.evidence else None,
        "inner_regressions": result.evidence.regressions if result.evidence else None,
        "inner_utility_delta": result.evidence.utility_delta if result.evidence else None,
        "inner_utility_interval": result.evidence.utility_interval if result.evidence else None,
        "outer_repairs": outer_evidence.repairs,
        "outer_regressions": outer_evidence.regressions,
        "outer_utility_delta": outer_evidence.utility_delta,
        "outer_utility_interval": outer_evidence.utility_interval,
        "provider_budget": budget.as_dict(),
        "results_claimed": False,
    }
    if outer_accepted:
        export_skill_pack(
            replace(result.candidate, verification_trial_ids=tuple(sorted(
                set(result.candidate.verification_trial_ids) | set(outer_evidence.evidence_trial_ids)))),
            output / "skill-pack",
            agent_revision=manifest["agent_revision"], model_identity=patch_proposal.model_identity,
            verifier_revision=manifest["verifier_revision"], task_population=manifest["task_population"],
        )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split-manifest", "viability-receipts", "source-receipts",
                 "source-evidence", "parent", "approval", "pricing", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="execute the explicitly approved C-stage run")
    args = parser.parse_args(argv)
    manifest = _read_json(args.manifest)
    split_manifest = _read_json(args.split_manifest)
    b_payload = _read_json(args.viability_receipts)
    source_payload = _read_json(args.source_receipts)
    b_rows = _rows(b_payload, "episodes")
    source_rows = _rows(source_payload, "receipts")
    evidence = _evidence_bundle(_read_json(args.source_evidence))
    parent = SkillContract.from_dict(_read_json(args.parent))
    plan = validate_learning_inputs(manifest, split_manifest, b_rows, source_rows, evidence)
    if not args.live:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise FileExistsError("live output path must be new")
    if not manifest.get("output_dir") or args.output.resolve() != Path(manifest["output_dir"]).resolve():
        raise ProviderBudgetExceeded("run_output_directory_mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("live execution requires a clean committed checkout")
    if manifest.get("viability_receipts_sha256") != manifest_digest(b_payload):
        raise ProviderBudgetExceeded("B_receipt_file_digest_mismatch")
    if manifest.get("source_receipts_sha256") != manifest_digest(source_payload):
        raise ProviderBudgetExceeded("source_receipt_file_digest_mismatch")
    approval, pricing = _read_json(args.approval), _read_json(args.pricing)
    base_url = manifest.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ProviderBudgetExceeded("approved_api_base_required")
    budget = prepare_skill_learning(
        manifest, split_manifest, b_rows, source_rows, evidence, approval, pricing,
        revision=revision, base_url=base_url,
    )
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
    report = run_live_learning(
        manifest, split_manifest, b_rows, source_rows, evidence, parent, budget,
        data_root=args.data_root, output=args.output, api_key=api_key, base_url=base_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
