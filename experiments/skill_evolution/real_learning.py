"""Approved train_dev skill induction and train_inner feedback evaluation.

Dry-run is the default. Live calls require a C-scoped ApprovedProviderBudget,
eligible B viability receipts, disjoint frozen templates, and a complete panel
budget before the provider client or environment is created.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
    verify_inner_pairs,
    export_skill_pack,
)

PHASE4_T2_MODE = "PANEL_A_OFF_V1_PANEL_B_OFF_V1_V2"
PHASE4_T2_MODE_VERSION = 1
PHASE4_SELECTION_POLICY = "MIN_REGRESSION_MAX_NET_SUCCESS_COST_TOKENS_TIE_V1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _redact_feedback_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = re.sub(r"(?i)\b([a-z]+)\s+\d+\b", r"\1", value)
    value = re.sub(r"(?i)(?:[A-Z]:\\|/(?:home|Users|mnt|private|tmp)/)\S+", "<path>", value)
    return value


def _evidence_bundle(value: Mapping[str, Any]) -> EvidenceBundle:
    tuple_fields = {
        "source_task_ids", "source_template_ids", "source_cluster_ids", "source_trace_ids", "source_trace_hashes",
        "successful_trajectory_summaries", "failed_trajectory_summaries", "first_divergence_events",
        "verified_positive_patterns", "verified_failure_patterns", "subgoal_compliance_events",
        "action_rejections", "loop_or_stall_events", "budget_failures",
        "paired_execution_metrics",
    }
    return EvidenceBundle(**{
        field.name: tuple(value[field.name]) if field.name in tuple_fields else value[field.name]
        for field in fields(EvidenceBundle) if field.name in value
    })


def source_bundle_from_receipts(source_rows: Sequence[Mapping[str, Any]],
                                split_manifest: Mapping[str, Any], *, source_mode: str) -> EvidenceBundle:
    """Project verified dev trajectories, never expert/state/reasoning fields."""
    from experiments.external_real.alfworld_runner import task_family_name
    members = {row["task_id"]: row for row in split_manifest["task_membership"]}
    if not source_rows:
        raise ValueError("real train_dev source receipts are required")
    positive, negative, families, agents = [], [], set(), set()
    for row in source_rows:
        member = members.get(row.get("task_id"), {})
        if (row.get("split") != "train_dev" or member.get("split") != "train_dev"
                or row.get("evidence_origin") != "REAL_AGENT" or row.get("teacher_only_source")
                or (row.get("provider_budget") or {}).get("tokens_unknown", True)
                or not row.get("trajectory") or receipt_outcome(row) is None
                or row.get("cluster_id") != member.get("cluster_id")
                or row.get("template_id") != member.get("template_id")):
            raise ValueError("source must be a complete autonomous resolved train_dev receipt")
        if row.get("receipt_sha256") != manifest_digest({k: v for k, v in row.items() if k != "receipt_sha256"}):
            raise ValueError("source receipt digest mismatch")
        families.add(task_family_name(member.get("task_type")))
        agents.add(row.get("agent_revision_sha256"))
        visible = [{key: _redact_feedback_text(turn[key]) for key in ("observation", "selected_action", "fallback_reason")
                    if key in turn} for turn in row["trajectory"]]
        # Bounded source summaries disclose truncation; labels still use the
        # complete official receipt, never a model's self-reported success.
        text = json.dumps({"official_success": receipt_outcome(row), "visible_events": visible}, ensure_ascii=False)
        text = text[:2400]
        (positive if receipt_outcome(row) else negative).append(text)
    if len(families) != 1 or len(agents) != 1 or None in agents:
        raise ValueError("induction sources require one frozen agent and one family")
    if source_mode == "SUCCESS_ONLY_INDUCTION" and negative:
        raise ValueError("success-only sources cannot include failed episodes")
    if source_mode == "CONTRASTIVE_REPAIR" and (not positive or not negative):
        raise ValueError("contrastive repair requires genuine success and failure episodes")
    disclosure = {"fields": ["observation", "selected_action", "fallback_reason"],
                  "redaction": "entity instance numbers removed", "summary_character_cap_per_trace": 2400,
                  "labels": "complete official receipts", "teacher_or_reasoning_or_state_fields": "excluded"}
    bundle = EvidenceBundle(source_split="train_dev", source_mode=source_mode,
        source_task_ids=tuple(row["task_id"] for row in source_rows),
        source_template_ids=tuple(row["template_id"] for row in source_rows),
        source_cluster_ids=tuple(row["cluster_id"] for row in source_rows),
        source_trace_ids=tuple(row["trial_id"] for row in source_rows),
        source_trace_hashes=tuple(row["receipt_sha256"] for row in source_rows),
        train_split_manifest_sha256=manifest_digest(split_manifest), task_family=next(iter(families)),
        agent_hash=next(iter(agents)), successful_trajectory_summaries=tuple(positive),
        failed_trajectory_summaries=tuple(negative),
        verified_failure_patterns=("official non-success episode",) if negative else (),
        redaction_report_sha256=manifest_digest(disclosure))
    bundle.validate()
    return bundle


def _feedback_bundle(source: EvidenceBundle, receipts: Sequence[PairReceipt],
                     raw_receipts: Sequence[Mapping[str, Any]] = ()) -> EvidenceBundle:
    """Derive patch feedback only from completed official inner pair receipts."""
    pairs: dict[str, dict[str, PairReceipt]] = {}
    for receipt in receipts:
        pairs.setdefault(receipt.pair_id, {})[receipt.arm] = receipt
    if not pairs or any(set(arms) != {"off", "on"} for arms in pairs.values()):
        raise ValueError("patch feedback requires complete inner off/on pairs")
    ordered = [pairs[key] for key in sorted(pairs)]
    summaries = []
    for arms in ordered:
        off, on = arms["off"], arms["on"]
        summaries.append((off, on))
    successes = tuple(
        f"{arm.arm} arm official outcome: success; verifier_status={arm.verifier_status}; utility={arm.utility:g}."
        for pair in summaries for arm in pair if arm.success is True
    )
    failures = tuple(
        f"{arm.arm} arm official outcome: failure; verifier_status={arm.verifier_status}; utility={arm.utility:g}."
        for pair in summaries for arm in pair if arm.success is False
    )
    traces = tuple(arm.trial_id for pair in summaries for arm in pair)
    hashes = tuple(arm.receipt_sha256 for pair in summaries for arm in pair)
    if any(not digest for digest in hashes):
        raise ValueError("inner feedback requires receipt hashes for every arm")
    if not successes:
        raise ValueError("inner feedback has no verified success; stop before patching")
    metrics = []
    for row in raw_receipts:
        events = row.get("feedback_events", [])
        informative = [event for event in events if event.get("fallback_reason")]
        selected_events = informative[:2] if informative else (events[:1] + events[-1:] if events else [])
        visible_events = [json.dumps(event, ensure_ascii=False, sort_keys=True)[:160]
                          for event in selected_events]
        metrics.append(
            f"{row.get('pair_id')} {row.get('arm')}: outcome={row.get('success')}; "
            f"input_tokens={row.get('input_tokens')}; completion_tokens={row.get('completion_tokens')}; "
            f"reasoning_tokens={row.get('reasoning_tokens')}; usd={row.get('cost_usd')}; "
            f"environment_steps={row.get('steps')}; actions={row.get('actions')}; "
            f"visible_events={visible_events}; full_event_count={len(events)}"
        )
    return EvidenceBundle(
        source_split="train_inner", source_mode=source.source_mode,
        source_task_ids=tuple(pair[0].task_id for pair in summaries),
        source_template_ids=tuple(pair[0].template_id for pair in summaries),
        source_cluster_ids=tuple(pair[0].cluster_id for pair in summaries),
        source_trace_ids=traces, source_trace_hashes=hashes,
        train_split_manifest_sha256=source.train_split_manifest_sha256,
        task_family=source.task_family, agent_hash=source.agent_hash,
        successful_trajectory_summaries=successes,
        failed_trajectory_summaries=failures,
        verified_failure_patterns=("official_environment_failure",) if failures else (),
        paired_execution_metrics=tuple(metrics),
        redaction_report_sha256=source.redaction_report_sha256,
    )


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    rows = payload.get(key) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} must be a JSON list of receipt objects")
    return rows


def freeze_pair_orders(pairs: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Balance OFF-first and ON-first using metadata only, before outcomes."""
    if type(seed) is not int:
        raise ValueError("predeclared pair order seed is required")
    ranked = sorted(pairs, key=lambda pair: manifest_digest({"seed": seed, "pair_id": pair["pair_id"]}))
    orders = {pair["pair_id"]: (["off", "on"] if index % 2 == 0 else ["on", "off"])
              for index, pair in enumerate(ranked)}
    return [dict(pair, arm_order=orders[pair["pair_id"]]) for pair in pairs]


def freeze_triplet_orders(pairs: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Freeze balanced deterministic OFF/v1/v2 permutations before outcomes."""
    if type(seed) is not int:
        raise ValueError("predeclared panel B arm order seed is required")
    orders = (
        ["off", "v1", "v2"], ["v1", "v2", "off"], ["v2", "off", "v1"],
        ["off", "v2", "v1"], ["v2", "v1", "off"], ["v1", "off", "v2"],
    )
    ranked = sorted(pairs, key=lambda pair: manifest_digest({"seed": seed, "pair_id": pair["pair_id"]}))
    order_by_id = {pair["pair_id"]: orders[index % len(orders)] for index, pair in enumerate(ranked)}
    return [dict(pair, arm_order=order_by_id[pair["pair_id"]]) for pair in pairs]


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
    if b_gate.get("agent_viability") != "VERIFIED_TRAIN_VIABLE":
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
    if None in win_clusters:
        raise ProviderBudgetExceeded("B_viability_receipts_require_semantic_clusters")
    b_clusters = win_clusters
    b_templates = {row.get("template_id") for row in (*b_successes, *b_failures)}
    if None in b_templates:
        raise ProviderBudgetExceeded("B_viability_requires_template_identity")

    source_by_id = {row.get("trial_id"): row for row in source_receipts}
    if len(source_by_id) != len(source_receipts) or None in source_by_id:
        raise ValueError("train_dev source receipt IDs must be unique")
    if set(evidence.source_trace_ids) != set(source_by_id):
        raise ValueError("evidence trace IDs do not match the frozen source receipt set")
    if evidence.source_split != "train_dev":
        raise ValueError("initial skill induction sources must be train_dev")
    if any(row.get("split") != "train_dev" or receipt_outcome(row) is None
           or row.get("evidence_origin") != "REAL_AGENT" or row.get("teacher_only_source", False)
           for row in source_receipts):
        raise ValueError("induction sources must be autonomous, verifier-resolved train_dev receipts")
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
    if (manifest.get("phase4_mode") != PHASE4_T2_MODE
            or type(manifest.get("phase4_mode_version")) is not int
            or manifest.get("phase4_mode_version") != PHASE4_T2_MODE_VERSION):
        raise ProviderBudgetExceeded("explicit_versioned_phase4_t2_mode_required")
    if manifest.get("source_mode") != evidence.source_mode:
        raise ProviderBudgetExceeded("source_mode_manifest_evidence_mismatch")
    selection_policy = manifest.get("selection_policy")
    if selection_policy != {"id": PHASE4_SELECTION_POLICY, "version": 1}:
        raise ProviderBudgetExceeded("frozen_phase4_selection_policy_required")
    plan_data = manifest.get("plan", {})
    panel_a = plan_data.get("panel_a_pairs")
    panel_b = plan_data.get("panel_b_pairs")
    if not isinstance(panel_a, list) or not isinstance(panel_b, list):
        raise ValueError("Phase4 manifest requires frozen panel_a_pairs and panel_b_pairs")
    if len(panel_a) != len(panel_b) or len(panel_a) not in {6, 8}:
        raise ValueError("Phase4 panels must both contain exactly 6 (MIN) or 8 (STANDARD) clusters")
    if len({pair.get("pair_id") for pair in panel_a + panel_b}) != len(panel_a) + len(panel_b):
        raise ValueError("panel A and B pair IDs must be unique")
    if "outer_pairs" in manifest.get("plan", {}):
        raise ProviderBudgetExceeded("outer evaluation requires a separate T3 approval")
    pair_templates: dict[str, set[str]] = {"panel_a": set(), "panel_b": set()}
    pair_clusters: dict[str, set[str]] = {"panel_a": set(), "panel_b": set()}
    pair_task_ids: dict[str, set[str]] = {"panel_a": set(), "panel_b": set()}
    for panel_name, pairs, expected_arms, order_seed, freezer in (
        ("panel_a", panel_a, {"off", "on"}, plan_data.get("pair_order_seed"), freeze_pair_orders),
        ("panel_b", panel_b, {"off", "v1", "v2"}, plan_data.get("panel_b_arm_order_seed"), freeze_triplet_orders),
    ):
        if pairs != freezer(pairs, order_seed):
            raise ValueError(f"{panel_name} arm order differs from frozen randomization")
        for pair in pairs:
            split = "train_inner"
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
            if pair.get("arm_order") is None or set(pair["arm_order"]) != expected_arms or len(pair["arm_order"]) != len(expected_arms):
                raise ValueError(f"{panel_name} must freeze each required arm exactly once")
            if (task_id in pair_task_ids[panel_name] or template_id in pair_templates[panel_name]
                    or cluster_id in pair_clusters[panel_name]):
                raise ValueError(f"{panel_name} requires unique task, template, and semantic cluster identities")
            pair_task_ids[panel_name].add(task_id)
            pair_templates[panel_name].add(template_id)
            pair_clusters[panel_name].add(cluster_id)
    for panel_name, values in (("task", pair_task_ids), ("template", pair_templates), ("cluster", pair_clusters)):
        if values["panel_a"] & values["panel_b"]:
            raise ValueError(f"panel A and B {panel_name} sets must be disjoint")
    if not set(evidence.source_task_ids).issubset(split_by_task):
        raise ValueError("source task missing from split manifest")
    if source_clusters != {split_by_task[task_id].get("cluster_id") for task_id in source_ids}:
        raise ValueError("source semantic clusters do not match the frozen task membership")
    for row in source_receipts:
        task = split_by_task.get(row["task_id"])
        if (task is None or task.get("split") != "train_dev"
                or task.get("template_id") != row.get("template_id")
                or task.get("cluster_id") != row.get("cluster_id")
                or task.get("identity_status") != "IDENTIFIED"):
            raise ValueError("source receipt does not match the frozen train_dev manifest")
    from experiments.external_real.alfworld_runner import TASK_FAMILIES, task_family_name
    stable_families = set(TASK_FAMILIES.values())
    def normalize_family(value: Any) -> str:
        if value in stable_families:
            return value
        return task_family_name(value)
    source_families = set()
    for row in source_receipts:
        membership_family = normalize_family(split_by_task[row["task_id"]].get("task_type"))
        declared = row.get("task_family")
        if declared is not None and normalize_family(declared) != membership_family:
            raise ValueError("source receipt family differs from canonical six-family mapping")
        source_families.add(membership_family)
    source_outcomes = {receipt_outcome(row) for row in source_receipts}
    if source_families != {evidence.task_family} or True not in source_outcomes:
        raise ValueError("train_dev induction sources require a same-family verified success")
    if evidence.source_mode == "SUCCESS_ONLY_INDUCTION":
        if source_outcomes != {True} or evidence.failed_trajectory_summaries:
            raise ValueError("success-only induction cannot include source failures")
    elif source_outcomes != {True, False}:
        raise ValueError("contrastive induction requires real resolved success and failure sources")

    max_steps = manifest.get("max_steps")
    input_cap = manifest.get("max_input_per_request")
    completion_cap = manifest.get("max_completion_per_request")
    if any(type(value) is not int or value <= 0 for value in (max_steps, input_cap, completion_cap)):
        raise ProviderBudgetExceeded("finite step and per-request token caps are required")
    try:
        episode_usd_cap = Decimal(str(manifest["max_usd_per_episode"]))
    except (KeyError, ValueError, TypeError, InvalidOperation):
        raise ProviderBudgetExceeded("finite per-episode USD cap is required") from None
    if (not episode_usd_cap.is_finite() or episode_usd_cap <= 0
            or type(manifest.get("max_calls_per_episode")) is not int
            or manifest["max_calls_per_episode"] < max_steps
            or type(manifest.get("max_episode_wall_seconds")) is not int
            or manifest["max_episode_wall_seconds"] <= 0):
        raise ProviderBudgetExceeded("finite per-episode call, USD, and wall caps are required")
    proposal_calls = manifest.get("proposal_completion_calls")
    if proposal_calls != 1:
        raise ProviderBudgetExceeded("one bounded proposal completion is required for this patch stage")
    patch_calls = manifest.get("patch_completion_calls", 1)
    if patch_calls != 1:
        raise ProviderBudgetExceeded("one bounded inner-feedback patch completion is required")
    required_calls = proposal_calls + patch_calls + (len(panel_a) * 2 + len(panel_b) * 3) * max_steps
    if manifest.get("max_calls") != required_calls:
        raise ProviderBudgetExceeded("manifest call cap must equal complete proposal and paired schedule")
    return {
        "status": "READY_FOR_APPROVAL",
        "provider_calls": 0,
        "planned_provider_calls": required_calls,
        "phase4_mode": PHASE4_T2_MODE,
        "phase4_mode_version": PHASE4_T2_MODE_VERSION,
        "panel_a_clusters": len(panel_a),
        "panel_b_clusters": len(panel_b),
        "outer_pairs": 0,
        "source_tasks": len(source_receipts),
        "B_independent_wins": len(win_clusters),
        "B_verified_failures": len({row.get("cluster_id") for row in viability_receipts if (
            receipt_outcome(row) is False and row.get("evidence_origin") == "REAL_AGENT"
            and not row.get("teacher_only_source", False))}),
        "source_templates": sorted(source_templates),
        "source_clusters": sorted(source_clusters),
        "paired_templates": {panel: sorted(values) for panel, values in pair_templates.items()},
        "paired_clusters": {panel: sorted(values) for panel, values in pair_clusters.items()},
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


def _success_only_inner_screen(receipts: Sequence[Mapping[str, Any]], *, split: str = "train_inner") -> dict[str, Any]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in receipts:
        if row.get("split") == split:
            pairs.setdefault(str(row.get("pair_id")), {})[str(row.get("arm"))] = row
    regressions = sum(arms.get("off", {}).get("success") is True
                      and arms.get("on", {}).get("success") is False for arms in pairs.values())
    complete = [arms for arms in pairs.values() if set(arms) == {"off", "on"}
                and arms["off"].get("success") is not None and arms["on"].get("success") is not None]
    missing = len(pairs) - len(complete)
    both_success = [arms for arms in complete if arms["off"].get("success") is True
                    and arms["on"].get("success") is True]
    savings = []
    for arms in both_success:
        off_input, off_completion = arms["off"].get("input_tokens"), arms["off"].get("completion_tokens")
        on_input, on_completion = arms["on"].get("input_tokens"), arms["on"].get("completion_tokens")
        if not all(type(value) is int for value in (off_input, off_completion, on_input, on_completion)):
            continue
        off_tokens, on_tokens = off_input + off_completion, on_input + on_completion
        if type(off_tokens) is int and type(on_tokens) is int and off_tokens > 0:
            savings.append((off_tokens - on_tokens) / off_tokens)
    savings.sort()
    median_saving = ((savings[len(savings) // 2 - 1] + savings[len(savings) // 2]) / 2
                     if savings and len(savings) % 2 == 0 else savings[len(savings) // 2] if savings else None)
    status = ("REJECTED_REGRESSION" if regressions else "UNDECIDED_INCOMPLETE" if missing
              else "READY_FOR_T3_PILOT" if median_saving is not None and median_saving >= 0.10
              else "NO_NET_EFFICIENCY_GAIN")
    return {"status": status, "regressions": regressions, "missing_pairs": missing,
            "both_success_pairs": len(both_success), "median_total_token_saving_fraction": median_saving,
            "usd_secondary_endpoint": [row.get("cost_usd") for arms in both_success for row in arms.values()]}


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def select_inner_candidate(rows: Sequence[Mapping[str, Any]], source_mode: str,
                           panel_b_pairs: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Select only from complete same-cluster panel B triplets; ties favor v1."""
    route = "SUCCESS_ONLY_EFFICIENCY" if source_mode == "SUCCESS_ONLY_INDUCTION" else source_mode
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("panel") == "B":
            arms = grouped.setdefault(str(row.get("pair_id")), {})
            arm = str(row.get("arm"))
            if arm in arms:
                raise ValueError("duplicate panel B receipt arm")
            arms[arm] = row
    planned = {str(pair["pair_id"]) for pair in panel_b_pairs} if panel_b_pairs is not None else set(grouped)
    complete = {
        pair_id: arms for pair_id, arms in grouped.items()
        if pair_id in planned and set(arms) == {"off", "v1", "v2"}
        and all(type(arms[arm].get("success")) is bool for arm in ("off", "v1", "v2"))
    }
    incomplete = sorted(planned - set(complete))
    screens: dict[str, dict[str, Any]] = {}
    for version in ("v1", "v2"):
        repairs = regressions = 0
        both_success_count = 0
        both_success_tokens: list[float] = []
        both_success_usd_savings: list[float] = []
        for arms in complete.values():
            off, treated = arms["off"], arms[version]
            repairs += off["success"] is False and treated["success"] is True
            regressions += off["success"] is True and treated["success"] is False
            if off["success"] is not True or treated["success"] is not True:
                continue
            both_success_count += 1
            token_values = [off.get("input_tokens"), off.get("completion_tokens"),
                            treated.get("input_tokens"), treated.get("completion_tokens")]
            if all(type(value) is int for value in token_values):
                off_tokens = token_values[0] + token_values[1]
                treated_tokens = token_values[2] + token_values[3]
                if off_tokens > 0:
                    both_success_tokens.append((off_tokens - treated_tokens) / off_tokens)
            off_cost, treated_cost = off.get("cost_usd"), treated.get("cost_usd")
            if off_cost is not None and treated_cost is not None:
                both_success_usd_savings.append(float(Decimal(str(off_cost)) - Decimal(str(treated_cost))))
        median_token_saving = _median(both_success_tokens)
        median_usd_saving = _median(both_success_usd_savings)
        net_success = repairs - regressions
        eligible = bool(complete) and not incomplete and regressions == 0 and (
            median_token_saving is not None and median_token_saving >= 0.10
            if route == "SUCCESS_ONLY_EFFICIENCY" else net_success > 0
        )
        screens[version] = {
            "complete_triplets": len(complete), "incomplete_triplets_excluded": len(incomplete),
            "repairs": repairs, "regressions": regressions, "net_success": net_success,
            "both_success_pairs": both_success_count,
            "both_success_token_pairs": len(both_success_tokens),
            "median_both_success_total_token_saving_fraction": median_token_saving,
            "median_both_success_usd_saving": median_usd_saving,
            "eligible_for_outer_pilot": eligible,
        }
    eligible = [version for version in ("v1", "v2") if screens[version]["eligible_for_outer_pilot"]]
    # Lexicographic safety-first rule: fewest regressions, highest net success,
    # then lower measured USD and total tokens on both-success pairs; tie -> v1.
    def rank(version: str) -> tuple[float, ...]:
        screen = screens[version]
        usd = screen["median_both_success_usd_saving"]
        tokens = screen["median_both_success_total_token_saving_fraction"]
        return (screen["regressions"], -screen["net_success"],
                -(usd if usd is not None else float("-inf")),
                -(tokens if tokens is not None else float("-inf")),
                0 if version == "v1" else 1)
    selected = min(eligible, key=rank) if eligible else "no_skill"
    pairwise = {}
    for left, right in (("off", "v1"), ("off", "v2"), ("v1", "v2")):
        valid = [(arms[left], arms[right]) for arms in complete.values()]
        wins = sum(a["success"] is False and b["success"] is True for a, b in valid)
        losses = sum(a["success"] is True and b["success"] is False for a, b in valid)
        pairwise[f"{left}_vs_{right}"] = {
            "valid_pairs": len(valid), "left_only_success": losses, "right_only_success": wins,
            "incomplete_triplets_excluded": len(incomplete),
        }
    return {
        "selected": selected, "candidate_screens": screens, "selection_count": 2,
        "research_route": route, "panel_b_complete_triplets": len(complete),
        "panel_b_incomplete_triplet_ids": incomplete, "pairwise_comparisons": pairwise,
        "selection_policy": PHASE4_SELECTION_POLICY,
        "status": "READY_FOR_T3_PILOT" if eligible else "NO_QUALIFIED_INNER_CANDIDATE",
        "claim": "exploratory panel B selection only; panel A is adaptation evidence; separate T3 required",
    }


def _selected_panel_b_pairs(rows: Sequence[Mapping[str, Any]], version: str) -> list[dict[str, Any]]:
    """Normalize complete raw panel B triplets to the existing OFF/ON pair contract."""
    if version not in {"v1", "v2"}:
        raise ValueError("selected inner version must be v1 or v2")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("panel") == "B":
            grouped.setdefault(str(row.get("pair_id")), {})[str(row.get("arm"))] = row
    normalized = []
    for pair_id, arms in sorted(grouped.items()):
        if set(arms) != {"off", "v1", "v2"} or any(type(arms[arm].get("success")) is not bool
                                                         for arm in ("off", "v1", "v2")):
            continue
        for source_arm, pair_arm in (("off", "off"), (version, "on")):
            normalized.append({**arms[source_arm], "arm": pair_arm, "pair_id": f"{pair_id}:{version}"})
    return normalized


def _run_panel_cluster(data_root: Path, client: Any, pair: Mapping[str, Any],
                       candidates: Mapping[str, SkillContract | None], *, split: str,
                       max_steps: int, run_id: str, agent_revision: str,
                       benchmark_revision: str, provenance: Mapping[str, Any],
                       on_arm: Callable[[str, dict[str, Any]], None],
                       environment_factory: Callable[..., Any] | None = None) -> dict[str, dict[str, Any]]:
    from experiments.external_real.alfworld_stateful_runner import run_episode
    if environment_factory is None:
        from experiments.skill_evolution.native_verifier import make_selected_environment
        environment_factory = make_selected_environment

    gamefile = pair["gamefile"]
    rows: dict[str, dict[str, Any]] = {}
    if set(pair["arm_order"]) != set(candidates):
        raise ValueError("frozen cluster arm order does not match candidate arms")
    if isinstance(client.budget, ApprovedProviderBudget):
        arm_count = len(pair["arm_order"])
        client.budget.ensure_capacity(
            arm_count * max_steps, arm_count * max_steps * client.max_input_per_request,
            arm_count * max_steps * client.max_completion_tokens,
        )
    for arm in pair["arm_order"]:
        if isinstance(client.budget, ApprovedProviderBudget):
            client.budget.begin_episode(
                max_usd=str(provenance["max_usd_per_episode"]),
                max_calls=provenance["max_calls_per_episode"],
                max_wall_seconds=provenance["max_episode_wall_seconds"],
            )
        candidate = candidates[arm]
        arm_id = f"{run_id}:{pair['pair_id']}:{arm}"
        receipt_metadata = {
            "study_id": provenance.get("study_id", "skill_evolution_real_v2"), "run_id": run_id,
            "pair_id": pair["pair_id"], "trial_id": arm_id,
            "task_id": pair["task_id"], "template_id": pair["template_id"],
            "cluster_id": pair["cluster_id"],
            "arm": arm, "split": split, "evidence_origin": getattr(client, "evidence_origin", "REAL_AGENT"),
            "skill_digest": candidate.digest if candidate is not None else None,
            "skill_version_sha256": candidate.digest if candidate is not None else None,
            "skill_sha": candidate.digest if candidate is not None else None,
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
        environment = environment_factory(gamefile, max_steps=max_steps, seed=int(pair["seed"]))
        try:
            row = run_episode(
                environment, client,
                skill=candidate.body if candidate is not None else None,
                skill_version=str(candidate.version) if candidate is not None else None,
                max_steps=max_steps,
                receipt_metadata=receipt_metadata,
                expected_gamefiles=[gamefile],
            )
        finally:
            environment.close()
        row["receipt_sha256"] = manifest_digest(row)
        rows[arm] = row
        on_arm(arm, row)
    return rows


def _run_one_pair(data_root: Path, client: Any, pair: Mapping[str, Any], candidate: SkillContract,
                  *, split: str, max_steps: int, run_id: str, agent_revision: str,
                  benchmark_revision: str, provenance: Mapping[str, Any],
                  on_arm: Callable[[str, dict[str, Any]], None],
                  environment_factory: Callable[..., Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _run_panel_cluster(data_root, client, pair, {"off": None, "on": candidate}, split=split,
        max_steps=max_steps, run_id=run_id, agent_revision=agent_revision,
        benchmark_revision=benchmark_revision, provenance=provenance, on_arm=on_arm,
        environment_factory=environment_factory)
    return rows["off"], rows["on"]


def run_live_learning(
    manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], viability_receipts: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]], evidence: EvidenceBundle, parent: SkillContract,
    budget: ApprovedProviderBudget, *, data_root: Path, output: Path, api_key: str, base_url: str,
    client_factory: Callable[..., Any] | None = None,
    environment_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Generate v1 from train_dev, patch it from real inner feedback, and stop at T2."""
    if parent.task_family != evidence.task_family or evidence.source_split != "train_dev":
        raise ValueError("initial proposal requires same-family train_dev source evidence")
    validate_learning_inputs(manifest, split_manifest, viability_receipts, source_receipts, evidence)
    for cap_name in ("max_usd_per_episode", "max_calls_per_episode", "max_episode_wall_seconds"):
        if str(manifest[cap_name]) != str(budget.approval.get(cap_name)):
            raise ProviderBudgetExceeded(f"episode_cap_manifest_approval_mismatch:{cap_name}")
    if "outer_pairs" in manifest.get("plan", {}):
        raise ProviderBudgetExceeded("T2 cannot contain outer tasks; submit a separate T3 manifest")
    if client_factory is None:
        from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
        client_factory = DeepSeekStatefulClient
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps({
        "manifest": manifest, "split_manifest_sha256": manifest["split_manifest_sha256"],
        "plan_sha256": manifest["plan_sha256"], "parent_digest": parent.digest,
        "candidate_status": "initial_proposal_pending", "results_claimed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "arm_trajectories.jsonl").write_text("", encoding="utf-8")
    with (output / "provider_journal.jsonl").open("x", encoding="utf-8"):
        pass
    with (output / "proposal_journal.jsonl").open("x", encoding="utf-8"):
        pass

    def append(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def persist_budget(snapshot: dict[str, Any]) -> None:
        append(output / "provider_journal.jsonl", snapshot)

    client = client_factory(api_key, base_url, manifest["model"], budget=budget,
        max_completion_tokens=manifest["max_completion_per_request"], persist_budget=persist_budget)
    client.max_input_per_request = manifest["max_input_per_request"]
    client.receipt_metadata = {"study_id": manifest["study_id"], "run_id": manifest["run_id"],
        "approval_reference": budget.approval["approval_reference"]}
    proposer = ModelSkillProposer(client, max_completion_calls=2,
        max_completion_tokens=manifest["max_completion_per_request"],
        max_skill_characters=manifest.get("max_skill_characters", 1800),
        on_completion=lambda record: append(output / "proposal_journal.jsonl", record))
    try:
        budget.begin_episode(max_usd=str(manifest.get("max_usd_per_proposal", "0.05")),
                             max_calls=1, max_wall_seconds=3600)
        initial = proposer.propose_initial(evidence, skill_id=parent.skill_id, subtask=parent.subtask,
                                           constraints=manifest.get("skill_constraints", ()))
    except (ValueError, ProviderBudgetExceeded) as exc:
        report = {"status": "REJECTED_INITIAL_PROPOSAL", "reason": str(exc),
                  "candidate_identity": None, "provider_budget": budget.as_dict(),
                  "canonical_promotion": False, "outer_status": "NOT_RUN", "results_claimed": False}
        append(output / "proposal_journal.jsonl", {"status": report["status"], "reason": str(exc),
            "source_trace_ids": list(evidence.source_trace_ids), "source_trace_hashes": list(evidence.source_trace_hashes)})
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    (output / "candidate_v1.json").write_text(json.dumps(initial.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    receipt_rows: list[dict[str, Any]] = []

    def run_panel(pairs: Sequence[Mapping[str, Any]], panel: str,
                  candidates: Mapping[str, SkillContract | None]) -> None:
        for pair in pairs:
            tagged = dict(pair)
            arm_rows = _run_panel_cluster(data_root, client, tagged, candidates, split="train_inner",
                max_steps=manifest["max_steps"], run_id=manifest["run_id"],
                agent_revision=manifest["agent_revision"], benchmark_revision=manifest["benchmark_revision"],
                provenance=manifest, environment_factory=environment_factory,
                on_arm=lambda arm, row: append(output / "arm_trajectories.jsonl",
                    {"split": "train_inner", "panel": panel, "pair_id": tagged["pair_id"],
                     "arm": arm, "episode": row}))
            for arm in pair["arm_order"]:
                row = arm_rows[arm]
                trajectory = row.get("trajectory", [])
                def total(field: str) -> int | None:
                    values = [item.get(field) for item in trajectory if item.get(field) is not None]
                    return sum(values) if len(values) == len(trajectory) and all(type(value) is int for value in values) else None
                receipt_rows.append({
                    "trial_id": row["trial_id"], "panel": panel, "pair_id": tagged["pair_id"],
                    "task_id": pair["task_id"], "template_id": pair["template_id"],
                    "cluster_id": pair["cluster_id"], "seed": pair["seed"], "split": "train_inner",
                    "arm": arm, "skill_digest": candidates[arm].digest if candidates[arm] else None,
                    "agent_revision": manifest["agent_revision"], "benchmark_revision": manifest["benchmark_revision"],
                    "success": receipt_outcome(row), "status": row["status"], "steps": row["steps"],
                    "provider_budget": row["provider_budget"], "evidence_origin": row["evidence_origin"],
                    "input_tokens": row.get("input_tokens"), "completion_tokens": row.get("completion_tokens"),
                    "reasoning_tokens": total("reasoning_tokens"), "actions": row.get("env_steps"),
                    "cost_usd": (str(sum((Decimal(item["estimated_provider_usd"]) for item in trajectory
                                             if item.get("estimated_provider_usd") is not None), Decimal(0)))
                                 if trajectory and all(item.get("estimated_provider_usd") is not None for item in trajectory)
                                 else None),
                    "feedback_events": [{key: _redact_feedback_text(item.get(key)) for key in
                        ("step", "observation", "selected_action", "fallback_reason", "subgoal")
                        if item.get(key) is not None} for item in trajectory] + (
                        [{"step": len(trajectory), "event": "terminal_visible_observation",
                          "observation": _redact_feedback_text(row["state"]["last_observation"])}]
                        if row.get("env_steps", 0) > 0 else []),
                    "receipt_sha256": row["receipt_sha256"],
                })
            (output / "paired_receipts.json").write_text(
                json.dumps(receipt_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_report(report: dict[str, Any]) -> dict[str, Any]:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    plan = manifest["plan"]
    run_panel(plan["panel_a_pairs"], "A", {"off": None, "on": initial})
    panel_a_rows = [row for row in receipt_rows if row["panel"] == "A"]
    if any(row["success"] is not True and row["success"] is not False for row in panel_a_rows):
        return save_report({"status": "PANEL_A_INCOMPLETE_NO_PATCH", "patch_attempt_state": "NOT_ATTEMPTED",
            "candidate_identity": initial.digest, "panel_a_receipts": panel_a_rows,
            "panel_b_status": "NOT_RUN", "outer_status": "NOT_RUN", "provider_budget": budget.as_dict(),
            "canonical_promotion": False, "results_claimed": False})
    panel_a_pair_ids = {row["pair_id"] for row in panel_a_rows}
    patch_feedback_rows = [row for row in panel_a_rows if row["pair_id"] in panel_a_pair_ids]
    v1_receipts = [
        PairReceipt(trial_id=row["trial_id"], pair_id=row["pair_id"], task_id=row["task_id"], seed=row["seed"],
            arm="off" if row["arm"] == "off" else "on", split="train_inner",
            skill_digest=row["skill_digest"], agent_revision=row["agent_revision"],
            benchmark_revision=row["benchmark_revision"], success=row["success"], utility=float(row["success"]),
            verifier_status="verified", hard_gates_passed=True, template_id=row["template_id"],
            cluster_id=row["cluster_id"], receipt_sha256=row["receipt_sha256"])
        for row in panel_a_rows
    ]
    try:
        inner_feedback = _feedback_bundle(evidence, v1_receipts, patch_feedback_rows)
    except ValueError as exc:
        return save_report({"status": "PANEL_A_NO_PATCHABLE_FEEDBACK", "reason": str(exc),
            "patch_attempt_state": "NOT_ATTEMPTED", "candidate_identity": initial.digest,
            "panel_a_receipts": panel_a_rows, "panel_b_status": "NOT_RUN", "outer_status": "NOT_RUN",
            "provider_budget": budget.as_dict(), "canonical_promotion": False, "results_claimed": False})
    try:
        budget.begin_episode(max_usd=str(manifest.get("max_usd_per_proposal", "0.05")),
                             max_calls=1, max_wall_seconds=3600)
        patch_proposal = proposer.propose_patch(initial, inner_feedback)
    except (ValueError, ProviderBudgetExceeded) as exc:
        attempts = [json.loads(line) for line in (output / "proposal_journal.jsonl").read_text(encoding="utf-8").splitlines()]
        response = next((item for item in reversed(attempts) if item.get("status") == "received_unvalidated"), {})
        try:
            parsed = json.loads(response.get("raw_output", ""))
        except (TypeError, json.JSONDecodeError):
            parsed = None
        patch_state = "PATCH_NOOP" if isinstance(parsed, dict) and (
            parsed.get("status") == "NO_OP" or (isinstance(parsed.get("edit_operation"), dict)
                                                   and parsed["edit_operation"].get("op", "").casefold() in {"no_op", "noop", "none"})
        ) else "PATCH_REJECTED"
        status = "PATCH_NOOP" if patch_state == "PATCH_NOOP" else "REJECTED_PATCH"
        append(output / "proposal_journal.jsonl", {"status": status, "reason": str(exc),
            "parent_digest": initial.digest, "feedback_receipt_hashes": [r.receipt_sha256 for r in v1_receipts]})
        return save_report({"status": status, "reason": str(exc), "patch_attempt_state": patch_state,
            "candidate_identity": initial.digest, "panel_a_receipts": panel_a_rows,
            "panel_b_status": "NOT_RUN", "outer_status": "NOT_RUN", "provider_budget": budget.as_dict(),
            "canonical_promotion": False, "results_claimed": False})
    optimizer = SkillPatchOptimizer(max_edit_ops_per_round=1, max_iterations=1, max_candidate_count=1,
        max_skill_characters=manifest.get("max_skill_characters", 1800))
    candidate, errors = optimizer.apply(initial, (patch_proposal.patch,), iteration=1)
    if candidate is None or candidate.parent_digest != initial.digest:
        reason = ";".join(errors) if errors else "patch_parent_digest_mismatch"
        append(output / "proposal_journal.jsonl", {"status": "REJECTED_PATCH", "reason": reason,
            "parent_digest": initial.digest, "candidate_digest": candidate.digest if candidate else None,
            "feedback_receipt_hashes": [r.receipt_sha256 for r in v1_receipts]})
        return save_report({"status": "REJECTED_PATCH", "reason": reason, "patch_attempt_state": "PATCH_REJECTED",
            "candidate_identity": initial.digest, "panel_a_receipts": panel_a_rows,
            "panel_b_status": "NOT_RUN", "outer_status": "NOT_RUN", "provider_budget": budget.as_dict(),
            "canonical_promotion": False, "results_claimed": False})
    candidate = replace(candidate, source_split=initial.source_split, source_mode=initial.source_mode,
        evidence_basis=initial.evidence_basis,
        patch_feedback_receipt_hashes=tuple(sorted(set(r.receipt_sha256 for r in v1_receipts))),
        generator_provider="deepseek", model_identity=patch_proposal.model_identity,
        proposal_prompt_sha256=patch_proposal.prompt_sha256, proposal_output_sha256=patch_proposal.output_sha256)
    (output / "candidate_v2.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_panel(plan["panel_b_pairs"], "B", {"off": None, "v1": initial, "v2": candidate})
    panel_b_plan = [dict(pair) for pair in plan["panel_b_pairs"]]
    selection = select_inner_candidate(receipt_rows, evidence.source_mode, panel_b_plan)
    v2_receipts = []
    complete_b = {pair_id: arms for pair_id, arms in {
        pair_id: {row["arm"]: row for row in receipt_rows if row["panel"] == "B" and row["pair_id"] == pair_id}
        for pair_id in {row["pair_id"] for row in receipt_rows if row["panel"] == "B"}
    }.items() if set(arms) == {"off", "v1", "v2"}
       and all(type(arms[arm]["success"]) is bool for arm in ("off", "v1", "v2"))}
    for pair_id, arms in complete_b.items():
        for arm, receipt_arm in (("off", "off"), ("v2", "on")):
            row = arms[arm]
            v2_receipts.append(PairReceipt(trial_id=row["trial_id"], pair_id=pair_id, task_id=row["task_id"],
                seed=row["seed"], arm=receipt_arm, split="train_inner",
                skill_digest=candidate.digest if arm == "v2" else None,
                agent_revision=row["agent_revision"], benchmark_revision=row["benchmark_revision"],
                success=row["success"], utility=float(row["success"]), verifier_status="verified",
                hard_gates_passed=True, template_id=row["template_id"], cluster_id=row["cluster_id"],
                receipt_sha256=row["receipt_sha256"]))
    result = None
    if v2_receipts:
        result = SkillEvolution(SkillPatchOptimizer(max_candidate_count=1),
            SkillEvolutionStore(output / "evolution")).run_round(initial, (patch_proposal.patch,), v2_receipts,
            iteration=1, split="train_inner", patch_feedback_receipt_hashes=inner_feedback.source_trace_hashes,
            train_split_manifest_sha256=evidence.train_split_manifest_sha256,
            generator_provider="deepseek", model_identity=patch_proposal.model_identity,
            proposal_prompt_sha256=patch_proposal.prompt_sha256, proposal_output_sha256=patch_proposal.output_sha256,
            runtime_agent_sha=evidence.agent_hash)
    selected_candidate = {"v1": initial, "v2": candidate}.get(selection["selected"])
    if selected_candidate is not None:
        export_skill_pack(selected_candidate, output / "selected_skill_pack",
            agent_revision=manifest["agent_revision"], model_identity=selected_candidate.model_identity,
            verifier_revision=manifest["benchmark_revision"], task_population="ALFWorld train pilot, not a global certificate")
    stage_status = selection["status"]
    report = {"status": stage_status, "inner_status": result.assessment_status if result else "NOT_ASSESSED_NO_COMPLETE_TRIPLETS",
        "phase4_mode": PHASE4_T2_MODE, "phase4_mode_version": PHASE4_T2_MODE_VERSION,
        "patch_attempt_state": "PATCH_APPLIED", "research_route": selection["research_route"],
        "outer_status": "NOT_RUN_REQUIRES_SEPARATE_T3_APPROVAL", "initial_candidate_digest": initial.digest,
        "candidate_digest": selected_candidate.digest if selected_candidate else None,
        "inner_selection": selection,
        "initial_proposal_prompt_sha256": initial.proposal_prompt_sha256,
        "initial_proposal_output_sha256": initial.proposal_output_sha256,
        "patch_prompt_sha256": patch_proposal.prompt_sha256, "patch_output_sha256": patch_proposal.output_sha256,
        "panel_a_receipts": panel_a_rows, "panel_b_triplet_receipts": [row for row in receipt_rows if row["panel"] == "B"],
        "panel_b_plan": panel_b_plan,
        "selected_frozen": ({
            "version": selection["selected"], "candidate_digest": selected_candidate.digest,
            "body_sha256": selected_candidate.body_digest, "selection_policy": PHASE4_SELECTION_POLICY,
            "panel": "B", "raw_triplet_receipts_sha256": manifest_digest(
                {"receipts": [row for row in receipt_rows if row["panel"] == "B"]}),
            "panel_b_plan_sha256": manifest_digest({"pairs": panel_b_plan}),
            "panel_b_arm_order_seed": plan["panel_b_arm_order_seed"],
        } if selected_candidate is not None else None),
        # Kept as the T3 CLI handoff field; it now contains all raw A and B rows.
        "inner_v1_and_v2_receipts": receipt_rows,
        "inner_receipts_sha256": manifest_digest({"receipts": receipt_rows}),
        "panel_a_pair_count": len(plan["panel_a_pairs"]), "panel_b_triplet_count": len(plan["panel_b_pairs"]),
        "canonical_promotion": False, "provider_budget": budget.as_dict(), "results_claimed": False}
    return save_report(report)


def validate_outer_inputs(manifest: Mapping[str, Any], parent_split: Mapping[str, Any],
                          extension: Mapping[str, Any], candidate: SkillContract,
                          parent: SkillContract | None, inner_rows: Sequence[Mapping[str, Any]], *,
                          panel_b_pairs: Sequence[Mapping[str, Any]] | None = None,
                          selected_frozen: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed unless a separately approved frozen T3 candidate has fresh child tasks."""
    if manifest.get("stage") != "skill_outer_validation":
        raise ProviderBudgetExceeded("outer_validation_requires_T3_stage")
    if manifest.get("selected_inner_version") not in {"v1", "v2"}:
        raise ProviderBudgetExceeded("selected_inner_version_required")
    if not manifest.get("outer_consumption_journal"):
        raise ProviderBudgetExceeded("shared_outer_consumption_journal_required")
    if manifest.get("parent_split_manifest_sha256") != manifest_digest(parent_split):
        raise ProviderBudgetExceeded("outer_parent_split_digest_mismatch")
    if manifest.get("split_extension_sha256") != manifest_digest(extension):
        raise ProviderBudgetExceeded("outer_split_extension_digest_mismatch")
    if (manifest.get("frozen_candidate_sha256") != candidate.digest
            or candidate.parent_digest != (parent.digest if parent is not None else None)):
        raise ProviderBudgetExceeded("outer_candidate_or_parent_identity_mismatch")
    if extension.get("parent_manifest_sha256") != manifest.get("parent_split_manifest_sha256"):
        raise ProviderBudgetExceeded("outer_extension_parent_mismatch")
    if manifest.get("inner_receipts_sha256") != manifest_digest({"receipts": list(inner_rows)}):
        raise ProviderBudgetExceeded("outer_inner_receipts_digest_mismatch")
    child_rows = extension.get("child_membership")
    if not isinstance(child_rows, list):
        raise ValueError("outer split extension has no child membership")
    child_by_task = {row.get("task_id"): row for row in child_rows if isinstance(row, Mapping)}
    parent_rows = parent_split.get("task_membership")
    if not isinstance(parent_rows, list):
        raise ValueError("parent split has no canonical task membership")
    parent_by_task = {row.get("task_id"): row for row in parent_rows if isinstance(row, Mapping)}
    plan = manifest.get("plan") or {}
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("T3 requires a frozen nonempty outer pair plan")
    if pairs != freeze_pair_orders(pairs, plan.get("pair_order_seed")):
        raise ValueError("outer pair execution order differs from frozen balanced randomization")
    if not isinstance(panel_b_pairs, Sequence) or not panel_b_pairs or not isinstance(selected_frozen, Mapping):
        raise ProviderBudgetExceeded("versioned_T2_panel_B_handoff_required")
    panel_b_seed = selected_frozen.get("panel_b_arm_order_seed")
    if panel_b_pairs != freeze_triplet_orders(panel_b_pairs, panel_b_seed):
        raise ValueError("T2 panel B order differs from its frozen handoff")
    if selected_frozen.get("panel_b_plan_sha256") != manifest_digest({"pairs": list(panel_b_pairs)}):
        raise ValueError("T2 panel B plan digest differs from its frozen handoff")
    planned_ids = {str(pair.get("pair_id")) for pair in panel_b_pairs}
    receipt_ids = {str(row.get("pair_id")) for row in inner_rows if row.get("panel") == "B"}
    if not receipt_ids.issubset(planned_ids):
        raise ValueError("panel B receipts contain a cluster outside the frozen plan")
    planned_by_id = {str(pair["pair_id"]): pair for pair in panel_b_pairs}
    for row in inner_rows:
        if row.get("panel") != "B":
            continue
        pair = planned_by_id[str(row["pair_id"])]
        if (row.get("arm") not in {"off", "v1", "v2"}
                or row.get("task_id") != pair.get("task_id")
                or row.get("template_id") != pair.get("template_id")
                or row.get("cluster_id") != pair.get("cluster_id")
                or row.get("seed") != pair.get("seed")):
            raise ValueError("panel B receipt identity differs from the frozen plan")
    selection = select_inner_candidate(inner_rows, candidate.source_mode, panel_b_pairs)
    if selection["selected"] != manifest["selected_inner_version"]:
        raise ValueError("outer candidate was not selected by the frozen inner screen")
    if selected_frozen is not None and (
        selected_frozen.get("version") != selection["selected"]
        or selected_frozen.get("candidate_digest") != candidate.digest
        or selected_frozen.get("body_sha256") != candidate.body_digest
        or selected_frozen.get("selection_policy") != PHASE4_SELECTION_POLICY
        or selected_frozen.get("panel") != "B"
        or selected_frozen.get("raw_triplet_receipts_sha256") != manifest_digest({
            "receipts": [row for row in inner_rows if row.get("panel") == "B"]
        })
    ):
        raise ValueError("T3 frozen candidate provenance differs from raw panel B evidence")
    if manifest.get("selected_frozen_sha256") != manifest_digest(selected_frozen):
        raise ProviderBudgetExceeded("T3 approval does not bind the selected frozen inner candidate")
    selected_version = manifest["selected_inner_version"]
    selected_rows = _selected_panel_b_pairs(inner_rows, selected_version)
    selected_on = [row for row in selected_rows if row.get("arm") == "on"]
    if not selected_on:
        raise ValueError("selected candidate has no complete panel B triplets")
    if any(row.get("skill_digest") != candidate.digest for row in selected_on):
        raise ValueError("selected inner receipts do not belong to the frozen candidate")
    inner_clusters = {row.get("cluster_id") for row in inner_rows}
    candidate_sources = set(candidate.source_cluster_ids)
    seen: set[str] = set()
    journal = Path(manifest["outer_consumption_journal"])
    consumed: set[str] = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["split_extension_sha256"] == manifest["split_extension_sha256"]:
                consumed.update(event["cluster_ids"])
    for pair in pairs:
        task_id = pair.get("task_id")
        child = child_by_task.get(task_id)
        frozen_task = parent_by_task.get(task_id)
        if (not isinstance(child, Mapping) or child.get("child_split") != "outer_skill_validation"
                or child.get("parent_split") != "train_outer"
                or pair.get("split") != "outer_skill_validation"
                or pair.get("cluster_id") != child.get("cluster_id")
                or not isinstance(frozen_task, Mapping) or frozen_task.get("split") != "train_outer"
                or not pair.get("gamefile") or pair.get("gamefile") != frozen_task.get("gamefile")
                or pair.get("arm_order") not in (["off", "on"], ["on", "off"])):
            raise ValueError("outer pair does not match the frozen outer_skill_validation child manifest")
        cluster = child.get("cluster_id")
        if cluster in seen or cluster in inner_clusters or cluster in candidate_sources or cluster in consumed:
            raise ValueError("outer cluster is repeated, consumed by inner, or used as an induction source")
        seen.add(cluster)
    max_steps = manifest.get("max_steps")
    if type(max_steps) is not int or max_steps <= 0:
        raise ProviderBudgetExceeded("finite_outer_episode_cap_required")
    required_calls = len(pairs) * 2 * max_steps
    if manifest.get("max_calls") != required_calls:
        raise ProviderBudgetExceeded("outer_budget_must_reserve_every_complete_pair")
    return {"status": "READY_FOR_T3_APPROVAL", "provider_calls": 0,
            "planned_provider_calls": required_calls, "outer_pairs": len(pairs),
            "candidate_digest": candidate.digest, "selected_inner_version": selected_version,
            "selected_inner_pairs": len(selected_rows) // 2,
            "panel_b_incomplete_triplets_excluded": selection["panel_b_incomplete_triplet_ids"],
            "outer_clusters": sorted(seen),
            "canonical_promotion": False, "results_claimed": False}


def _pair_receipt_from_row(row: Mapping[str, Any], candidate_digest: str) -> PairReceipt:
    return PairReceipt(trial_id=row["trial_id"], pair_id=row["pair_id"], task_id=row["task_id"],
        seed=row["seed"], arm=row["arm"], split="train_inner",
        skill_digest=candidate_digest if row["arm"] == "on" else None,
        agent_revision=row["agent_revision"], benchmark_revision=row["benchmark_revision"],
        success=row["success"], utility=float(row["success"]), verifier_status="verified",
        hard_gates_passed=True, template_id=row["template_id"], cluster_id=row["cluster_id"],
        receipt_sha256=row["receipt_sha256"])


def run_frozen_outer(manifest: Mapping[str, Any], parent_split: Mapping[str, Any],
                     extension: Mapping[str, Any], candidate: SkillContract,
                     parent: SkillContract | None, inner_rows: Sequence[Mapping[str, Any]],
                     panel_b_pairs: Sequence[Mapping[str, Any]], selected_frozen: Mapping[str, Any],
                     approval: Mapping[str, Any], pricing: Mapping[str, Any], *,
                     revision: str, base_url: str, api_key: str | None, data_root: Path,
                     output: Path, client_factory: Callable[..., Any] | None = None,
                     environment_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    plan = validate_outer_inputs(manifest, parent_split, extension, candidate, parent, inner_rows,
        panel_b_pairs=panel_b_pairs, selected_frozen=selected_frozen)
    if manifest.get("controller_commit") != revision:
        raise ProviderBudgetExceeded("outer_run_revision_mismatch")
    if output.exists() or Path(manifest.get("output_dir", "")).resolve() != output.resolve():
        raise ProviderBudgetExceeded("outer_output_must_be_new_and_match_manifest")
    budget = ApprovedProviderBudget(manifest, approval, pricing, controller_commit=revision,
        stage="skill_outer_validation", model=manifest["model"], base_url=base_url)
    budget.ensure_capacity(plan["planned_provider_calls"],
        plan["planned_provider_calls"] * manifest["max_input_per_request"],
        plan["planned_provider_calls"] * manifest["max_completion_per_request"])
    if client_factory is None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required at runtime")
        from experiments.external_real.alfworld_stateful_runner import DeepSeekStatefulClient
        client_factory = DeepSeekStatefulClient
    def persist_budget(snapshot: dict[str, Any]) -> None:
        with (output / "provider_journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "provider_journal.jsonl").open("x", encoding="utf-8"):
        pass
    client = client_factory(api_key, base_url, manifest["model"], budget=budget,
        max_completion_tokens=manifest["max_completion_per_request"], persist_budget=persist_budget)
    client.max_input_per_request = manifest["max_input_per_request"]
    rows: list[dict[str, Any]] = []
    receipts: list[PairReceipt] = []
    # This study-level journal survives rejected candidates and different run
    # output directories. Record consumption before reset, not after a good result.
    append_jsonl(Path(manifest["outer_consumption_journal"]), {
        "run_id": manifest["run_id"], "candidate_sha256": candidate.digest,
        "split_extension_sha256": manifest["split_extension_sha256"], "cluster_ids": plan["outer_clusters"],
        "status": "RESERVED_FOR_OUTER_EXECUTION"})
    for pair in manifest["plan"]["pairs"]:
        off, on = _run_one_pair(data_root, client, pair, candidate, split="train_outer",
            max_steps=manifest["max_steps"], run_id=manifest["run_id"],
            agent_revision=manifest["agent_revision"], benchmark_revision=manifest["benchmark_revision"],
            provenance=manifest, environment_factory=environment_factory,
            on_arm=lambda arm, row: append_jsonl(output / "arm_trajectories.jsonl",
                {"pair_id": pair["pair_id"], "arm": arm, "episode": row}))
        for arm, row in (("off", off), ("on", on)):
            outcome = receipt_outcome(row)
            rows.append({"pair_id": pair["pair_id"], "arm": arm, "trial_id": row["trial_id"],
                "task_id": pair["task_id"], "cluster_id": pair["cluster_id"], "split": "outer_skill_validation",
                "success": outcome, "receipt_sha256": row["receipt_sha256"],
                "input_tokens": row.get("input_tokens"), "completion_tokens": row.get("completion_tokens"),
                "reasoning_tokens": sum((step.get("reasoning_tokens") or 0) for step in row.get("trajectory", [])),
                "cost_usd": (str(sum((Decimal(step["estimated_provider_usd"]) for step in row.get("trajectory", [])
                                      if step.get("estimated_provider_usd") is not None), Decimal(0)))
                             if row.get("trajectory") and all(step.get("estimated_provider_usd") is not None
                                                              for step in row.get("trajectory", [])) else None),
                "input_tokens": row.get("input_tokens"), "completion_tokens": row.get("completion_tokens")})
            receipts.append(PairReceipt(trial_id=row["trial_id"], pair_id=pair["pair_id"],
                task_id=pair["task_id"], seed=pair["seed"], arm=arm, split="train_outer",
                skill_digest=candidate.digest if arm == "on" else None,
                agent_revision=manifest["agent_revision"], benchmark_revision=manifest["benchmark_revision"],
                success=outcome, utility=float(outcome) if outcome is not None else 0.0,
                verifier_status="verified" if outcome is not None else row["status"], hard_gates_passed=outcome is not None,
                template_id=pair["template_id"], cluster_id=pair["cluster_id"],
                receipt_sha256=row["receipt_sha256"]))
    (output / "outer_receipts.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(row["success"] is None for row in rows):
        report = {"status": "REAL_OUTER_EVALUATED_UNDECIDED", "reason": "INCOMPLETE_PAIR_UNKNOWN",
                  "canonical_promotion": False, "provider_budget": budget.as_dict(), "results_claimed": False}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    selected_version = manifest.get("selected_inner_version")
    inner_pair_receipts = [_pair_receipt_from_row(row, candidate.digest)
                           for row in _selected_panel_b_pairs(inner_rows, selected_version)]
    inner_evidence = verify_inner_pairs(replace(candidate, verification_trial_ids=()), inner_pair_receipts)
    outer_evidence = approve_outer_validation(replace(candidate, verification_trial_ids=()), candidate.digest, inner_evidence, receipts,
        split="train_outer", alpha=manifest.get("alpha", 0.05))
    store = SkillEvolutionStore(output)
    store.save_candidate(candidate)
    if candidate.source_mode == "SUCCESS_ONLY_INDUCTION":
        screen = _success_only_inner_screen(rows, split="outer_skill_validation")
        accepted = False
        if screen["status"] == "REJECTED_REGRESSION":
            store.rollback(candidate, parent, "outer_success_regression", evidence_ids=outer_evidence.evidence_trial_ids)
            outer_status = "REAL_OUTER_REJECTED"
        else:
            outer_status = ("REAL_OUTER_PILOT_FAVORABLE" if screen["status"] == "READY_FOR_T3_PILOT"
                            else "REAL_OUTER_EVALUATED_UNDECIDED")
            store.record_evidence(candidate, outer_evidence)
            store._append_decision(candidate, outer_status.lower(), screen["status"], outer_evidence.evidence_trial_ids)
    else:
        screen = None
        accepted = store.record_outer_assessment(candidate, parent, outer_evidence)
        outer_status = outer_evidence.assessment_status
    report = {"status": outer_status, "candidate_digest": candidate.digest,
        "success_only_efficiency_screen": screen,
        "outer_child_split": "outer_skill_validation", "outer_pairs": len(manifest["plan"]["pairs"]),
        "outer_regressions": outer_evidence.regressions, "outer_repairs": outer_evidence.repairs,
        "outer_utility_delta": outer_evidence.utility_delta, "accepted_for_further_review": accepted,
        "canonical_promotion": False, "provider_budget": budget.as_dict(), "results_claimed": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def outer_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a separately approved frozen T3 outer validation.")
    for name in ("manifest", "split-manifest", "split-extension", "candidate",
                 "inner-receipts", "approval", "pricing", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    manifest = _read_json(args.manifest)
    parent_split, extension = _read_json(args.split_manifest), _read_json(args.split_extension)
    candidate = SkillContract.from_dict(_read_json(args.candidate))
    parent = SkillContract.from_dict(_read_json(args.parent)) if args.parent else None
    inner_payload = _read_json(args.inner_receipts)
    inner_rows = inner_payload.get("inner_v1_and_v2_receipts", inner_payload.get("receipts", inner_payload))
    if not isinstance(inner_rows, list):
        raise ValueError("inner receipts must be a JSON list")
    panel_b_plan = inner_payload.get("panel_b_plan") if isinstance(inner_payload, Mapping) else None
    selected_frozen = inner_payload.get("selected_frozen") if isinstance(inner_payload, Mapping) else None
    plan = validate_outer_inputs(manifest, parent_split, extension, candidate, parent, inner_rows,
        panel_b_pairs=panel_b_plan, selected_frozen=selected_frozen)
    if not args.live:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists() or Path(manifest.get("output_dir", "")).resolve() != args.output.resolve():
        raise ProviderBudgetExceeded("outer_output_must_be_new_and_match_manifest")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("live outer execution requires a clean committed checkout")
    if manifest.get("controller_commit") != revision:
        raise ProviderBudgetExceeded("outer_run_revision_mismatch")
    approval, pricing = _read_json(args.approval), _read_json(args.pricing)
    base_url = manifest.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ProviderBudgetExceeded("approved_api_base_required")
    report = run_frozen_outer(manifest, parent_split, extension, candidate, parent, inner_rows,
        panel_b_plan, selected_frozen,
        approval, pricing, revision=revision, base_url=base_url, api_key=None,
        data_root=args.data_root, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv and actual_argv[0] == "outer":
        return outer_main(actual_argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split-manifest", "viability-receipts", "source-receipts",
                 "parent", "approval", "pricing", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, help="optional frozen bundle; must match the raw-trajectory projection")
    parser.add_argument("--live", action="store_true", help="execute the explicitly approved C-stage run")
    args = parser.parse_args(argv)
    manifest = _read_json(args.manifest)
    split_manifest = _read_json(args.split_manifest)
    b_payload = _read_json(args.viability_receipts)
    source_payload = _read_json(args.source_receipts)
    b_rows = _rows(b_payload, "episodes")
    source_rows = _rows(source_payload, "receipts")
    evidence = source_bundle_from_receipts(source_rows, split_manifest, source_mode=manifest["source_mode"])
    if args.source_evidence and _evidence_bundle(_read_json(args.source_evidence)) != evidence:
        raise ValueError("supplied source bundle differs from the policy-visible receipt projection")
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
