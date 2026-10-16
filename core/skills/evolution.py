"""Train-only contrast mining, bounded patch verification, and Rule OS storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from core.governance import EvolutionDecision
from core.lifecycle import apply_lifecycle_decision
from core.models import EvidenceEvent, RuleState, identifier_digest
from core.mutation_journal import MutationJournal
from core.state_store import StateStore

from .contract import SkillContract, validate_skill_text


@dataclass(frozen=True)
class TrialTrace:
    trial_id: str
    task_id: str
    task_group: str
    split: str
    success: bool | None
    verifier_status: str
    failure_code: str | None
    actions: tuple[str, ...]
    task_family: str
    context_key: str = "default"

    def __post_init__(self) -> None:
        if self.split not in {"train", "calibration", "sealed"}:
            raise ValueError("unknown trace split")
        if self.success is not None and self.verifier_status != "verified":
            raise ValueError("trace outcome requires an independent verified status")


@dataclass(frozen=True)
class Contrast:
    success_trial_id: str
    failure_trial_id: str
    task_group: str
    task_family: str
    success_steps: tuple[str, ...]
    missing_steps: tuple[str, ...]


def _action_step(action: str) -> str | None:
    lowered = action.casefold().strip()
    if lowered.startswith(("go to ", "open ", "look")):
        return "locate_object"
    if lowered.startswith(("pick up ", "take ")):
        return "acquire_object"
    for transform in ("clean", "heat", "cool"):
        if lowered.startswith(transform + " "):
            return f"transform_{transform}"
    if lowered.startswith(("put ", "place ")):
        return "place_object"
    return None


class TraceContrastMiner:
    """Compare verified train trials; free-form observations are never ingested."""

    _excluded_failures = {"api_contract", "budget", "provider_error", "verifier_unavailable", "controller_failure"}

    def mine(self, traces: Sequence[TrialTrace]) -> tuple[Contrast, ...]:
        for trace in traces:
            if trace.split != "train":
                raise ValueError("trace mining accepts train split only")
        eligible = [trace for trace in traces if trace.verifier_status == "verified" and (
            trace.success is True or (trace.success is False and trace.failure_code not in self._excluded_failures)
        )]
        groups: dict[tuple[str, str, str], dict[bool, list[TrialTrace]]] = {}
        for trace in eligible:
            pair_key = (trace.task_group, trace.task_family, trace.context_key)
            groups.setdefault(pair_key, {}).setdefault(bool(trace.success), []).append(trace)
        contrasts: list[Contrast] = []
        for (task_group, task_family, _), arms in sorted(groups.items()):
            if True not in arms or False not in arms:
                continue
            successes = sorted(arms[True], key=lambda item: item.trial_id)
            failures = sorted(arms[False], key=lambda item: item.trial_id)
            for successful, failed in zip(successes, failures):
                success_steps = tuple(step for action in successful.actions if (step := _action_step(action)))
                failure_steps = tuple(step for action in failed.actions if (step := _action_step(action)))
                missing = tuple(step for step in success_steps if step not in failure_steps)
                contrasts.append(Contrast(successful.trial_id, failed.trial_id, task_group, task_family, success_steps, missing))
        return tuple(contrasts)

    def extract_subskill(self, skill_id: str, traces: Sequence[TrialTrace], *, subtask: str = "verified_procedure") -> SkillContract:
        contrasts = self.mine(traces)
        if not contrasts:
            raise ValueError("no matched verified train success/failure contrast")
        steps = tuple(dict.fromkeys(step for contrast in contrasts for step in contrast.success_steps))
        if not steps:
            raise ValueError("matched traces contain no recognized reusable actions")
        family = contrasts[0].task_family
        if any(item.task_family != family for item in contrasts):
            raise ValueError("subskill extraction requires one task family")
        transformation = next((step.removeprefix("transform_") for step in steps if step.startswith("transform_")), None)
        required = ("locate_object", "acquire_object") + ((f"transform_{transformation}",) if transformation else ()) + ("place_object",)
        procedure = tuple({
            "locate_object": "Locate the goal object using visible observations and admissible navigation actions.",
            "acquire_object": "Acquire the goal object and confirm the inventory change from observation.",
            "transform_clean": "Clean the carried object and confirm the clean state from observation.",
            "transform_heat": "Heat the carried object and confirm the heated state from observation.",
            "transform_cool": "Cool the carried object and confirm the cooled state from observation.",
            "place_object": "Place the object in the requested receptacle and inspect the result.",
        }[step] for step in required)
        ids = tuple(sorted({item for contrast in contrasts for item in (contrast.success_trial_id, contrast.failure_trial_id)}))
        return SkillContract(
            skill_id=skill_id,
            version=1,
            task_family=family,
            subtask=subtask,
            intent=f"Complete the {subtask.replace('_', ' ')} with visible state checks.",
            applicability=(f"ALFWorld {family} tasks",),
            procedure=procedure,
            procedure_step_ids=required,
            required_procedure_step_ids=required,
            preconditions=("goal and admissible actions are visible",),
            postconditions=("environment verifier confirms task completion",),
            protected_semantics=("use only admissible actions", "confirm transformations from visible observations"),
            source_trial_ids=ids,
        )


@dataclass(frozen=True)
class Patch:
    operation: str
    index: int
    text: str = ""


@dataclass(frozen=True)
class PairReceipt:
    trial_id: str
    pair_id: str
    task_id: str
    seed: int
    arm: str
    split: str
    skill_digest: str | None
    agent_revision: str
    benchmark_revision: str
    success: bool
    utility: float
    verifier_status: str = "verified"
    hard_gates_passed: bool = True
    template_id: str | None = None
    cluster_id: str | None = None
    receipt_sha256: str | None = None


@dataclass(frozen=True)
class PairResult:
    pair_id: str
    off_trial_id: str
    on_trial_id: str
    off_success: bool
    on_success: bool
    off_utility: float
    on_utility: float
    task_id: str
    seed: int
    agent_revision: str
    benchmark_revision: str
    template_id: str | None = None
    cluster_id: str | None = None
    off_receipt_sha256: str | None = None
    on_receipt_sha256: str | None = None


@dataclass(frozen=True)
class PairedEvidence:
    skill_digest: str
    pairs: tuple[PairResult, ...]
    repairs: int
    regressions: int
    utility_delta: float
    hard_gates_passed: bool
    utility_interval: tuple[float, float]
    assessment_status: str
    split: str = "train"
    alpha: float = 0.05

    @property
    def point_gain_smoke(self) -> bool:
        """Fixture-friendly point estimate; it is not empirical acceptance."""
        return self.utility_delta > 0

    @property
    def evidence_trial_ids(self) -> tuple[str, ...]:
        return tuple(sorted({trial for pair in self.pairs for trial in (pair.off_trial_id, pair.on_trial_id)}))


def _verify_paired_receipts(
    contract: SkillContract,
    receipts: Sequence[PairReceipt],
    *,
    split: str,
    alpha: float,
    forbidden_trial_ids: Sequence[str] = (),
    forbidden_task_ids: Sequence[str] = (),
    forbidden_template_ids: Sequence[str] = (),
    forbidden_cluster_ids: Sequence[str] = (),
) -> PairedEvidence:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not receipts:
        raise ValueError("paired verifier requires receipts")
    by_pair: dict[str, dict[str, PairReceipt]] = {}
    seen_trials: set[str] = set()
    seen_tasks: set[str] = set()
    seen_templates: set[str] = set()
    seen_clusters: set[str] = set()
    forbidden = set(forbidden_trial_ids) | set(contract.source_trial_ids) | set(contract.verification_trial_ids)
    forbidden_tasks = set(forbidden_task_ids) | set(contract.source_task_ids)
    forbidden_templates = set(forbidden_template_ids) | set(contract.source_template_ids)
    forbidden_clusters = set(forbidden_cluster_ids) | set(contract.source_cluster_ids)
    for receipt in receipts:
        if receipt.split != split:
            if split in {"train", "train_inner"}:
                raise ValueError("paired verification accepts train split only")
            raise ValueError(f"paired verification requires {split} split")
        if receipt.trial_id in seen_trials or receipt.trial_id in forbidden:
            raise ValueError("trial evidence reused across mining or verification splits")
        seen_trials.add(receipt.trial_id)
        if receipt.verifier_status != "verified":
            raise ValueError("paired receipt lacks independent verifier status")
        if receipt.receipt_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", receipt.receipt_sha256):
            raise ValueError("paired receipt hash must be a lowercase SHA256 digest")
        if not -1.0 <= receipt.utility <= 1.0:
            raise ValueError("paired utility must be in [-1, 1]")
        expected_digest = contract.digest if receipt.arm == "on" else None
        if receipt.arm not in {"on", "off"} or receipt.skill_digest != expected_digest:
            raise ValueError("paired receipt arm does not match skill version")
        pair = by_pair.setdefault(receipt.pair_id, {})
        if receipt.arm in pair:
            raise ValueError("duplicate arm in paired trial")
        pair[receipt.arm] = receipt
    if any(set(arms) != {"on", "off"} for arms in by_pair.values()):
        raise ValueError("paired verifier requires both arms for every pair")
    results: list[PairResult] = []
    for pair_id, arms in sorted(by_pair.items()):
        off, on = arms["off"], arms["on"]
        if (off.task_id, off.seed, off.agent_revision, off.benchmark_revision) != (
            on.task_id, on.seed, on.agent_revision, on.benchmark_revision
        ):
            raise ValueError("paired arms differ in task, seed, agent, or benchmark contract")
        if off.template_id != on.template_id:
            raise ValueError("paired arms differ in task template")
        if contract.source_template_ids and off.template_id is None:
            raise ValueError("template identity is required when source templates are recorded")
        if off.template_id is not None:
            if off.template_id in forbidden_templates:
                raise ValueError("paired task template overlaps source or prior verification templates")
            if off.template_id in seen_templates:
                raise ValueError("duplicate independent task template across paired evidence")
            seen_templates.add(off.template_id)
        if off.cluster_id != on.cluster_id:
            raise ValueError("paired arms differ in semantic task cluster")
        if contract.source_cluster_ids and off.cluster_id is None:
            raise ValueError("semantic cluster identity is required when source clusters are recorded")
        if off.cluster_id is not None:
            if off.cluster_id in forbidden_clusters:
                raise ValueError("paired semantic cluster overlaps source or prior verification clusters")
            if off.cluster_id in seen_clusters:
                raise ValueError("duplicate independent semantic cluster across paired evidence")
            seen_clusters.add(off.cluster_id)
        if off.task_id in seen_tasks:
            raise ValueError("duplicate independent task and seed group; task ids cannot repeat across seeds")
        if off.task_id in forbidden_tasks:
            raise ValueError("paired task overlaps source or prior verification tasks")
        seen_tasks.add(off.task_id)
        results.append(PairResult(pair_id, off.trial_id, on.trial_id, off.success, on.success, off.utility, on.utility,
                                  off.task_id, off.seed, off.agent_revision, off.benchmark_revision, off.template_id,
                                  off.cluster_id, off.receipt_sha256, on.receipt_sha256))
    repairs = sum(not row.off_success and row.on_success for row in results)
    regressions = sum(row.off_success and not row.on_success for row in results)
    utility_delta = sum(row.on_utility - row.off_utility for row in results) / len(results)
    # Hoeffding interval for paired utility deltas, each bounded to [-2, 2].
    # It intentionally remains wide for small samples; point gain alone is not acceptance.
    import math
    radius = 4.0 * math.sqrt(math.log(2.0 / alpha) / (2 * len(results)))
    interval = (max(-2.0, utility_delta - radius), min(2.0, utility_delta + radius))
    hard_gates_passed = all(arms["off"].hard_gates_passed and arms["on"].hard_gates_passed for arms in by_pair.values())
    assessment = "REJECTED" if not hard_gates_passed or regressions or interval[1] < 0 else (
        "CANDIDATE_ACCEPTED" if interval[0] > 0 else "UNDECIDED"
    )
    return PairedEvidence(
        contract.digest,
        tuple(results),
        repairs,
        regressions,
        utility_delta,
        hard_gates_passed,
        interval,
        assessment,
        split,
        alpha,
    )


def verify_training_pairs(
    contract: SkillContract,
    receipts: Sequence[PairReceipt],
    *,
    forbidden_trial_ids: Sequence[str] = (),
) -> PairedEvidence:
    """Compatibility entry point for the original train-only, 95% verifier."""
    return _verify_paired_receipts(
        contract, receipts, split="train", alpha=0.05,
        forbidden_trial_ids=forbidden_trial_ids,
    )


def verify_inner_pairs(
    contract: SkillContract,
    receipts: Sequence[PairReceipt],
    *,
    alpha: float = 0.05,
) -> PairedEvidence:
    """Verify new train_inner pairs against the proposal's source provenance."""
    return _verify_paired_receipts(contract, receipts, split="train_inner", alpha=alpha)


def approve_outer_validation(
    contract: SkillContract,
    frozen_candidate_digest: str,
    inner_evidence: PairedEvidence,
    outer_receipts: Sequence[PairReceipt],
    *,
    split: str,
    alpha: float,
) -> PairedEvidence:
    """Assess a pre-frozen candidate against held-out evidence without promotion."""
    if split not in {"train_outer", "calibration", "sealed"}:
        raise ValueError("outer validation requires train_outer, calibration, or sealed split")
    if contract.digest != frozen_candidate_digest:
        raise ValueError("candidate differs from its frozen digest")
    if inner_evidence.skill_digest != frozen_candidate_digest:
        raise ValueError("inner evidence refers to a different frozen candidate")
    inner_trial_ids = inner_evidence.evidence_trial_ids
    inner_task_ids = tuple(pair.task_id for pair in inner_evidence.pairs)
    inner_template_ids = tuple(pair.template_id for pair in inner_evidence.pairs if pair.template_id is not None)
    inner_cluster_ids = tuple(pair.cluster_id for pair in inner_evidence.pairs if pair.cluster_id is not None)
    outer = _verify_paired_receipts(
        contract,
        outer_receipts,
        split=split,
        alpha=alpha,
        forbidden_trial_ids=inner_trial_ids,
        forbidden_task_ids=inner_task_ids,
        forbidden_template_ids=inner_template_ids,
        forbidden_cluster_ids=inner_cluster_ids,
    )
    if inner_evidence.assessment_status == "REJECTED" or outer.assessment_status == "REJECTED":
        assessment = "REJECTED"
    elif inner_evidence.assessment_status != "CANDIDATE_ACCEPTED" or outer.assessment_status != "CANDIDATE_ACCEPTED":
        assessment = "UNDECIDED"
    else:
        assessment = "CANDIDATE_ACCEPTED"
    return replace(outer, assessment_status=assessment)


class SkillPatchOptimizer:
    def __init__(
        self,
        *,
        max_edit_ops_per_round: int = 1,
        max_iterations: int = 3,
        max_candidate_count: int = 8,
        max_skill_characters: int = 1800,
    ) -> None:
        self.max_edit_ops_per_round = max_edit_ops_per_round
        self.max_iterations = max_iterations
        self.max_candidate_count = max_candidate_count
        self.max_skill_characters = max_skill_characters
        self._candidate_count = 0
        if min(max_edit_ops_per_round, max_iterations, max_candidate_count, max_skill_characters) < 1:
            raise ValueError("optimizer bounds must be positive")

    def apply(self, parent: SkillContract, patches: Sequence[Patch], *, iteration: int = 1) -> tuple[SkillContract | None, tuple[str, ...]]:
        if iteration > self.max_iterations:
            return None, ("iteration_budget_exhausted",)
        if self._candidate_count >= self.max_candidate_count:
            return None, ("candidate_budget_exhausted",)
        if len(patches) > self.max_edit_ops_per_round:
            return None, ("edit_budget_exceeded",)
        procedure = list(parent.procedure)
        procedure_step_ids = list(parent.procedure_step_ids)
        preconditions = list(parent.preconditions)
        applicability = list(parent.applicability)
        for patch in patches:
            if patch.operation == "ADD_STEP" and 0 <= patch.index <= len(procedure) and patch.text.strip():
                procedure.insert(patch.index, patch.text.strip())
                procedure_step_ids.insert(patch.index, f"optional_{parent.version + iteration}_{patch.index}")
            elif patch.operation == "DELETE_STEP" and 0 <= patch.index < len(procedure):
                if procedure_step_ids[patch.index] in parent.required_procedure_step_ids:
                    return None, ("required_procedure_step_removed",)
                procedure.pop(patch.index)
                procedure_step_ids.pop(patch.index)
            elif patch.operation == "REPLACE_STEP" and 0 <= patch.index < len(procedure) and patch.text.strip():
                if procedure_step_ids[patch.index] in parent.required_procedure_step_ids:
                    return None, ("required_procedure_step_replaced",)
                procedure[patch.index] = patch.text.strip()
            elif patch.operation == "TIGHTEN_PRECONDITION" and patch.text.strip():
                if patch.text.strip() not in preconditions:
                    preconditions.append(patch.text.strip())
            elif patch.operation == "NARROW_SCOPE" and patch.text.strip():
                if patch.text.strip() not in applicability:
                    applicability.append(patch.text.strip())
            else:
                return None, ("invalid_patch_operation_or_index",)
        candidate = replace(
            parent,
            version=parent.version + iteration,
            parent_digest=parent.digest,
            parent_version=parent.version,
            procedure=tuple(procedure),
            procedure_step_ids=tuple(procedure_step_ids),
            preconditions=tuple(preconditions),
            applicability=tuple(applicability),
            verification_trial_ids=(),
            status="candidate",
        )
        errors = validate_skill_text(candidate, max_characters=self.max_skill_characters)
        if not candidate.procedure:
            errors.append("empty_procedure")
        for semantic in parent.protected_semantics:
            if semantic not in candidate.protected_semantics:
                errors.append("protected_semantics_changed")
                break
        if errors:
            return None, tuple(errors)
        self._candidate_count += 1
        return candidate, ()

    @staticmethod
    def assess(evidence: PairedEvidence) -> str:
        return evidence.assessment_status


class SkillEvolutionStore:
    """Persist candidates/evidence through RuleSpec, RuleState, StateStore, and journal."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_store = StateStore(self.root)
        self.journal = MutationJournal(self.root / "evolution" / "mutation_journal.jsonl")
        self.events_path = self.root / "evolution" / "skill_evolution.jsonl"

    def _ensure_genesis(self) -> None:
        if not self.journal.entries():
            self.journal.initialize_genesis(self.root)

    def save_candidate(self, contract: SkillContract) -> Path:
        self._ensure_genesis()
        state = RuleState(rule_id=contract.skill_id, version=contract.version, status="candidate")
        state_path = self.state_store.path_for(state)
        spec_path = state_path.with_name(state_path.name.removesuffix(".state.json") + ".json")
        spec = contract.to_rule_spec().to_dict()
        spec_payload = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
        if spec_path.exists():
            if spec_path.read_text(encoding="utf-8") != spec_payload:
                raise ValueError("immutable skill revision already exists with different content")
        else:
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            with spec_path.open("x", encoding="utf-8") as handle:
                handle.write(spec_payload)
            self.journal.append(
                "add_v2_spec", contract.skill_id, version=contract.version,
                artifact_path=str(spec_path.relative_to(self.root)).replace("\\", "/"),
                digest=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            )
        if state_path.exists():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            desired = state.to_dict()
            existing_semantics = {key: value for key, value in existing.items() if key != "last_updated"}
            desired_semantics = {key: value for key, value in desired.items() if key != "last_updated"}
            if existing_semantics != desired_semantics:
                raise ValueError("skill revision state already transitioned")
        else:
            state_payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n"
            state_path.write_text(state_payload, encoding="utf-8")
            self.journal.append(
                "update_state", contract.skill_id, version=contract.version,
                artifact_path=str(state_path.relative_to(self.root)).replace("\\", "/"),
                digest=hashlib.sha256(state_path.read_bytes()).hexdigest(),
            )
        return spec_path

    def record_evidence(self, contract: SkillContract, evidence: PairedEvidence) -> None:
        if evidence.skill_digest != contract.digest:
            raise ValueError("evidence refers to a different skill contract")
        if evidence.assessment_status not in {"CANDIDATE_ACCEPTED", "UNDECIDED", "REJECTED"}:
            raise ValueError("unknown paired evidence assessment")
        if evidence.split not in {"train", "train_inner", "train_outer", "calibration", "sealed"}:
            raise ValueError("unknown paired evidence split")
        if not evidence.pairs:
            raise ValueError("cannot persist empty paired evidence")
        existing_ids: set[str] = set()
        existing_trial_ids: set[str] = set()
        existing_task_ids: set[str] = set()
        shared_base: dict[str, tuple] = {}
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if "event_id" in event:
                    existing_ids.add(event["event_id"])
                    artifacts = event.get("artifacts", {})
                    same_skill = event.get("versions", {}).get("skill") == contract.digest
                    on_id, off_id = artifacts.get("on_trial_id"), artifacts.get("off_trial_id")
                    if on_id is not None:
                        existing_trial_ids.add(on_id)
                    if off_id is not None:
                        if same_skill:
                            existing_trial_ids.add(off_id)
                        shared_base[off_id] = (
                            event["context"].get("task_id"), event["context"].get("seed"),
                            event["versions"].get("agent"), event["versions"].get("benchmark"),
                            artifacts.get("off_utility"), artifacts.get("off_success"),
                        )
                    task_id = event.get("context", {}).get("task_id")
                    if task_id is not None and event.get("versions", {}).get("skill") == contract.digest:
                        existing_task_ids.add(task_id)
        new_trial_ids = [trial_id for pair in evidence.pairs for trial_id in (pair.off_trial_id, pair.on_trial_id)]
        new_task_ids = [pair.task_id for pair in evidence.pairs]
        for pair in evidence.pairs:
            if pair.on_trial_id in shared_base:
                raise ValueError("a base trial cannot be reused as a skill-treatment trial")
            if pair.off_trial_id in shared_base and shared_base[pair.off_trial_id] != (
                pair.task_id, pair.seed, pair.agent_revision, pair.benchmark_revision,
                pair.off_utility, pair.off_success,
            ):
                raise ValueError("shared base trial has conflicting outcome or execution identity")
        if len(set(new_trial_ids)) != len(new_trial_ids) or existing_trial_ids.intersection(new_trial_ids):
            raise ValueError("paired trial evidence already recorded")
        if len(set(new_task_ids)) != len(new_task_ids) or existing_task_ids.intersection(new_task_ids):
            raise ValueError("paired independent task already recorded")
        new_ids = {
            f"{contract.skill_id}.v{contract.version}.{evidence.split}.{pair.pair_id}"
            for pair in evidence.pairs
        }
        if evidence.split in {"train", "train_inner"}:
            new_ids.update(
                f"{contract.skill_id}.v{contract.version}.{pair.pair_id}"
                for pair in evidence.pairs
            )
        if existing_ids.intersection(new_ids):
            raise ValueError("paired evidence event already recorded")
        self._ensure_genesis()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        old_digest = hashlib.sha256(self.events_path.read_bytes()).hexdigest() if self.events_path.exists() else ""
        with self.events_path.open("a", encoding="utf-8") as handle:
            for pair in evidence.pairs:
                event = EvidenceEvent(
                    schema_version=2,
                    event_id=f"{contract.skill_id}.v{contract.version}.{evidence.split}.{pair.pair_id}",
                    context={"task_id": pair.task_id, "seed": pair.seed, "task_family": contract.task_family, "skill_digest": contract.digest},
                    assignment={"interventions": {contract.skill_id: 1}, "propensity": 0.5,
                                "design_id": f"matched-{evidence.split}-skill-v1"},
                    outcome_vector={"utility": (pair.on_utility - pair.off_utility) / 2.0},
                    scientific_gates={"independent_verifier": True, "hard_gates": evidence.hard_gates_passed},
                    artifacts={"off_trial_id": pair.off_trial_id, "on_trial_id": pair.on_trial_id,
                               "off_receipt_sha256": pair.off_receipt_sha256,
                               "on_receipt_sha256": pair.on_receipt_sha256,
                               "off_utility": pair.off_utility, "on_utility": pair.on_utility,
                               "off_success": pair.off_success, "on_success": pair.on_success,
                               "raw_paired_delta": pair.on_utility - pair.off_utility,
                               "utility_scale": "raw_paired_delta_divided_by_2"},
                    versions={"skill": contract.digest, "agent": pair.agent_revision, "benchmark": pair.benchmark_revision},
                    source_id=pair.on_trial_id,
                    independence_group=pair.pair_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    evidence_stream="representative",
                    query_id=pair.pair_id,
                    trust_zone="local",
                    attacker_controlled_fields=[],
                    evidence_role="synthesis" if evidence.split in {"train", "train_inner"} else "validation",
                )
                handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(self.events_path.read_bytes()).hexdigest()
        self.journal.append(
            "add_evidence", contract.skill_id, version=contract.version,
            artifact_path=str(self.events_path.relative_to(self.root)).replace("\\", "/"),
            digest=digest, old_digest=old_digest or None,
            operation_detail=f"{evidence.split}_paired_verification",
        )

    def record_rejection(self, contract: SkillContract, reason: str, *, evidence_ids: Sequence[str] = ()) -> None:
        self._ensure_genesis()
        state = self.state_store.load("rule", contract.skill_id, contract.version)
        if state.status != "retired":
            decision = EvolutionDecision("rule", contract.skill_id, "RETIRE", "rejected", "none", reason, tuple(evidence_ids))
            state = apply_lifecycle_decision(decision, state, self.journal, self.state_store)
        self._append_decision(contract, "rejected", reason, evidence_ids)

    def record_acceptance(self, contract: SkillContract, evidence: PairedEvidence) -> None:
        if evidence.skill_digest != contract.digest:
            raise ValueError("evidence refers to a different skill contract")
        if evidence.assessment_status != "CANDIDATE_ACCEPTED":
            raise ValueError("only candidate-accepted evidence can record acceptance")
        self.record_evidence(contract, evidence)
        self._append_decision(contract, "verified_candidate", "bounded_interval_above_zero", evidence.evidence_trial_ids)

    def record_outer_assessment(self, candidate: SkillContract, parent: SkillContract,
                                evidence: PairedEvidence) -> bool:
        """Persist independent validation; inner acceptance cannot hide outer failure."""
        if evidence.split not in {"train_outer", "calibration", "sealed"}:
            raise ValueError("outer assessment requires independent validation evidence")
        if candidate.parent_digest != parent.digest:
            raise ValueError("outer assessment parent differs from candidate lineage")
        if evidence.assessment_status == "CANDIDATE_ACCEPTED":
            self.record_acceptance(candidate, evidence)
            return True
        self.record_evidence(candidate, evidence)
        if evidence.assessment_status == "REJECTED":
            self.rollback(candidate, parent, "independent_outer_validation_failed",
                          evidence_ids=evidence.evidence_trial_ids)
        else:
            self._append_decision(candidate, "outer_undecided", "independent_outer_validation_insufficient",
                                  evidence.evidence_trial_ids)
        return False

    def _append_decision(self, contract: SkillContract, outcome: str, reason: str, evidence_ids: Sequence[str]) -> None:
        old_digest = hashlib.sha256(self.events_path.read_bytes()).hexdigest() if self.events_path.exists() else ""
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "skill_lifecycle_decision", "skill_id": contract.skill_id,
                "version": contract.version, "skill_digest": contract.digest,
                "outcome": outcome, "reason": reason, "evidence_trial_ids": list(evidence_ids),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, sort_keys=True, ensure_ascii=False) + "\n")
        self.journal.append(
            "add_evidence", contract.skill_id, version=contract.version,
            artifact_path=str(self.events_path.relative_to(self.root)).replace("\\", "/"),
            digest=hashlib.sha256(self.events_path.read_bytes()).hexdigest(), old_digest=old_digest or None,
            operation_detail=outcome,
        )

    def rollback(self, candidate: SkillContract, parent: SkillContract | None, reason: str, *, evidence_ids: Sequence[str] = ()) -> SkillContract | None:
        if parent is not None and candidate.parent_digest != parent.digest:
            raise ValueError("rollback parent does not match candidate lineage")
        self.record_rejection(candidate, reason, evidence_ids=evidence_ids)
        return parent


@dataclass(frozen=True)
class EvolutionResult:
    accepted: bool
    current: SkillContract
    candidate: SkillContract | None
    rejection_reasons: tuple[str, ...]
    evidence: PairedEvidence | None
    assessment_status: str


class SkillEvolution:
    def __init__(self, optimizer: SkillPatchOptimizer, store: SkillEvolutionStore) -> None:
        self.optimizer = optimizer
        self.store = store

    def run_round(
        self,
        parent: SkillContract,
        patches: Sequence[Patch],
        receipts: Sequence[PairReceipt],
        *,
        iteration: int = 1,
        split: str = "train",
        source_trial_ids: Sequence[str] = (),
        source_task_ids: Sequence[str] = (),
        source_template_ids: Sequence[str] = (),
        source_cluster_ids: Sequence[str] = (),
        source_trace_hashes: Sequence[str] = (),
        patch_feedback_receipt_hashes: Sequence[str] = (),
        train_split_manifest_sha256: str | None = None,
        generator_provider: str | None = None,
        model_identity: str | None = None,
        proposal_prompt_sha256: str | None = None,
        proposal_output_sha256: str | None = None,
        runtime_agent_sha: str | None = None,
    ) -> EvolutionResult:
        if split not in {"train", "train_inner"}:
            raise ValueError("candidate refinement accepts train or train_inner evidence only")
        candidate, errors = self.optimizer.apply(parent, patches, iteration=iteration)
        if candidate is None:
            self.store._append_decision(parent, "rejected_patch", ";".join(errors), ())
            return EvolutionResult(False, parent, None, errors, None, "REJECTED")
        if any((source_trial_ids, source_task_ids, source_template_ids, source_cluster_ids, source_trace_hashes,
                train_split_manifest_sha256, generator_provider, model_identity,
                proposal_prompt_sha256, proposal_output_sha256, runtime_agent_sha)):
            candidate = replace(
                candidate,
                source_split=parent.source_split,
                source_trial_ids=tuple(sorted(set(parent.source_trial_ids) | set(source_trial_ids))),
                source_task_ids=tuple(sorted(set(parent.source_task_ids) | set(source_task_ids))),
                source_template_ids=tuple(sorted(set(parent.source_template_ids) | set(source_template_ids))),
                source_cluster_ids=tuple(sorted(set(parent.source_cluster_ids) | set(source_cluster_ids))),
                source_trace_hashes=tuple(sorted(set(parent.source_trace_hashes) | set(source_trace_hashes))),
                train_split_manifest_sha256=train_split_manifest_sha256 or parent.train_split_manifest_sha256,
                generator_provider=generator_provider or parent.generator_provider,
                model_identity=model_identity or parent.model_identity,
                proposal_prompt_sha256=proposal_prompt_sha256 or parent.proposal_prompt_sha256,
                proposal_output_sha256=proposal_output_sha256 or parent.proposal_output_sha256,
                runtime_agent_sha=runtime_agent_sha or parent.runtime_agent_sha,
                patch_feedback_receipt_hashes=tuple(sorted(set(parent.patch_feedback_receipt_hashes) | set(patch_feedback_receipt_hashes))),
            )
        elif patch_feedback_receipt_hashes:
            candidate = replace(candidate, patch_feedback_receipt_hashes=tuple(sorted(
                set(parent.patch_feedback_receipt_hashes) | set(patch_feedback_receipt_hashes))))
        try:
            evidence = verify_inner_pairs(candidate, receipts) if split == "train_inner" else verify_training_pairs(candidate, receipts)
        except ValueError as exc:
            self.store.save_candidate(candidate)
            self.store.rollback(candidate, parent, str(exc))
            return EvolutionResult(False, parent, candidate, (str(exc),), None, "REJECTED")
        candidate = replace(candidate, verification_trial_ids=evidence.evidence_trial_ids)
        self.store.save_candidate(candidate)
        assessment = self.optimizer.assess(evidence)
        if assessment == "REJECTED":
            self.store.record_evidence(candidate, evidence)
            self.store.rollback(candidate, parent, "paired_non_regression_gate_failed", evidence_ids=evidence.evidence_trial_ids)
            return EvolutionResult(False, parent, candidate, ("paired_assessment_rejected",), evidence, assessment)
        if assessment == "UNDECIDED":
            self.store.record_evidence(candidate, evidence)
            self.store._append_decision(candidate, "undecided", "bounded_interval_overlaps_zero", evidence.evidence_trial_ids)
            return EvolutionResult(False, parent, candidate, (), evidence, assessment)
        self.store.record_acceptance(candidate, evidence)
        return EvolutionResult(True, candidate, candidate, (), evidence, assessment)
