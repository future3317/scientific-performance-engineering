"""Development candidate synthesis and independent calibration through ACRE."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from core.acre import AcreEngine
from core.acre.typed_gate import decision_from_synthesis
from core.acre.cegis import SynthesisResult, synthesize_applicability
from core.acre.budget import StatisticalBudget
from benchmark.families import family_predicate_grammar
from core.acre.regimes import (
    SupportGroup,
    SupportRegime,
    SupportRegimeDecision,
    SupportScope,
    classify_support_regime,
)
from core.acre.cegis import synthesis_certificate_valid
from core.governance import ValidationCertificate, ValidationKind, apply_promotion
from core.models import EvidenceEvent, RevisionRef, RuleSpec, RuleState, TaskContext
from core.predicates import match_predicate
from core.sequential_stats import bounded_mean_interval, mixture_lower_bound
from core.utility import utility_effect


@dataclass(frozen=True)
class DevelopmentCandidate:
    status: str
    spec: RuleSpec
    candidate_spec: RuleSpec | None
    state: RuleState
    synthesis: SynthesisResult
    events: tuple[EvidenceEvent, ...]
    task_context: TaskContext
    development_families: frozenset[str]
    development_pair_ids: frozenset[str]
    excluded_pairs: tuple[tuple[str, str], ...]
    applicability_family_id: str


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    route: Any
    route_engine: AcreEngine
    regime: SupportRegimeDecision
    events: tuple[EvidenceEvent, ...]
    assessment: Any
    task_context: TaskContext
    calibration_families: frozenset[str]
    validation_certificates: tuple[ValidationCertificate, ...]
    governance_errors: tuple[tuple[str, str], ...]
    excluded_pairs: tuple[tuple[str, str], ...]
    reason: str


_OUTCOME_FEATURE_PARTS = {
    "correctness", "effect", "elapsed", "latency", "log_benefit", "outcome",
    "receipt", "runtime", "seconds", "speedup", "status", "utility",
}
_LAYOUT_FEATURES = {
    "batch", "horizon", "conv2d_module_count", "input_channels",
    "input_height", "input_width",
}


def _has_canonical_layout_grammar(family_id: str) -> bool:
    """Require the registered, pre-execution channels-last feature lattice."""
    if family_id != "channels_last":
        return False
    grammar = family_predicate_grammar(family_id)
    features = {
        str(feature.get("path", "")).rsplit(".", 1)[-1]
        for feature in grammar.get("features", ())
        if isinstance(feature, Mapping) and feature.get("type") == "numeric"
    }
    return _LAYOUT_FEATURES <= features


def _spec_state(engine: AcreEngine, rule_id: str) -> tuple[RuleSpec, RuleState]:
    specs = [spec for spec in engine.rule_specs if spec.rule_id == rule_id]
    state = engine.rule_states.get(rule_id)
    if len(specs) != 1 or state is None:
        raise ValueError(f"engine must contain one materialized rule and state for {rule_id}")
    if specs[0].version != state.version or state.rule_id != rule_id:
        raise ValueError("rule spec and state identity/version do not match")
    return specs[0], RuleState.from_dict(state.to_dict())


def _validate_feature_names(features: Mapping[str, Any], allowed: Sequence[str]) -> None:
    names = set(features)
    if any(not isinstance(name, str) for name in names):
        raise ValueError("pre-execution feature names must be strings")
    if names - set(allowed):
        raise ValueError("features contain undeclared or non-pre-execution fields")
    for name in names:
        parts = set("".join(character.lower() if character.isalnum() else " " for character in name).split())
        if parts & _OUTCOME_FEATURE_PARTS:
            raise ValueError(f"post-execution feature is forbidden: {name}")


def _context_features(pair: Mapping[str, Any], allowed: Sequence[str]) -> dict[str, Any]:
    features = pair.get("features")
    if not isinstance(features, Mapping):
        context = pair.get("context")
        if isinstance(context, Mapping):
            workload = context.get("workload")
            if isinstance(workload, Mapping):
                features = workload
            else:
                features = {name: context[name] for name in allowed if name in context}
                qualification = context.get("qualification")
                summary = qualification.get("model_summary") if isinstance(qualification, Mapping) else None
                summary = summary if isinstance(summary, Mapping) else {}
                for name in ("conv2d_module_count",):
                    if name not in features and name in summary:
                        features[name] = summary[name]
                shapes = (context.get("actual_input_shapes") or summary.get("input_shapes") or {})
                if isinstance(shapes, Mapping):
                    dims = next((value for value in shapes.values()
                                 if isinstance(value, list) and len(value) == 4), None)
                    if dims:
                        features.setdefault("input_channels", dims[1])
                        features.setdefault("input_height", dims[-2])
                        features.setdefault("input_width", dims[-1])
    if not isinstance(features, Mapping):
        raise ValueError("paired receipt needs its frozen pre-execution workload features")
    result = dict(features)
    _validate_feature_names(result, allowed)
    return result


def _pair_id(pair: Mapping[str, Any]) -> str:
    value = str(pair.get("pair_id", ""))
    if not value:
        raise ValueError("paired receipt needs pair_id")
    return value


def _context_key(features: Mapping[str, Any]) -> str:
    return json.dumps(features, sort_keys=True, separators=(",", ":"), default=str)


def _stratum_key(features: Mapping[str, Any]) -> str:
    if "batch" not in features or "horizon" not in features:
        raise ValueError("paired context needs frozen batch and horizon stratum fields")
    return _context_key({"batch": features.get("batch"), "horizon": features.get("horizon")})


def _stratum_region_key(features: Mapping[str, Any], inside: bool) -> str:
    return _context_key({
        "batch": features.get("batch"),
        "horizon": features.get("horizon"),
        "inside_candidate": inside,
    })


def _representative_features(
    family_features: Mapping[str, Mapping[str, Any]], allowed: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in allowed:
        values = [features[name] for features in family_features.values() if name in features]
        if not values:
            continue
        if all(not isinstance(value, bool) and isinstance(value, Real) for value in values):
            result[name] = median(values)
        elif all(value == values[0] for value in values):
            result[name] = values[0]
    return result


def _context_digest(context_key: str) -> str:
    return hashlib.sha256(context_key.encode("utf-8")).hexdigest()[:12]


def _equivalent_baseline_for_effect(effect: float) -> float:
    finite_limit = math.nextafter(1.0, 0.0)
    effect = min(finite_limit, max(-finite_limit, effect))
    return math.exp(math.log(2.0) * math.atanh(effect))


def _split(pair: Mapping[str, Any], expected: str) -> None:
    if str(pair.get("split", "")).lower() != expected:
        raise ValueError(f"{expected} policy evidence must contain only {expected} pairs")


def _status(pair: Mapping[str, Any]) -> str:
    return str(pair.get("effect_status", pair.get("status", "unknown"))).upper()


def _cold_seconds(pair: Mapping[str, Any], arm: str) -> float | None:
    receipt = pair.get(arm)
    timing = receipt.get("timing") if isinstance(receipt, Mapping) else None
    value = timing.get("cold_seconds") if isinstance(timing, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
        return None
    return float(value)


def _gates_pass(pair: Mapping[str, Any]) -> bool:
    classification = pair.get("classification")
    if isinstance(classification, Mapping):
        if classification.get("scientific_correctness") != "PASS":
            return False
    else:
        gates = pair.get("scientific_gates")
        if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
            return False
    for arm in ("S0", "S1"):
        receipt = pair.get(arm)
        if isinstance(receipt, Mapping) and receipt.get("scientific_correctness", "PASS") != "PASS":
            return False
    return True


def _known_harm(pair: Mapping[str, Any]) -> bool:
    gates = pair.get("scientific_gates")
    if isinstance(gates, Mapping) and gates.get("candidate_correctness") is False:
        return True
    candidate = pair.get("S1")
    return (isinstance(candidate, Mapping)
            and candidate.get("scientific_correctness") == "FAIL")


def _paired_receipts_match(
    pair: Mapping[str, Any], expected_gpu_uuid: str | None,
    runtime_versions: Mapping[str, str],
) -> bool:
    classification = pair.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(key) != expected
        for key, expected in (
            ("arm_execution", "COMPLETE"),
            ("scientific_correctness", "PASS"),
            ("paired_comparability", "MATCHED"),
            ("policy_evidence", "ELIGIBLE"),
        )
    ):
        return False
    if not expected_gpu_uuid:
        return False
    identity_fields = (
        "runtime_gpu_uuid", "inputs_sha256", "weights_sha256", "source_commit",
        "config_digest", "context_digest",
    )
    arms = [pair.get("S0"), pair.get("S1")]
    if not all(isinstance(arm, Mapping) for arm in arms):
        return False
    first, second = arms
    def stable_runtime_identity(arm: Mapping[str, Any]) -> dict[str, Any] | None:
        runtime = arm.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            return None
        driver = runtime.get("driver")
        if not isinstance(driver, Mapping):
            return None
        return {
            "device_uuid": runtime.get("device_uuid"),
            "device_name": runtime.get("device_name"),
            "torch_version": runtime.get("torch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
            "driver_version": driver.get("driver_version"),
            "packages": dict(runtime.get("packages") or {}),
        }
    first_runtime = stable_runtime_identity(first)
    second_runtime = stable_runtime_identity(second)
    if first_runtime is None or first_runtime != second_runtime:
        return False
    for arm in arms:
        receipt_digest = arm.get("raw_receipt_sha256")
        if not isinstance(receipt_digest, str) or len(receipt_digest) != 64 or any(
            character not in "0123456789abcdef" for character in receipt_digest.lower()
        ):
            return False
        if arm.get("runtime_gpu_uuid") != expected_gpu_uuid:
            return False
        runtime = arm.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            return False
        driver = runtime.get("driver")
        packages = runtime.get("packages")
        observed = {
            "gpu_uuid": runtime.get("device_uuid"),
            "torch": runtime.get("torch_version"),
            "torch_version": runtime.get("torch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
            "driver_version": driver.get("driver_version") if isinstance(driver, Mapping) else None,
            "source_commit": arm.get("source_commit"),
            **({str(key): value for key, value in packages.items()} if isinstance(packages, Mapping) else {}),
        }
        for key, value in runtime_versions.items():
            if key == "skill_digest":
                continue
            if key not in observed or observed[key] is None or str(observed[key]) != str(value):
                return False
    return all(first.get(key) and first.get(key) == second.get(key) for key in identity_fields)


def _pair_utility(pair: Mapping[str, Any]) -> float | None:
    control, treatment = _cold_seconds(pair, "S0"), _cold_seconds(pair, "S1")
    if control is None or treatment is None:
        return None
    return utility_effect(treatment, control, higher_is_better=False,
                          log_scale=math.log(2.0))


def _event_pair(
    pair: Mapping[str, Any], *, rule_id: str, rule_version: int,
    context: Mapping[str, Any], utility: float | None,
    versions: Mapping[str, str], source_id: str, timestamp: str,
    evidence_role: str,
) -> tuple[EvidenceEvent, EvidenceEvent]:
    adversarial = evidence_role == "adversarial"
    gates = dict(pair.get("scientific_gates", {}))
    if not gates:
        gates = {"correctness": _gates_pass(pair)}
    artifacts: dict[str, Any] = {
        "pair_status": _status(pair),
        "paired_receipt_refs": {
            arm: {
                "pair_id": _pair_id(pair),
                "receipt_sha256": ((pair.get(arm) or {}).get("raw_receipt_sha256")
                                   if isinstance(pair.get(arm), Mapping) else None),
            }
            for arm in ("S0", "S1")
        },
    }
    contrast = {
        "seconds_control": _cold_seconds(pair, "S0"),
        "seconds_treatment": _cold_seconds(pair, "S1"),
    }
    if utility is not None:
        contrast["utility_effect"] = utility
    artifacts["paired_contrast"] = contrast
    if adversarial:
        artifacts["counterexample"] = True
    common = {
        "schema_version": 2,
        "outcome_vector": ({"utility": utility} if utility is not None else {}),
        "scientific_gates": gates,
        "artifacts": artifacts,
        "versions": {str(key): str(value) for key, value in versions.items()},
        "source_id": source_id,
        "independence_group": str(pair["family_id"]),
        "timestamp": timestamp,
        "evidence_stream": "adversarial" if adversarial else "representative",
        "query_id": _pair_id(pair),
        "trust_zone": "local",
        "attacker_controlled_fields": [],
        "evidence_role": evidence_role,
    }
    return tuple(
        EvidenceEvent(
            **common,
            event_id=f"{_pair_id(pair)}-{arm.lower()}",
            assignment={
                "propensity": 0.5,
                "design_id": "v4-fixed-paired",
                "interventions": {rule_id: value},
            },
            context={**context, "rule_versions": {rule_id: rule_version}},
        )
        for arm, value in (("S1", 1), ("S0", 0))
    )  # type: ignore[return-value]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")).hexdigest()


def _promote_from_calibration(
    *,
    engine: AcreEngine,
    candidate: DevelopmentCandidate,
    groups: Sequence[SupportGroup],
    family_rows: Mapping[str, Sequence[tuple[Mapping[str, Any], float | None, bool, bool, bool]]],
    family_role: Mapping[str, bool],
    context_strata_evidence: Sequence[Mapping[str, Any]],
    source_id: str,
    minimum_support_families: int,
    minimum_non_support_families: int,
) -> tuple[AcreEngine | None, str]:
    """Apply core governance only when calibration contains disjoint family pools."""
    if engine.state_store is None or candidate.candidate_spec is None:
        return None, "canonical_store_or_candidate_spec_missing"

    by_id = {group.group_id: group for group in groups}
    supported = sorted([
        by_id[family] for family, inside in family_role.items()
        if inside is True and family in by_id
    ], key=lambda group: group.group_id)
    unsupported = sorted([
        by_id[family] for family, inside in family_role.items()
        if inside is False and family in by_id
    ], key=lambda group: group.group_id)
    # Reserve one positive replication and one independent boundary holdout.
    if len(supported) < minimum_support_families or len(unsupported) < minimum_non_support_families:
        return None, "calibration_lacks_registered_support_or_non_support_families"
    promotion_groups = supported[:-1]
    heldout_group = supported[-1]
    if len(promotion_groups) < 2:
        return None, "calibration_lacks_two_independent_promotion_families"
    if any(
        not group.hard_gates_passed or group.effect_lower <= 0.0
        for group in promotion_groups
    ):
        return None, "promotion_representative_family_failed_calibration_effect_or_correctness_gate"
    if any(
        not group.hard_gates_passed or group.effect_upper > 0.0
        for group in unsupported
    ):
        return None, "non_support_family_did_not_clear_calibration_boundary_gate"
    if not heldout_group.hard_gates_passed or heldout_group.effect_lower <= 0.0:
        return None, "reserved_holdout_family_failed_calibration_effect_or_correctness_gate"
    heldout_decision = decision_from_synthesis(candidate.synthesis, heldout_group.context)
    if heldout_decision.reason_code != "identified_match":
        return None, "heldout_family_did_not_clear_canonical_synthesis_gate"
    poison_groups = unsupported
    poison_decisions = [
        decision_from_synthesis(candidate.synthesis, group.context)
        for group in poison_groups
    ]
    if any(not group.hard_gates_passed for group in poison_groups):
        return None, "reserved_boundary_family_failed_scientific_correctness_gate"
    if any(decision.reason_code != "outside_predicate" for decision in poison_decisions):
        return None, "boundary_family_was_not_rejected_by_canonical_synthesis_gate"

    def receipt_refs(family: str) -> list[dict[str, Any]]:
        return [
            {
                "pair_id": _pair_id(pair),
                "S0_receipt_sha256": pair["S0"].get("raw_receipt_sha256"),
                "S1_receipt_sha256": pair["S1"].get("raw_receipt_sha256"),
            }
            for pair, _, _, valid, _ in family_rows[family]
            if valid
        ]

    from core.acre.experiments import ReplaySequentialCertificate

    replay_gate = ReplaySequentialCertificate(minimum_groups=len(promotion_groups))
    probability_lcb = mixture_lower_bound(
        len(promotion_groups), len(promotion_groups),
        replay_gate.statistical_budget.replay_minimum_delta,
    )
    if probability_lcb < replay_gate.p_min:
        return None, "calibration_families_do_not_clear_core_promotion_probability_gate"
    utility_lcb = bounded_mean_interval(
        [group.effect_lower for group in promotion_groups],
        replay_gate.statistical_budget.replay_minimum_delta,
    )[0]
    utility_ucb = bounded_mean_interval(
        [group.effect_upper for group in promotion_groups],
        replay_gate.statistical_budget.replay_minimum_delta,
    )[1]
    if utility_lcb <= 0.0:
        return None, "calibration_families_do_not_clear_positive_utility_bound"

    synthesis_events = [
        event for event in candidate.events if event.evidence_role == "synthesis"
    ]
    synthesis_by_family: dict[str, str] = {}
    for event in synthesis_events:
        synthesis_by_family.setdefault(event.independence_group, event.query_id)
    if not synthesis_by_family:
        return None, "development_has_no_canonical_synthesis_events"

    promotion_cases = [
        {
            "case_id": f"promotion:{group.group_id}",
            "independence_group": group.group_id,
            "receipt_refs": receipt_refs(group.group_id),
            "effect_interval": {"lcb": group.effect_lower, "ucb": group.effect_upper},
        }
        for group in promotion_groups
    ]
    heldout = {
        "case_id": f"heldout:{heldout_group.group_id}",
        "holdout_class": "replication",
        "executed": True,
        "execution_source": source_id,
        "scientific_ok": True,
        "effect_lcb": heldout_group.effect_lower,
        "gate_decision": "reuse",
        "receipt_refs": receipt_refs(heldout_group.group_id),
        "effect_interval": {"lcb": heldout_group.effect_lower, "ucb": heldout_group.effect_upper},
    }
    poison_cases = [
        {
            "case_id": f"boundary:{group.group_id}",
            "holdout_class": "adversarial",
            "executed": True,
            "execution_source": source_id,
            "accepted": False,
            "gate_decision": decision.action,
            "gate_reason": decision.reason_code,
            "receipt_refs": receipt_refs(group.group_id),
            "effect_interval": {"lcb": group.effect_lower, "ucb": group.effect_upper},
        }
        for group, decision in zip(poison_groups, poison_decisions)
    ]
    validation = {
        "promotion_case_ids": [case["case_id"] for case in promotion_cases],
        "synthesis_case_ids": list(synthesis_by_family.values()),
        "synthesis_independence_groups": list(synthesis_by_family),
        "promotion_independence_groups": [group.group_id for group in promotion_groups],
        "synthesis_cases": [
            {"case_id": case_id, "independence_group": family}
            for family, case_id in synthesis_by_family.items()
        ],
        "promotion_cases": promotion_cases,
        "heldout_regression_cases": [heldout],
        "poison_probe_cases": poison_cases,
        "context_strata_validation": list(context_strata_evidence),
        "regression_tolerance": 0.0,
    }
    root = Path(engine.state_store.root)
    validation_path = Path("evolution") / "validation" / f"{candidate.spec.rule_id}-v{candidate.candidate_spec.version}.json"
    validation_file = root / validation_path
    validation_file.parent.mkdir(parents=True, exist_ok=True)
    validation_file.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replay_result = {
        "p_min": replay_gate.p_min,
        "promotion_probability_lower_bound": probability_lcb,
        "utility_effect_lcb": utility_lcb,
        "utility_effect_ucb": utility_ucb,
        "mean_effect": sum(group.effect_lower for group in promotion_groups) / len(promotion_groups),
    }
    replay_manifest = {
        "outcome": "passed",
        "execution_source": source_id,
        "result": replay_result,
        "promotion_record": {
            "representative_groups": [group.group_id for group in promotion_groups],
            "promotion_case_ids": [case["case_id"] for case in promotion_cases],
            "heldout_regression_digest": _digest([heldout]),
            "poison_gate": {"passed": True, "case_id": poison_cases[-1]["case_id"]},
            "promotion_probability_lcb": probability_lcb,
            "utility_effect_cs": {"lcb": utility_lcb, "ucb": utility_ucb},
            "replay_manifest_digest": _digest({"cases": promotion_cases, "result": replay_result}),
            "validation_artifact_path": validation_path.as_posix(),
            "validation_artifact_digest": _digest(validation),
        },
    }
    decision = apply_promotion(
        root, candidate.candidate_spec.to_dict(), replay_manifest,
        replay_path=f"{source_id}/calibration-replay.json",
    )
    if not decision.allowed:
        return None, decision.reason
    return AcreEngine.from_store(root), "canonical_promotion_applied_and_reloaded"


def build_development_candidate(
    *,
    engine: AcreEngine,
    rule_id: str,
    development_pairs: Sequence[Mapping[str, Any]],
    candidate_features: Mapping[str, Any],
    pre_execution_features: Sequence[str],
    family_id: str,
    support_scope: SupportScope,
    runtime_versions: Mapping[str, str],
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
    source_id: str,
    timestamp: str,
) -> DevelopmentCandidate:
    """Use development receipts only to propose a rule; never certify/deploy it."""
    spec, _ = _spec_state(engine, rule_id)
    allowed = tuple(pre_execution_features)
    _validate_feature_names(candidate_features, allowed)
    if not family_id:
        raise ValueError("a registered ACRE family_id is required for synthesis")
    context = {
        "domain": spec.domain,
        "workload": dict(candidate_features),
        "hardware": dict(hardware),
        "software": dict(software),
        "evidence": {"source": "development", "rule_versions": {rule_id: spec.version}},
    }
    task_context = TaskContext(**context)
    contract_digest = spec.intervention.get("contract_digest")
    if not isinstance(contract_digest, str) or not contract_digest:
        raise ValueError("the frozen intervention needs a contract_digest for certificate scope")
    scope_versions = {str(key): str(value) for key, value in runtime_versions.items()}
    required_scope = {
        "gpu_uuid", "source_commit", "driver_version", "torch_cuda_version",
    }
    if not required_scope.issubset(scope_versions) or not (
        "torch" in scope_versions or "torch_version" in scope_versions
    ):
        raise ValueError("certificate scope must bind GPU UUID, source commit, and PyTorch version")
    scope_versions["skill_digest"] = contract_digest
    if scope_versions != dict(support_scope.versions):
        raise ValueError("development synthesis scope must bind the frozen runtime and intervention")

    grouped: dict[str, dict[str, Any]] = {}
    events: list[EvidenceEvent] = []
    excluded: list[tuple[str, str]] = []
    pair_ids: set[str] = set()
    families: set[str] = set()
    for pair in development_pairs:
        _split(pair, "development")
        pair_id = _pair_id(pair)
        if pair_id in pair_ids:
            raise ValueError(f"duplicate development pair_id: {pair_id}")
        pair_ids.add(pair_id)
        family = str(pair.get("family_id", ""))
        if not family:
            raise ValueError("development pair needs independent family_id")
        features = _context_features(pair, allowed)
        pair_context = {**context, "workload": features}
        expected_gpu = hardware.get("gpu_uuid", hardware.get("selected_gpu_uuid"))
        valid = (
            _status(pair) in {"VALID", "COMPLETE"}
            and _gates_pass(pair)
            and _paired_receipts_match(
                pair, str(expected_gpu) if expected_gpu else None, runtime_versions,
            )
        )
        utility = _pair_utility(pair)
        if not valid or utility is None:
            excluded.append((pair_id, _status(pair)))
            if _known_harm(pair):
                events.extend(_event_pair(
                    pair, rule_id=rule_id, rule_version=spec.version,
                    context=pair_context, utility=utility, versions=scope_versions,
                    source_id=source_id, timestamp=timestamp, evidence_role="adversarial",
                ))
            continue
        families.add(family)
        context_key = _stratum_key(features)
        cell = grouped.setdefault(context_key, {
            "family_features": {},
            "families": {},
        })
        cell["family_features"].setdefault(family, features)
        cell["families"].setdefault(family, []).append(pair)
        events.extend(_event_pair(
            pair, rule_id=rule_id, rule_version=spec.version,
            context=pair_context, utility=utility, versions=scope_versions,
            source_id=source_id, timestamp=timestamp, evidence_role="synthesis",
        ))

    synthesis_budget = StatisticalBudget()
    cases = []
    development_grid = set(grouped)
    development_grid_complete = bool(development_grid) and all(
        {context_key for context_key, cell in grouped.items()
         if family in cell["families"]} == development_grid
        for family in families
    )
    for context_key, cell in sorted(grouped.items()):
        family_pairs = cell["families"]
        family_effects = [
            sum(_pair_utility(pair) for pair in rows) / len(rows)
            for _, rows in sorted(family_pairs.items())
        ]
        cases.append({
            "case_id": f"development-context:{_context_digest(context_key)}",
            "context": {"domain": spec.domain, "workload": _representative_features(
                cell["family_features"], allowed,
            )},
            # Encode each family-cluster mean utility as an equivalent positive
            # pair because the core CEGIS adapter accepts paired measurements.
            # This preserves mean log utility instead of taking a log after
            # pooling sessions or treating them as independent families.
            "intervention_measurements": [1.0 for _ in family_effects],
            "baseline_measurements": [
                _equivalent_baseline_for_effect(effect) for effect in family_effects
            ],
            "cluster_families": sorted(family_pairs),
            "higher_is_better": False,
            "utility_scale": math.log(2.0),
            "scientific_ok": True,
            "quality_ok": True,
        })
    has_layout_grammar = _has_canonical_layout_grammar(family_id)
    synthesis = (
        synthesize_applicability(
            cases,
            family_id=family_id,
            require_identified=True,
            certificate_scope=support_scope.to_dict(),
            statistical_budget=synthesis_budget,
        )
        if has_layout_grammar and development_grid_complete else
        SynthesisResult(
            "unavailable" if not has_layout_grammar else "insufficient_evidence", None, (), (),
            provenance={
                "reason": (
                    "no_canonical_layout_applicability_grammar"
                    if not has_layout_grammar else
                    "development_context_grid_incomplete_across_families"
                ),
                "family_id": family_id,
            },
        )
    )
    candidate_spec = None
    status = "CANDIDATE_UNAVAILABLE" if has_layout_grammar else "ACRE_NOT_FITTED"
    if (
        synthesis.status == "identified"
        and synthesis.predicate is not None
        and synthesis_certificate_valid(synthesis)
    ):
        parent_digest = hashlib.sha256(json.dumps(
            spec.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        candidate_spec = replace(
            spec,
            version=spec.version + 1,
            parent=RevisionRef(spec.rule_id, spec.version, parent_digest),
            applicability={"all": [spec.applicability, dict(synthesis.predicate)]},
        )
        status = "CANDIDATE_ONLY"
    return DevelopmentCandidate(
        status, spec, candidate_spec,
        RuleState(rule_id, (candidate_spec.version if candidate_spec else spec.version), status="candidate"),
        synthesis, tuple(events), task_context, frozenset(families),
        frozenset(pair_ids), tuple(excluded), family_id,
    )


def validate_calibration_and_route(
    *,
    engine: AcreEngine,
    candidate: DevelopmentCandidate,
    calibration_pairs: Sequence[Mapping[str, Any]],
    candidate_features: Mapping[str, Any],
    pre_execution_features: Sequence[str],
    support_scope: SupportScope,
    runtime_versions: Mapping[str, str],
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
    source_id: str,
    timestamp: str,
    min_calibration_families: int,
    planned_calibration_pair_ids: Sequence[str],
    minimum_support_families: int | None = None,
    minimum_non_support_families: int | None = None,
) -> CalibrationResult:
    """Validate independent calibration receipts and route only governed rules."""
    if min_calibration_families < 1:
        raise ValueError("min_calibration_families must be positive")
    minimum_support_families = minimum_support_families or min_calibration_families
    minimum_non_support_families = minimum_non_support_families or min_calibration_families
    if minimum_support_families < 1 or minimum_non_support_families < 1:
        raise ValueError("support and non-support family minima must be positive")
    _validate_feature_names(candidate_features, pre_execution_features)
    rule_version = (
        candidate.candidate_spec.version
        if candidate.candidate_spec is not None else candidate.spec.version
    )
    context = {
        "domain": candidate.spec.domain,
        "workload": dict(candidate_features),
        "hardware": dict(hardware),
        "software": dict(software),
        "evidence": {
            "source": "calibration",
            "rule_versions": {candidate.spec.rule_id: rule_version},
        },
    }
    task_context = TaskContext(**context)
    has_layout_grammar = _has_canonical_layout_grammar(candidate.applicability_family_id)
    versions = {str(key): str(value) for key, value in runtime_versions.items()}
    digest = candidate.spec.intervention.get("contract_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("candidate intervention has no frozen contract_digest")
    versions["skill_digest"] = digest
    if versions != dict(support_scope.versions):
        raise ValueError("calibration scope differs from the development candidate scope")
    pair_ids: set[str] = set()
    families: set[str] = set()
    grouped: dict[str, list[tuple[Mapping[str, Any], float | None, bool, bool, bool]]] = {}
    events: list[EvidenceEvent] = []
    excluded: list[tuple[str, str]] = []
    for pair in calibration_pairs:
        _split(pair, "calibration")
        pair_id = _pair_id(pair)
        if pair_id in pair_ids:
            raise ValueError(f"duplicate calibration pair_id: {pair_id}")
        if pair_id in candidate.development_pair_ids:
            raise ValueError(f"pair crosses development/calibration boundary: {pair_id}")
        pair_ids.add(pair_id)
        family = str(pair.get("family_id", ""))
        if not family:
            raise ValueError("calibration pair needs independent family_id")
        if family in candidate.development_families:
            raise ValueError(f"family crosses development/calibration boundary: {family}")
        families.add(family)
        features = _context_features(pair, pre_execution_features)
        pair_context = {**context, "workload": features, "rule_versions": {
            candidate.spec.rule_id: rule_version,
        }}
        gates_passed = _gates_pass(pair)
        utility = _pair_utility(pair)
        expected_gpu = hardware.get("gpu_uuid", hardware.get("selected_gpu_uuid"))
        is_valid = (
            _status(pair) in {"VALID", "COMPLETE"}
            and gates_passed
            and utility is not None
            and _paired_receipts_match(
                pair, str(expected_gpu) if expected_gpu else None, runtime_versions,
            )
        )
        role = "validation" if is_valid else "adversarial"
        known_harm = _known_harm(pair)
        if not is_valid and not known_harm:
            excluded.append((pair_id, _status(pair)))
        if is_valid or known_harm:
            events.extend(_event_pair(
                pair, rule_id=candidate.spec.rule_id, rule_version=rule_version,
                context=pair_context, utility=utility, versions=versions,
                source_id=source_id, timestamp=timestamp, evidence_role=role,
            ))
        grouped.setdefault(family, []).append((pair, utility, gates_passed, is_valid, known_harm))

    transition = engine.maintain(events=tuple(events))
    assessment = transition.assessment
    groups: list[SupportGroup] = []
    family_role: dict[str, bool] = {}
    validation_certificates: list[ValidationCertificate] = []
    governance_errors: list[tuple[str, str]] = []
    planned_ids = tuple(str(item) for item in planned_calibration_pair_ids)
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("planned calibration pair ids must be unique")
    calibration_complete = pair_ids == set(planned_ids)
    all_calibration_pairs_valid = calibration_complete and all(
        is_valid for rows in grouped.values() for _, _, _, is_valid, _ in rows
    )
    family_context_values: dict[str, dict[str, list[float]]] = {}
    family_context_rows: dict[str, dict[str, list[tuple[Mapping[str, Any], float | None, bool, bool, bool]]]] = {}
    family_grid: dict[str, set[str]] = {}
    cell_features: dict[str, dict[str, Any]] = {}
    cell_inside: dict[str, bool] = {}
    for family, rows in grouped.items():
        family_context_values[family] = {}
        family_context_rows[family] = {}
        family_grid[family] = set()
        for row in rows:
            pair, value, passed, is_valid, _ = row
            features = _context_features(pair, pre_execution_features)
            context_key = _stratum_key(features)
            family_grid[family].add(context_key)
            inside = (
                match_predicate(candidate.synthesis.predicate,
                                {**context, "workload": features})
                if candidate.synthesis.predicate is not None else None
            )
            if inside is None:
                continue
            cell_key = _stratum_region_key(features, inside)
            cell_features.setdefault(cell_key, features)
            cell_inside[cell_key] = inside
            family_context_rows[family].setdefault(cell_key, []).append(row)
            if value is not None and passed and is_valid:
                family_context_values[family].setdefault(cell_key, []).append(value)

    context_keys = sorted(cell_features)
    frozen_context_grid = set().union(*family_grid.values()) if family_grid else set()
    grid_complete = bool(frozen_context_grid) and all(
        family_grid[family] == frozen_context_grid for family in families
    )
    budget = StatisticalBudget()
    context_delta = budget.validation / max(1, len(context_keys))
    context_behavior_valid = grid_complete and candidate.synthesis.predicate is not None
    context_strata_evidence: list[dict[str, Any]] = []
    for key in context_keys:
        family_means = {
            family: sum(values) / len(values)
            for family in sorted(families)
            if (values := family_context_values[family].get(key))
        }
        if not family_means:
            context_behavior_valid = False
            continue
        lower, upper = bounded_mean_interval(list(family_means.values()), context_delta)
        refs = []
        for family in sorted(family_means):
            for pair, _, _, valid, _ in family_context_rows[family].get(key, ()):
                if valid:
                    refs.append({
                        "family_id": family,
                        "pair_id": _pair_id(pair),
                        "S0_receipt_sha256": pair["S0"].get("raw_receipt_sha256"),
                        "S1_receipt_sha256": pair["S1"].get("raw_receipt_sha256"),
                    })
        cell_features_row = cell_features[key]
        context_strata_evidence.append({
            "batch": cell_features_row["batch"],
            "horizon": cell_features_row["horizon"],
            "candidate_region": "support" if cell_inside[key] else "boundary",
            "alpha": context_delta,
            "independence_groups": sorted(family_means),
            "family_mean_effects": family_means,
            "effect_interval": {"lcb": lower, "ucb": upper},
            "paired_receipt_refs": refs,
        })
        if cell_inside[key]:
            context_behavior_valid &= lower > 0.0
        else:
            context_behavior_valid &= upper <= 0.0

    support_contexts = {key for key, inside in cell_inside.items() if inside}
    boundary_contexts = {key for key, inside in cell_inside.items() if not inside}
    support_families = {
        family for family in families
        if support_contexts.intersection(family_context_values[family])
    }
    boundary_families = {
        family for family in families
        if boundary_contexts.intersection(family_context_values[family])
    }

    # Families observed in several contexts stay one inferential unit. Assign
    # each family to exactly one canonical promotion role; its assigned role
    # uses only the corresponding predeclared support or boundary strata.
    support_only = sorted(support_families - boundary_families)
    boundary_only = sorted(boundary_families - support_families)
    mixed = sorted(support_families & boundary_families)
    support_ids, boundary_ids = set(support_only), set(boundary_only)
    next_mixed = 0
    while len(support_ids) < minimum_support_families and next_mixed < len(mixed):
        support_ids.add(mixed[next_mixed])
        next_mixed += 1
    while len(boundary_ids) < minimum_non_support_families and next_mixed < len(mixed):
        boundary_ids.add(mixed[next_mixed])
        next_mixed += 1
    support_ids.update(mixed[next_mixed:])

    promotion_rows: dict[str, list[tuple[Mapping[str, Any], float | None, bool, bool, bool]]] = {}
    for family, inside in [(item, True) for item in sorted(support_ids)] + [
        (item, False) for item in sorted(boundary_ids)
    ]:
        selected_contexts = support_contexts if inside else boundary_contexts
        rows = [
            row for key in sorted(selected_contexts)
            for row in family_context_rows[family].get(key, ())
        ]
        cell_means = [
            sum(family_context_values[family][key]) / len(family_context_values[family][key])
            for key in sorted(selected_contexts)
            if family_context_values[family].get(key)
        ]
        if not rows or not cell_means:
            continue
        group_effect = sum(cell_means) / len(cell_means)
        group_context = {**context, "workload": cell_features[min(selected_contexts)]}
        group = SupportGroup(
            family, group_context, group_effect, group_effect,
            not any(harm for _, _, _, _, harm in rows), versions=versions,
        )
        groups.append(group)
        family_role[family] = inside
        promotion_rows[family] = rows
        certificate = ValidationCertificate(
            kind=ValidationKind.REPLICATION if inside else ValidationKind.BOUNDARY,
            case_id=family,
            executed=True,
            execution_source=source_id,
            scientific_ok=group.hard_gates_passed,
            effect_lcb=group.effect_lower if inside else group.effect_upper,
            abstained=not inside,
        )
        validation_certificates.append(certificate)
        validation_error = certificate.validate(tolerance=0.0)
        if validation_error:
            governance_errors.append((family, validation_error))

    enough_calibration = (
        len(families) >= min_calibration_families
        and len(groups) >= min_calibration_families
        and len(support_ids) >= minimum_support_families
        and len(boundary_ids) >= minimum_non_support_families
    )
    selective_groups = [group for group in groups if family_role.get(group.group_id) is True]
    unsupported_groups = [group for group in groups if family_role.get(group.group_id) is False]
    selective_validated = (
        len(selective_groups) >= minimum_support_families
        and len(unsupported_groups) >= minimum_non_support_families
        and not governance_errors
        and context_behavior_valid
        and all(group.hard_gates_passed and group.effect_lower > 0.0 for group in selective_groups)
        and all(group.hard_gates_passed and group.effect_upper <= 0.0 for group in unsupported_groups)
    )
    selective = (
        candidate.synthesis
        if enough_calibration and all_calibration_pairs_valid and selective_validated
        else None
    )
    regime_context = (
        {**context, "workload": _context_features(calibration_pairs[0], pre_execution_features)}
        if calibration_pairs else context
    )
    regime = classify_support_regime(
        groups,
        scope=support_scope,
        target_context=regime_context,
        runtime_versions=versions,
        selective=selective,
        # These groups are role-stratified (support families and boundary
        # families), not a global representative sample for a pooled claim.
        fixed_time_complete=False,
    )
    deployable_certificate = (
        has_layout_grammar
        and enough_calibration
        and all_calibration_pairs_valid
        and not governance_errors
        and synthesis_certificate_valid(candidate.synthesis)
        and regime.action == "reuse"
        and regime.certificate_is_scope_bound()
    )
    governed_engine = None
    promotion_reason = "calibration_regime_did_not_clear_core_support_gate"
    if deployable_certificate and candidate.status == "CANDIDATE_ONLY":
        governed_engine, promotion_reason = _promote_from_calibration(
            engine=engine,
            candidate=candidate,
            groups=groups,
            family_rows=promotion_rows,
            family_role=family_role,
            context_strata_evidence=context_strata_evidence,
            source_id=source_id,
            minimum_support_families=minimum_support_families,
            minimum_non_support_families=minimum_non_support_families,
        )
    governed_match = governed_engine is not None
    route_engine = governed_engine or engine
    route = route_engine.route(
        task_context,
        support_regimes={candidate.spec.rule_id: regime if deployable_certificate and governed_match else None},
        runtime_versions=runtime_versions,
        candidate_ids=[candidate.spec.rule_id],
    )
    fitted = governed_match and deployable_certificate
    reason = (
        "scope-bound canonical promotion established; route each target context separately"
        if fitted else
        "no canonical layout applicability grammar exists"
        if not has_layout_grammar else
        f"candidate, independent calibration, canonical governance, scope, or route gate did not pass: {promotion_reason}"
    )
    return CalibrationResult(
        "FITTED" if fitted else (
            "ACRE_NOT_FITTED"
            if not has_layout_grammar
            else "NOT_FITTED"
        ), route, route_engine, regime, tuple(events),
        assessment, task_context, frozenset(families),
        tuple(validation_certificates), tuple(governance_errors), tuple(excluded), reason,
    )
