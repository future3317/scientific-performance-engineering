"""Conservative routing over the canonical rule/relation contracts."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.models import RelationSpec, RelationState, RuleSpec, RuleState, TaskContext
from core.predicates import match_predicate
from core.cost import PromptCostModel


@dataclass(frozen=True)
class BundleCertificate:
    bundle_ids: tuple[str, ...]
    context_predicate: Mapping[str, Any]
    residual_lcb: float
    residual_ucb: float
    status: str

    @property
    def bounded_auto_allowed(self) -> bool:
        # A non-zero higher-order residual is a hyperedge, not evidence that
        # the pairwise graph is complete.  It must not silently pass the
        # bounded-auto gate.
        return self.status in {"pairwise_certified", "not_applicable"}


@dataclass(frozen=True)
class RoutingDecision:
    selected_rule_ids: tuple[str, ...]
    objective: float
    rejected_reasons: Mapping[str, tuple[str, ...]]
    bundle_certificate: BundleCertificate | None = None
    optimizer_mode: str = "exact"
    upper_bound: float | None = None
    optimality_gap: float | None = None
    blockers: tuple[Mapping[str, Any], ...] = ()


def validate_relation_nonoverlap(
    relation_specs: Sequence[RelationSpec], contexts: Sequence[Mapping[str, Any]] = ()
) -> list[str]:
    """Return active relation overlap errors for a registered relation set."""
    errors: list[str] = []
    grouped: dict[frozenset[str], list[RelationSpec]] = {}
    for spec in relation_specs:
        grouped.setdefault(frozenset(spec.endpoints.values()), []).append(spec)
    for pair, specs in grouped.items():
        for index, left in enumerate(specs):
            for right in specs[index + 1 :]:
                if left.relation_id == right.relation_id:
                    continue
                overlap = left.applicability == right.applicability
                if contexts:
                    overlap = overlap or any(
                        match_predicate(left.applicability, context)
                        and match_predicate(right.applicability, context)
                        for context in contexts
                    )
                if overlap:
                    errors.append(
                        f"relation applicability overlap for {left.relation_id} and {right.relation_id} on {sorted(pair)}"
                    )
    return errors


class ConservativeCausalRouter:
    def __init__(self, *, token_budget: int, zeta: float = 0.05, lambda_tokens: float = 0.01) -> None:
        if token_budget < 1 or zeta < 0.0 or lambda_tokens < 0.0:
            raise ValueError("token budget must be positive and penalties non-negative")
        self.token_budget = token_budget
        self.zeta = zeta
        self.lambda_tokens = lambda_tokens

    @staticmethod
    def _context(value: TaskContext | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(value, TaskContext):
            return value.to_dict()
        return value

    @staticmethod
    def _utility(state: RuleState) -> float:
        value = state.confidence_sequence.get("utility_effect_lcb")
        if value is None:
            value = state.effect.get("lower_utility", state.effect.get("utility", state.retrieval_utility))
        if not math.isfinite(float(value)):
            raise ValueError("rule state utility must be finite")
        return float(value)

    @staticmethod
    def _tokens(spec: RuleSpec) -> int:
        # Explicit measured runtime cost is authoritative when present;
        # otherwise use the canonical worker-visible serialization.  Formal
        # candidates populate the former from PromptCostModel, so C and D
        # still share one cost contract while old in-memory fixtures remain
        # usable.
        declared = spec.runtime_cost.get("tokens", spec.runtime_cost.get("token_cost"))
        if declared is not None:
            if float(declared) < 1:
                raise ValueError(f"rule {spec.rule_id} token cost must be positive")
            return int(math.ceil(float(declared)))
        view = {
            "rule_id": spec.rule_id,
            "version": spec.version,
            "action": spec.intervention,
            "mechanism": spec.expected_mechanism,
            "applicability": spec.applicability,
            "invariants": spec.scientific_invariants,
            "abstain_conditions": spec.abstain_conditions,
            "provenance": spec.provenance_policy,
        }
        return PromptCostModel().cost(view)

    @staticmethod
    def _relation_map(specs: Sequence[RelationSpec]) -> dict[frozenset[str], tuple[RelationSpec, ...]]:
        pairs: dict[frozenset[str], list[RelationSpec]] = {}
        for spec in specs:
            key = frozenset(spec.endpoints.values())
            if len(key) != 2:
                continue
            pairs.setdefault(key, []).append(spec)
        return {key: tuple(value) for key, value in pairs.items()}

    @staticmethod
    def _certificate(
        bundle: tuple[RuleSpec, ...],
        context: Mapping[str, Any],
        certificates: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        ids = frozenset(spec.rule_id for spec in bundle)
        for certificate in (certificates or {}).values():
            if not isinstance(certificate, Mapping):
                continue
            candidate_ids = certificate.get("bundle_ids") or certificate.get("rule_ids")
            if not isinstance(candidate_ids, list) or frozenset(str(item) for item in candidate_ids) != ids:
                continue
            versions = certificate.get("bundle_versions") or certificate.get("rule_versions")
            if not isinstance(versions, Mapping):
                continue
            expected = {spec.rule_id: int(spec.version) for spec in bundle}
            if {str(key): int(value) for key, value in versions.items()} != expected:
                continue
            predicate = certificate.get("context_predicate") or certificate.get("applicability") or {"all": []}
            if match_predicate(predicate, context):
                return certificate
        return None

    @staticmethod
    def _active(spec: RelationSpec, states: Mapping[str, RelationState], context: Mapping[str, Any]) -> bool:
        state = states.get(spec.relation_id)
        # Cross-context parent certificates are audit objects; only typed
        # relational-CEGIS children may enter deployment.
        if spec.kind == "context_dependent_interaction":
            return False
        return state is not None and state.status == "canonical" and state.drift_state == "stable" and match_predicate(spec.applicability, context)

    @classmethod
    def _matching_relations(
        cls,
        pair: frozenset[str],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
    ) -> tuple[tuple[RelationSpec, ...], bool]:
        active = [spec for spec in relation_map.get(pair, ()) if cls._active(spec, relation_states, context)]
        # A single canonical semantic relation is the only safe router state.
        # Multiple versions of the same relation resolve to the newest one;
        # distinct active relations are an overlap conflict regardless of kind
        # (independence and redundancy are not interchangeable).
        newest: dict[str, RelationSpec] = {}
        for spec in active:
            previous = newest.get(spec.relation_id)
            if previous is None or spec.version > previous.version:
                newest[spec.relation_id] = spec
        matches = tuple(sorted(newest.values(), key=lambda item: (item.relation_id, item.version)))
        return matches, len(matches) > 1

    @staticmethod
    def _lower_bound(spec: RelationSpec, state: RelationState | None) -> float:
        if spec.kind in {"independence", "redundancy"} or state is None:
            return 0.0
        bounds = state.contrast_bounds or {}
        if spec.kind == "prerequisite":
            key = "delta_a_given_b1" if spec.orientation == "left_to_right" else "delta_b_given_a1"
            return float(bounds.get(key, {}).get("lcb", state.confidence_sequence.get("utility_effect_lcb", state.estimate)))
        return float(bounds.get("gamma", {}).get("lcb", state.confidence_sequence.get("utility_effect_lcb", state.estimate)))

    def _invalid_reasons(
        self,
        bundle: tuple[RuleSpec, ...],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
        higher_order_certificates: Mapping[str, Any] | None = None,
        require_higher_order_certificate: bool = False,
    ) -> tuple[str, ...]:
        ids = {spec.rule_id for spec in bundle}
        reasons: set[str] = set()
        if sum(self._tokens(spec) for spec in bundle) > self.token_budget:
            reasons.add("token_budget")
        if len(bundle) >= 3 and require_higher_order_certificate:
            key = ":".join(sorted(spec.rule_id for spec in bundle))
            certificate = self._certificate(bundle, context, higher_order_certificates)
            if not isinstance(certificate, Mapping) or certificate.get("status") != "pairwise_certified":
                reasons.add("higher_order_certificate_required")
        # A directed prerequisite is a closure constraint, not merely a
        # pairwise bonus.  Reject a dependent rule when its prerequisite is
        # absent from the candidate bundle, even if the pair is not selected.
        for pair, candidates in relation_map.items():
            for relation in candidates:
                if relation.kind != "prerequisite" or not self._active(relation, relation_states, context):
                    continue
                left, right = relation.endpoints["left"], relation.endpoints["right"]
                prerequisite, dependent = (left, right) if relation.orientation == "left_to_right" else (right, left) if relation.orientation == "right_to_left" else (None, None)
                if prerequisite is not None and dependent in ids and prerequisite not in ids:
                    reasons.add("missing_prerequisite:" + prerequisite)
        for left, right in itertools.combinations(bundle, 2):
            matches, overlap_conflict = self._matching_relations(frozenset((left.rule_id, right.rule_id)), relation_map, relation_states, context)
            if overlap_conflict:
                reasons.add("relation_overlap_conflict")
            relation = matches[0] if len(matches) == 1 else None
            if relation is None:
                # One scientifically sensitive endpoint is sufficient to
                # make an unknown joint deployment unsafe.  The absence of a
                # certificate is not evidence of independence.
                if left.scientific_invariants or right.scientific_invariants:
                    reasons.add("unknown_scientific_interaction")
                continue
            if relation.kind == "semantic_conflict":
                reasons.add("hard_conflict")
        return tuple(sorted(reasons))

    def _objective(
        self,
        bundle: tuple[RuleSpec, ...],
        states: Mapping[str, RuleState],
        relation_map: Mapping[frozenset[str], tuple[RelationSpec, ...]],
        relation_states: Mapping[str, RelationState],
        context: Mapping[str, Any],
    ) -> float:
        score = sum(self._utility(states[spec.rule_id]) for spec in bundle)
        uncertainty_penalty = 0.0
        for left, right in itertools.combinations(bundle, 2):
            matches, _ = self._matching_relations(frozenset((left.rule_id, right.rule_id)), relation_map, relation_states, context)
            relation = matches[0] if len(matches) == 1 else None
            if relation is not None:
                state = relation_states.get(relation.relation_id)
                # Relation semantics are not interchangeable utility bonuses:
                # conflicts reject, prerequisites constrain closure, redundancy
                # removes duplicate node gain, and only synergy/antagonism
                # contribute a pairwise effect term.
                if relation.kind == "semantic_conflict":
                    continue
                if relation.kind == "prerequisite":
                    continue
                if relation.kind == "redundancy":
                    score -= min(self._utility(states[spec.rule_id]) for spec in (left, right))
                elif relation.kind in {"synergy", "antagonism"}:
                    score += self._lower_bound(relation, state)
                bounds = state.contrast_bounds.get("gamma", {}) if state is not None else {}
                if isinstance(bounds, Mapping) and {"lcb", "ucb"}.issubset(bounds):
                    uncertainty_penalty += max(0.0, float(bounds["ucb"]) - float(bounds["lcb"])) / 2.0
            else:
                # Unknown pairwise evidence is not independence.  It reduces
                # the robust value of a bundle even when neither endpoint is
                # scientifically sensitive enough to hard-block it.
                uncertainty_penalty += 0.05
        if len(bundle) > 1:
            uncertainty_penalty += self.zeta
        if len(bundle) >= 3:
            uncertainty_penalty += 0.05 * (len(bundle) - 2)
        score -= uncertainty_penalty
        score -= self.lambda_tokens * sum(self._tokens(spec) for spec in bundle)
        return score

    def route(
        self,
        rule_specs: Sequence[RuleSpec],
        rule_states: Mapping[str, RuleState],
        relation_specs: Sequence[RelationSpec],
        relation_states: Mapping[str, RelationState],
        context: TaskContext | Mapping[str, Any],
        higher_order_evidence: Mapping[str, float] | None = None,
        higher_order_certificates: Mapping[str, Any] | None = None,
        require_higher_order_certificate: bool = False,
    ) -> RoutingDecision:
        by_id = {spec.rule_id: spec for spec in rule_specs}
        if len(by_id) != len(rule_specs):
            raise ValueError("rule spec ids must be unique")
        if any(spec.rule_id not in rule_states for spec in rule_specs):
            raise ValueError("rule states must cover every rule spec")
        relation_map = self._relation_map(relation_specs)
        context_map = self._context(context)
        context_domain = context_map.get("domain")

        def domain_matches(spec: RuleSpec) -> bool:
            # ``runtime`` is the historical default for cards authored before
            # domain-aware routing; an explicit non-default domain must match
            # the full TaskContext root.
            return context_domain is None or spec.domain in {"runtime", str(context_domain)}

        def eligible(spec: RuleSpec, state: RuleState) -> bool:
            if state.status != "canonical" or state.drift_state != "stable":
                return False
            if not domain_matches(spec) or not match_predicate(spec.applicability, context_map):
                return False
            return not (spec.abstain_conditions and match_predicate(spec.abstain_conditions, context_map))

        applicable = [spec for spec in rule_specs if eligible(spec, rule_states[spec.rule_id])]
        rejected: dict[str, set[str]] = {spec.rule_id: set() for spec in rule_specs}
        for spec in rule_specs:
            state = rule_states[spec.rule_id]
            if state.status != "canonical" or state.drift_state != "stable":
                rejected[spec.rule_id].add("rule_state_ineligible")
            elif not domain_matches(spec):
                rejected[spec.rule_id].add("domain_mismatch")
            elif spec.abstain_conditions and match_predicate(spec.abstain_conditions, context_map):
                rejected[spec.rule_id].add("abstain_condition")
        valid: list[tuple[tuple[RuleSpec, ...], float]] = []
        optimizer_mode = "exact" if len(applicable) <= 16 else "conservative-greedy"
        bundles: list[tuple[RuleSpec, ...]]
        if len(applicable) > 16:
            selected: tuple[RuleSpec, ...] = ()
            for spec in sorted(applicable, key=lambda item: self._utility(rule_states[item.rule_id]), reverse=True):
                proposal = tuple(sorted((*selected, spec), key=lambda item: item.rule_id))
                if not self._invalid_reasons(proposal, relation_map, relation_states, context_map, higher_order_certificates, require_higher_order_certificate):
                    selected = proposal
            bundles = [selected]
        else:
            bundles = [selected for width in range(0, len(applicable) + 1) for selected in itertools.combinations(applicable, width)]
        for selected in bundles:
            bundle = tuple(sorted(selected, key=lambda spec: spec.rule_id))
            reasons = self._invalid_reasons(
                bundle, relation_map, relation_states, context_map,
                higher_order_certificates, require_higher_order_certificate,
            )
            if reasons:
                for spec in bundle:
                    rejected[spec.rule_id].update(reasons)
                continue
            valid.append((bundle, self._objective(bundle, rule_states, relation_map, relation_states, context_map)))
        if not valid:
            raise ValueError("no valid rule bundle under the token budget")
        bundle, objective = max(valid, key=lambda item: (item[1], tuple(spec.rule_id for spec in item[0])))
        selected_ids = {spec.rule_id for spec in bundle}
        for spec in rule_specs:
            if spec.rule_id not in selected_ids and not eligible(spec, rule_states[spec.rule_id]):
                rejected[spec.rule_id].add("not_applicable")
        evidence = dict(higher_order_evidence or {})
        if len(bundle) < 3:
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, 0.0, 0.0, "not_applicable")
        elif (stored := self._certificate(bundle, context_map, higher_order_certificates)) is not None:
            certificate = BundleCertificate(
                tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)},
                float(stored.get("residual_lcb", stored.get("lcb", -1.0))),
                float(stored.get("residual_ucb", stored.get("ucb", 1.0))),
                str(stored.get("status", "higher_order_suspected")),
            )
        elif "lcb" in evidence and "ucb" in evidence:
            lcb, ucb = float(evidence["lcb"]), float(evidence["ucb"])
            eta = float(evidence.get("practical_margin", 0.05))
            status = "pairwise_certified" if lcb >= -eta and ucb <= eta else "hyperedge_required"
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, lcb, ucb, status)
        else:
            certificate = BundleCertificate(tuple(spec.rule_id for spec in bundle), {"context": dict(context_map)}, -1.0, 1.0, "higher_order_suspected")
        blockers: list[Mapping[str, Any]] = []
        if len(bundle) >= 3 and certificate.status == "higher_order_suspected":
            blockers.append({"type": "higher_order", "bundle": list(spec.rule_id for spec in bundle), "required_arms": ["000", "001", "010", "011", "100", "101", "110", "111"]})
        if optimizer_mode == "exact":
            upper_bound = objective
            optimality_gap = 0.0
        else:
            optimistic = sum(max(0.0, self._utility(rule_states[spec.rule_id])) for spec in applicable)
            for left, right in itertools.combinations(applicable, 2):
                matches, _ = self._matching_relations(frozenset((left.rule_id, right.rule_id)), relation_map, relation_states, context_map)
                if len(matches) == 1:
                    optimistic += max(0.0, self._lower_bound(matches[0], relation_states.get(matches[0].relation_id)))
            upper_bound = optimistic
            optimality_gap = max(0.0, upper_bound - objective)
        return RoutingDecision(
            selected_rule_ids=tuple(spec.rule_id for spec in bundle),
            objective=objective,
            rejected_reasons={rule_id: tuple(sorted(reasons)) for rule_id, reasons in rejected.items() if reasons},
            bundle_certificate=certificate,
            optimizer_mode=optimizer_mode,
            upper_bound=upper_bound,
            optimality_gap=optimality_gap,
            blockers=tuple(blockers),
        )
