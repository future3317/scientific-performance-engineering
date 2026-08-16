"""Family-owned environment semantics shared by evolution conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math


@dataclass(frozen=True)
class EpisodeEnvironmentState:
    """Persistent regime state for one sequential evolution episode."""

    runtime_version: str = "A"
    hardware_regime: str = "default"
    scale_regime: str = "default"
    scientific_regime: str = "default"
    harness_regime: str = "default"
    active_poison: tuple[str, ...] = ()

    def apply(self, transformation: Mapping[str, Any] | Any) -> "EpisodeEnvironmentState":
        if hasattr(transformation, "kind"):
            kind = str(transformation.kind)
            parameters = dict(getattr(transformation, "parameters", {}) or {})
        else:
            kind = str(transformation.get("kind", ""))
            parameters = dict(transformation.get("parameters", {}) or {})
        values = self.__dict__.copy()
        if kind == "software":
            values["runtime_version"] = str(parameters.get("to_runtime", parameters.get("runtime_version", self.runtime_version)))
        elif kind == "hardware":
            values["hardware_regime"] = str(parameters.get("regime", parameters.get("hardware_regime", "shifted")))
        elif kind == "scale":
            values["scale_regime"] = str(parameters.get("regime", parameters.get("scale_regime", "shifted")))
        elif kind in {"scientific_regime", "science"}:
            values["scientific_regime"] = str(parameters.get("regime", "shifted"))
        elif kind == "harness":
            values["harness_regime"] = str(parameters.get("regime", "shifted"))
        elif kind == "poison":
            operator = str(parameters.get("operator", "unknown"))
            values["active_poison"] = tuple(sorted(set(self.active_poison) | {operator}))
        elif kind in {"revalidation", "recovery"}:
            values["runtime_version"] = str(parameters.get("runtime_version", "A"))
            values["active_poison"] = ()
        return EpisodeEnvironmentState(**values)


@dataclass(frozen=True)
class EnvironmentOutcome:
    utility: float
    scientific_gates: Mapping[str, bool]
    oracle_bundle: tuple[str, ...]


@dataclass(frozen=True)
class FamilyEnvironment:
    family_id: str

    def _state(self, transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None) -> EpisodeEnvironmentState:
        if transformation_state is None:
            return EpisodeEnvironmentState()
        if isinstance(transformation_state, EpisodeEnvironmentState):
            return transformation_state
        values = EpisodeEnvironmentState().__dict__.copy()
        for key in ("runtime_version", "hardware_regime", "scale_regime", "scientific_regime", "harness_regime"):
            if key in transformation_state:
                values[key] = str(transformation_state[key])
        state = EpisodeEnvironmentState(**values)
        if transformation_state.get("drifted"):
            state = state.apply({"kind": "software", "parameters": {"to_runtime": "B"}})
        if transformation_state.get("poison"):
            state = state.apply({"kind": "poison", "parameters": {"operator": "phase"}})
        return state

    def legal_bundles(self, context: Mapping[str, Any], transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None = None) -> tuple[tuple[str, ...], ...]:
        from .catalog import FAMILY_SPECS, resolve_family_id
        try:
            spec = FAMILY_SPECS[resolve_family_id(self.family_id)]
        except KeyError:
            return ((),)
        actions = tuple(spec.action_specs)
        return ((),) + tuple((action,) for action in actions)

    def evaluate(
        self,
        context: Mapping[str, Any],
        deployed_interventions: Sequence[str],
        transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None = None,
    ) -> EnvironmentOutcome:
        state = self._state(transformation_state)
        deployed = set(str(item) for item in deployed_interventions)
        workload = dict(context.get("workload", context) if isinstance(context, Mapping) else {})
        # Applicability is owned by FamilySpec.  The family-specific branches
        # below only translate that truth into outcome semantics under the
        # persistent regime state.
        from .catalog import FAMILY_SPECS, resolve_family_id
        try:
            family_spec = FAMILY_SPECS[resolve_family_id(self.family_id)]
            try:
                family_applicable = bool(family_spec.applicability(workload))
            except (KeyError, TypeError, ValueError):
                # A partial context is still sufficient to expose the family
                # action policy (for example when computing an oracle bundle
                # during a drift probe); missing parameters do not fabricate
                # applicability truth.
                family_applicable = not bool(workload)
            model = dict(family_spec.outcome_model)
        except (KeyError, TypeError, ValueError):
            family_spec = None
            family_applicable = True
            model = {"baseline": 0.60, "preferred": 0.80, "mismatch": 0.35, "poison_penalty": 0.20}
        # FamilySpec owns the action semantics.  The environment only maps
        # persistent regime state to the declared policy, so Boundary,
        # Interaction, and Evolution cannot drift into separate action maps.
        policy = dict(getattr(family_spec, "action_policy", {}) or {})
        shifted = any((
            state.runtime_version != "A",
            state.hardware_regime != "default",
            state.scale_regime != "default",
            state.scientific_regime != "default",
            state.harness_regime != "default",
        ))
        policy_key = "shifted" if shifted else "inapplicable" if not family_applicable else "default"
        preferred = str(policy.get(policy_key, ""))
        legal_actions = set(getattr(family_spec, "action_specs", {}) or {})
        if preferred and preferred not in legal_actions:
            raise ValueError(f"family {self.family_id} action policy references undeclared action {preferred}")
        utility = float(model["baseline"])
        if deployed:
            # Evaluate the concrete action against the FamilySpec action-level
            # contract.  A family-level applicability label never directly
            # supplies a utility value.
            action = sorted(deployed)[0]
            regime = "shifted" if shifted else "default"
            if family_spec is not None and family_spec.action_applicable(action, workload, regime=regime):
                utility = family_spec.action_effect(action, workload, regime=regime)
            else:
                utility = float(model["mismatch"])
            if state.active_poison:
                utility -= float(model["poison_penalty"])
        if family_spec is not None:
            invariant_names = tuple(getattr(family_spec, "scientific_invariants", ()) or ("finite_loss",))
            raw_gates = {
                name: (not deployed) or (bool(preferred) and action == preferred and math.isfinite(float(utility)) and not state.active_poison)
                for name in invariant_names
            }
            gates = family_spec.policy_spec().evaluate(raw_gates)
        else:
            gates = {"finite_loss": True}
        return EnvironmentOutcome(utility, gates, (preferred,) if preferred else ())

    def oracle(self, context: Mapping[str, Any], transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None = None) -> EnvironmentOutcome:
        """Enumerate legal bundles and select the hindsight-best outcome."""
        candidates = [self.evaluate(context, bundle, transformation_state) for bundle in self.legal_bundles(context, transformation_state)]
        return max(candidates, key=lambda outcome: outcome.utility)
