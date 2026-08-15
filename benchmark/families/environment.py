"""Family-owned environment semantics shared by evolution conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
        bundles = {
            "compile": ((), ("reuse_compile_cache",), ("revalidate_compile_cache",)),
            "graph_cache": ((), ("reuse_graph_cache",), ("rebuild_graph_cache",)),
            "h2d_pipeline": ((), ("pin_memory_pipeline",), ("prefetch_pipeline",)),
            "checkpoint": ((), ("checkpoint_recompute",), ("retained_graph",)),
            "scalar_sync": ((), ("aggregate_scalars",), ("defer_scalar_sync",)),
        }
        return bundles.get(self.family_id, ((),))

    def evaluate(
        self,
        context: Mapping[str, Any],
        deployed_interventions: Sequence[str],
        transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None = None,
    ) -> EnvironmentOutcome:
        state = self._state(transformation_state)
        deployed = set(str(item) for item in deployed_interventions)
        if self.family_id == "compile":
            drifted = state.runtime_version != "A"
            preferred = "revalidate_compile_cache" if drifted else "reuse_compile_cache"
            utility = 0.35 if drifted and "reuse_compile_cache" in deployed else 0.80 if preferred in deployed else 0.60
            if state.active_poison and "reuse_compile_cache" in deployed:
                utility -= 0.20
            return EnvironmentOutcome(utility, {"finite_loss": True}, (preferred,))
        if self.family_id == "graph_cache":
            drifted = state.scientific_regime != "default"
            preferred = "rebuild_graph_cache" if drifted else "reuse_graph_cache"
            utility = 0.35 if drifted and "reuse_graph_cache" in deployed else 0.78 if preferred in deployed else 0.60
            return EnvironmentOutcome(utility, {"finite_loss": True}, (preferred,))
        if self.family_id == "h2d_pipeline":
            shifted = state.hardware_regime != "default" or state.scale_regime != "default"
            preferred = "prefetch_pipeline" if shifted else "pin_memory_pipeline"
            utility = 0.76 if preferred in deployed else 0.58
            return EnvironmentOutcome(utility, {"finite_loss": True}, (preferred,))
        if self.family_id == "checkpoint":
            shifted = state.scale_regime != "default"
            preferred = "retained_graph" if shifted else "checkpoint_recompute"
            utility = 0.76 if preferred in deployed else 0.58
            return EnvironmentOutcome(utility, {"finite_loss": True}, (preferred,))
        if self.family_id == "scalar_sync":
            shifted = state.harness_regime != "default"
            preferred = "defer_scalar_sync" if shifted else "aggregate_scalars"
            utility = 0.76 if preferred in deployed else 0.58
            return EnvironmentOutcome(utility, {"finite_loss": True}, (preferred,))
        return EnvironmentOutcome(0.60 if not deployed else 0.80, {"finite_loss": True}, ())

    def oracle(self, context: Mapping[str, Any], transformation_state: Mapping[str, Any] | EpisodeEnvironmentState | None = None) -> EnvironmentOutcome:
        """Enumerate legal bundles and select the hindsight-best outcome."""
        candidates = [self.evaluate(context, bundle, transformation_state) for bundle in self.legal_bundles(context, transformation_state)]
        return max(candidates, key=lambda outcome: outcome.utility)
