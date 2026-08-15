"""Environment semantics shared by C/D evolution conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EnvironmentOutcome:
    utility: float
    scientific_gates: Mapping[str, bool]
    oracle_bundle: tuple[str, ...]


@dataclass(frozen=True)
class FamilyEnvironment:
    family_id: str

    def legal_bundles(self, context: Mapping[str, Any], transformation_state: Mapping[str, Any] | None = None) -> tuple[tuple[str, ...], ...]:
        if self.family_id == "compile":
            return ((), ("reuse_compile_cache",), ("revalidate_compile_cache",))
        return ((),)

    def evaluate(
        self,
        context: Mapping[str, Any],
        deployed_interventions: Sequence[str],
        transformation_state: Mapping[str, Any] | None = None,
    ) -> EnvironmentOutcome:
        state = dict(transformation_state or {})
        drifted = bool(state.get("drifted") or state.get("to_runtime") == "B")
        poison = bool(state.get("poison"))
        deployed = set(str(item) for item in deployed_interventions)
        if self.family_id == "compile":
            if drifted and "reuse_compile_cache" in deployed:
                utility = 0.35
                oracle = ("revalidate_compile_cache",)
            elif drifted:
                utility = 0.70
                oracle = ("revalidate_compile_cache",)
            elif "reuse_compile_cache" in deployed or "compile-cache-rule" in deployed:
                utility = 0.80
                oracle = ("reuse_compile_cache",)
            else:
                utility = 0.60
                oracle = ("reuse_compile_cache",)
            if poison and "reuse_compile_cache" in deployed:
                utility = 0.45
            return EnvironmentOutcome(utility, {"finite_loss": True}, oracle)
        return EnvironmentOutcome(0.60 if not deployed else 0.80, {"finite_loss": True}, ())

    def oracle(self, context: Mapping[str, Any], transformation_state: Mapping[str, Any] | None = None) -> EnvironmentOutcome:
        """Enumerate legal bundles and select the hindsight-best outcome."""
        candidates = [self.evaluate(context, bundle, transformation_state) for bundle in self.legal_bundles(context, transformation_state)]
        return max(candidates, key=lambda outcome: outcome.utility)
