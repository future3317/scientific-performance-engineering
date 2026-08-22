"""Canonical workload families and deterministic instance projections.

The catalog is intentionally small: it describes the parameter axes and
interventions shared by task, boundary, interaction, and evolution views.
Task workspaces remain the executable anchors; this module does not duplicate
their benchmark or oracle implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Callable, Mapping
from core.models import ScientificPolicySpec


@dataclass(frozen=True)
class FamilyTransformation:
    """A legal change of regime applied between sequential episode phases."""

    name: str
    kind: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionSpec:
    """Canonical pairwise composition contract shared by interaction views."""

    left_family: str
    right_family: str
    context: Mapping[str, Any] = field(default_factory=dict)
    mechanism: str | None = None


@dataclass(frozen=True)
class InteractionOracle:
    """Generate factorial outcomes from family parameters, not labels."""

    spec: CompositionSpec

    def evaluate(self, left: "FamilyInstance", right: "FamilyInstance", context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ctx = dict(self.spec.context or {})
        ctx.update(context or {})
        numeric_left = [float(v) for v in left.parameters.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        numeric_right = [float(v) for v in right.parameters.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        left_score = sum(numeric_left) / max(1, len(numeric_left))
        right_score = sum(numeric_right) / max(1, len(numeric_right))
        left_active, right_active = left.applicable, right.applicable
        # The relation is a consequence of regime and parameter interaction;
        # no surface index or pre-assigned relation is consulted.
        # Relation truth is derived from the two family parameter points and
        # an explicit runtime regime.  A worker- or benchmark-supplied label
        # is never accepted as causal truth.  The pilot pair has a declared
        # mechanistic composition; other pairs use the conservative generic
        # applicability/score rule below.
        regime = str(ctx.get("regime", "baseline"))
        # Normalize parameter ownership before evaluating the composition so
        # swapping left/right family arguments cannot change hidden truth.
        by_family = {left.family_id: left.parameters, right.family_id: right.parameters}
        compile_parameters = by_family.get("compile", {})
        h2d_parameters = by_family.get("h2d_pipeline", {})
        worker_count = int(h2d_parameters.get("worker_count", -1))
        dynamic_rate = float(compile_parameters.get("dynamic_shape_rate", -1.0))
        pair_is_pipeline_compile = {
            self.spec.left_family,
            self.spec.right_family,
        } == {"h2d_pipeline", "compile"}
        sign_flip = pair_is_pipeline_compile and worker_count > 6 and dynamic_rate > 0.5
        semantic_conflict = pair_is_pipeline_compile and worker_count > 6 and dynamic_rate <= 0.5
        redundancy_regime = pair_is_pipeline_compile and worker_count <= 4 and dynamic_rate <= 0.2
        if semantic_conflict:
            relation = "semantic_conflict"
        elif redundancy_regime:
            relation = "redundancy"
        elif sign_flip and regime == "shifted":
            relation = "antagonism"
        elif left_active and right_active and left_score > 0 and right_score > 0:
            relation = "synergy"
        elif left_active and not right_active:
            relation = "prerequisite_a_to_b"
        elif right_active and not left_active:
            relation = "prerequisite_b_to_a"
        else:
            relation = "independence"
        a = min(0.15, 0.08 + abs(left_score) % 0.07)
        b = min(0.15, 0.08 + abs(right_score) % 0.07)
        if relation == "redundancy":
            a, b = 0.48, 0.42
        # Keep the latent factorial contrast inside the utility bounds.  The
        # interaction value is gamma, not an unbounded additive effect later
        # clipped into a different hidden relation.
        interaction = {"synergy": 0.12, "antagonism": -0.12, "independence": 0.0, "redundancy": 0.0, "prerequisite_a_to_b": 0.12, "prerequisite_b_to_a": 0.12, "semantic_conflict": 0.0}[relation]
        if sign_flip and regime == "shifted":
            interaction = -interaction if interaction else -0.12
        baseline = -0.20 if relation in {"synergy", "antagonism", "prerequisite_a_to_b", "prerequisite_b_to_a"} else 0.0
        outcomes = {"00": baseline, "10": baseline + a, "01": baseline + b, "11": baseline + a + b + 4.0 * interaction}
        if not all(-1.0 <= value <= 1.0 for value in outcomes.values()):
            raise ValueError("composition oracle produced an out-of-range factorial cell")
        if relation == "prerequisite_a_to_b": outcomes["01"] = baseline
        if relation == "prerequisite_b_to_a": outcomes["10"] = baseline
        if relation == "redundancy": outcomes["11"] = max(outcomes["10"], outcomes["01"])
        gates = {arm: True for arm in ("00", "10", "01", "11")}
        if relation == "semantic_conflict":
            gates["11"] = False
        # Hidden truth and noisy estimation share one semantic decision policy.
        # The oracle supplies degenerate (point) confidence sets; it does not
        # maintain a second threshold implementation.
        from core.acre.policy import RelationDecisionPolicy
        gamma = (outcomes["11"] - outcomes["10"] - outcomes["01"] + outcomes["00"]) / 4.0
        intervals = {
            "gamma": (gamma, gamma),
            "delta_a_given_b0": ((outcomes["10"] - outcomes["00"]) / 2.0,) * 2,
            "delta_a_given_b1": ((outcomes["11"] - outcomes["01"]) / 2.0,) * 2,
            "delta_b_given_a0": ((outcomes["01"] - outcomes["00"]) / 2.0,) * 2,
            "delta_b_given_a1": ((outcomes["11"] - outcomes["10"]) / 2.0,) * 2,
            "redundancy": ((outcomes["11"] - max(outcomes["10"], outcomes["01"])) / 2.0,) * 2,
        }
        policy = RelationDecisionPolicy(0.05)
        if gates["11"] is False and all(gates.get(arm, False) for arm in ("00", "10", "01")):
            derived_relation = "semantic_conflict"
        else:
            derived_relation = policy.decide(intervals, gates, kind_hint="redundancy")
            if derived_relation == "unresolved":
                derived_relation = policy.decide(intervals, gates, kind_hint="prerequisite")
            if derived_relation == "unresolved":
                derived_relation = policy.decide(intervals, gates)
        derived_relation = {
            "confirmed_synergy": "synergy",
            "confirmed_antagonism": "antagonism",
            "confirmed_independence": "independence",
            "confirmed_redundancy": "redundancy",
        }.get(derived_relation, derived_relation)
        return {"outcomes": outcomes, "hidden_relation": derived_relation, "target_gamma": gamma, "scientific_gates": gates, "higher_order_residual": float(ctx.get("higher_order_residual", 0.0))}


@dataclass(frozen=True)
class FamilyInstance:
    """One frozen point in a family parameter space."""

    family_id: str
    instance_id: str
    parameters: Mapping[str, Any]
    polarity: str = "positive"
    difficulty_tier: str = "medium"
    anchor_task_id: str | None = None
    lineage: Mapping[str, Any] | None = None
    applicable: bool = True
    scientific_truth: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "parameters": dict(self.parameters),
            "polarity": self.polarity,
            "difficulty_tier": self.difficulty_tier,
            "anchor_task_id": self.anchor_task_id,
            "lineage": dict(self.lineage or {}),
            "applicable": self.applicable,
            "scientific_truth": dict(self.scientific_truth or {}),
        }


@dataclass(frozen=True)
class FamilySurfaceSpec:
    """Frozen, disjoint partitions shared by every benchmark view."""

    decision_lattice_id: str
    synthesis_ids: tuple[str, ...]
    promotion_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_lattice_id": self.decision_lattice_id,
            "synthesis_ids": list(self.synthesis_ids),
            "promotion_ids": list(self.promotion_ids),
            "validation_ids": list(self.validation_ids),
        }


def family_instance_digest(family_id: str, parameters: Mapping[str, Any]) -> str:
    """Stable identity for a canonical family point."""
    payload = {"family_id": family_id, "parameters": dict(parameters)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    generator_family_id: str
    track: str
    scientific_contract_id: str
    parameter_space: tuple[str, ...]
    interventions: tuple[str, ...]
    transformations: tuple[str, ...]
    generator: Callable[[int, int], Mapping[str, Any]]
    anchors: tuple[str, ...] = ()
    applicability: Callable[[Mapping[str, Any]], bool] = lambda _parameters: True
    scientific_truth: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda parameters: {
        "applicable": True,
        "parameters": dict(parameters),
    }
    anchor_parameters: Mapping[str, Mapping[str, Any]] | None = None
    predicate_features: tuple[Mapping[str, str], ...] = ()
    threshold_universe: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    scientific_invariants: tuple[str, ...] = ()
    default_severity: str = "P2"
    # All views consume these declarations.  They are projections of the
    # family contract, not parallel benchmark-specific maps.
    action_specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    decision_lattice: tuple[Mapping[str, Any], ...] = ()
    legal_compositions: tuple[CompositionSpec, ...] = ()
    validation_scenarios: tuple[Mapping[str, Any], ...] = ()
    formal_predicate_grammar: bool = True
    # Named contract components are kept on the family even when the default
    # environment adapter supplies the executable implementation.
    public_features: tuple[str, ...] = ()
    feature_domains: Mapping[str, Any] = field(default_factory=dict)
    scientific_policy: Mapping[str, Any] = field(default_factory=dict)
    realization_validators: tuple[str, ...] = ()
    outcome_model: Mapping[str, float] = field(default_factory=lambda: {
        "baseline": 0.60,
        "preferred": 0.80,
        "mismatch": 0.35,
        "poison_penalty": 0.20,
    })
    action_policy: Mapping[str, str] = field(default_factory=dict)

    def policy_spec(self) -> ScientificPolicySpec:
        value = self.scientific_policy or {"policy_id": self.scientific_contract_id, "required_gates": list(self.scientific_invariants)}
        return ScientificPolicySpec(str(value.get("policy_id", self.scientific_contract_id)), tuple(str(item) for item in value.get("required_gates", self.scientific_invariants)), dict(value.get("tolerance", {})))

    def action_policy_spec(self, action_id: str) -> ScientificPolicySpec:
        metadata = self.action_specs.get(action_id)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"unknown action {action_id} for family {self.family_id}")
        policy_id = str(metadata.get("scientific_policy_ref", self.scientific_contract_id))
        return ScientificPolicySpec(policy_id, _ACTION_POLICY_GATES.get(policy_id, self.scientific_invariants), {})

    def action_applicable(self, action_id: str, parameters: Mapping[str, Any], *, regime: str = "default") -> bool:
        """Return applicability for one declared action, not the family label.

        The family predicate answers whether the workload is relevant; this
        action-level contract answers whether the specific intervention is
        deployable in the current regime.
        """
        if action_id not in self.action_specs:
            return False
        declared = self.action_specs[action_id].get("applicability") if isinstance(self.action_specs[action_id], Mapping) else None
        if callable(declared):
            return bool(declared(parameters))
        if isinstance(declared, Mapping):
            from core.predicates import match_predicate
            return bool(match_predicate(dict(declared), {"workload": dict(parameters), **dict(parameters)}))
        try:
            family_ok = bool(self.applicability(parameters))
        except (KeyError, TypeError, ValueError):
            # Environment oracle construction may only have a regime context;
            # it must not manufacture a negative action solely because a
            # workload lattice point is unavailable.
            family_ok = True
        return family_ok

    def action_bundle_applicable(self, action_ids: tuple[str, ...], parameters: Mapping[str, Any], *, regime: str = "default") -> bool:
        return bool(action_ids) and all(self.action_applicable(action_id, parameters, regime=regime) for action_id in action_ids)

    def action_bundle_effect(self, action_ids: tuple[str, ...], parameters: Mapping[str, Any], *, regime: str = "default") -> float:
        if not action_ids:
            return float(self.outcome_model.get("baseline", 0.0))
        baseline = float(self.outcome_model.get("baseline", 0.0))
        if not self.action_bundle_applicable(action_ids, parameters, regime=regime):
            return float(self.outcome_model.get("mismatch", baseline))
        # Bundle semantics are additive over independent semantic actions,
        # with each action's declared effect measured from the same control.
        # A family may override this by declaring ``bundle_effect`` in its
        # action metadata; no action is silently replaced by the first item.
        declared = [self.action_specs[action_id].get("bundle_effect") for action_id in action_ids if isinstance(self.action_specs.get(action_id), Mapping)]
        if len(action_ids) > 1 and all(isinstance(value, (int, float)) for value in declared):
            return max(-1.0, min(1.0, float(declared[-1])))
        return max(-1.0, min(1.0, baseline + sum(self.action_effect(action_id, parameters, regime=regime) - baseline for action_id in action_ids)))

    def action_effect(self, action_id: str, parameters: Mapping[str, Any], *, regime: str = "default") -> float:
        """Return the bounded utility for a concrete action in this context."""
        model = self.action_specs.get(action_id, {})
        if isinstance(model, Mapping) and isinstance(model.get("effect"), (int, float)):
            return float(model["effect"])
        return float(self.outcome_model.get("preferred", self.outcome_model.get("baseline", 0.0))) if self.action_applicable(action_id, parameters, regime=regime) else float(self.outcome_model.get("mismatch", self.outcome_model.get("baseline", 0.0)))

    def generate(self, count: int, seed: int = 0) -> list[FamilyInstance]:
        if count < 1:
            raise ValueError("count must be positive")
        result: list[FamilyInstance] = []
        for index in range(count):
            params = dict(self.generator(index, seed))
            # Generators may carry legacy metadata, but truth is never read
            # from it.  Applicability is the only source of polarity.
            params.pop("polarity", None)
            difficulty = str(params.pop("difficulty_tier", "medium"))
            applicable = bool(self.applicability(params))
            truth = dict(self.scientific_truth(params) or {})
            truth["applicable"] = applicable
            result.append(
                FamilyInstance(
                    family_id=self.family_id,
                    instance_id=f"{self.family_id}-{seed:04d}-{index:04d}",
                    parameters=params,
                    polarity="positive" if applicable else "counterexample",
                    difficulty_tier=difficulty,
                    lineage={"generator_family_id": self.generator_family_id, "seed": seed, "index": index},
                    applicable=applicable,
                    scientific_truth=truth,
                )
            )
        return result

    def reconstruct_anchor(self, task_id: str) -> FamilyInstance:
        """Rebuild a declared anchor from the family generator.

        Anchors are fixed points in the family parameter space.  Their position
        is the declared order in ``anchors``; using the same generator and seed
        makes reconstruction deterministic without copying task workspaces.
        """
        if task_id not in self.anchors:
            raise ValueError(f"{task_id} is not a canonical anchor of family {self.family_id}")
        index = self.anchors.index(task_id)
        if not self.anchor_parameters or task_id not in self.anchor_parameters:
            raise ValueError(f"{self.family_id} anchor {task_id} lacks explicit parameters")
        params = dict(self.anchor_parameters[task_id])
        applicable = bool(self.applicability(params))
        truth = dict(self.scientific_truth(params) or {})
        truth["applicable"] = applicable
        truth["anchor_task_id"] = task_id
        return FamilyInstance(
            family_id=self.family_id,
            instance_id=task_id,
            parameters=params,
            polarity="positive" if applicable else "counterexample",
            difficulty_tier="medium",
            anchor_task_id=task_id,
            lineage={"generator_family_id": self.generator_family_id, "role": "anchor", "anchor_index": index},
            applicable=applicable,
            scientific_truth=truth,
        )


def _compile(index: int, seed: int) -> Mapping[str, Any]:
    return {
        "logical_steps": 32 + ((index + seed) % 16) * 32,
        "graph_size": 32 + ((index * 3 + seed) % 10) * 32,
        "dynamic_shape_rate": ((index + seed) % 8) / 10.0,
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _graph_cache(index: int, seed: int) -> Mapping[str, Any]:
    displacement = 0.005 + ((index + seed) % 40) * 0.003
    return {
        "geometry_displacement": displacement,
        "skin": 0.2 + ((index * 2 + seed) % 7) * 0.1,
        "graph_size": 32 + ((index + seed) % 10) * 32,
        "dynamic_rate": ((index + seed) % 8) / 10.0,
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _h2d(index: int, seed: int) -> Mapping[str, Any]:
    return {
        "batch_size": 16 + ((index + seed) % 8) * 16,
        "worker_count": (index + seed) % 9,
        "prefetch_factor": 2 + ((index + seed) % 4),
        "pin_memory": bool((index + seed) % 2),
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _checkpoint(index: int, seed: int) -> Mapping[str, Any]:
    pressure = 0.25 + ((index + seed) % 10) * 0.08
    return {
        "memory_pressure": pressure,
        "segment_count": 2 + ((index + seed) % 8),
        "recompute_ratio": 0.1 + ((index + seed) % 6) * 0.1,
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _scalar_sync(index: int, seed: int) -> Mapping[str, Any]:
    syncs = 1 + ((index + seed) % 32)
    return {
        "scalar_syncs_per_step": syncs,
        "metric_cadence": 1 + ((index * 3 + seed) % 16),
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _legacy(index: int, seed: int) -> Mapping[str, Any]:
    return {"fixture_index": index, "seed": seed, "difficulty_tier": "medium"}


def _repeated_compute(index: int, seed: int) -> Mapping[str, Any]:
    return {"repeat_count": 2 + ((index + seed) % 8), "backbone_width": 128 + ((index + seed) % 4) * 128, "batch_size": 16 + ((index * 3 + seed) % 4) * 16, "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _autograd(index: int, seed: int) -> Mapping[str, Any]:
    return {"output_count": 2 + ((index + seed) % 16), "input_dim": 64 + ((index + seed) % 4) * 64, "jacobian_density": 0.2 + ((index * 3 + seed) % 5) * 0.15, "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _equivariant_head(index: int, seed: int) -> Mapping[str, Any]:
    return {"irrep_order": 1 + ((index + seed) % 3), "node_count": 32 + ((index + seed) % 4) * 32, "recompute_rate": 0.1 + ((index * 2 + seed) % 7) * 0.1, "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _crystal_generation(index: int, seed: int) -> Mapping[str, Any]:
    return {"atom_count": 16 + ((index + seed) % 8) * 8, "diffusion_steps": 50 + ((index + seed) % 4) * 50, "guidance_scale": 1.0 + ((index + seed) % 4), "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _crystal_sampling(index: int, seed: int) -> Mapping[str, Any]:
    return {"neighbor_count": 8 + ((index + seed) % 8) * 4, "sample_count": 16 + ((index + seed) % 8) * 8, "geometry_variation": 0.05 + ((index * 2 + seed) % 8) * 0.1, "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _episode(index: int, seed: int) -> Mapping[str, Any]:
    return {"runtime_version": "A" if (index + seed) % 2 == 0 else "B", "context_width": 2 + ((index + seed) % 8), "drift_rate": 0.05 + ((index * 2 + seed) % 8) * 0.1, "fixture_index": index, "seed": seed, "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3]}


def _truth_from_polarity(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": bool(parameters.get("applicable", True)), "parameters": dict(parameters)}


def _legacy_applicability(parameters: Mapping[str, Any]) -> bool:
    return bool(parameters.get("applicable", True))


def _compile_applicability(parameters: Mapping[str, Any]) -> bool:
    logical_steps = int(parameters["logical_steps"])
    dynamic_rate = float(parameters["dynamic_shape_rate"])
    graph_size = int(parameters.get("graph_size", 0))
    if dynamic_rate > 0.0:
        return logical_steps >= 128 and dynamic_rate <= 0.4
    # Cold compilation is not worth applying to the calibrated tiny fixed-shape
    # graph-break point (graph_size=64); keep that point as an explicit
    # counterexample instead of claiming a positive graph-break repair.
    return logical_steps >= 128 and graph_size >= 128
def _compile_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    applicable = _compile_applicability(parameters)
    logical_steps = int(parameters["logical_steps"])
    dynamic_rate = float(parameters["dynamic_shape_rate"])
    graph_size = int(parameters.get("graph_size", 0))
    if graph_size >= 320:
        mechanism = "kernel_fusion"
        preferred_action = "fuse_pointwise_chain"
    elif logical_steps < 128:
        mechanism = "compile_tiny_graphs"
        preferred_action = "bypass_compile"
    elif dynamic_rate > 0.0:
        mechanism = "compile_dynamic_shapes"
        preferred_action = "stabilize_dynamic_guards"
    else:
        mechanism = "compile_graph_break"
        preferred_action = "remove_compile_graph_break"
    return {
        "applicable": applicable,
        "mechanism": mechanism,
        "preferred_action": preferred_action,
        "boundary": {"logical_steps": 128, "dynamic_shape_rate": 0.4, "graph_size": 128},
    }


def _graph_applicability(parameters: Mapping[str, Any]) -> bool:
    return float(parameters["geometry_displacement"]) <= 0.05


def _graph_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": _graph_applicability(parameters), "boundary": {"geometry_displacement": 0.05}}


def _h2d_applicability(parameters: Mapping[str, Any]) -> bool:
    return bool(parameters["pin_memory"]) and int(parameters["worker_count"]) <= 4


def _h2d_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": _h2d_applicability(parameters), "boundary": {"worker_count": 4, "pin_memory": True}}


def _checkpoint_applicability(parameters: Mapping[str, Any]) -> bool:
    return float(parameters["memory_pressure"]) >= 0.57


def _checkpoint_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": _checkpoint_applicability(parameters), "boundary": {"memory_pressure": 0.57}}


def _scalar_applicability(parameters: Mapping[str, Any]) -> bool:
    return int(parameters["scalar_syncs_per_step"]) > 8


def _scalar_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": _scalar_applicability(parameters), "boundary": {"scalar_syncs_per_step": 8}}


_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec("compile", "GEN-COMPILER", "spe_core", "CONTRACT-COMPILER-CACHE", ("logical_steps", "graph_size", "dynamic_shape_rate"), ("compile", "checkpoint"), ("software", "hardware", "scale", "model"), _compile, ("CORE-COMPILE-RECOMPILE-04", "CORE-COMPILE-DYNAMIC-11", "CORE-COMPILE-TINY-12", "CORE-KERNEL-FUSION-09R2"), _compile_applicability, _compile_truth, {"CORE-COMPILE-RECOMPILE-04": False, "CORE-COMPILE-DYNAMIC-11": True, "CORE-COMPILE-TINY-12": False, "CORE-KERNEL-FUSION-09R2": True}),
    FamilySpec("graph_cache", "GEN-GRAPH-CACHE", "sciml", "CONTRACT-ENERGY-FORCE", ("geometry_displacement", "skin", "graph_size", "dynamic_rate"), ("graph_cache", "geometry_motion"), ("software", "scale", "scientific_regime"), _graph_cache, ("SCIML-GNN-RAGGED-05R2", "SCIML-GNN-STATIC-GRAPH-CACHE-17R2", "SCIML-GNN-DYNAMIC-GRAPH-18R2", "SCIML-FORCE-AUTOGRAD-19R2", "SCIML-GRAPH-CACHE-INTERIOR-27"), _graph_applicability, _graph_truth, {"SCIML-GNN-RAGGED-05R2": True, "SCIML-GNN-STATIC-GRAPH-CACHE-17R2": True, "SCIML-GNN-DYNAMIC-GRAPH-18R2": False, "SCIML-FORCE-AUTOGRAD-19R2": True, "SCIML-GRAPH-CACHE-INTERIOR-27": True}),
    FamilySpec("h2d_pipeline", "GEN-H2D-PIPELINE", "spe_core", "CONTRACT-DATA-PIPELINE", ("batch_size", "worker_count", "prefetch_factor", "pin_memory"), ("pin_memory", "non_blocking", "prefetch"), ("hardware", "scale", "harness"), _h2d, ("CORE-H2D-PIPELINE-03R2", "CORE-DATALOADER-FANOUT-16R2", "CORE-H2D-OVERFANOUT-23R3"), _h2d_applicability, _h2d_truth, {"CORE-H2D-PIPELINE-03R2": True, "CORE-DATALOADER-FANOUT-16R2": True, "CORE-H2D-OVERFANOUT-23R3": False}),
    FamilySpec("checkpoint", "GEN-CHECKPOINT", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("memory_pressure", "segment_count", "recompute_ratio"), ("checkpoint", "gradient_accumulation"), ("scale", "hardware", "scientific_regime"), _checkpoint, ("CORE-MEM-RETAINED-GRAPH-13R2", "CORE-CHECKPOINT-AMPLE-MEM-14R2", "CORE-CHECKPOINT-HIGH-PRESSURE-24R2"), _checkpoint_applicability, _checkpoint_truth, {"CORE-MEM-RETAINED-GRAPH-13R2": True, "CORE-CHECKPOINT-AMPLE-MEM-14R2": False, "CORE-CHECKPOINT-HIGH-PRESSURE-24R2": True}),
    FamilySpec("scalar_sync", "GEN-SCALAR-SYNC", "spe_core", "CONTRACT-TRAINING-LOOP", ("scalar_syncs_per_step", "metric_cadence"), ("metric_aggregation", "device_sync"), ("scale", "harness"), _scalar_sync, ("CORE-SCALAR-SYNC-01R2", "CORE-SCALAR-SYNC-LOW-CADENCE-21"), _scalar_applicability, _scalar_truth, {"CORE-SCALAR-SYNC-01R2": True, "CORE-SCALAR-SYNC-LOW-CADENCE-21": False}),
    FamilySpec("repeated_compute", "GEN-REPEATED-COMPUTE", "spe_core", "CONTRACT-REPEATED-COMPUTE", ("repeat_count", "backbone_width", "batch_size", "fixture_index", "seed"), ("backbone_reuse",), ("scale",), _repeated_compute, ("CORE-REPEATED-BACKBONE-02R2", "CORE-REPEATED-BACKBONE-LOW-REUSE-22"), lambda p: int(p["repeat_count"]) >= 4, _truth_from_polarity, {"CORE-REPEATED-BACKBONE-02R2": {"repeat_count": 4, "backbone_width": 256, "batch_size": 32, "fixture_index": 0, "seed": 0}, "CORE-REPEATED-BACKBONE-LOW-REUSE-22": {"repeat_count": 2, "backbone_width": 128, "batch_size": 16, "fixture_index": 1, "seed": 0}}),
    FamilySpec("autograd", "GEN-AUTOGRAD-VJP", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("output_count", "input_dim", "jacobian_density", "fixture_index", "seed"), ("batched_vjp",), ("scale",), _autograd, ("CORE-AUTOGRAD-BATCHED-VJP-15R2", "CORE-AUTOGRAD-VJP-SMALL-25R2"), lambda p: int(p["output_count"]) >= 8, _truth_from_polarity, {"CORE-AUTOGRAD-BATCHED-VJP-15R2": {"output_count": 8, "input_dim": 128, "jacobian_density": 0.5, "fixture_index": 0, "seed": 0}, "CORE-AUTOGRAD-VJP-SMALL-25R2": {"output_count": 2, "input_dim": 64, "jacobian_density": 0.5, "fixture_index": 1, "seed": 0}}),
    FamilySpec("equivariant_head", "GEN-EQUIVARIANT_HEAD", "sciml", "CONTRACT-EQUIVARIANCE", ("irrep_order", "node_count", "recompute_rate", "fixture_index", "seed"), ("equivariant_recompute",), ("scientific_regime",), _equivariant_head, ("SCIML-EQUIV-RECOMPUTE-06R2", "SCIML-EQUIV-LOWORDER-26R2"), lambda p: int(p["irrep_order"]) < 2, _truth_from_polarity, {"SCIML-EQUIV-RECOMPUTE-06R2": {"irrep_order": 2, "node_count": 64, "recompute_rate": 0.5, "fixture_index": 0, "seed": 0}, "SCIML-EQUIV-LOWORDER-26R2": {"irrep_order": 1, "node_count": 64, "recompute_rate": 0.75, "fixture_index": 1, "seed": 0}}),
    FamilySpec("crystal_generation", "GEN-CRYSTAL_GENERATION", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("atom_count", "diffusion_steps", "guidance_scale", "fixture_index", "seed"), ("sampling",), ("scale", "scientific_regime"), _crystal_generation, ("SCIML-CRYSTAL-DIFFUSION-07R2", "SCIML-CRYSTAL-HIGH-GUIDANCE-28"), lambda p: float(p["guidance_scale"]) <= 3.0, _truth_from_polarity, {"SCIML-CRYSTAL-DIFFUSION-07R2": {"atom_count": 32, "diffusion_steps": 100, "guidance_scale": 2.0, "fixture_index": 0, "seed": 0}, "SCIML-CRYSTAL-HIGH-GUIDANCE-28": {"atom_count": 32, "diffusion_steps": 100, "guidance_scale": 4.0, "fixture_index": 1, "seed": 0}}),
    FamilySpec("crystal_sampling", "GEN-CRYSTAL_SAMPLING", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("neighbor_count", "sample_count", "geometry_variation", "fixture_index", "seed"), ("graph_rebuild",), ("scale", "scientific_regime"), _crystal_sampling, ("SCIML-GRAPH-REBUILD-08R2", "SCIML-CRYSTAL-STATIC-SAMPLING-29R2"), lambda p: float(p["geometry_variation"]) <= 0.35, _truth_from_polarity, {"SCIML-GRAPH-REBUILD-08R2": {"neighbor_count": 12, "sample_count": 24, "geometry_variation": 0.7, "fixture_index": 0, "seed": 0}, "SCIML-CRYSTAL-STATIC-SAMPLING-29R2": {"neighbor_count": 8, "sample_count": 32, "geometry_variation": 0.2, "fixture_index": 1, "seed": 0}}),
    FamilySpec("episode", "GEN-EPISODE", "evolution", "CONTRACT-EVOLUTION-GOVERNANCE", ("runtime_version", "context_width", "drift_rate", "fixture_index", "seed"), ("rule_update",), ("software", "hardware", "scale", "scientific_regime", "harness"), _episode, ("EVOL-EPISODE-POISON-10", "EVOL-COMPILER-DRIFT-20", "EVOL-EQUIVARIANT-SPECIALIZE-30"), lambda p: float(p["drift_rate"]) <= 0.45, _truth_from_polarity, {"EVOL-EPISODE-POISON-10": {"runtime_version": "A", "context_width": 2, "drift_rate": 0.05, "fixture_index": 0, "seed": 0}, "EVOL-COMPILER-DRIFT-20": {"runtime_version": "B", "context_width": 3, "drift_rate": 0.25, "fixture_index": 1, "seed": 0}, "EVOL-EQUIVARIANT-SPECIALIZE-30": {"runtime_version": "A", "context_width": 4, "drift_rate": 0.3, "fixture_index": 2, "seed": 0}}),
)

# Explicit anchor coordinates are the canonical task truth.  The legacy
# generator-index projections above remain useful for unmaterialized pools,
# but never define an anchor's applicability.
_EXPLICIT_ANCHORS: dict[str, dict[str, Mapping[str, Any]]] = {
    "compile": {
        "CORE-COMPILE-RECOMPILE-04": {"logical_steps": 128, "graph_size": 64, "dynamic_shape_rate": 0.0},
        "CORE-COMPILE-DYNAMIC-11": {"logical_steps": 128, "graph_size": 128, "dynamic_shape_rate": 0.3},
        "CORE-COMPILE-TINY-12": {"logical_steps": 8, "graph_size": 64, "dynamic_shape_rate": 0.8},
        "CORE-KERNEL-FUSION-09R2": {"logical_steps": 192, "graph_size": 320, "dynamic_shape_rate": 0.2},
    },
    "graph_cache": {
        "SCIML-GNN-RAGGED-05R2": {"geometry_displacement": 0.02, "skin": 0.2, "graph_size": 64, "dynamic_rate": 0.0},
        "SCIML-GNN-STATIC-GRAPH-CACHE-17R2": {"geometry_displacement": 0.03, "skin": 0.4, "graph_size": 96, "dynamic_rate": 0.1},
        "SCIML-GNN-DYNAMIC-GRAPH-18R2": {"geometry_displacement": 0.08, "skin": 0.6, "graph_size": 96, "dynamic_rate": 0.8},
        "SCIML-FORCE-AUTOGRAD-19R2": {"geometry_displacement": 0.02, "skin": 0.8, "graph_size": 128, "dynamic_rate": 0.3},
        "SCIML-GRAPH-CACHE-INTERIOR-27": {"geometry_displacement": 0.03, "skin": 0.4, "graph_size": 128, "dynamic_rate": 0.2},
    },
    "h2d_pipeline": {
        "CORE-H2D-PIPELINE-03R2": {"batch_size": 4096, "worker_count": 2, "prefetch_factor": 2, "pin_memory": True},
        "CORE-DATALOADER-FANOUT-16R2": {"batch_size": 64, "worker_count": 3, "prefetch_factor": 4, "pin_memory": True},
        "CORE-H2D-OVERFANOUT-23R3": {"batch_size": 32, "worker_count": 5, "prefetch_factor": 4, "pin_memory": True},
    },
    "checkpoint": {
        "CORE-MEM-RETAINED-GRAPH-13R2": {"memory_pressure": 0.7, "segment_count": 4, "recompute_ratio": 0.2},
        "CORE-CHECKPOINT-AMPLE-MEM-14R2": {"memory_pressure": 0.4, "segment_count": 3, "recompute_ratio": 0.2},
        "CORE-CHECKPOINT-HIGH-PRESSURE-24R2": {"memory_pressure": 0.8, "segment_count": 4, "recompute_ratio": 0.5},
    },
    "scalar_sync": {"CORE-SCALAR-SYNC-01R2": {"scalar_syncs_per_step": 12, "metric_cadence": 4}, "CORE-SCALAR-SYNC-LOW-CADENCE-21": {"scalar_syncs_per_step": 4, "metric_cadence": 8}},
    "repeated_compute": {"CORE-REPEATED-BACKBONE-02R2": {"repeat_count": 4, "backbone_width": 256, "batch_size": 32, "fixture_index": 0, "seed": 0}, "CORE-REPEATED-BACKBONE-LOW-REUSE-22": {"repeat_count": 2, "backbone_width": 128, "batch_size": 16, "fixture_index": 1, "seed": 0}},
    "autograd": {"CORE-AUTOGRAD-BATCHED-VJP-15R2": {"output_count": 8, "input_dim": 128, "jacobian_density": 0.5, "fixture_index": 0, "seed": 0}, "CORE-AUTOGRAD-VJP-SMALL-25R2": {"output_count": 2, "input_dim": 64, "jacobian_density": 0.5, "fixture_index": 1, "seed": 0}},
    "equivariant_head": {"SCIML-EQUIV-RECOMPUTE-06R2": {"irrep_order": 2, "node_count": 64, "recompute_rate": 0.5, "fixture_index": 0, "seed": 0}, "SCIML-EQUIV-LOWORDER-26R2": {"irrep_order": 1, "node_count": 64, "recompute_rate": 0.75, "fixture_index": 1, "seed": 0}},
    "crystal_generation": {"SCIML-CRYSTAL-DIFFUSION-07R2": {"atom_count": 32, "diffusion_steps": 100, "guidance_scale": 2.0, "fixture_index": 0, "seed": 0}, "SCIML-CRYSTAL-HIGH-GUIDANCE-28": {"atom_count": 32, "diffusion_steps": 100, "guidance_scale": 4.0, "fixture_index": 1, "seed": 0}},
    "crystal_sampling": {"SCIML-GRAPH-REBUILD-08R2": {"neighbor_count": 12, "sample_count": 24, "geometry_variation": 0.7, "fixture_index": 0, "seed": 0}, "SCIML-CRYSTAL-STATIC-SAMPLING-29R2": {"neighbor_count": 8, "sample_count": 32, "geometry_variation": 0.2, "fixture_index": 1, "seed": 0}},
    "episode": {"EVOL-EPISODE-POISON-10": {"runtime_version": "A", "context_width": 2, "drift_rate": 0.05, "fixture_index": 0, "seed": 0}, "EVOL-COMPILER-DRIFT-20": {"runtime_version": "B", "context_width": 3, "drift_rate": 0.25, "fixture_index": 1, "seed": 0}, "EVOL-EQUIVARIANT-SPECIALIZE-30": {"runtime_version": "A", "context_width": 4, "drift_rate": 0.3, "fixture_index": 2, "seed": 0}},
}

_CANONICAL_SPECS: list[FamilySpec] = []
_PREDICATE_FEATURES: dict[str, tuple[Mapping[str, str], ...]] = {
    "compile": (
        {"path": "workload.logical_steps", "type": "numeric"},
        {"path": "workload.dynamic_shape_rate", "type": "numeric"},
        {"path": "workload.graph_size", "type": "numeric"},
    ),
    "graph_cache": (
        {"path": "workload.geometry_displacement", "type": "numeric"},
        {"path": "workload.dynamic_rate", "type": "numeric"},
        {"path": "workload.graph_size", "type": "numeric"},
    ),
    "h2d_pipeline": (
        {"path": "workload.worker_count", "type": "numeric"},
        {"path": "workload.pin_memory", "type": "categorical"},
        {"path": "workload.batch_size", "type": "numeric"},
    ),
    "checkpoint": (
        {"path": "workload.memory_pressure", "type": "numeric"},
        {"path": "workload.segment_count", "type": "numeric"},
    ),
    "scalar_sync": (
        {"path": "workload.scalar_syncs_per_step", "type": "numeric"},
        {"path": "workload.metric_cadence", "type": "numeric"},
    ),
    "repeated_compute": (
        {"path": "workload.repeat_count", "type": "numeric"},
        {"path": "workload.backbone_width", "type": "numeric"},
        {"path": "workload.batch_size", "type": "numeric"},
    ),
    "autograd": (
        {"path": "workload.output_count", "type": "numeric"},
        {"path": "workload.input_dim", "type": "numeric"},
        {"path": "workload.jacobian_density", "type": "numeric"},
    ),
    "equivariant_head": (
        {"path": "workload.irrep_order", "type": "numeric"},
        {"path": "workload.node_count", "type": "numeric"},
        {"path": "workload.recompute_rate", "type": "numeric"},
    ),
    "crystal_generation": (
        {"path": "workload.atom_count", "type": "numeric"},
        {"path": "workload.diffusion_steps", "type": "numeric"},
        {"path": "workload.guidance_scale", "type": "numeric"},
    ),
    "crystal_sampling": (
        {"path": "workload.neighbor_count", "type": "numeric"},
        {"path": "workload.sample_count", "type": "numeric"},
        {"path": "workload.geometry_variation", "type": "numeric"},
    ),
    "episode": (
        {"path": "workload.runtime_version", "type": "categorical"},
        {"path": "workload.context_width", "type": "numeric"},
        {"path": "workload.drift_rate", "type": "numeric"},
    ),
}
_THRESHOLDS: dict[str, dict[str, tuple[float, ...]]] = {
    "compile": {"workload.logical_steps": (64.0, 128.0, 192.0), "workload.dynamic_shape_rate": (0.0, 0.2, 0.4, 0.6), "workload.graph_size": (64.0, 128.0, 256.0, 320.0)},
    "graph_cache": {"workload.geometry_displacement": (0.02, 0.05, 0.08), "workload.dynamic_rate": (0.2, 0.6), "workload.graph_size": (64.0, 128.0, 256.0)},
    "h2d_pipeline": {"workload.worker_count": (2.0, 4.0, 6.0), "workload.batch_size": (32.0, 64.0, 4096.0), "workload.prefetch_factor": (2.0, 4.0)},
    "checkpoint": {"workload.memory_pressure": (0.4, 0.57, 0.7), "workload.segment_count": (4.0, 6.0), "workload.recompute_ratio": (0.2, 0.5)},
    "scalar_sync": {"workload.scalar_syncs_per_step": (4.0, 8.0, 12.0), "workload.metric_cadence": (4.0, 8.0, 12.0)},
    "repeated_compute": {"workload.repeat_count": (2.0, 4.0, 8.0), "workload.backbone_width": (128.0, 256.0, 512.0), "workload.batch_size": (16.0, 32.0, 64.0)},
    "autograd": {"workload.output_count": (2.0, 8.0, 16.0), "workload.input_dim": (64.0, 128.0, 256.0), "workload.jacobian_density": (0.25, 0.5, 0.75)},
    "equivariant_head": {"workload.irrep_order": (1.0, 2.0, 3.0), "workload.node_count": (32.0, 64.0, 128.0), "workload.recompute_rate": (0.25, 0.5, 0.75)},
    "crystal_generation": {"workload.atom_count": (16.0, 32.0, 64.0), "workload.diffusion_steps": (50.0, 100.0, 200.0), "workload.guidance_scale": (1.0, 2.0, 4.0)},
    "crystal_sampling": {"workload.neighbor_count": (8.0, 16.0, 32.0), "workload.sample_count": (16.0, 32.0, 64.0), "workload.geometry_variation": (0.1, 0.3, 0.6)},
    "episode": {"workload.context_width": (2.0, 4.0, 8.0), "workload.drift_rate": (0.1, 0.3, 0.6)},
}
_SCIENTIFIC_INVARIANTS: dict[str, tuple[str, ...]] = {
    "compile": ("compile_correctness",),
    "graph_cache": ("energy_force_consistency",),
    "h2d_pipeline": ("batch_semantics_preserved",),
    "checkpoint": ("gradient_equivalence",),
    "scalar_sync": ("metric_semantics_preserved",),
    "repeated_compute": ("output_equivalence",),
    "autograd": ("gradient_equivalence",),
    "equivariant_head": ("equivariance_preserved",),
    "crystal_generation": ("structure_validity",),
    "crystal_sampling": ("neighbor_consistency",),
    "episode": ("state_transition_valid",),
}
_ACTION_POLICY_GATES: dict[str, tuple[str, ...]] = {
    "CONTRACT-COMPILER-CACHE": ("compile_correctness",),
    "CONTRACT-COMPILER-FUSION": ("finite_output", "output_moment_match"),
    "CONTRACT-ENERGY-FORCE": ("energy_force_consistency",),
    "CONTRACT-DATA-PIPELINE": ("batch_semantics_preserved",),
    "CONTRACT-AUTOGRAD-GRAPH": ("gradient_equivalence",),
    "CONTRACT-TRAINING-LOOP": ("metric_semantics_preserved",),
    "CONTRACT-REPEATED-COMPUTE": ("output_equivalence",),
    "CONTRACT-EQUIVARIANCE": ("equivariance_preserved",),
    "CONTRACT-CRYSTAL-VALIDITY": ("structure_validity",),
    "CONTRACT-EVOLUTION-GOVERNANCE": ("state_transition_valid",),
}
_ACTION_SPECS: dict[str, dict[str, Mapping[str, Any]]] = {
    "compile": {
        # Legacy cache actions remain only in action_policy as migration
        # labels; they have no calibrated anchor and are not deployable.
        "stabilize_dynamic_guards": {"family": "compile", "mechanism": "compile_dynamic_shapes", "applicability": {"all": [{"compare": {"workload.logical_steps": {"gte": 128}}}, {"compare": {"workload.graph_size": {"lt": 320}}}, {"compare": {"workload.dynamic_shape_rate": {"gt": 0.0}}}, {"compare": {"workload.dynamic_shape_rate": {"lte": 0.4}}}]}, "risk_class": "review", "scientific_policy_ref": "CONTRACT-COMPILER-CACHE", "activation_validator": "compile_dynamic_guard_stability", "realization_interface": "source_patch"},
        "remove_compile_graph_break": {"family": "compile", "mechanism": "compile_graph_break", "applicability": {"all": [{"compare": {"workload.logical_steps": {"gte": 128}}}, {"compare": {"workload.graph_size": {"gte": 128}}}, {"compare": {"workload.graph_size": {"lt": 320}}}, {"equals": {"workload.dynamic_shape_rate": 0.0}}, {"any": [{"compare": {"workload.evidence.graph_break_count": {"gt": 0}}}, {"equals": {"workload.dynamic_shape_rate": 0.0}}]}]}, "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-COMPILER-CACHE", "activation_validator": "compile_graph_break_removed", "realization_interface": "source_patch"},
        "bypass_compile": {"family": "compile", "mechanism": "compile_tiny_graphs", "applicability": {"any": [{"equals": {"workload.evidence.compile_worthiness": False}}, {"compare": {"workload.logical_steps": {"lt": 128}}}]}, "effect": 0.60, "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-COMPILER-CACHE", "activation_validator": "", "realization_interface": "source_patch"},
        "fuse_pointwise_chain": {"family": "compile", "mechanism": "launch_fragmentation", "applicability": {"all": [{"compare": {"workload.logical_steps": {"gte": 128}}}, {"compare": {"workload.graph_size": {"gte": 320}}}]}, "risk_class": "review", "scientific_policy_ref": "CONTRACT-COMPILER-FUSION", "activation_validator": "kernel_fusion_operator_trace", "realization_interface": "source_patch"},
    },
    "graph_cache": {
        "reuse_graph_cache": {"family": "graph_cache", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-ENERGY-FORCE", "activation_validator": "graph_cache_hit_without_rebuild", "realization_interface": "source_patch"},
        "rebuild_graph_cache": {"family": "graph_cache", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-ENERGY-FORCE", "activation_validator": "graph_cache_rebuild_trace", "realization_interface": "source_patch"},
        "reuse_neighbor_graph": {"family": "graph_cache", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-ENERGY-FORCE", "activation_validator": "neighbor_graph_reuse_trace", "realization_interface": "source_patch"},
        "batched_force_vjp": {"family": "graph_cache", "risk_class": "review", "scientific_policy_ref": "CONTRACT-ENERGY-FORCE", "activation_validator": "batched_force_vjp_trace", "realization_interface": "source_patch"},
    },
    "h2d_pipeline": {
        "pin_memory_pipeline": {"family": "h2d_pipeline", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-DATA-PIPELINE", "activation_validator": "h2d_pin_nonblocking_trace", "realization_interface": "source_patch"},
        "prefetch_pipeline": {"family": "h2d_pipeline", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-DATA-PIPELINE", "activation_validator": "h2d_prefetch_fanout_trace", "realization_interface": "source_patch"},
    },
    "checkpoint": {
        "checkpoint_recompute": {"family": "checkpoint", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-AUTOGRAD-GRAPH", "activation_validator": "checkpoint_recompute_trace", "realization_interface": "source_patch"},
        "retained_graph": {"family": "checkpoint", "risk_class": "review", "scientific_policy_ref": "CONTRACT-AUTOGRAD-GRAPH", "activation_validator": "retained_graph_trace", "realization_interface": "source_patch"},
    },
    "scalar_sync": {
        "aggregate_scalars": {"family": "scalar_sync", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-TRAINING-LOOP", "activation_validator": "scalar_aggregation_sync_count", "realization_interface": "source_patch"},
        "defer_scalar_sync": {"family": "scalar_sync", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-TRAINING-LOOP", "activation_validator": "scalar_deferred_sync_count", "realization_interface": "source_patch"},
    },
    "repeated_compute": {"reuse_backbone": {"family": "repeated_compute", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-REPEATED-COMPUTE", "activation_validator": "backbone_reuse_trace", "realization_interface": "source_patch"}},
    "autograd": {"batched_vjp": {"family": "autograd", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-AUTOGRAD-GRAPH", "activation_validator": "batched_vjp_trace", "realization_interface": "source_patch"}},
    "equivariant_head": {"equivariant_recompute": {"family": "equivariant_head", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-EQUIVARIANCE", "activation_validator": "equivariant_path_trace", "realization_interface": "source_patch"}},
    "crystal_generation": {"sampling": {"family": "crystal_generation", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-CRYSTAL-VALIDITY", "activation_validator": "crystal_sampling_trace", "realization_interface": "source_patch"}},
    "crystal_sampling": {"graph_rebuild": {"family": "crystal_sampling", "risk_class": "bounded", "scientific_policy_ref": "CONTRACT-CRYSTAL-VALIDITY", "activation_validator": "graph_rebuild_trace", "realization_interface": "source_patch"}},
    "episode": {"rule_update": {"family": "episode", "risk_class": "review", "scientific_policy_ref": "CONTRACT-EVOLUTION-GOVERNANCE", "activation_validator": "episode_transition_trace", "realization_interface": "source_patch"}},
}
_OUTCOME_MODELS: dict[str, Mapping[str, float]] = {
    "compile": {"baseline": 0.60, "preferred": 0.80, "mismatch": 0.35, "poison_penalty": 0.20},
    "graph_cache": {"baseline": 0.60, "preferred": 0.78, "mismatch": 0.35, "poison_penalty": 0.20},
    # A wrongly deployed action has a measurable, bounded cost.  Keeping
    # mismatch equal to baseline would make a non-applicable action
    # statistically indistinguishable from the control and leave Boundary
    # CEGIS without a certifiable counterexample.
    "h2d_pipeline": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "checkpoint": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "scalar_sync": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "repeated_compute": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "autograd": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "equivariant_head": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "crystal_generation": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "crystal_sampling": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
    "episode": {"baseline": 0.58, "preferred": 0.76, "mismatch": 0.38, "poison_penalty": 0.20},
}
_ACTION_POLICIES: dict[str, Mapping[str, str]] = {
    "compile": {"default": "reuse_compile_cache", "shifted": "revalidate_compile_cache", "inapplicable": ""},
    "graph_cache": {"default": "reuse_graph_cache", "shifted": "rebuild_graph_cache", "inapplicable": ""},
    "h2d_pipeline": {"default": "pin_memory_pipeline", "shifted": "prefetch_pipeline", "inapplicable": ""},
    "checkpoint": {"default": "checkpoint_recompute", "shifted": "retained_graph", "inapplicable": ""},
    "scalar_sync": {"default": "aggregate_scalars", "shifted": "defer_scalar_sync", "inapplicable": ""},
    "repeated_compute": {"default": "reuse_backbone", "shifted": "reuse_backbone", "inapplicable": ""},
    "autograd": {"default": "batched_vjp", "shifted": "batched_vjp", "inapplicable": ""},
    "equivariant_head": {"default": "equivariant_recompute", "shifted": "equivariant_recompute", "inapplicable": ""},
    "crystal_generation": {"default": "sampling", "shifted": "sampling", "inapplicable": ""},
    "crystal_sampling": {"default": "graph_rebuild", "shifted": "graph_rebuild", "inapplicable": ""},
    "episode": {"default": "rule_update", "shifted": "rule_update", "inapplicable": ""},
}
_LEGAL_COMPOSITIONS: dict[str, tuple[CompositionSpec, ...]] = {
    "compile": (CompositionSpec("compile", "h2d_pipeline"), CompositionSpec("compile", "graph_cache"), CompositionSpec("compile", "scalar_sync")),
    "h2d_pipeline": (CompositionSpec("h2d_pipeline", "compile"), CompositionSpec("h2d_pipeline", "checkpoint")),
    "graph_cache": (CompositionSpec("graph_cache", "compile"), CompositionSpec("graph_cache", "h2d_pipeline")),
    "checkpoint": (CompositionSpec("checkpoint", "compile"), CompositionSpec("checkpoint", "scalar_sync")),
    "scalar_sync": (CompositionSpec("scalar_sync", "h2d_pipeline"), CompositionSpec("scalar_sync", "compile")),
}
_family_order = [spec.family_id for spec in _SPECS]
for _index, _family_id in enumerate(_family_order):
    _next = CompositionSpec(_family_id, _family_order[(_index + 1) % len(_family_order)])
    existing = tuple(_LEGAL_COMPOSITIONS.get(_family_id, ()))
    if _next.right_family not in {item.right_family for item in existing}:
        _LEGAL_COMPOSITIONS[_family_id] = existing + (_next,)
for _spec in _SPECS:
    _CANONICAL_SPECS.append(replace(
        _spec,
        applicability=_spec.applicability,
        anchor_parameters=_EXPLICIT_ANCHORS[_spec.family_id],
        predicate_features=_PREDICATE_FEATURES.get(_spec.family_id, ()),
        threshold_universe=_THRESHOLDS.get(_spec.family_id, {}),
        scientific_invariants=_SCIENTIFIC_INVARIANTS.get(_spec.family_id, ()),
        scientific_policy={
            "policy_id": _spec.scientific_contract_id,
            "required_gates": list(_SCIENTIFIC_INVARIANTS.get(_spec.family_id, ())),
            "tolerance": {},
        },
        action_specs=_ACTION_SPECS.get(_spec.family_id, {}),
        outcome_model=_OUTCOME_MODELS.get(_spec.family_id, {
            "baseline": 0.60,
            "preferred": 0.80,
            "mismatch": 0.35,
            "poison_penalty": 0.20,
        }),
        action_policy=_ACTION_POLICIES.get(_spec.family_id, {}),
        legal_compositions=_LEGAL_COMPOSITIONS.get(_spec.family_id, ()),
    ))
_SPECS = tuple(_CANONICAL_SPECS)

FAMILY_SPECS = {spec.family_id: spec for spec in _SPECS}
_ALIASES = {
    "compiler": "compile", "GEN-COMPILER": "compile", "GEN-COMPILER-DYNAMIC": "compile", "GEN-COMPILER-TINY": "compile",
    "h2d": "h2d_pipeline", "graph-cache": "graph_cache", "scalar-sync": "scalar_sync",
    "graph_energy_force": "graph_cache", "GEN-GNN-STATIC": "graph_cache", "GEN-GNN-DYNAMIC": "graph_cache", "GEN-GRAPH_ENERGY_FORCE": "graph_cache", "GEN-GNN-FORCE": "graph_cache",
    "data_pipeline": "h2d_pipeline", "GEN-DATA_PIPELINE": "h2d_pipeline", "GEN-DATA-FANOUT": "h2d_pipeline",
    "memory": "checkpoint", "GEN-MEMORY-CHECKPOINT": "checkpoint", "GEN-MEMORY-RETAINED": "checkpoint",
    "training_loop_overhead": "scalar_sync", "GEN-TRAINING_LOOP_OVERHEAD": "scalar_sync",
    "repeated_compute": "repeated_compute", "GEN-REPEATED_COMPUTE": "repeated_compute",
    "autograd": "autograd", "GEN-AUTOGRAD-VJP": "autograd",
    "equivariant_head": "equivariant_head", "GEN-EQUIVARIANT_HEAD": "equivariant_head",
    "crystal_generation": "crystal_generation", "GEN-CRYSTAL_GENERATION": "crystal_generation",
    "crystal_sampling": "crystal_sampling", "GEN-CRYSTAL_SAMPLING": "crystal_sampling",
    "episode": "episode", "GEN-EPISODE": "episode", "GEN-EVOLUTION-DRIFT": "episode",
}


def resolve_family_id(value: str) -> str:
    key = str(value)
    resolved = _ALIASES.get(key, key)
    if resolved not in FAMILY_SPECS:
        raise KeyError(f"unknown workload family: {value}")
    return resolved


def family_instances(family_id: str, *, count: int, seed: int = 0) -> list[FamilyInstance]:
    return FAMILY_SPECS[resolve_family_id(family_id)].generate(count, seed)


_SURFACE_CACHE: dict[tuple[str, int], tuple[FamilySurfaceSpec, tuple[FamilyInstance, ...]]] = {}


def family_surface(family_id: str, *, seed: int = 0) -> tuple[FamilySurfaceSpec, tuple[FamilyInstance, ...]]:
    """Return the one frozen lattice and its disjoint view membership."""
    resolved = resolve_family_id(family_id)
    key = (resolved, int(seed))
    if key not in _SURFACE_CACHE:
        # 108 promotion contexts are the minimum needed by the preregistered
        # p_min=0.8, delta_mix=0.0125 mixture gate.  Keep all three views
        # equal-sized and frozen so every campaign has enough independent
        # groups without borrowing from synthesis or validation.
        instances = tuple(family_instances(resolved, count=324, seed=int(seed)))
        surface = FamilySurfaceSpec(
            decision_lattice_id=f"{resolved}-lattice-{int(seed):04d}",
            synthesis_ids=tuple(item.instance_id for item in instances[108:216]),
            promotion_ids=tuple(item.instance_id for item in instances[:108]),
            validation_ids=tuple(item.instance_id for item in instances[216:]),
        )
        _SURFACE_CACHE[key] = (surface, instances)
    return _SURFACE_CACHE[key]


def family_decision_lattice(family_id: str, *, seed: int = 0, count: int | None = 264) -> list[dict[str, Any]]:
    """Canonical CEGIS lattice; all other views are disjoint partitions."""
    _surface, instances = family_surface(family_id, seed=seed)
    selected = instances if count is None else instances[: int(count)]
    return [{"workload": dict(item.parameters), "context_id": item.instance_id} for item in selected]


def family_predicate_grammar(family_id: str) -> dict[str, Any]:
    """Return the family-owned typed grammar used by harness CEGIS."""
    spec = FAMILY_SPECS[resolve_family_id(family_id)]
    if not spec.predicate_features:
        return {}
    return {
        "schema_version": 1,
        "features": [dict(feature) for feature in spec.predicate_features],
        "max_depth": 3,
        "max_literals": 4,
        "threshold_universe": {key: list(values) for key, values in spec.threshold_universe.items()},
        "formal": bool(spec.formal_predicate_grammar),
        "public_features": list(spec.public_features or spec.parameter_space),
        "feature_domains": dict(spec.feature_domains),
    }


def family_views(family_id: str, *, count: int = 24, seed: int = 0) -> dict[str, list[FamilyInstance]]:
    if count < 6:
        raise ValueError("count must be at least 6 to form three disjoint pools")
    surface, instances = family_surface(family_id, seed=seed)
    per_pool = min(count // 3, len(surface.promotion_ids))
    sealed_size = min(count - 2 * per_pool, len(surface.validation_ids))
    by_id = {item.instance_id: item for item in instances}
    return {
        "representative_pool": [by_id[item] for item in surface.promotion_ids[:per_pool]],
        "active_query_pool": [by_id[item] for item in surface.synthesis_ids[:per_pool]],
        "sealed_boundary_pool": [by_id[item] for item in surface.validation_ids[:sealed_size]],
    }


def transformation(family_id: str, kind: str, **parameters: Any) -> FamilyTransformation:
    spec = FAMILY_SPECS[resolve_family_id(family_id)]
    if kind not in spec.transformations:
        raise ValueError(f"{kind!r} is not a legal transformation for {spec.family_id}")
    return FamilyTransformation(name=f"{spec.family_id}:{kind}", kind=kind, parameters=dict(parameters))


def poisoning_transformation(family_id: str, operator: str, **parameters: Any) -> FamilyTransformation:
    """Describe an evidence-graph attack without exposing hidden truth labels."""
    allowed = {"duplicate_provenance", "small_error_mass", "overbroad_rule", "context_backdoor", "colluding_sources", "interaction_mismatch"}
    if operator not in allowed:
        raise ValueError(f"unknown poisoning operator: {operator}")
    resolved = resolve_family_id(family_id)
    return FamilyTransformation(
        name=f"{resolved}:poison:{operator}",
        kind="poison",
        parameters={"family_id": resolved, "operator": operator, **parameters},
    )


def anchor_projection(task_id: str, family_value: str) -> FamilyInstance:
    family_id = resolve_family_id(family_value)
    return FAMILY_SPECS[family_id].reconstruct_anchor(task_id)


def reconstruct_anchor_instance(task_id: str, family_id: str | None = None) -> FamilyInstance:
    """Rebuild any of the 30 materialized anchors from its FamilySpec."""
    if family_id is not None:
        return anchor_projection(task_id, family_id)
    matches = [spec for spec in FAMILY_SPECS.values() if task_id in spec.anchors]
    if len(matches) != 1:
        raise ValueError(f"anchor must belong to exactly one family: {task_id}")
    return matches[0].reconstruct_anchor(task_id)


def all_anchor_instances() -> list[FamilyInstance]:
    return [spec.reconstruct_anchor(anchor) for spec in FAMILY_SPECS.values() for anchor in spec.anchors]
