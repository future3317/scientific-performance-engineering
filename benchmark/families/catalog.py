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
        worker_count = int(left.parameters.get("worker_count", -1))
        dynamic_rate = float(right.parameters.get("dynamic_shape_rate", -1.0))
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
        selected = self.action_policy.get(regime) or self.action_policy.get("default", "")
        if not selected or selected != action_id:
            return False
        try:
            family_ok = bool(self.applicability(parameters))
        except (KeyError, TypeError, ValueError):
            # Environment oracle construction may only have a regime context;
            # it must not manufacture a negative action solely because a
            # workload lattice point is unavailable.
            family_ok = True
        return family_ok

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


def _truth_from_polarity(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"applicable": bool(parameters.get("applicable", True)), "parameters": dict(parameters)}


def _legacy_applicability(parameters: Mapping[str, Any]) -> bool:
    return bool(parameters.get("applicable", True))


def _compile_applicability(parameters: Mapping[str, Any]) -> bool:
    return float(parameters["logical_steps"]) >= 128 and float(parameters["dynamic_shape_rate"]) <= 0.4


def _compile_truth(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    applicable = _compile_applicability(parameters)
    return {"applicable": applicable, "boundary": {"logical_steps": 128, "dynamic_shape_rate": 0.4}}


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
    FamilySpec("compile", "GEN-COMPILER", "spe_core", "CONTRACT-COMPILER-CACHE", ("logical_steps", "graph_size", "dynamic_shape_rate"), ("compile", "checkpoint"), ("software", "hardware", "scale", "model"), _compile, ("CORE-COMPILE-RECOMPILE-04", "CORE-COMPILE-DYNAMIC-11", "CORE-COMPILE-TINY-12", "CORE-KERNEL-FUSION-09"), _compile_applicability, _compile_truth, {"CORE-COMPILE-RECOMPILE-04": True, "CORE-COMPILE-DYNAMIC-11": True, "CORE-COMPILE-TINY-12": False, "CORE-KERNEL-FUSION-09": True}),
    FamilySpec("graph_cache", "GEN-GRAPH-CACHE", "sciml", "CONTRACT-ENERGY-FORCE", ("geometry_displacement", "skin", "graph_size", "dynamic_rate"), ("graph_cache", "geometry_motion"), ("software", "scale", "scientific_regime"), _graph_cache, ("SCIML-GNN-RAGGED-05", "SCIML-GNN-STATIC-GRAPH-CACHE-17", "SCIML-GNN-DYNAMIC-GRAPH-18", "SCIML-FORCE-AUTOGRAD-19"), _graph_applicability, _graph_truth, {"SCIML-GNN-RAGGED-05": True, "SCIML-GNN-STATIC-GRAPH-CACHE-17": True, "SCIML-GNN-DYNAMIC-GRAPH-18": False, "SCIML-FORCE-AUTOGRAD-19": True}),
    FamilySpec("h2d_pipeline", "GEN-H2D-PIPELINE", "spe_core", "CONTRACT-DATA-PIPELINE", ("batch_size", "worker_count", "prefetch_factor", "pin_memory"), ("pin_memory", "non_blocking", "prefetch"), ("hardware", "scale", "harness"), _h2d, ("CORE-H2D-PIPELINE-03", "CORE-DATALOADER-FANOUT-16"), _h2d_applicability, _h2d_truth, {"CORE-H2D-PIPELINE-03": True, "CORE-DATALOADER-FANOUT-16": True}),
    FamilySpec("checkpoint", "GEN-CHECKPOINT", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("memory_pressure", "segment_count", "recompute_ratio"), ("checkpoint", "gradient_accumulation"), ("scale", "hardware", "scientific_regime"), _checkpoint, ("CORE-MEM-RETAINED-GRAPH-13", "CORE-CHECKPOINT-AMPLE-MEM-14"), _checkpoint_applicability, _checkpoint_truth, {"CORE-MEM-RETAINED-GRAPH-13": True, "CORE-CHECKPOINT-AMPLE-MEM-14": False}),
    FamilySpec("scalar_sync", "GEN-SCALAR-SYNC", "spe_core", "CONTRACT-TRAINING-LOOP", ("scalar_syncs_per_step", "metric_cadence"), ("metric_aggregation", "device_sync"), ("scale", "harness"), _scalar_sync, ("CORE-SCALAR-SYNC-01",), _scalar_applicability, _scalar_truth, {"CORE-SCALAR-SYNC-01": True}),
    FamilySpec("repeated_compute", "GEN-REPEATED_COMPUTE", "spe_core", "CONTRACT-REPEATED-COMPUTE", ("fixture_index", "seed"), ("backbone_reuse",), ("scale",), _legacy, ("CORE-REPEATED-BACKBONE-02",), lambda p: True, _truth_from_polarity, {"CORE-REPEATED-BACKBONE-02": True}),
    FamilySpec("autograd", "GEN-AUTOGRAD-VJP", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("fixture_index", "seed"), ("batched_vjp",), ("scale",), _legacy, ("CORE-AUTOGRAD-BATCHED-VJP-15",), lambda p: True, _truth_from_polarity, {"CORE-AUTOGRAD-BATCHED-VJP-15": True}),
    FamilySpec("equivariant_head", "GEN-EQUIVARIANT_HEAD", "sciml", "CONTRACT-EQUIVARIANCE", ("fixture_index", "seed"), ("equivariant_recompute",), ("scientific_regime",), _legacy, ("SCIML-EQUIV-RECOMPUTE-06",), lambda p: True, _truth_from_polarity, {"SCIML-EQUIV-RECOMPUTE-06": True}),
    FamilySpec("crystal_generation", "GEN-CRYSTAL_GENERATION", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("fixture_index", "seed"), ("sampling",), ("scale", "scientific_regime"), _legacy, ("SCIML-CRYSTAL-DIFFUSION-07",), lambda p: True, _truth_from_polarity, {"SCIML-CRYSTAL-DIFFUSION-07": True}),
    FamilySpec("crystal_sampling", "GEN-CRYSTAL_SAMPLING", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("fixture_index", "seed"), ("graph_rebuild",), ("scale", "scientific_regime"), _legacy, ("SCIML-GRAPH-REBUILD-08",), lambda p: True, _truth_from_polarity, {"SCIML-GRAPH-REBUILD-08": True}),
    FamilySpec("episode", "GEN-EPISODE", "evolution", "CONTRACT-EVOLUTION-GOVERNANCE", ("fixture_index", "seed"), ("rule_update",), ("software", "hardware", "scale", "scientific_regime", "harness"), _legacy, ("EVOL-EPISODE-POISON-10", "EVOL-COMPILER-DRIFT-20"), lambda p: True, _truth_from_polarity, {"EVOL-EPISODE-POISON-10": True, "EVOL-COMPILER-DRIFT-20": True}),
)

# Explicit anchor coordinates are the canonical task truth.  The legacy
# generator-index projections above remain useful for unmaterialized pools,
# but never define an anchor's applicability.
_EXPLICIT_ANCHORS: dict[str, dict[str, Mapping[str, Any]]] = {
    "compile": {
        "CORE-COMPILE-RECOMPILE-04": {"logical_steps": 128, "graph_size": 64, "dynamic_shape_rate": 0.2},
        "CORE-COMPILE-DYNAMIC-11": {"logical_steps": 256, "graph_size": 128, "dynamic_shape_rate": 0.3},
        "CORE-COMPILE-TINY-12": {"logical_steps": 64, "graph_size": 64, "dynamic_shape_rate": 0.8},
        "CORE-KERNEL-FUSION-09": {"logical_steps": 192, "graph_size": 320, "dynamic_shape_rate": 0.2},
    },
    "graph_cache": {
        "SCIML-GNN-RAGGED-05": {"geometry_displacement": 0.02, "skin": 0.2, "graph_size": 64, "dynamic_rate": 0.0},
        "SCIML-GNN-STATIC-GRAPH-CACHE-17": {"geometry_displacement": 0.03, "skin": 0.4, "graph_size": 96, "dynamic_rate": 0.1},
        "SCIML-GNN-DYNAMIC-GRAPH-18": {"geometry_displacement": 0.08, "skin": 0.6, "graph_size": 96, "dynamic_rate": 0.8},
        "SCIML-FORCE-AUTOGRAD-19": {"geometry_displacement": 0.02, "skin": 0.8, "graph_size": 128, "dynamic_rate": 0.3},
    },
    "h2d_pipeline": {
        "CORE-H2D-PIPELINE-03": {"batch_size": 32, "worker_count": 2, "prefetch_factor": 2, "pin_memory": True},
        "CORE-DATALOADER-FANOUT-16": {"batch_size": 64, "worker_count": 3, "prefetch_factor": 4, "pin_memory": True},
    },
    "checkpoint": {
        "CORE-MEM-RETAINED-GRAPH-13": {"memory_pressure": 0.7, "segment_count": 4, "recompute_ratio": 0.2},
        "CORE-CHECKPOINT-AMPLE-MEM-14": {"memory_pressure": 0.4, "segment_count": 3, "recompute_ratio": 0.2},
    },
    "scalar_sync": {"CORE-SCALAR-SYNC-01": {"scalar_syncs_per_step": 12, "metric_cadence": 4}},
    "repeated_compute": {"CORE-REPEATED-BACKBONE-02": {"fixture_index": 0, "seed": 0}},
    "autograd": {"CORE-AUTOGRAD-BATCHED-VJP-15": {"fixture_index": 0, "seed": 0}},
    "equivariant_head": {"SCIML-EQUIV-RECOMPUTE-06": {"fixture_index": 0, "seed": 0, "applicable": False}},
    "crystal_generation": {"SCIML-CRYSTAL-DIFFUSION-07": {"fixture_index": 0, "seed": 0}},
    "crystal_sampling": {"SCIML-GRAPH-REBUILD-08": {"fixture_index": 0, "seed": 0, "applicable": False}},
    "episode": {"EVOL-EPISODE-POISON-10": {"fixture_index": 0, "seed": 0}, "EVOL-COMPILER-DRIFT-20": {"fixture_index": 1, "seed": 0}},
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
}
_THRESHOLDS: dict[str, dict[str, tuple[float, ...]]] = {
    "compile": {"workload.logical_steps": (64.0, 128.0, 192.0), "workload.dynamic_shape_rate": (0.2, 0.4, 0.6), "workload.graph_size": (64.0, 128.0, 256.0)},
    "graph_cache": {"workload.geometry_displacement": (0.02, 0.05, 0.08), "workload.dynamic_rate": (0.2, 0.6), "workload.graph_size": (64.0, 128.0, 256.0)},
    "h2d_pipeline": {"workload.worker_count": (2.0, 4.0, 6.0), "workload.batch_size": (32.0, 64.0), "workload.prefetch_factor": (2.0, 4.0)},
    "checkpoint": {"workload.memory_pressure": (0.4, 0.57, 0.7), "workload.segment_count": (4.0, 6.0), "workload.recompute_ratio": (0.2, 0.5)},
    "scalar_sync": {"workload.scalar_syncs_per_step": (4.0, 8.0, 12.0), "workload.metric_cadence": (4.0, 8.0, 12.0)},
}
_SCIENTIFIC_INVARIANTS: dict[str, tuple[str, ...]] = {
    "compile": ("compile_correctness",),
    "graph_cache": ("energy_force_consistency",),
    "h2d_pipeline": ("batch_semantics_preserved",),
    "checkpoint": ("gradient_equivalence",),
    "scalar_sync": ("metric_semantics_preserved",),
}
_ACTION_SPECS: dict[str, dict[str, Mapping[str, Any]]] = {
    "compile": {
        "reuse_compile_cache": {"family": "compile", "risk_class": "bounded"},
        "revalidate_compile_cache": {"family": "compile", "risk_class": "review"},
    },
    "graph_cache": {
        "reuse_graph_cache": {"family": "graph_cache", "risk_class": "bounded"},
        "rebuild_graph_cache": {"family": "graph_cache", "risk_class": "bounded"},
    },
    "h2d_pipeline": {
        "pin_memory_pipeline": {"family": "h2d_pipeline", "risk_class": "bounded"},
        "prefetch_pipeline": {"family": "h2d_pipeline", "risk_class": "bounded"},
    },
    "checkpoint": {
        "checkpoint_recompute": {"family": "checkpoint", "risk_class": "bounded"},
        "retained_graph": {"family": "checkpoint", "risk_class": "review"},
    },
    "scalar_sync": {
        "aggregate_scalars": {"family": "scalar_sync", "risk_class": "bounded"},
        "defer_scalar_sync": {"family": "scalar_sync", "risk_class": "bounded"},
    },
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
}
_ACTION_POLICIES: dict[str, Mapping[str, str]] = {
    "compile": {"default": "reuse_compile_cache", "shifted": "revalidate_compile_cache", "inapplicable": ""},
    "graph_cache": {"default": "reuse_graph_cache", "shifted": "rebuild_graph_cache", "inapplicable": ""},
    "h2d_pipeline": {"default": "pin_memory_pipeline", "shifted": "prefetch_pipeline", "inapplicable": ""},
    "checkpoint": {"default": "checkpoint_recompute", "shifted": "retained_graph", "inapplicable": ""},
    "scalar_sync": {"default": "aggregate_scalars", "shifted": "defer_scalar_sync", "inapplicable": ""},
}
_LEGAL_COMPOSITIONS: dict[str, tuple[CompositionSpec, ...]] = {
    "compile": (CompositionSpec("compile", "h2d_pipeline"), CompositionSpec("compile", "graph_cache"), CompositionSpec("compile", "scalar_sync")),
    "h2d_pipeline": (CompositionSpec("h2d_pipeline", "compile"), CompositionSpec("h2d_pipeline", "checkpoint")),
    "graph_cache": (CompositionSpec("graph_cache", "compile"), CompositionSpec("graph_cache", "h2d_pipeline")),
    "checkpoint": (CompositionSpec("checkpoint", "compile"), CompositionSpec("checkpoint", "scalar_sync")),
    "scalar_sync": (CompositionSpec("scalar_sync", "h2d_pipeline"), CompositionSpec("scalar_sync", "compile")),
}
for _spec in _SPECS:
    _applicability = _legacy_applicability if _spec.family_id in {"repeated_compute", "autograd", "equivariant_head", "crystal_generation", "crystal_sampling", "episode"} else _spec.applicability
    _CANONICAL_SPECS.append(replace(
        _spec,
        applicability=_applicability,
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
    instances = family_instances(family_id, count=count, seed=seed)
    first = count // 3
    second = 2 * count // 3
    return {"representative_pool": instances[:first], "active_query_pool": instances[first:second], "sealed_boundary_pool": instances[second:]}


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
    """Rebuild any of the 20 materialized anchors from its FamilySpec."""
    if family_id is not None:
        return anchor_projection(task_id, family_id)
    matches = [spec for spec in FAMILY_SPECS.values() if task_id in spec.anchors]
    if len(matches) != 1:
        raise ValueError(f"anchor must belong to exactly one family: {task_id}")
    return matches[0].reconstruct_anchor(task_id)


def all_anchor_instances() -> list[FamilyInstance]:
    return [spec.reconstruct_anchor(anchor) for spec in FAMILY_SPECS.values() for anchor in spec.anchors]
