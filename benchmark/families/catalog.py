"""Canonical workload families and deterministic instance projections.

The catalog is intentionally small: it describes the parameter axes and
interventions shared by task, boundary, interaction, and evolution views.
Task workspaces remain the executable anchors; this module does not duplicate
their benchmark or oracle implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class FamilyTransformation:
    """A legal change of regime applied between sequential episode phases."""

    name: str
    kind: str
    parameters: Mapping[str, Any]


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "parameters": dict(self.parameters),
            "polarity": self.polarity,
            "difficulty_tier": self.difficulty_tier,
            "anchor_task_id": self.anchor_task_id,
            "lineage": dict(self.lineage or {}),
        }


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

    def generate(self, count: int, seed: int = 0) -> list[FamilyInstance]:
        if count < 1:
            raise ValueError("count must be positive")
        result: list[FamilyInstance] = []
        for index in range(count):
            params = dict(self.generator(index, seed))
            polarity = str(params.pop("polarity", "positive"))
            difficulty = str(params.pop("difficulty_tier", "medium"))
            result.append(
                FamilyInstance(
                    family_id=self.family_id,
                    instance_id=f"{self.family_id}-{seed:04d}-{index:04d}",
                    parameters=params,
                    polarity=polarity,
                    difficulty_tier=difficulty,
                    lineage={"generator_family_id": self.generator_family_id, "seed": seed, "index": index},
                )
            )
        return result


def _compile(index: int, seed: int) -> Mapping[str, Any]:
    return {
        "logical_steps": 32 + ((index + seed) % 16) * 32,
        "graph_size": 32 + ((index * 3 + seed) % 10) * 32,
        "dynamic_shape_rate": ((index + seed) % 8) / 10.0,
        "polarity": "positive" if ((index + seed) % 3) else "counterexample",
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _graph_cache(index: int, seed: int) -> Mapping[str, Any]:
    displacement = 0.005 + ((index + seed) % 40) * 0.003
    return {
        "geometry_displacement": displacement,
        "skin": 0.2 + ((index * 2 + seed) % 7) * 0.1,
        "graph_size": 32 + ((index + seed) % 10) * 32,
        "dynamic_rate": ((index + seed) % 8) / 10.0,
        "polarity": "positive" if displacement <= 0.05 else "counterexample",
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _h2d(index: int, seed: int) -> Mapping[str, Any]:
    return {
        "batch_size": 16 + ((index + seed) % 8) * 16,
        "worker_count": (index + seed) % 9,
        "prefetch_factor": 2 + ((index + seed) % 4),
        "pin_memory": bool((index + seed) % 2),
        "polarity": "positive" if ((index + seed) % 4) != 0 else "counterexample",
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _checkpoint(index: int, seed: int) -> Mapping[str, Any]:
    pressure = 0.25 + ((index + seed) % 10) * 0.08
    return {
        "memory_pressure": pressure,
        "segment_count": 2 + ((index + seed) % 8),
        "recompute_ratio": 0.1 + ((index + seed) % 6) * 0.1,
        "polarity": "positive" if pressure >= 0.57 else "counterexample",
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _scalar_sync(index: int, seed: int) -> Mapping[str, Any]:
    syncs = 1 + ((index + seed) % 32)
    return {
        "scalar_syncs_per_step": syncs,
        "metric_cadence": 1 + ((index * 3 + seed) % 16),
        "polarity": "positive" if 5 <= syncs <= 10 else "counterexample",
        "difficulty_tier": ("easy", "medium", "hard")[(index + seed) % 3],
    }


def _legacy(index: int, seed: int) -> Mapping[str, Any]:
    return {"fixture_index": index, "seed": seed, "difficulty_tier": "medium", "polarity": "positive"}


_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec("compile", "GEN-COMPILER", "spe_core", "CONTRACT-COMPILER-CACHE", ("logical_steps", "graph_size", "dynamic_shape_rate"), ("compile", "checkpoint"), ("software", "hardware", "scale", "model"), _compile, ("CORE-COMPILE-RECOMPILE-04", "CORE-COMPILE-DYNAMIC-11", "CORE-COMPILE-TINY-12", "CORE-KERNEL-FUSION-09")),
    FamilySpec("graph_cache", "GEN-GRAPH-CACHE", "sciml", "CONTRACT-ENERGY-FORCE", ("geometry_displacement", "skin", "graph_size", "dynamic_rate"), ("graph_cache", "geometry_motion"), ("software", "scale", "scientific_regime"), _graph_cache, ("SCIML-GNN-RAGGED-05", "SCIML-GNN-STATIC-GRAPH-CACHE-17", "SCIML-GNN-DYNAMIC-GRAPH-18", "SCIML-FORCE-AUTOGRAD-19")),
    FamilySpec("h2d_pipeline", "GEN-H2D-PIPELINE", "spe_core", "CONTRACT-DATA-PIPELINE", ("batch_size", "worker_count", "prefetch_factor", "pin_memory"), ("pin_memory", "non_blocking", "prefetch"), ("hardware", "scale", "harness"), _h2d, ("CORE-H2D-PIPELINE-03", "CORE-DATALOADER-FANOUT-16")),
    FamilySpec("checkpoint", "GEN-CHECKPOINT", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("memory_pressure", "segment_count", "recompute_ratio"), ("checkpoint", "gradient_accumulation"), ("scale", "hardware", "scientific_regime"), _checkpoint, ("CORE-MEM-RETAINED-GRAPH-13", "CORE-CHECKPOINT-AMPLE-MEM-14")),
    FamilySpec("scalar_sync", "GEN-SCALAR-SYNC", "spe_core", "CONTRACT-TRAINING-LOOP", ("scalar_syncs_per_step", "metric_cadence"), ("metric_aggregation", "device_sync"), ("scale", "harness"), _scalar_sync, ("CORE-SCALAR-SYNC-01",)),
    FamilySpec("repeated_compute", "GEN-REPEATED_COMPUTE", "spe_core", "CONTRACT-REPEATED-COMPUTE", ("fixture_index", "seed"), ("backbone_reuse",), ("scale",), _legacy, ("CORE-REPEATED-BACKBONE-02",)),
    FamilySpec("autograd", "GEN-AUTOGRAD-VJP", "spe_core", "CONTRACT-AUTOGRAD-GRAPH", ("fixture_index", "seed"), ("batched_vjp",), ("scale",), _legacy, ("CORE-AUTOGRAD-BATCHED-VJP-15",)),
    FamilySpec("equivariant_head", "GEN-EQUIVARIANT_HEAD", "sciml", "CONTRACT-EQUIVARIANCE", ("fixture_index", "seed"), ("equivariant_recompute",), ("scientific_regime",), _legacy, ("SCIML-EQUIV-RECOMPUTE-06",)),
    FamilySpec("crystal_generation", "GEN-CRYSTAL_GENERATION", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("fixture_index", "seed"), ("sampling",), ("scale", "scientific_regime"), _legacy, ("SCIML-CRYSTAL-DIFFUSION-07",)),
    FamilySpec("crystal_sampling", "GEN-CRYSTAL_SAMPLING", "sciml", "CONTRACT-CRYSTAL-VALIDITY", ("fixture_index", "seed"), ("graph_rebuild",), ("scale", "scientific_regime"), _legacy, ("SCIML-GRAPH-REBUILD-08",)),
    FamilySpec("episode", "GEN-EPISODE", "evolution", "CONTRACT-EVOLUTION-GOVERNANCE", ("fixture_index", "seed"), ("rule_update",), ("software", "hardware", "scale", "scientific_regime", "harness"), _legacy, ("EVOL-EPISODE-POISON-10", "EVOL-COMPILER-DRIFT-20")),
)

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
    spec = FAMILY_SPECS[family_id]
    if task_id not in spec.anchors:
        raise ValueError(f"{task_id} is not a canonical anchor of family {family_id}")
    return FamilyInstance(family_id, task_id, {}, anchor_task_id=task_id, lineage={"role": "anchor"})
