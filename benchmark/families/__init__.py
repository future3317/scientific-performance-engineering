"""Shared workload-family generation for all SPE-EvoBench views."""

from .catalog import (
    FamilyInstance,
    FamilySpec,
    FamilyTransformation,
    family_instances,
    family_views,
    resolve_family_id,
    transformation,
    anchor_projection,
    poisoning_transformation,
    reconstruct_anchor_instance,
    all_anchor_instances,
    family_instance_digest,
    family_predicate_grammar,
    CompositionSpec,
    InteractionOracle,
)
from .environment import EnvironmentOutcome, EpisodeEnvironmentState, FamilyEnvironment

__all__ = [
    "FamilyInstance",
    "FamilySpec",
    "FamilyTransformation",
    "family_instances",
    "family_views",
    "resolve_family_id",
    "transformation",
    "anchor_projection",
    "poisoning_transformation",
    "reconstruct_anchor_instance",
    "all_anchor_instances",
    "family_instance_digest",
    "family_predicate_grammar",
    "CompositionSpec",
    "InteractionOracle",
    "EnvironmentOutcome", "EpisodeEnvironmentState", "FamilyEnvironment",
    "PILOT_FAMILIES",
    "validate_cross_view_consistency",
]


def __getattr__(name: str):
    if name in {"PILOT_FAMILIES", "validate_cross_view_consistency"}:
        from .consistency import PILOT_FAMILIES, validate_cross_view_consistency

        return {"PILOT_FAMILIES": PILOT_FAMILIES, "validate_cross_view_consistency": validate_cross_view_consistency}[name]
    raise AttributeError(name)
