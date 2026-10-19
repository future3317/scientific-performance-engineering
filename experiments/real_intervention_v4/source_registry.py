"""Frozen architecture-family population for the channels-last study."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any


def family_lineage_id(candidate: dict[str, Any]) -> str | None:
    """Return the mapped architecture lineage; unmapped candidates stay ungrouped."""
    value = candidate.get("lineage_id") or candidate.get("family") or candidate.get("family_id")
    if value is None or not str(value).strip():
        return None
    return str(value).strip().casefold()


def candidate_source_identity(candidate: dict[str, Any], *,
                              source_revisions: dict[str, str] | None = None) -> dict[str, Any]:
    """Describe the registered model source and its explicit source revisions."""
    source = candidate.get("source")
    model = candidate.get("model")
    if not source or not model:
        raise ValueError("registered candidate requires source and model")
    registration = {
        name: candidate[name]
        for name in (
            "availability", "eval_deterministic", "metadata_supported", "input_kind",
            "availability_detail", "installed_source_version", "input_shape",
            "pretrained", "num_classes",
        )
        if name in candidate
    }
    return {
        "candidate_id": str(candidate.get("candidate_id", f"{source}:{model}")),
        "source": str(source),
        "model": str(model),
        "family_lineage_id": family_lineage_id(candidate),
        "source_revisions": dict(sorted((candidate.get("source_revisions")
                                          or source_revisions or {}).items())),
        "registration": registration,
    }


def register_candidates(candidates: list[dict[str, Any]], *,
                        source_revisions: dict[str, str] | None = None
                        ) -> list[dict[str, Any]]:
    """Retain the full roster while assigning aliases to one architecture lineage."""
    registered = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        row = dict(candidate)
        candidate_id = str(row.get("candidate_id", f"{row.get('source')}:{row.get('model')}"))
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate identity: {candidate_id}")
        seen_ids.add(candidate_id)
        lineage = family_lineage_id(row)
        row["candidate_id"] = candidate_id
        row["family_id"] = lineage
        row["family_lineage_id"] = lineage
        revisions = {**(source_revisions or {}), **row.get("source_revisions", {})}
        row["source_revisions"] = revisions
        row["source_identity"] = candidate_source_identity(row, source_revisions=revisions)
        registered.append(row)
    return registered


def candidate_universe(torchbench_root: str | Path, *,
                       source_revisions: dict[str, str] | None = None,
                       supplemental_sources: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Use the already frozen v3 candidate pool without importing v3 outcomes."""
    from experiments.real_intervention_v3.run_all import baseline_pool
    from experiments.real_intervention_v3.source_adapters import model_universe

    rows = []
    for candidate in baseline_pool(model_universe(torchbench_root)):
        availability = str(candidate["availability"]).upper()
        rows.append({
            **candidate,
            "candidate_id": f"{candidate['source']}:{candidate['model']}",
            "availability": availability,
        })
    supplemental_sources = supplemental_sources or {}
    for source, registration in sorted(supplemental_sources.items()):
        if source != "timm":
            raise ValueError(f"unsupported supplemental model source: {source}")
        expected_version = str(registration["version"])
        try:
            installed_version = importlib.metadata.version(source)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        availability = "available" if installed_version == expected_version else "source_unavailable"
        for model in registration["models"]:
            family = str(model["family"])
            rows.append({
                "source": source, "model": str(model["model"]), "family": family,
                "candidate_id": f"{source}:{model['model']}",
                "availability": availability,
                "installed_source_version": installed_version,
                "availability_detail": (
                    None if availability == "available" else
                    f"expected {source} {expected_version}, found {installed_version or 'not installed'}"
                ),
                "input_shape": list(model["input_shape"]),
                "input_kind": "SEEDED_RANDOM_MODEL_INPUT",
                "pretrained": False, "num_classes": 1000,
                "source_revisions": {
                    "repository_commit": (source_revisions or {}).get("repository_commit", ""),
                    "timm_version": expected_version,
                    "timm_git_commit": str(registration["git_commit"]),
                },
            })
    return register_candidates(rows, source_revisions=source_revisions)


def model_spec(candidate: dict[str, Any], *, seed: int, batch: int, horizon: int,
               shape: tuple[int, int, int] = (3, 224, 224),
               torchbench_root: str | Path, reference_path: str | Path,
               correctness: dict[str, float]) -> dict[str, Any]:
    """Materialize one immutable context without observing treatment outcomes."""
    return {
        **candidate,
        "seed": seed,
        "batch": batch,
        "shape": list(shape),
        "horizon": horizon,
        "torchbench_root": str(Path(torchbench_root).resolve()),
        "reference_path": str(Path(reference_path).resolve()),
        "correctness": dict(correctness),
        "precision": "float32",
        "mode": "eval",
    }
