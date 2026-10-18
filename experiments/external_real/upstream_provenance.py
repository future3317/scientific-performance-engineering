from __future__ import annotations

import json
from pathlib import Path


_EXPECTED = {
    "sampled_trajectories": "logs/sampling",
    "extracted_action_contributions": "logs/skill_extraction",
    "skill_graph": "logs/skill_extraction",
    "embeddings": "logs/skill_extraction",
    "retrieval_artifacts": "logs/skill_extraction",
    "domain_mapping": "logs/skill_extraction",
}


def _files(root: Path, relative: str) -> list[str]:
    path = root / relative
    if not path.exists():
        return []
    return sorted(str(item.relative_to(root)) for item in path.rglob("*") if item.is_file())


def audit_provenance(source_root: Path) -> dict[str, object]:
    artifacts = {name: _files(source_root, relative) for name, relative in _EXPECTED.items()}
    present = {name: paths for name, paths in artifacts.items() if paths}
    reusable = bool(present)
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "artifact_inventory": artifacts,
        "frozen_train_derived_artifact_usable": False,
        "outcome_label_leakage_check": "not_applicable_no_candidate_artifacts" if not reusable else "requires_manual_artifact_review",
        "prospective_audit_authorized": False,
        "decision": "reuse_only_if_train_derived_and_outcome_label_free" if reusable else "enter_u8_resampling",
        "reason": (
            "No committed sampling, graph, embedding, retrieval, or domain-mapping artifacts were released."
            if not reusable
            else "Candidate artifacts exist but provenance and outcome-label separation must be checked before reuse."
        ),
    }


def write_provenance_audit(source_root: Path, output_path: Path) -> dict[str, object]:
    report = audit_provenance(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
