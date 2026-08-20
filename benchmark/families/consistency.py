"""Cross-view checks for the canonical Family workload source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml

from .catalog import FAMILY_SPECS, family_instances, family_views, resolve_family_id, transformation, poisoning_transformation, family_instance_digest


PILOT_FAMILIES = tuple(sorted(FAMILY_SPECS))
_CONTRACT_ALIASES = {
    "CONTRACT-COMPILER": "CONTRACT-COMPILER-CACHE",
    "CONTRACT-DATA-PIPELINE": "CONTRACT-DATA-PIPELINE",
    "CONTRACT-DATA_PIPELINE": "CONTRACT-DATA-PIPELINE",
    "CONTRACT-DATA-ORDER": "CONTRACT-DATA-PIPELINE",
    "CONTRACT-TRAINING_LOOP_OVERHEAD": "CONTRACT-TRAINING-LOOP",
    "CONTRACT-REPEATED_COMPUTE": "CONTRACT-REPEATED-COMPUTE",
    "CONTRACT-GRAPH_ENERGY_FORCE": "CONTRACT-ENERGY-FORCE",
    "CONTRACT-EQUIVARIANT_HEAD": "CONTRACT-EQUIVARIANCE",
    "CONTRACT-CRYSTAL_GENERATION": "CONTRACT-CRYSTAL-VALIDITY",
    "CONTRACT-CRYSTAL_SAMPLING": "CONTRACT-CRYSTAL-VALIDITY",
}


def validate_cross_view_consistency(
    *, tasks_root: str | Path | None = None, surface_count: int = 100, seed: int = 0
) -> dict[str, Any]:
    """Validate that task, boundary, interaction, and evolution views share lineage.

    This is a structural validator: it checks that all views are derived from
    the same FamilySpec and that applicability/scientific truth are not
    redefined by a view-specific generator.
    """
    if surface_count < 6:
        raise ValueError("surface_count must be at least 6")
    errors: list[str] = []
    family_report: dict[str, Any] = {}
    reconstructed_anchors: list[str] = []
    for family_id, spec in FAMILY_SPECS.items():
        for anchor_id in spec.anchors:
            anchor = spec.reconstruct_anchor(anchor_id)
            if anchor.instance_id != anchor_id or anchor.anchor_task_id != anchor_id:
                errors.append(f"{family_id}: anchor reconstruction failed for {anchor_id}")
            if bool((anchor.scientific_truth or {}).get("applicable")) != anchor.applicable:
                errors.append(f"{family_id}: anchor truth mismatch for {anchor_id}")
            reconstructed_anchors.append(anchor_id)
    for family_id in PILOT_FAMILIES:
        spec = FAMILY_SPECS[family_id]
        views = family_views(family_id, count=surface_count, seed=seed)
        instances = [item for pool in views.values() for item in pool]
        by_id = {item.instance_id: item for item in instances}
        view_ids: set[str] = set()
        for view_name, pool in views.items():
            for item in pool:
                if item.instance_id not in by_id:
                    errors.append(f"{family_id}: {view_name} contains unknown instance {item.instance_id}")
                view_ids.add(item.instance_id)
                truth = dict(item.scientific_truth or {})
                if bool(truth.get("applicable")) != bool(item.applicable):
                    errors.append(f"{family_id}: instance truth mismatch for {item.instance_id}")
        if len(view_ids) != surface_count:
            errors.append(f"{family_id}: views do not partition {surface_count} instances")
        family_report[family_id] = {
            "surface_count": surface_count,
            "pool_sizes": {name: len(pool) for name, pool in views.items()},
            "applicable_count": sum(item.applicable for item in instances),
            "scientific_contract_id": spec.scientific_contract_id,
        }

    # Interaction views are generated from the same instance IDs.  Pair each
    # pilot family with a different pilot family so every family participates.
    from benchmark.interaction.factorial_bench import generate_family_interaction_surface

    for family_id in PILOT_FAMILIES:
        spec = FAMILY_SPECS[family_id]
        partner = spec.legal_compositions[0].right_family if spec.legal_compositions else family_id
        surfaces = generate_family_interaction_surface((family_id, partner), count=min(surface_count, 128), seed=seed)
        left = {item.instance_id for item in family_instances(family_id, count=len(surfaces), seed=seed)}
        right = {item.instance_id for item in family_instances(partner, count=len(surfaces), seed=seed + 1)}
        for surface in surfaces:
            ids = list(surface.get("instance_ids", []))
            if len(ids) != 2 or ids[0] not in left or ids[1] not in right:
                errors.append(f"interaction: {surface.get('surface_id')} has inconsistent family lineage")

    # Evolution transformations are validated against the FamilySpec's legal
    # transformation set; poisoning is a harness transformation, not a second
    # workload source.
    for family_id in PILOT_FAMILIES:
        spec = FAMILY_SPECS[family_id]
        for kind in spec.transformations:
            transformation(family_id, kind)
        poisoning_transformation(family_id, "overbroad_rule")

    if tasks_root is not None:
        tasks_root = Path(tasks_root)
        seen: dict[str, str] = {}
        manifest_path = tasks_root.parent / "pilot_population.json"
        active_task_ids: set[str] | None = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "active_manifest" and isinstance(manifest.get("task_ids"), list):
                    active_task_ids = {str(task_id) for task_id in manifest["task_ids"]}
            except (OSError, json.JSONDecodeError):
                active_task_ids = None
        task_dirs = (
            [tasks_root / task_id for task_id in sorted(active_task_ids)]
            if active_task_ids is not None
            else sorted(tasks_root.iterdir())
        )
        for task_dir in task_dirs:
            task_path = task_dir / "task.yaml"
            if not task_path.is_file():
                continue
            task = miniyaml.load(str(task_path))
            family_id = str(task.get("family_id", ""))
            anchor_id = str(task.get("anchor_instance_id", ""))
            if not anchor_id or family_id not in FAMILY_SPECS:
                errors.append(f"{task_dir.name}: missing canonical family anchor projection")
                continue
            if anchor_id != str(task.get("task_id")):
                errors.append(f"{task_dir.name}: anchor_instance_id must equal task_id")
            if anchor_id not in FAMILY_SPECS[family_id].anchors:
                errors.append(f"{task_dir.name}: anchor is not declared by {family_id}")
            spec = FAMILY_SPECS[family_id]
            contract_id = _CONTRACT_ALIASES.get(str(task.get("scientific_contract_id", "")), str(task.get("scientific_contract_id", "")))
            if contract_id != spec.scientific_contract_id:
                errors.append(f"{task_dir.name}: scientific contract differs from {family_id}")
            try:
                generator_family = resolve_family_id(str(task.get("generator_family_id", "")))
            except KeyError:
                generator_family = ""
            if generator_family != family_id:
                errors.append(f"{task_dir.name}: generator family differs from {family_id}")
            anchor = spec.reconstruct_anchor(anchor_id)
            if task.get("family_parameters") != dict(anchor.parameters):
                errors.append(f"{task_dir.name}: family_parameters do not match canonical anchor")
            if str(task.get("family_instance_digest", "")) != family_instance_digest(family_id, anchor.parameters):
                errors.append(f"{task_dir.name}: family_instance_digest does not match canonical anchor")
            expected_kind = "positive" if anchor.applicable else "counterexample"
            if str(task.get("kind")) not in {expected_kind, "do_not_apply" if not anchor.applicable else expected_kind}:
                errors.append(f"{task_dir.name}: task polarity differs from FamilySpec applicability")
            if anchor_id in seen and seen[anchor_id] != family_id:
                errors.append(f"{task_dir.name}: anchor is assigned to multiple families")
            seen[anchor_id] = family_id
        declared = {anchor for spec in FAMILY_SPECS.values() for anchor in spec.anchors}
        if set(seen) != declared:
            errors.append(f"anchor population mismatch: expected {len(declared)}, found {len(seen)}")

    return {
        "schema_version": 1,
        "surface_count": surface_count,
        "families": family_report,
        "anchor_count": sum(len(spec.anchors) for spec in FAMILY_SPECS.values()),
        "reconstructed_anchor_count": len(reconstructed_anchors),
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=Path(__file__).resolve().parents[1] / "tasks")
    parser.add_argument("--surface-count", type=int, default=100)
    args = parser.parse_args()
    report = validate_cross_view_consistency(tasks_root=args.tasks_root, surface_count=args.surface_count)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
