"""Formal release manifest and sealed-package materialization helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark.families import FAMILY_SPECS, FamilyInstance, family_instance_digest, resolve_family_id
from benchmark.harness import miniyaml
from benchmark.formal.attest import task_package_digest
from benchmark.taskgen.generate import ast_skeleton_hash, assert_executable_projection


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _mechanism_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _formal_source_candidates(tasks_root: Path, slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_manifest = tasks_root.parent / "pilot_population.json"
    if not active_manifest.is_file():
        raise ValueError(f"active population manifest missing: {active_manifest}")
    active_ids = set(json.loads(active_manifest.read_text(encoding="utf-8")).get("task_ids", []))
    if not active_ids:
        raise ValueError(f"active population manifest has no task_ids: {active_manifest}")
    requested = {
        (
            str(slot.get("family_id")),
            _mechanism_key(slot.get("mechanism")),
            str(slot.get("polarity")),
            str(slot.get("difficulty")),
        )
        for slot in slots
    }
    candidates: list[dict[str, Any]] = []
    for task_dir in sorted(tasks_root.iterdir()):
        task_path = task_dir / "task.yaml"
        metadata_path = task_dir / "metadata.json"
        if task_dir.name not in active_ids:
            continue
        if not task_dir.is_dir() or not task_path.is_file() or not metadata_path.is_file():
            continue
        spec = miniyaml.load(str(task_path))
        try:
            family_id = resolve_family_id(str(spec.get("family_id", spec.get("family", ""))))
        except (KeyError, TypeError, ValueError):
            continue
        candidate_key = (
            family_id,
            _mechanism_key(spec.get("mechanism")),
            str(spec.get("kind")),
            str(spec.get("difficulty_tier")),
        )
        if candidate_key not in requested:
            continue
        declared_parameters = {
            key: value
            for key, value in dict(spec.get("family_parameters") or {}).items()
            if key not in {"seed", "fixture_index"}
        }
        if not declared_parameters:
            continue
        try:
            truth = dict(FAMILY_SPECS[family_id].scientific_truth(declared_parameters) or {})
        except (KeyError, TypeError, ValueError):
            continue
        instance = FamilyInstance(
            family_id=family_id,
            instance_id=str(spec["task_id"]),
            parameters=dict(declared_parameters),
            polarity=str(spec.get("kind", "positive")),
            difficulty_tier=str(spec.get("difficulty_tier", "medium")),
            anchor_task_id=None,
            lineage={"source": "executable_projection", "task_id": str(spec["task_id"])},
            applicable=bool(truth.get("applicable", spec.get("kind") == "positive")),
            scientific_truth=truth,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_group = metadata.get("split_group")
        if not isinstance(split_group, dict) or not split_group:
            continue
        candidates.append({
            "task_id": str(spec["task_id"]),
            "path": task_dir,
            "family_id": family_id,
            "mechanism": spec.get("mechanism"),
            "kind": str(spec.get("kind")),
            "difficulty": str(spec.get("difficulty_tier")),
            "instance": instance,
            "projection_status": "declared_template",
        })
    return candidates


def _select_formal_source(slot: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    family_id = resolve_family_id(str(slot.get("family_id", "")))
    mechanism = _mechanism_key(slot.get("mechanism"))
    polarity = str(slot.get("polarity"))
    matches = [
        candidate for candidate in candidates
        if candidate["family_id"] == family_id
        and _mechanism_key(candidate["mechanism"]) == mechanism
        and candidate["kind"] == polarity
        and candidate["difficulty"] == str(slot.get("difficulty"))
    ]
    if not matches:
        raise ValueError(
            f"no executable source package for {slot.get('slot_id')}: "
            f"family={family_id}, mechanism={slot.get('mechanism')}, polarity={slot.get('polarity')}, "
            f"difficulty={slot.get('difficulty')}"
        )
    return matches[int(slot["seed"]) % len(matches)]


def _inject_projection_metadata(package_dir: Path, source: dict[str, Any], formal_seed: int) -> None:
    family_id = str(source["family_id"])
    parameters = dict(source["instance"].parameters)
    extras: dict[str, Any] = {}
    if family_id == "autograd" and "jacobian_density" in parameters:
        extras["jacobian_density"] = parameters["jacobian_density"]
    elif family_id == "graph_cache":
        extras["graph_cache_config"] = {
            key: parameters[key]
            for key in ("geometry_displacement", "skin", "graph_size", "dynamic_rate")
            if key in parameters
        }
    elif family_id == "checkpoint":
        extras["checkpoint_config"] = {
            key: parameters[key]
            for key in ("memory_pressure", "recompute_ratio", "segment_count")
            if key in parameters
        }
        if "logical_batch_size" in parameters:
            extras["logical_batch_size"] = parameters["logical_batch_size"]
    elif family_id == "h2d_pipeline":
        extras["data_config"] = {"batch_size": parameters.get("batch_size")}
        extras["worker_count"] = parameters.get("worker_count")
        extras["prefetch_factor"] = parameters.get("prefetch_factor")
        extras["pin_memory"] = parameters.get("pin_memory")
    elif family_id == "scalar_sync":
        extras["scalar_syncs_per_step"] = parameters.get("scalar_syncs_per_step")
        extras["metric_cadence"] = parameters.get("metric_cadence")
    elif family_id == "equivariant_head":
        extras["node_count"] = parameters.get("node_count")
        extras["irrep_order"] = parameters.get("irrep_order")
        extras["recompute_rate"] = parameters.get("recompute_rate")
    elif family_id == "crystal_sampling":
        extras["neighbor_count"] = parameters.get("neighbor_count")
        extras["geometry_variation"] = parameters.get("geometry_variation")
    elif family_id == "crystal_generation":
        extras["config"] = {
            "num_atoms": parameters.get("atom_count"),
            "num_steps": parameters.get("diffusion_steps"),
        }
        extras["guidance_scale"] = parameters.get("guidance_scale")
    benchmark_path = package_dir / "benchmark.py"
    text = benchmark_path.read_text(encoding="utf-8")
    marker = "# FORMAL_SLOT_PROJECTION_METADATA"
    if marker in text:
        return
    assignments = "\n".join(f"    fixtures[{key!r}] = {value!r}" for key, value in extras.items())
    wrapper = (
        f"\n\n{marker}\n"
        "_FORMAL_ORIGINAL_MAKE_FIXTURES = make_fixtures\n"
        f"_FORMAL_VARIANT_SEED = {int(formal_seed)}\n"
        f"_FORMAL_VARIANT_PARAMETERS = {parameters!r}\n"
        f"_FORMAL_FAMILY_ID = {family_id!r}\n"
        "def make_fixtures(seed=0, device='cpu'):\n"
        "    fixtures = _FORMAL_ORIGINAL_MAKE_FIXTURES(int(seed) + _FORMAL_VARIANT_SEED, device)\n"
        "    if isinstance(fixtures, dict):\n"
        "        fixtures['formal_variant_seed'] = _FORMAL_VARIANT_SEED\n"
        "        fixtures['formal_variant_parameters'] = dict(_FORMAL_VARIANT_PARAMETERS)\n"
        "        if _FORMAL_FAMILY_ID == 'episode':\n"
        "            public_context = fixtures.get('public_context')\n"
        "            workload = public_context.get('workload') if isinstance(public_context, dict) else None\n"
        "            if isinstance(workload, dict):\n"
        "                workload['formal_variant_seed'] = _FORMAL_VARIANT_SEED\n"
        "                workload['formal_variant_parameters'] = dict(_FORMAL_VARIANT_PARAMETERS)\n"
        f"{assignments}\n"
        "    return fixtures\n"
    )
    benchmark_path.write_text(text + wrapper, encoding="utf-8")


def _rewrite_formal_identity(package_dir: Path, slot: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    slot_id = str(slot["slot_id"])
    source_id = str(source["task_id"])
    formal_seed = int(slot["seed"])
    _inject_projection_metadata(package_dir, source, formal_seed)
    solution = package_dir / "workspace" / "solution.py"
    if solution.is_file():
        text = solution.read_text(encoding="utf-8")
        marker = f'FORMAL_SLOT_VARIANT = "{slot_id}"\n'
        if "FORMAL_SLOT_VARIANT = " not in text:
            future = "from __future__ import annotations\n"
            text = text.replace(future, future + "\n" + marker, 1) if future in text else marker + text
            solution.write_text(text, encoding="utf-8")
    spec_path = package_dir / "task.yaml"
    spec = miniyaml.load(str(spec_path))
    instance = source["instance"]
    spec.update({
        "task_id": slot_id,
        "anchor_instance_id": slot_id,
        "family_id": source["family_id"],
        "family_parameters": dict(instance.parameters),
        "family_instance_digest": family_instance_digest(source["family_id"], instance.parameters),
        "formal_variant_seed": formal_seed,
        "formal_variant_parameters": dict(instance.parameters),
        "public_context": {"workload": dict(instance.parameters)},
        "lineage": {
            "source": "sealed_generator",
            "mutation_template_id": f"FORMAL-{slot_id}",
            "mutation_seed": int(slot["seed"]),
            "formal_variant_seed": formal_seed,
            "source_task_id": source_id,
        },
        "oracle_fix_pattern_id": f"FORMAL-{slot_id}-INTERVENTION",
        "workspace_ast_skeleton_hash": ast_skeleton_hash(package_dir),
        "ast_skeleton_version": 2,
    })
    miniyaml.save(spec, str(spec_path))
    metadata_path = package_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "task_id": slot_id,
        "family_id": source["family_id"],
        "anchor_instance_id": slot_id,
        "formal_variant_seed": formal_seed,
        "formal_variant_parameters": dict(instance.parameters),
        "lineage": dict(spec["lineage"]),
        "split_group": {
            "family": source["family_id"],
            "mechanism": slot.get("mechanism"),
            "source": f"sealed-seed-{slot['seed']}",
            "mutation_template_id": f"FORMAL-{slot_id}",
        },
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "family_instance": FamilyInstance(
            family_id=source["family_id"],
            instance_id=slot_id,
            parameters=dict(instance.parameters),
            polarity=str(slot["polarity"]),
            difficulty_tier=str(slot["difficulty"]),
            lineage={"seed": formal_seed, "formal_variant_seed": formal_seed, "source_task_id": source_id},
            applicable=instance.applicable,
            scientific_truth=dict(instance.scientific_truth or {}),
        ).to_dict(),
        "split": metadata["split_group"],
    }


def materialize_frozen_slots(
    preregistered: dict[str, Any], sealed_root: str | Path, *, tasks_root: str | Path
) -> dict[str, Any]:
    """Materialize frozen slots into executable private packages.

    Each slot selects a same-family canonical executable package using its
    frozen seed, rewrites only package identity/lineage metadata, and proves
    the resulting package still projects to its declared FamilyInstance.
    """
    slots = list(preregistered.get("slots", []))
    if preregistered.get("status") not in {"preregistered_content_withheld", "materialized_frozen"}:
        raise ValueError("population manifest is not in a materializable state")
    root = Path(sealed_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = _formal_source_candidates(Path(tasks_root).resolve(), slots)
    materialized: list[dict[str, Any]] = []
    for slot in slots:
        item = dict(slot)
        package_dir = root / str(item["slot_id"])
        source_record = _select_formal_source(item, candidates)
        source = Path(source_record["path"]).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if package_dir.exists():
            shutil.rmtree(package_dir)
        shutil.copytree(source, package_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        identity = _rewrite_formal_identity(package_dir, item, source_record)
        projection = assert_executable_projection(package_dir)
        item["task_id"] = str(package_dir.name)
        item["source_task_id"] = source_record["task_id"]
        item["source_template_cluster"] = source_record["task_id"]
        item["source_task_dir"] = str(source)
        item["formal_variant_seed"] = int(item["seed"])
        item["formal_variant_parameters"] = dict(source_record["instance"].parameters)
        item["package_digest"] = task_package_digest(package_dir)
        item["family_instance"] = identity["family_instance"]
        item["projection"] = {"status": projection["status"], "matches": projection.get("matches", [])}
        item["split"] = identity["split"]
        item["materialization"] = "materialized"
        materialized.append(item)
    return {
        "schema_version": 1,
        "population_id": preregistered.get("population_id"),
        "status": "materialized_frozen",
        "primary_population": preregistered.get("primary_population"),
        "sealed_root": str(root),
        "slots": materialized,
        "slot_count": len(materialized),
        "public_dev_total": sum(item.get("visibility") == "public_dev" for item in materialized),
        "sealed_total": sum(item.get("visibility") == "sealed" for item in materialized),
    }


def validate_materialized_manifest(manifest: dict[str, Any], sealed_root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("formal population manifest schema_version must be 1")
    if manifest.get("status") != "materialized_frozen":
        errors.append("formal population must be materialized_frozen")
    slots = manifest.get("slots")
    if not isinstance(slots, list) or len(slots) != 50:
        errors.append("materialized formal manifest must contain 50 slots")
    if sum(isinstance(slot, dict) and slot.get("visibility") == "public_dev" for slot in slots or []) != 15:
        errors.append("materialized formal manifest must contain 15 public_dev slots")
    if sum(isinstance(slot, dict) and slot.get("visibility") == "sealed" for slot in slots or []) != 35:
        errors.append("materialized formal manifest must contain 35 sealed slots")
    seen: set[str] = set()
    seen_splits: set[str] = set()
    for slot in slots or []:
        if not isinstance(slot, dict):
            errors.append("slot must be an object")
            continue
        for key in ("slot_id", "task_id", "package_digest", "visibility", "track", "split", "source_task_id", "source_template_cluster", "lineage_class", "family_instance", "projection"):
            if not slot.get(key):
                errors.append(f"slot missing {key}")
        if not isinstance(slot.get("formal_variant_seed"), int):
            errors.append(f"slot missing formal_variant_seed: {slot.get('slot_id')}")
        slot_id = str(slot.get("slot_id"))
        if slot_id in seen:
            errors.append(f"duplicate slot_id: {slot_id}")
        seen.add(slot_id)
        if slot.get("visibility") not in {"sealed", "public_dev"}:
            errors.append(f"invalid visibility for {slot_id}")
        if slot.get("visibility") == "sealed" and slot.get("lineage_class") != "seeded-template-instance":
            errors.append(f"sealed slot lineage class is not independent: {slot_id}")
        if slot.get("source_template_cluster") != slot.get("source_task_id"):
            errors.append(f"slot source template cluster mismatch: {slot_id}")
        split_key = json.dumps(slot.get("split"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if split_key in seen_splits:
            errors.append(f"duplicate split group: {slot_id}")
        seen_splits.add(split_key)
        if not isinstance(slot.get("family_instance"), dict) or slot["family_instance"].get("instance_id") != slot_id:
            errors.append(f"family instance identity mismatch: {slot_id}")
        if not isinstance(slot.get("projection"), dict) or slot["projection"].get("status") != "pass":
            errors.append(f"executable projection failed: {slot_id}")
        if sealed_root and slot.get("visibility") == "sealed":
            package = Path(sealed_root) / slot_id
            if not package.is_dir():
                errors.append(f"sealed package missing: {slot_id}")
            elif slot.get("package_digest") != task_package_digest(package):
                errors.append(f"sealed package digest mismatch: {slot_id}")
    return errors


def build_release_manifest(*, repo_root: str | Path, population: dict[str, Any], approval: dict[str, Any], contamination: dict[str, Any], claims_path: Path, protocol_path: Path, campaign_config: Path, executor_image: str) -> dict[str, Any]:
    root = Path(repo_root)
    if not isinstance(executor_image, str) or not executor_image.strip():
        raise ValueError("executor_image must be a non-empty image digest or reference")
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "git_commit": str(approval.get("benchmark_revision", "")),
        "population_manifest_digest": digest_json(population),
        "approval_digest": approval.get("approval_digest"),
        "contamination_report_digest": digest_json(contamination),
        "claims_digest": hashlib.sha256(claims_path.read_bytes()).hexdigest(),
        "statistical_protocol_digest": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "campaign_config_digest": hashlib.sha256(campaign_config.read_bytes()).hexdigest(),
        "agent_policy_digest": hashlib.sha256((root / "benchmark" / "formal" / "agent_backbones.yaml").read_bytes()).hexdigest(),
        "executor_image": executor_image,
    }
    manifest["release_digest"] = digest_json({k: v for k, v in manifest.items() if k != "release_digest"})
    return manifest


def validate_formal_release(
    manifest: dict[str, Any],
    *,
    repo_root: str | Path,
    population_manifest: Path,
    approval_path: Path,
    claims_path: Path,
    protocol_path: Path,
    campaign_config: Path,
    analysis_plan_path: Path | None = None,
) -> list[str]:
    """Validate the frozen release bindings at every formal entry point."""
    root = Path(repo_root)
    errors: list[str] = []
    if manifest.get("status") not in {"frozen", "ready", "released"}:
        errors.append("formal release manifest status must be frozen, ready, or released")
    expected_digest = digest_json({key: value for key, value in manifest.items() if key != "release_digest"})
    if manifest.get("release_digest") != expected_digest:
        errors.append("formal release release_digest mismatch")
    try:
        revision = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, __import__("subprocess").CalledProcessError):
        revision = None
    if revision and manifest.get("git_commit") != revision:
        errors.append("formal release git_commit mismatch")
    try:
        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [f"formal release binding unavailable: {exc}"]
    bindings = {
        "population_manifest_digest": digest_json(population),
        "approval_digest": approval.get("approval_digest"),
        "claims_digest": hashlib.sha256(claims_path.read_bytes()).hexdigest() if claims_path.is_file() else None,
        "statistical_protocol_digest": hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.is_file() else None,
        "campaign_config_digest": hashlib.sha256(campaign_config.read_bytes()).hexdigest() if campaign_config.is_file() else None,
        "agent_policy_digest": hashlib.sha256((root / "benchmark" / "formal" / "agent_backbones.yaml").read_bytes()).hexdigest() if (root / "benchmark" / "formal" / "agent_backbones.yaml").is_file() else None,
    }
    for key, value in bindings.items():
        if value is None or manifest.get(key) != value:
            errors.append(f"formal release {key} mismatch")
    if analysis_plan_path is not None:
        analysis_digest = digest_json(json.loads(analysis_plan_path.read_text(encoding="utf-8"))) if analysis_plan_path.is_file() else None
        if analysis_digest is None or manifest.get("analysis_plan_digest") != analysis_digest:
            errors.append("formal release analysis_plan_digest mismatch")
    if not isinstance(manifest.get("executor_image"), str) or not manifest.get("executor_image"):
        errors.append("formal release executor_image is missing")
    return errors
