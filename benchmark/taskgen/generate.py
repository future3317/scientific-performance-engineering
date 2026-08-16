#!/usr/bin/env python3
"""Generate the v1.0-20 pilot population and canonical lineage metadata."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark.harness import miniyaml
from benchmark.families import resolve_family_id, family_instance_digest
from benchmark.families.catalog import FAMILY_SPECS
from benchmark.families import family_instances


NEW_TASKS: tuple[dict[str, Any], ...] = (
    {"task_id": "CORE-COMPILE-DYNAMIC-11", "track": "spe_core", "family": "compiler", "mechanism": "compile_dynamic_shapes", "kind": "positive", "source": "synthetic", "template": "MT-COMPILE-DYNAMIC-V1", "generator": "GEN-COMPILER-DYNAMIC", "fix": "FIX-COMPILE-DYNAMIC-GUARD", "contract": "CONTRACT-COMPILER-CACHE", "difficulty": "medium", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "CORE-COMPILE-TINY-12", "track": "spe_core", "family": "compiler", "mechanism": "compile_tiny_graphs", "kind": "counterexample", "source": "synthetic", "template": "MT-COMPILE-TINY-V1", "generator": "GEN-COMPILER-TINY", "fix": "FIX-COMPILE-TINY-BYPASS", "contract": "CONTRACT-COMPILER-CACHE", "difficulty": "hard", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "CORE-MEM-RETAINED-GRAPH-13", "track": "spe_core", "family": "memory", "mechanism": "retained_graph", "kind": "positive", "source": "synthetic", "template": "MT-MEM-RETAINED-GRAPH-V1", "generator": "GEN-MEMORY-RETAINED", "fix": "FIX-MEM-RETAINED-DETACH", "contract": "CONTRACT-AUTOGRAD-GRAPH", "difficulty": "hard", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "CORE-CHECKPOINT-AMPLE-MEM-14", "track": "spe_core", "family": "memory", "mechanism": "checkpoint_ample_memory", "kind": "counterexample", "source": "synthetic", "template": "MT-CHECKPOINT-AMPLE-MEM-V1", "generator": "GEN-MEMORY-CHECKPOINT", "fix": "FIX-CHECKPOINT-SELECTIVE", "contract": "CONTRACT-AUTOGRAD-GRAPH", "difficulty": "medium", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "CORE-AUTOGRAD-BATCHED-VJP-15", "track": "spe_core", "family": "autograd", "mechanism": "batched_vjp", "kind": "positive", "source": "synthetic", "template": "MT-AUTOGRAD-BATCHED-VJP-V1", "generator": "GEN-AUTOGRAD-VJP", "fix": "FIX-AUTOGRAD-BATCHED-VJP", "contract": "CONTRACT-AUTOGRAD-GRAPH", "difficulty": "hard", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "CORE-DATALOADER-FANOUT-16", "track": "spe_core", "family": "data_pipeline", "mechanism": "dataloader_worker_fanout", "kind": "positive", "source": "synthetic", "template": "MT-DATALOADER-FANOUT-V1", "generator": "GEN-DATA-FANOUT", "fix": "FIX-DATALOADER-TOPOLOGY", "contract": "CONTRACT-DATA-ORDER", "difficulty": "medium", "base": "CORE-COMPILE-RECOMPILE-04"},
    {"task_id": "SCIML-GNN-STATIC-GRAPH-CACHE-17", "track": "sciml", "family": "graph_energy_force", "mechanism": "static_graph_cache", "kind": "positive", "source": "fairchem-shaped", "template": "MT-GNN-STATIC-CACHE-V1", "generator": "GEN-GNN-STATIC", "fix": "FIX-GNN-STATIC-CACHE", "contract": "CONTRACT-ENERGY-FORCE", "difficulty": "medium", "base": "SCIML-GNN-RAGGED-05"},
    {"task_id": "SCIML-GNN-DYNAMIC-GRAPH-18", "track": "sciml", "family": "graph_energy_force", "mechanism": "dynamic_graph_rebuild", "kind": "counterexample", "source": "fairchem-shaped", "template": "MT-GNN-DYNAMIC-GRAPH-V1", "generator": "GEN-GNN-DYNAMIC", "fix": "FIX-GNN-DYNAMIC-REBUILD", "contract": "CONTRACT-ENERGY-FORCE", "difficulty": "hard", "base": "SCIML-GNN-RAGGED-05"},
    {"task_id": "SCIML-FORCE-AUTOGRAD-19", "track": "sciml", "family": "graph_energy_force", "mechanism": "force_autograd", "kind": "positive", "source": "fairchem-shaped", "template": "MT-FORCE-AUTOGRAD-V1", "generator": "GEN-GNN-FORCE", "fix": "FIX-FORCE-BATCHED-VJP", "contract": "CONTRACT-ENERGY-FORCE", "difficulty": "hard", "base": "SCIML-GNN-RAGGED-05"},
    {"task_id": "EVOL-COMPILER-DRIFT-20", "track": "evolution", "family": "episode", "mechanism": ["compile_recompile", "runtime_drift"], "kind": "positive", "source": "synthetic", "template": "MT-EPISODE-COMPILER-DRIFT-V1", "generator": "GEN-EVOLUTION-DRIFT", "fix": "FIX-EVOLUTION-DRIFT-REVALIDATE", "contract": "CONTRACT-EVOLUTION-GOVERNANCE", "difficulty": "hard", "base": "EVOL-EPISODE-POISON-10"},
)

NEW_TASK_IDS = {str(item["task_id"]) for item in NEW_TASKS}


def generate_family_slots(family_id: str, *, count: int, seed: int = 0) -> list[dict[str, Any]]:
    """Return frozen-slot metadata without materializing formal task packages.

    The pilot deliberately calls this only for calibration and view tests. A
    formal sealed campaign must persist the returned slot metadata once and
    then use the same instances for A/B/C/D; this helper does not create the
    remaining v1.0-50 tasks.
    """
    return [item.to_dict() for item in family_instances(family_id, count=count, seed=seed)]


def ast_skeleton_hash(task_dir: Path) -> str:
    """Versioned interpreter-independent AST skeleton identity."""
    def normalize(node: ast.AST) -> Any:
        fields: list[Any] = [node.__class__.__name__]
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                fields.append((name, normalize(value)))
            elif isinstance(value, list):
                fields.append((name, [normalize(item) if isinstance(item, ast.AST) else type(item).__name__ for item in value]))
            elif name in {"ctx", "type_comment", "kind"}:
                fields.append((name, type(value).__name__ if value is not None else None))
            else:
                fields.append((name, type(value).__name__))
        return fields
    chunks: list[str] = []
    for path in sorted((task_dir / "workspace").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        chunks.append(path.relative_to(task_dir).as_posix() + "\n" + json.dumps(normalize(tree), sort_keys=True, separators=(",", ":")))
    return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()


def _annotate_task(task_dir: Path, metadata: dict[str, Any]) -> None:
    source_task_id = metadata.get("base")
    if source_task_id and source_task_id != metadata["task_id"]:
        # A generated package may reuse a fixture implementation, but its
        # public package artifacts must not retain the prototype task identity.
        text_suffixes = {".py", ".md", ".json", ".diff", ".yaml", ".yml", ".txt"}
        for path in task_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in text_suffixes:
                text = path.read_text(encoding="utf-8")
                replaced = text.replace(str(source_task_id), str(metadata["task_id"]))
                if replaced != text:
                    path.write_text(replaced, encoding="utf-8")
    workspace_solution = task_dir / "workspace" / "solution.py"
    if workspace_solution.is_file() and metadata["task_id"] in NEW_TASK_IDS:
        text = workspace_solution.read_text(encoding="utf-8")
        marker = f'TASK_VARIANT = "{metadata["task_id"]}"\n'
        # Keep module docstrings and ``from __future__`` imports valid.  The
        # marker is metadata only and must not become a statement before a
        # future import (which would make every generated fixture uncompilable).
        # Remove any previous generator marker before placing the canonical
        # one.  This also repairs an interrupted/older generation run.
        import re
        text = re.sub(r'^TASK_VARIANT = "[A-Z0-9-]+"\n\s*', "", text, flags=re.MULTILINE)
        if "from __future__ import annotations\n" in text:
            needle = "from __future__ import annotations\n"
            text = text.replace(needle, needle + "\n" + marker, 1)
        else:
            text = marker + text
        workspace_solution.write_text(text, encoding="utf-8")
    spec = miniyaml.load(str(task_dir / "task.yaml"))
    spec["task_id"] = metadata["task_id"]
    spec["track"] = metadata["track"]
    spec["family"] = metadata["family"]
    spec["mechanism"] = metadata["mechanism"]
    spec["kind"] = metadata["kind"]
    spec["lineage"] = {
        "source": metadata["source"],
        "mutation_template_id": metadata["template"],
        "mutation_seed": int(hashlib.sha256(metadata["task_id"].encode()).hexdigest()[:8], 16),
    }
    spec["generator_family_id"] = metadata["generator"]
    try:
        family_id = resolve_family_id(metadata["generator"])
    except KeyError:
        family_id = None
    if family_id is not None:
        spec["family_id"] = family_id
        if metadata["task_id"] in FAMILY_SPECS[family_id].anchors:
            spec["anchor_instance_id"] = metadata["task_id"]
            anchor = FAMILY_SPECS[family_id].reconstruct_anchor(metadata["task_id"])
            spec["family_parameters"] = dict(anchor.parameters)
            spec["family_instance_digest"] = family_instance_digest(family_id, anchor.parameters)
            spec["public_context"] = {"workload": dict(anchor.parameters)}
    spec["oracle_fix_pattern_id"] = metadata["fix"]
    # The family policy is the sole scientific-contract authority.  Legacy
    # task metadata may carry an older label, but materialized anchors always
    # project the canonical FamilySpec policy into the task contract.
    if family_id is not None:
        spec["scientific_contract_id"] = FAMILY_SPECS[family_id].policy_spec().policy_id
    else:
        spec["scientific_contract_id"] = metadata["contract"]
    spec["difficulty_tier"] = metadata["difficulty"]
    if metadata["kind"] in {"counterexample", "do_not_apply"}:
        spec["oracle"]["expected_speedup_range"] = [0.9, 1.1]
    spec["workspace_ast_skeleton_hash"] = ast_skeleton_hash(task_dir)
    spec["ast_skeleton_version"] = 2
    miniyaml.save(spec, str(task_dir / "task.yaml"))
    metadata_path = task_dir / "metadata.json"
    if metadata_path.is_file():
        package_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        package_metadata.update(
            {
                "task_id": metadata["task_id"],
                "title": spec.get("title", metadata["task_id"]),
                "track": metadata["track"],
                "family": metadata["family"],
                "mechanism": metadata["mechanism"],
                "kind": metadata["kind"],
                "lineage": spec["lineage"],
                "difficulty": metadata["difficulty"],
                "split_group": {
                    "family": metadata["family"],
                    "mechanism": metadata["mechanism"],
                    "source": metadata["source"],
                    "mutation_template_id": metadata["template"],
                },
            }
        )
        if spec.get("family_id"):
            package_metadata["family_id"] = spec["family_id"]
        if spec.get("anchor_instance_id"):
            package_metadata["anchor_instance_id"] = spec["anchor_instance_id"]
        metadata_path.write_text(json.dumps(package_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    oracle = task_dir / "oracle"
    oracle.mkdir(exist_ok=True)
    (oracle / "tempting_wrong_patch.md").write_text(
        f"# Tempting but wrong patch\n\nDo not apply an unconditional `{metadata['fix']}` rewrite: it changes the scientific or workload contract outside the measured trigger.\n",
        encoding="utf-8",
    )
    (oracle / "noise_floor.json").write_text(
        json.dumps({"declared_percent": float(spec["measurement"].get("noise_floor_percent", 2.0)), "measurement": "paired_control_runs"}, indent=2) + "\n",
        encoding="utf-8",
    )
    oracle_reference = "oracle/reference_patch.diff" if (oracle / "reference_patch.diff").is_file() else "hidden_verifier/reference.py"
    (oracle / "validation.json").write_text(
        json.dumps({
            "baseline_validation": {"entrypoint": "benchmark.py::run_correctness", "fresh_inputs": int(spec["correctness"]["num_fresh_inputs"])},
            "oracle_validation": {"artifact": oracle_reference, "scientific_gates": spec.get("scientific_gates", [])},
            "anti_cheat": "harness S1 canary/static scan",
            "deterministic_fixture": "benchmark.py::make_fixtures(seed=0)",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate(root: str | Path) -> list[str]:
    root = Path(root)
    tasks_root = root / "benchmark" / "tasks"
    for metadata in NEW_TASKS:
        target = tasks_root / metadata["task_id"]
        if not target.exists():
            source = tasks_root / metadata["base"]
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "episode_result.json", "store"))
            metadata["new"] = True
            if metadata["track"] == "evolution":
                old = target / "episodes" / "poison_episode.yaml"
                new = target / "episodes" / "compiler_drift_episode.yaml"
                old.rename(new)
                benchmark_py = target / "benchmark.py"
                benchmark_py.write_text(benchmark_py.read_text(encoding="utf-8").replace("poison_episode.yaml", "compiler_drift_episode.yaml"), encoding="utf-8")
        _annotate_task(target, metadata)
    existing = []
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.yaml").is_file():
            continue
        spec = miniyaml.load(str(task_dir / "task.yaml"))
        if all(key in spec for key in ("generator_family_id", "oracle_fix_pattern_id", "scientific_contract_id", "difficulty_tier")):
            metadata = {
                "task_id": spec["task_id"], "track": spec["track"], "family": spec["family"],
                "mechanism": spec["mechanism"], "kind": spec["kind"], "source": spec["lineage"]["source"],
                "template": spec["lineage"]["mutation_template_id"], "generator": spec["generator_family_id"],
                "fix": spec["oracle_fix_pattern_id"], "contract": spec["scientific_contract_id"],
                "difficulty": spec["difficulty_tier"],
            }
        elif spec.get("track") == "evolution":
            metadata = {"task_id": spec["task_id"], "track": spec["track"], "family": spec["family"], "mechanism": spec["mechanism"], "kind": spec["kind"], "source": spec["lineage"]["source"], "template": spec["lineage"]["mutation_template_id"], "generator": "GEN-" + str(spec["family"]).upper(), "fix": "FIX-" + str(spec["task_id"]), "contract": "CONTRACT-EVOLUTION-GOVERNANCE", "difficulty": "hard"}
        else:
            metadata = {"task_id": spec["task_id"], "track": spec["track"], "family": spec["family"], "mechanism": spec["mechanism"], "kind": spec["kind"], "source": spec["lineage"]["source"], "template": spec["lineage"]["mutation_template_id"], "generator": "GEN-" + str(spec["family"]).upper(), "fix": "FIX-" + str(spec["task_id"]), "contract": "CONTRACT-" + str(spec["family"]).upper(), "difficulty": "medium"}
        _annotate_task(task_dir, metadata)
        existing.append(spec["task_id"])
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps({"tasks": generate(args.root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
