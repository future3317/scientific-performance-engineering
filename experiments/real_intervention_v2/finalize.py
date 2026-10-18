from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any

from .contracts import digest
from .replay import aggregate_sessions


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == "compute_processes" and isinstance(item, list):
                output["compute_process_count"] = len(item)
                output["compute_process_memory_mb"] = [row.get("memory_mb") for row in item]
            elif key in {"pid", "name"}:
                continue
            else:
                output[key] = _redact(item)
        return output
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]


def _effect_summary(records: list[dict[str, Any]], delta: float, seed: int) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_split.setdefault(row["context"]["split"], []).append(row)
    output = {}
    for split, rows in by_split.items():
        valid = [row for row in rows if all(row["arms"].get(arm, {}).get("status") == "valid"
                                             for arm in ("eager", "compile"))
                 and row.get("correctness", {}).get("passed")]
        effects = [math.log(float(row["arms"]["eager"]["total_job_s"])
                            / float(row["arms"]["compile"]["total_job_s"])) for row in valid]
        groups: dict[str, list[float]] = {}
        for row, effect in zip(valid, effects):
            groups.setdefault(str(row["context"]["lineage_id"]), []).append(effect)
        group_values = [sum(values) / len(values) for values in groups.values()]
        rng = random.Random(seed)
        boot = []
        if len(group_values) >= 2:
            for _ in range(10000):
                sample = [group_values[rng.randrange(len(group_values))] for _ in group_values]
                boot.append(sum(sample) / len(sample))
        output[split] = {
            "scheduled_contexts": len(rows),
            "valid_contexts": len(valid),
            "valid_independent_lineages": len(groups),
            "positive": sum(value > delta for value in effects),
            "negative": sum(value < -delta for value in effects),
            "unresolved": len(rows) - sum(abs(value) > delta for value in effects),
            "mean_log_benefit": sum(effects) / len(effects) if effects else None,
            "median_log_benefit": _percentile(effects, 0.5),
            "lineage_cluster_bootstrap_95_ci": [_percentile(boot, 0.025), _percentile(boot, 0.975)] if boot else None,
        }
    return output


def _policy_csv(policy_results: dict[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    fields = ["policy", "context_id", "deploy", "benefit_s", "log_effect", "realized_saving_s",
              "harm", "regret_s", "outcome_valid", "candidate_status"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for policy, summary in policy_results.get("results", {}).items():
        for row in summary.get("rows", []):
            writer.writerow({"policy": policy, **{key: row.get(key) for key in fields[1:]}})
    return stream.getvalue().encode()


def finalize(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    state = _read_json(root / "RUN_PROGRESS.json")
    policy = _read_json(root / "POLICY_FROZEN.json") if (root / "POLICY_FROZEN.json").exists() else None
    policy_results = _read_json(root / "POLICY_RESULTS.json") if (root / "POLICY_RESULTS.json").exists() else None
    raw_pairs = [_read_json(path) for path in sorted((root / "pairs").glob("*.json"))]
    contexts = aggregate_sessions(raw_pairs) if raw_pairs else []
    if (root / "PAIRS.jsonl").exists() or (root / "METRICS.json").exists() or (root / "EVIDENCE_BUNDLE.tar.gz").exists():
        raise FileExistsError("final outputs already exist; refusing to replace evidence")

    pair_lines = []
    raw_pair_hashes = {}
    for path in sorted((root / "pairs").glob("*.json")):
        raw_pair_hashes[str(path.relative_to(root))] = _sha(path.read_bytes())
    for row in contexts:
        arms = row["arms"]
        pair_lines.append({
            "context_id": row["context_id"],
            "lineage_id": row["context"]["lineage_id"],
            "split": row["context"]["split"],
            "preexecution_context": row["context"],
            "sessions": [{"session_id": session["session_id"],
                          "order": session["order"],
                          "eager_status": session["arms"].get("eager", {}).get("status"),
                          "compile_status": session["arms"].get("compile", {}).get("status"),
                          "eager_total_job_s": session["arms"].get("eager", {}).get("total_job_s"),
                          "compile_total_job_s": session["arms"].get("compile", {}).get("total_job_s"),
                          "correctness": session.get("correctness"),
                          "eager_parent_wall_s": session["arms"].get("eager", {}).get("parent_wall_s"),
                          "compile_parent_wall_s": session["arms"].get("compile", {}).get("parent_wall_s")}
                         for session in row.get("sessions", [])],
            "context_eager_total_job_s": arms.get("eager", {}).get("total_job_s"),
            "context_compile_total_job_s": arms.get("compile", {}).get("total_job_s"),
            "correctness": row.get("correctness"),
        })
    pairs_bytes = b"".join((json.dumps(_redact(row), sort_keys=True, allow_nan=False) + "\n").encode()
                           for row in pair_lines)
    metrics = {
        "experiment": manifest["run_id"],
        "status": state["status"],
        "run_identity": manifest["run_identity"],
        "source_commit": manifest["source_commit"],
        "torchbench_commit": manifest["torchbench_commit"],
        "intervention_id": manifest["intervention_id"],
        "provider_calls": 0,
        "model_lineages": {"scheduled": len({row["lineage_id"] for row in manifest["contexts"]}),
                            "by_split": {split: len({row["lineage_id"] for row in manifest["contexts"] if row["split"] == split})
                                         for split in ("development", "gate_calibration", "final_untouched")}},
        "contexts": {"scheduled": len(manifest["contexts"]), "paired_contexts_observed": len(contexts),
                      "paired_sessions_scheduled": manifest["planned_pair_sessions"],
                      "paired_sessions_observed": len(raw_pairs)},
        "effect_by_split": _effect_summary(contexts, float((policy or {}).get("minimum_effect_log", 0.04879016416943205)),
                                             int(manifest["seed"])),
        "resource_usage": state.get("resource_usage", {}),
        "a_a_noise": _read_json(root / "AA_NOISE.json") if (root / "AA_NOISE.json").exists() else None,
        "policy_frozen": {"policy_digest": policy.get("policy_digest"),
                          "final_decision_digest": policy.get("final_decision_digest"),
                          "acre_status": policy.get("acre_status")} if policy else None,
        "policy_results": policy_results,
        "science_external_zinc": "NOT_RUN",
        "raw_pair_receipts": raw_pair_hashes,
        "source_paths": {"manifest": "RUN_MANIFEST.json", "approval": "APPROVAL_BINDING.json",
                         "progress": "RUN_PROGRESS.json"},
    }
    metrics_bytes = (json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    csv_bytes = _policy_csv(policy_results or {})
    report = [f"# EvoSPE v2 real intervention result — {manifest['run_id']}", "",
              f"Status: **{state['status']}**", "",
              f"- TorchBench commit: `{manifest['torchbench_commit']}`",
              f"- Source commit: `{manifest['source_commit']}`",
              f"- Intervention: `{manifest['intervention_id']}` (`torch.compile`, inductor/default)",
              f"- Model lineages: {metrics['model_lineages']['scheduled']} scheduled",
              f"- Contexts: {len(contexts)} observed / {len(manifest['contexts'])} scheduled",
              f"- Matched sessions: {len(raw_pairs)} observed / {manifest['planned_pair_sessions']} scheduled",
              f"- Provider calls: 0", "",
              "## Effect summary", "",
              "Positive/negative counts use the frozen practical log-effect margin. Repeated sessions are first reduced within context; model lineage is the cluster unit.", ""]
    for split, summary in metrics["effect_by_split"].items():
        report.append(f"- `{split}`: n={summary['valid_contexts']} valid contexts across {summary['valid_independent_lineages']} lineages; "
                      f"positive={summary['positive']}, negative={summary['negative']}, unresolved={summary['unresolved']}; "
                      f"mean log benefit={summary['mean_log_benefit']}; 95% lineage-bootstrap CI={summary['lineage_cluster_bootstrap_95_ci']}")
    report.extend(["", "## Interpretation boundary", "",
                   "This is real TorchBench model-code runtime evidence for a fixed compile intervention. It does not validate automatic Skill generation or scientific-dataset time-to-quality. ACRE was scored only through the frozen policy path; where no governed support certificate existed, it abstained.",
                   "", "## Artifacts", "", "See `METRICS.json`, `PAIRS.jsonl`, `ACRE_POLICY_TABLE.csv`, and the sanitized evidence bundle."])
    report_bytes = ("\n".join(report) + "\n").encode()
    paper_ready = ("# Results paper-ready summary\n\n"
                   "The study evaluates one fixed executable intervention: eager execution versus `torch.compile` with the inductor backend and default mode. Models are pinned to the TorchBench source commit and split by model lineage before S0/S1 outcomes. The primary estimand is cold-inclusive model-job wall time over the planned inference horizon; warm repeated timing is reported separately.\n\n"
                   "## New evidence\n\n"
                   f"Run status: **{state['status']}**. The frozen cohort contains {metrics['model_lineages']['scheduled']} model lineages and {len(manifest['contexts'])} planned contexts; {len(contexts)} contexts have observed paired sessions. Provider calls are zero. See `METRICS.json` for split-specific validity, effects, uncertainty, failures, resource use, and frozen-policy results.\n\n"
                   "## Claim boundary\n\n"
                   "These measurements concern runtime behavior of real benchmark model implementations under one fixed compile intervention. They do not establish that ACRE or automatic skill generation improves scientific training quality. Any routing result is conditional on the frozen pre-execution policy and observed final receipts; pairwise oracle is a post-hoc upper bound only. PyG ZINC was not run in this result root.\n").encode()

    outputs = {
        "PAIRS.jsonl": pairs_bytes,
        "METRICS.json": metrics_bytes,
        "ACRE_POLICY_TABLE.csv": csv_bytes,
        "REPORT.md": report_bytes,
        "RESULTS_PAPER_READY.md": paper_ready,
    }
    for name, data in outputs.items():
        _write(root / name, data)

    members: dict[str, bytes] = {**outputs}
    for name in ("RUN_MANIFEST.json", "APPROVAL_BINDING.json", "MODEL_API_MATRIX.json", "AA_NOISE.json",
                 "POLICY_FROZEN.json", "POLICY_RESULTS.json", "RUN_PROGRESS.json"):
        path = root / name
        if path.exists():
            members[name] = path.read_bytes()
    for folder in ("pairs", "arms", "aa/pairs", "aa/arms"):
        directory = root / folder
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            rel = path.relative_to(root).as_posix()
            members[rel] = (json.dumps(_redact(_read_json(path)), sort_keys=True,
                                       indent=2, allow_nan=False) + "\n").encode()
    if len(members) != len(set(members)):
        raise ValueError("evidence bundle contains duplicate member names")
    index = {"members": [{"path": name, "sha256": _sha(data), "bytes": len(data)}
                         for name, data in sorted(members.items())],
             "contains_secrets": False, "contains_model_weights": False,
             "raw_server_receipts_retained": True}
    members["BUNDLE_INDEX.json"] = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "w:gz") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
