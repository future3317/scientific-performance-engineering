"""Recomputable family analysis and one terminal evidence export."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from experiments.real_intervention_v5.common import read_json, write_json, write_csv, digest, utc
from .policy_experiment import summarize


def analyze(root, rows):
    frozen = read_json(root / "POLICY_FREEZE.json") if (root / "POLICY_FREEZE.json").exists() else {}
    analyses = {}
    for contrast in sorted({r["contrast"] for r in rows}):
        per_split = {}
        for split in ("development", "calibration", "final"):
            subset = [r for r in rows if r["contrast"] == contrast and r["split"] == split]
            policies = frozen.get("by_contrast", {}).get(contrast) if split == "final" else None
            overhead = policies.get("routing_overhead_s") if policies else None
            per_split[split] = summarize(subset, policies, routing_overhead_s=overhead)
            if policies:
                actions = {(p["family_id"], p["context_id"]): p["predictions"] for p in policies["final_predictions"]}
                for name, summary in per_split[split].get("policies", {}).items():
                    summary["unsafe_deployment_rows"] = sum(r["primary_pair_status"] == "SCIENTIFIC_UNSAFE" and actions.get((r["family_id"], r["context_id"]), {}).get(name) is True for r in subset)
        analyses[contrast] = per_split
    return analyses


def bundle(root):
    members = {}
    payloads = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or "cache" in relative.parts or "references" in relative.parts
                or path.name in {"MEMBER_SHA256.json", "EVIDENCE_BUNDLE.tar.gz", "BUNDLE_VALIDATION.json", "runner.lock"}
                or path.suffix not in {".json", ".jsonl", ".csv", ".md", ".py", ".log"}):
            continue
        raw = path.read_bytes()
        exported = raw.decode("utf-8", errors="replace").replace("/home/workspace/lrh", "<ACCOUNT_WORKSPACE>").encode()
        name = str(relative)
        members[name] = {"bytes": len(exported), "sha256": hashlib.sha256(exported).hexdigest(), "original_sha256": hashlib.sha256(raw).hexdigest(), "redacted": exported != raw}
        payloads[name] = exported
    write_json(root / "MEMBER_SHA256.json", members)
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "w:gz") as archive:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or set(names) != set(members):
            raise RuntimeError("bundle membership is not unique and closed")
        for item in archive:
            data = archive.extractfile(item).read()
            if item.name.startswith("/") or ".." in Path(item.name).parts or len(data) != members[item.name]["bytes"] or hashlib.sha256(data).hexdigest() != members[item.name]["sha256"]:
                raise RuntimeError(f"bundle validation failed: {item.name}")
    write_json(root / "BUNDLE_VALIDATION.json", {"status": "PASS", "members_verified": len(members)})


def finish(root, status, rows, *, elapsed_s, blocker=None):
    root = Path(root)
    manifest = read_json(root / "FROZEN_MANIFEST.json") if (root / "FROZEN_MANIFEST.json").exists() else {}
    start = read_json(root / "RUN_START.json") if (root / "RUN_START.json").exists() else {}
    measured_rows = [r for r in rows if not r["contrast"].startswith("PROJECTED")]
    with (root / "PAIRS.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_csv(root / "PAIRS.csv", rows)
    attempts = []
    folders = {p.parent for pattern in ("intent.json", "started.json", "terminal.json", "recovery.json") for p in root.rglob(pattern)}
    for folder in sorted(folders):
        path = next(folder / name for name in ("terminal.json", "recovery.json", "intent.json", "started.json") if (folder / name).exists())
        arm = read_json(path)
        has_started = (folder / "started.json").exists()
        attempts.append({"path": str(path.relative_to(root)), "known_worker_started": has_started, "intent_present": (folder / "intent.json").exists(),
                         "status": arm.get("primary_pair_status", "UNKNOWN"), "execution_status": arm.get("execution_status", "UNKNOWN"),
                         "parent_wall_s": arm.get("parent_wall_s"), "child_cpu_s": arm.get("timing", {}).get("child_cpu_s"),
                         "worker_cuda_provenance": arm.get("worker_cuda_provenance"), "co_tenancy_exposure": arm.get("co_tenancy_exposure")})
    write_json(root / "RAW_ATTEMPT_INDEX.json", attempts)
    final_measured = any(r.get("analytically_comparable") for r in measured_rows if r["split"] == "final")
    scientific_status = ("INDEPENDENT_FINAL_PILOT" if final_measured and (root / "FINAL_RELEASE.json").exists()
                         else "DEVELOPMENT_CALIBRATION_DESCRIPTIVE_ONLY" if any(r.get("analytically_comparable") for r in measured_rows) else "NO_EFFECT_ESTIMATE")
    metrics = {"status": status, "scientific_status": scientific_status,
               "source_sha": manifest.get("source_sha", start.get("source_sha")), "study_id": manifest.get("study_id", start.get("config", {}).get("study_id")), "gpu": manifest.get("gpu"),
               "v5_raw_modified": False, "formal_contrast_rows": len(rows), "primary_status_counts": dict(Counter(r["primary_pair_status"] for r in rows)),
               "analytically_comparable_rows": sum(r.get("analytically_comparable", False) for r in measured_rows),
               "projected_rows_not_actual_pairs": len(rows) - len(measured_rows),
               "actual_context_sessions_planned": len({r["pair_id"] for r in measured_rows}),
               "independent_measured_families": len({r["family_id"] for r in measured_rows if r.get("analytically_comparable")}),
               "lifecycle_attempt_records": len(attempts), "definite_worker_starts": sum(a["known_worker_started"] for a in attempts),
               "unresolved_intents": sum(a["intent_present"] and a["status"] == "UNKNOWN" for a in attempts), "final_released": (root / "FINAL_RELEASE.json").exists(),
               "original_acre_certification": "MATHEMATICALLY_UNREACHABLE_ON_THIS_COHORT",
               "family_analysis": analyze(root, rows),
               "resources": {"runner_wall_s": elapsed_s, "controller_cpu_s": time.process_time(),
                   "recorded_child_cpu_s": sum(a["child_cpu_s"] or 0 for a in attempts), "cpu_accounting": "includes recorded completed-worker CPU only; timed-out child CPU unknown",
                   "gpu_busy_seconds": None, "gpu_busy_status": "NOT_INFERRED_FROM_SAMPLES",
                   "storage_bytes_before_bundle": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())},
               "blocker": blocker, "timestamp": utc()}
    write_json(root / "V6_METRICS.json", metrics)
    lines = [f"# {metrics['study_id'] or 'V6 engineering'}", "", f"Status: {status}; scientific status: {metrics['scientific_status']}",
             f"Source: {metrics['source_sha']}; V5 raw modified: NO", f"Analytically comparable contrast rows: {metrics['analytically_comparable_rows']} / {len(rows)}",
             f"Independent measured families: {metrics['independent_measured_families']}; final released: {metrics['final_released']}",
             "Original ACRE certification: mathematically unreachable on this cohort (necessary calibration lower bound42).", "",
             "UNKNOWN co-tenancy permits only exploratory paired descriptions; it does not prove uncontended execution.",
             "R100 is cost projection, not100 observed deployments or independent samples. Family is the independent unit.", "",
             "## Family effects", "", "Contrast | Split | Measured families | Positive / negative | Mean log benefit | Family95%CI", "--- | --- | ---: | --- | --- | ---"]
    for contrast, splits in metrics["family_analysis"].items():
        for split, result in splits.items():
            s = result["overall"]
            lines.append(f"{contrast} | {split} | {s['observed_families']} | {s['positive_families']}/{s['negative_families']} | {s['family_equal_mean_log_benefit']} | {s['family_bootstrap_ci95']}")
    lines += ["", "Policy results, raw seconds, paired research-minus-simple CIs and missingness bounds are in V6_METRICS.json.",
              "Primary costs include deployment instrumentation/construction/cache materialization/probes, not GPU-only time. No task accuracy claim."]
    if blocker:
        lines += ["", f"Blocker: {blocker}"]
    (root / "V6_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "PAPER_CLAIM_MATRIX.md").write_text("V6 supports only the completed conditional systems pilot at its recorded exposure grade.\nNo task-accuracy claim, no canonical ACRE certificate, no substitution of historical engineering for prospective final.\n")
    write_json(root / "RUN_EXIT.json", {"status": status, "timestamp": utc(), "scientific_effect_complete": status == "PILOT_COMPLETE"})
    if blocker:
        write_json(root / "BLOCKER.json", {"assertion": blocker, "status": status})
        write_json(root / "NEXT_VALID_ACTION.json", {"condition": blocker, "no_history_retry": True})
    bundle(root)
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.root / "PAIRS.jsonl").read_text().splitlines()]
    print(json.dumps(analyze(args.root, rows), indent=2))
