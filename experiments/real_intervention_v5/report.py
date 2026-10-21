"""Recomputable denominators, research claim boundaries, and a sanitized bundle."""
from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import tarfile

from .common import append_jsonl, digest, file_digest, read_json, utc, write_csv, write_json


def bundle(root):
    members = {}
    content = {}
    excluded = {"MEMBER_SHA256.json", "EVIDENCE_BUNDLE.tar.gz", "BUNDLE_VALIDATION.json"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.name in excluded or any(p in {"cache", "references"} for p in relative.parts): continue
        if path.suffix not in {".json", ".jsonl", ".csv", ".md", ".py"}: continue
        data = path.read_text(encoding="utf-8")
        # No environment dump, full process command lines, credentials or weights
        # are collected. Remove account-specific roots from the exported copy.
        data = data.replace("/home/workspace/lrh", "<ACCOUNT_WORKSPACE>")
        payload = data.encode()
        name = relative.as_posix()
        import hashlib
        members[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        content[name] = payload
    write_json(root / "MEMBER_SHA256.json", members)
    content["MEMBER_SHA256.json"] = (root / "MEMBER_SHA256.json").read_bytes()
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "w:gz") as archive:
        for name, payload in content.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe or duplicate bundle member")
        embedded = json.loads(archive.extractfile("MEMBER_SHA256.json").read())
        if embedded != members:
            raise ValueError("embedded bundle manifest differs")
        for name, expected in members.items():
            payload = archive.extractfile(name).read()
            import hashlib
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"bundle closure failed: {name}")
    write_json(root / "BUNDLE_VALIDATION.json", {"status": "PASS", "members_verified": len(members),
                                               "archive_sha256": file_digest(root / "EVIDENCE_BUNDLE.tar.gz"),
                                               "member_manifest_sha256": file_digest(root / "MEMBER_SHA256.json")})


def resource_ledger(root, pairs, wall_s):
    # B's R=1/10/100 contrasts share one actual execution. Count it only once.
    sessions = set(p["session_receipt"] for p in pairs if p.get("session_receipt"))
    cpu, parent, construction, prepare, forwards, peaks = 0., 0., 0., 0., 0., []
    samples, process_utilization_status = [], set()
    for relative in sessions:
        path = root / relative
        if not path.exists(): continue
        receipt = read_json(path)
        parent += receipt.get("timings", {}).get("session_wall_seconds", 0.)
        for arm in receipt.get("arms", []):
            payload = arm.get("worker_result", {})
            t = payload.get("timing", {})
            cpu += t.get("child_cpu_s", 0.)
            construction += t.get("model_construction_s", 0.)
            prepare += t.get("prepare_s", 0.)
            forwards += t.get("first_forward_s", 0.) + t.get("remaining_s", 0.)
            if payload.get("peak_allocated_bytes") is not None: peaks.append(payload["peak_allocated_bytes"])
        for sample in receipt.get("monitor", {}).get("samples", []):
            if isinstance(sample.get("utilization_percent"), (int, float)): samples.append(sample["utilization_percent"])
            process_utilization_status.add(str(sample.get("process_utilization_status", "UNAVAILABLE")))
    return {"runner_wall_s_including_wait": wall_s, "efficacy_session_parent_wall_s": parent,
            "efficacy_child_cpu_s_lower_bound": cpu, "cpu_exclusions": "qualification/shakedown/controller CPU not included in this lower bound",
            "efficacy_construction_s": construction, "efficacy_prepare_s": prepare, "efficacy_H_forward_s": forwards,
            "gpu_busy_seconds": None, "gpu_busy_seconds_status": "UNKNOWN_NOT_INFERRED_FROM_SAMPLED_DEVICE_UTILIZATION",
            "sampled_device_utilization_mean_percent": sum(samples) / len(samples) if samples else None,
            "device_utilization_samples": len(samples), "process_utilization_status": sorted(process_utilization_status),
            "peak_allocated_bytes": max(peaks) if peaks else None,
            "storage_bytes_before_bundle": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}


def finish(root, manifest, registry, qualification, pairs, status, feasibility, shakedown, wall_s):
    from .analysis import summarize, evaluate_policies
    config = manifest["config"]
    with (root / "PAIRS.jsonl").open("w") as f:
        for p in pairs: f.write(json.dumps(p, sort_keys=True, allow_nan=False) + "\n")
    fields = ["pair_id", "family_id", "split", "batch", "horizon", "session", "contrast", "execution_status", "scientific_status", "platform_status", "exposure_status", "analysis_status", "first_failing_gate", "error_code", "log_benefit", "s0_s", "s1_s"]
    write_csv(root / "PAIR_VALIDITY_LEDGER.csv", [{k: p.get(k) for k in fields} for p in pairs], fields)
    failures = Counter((p.get("execution_status"), p.get("scientific_status"), p.get("platform_status"), p.get("exposure_status"), p.get("analysis_status"), p.get("first_failing_gate")) for p in pairs)
    write_csv(root / "FAILURE_TAXONOMY.csv", [{"execution_status": k[0], "scientific_status": k[1], "platform_status": k[2], "exposure_status": k[3], "analysis_status": k[4], "first_failing_gate": k[5], "count": n} for k, n in failures.items()])
    contrasts = sorted({p.get("contrast", "LAYOUT") for p in pairs}) or (["LAYOUT"] if config["study"] == "A" else ["COMPILE_COLD", "PERSISTENT_R1", "PERSISTENT_R10", "PERSISTENT_R100"])
    summaries = {c: summarize([p for p in pairs if p.get("contrast", "LAYOUT") == c]) for c in contrasts}
    family_rows = []
    for c, summary in summaries.items():
        for split, values in summary["by_split"].items():
            family_rows.extend({"contrast": c, "split": split, "family_id": f, "mean_log_benefit": v} for f, v in values["family_mean_log_benefit"].items())
    write_csv(root / "PER_FAMILY_EFFECTS.csv", family_rows)
    qualified = [r for r in qualification if r["representative"]]
    families = {"registered": len({r["family_id"] for r in registry}), "registered_candidates": len(registry),
                "baseline_valid": sum(r["baseline_valid"] for r in qualified) if qualification else None,
                "candidate_correct": sum(r["candidate_correct"] for r in qualified) if qualification else None,
                "kernel_exposure_valid": sum(r.get("kernel_exposure_valid", False) for r in qualified) if qualification else None,
                "paired_valid": len({p["family_id"] for p in pairs if p["analysis_status"] == "VALID_PAIRED"})}
    split_counts = {s: {"registered": len({r["family_id"] for r in registry if r["split"] == s}),
                        "qualified": sum(r["eligibility"] == "ELIGIBLE" and r["split"] == s for r in qualified) if qualification else None,
                        "paired_valid": len({p["family_id"] for p in pairs if p["split"] == s and p["analysis_status"] == "VALID_PAIRED"})}
                    for s in ("development", "calibration", "final")}
    # Execution sessions and derived contrast rows have separate denominators.
    unique = {p.get("session_receipt", p["pair_id"]): p for p in pairs}
    counts = {"planned_eligible_sessions": len(unique) if qualification else None,
              "planned_registered_upper_bound_sessions": families["registered"] * len(config["batches"]) * len(config["horizons"]) * config["sessions_per_context"],
              "contrast_rows": len(pairs), "started_sessions": sum(p["execution_status"] != "NOT_STARTED" for p in unique.values()),
              "complete_sessions": sum(p["execution_status"] == "COMPLETE" for p in unique.values()),
              "valid_contrast_rows": sum(p["analysis_status"] == "VALID_PAIRED" for p in pairs),
              "unsafe_sessions": sum(p["scientific_status"] == "NUMERICAL_UNSAFE" for p in unique.values()),
              "contended_sessions": sum(p["exposure_status"] == "CONTENDED" for p in unique.values()),
              "unknown_or_incomplete_started_sessions": sum(p["execution_status"] != "NOT_STARTED" and p["exposure_status"] in {"UNKNOWN", "INCOMPLETE"} for p in unique.values()),
              "not_started_sessions": sum(p["execution_status"] == "NOT_STARTED" for p in unique.values())}
    policy_path = root / "POLICY_FREEZE.json"
    if not policy_path.exists(): write_json(policy_path, {"status": "NOT_FITTED", "reason": status, "ORIGINAL_ACRE": "NOT_FITTED", "ACRE_SYS": "NOT_IMPLEMENTED"})
    freeze = read_json(policy_path)
    policy_rows = []
    if "by_contrast" in freeze:
        for c, policy in freeze["by_contrast"].items():
            for split in ("development", "calibration", "final"):
                policy_rows.extend({**p, "contrast": c, "split": split} for p in evaluate_policies([p for p in pairs if p.get("contrast") == c and p["split"] == split], policy))
    elif freeze.get("status") != "NOT_FITTED":
        for split in ("development", "calibration", "final"):
            policy_rows.extend({**p, "split": split} for p in evaluate_policies([p for p in pairs if p["split"] == split], freeze))
    write_csv(root / "POLICY_RESULTS.csv", policy_rows)
    resources = resource_ledger(root, pairs, wall_s)
    claims = {"allowed": ["Programmatic necessary ACRE sample-size lower bound under unchanged historical certificate", "Registered-model systems performance on random inputs, conditional on numeric and platform validity, when observed"],
              "not_allowed": ["Scientific task accuracy or time-to-quality", "Original ACRE certified or fitted without canonical certificate", "Shared GPU samples with incomplete attribution treated as matched", "V3/V4 observations counted as V5 final"],
              "next_scientific_decision": "Assess fixed intervention from independent final and its uncertainty; retain null/negative results" if (root / "FINAL_RELEASE.json").exists() else "Resolve reported qualification/resource deficit before making new efficacy claims"}
    metrics = {"status": status, "study_id": config["study_id"], "source_sha": manifest["source_sha"], "manifest_sha256": digest(manifest),
               "families": families, "splits": split_counts, "pairs": counts, "contrasts": summaries, "certificate_feasibility": feasibility,
               "original_acre_status": "ORIGINAL_ACRE_NOT_FITTED", "independent_support_groups": None,
               "acre_sys_status": "NOT_IMPLEMENTED", "final_opened": (root / "FINAL_RELEASE.json").exists(),
               "shakedown": {k: v for k, v in shakedown.items() if k != "pairs"}, "gpu": manifest["gpu"], "versions": manifest["versions"],
               "resources": resources, "claims": claims, "timestamp": utc()}
    write_json(root / "V5_METRICS.json", metrics)
    lines = [f"STATUS / STUDY_ID / SOURCE_SHA / MANIFEST_SHA: {status} / {config['study_id']} / {manifest['source_sha']} / {digest(manifest)}",
             f"MODEL FAMILIES registered / baseline-valid / candidate-correct / paired-valid: {families['registered']} / {families['baseline_valid']} / {families['candidate_correct']} / {families['paired_valid']}",
             f"PAIRS: {json.dumps(counts, sort_keys=True)}", f"DEV/CAL/FINAL: {json.dumps(split_counts, sort_keys=True)}",
             f"FINAL opened: {metrics['final_opened']}", "NEVER / ALWAYS / STRONGEST SIMPLE: see frozen policy table; NOT_ESTIMABLE without independent measured outcomes",
             "ORIGINAL_ACRE: NOT_FITTED; ACRE_SYS: NOT_IMPLEMENTED",
             f"CANONICAL CERTIFICATE: feasible={feasibility['acre_canonical_certificate_feasible']} necessary only; fitted=false; required independent calibration groups={feasibility['required_independent_groups_lower_bound']}",
             f"GPU/CPU/WALL/STORAGE: {json.dumps({'gpu':manifest['gpu'], **resources}, sort_keys=True)}", "",
             f"CLAIM ALLOWED: {'; '.join(claims['allowed'])}", f"CLAIM NOT ALLOWED: {'; '.join(claims['not_allowed'])}", f"NEXT SCIENTIFIC DECISION: {claims['next_scientific_decision']}", ""]
    for c, summary in summaries.items():
        lines += [f"Contrast {c}", "", "Split | Valid families | Positive/negative | Family-equal log benefit | 95% family bootstrap CI", "--- | ---: | --- | ---: | ---"]
        for s in ("development", "calibration", "final"):
            v = summary["by_split"].get(s)
            lines.append(f"{s} | {v['observed_independent_families'] if v else 0} | {str(v['positive_families'])+'/'+str(v['negative_families']) if v else 'NOT_ESTIMABLE'} | {v['family_equal_mean_log_benefit'] if v else 'NOT_ESTIMABLE'} | {v['family_cluster_bootstrap_ci95'] if v else 'NOT_ESTIMABLE'}")
        lines.append("")
    lines += ["Unmeasured and excluded outcomes are not imputed. All-planned effects are unbounded without additional missingness assumptions.",
              "Each family is one independent sampling unit; contexts and sessions improve measurement precision only. Shared architectural ancestry and the small final sample limit external generalization.",
              "Model/input construction is reported separately; primary job time includes preparation and H forwards. Sampled GPU utilization is not continuous GPU busy time.",
              "Root retains raw receipts and references. Export excludes caches/weights and redacts the account workspace; MEMBER_SHA256.json attests exported bytes."]
    (root / "V5_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "PAPER_TABLES.md").write_text("\n".join(lines) + "\n")
    (root / "PAPER_CLAIM_MATRIX.md").write_text("\n".join([f"Allowed: {x}" for x in claims["allowed"]] + [f"Not allowed: {x}" for x in claims["not_allowed"]]) + "\n")
    (root / "EXECUTION_LOG.md").write_text(f"Study: {config['study_id']}\nSource: {manifest['source_sha']}\nTerminal: {status}\nWall including wait: {wall_s:.3f} seconds\nNo started identity retry. Full per-session lifecycle in raw session directories.\n")
    if not (root / "QUALIFICATION_MATRIX.csv").exists(): write_csv(root / "QUALIFICATION_MATRIX.csv", [], ["candidate_id", "eligibility", "baseline_valid", "candidate_correct"])
    write_json(root / "RUN_COMPLETE.json", {"status": status, "timestamp": utc(), "source_sha": manifest["source_sha"], "final_opened": metrics["final_opened"]})
    bundle(root)
    print("\n".join(lines[:12]), flush=True)
