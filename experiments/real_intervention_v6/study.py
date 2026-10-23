"""Freeze a conditional cohort, run paired deployments, and release a pilot final."""
from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import time

from experiments.real_intervention_v5.common import read_json, write_json, digest, utc, write_csv, file_digest
from experiments.real_intervention_v5.run_all import preflight
from experiments.real_intervention_v5.registry import TORCHBENCH
from .audit_v5 import V5_ROOT, V5_SHA
from .eligibility_and_split import build_cohort
from .execution_adapter import run_arm, publish_cache, primary_status


def old_root(study):
    return V5_ROOT / ("EVOSPE_V5_LAYOUT_BF16_20261021_BALANCED_SESSIONS" if study == "A" else "EVOSPE_V5_COMPILE_CACHE_20261021_BALANCED_SESSIONS")


def environment_identity():
    import importlib.metadata
    import torch
    import pynvml
    versions = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "triton", "nvidia-ml-py", "numpy")}
    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
    finally:
        pynvml.nvmlShutdown()
    versions.update({"cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "driver": driver.decode() if isinstance(driver, bytes) else str(driver)})
    return versions


def freeze(root, config, gpu):
    sha = preflight()
    old = old_root(config["study"])
    snapshot = read_json(old / "QUALIFICATION_SNAPSHOT.json")
    exposed = {p["records"][0]["family_id"] for p in read_json(old / "ENGINEERING_SHAKEDOWN.json")["pairs"]}
    cohort = build_cohort(snapshot, sorted(exposed), config["study"], config["split_seed"])
    if not cohort["sample_feasibility"]["independent_pilot"]["feasible"]:
        raise RuntimeError("correctness-qualified independent family cohort insufficient even for pilot")
    versions = environment_identity()
    manifest = {"study_id": config["study_id"], "source_sha": sha, "config": config, "gpu": gpu,
                "cohort": cohort, "versions": versions, "prior_qualification_source": V5_SHA,
                "prior_qualification_sha256": file_digest(old / "QUALIFICATION_SNAPSHOT.json"),
                "qualification_basis": "V5 correctness and mechanism-qualified conditional population; fresh arms recheck numeric/workload identity; no performance labels used for allocation",
                "historical_exposure": "V5 shakedown families prioritized into development. V3/V4 are known exploratory motivation; lineage independence is an assumption, no claim of universally untouched architecture.",
                "topology": {"gpu_workers": 1, "native_cpu_threads": 1, "loader_workers": 0, "seeds_concurrent": 1,
                             "cpu_affinity": sorted(os.sched_getaffinity(0)), "numa_binding": "unbound observed shared host"}}
    path = root / "FROZEN_MANIFEST.json"
    if path.exists():
        if read_json(path) != manifest:
            raise RuntimeError("resume code/env/config/qualification/GPU mismatch")
    else:
        write_json(path, manifest)
    write_json(root / "FAMILY_SPLIT.json", cohort)
    write_csv(root / "QUALIFICATION_AND_SPLIT.csv", cohort["rows"])
    write_json(root / "SAMPLE_SIZE_FEASIBILITY.json", {**cohort["sample_feasibility"],
        "ORIGINAL_ACRE_CERTIFICATION": "MATHEMATICALLY_UNREACHABLE_ON_THIS_COHORT",
        "RESEARCH_POLICY_COMPARISON": "INDEPENDENT_PILOT_UNDERPOWERED", "p_min": .8, "delta_mix": .0125,
        "E35": 1 / (36 * .8**35), "E36": 1 / (37 * .8**36), "minimum_necessary_groups": 42})
    original = {r["family_id"]: r for r in snapshot if r.get("eligibility") == "ELIGIBLE"}
    roster = {r["family_id"]: {**original[r["family_id"]], "split": r["split"]} for r in cohort["rows"]}
    plan = []
    for row in cohort["rows"]:
        family = row["family_id"]
        qualified = next(c["payload"] for c in roster[family]["checks"] if c["batch"] == 1)
        features = dict(qualified["features"])
        features["conv_fraction"] = features["conv2d"] / max(1, features["layers"])
        for horizon in config["horizons"]:
            for session in range(config["sessions_per_context"]):
                order = (int(digest([config["split_seed"], family, horizon])[:8], 16) + session) % 2
                plan.append({"family_id": family, "split": row["split"], "horizon": horizon, "session": session,
                             "pair_id": f"{family}-h{horizon}-s{session}", "order": "AB" if order == 0 else "BA", "features": features})
    write_json(root / "PAIR_PLAN.json", plan)
    return manifest, roster, plan


def spec_for(row, cell, config, gpu, root, arm, scope, engineering):
    qualified = next(c["payload"] for c in row["checks"] if c["batch"] == 1)
    compatibility = {"source_sha": read_json(root / "FROZEN_MANIFEST.json")["source_sha"], "versions": read_json(root / "FROZEN_MANIFEST.json")["versions"],
                     "gpu_uuid": gpu["uuid"], "family": row["family_id"], "shape": row["shape"], "batch": 1, "precision": config["precision"], "compile": config.get("compile"),
                     "weights_sha256": qualified["weights_sha256"], "inputs_sha256": qualified["inputs_sha256"], "context_id": scope}
    spec = {**row, "study_id": config["study_id"], "study": config["study"], "mode": "engineering" if engineering else "measurement", "arm": arm,
            "engineering": engineering, "split": "engineering" if engineering else cell["split"], "seed": config["construction_seed"],
            "batch": 1, "horizon": 1 if arm == "PREFILL" else cell["horizon"], "gpu_uuid": gpu["uuid"], "torchbench_root": TORCHBENCH,
            "correctness": config["correctness"], "timeout_s": config["arm_timeout_s"], "source_sha": compatibility["source_sha"],
            "qualification": {k: qualified[k] for k in ("weights_sha256", "inputs_sha256", "reference_sha256")},
            "results_root": str(root), "context_id": scope, "cache_compatibility": compatibility,
            "cache_path": str(root / "cache" / scope / arm),
            "reference_path": str(old_root(config["study"]) / "references" / f"{row['family_id']}-b1.pt")}
    if spec["split"] == "final":
        release = read_json(root / "FINAL_RELEASE.json")
        spec["final_identity"] = {"release_sha256": digest(release), "policy_sha256": release["policy_sha256"]}
    return spec


def measurable(arm):
    return arm.get("primary_pair_status") in ("VALID", "EXPOSURE_UNCERTAIN")


def measure(root, cell, row, config, gpu, *, engineering=False, deadline=None):
    scope = ("shakedown/" if engineering else f"pairs/{cell['split']}/") + cell["pair_id"]
    directory = root / scope
    summary = directory / "PAIR.json"
    if summary.exists():
        return read_json(summary)
    arms = {}
    def execute(arm, index=0, published=None):
        key = f"{arm}-{index}"
        spec = spec_for(row, cell, config, gpu, root, arm, scope, engineering)
        spec["cache_path"] += f"-{index}"
        if deadline is not None and time.monotonic() >= deadline:
            result = {"execution_status": "NOT_STARTED", "primary_pair_status": "NOT_STARTED", "error": "STUDY_WALL_CAP"}
        else:
            if deadline is not None:
                spec["timeout_s"] = min(spec["timeout_s"], max(1, deadline - time.monotonic()))
            result = run_arm(spec, directory / key, published_cache=published,
                             idle_wait_s=min(config["idle_wait_s"], max(0, deadline - time.monotonic())) if deadline else config["idle_wait_s"], campaign_deadline=deadline)
        arms[key] = result
        return result, spec
    count = 1 if engineering else config["actual_deployment_jobs"]
    published, prefill = None, None
    if config["study"] == "B":
        prefill, pspec = execute("PREFILL")
        if measurable(prefill):
            publication_start = time.monotonic()
            destination = root / "cache" / scope / "published"
            if destination.exists():
                manifest = read_json(destination.with_suffix(".manifest.json"))
                if manifest["compatibility"] != pspec["cache_compatibility"]:
                    raise RuntimeError("existing published artifact identity differs")
                if not (directory / "PREFILL_COST.json").exists():
                    raise RuntimeError("published cache has no closed publication-cost receipt; historical cost UNKNOWN, no replay")
                published = str(destination)
            else:
                published = publish_cache(pspec["cache_path"], destination, pspec["cache_compatibility"])
            write_json(directory / "CACHE_ARTIFACT.json", read_json(destination.with_suffix(".manifest.json")))
            cost_path = directory / "PREFILL_COST.json"
            if cost_path.exists():
                prefill_cost = read_json(cost_path)
            else:
                publication_s = time.monotonic() - publication_start
                prefill_cost = {"worker_parent_wall_s": prefill["parent_wall_s"], "artifact_publication_s": publication_s,
                                "full_prefill_job_s": prefill["parent_wall_s"] + publication_s}
                write_json(cost_path, prefill_cost)
    deployment_rows = []
    if config["study"] == "A" or published:
        for index in range(count):
            names = ["EAGER", "LAYOUT" if config["study"] == "A" else "RELOAD"]
            if (cell["order"] == "BA") ^ (index % 2 == 1):
                names.reverse()
            for arm in names:
                execute(arm, index, published if arm == "RELOAD" else None)
            deployment_rows.append((arms[f"EAGER-{index}"], arms[f"{names[0] if names[0] != 'EAGER' else names[1]}-{index}"]))
        if config["study"] == "B":
            execute("COLD")
    contrasts = [("LAYOUT", 1)] if config["study"] == "A" else [("COLD", 1)] + [(f"PERSISTENT_R{r}", r) for r in config["reported_actual_reuse_counts"]]
    if not engineering:
        contrasts += [(f"PROJECTED_R{r}", r) for r in config["projected_reuse_counts"]]
    records = []
    for contrast, reuse in contrasts:
        selected = []
        if contrast == "COLD":
            selected = [arms.get("EAGER-0", {}), arms.get("COLD-0", {})]
        else:
            for eager, candidate in deployment_rows[:min(reuse, count)]:
                selected += [eager, candidate]
            if prefill is not None:
                selected.append(prefill)
        expected = 2 if contrast == "COLD" else min(reuse, count) * 2 + (1 if prefill is not None else 0)
        usable = len(selected) == expected and bool(selected) and all(measurable(a) for a in selected)
        state = "NOT_STARTED" if not selected else "VALID" if usable and all(a["primary_pair_status"] == "VALID" for a in selected) else "EXPOSURE_UNCERTAIN" if usable else next((a.get("primary_pair_status", "UNKNOWN") for a in selected if not measurable(a)), "UNKNOWN")
        item = {**cell, "context_id": f"{cell['family_id']}-h{cell['horizon']}-{contrast}", "contrast": contrast, "reuse_count": reuse,
                "primary_pair_status": state, "analytically_comparable": usable, "confirmatory_eligible": state == "VALID",
                "evidence_scope": "PROJECTED_COST_SENSITIVITY" if contrast.startswith("PROJECTED") else "EXPLORATORY_SHARED_GPU" if state == "EXPOSURE_UNCERTAIN" else state,
                "s0_s": None, "s1_s": None, "log_benefit": None, "receipt": str(summary.relative_to(root)),
                "actual_reuse_jobs_observed": min(reuse, count), "independent_unit": "architecture_family"}
        if usable:
            if contrast == "COLD":
                s0, s1 = selected[0]["parent_wall_s"], selected[1]["parent_wall_s"]
            else:
                used = deployment_rows[:min(reuse, count)]
                multiplier = reuse / len(used) if contrast.startswith("PROJECTED") else 1
                s0 = sum(e["parent_wall_s"] for e, c in used) * multiplier
                s1 = sum(c["parent_wall_s"] for e, c in used) * multiplier + (prefill_cost["full_prefill_job_s"] if prefill else 0)
            item.update({"s0_s": s0, "s1_s": s1, "net_saved_s": s0 - s1, "log_benefit": math.log(s0 / s1)})
        records.append(item)
    outcome = {"records": records, "arm_states": dict(Counter(a["primary_pair_status"] for a in arms.values())),
               "timers_valid": bool(arms) and all(a.get("timer_crosscheck", {}).get("passed") is True for a in arms.values()) if engineering else None}
    write_json(summary, outcome)
    return outcome
