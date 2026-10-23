"""One controller, primary B then independent A if a lab GPU is available."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.real_intervention_v5.common import read_json, write_json, write_csv, utc
from experiments.real_intervention_v5.run_all import preflight, environment
from .gpu_attestation import idle_device
from .report import bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-parent", type=Path, required=True)
    parser.add_argument("--tests-receipt", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path)
    args = parser.parse_args()
    source = preflight()
    tests = read_json(args.tests_receipt)
    if tests["source_sha"] != source or tests["exit_code"] != 0:
        raise RuntimeError("campaign needs passed current-source targeted tests")
    root = args.results_parent.resolve()
    if not root.is_relative_to(Path("/home/workspace/lrh/RESULTS")) or "evospe-v6" not in root.name:
        raise ValueError("new V6 parent required")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "CAMPAIGN_COMPLETE.json").exists():
            print(read_json(root / "CAMPAIGN_COMPLETE.json"))
            return
        launch = {"source_sha": source, "pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0), "timestamp": utc()}
        if (root / "CAMPAIGN_LAUNCH.json").exists() and read_json(root / "CAMPAIGN_LAUNCH.json")["source_sha"] != source:
            raise RuntimeError("campaign source changed")
        if not (root / "CAMPAIGN_LAUNCH.json").exists():
            write_json(root / "CAMPAIGN_LAUNCH.json", launch)
        write_json(root / "CPU_TESTS.json", tests)
        studies, rows = {}, []
        for study, filename in (("B", "V6_MAIN_COMPILE.json"), ("A", "V6_LAYOUT_SECONDARY.json")):
            config_path = Path(__file__).parent / "config" / filename
            config = read_json(config_path)
            child_root = root / ("evospe-v6-compile-main" if study == "B" else "evospe-v6-layout-secondary")
            if study == "A" and "RESOURCE_BLOCKED" in str(studies.get("B", {}).get("blocker", "")):
                available, inventory = idle_device()
                if available is None:
                    studies[study] = {"status": "NOT_RUN_RESOURCE_CONSTRAINT", "gpu_inventory": inventory,
                                      "reason": "No idle GPU after primary's bounded admission; secondary conditional on lab resources"}
                    break
            command = [sys.executable, "-u", "-m", "experiments.real_intervention_v6.run_all", "--config", str(config_path),
                       "--results-root", str(child_root), "--tests-receipt", str(args.tests_receipt), "--resume-safe"]
            if args.diagnostics_root:
                command += ["--diagnostics-root", str(args.diagnostics_root)]
            with (root / f"study_{study}.log").open("a") as log:
                process = subprocess.Popen(command, env={**os.environ, **environment()}, stdout=log, stderr=subprocess.STDOUT)
                write_json(root / "CAMPAIGN_PROGRESS.json", {"study": study, "child_pid": process.pid, "source_sha": source, "timestamp": utc()})
                code = process.wait()
            if not (child_root / "V6_METRICS.json").exists():
                studies[study] = {"status": "CONTROLLER_EXIT_WITHOUT_CLOSED_STUDY", "exit_code": code, "root": str(child_root)}
                break
            studies[study] = read_json(child_root / "V6_METRICS.json")
            rows += [{**json.loads(line), "study": study} for line in (child_root / "PAIRS.jsonl").read_text().splitlines()]
        write_json(root / "V6_METRICS.json", {"source_sha": source, "studies": studies, "studies_never_pooled": True, "v5_raw_modified": False})
        with (root / "PAIRS.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        write_csv(root / "PAIRS.csv", rows)
        lines = ["# EvoSPE V6 campaign", "", f"Source: {source}; V5 raw modified: NO", "", "Study | Status | Comparable actual contrast rows | Independent measured families | Final released", "--- | --- | ---: | ---: | ---"]
        for name, metrics in studies.items():
            lines.append(f"{name} | {metrics['status']} | {metrics.get('analytically_comparable_rows', 0)} | {metrics.get('independent_measured_families', 0)} | {metrics.get('final_released', False)}")
            if metrics.get("blocker"):
                lines.append(f"\n{name} blocker: {metrics['blocker']}\n")
        lines += ["", "See each study V6_FINAL_REPORT.md and V6_METRICS.json for every contrast/split, signed family effects, bootstrap CIs and frozen-policy results. Studies are independent and never pooled.",
                  "Original ACRE: mathematically unreachable on this conditional cohort (necessary42 calibration groups); this does not gate the separate fixed-intervention pilot.",
                  "UNKNOWN co-tenancy retains exploratory scope; R100 is projection, not observed100-job deployment. No task-accuracy claim."]
        (root / "V6_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
        (root / "PAPER_CLAIM_MATRIX.md").write_text("V5 historical readiness remains unchanged. V6 effect claims require the recorded prospective final and exposure grade.\nNo original ACRE certificate, task-accuracy claim or unobserved deployment claim.\n")
        source_root = root / "analysis_source"
        for module in ("real_intervention_v6/report.py", "real_intervention_v6/policy_experiment.py", "real_intervention_v5/common.py"):
            destination = source_root / "experiments" / module
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path("experiments") / module, destination)
        shutil.copyfile(Path(__file__).with_name("recompute.py"), root / "recompute.py")
        for package in ("experiments", "experiments/real_intervention_v5", "experiments/real_intervention_v6"):
            (source_root / package / "__init__.py").write_text("")
        raw_index = []
        for child in root.glob("evospe-v6-*"):
            if (child / "RAW_ATTEMPT_INDEX.json").exists():
                raw_index += [{**r, "study_root": child.name} for r in read_json(child / "RAW_ATTEMPT_INDEX.json")]
        write_json(root / "RAW_ATTEMPT_INDEX.json", raw_index)
        primary_root = root / "evospe-v6-compile-main"
        if (primary_root / "V6_ROOT_CAUSE_REGISTER.json").exists():
            shutil.copyfile(primary_root / "V6_ROOT_CAUSE_REGISTER.json", root / "V6_ROOT_CAUSE_REGISTER.json")
        write_json(root / "CAMPAIGN_COMPLETE.json", {"source_sha": source, "study_statuses": {s: m["status"] for s, m in studies.items()}, "timestamp": utc()})
        bundle(root)
        print(read_json(root / "CAMPAIGN_COMPLETE.json"))


if __name__ == "__main__":
    main()
