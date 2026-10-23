"""Read the frozen V5 receipts without importing any historical effect labels."""
from datetime import datetime
from pathlib import Path
import subprocess

from experiments.real_intervention_v5.common import read_json, write_json

V5_SHA = "d8980253a91a52fa9491fa613eb4c20a1d16e804"
V5_ROOT = Path("/home/workspace/lrh/RESULTS/evospe-v5-20261021-balanced-sessions")


def audit(destination):
    incidents = []
    source = subprocess.check_output(["git", "show", f"{V5_SHA}:experiments/real_intervention_v5/runtime.py"], text=True)
    for study in ("A", "B"):
        root = V5_ROOT / ("EVOSPE_V5_LAYOUT_BF16_20261021_BALANCED_SESSIONS" if study == "A" else "EVOSPE_V5_COMPILE_CACHE_20261021_BALANCED_SESSIONS")
        summary = read_json(root / "ENGINEERING_SHAKEDOWN.json")
        for pair in summary["pairs"]:
            record = pair["records"][0]
            directory = root / record["session_receipt"]
            if record.get("error_code") == "WORKER_GPU_ATTRIBUTION_FAILED":
                terminal = read_json(directory)
                for arm in terminal["arms"]:
                    if arm.get("error_code") != "WORKER_GPU_ATTRIBUTION_FAILED":
                        continue
                    index = terminal["arms"].index(arm)
                    ready_path = directory.parent / f"arm-{index}.result.ready.json"
                    ready = read_json(ready_path)
                    ready_unix = datetime.fromisoformat(ready["timestamp"]).timestamp()
                    samples = []
                    import json
                    with (directory.parent / "telemetry.jsonl").open() as stream:
                        for line in stream:
                            item = json.loads(line)
                            if item.get("phase") == "WAITING_FOR_WORKER_ATTRIBUTION":
                                samples.append(item)
                    last = samples[-1] if samples else {}
                    observed = {"ready": ready, "last_attribution_sample": last,
                                "ready_minus_sample_start_s": ready_unix - last.get("timestamp_unix", ready_unix)}
                    incidents.append({"incident_id": f"{study}:{pair['pair_id']}:attribution", "observed": observed,
                        "source_paths": ["experiments/real_intervention_v5/runtime.py:700-738", "experiments/real_intervention_v5/worker.py:236-247"],
                        "source_commit": V5_SHA, "trace_receipts": [str(directory), str(ready_path)],
                        "leading_hypothesis": "Parent takes NVML sample before reading ready; a ready published during sampling is tested against a pre-ready process snapshot and immediately rejected.",
                        "counter_hypotheses": ["persistent NVML visibility limitation", "PID namespace mismatch", "wrong device"],
                        "diagnostic_test": "Compare sample capture time with ready time; deterministic interleaving fixture; sample again after reading ready in a phased CUDA probe.",
                        "result": "STALE_PRE_READY_SAMPLE" if observed["ready_minus_sample_start_s"] > 0 else "POST_READY_VISIBILITY_UNRESOLVED",
                        "certainty": "receipt_timestamps_and_source_order; underlying NVML latency requires live probe", "blocking": True})
            if study == "B" and not pair["timers_valid"] and record["execution_status"] == "COMPLETE":
                payload = record["arm_payloads"]["CACHE_PREFILL"]
                incidents.append({"incident_id": f"B:{pair['pair_id']}:timer", "observed": {"timing": payload["timing"], "timer_crosscheck": payload["timer_crosscheck"]},
                    "source_paths": ["experiments/real_intervention_v5/worker.py:161-180", "experiments/real_intervention_v5/worker.py:209-219", "experiments/real_intervention_v5/run_all.py:233"],
                    "source_commit": V5_SHA, "trace_receipts": [str(directory)],
                    "leading_hypothesis": "H=1 CACHE_PREFILL leaves compiled callable insufficiently warmed; ten-call manual probe can include subsequent compilation, whereas later benchmark sees warm state.",
                    "counter_hypotheses": ["host scheduling/foreign resource stall", "unequal call count/unit"],
                    "diagnostic_test": "V5 manual already divides by ten; record each call and compiler counters, raw benchmark times and number_per_run in fresh-process prefill reproducer.",
                    "result": "Manual is per-call seconds over ten calls, NOT 400 unnormalized calls. V5 did not retain raw_times or number_per_run; exact historical per-call cause unavailable.",
                    "certainty": "unit interpretation confirmed by frozen source; cold/warm cause pending reproduction", "blocking": True})
    report = {"v5_source": V5_SHA, "v5_raw_modified": False, "incidents": incidents,
              "v4_to_v5": {"v4": "CLI process snapshots and process-group ancestry", "v5": "direct NVML plus process-tree monitor and GO/ACK CUDA-owner handshake; immediate ready check uses earlier sample", "source_order_present": source.index('sample = take_sample("WAITING_FOR_WORKER_ATTRIBUTION"') < source.index('if ready.is_file():')}}
    write_json(Path(destination) / "V6_ROOT_CAUSE_REGISTER.json", report)
    return [{"incident": i["incident_id"], "result": i["result"], "ready_minus_sample_start_s": i["observed"].get("ready_minus_sample_start_s")} for i in incidents]
