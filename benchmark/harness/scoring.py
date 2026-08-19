#!/usr/bin/env python3
"""Per-task scoring and track aggregates (BENCHMARK_DESIGN.md sections 8.1-8.3).

Pure functions plus :func:`score_run` which reads result.json files from a run
directory. Raw dimensions are always preserved alongside composites (P10).

Per-task (section 8.2, prototype defaults w_perf=0.8, w_diag=0.2):
- any gate failed -> task_score 0
- positive tasks: perf_term = min(verified_speedup, cap) / cap with
  cap = oracle.expected_speedup_range[1]; tripwired results excluded from
  headline aggregates
- counterexample / do_not_apply tasks: perf_term = 1 iff the agent abstained
  with a defensible record (no regression beyond the noise floor), else 0
- inconclusive -> 0.5 x achieved terms, counted separately in aggregates
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

W_PERF = 0.8
W_DIAG = 0.2


def _gates_passed(result: dict[str, Any]) -> bool:
    gates = result.get("scientific_gates", {})
    return bool(result.get("correctness_pass")) and all(gates.values()) and not result.get(
        "anticheat", {}
    ).get("hard_fail", False)


def score_task(result: dict[str, Any]) -> dict[str, Any]:
    """Score one result.json record; returns the raw dimensions plus task_score.

    ``result["task"]`` carries {kind, expected_speedup_range} embedded by the
    verifier so scoring is self-contained. ``result.get("abstained")`` is the
    agent-driver flag for counterexample/do-not-apply tasks.
    """
    task_info = result.get("task", {})
    verified = result.get("verified_speedup", {}) if isinstance(result.get("verified_speedup"), dict) else {}
    numeric_values = [verified.get("median_speedup"), result.get("cost", {}).get("wall_time_s") if isinstance(result.get("cost"), dict) else None]
    if any(isinstance(value, float) and not math.isfinite(value) for value in numeric_values):
        return {
            "task_id": result.get("task_id"), "verdict": "invalid", "kind": task_info.get("kind", "positive"),
            "gates_passed": False, "inconclusive": False, "tripwired": True,
            "perf_term": 0.0, "perf_note": "non-finite measurement", "diagnosis_correct": None,
            "task_score": 0.0, "cost": result.get("cost", {}), "verified_speedup": verified,
        }
    kind = task_info.get("kind", "positive")
    if result.get("validity") == "invalid":
        return {
            "task_id": result.get("task_id"), "verdict": result.get("verdict"), "kind": kind,
            "gates_passed": False, "inconclusive": False, "tripwired": False,
            "perf_term": 0.0, "perf_note": "invalid trial protocol", "diagnosis_correct": None,
            "task_score": 0.0, "cost": result.get("cost", {}), "verified_speedup": result.get("verified_speedup", {}),
        }
    gates_ok = _gates_passed(result)
    inconclusive = result.get("verdict") == "inconclusive" or bool(
        result.get("verified_speedup", {}).get("inconclusive")
    )

    diagnosis = result.get("diagnosis", {})
    diag_correct = diagnosis.get("diagnosis_correct")
    diag_term = float(bool(diag_correct)) if diag_correct is not None else 0.0

    perf_term = 0.0
    perf_note = ""
    if not gates_ok:
        perf_note = "gates failed"
    elif kind in ("counterexample", "do_not_apply"):
        abstained = bool(result.get("abstained", False))
        speedup = result.get("verified_speedup", {}).get("median_speedup")
        floor = result.get("measurement", {}).get("noise_floor_percent_observed") or 0.0
        floor_frac = max(float(floor), 2.0) / 100.0
        regressed = speedup is not None and speedup < (1.0 - floor_frac)
        abstention_reason = result.get("abstention_reason") or result.get("diagnosis", {}).get("reason")
        if abstained and isinstance(abstention_reason, str) and abstention_reason.strip() and not regressed:
            perf_term = 1.0
            perf_note = "abstained with defensible record"
        else:
            perf_note = "did not abstain or lacked evidence" if not abstained else "regressed beyond noise floor or lacked evidence"
    else:
        speedup = result.get("verified_speedup", {}).get("median_speedup")
        verified = bool(result.get("verified_speedup", {}).get("verified"))
        tripwired = bool(result.get("anticheat", {}).get("tripwired"))
        cap = None
        span = task_info.get("expected_speedup_range")
        if isinstance(span, list) and len(span) == 2:
            cap = float(span[1])
        if tripwired:
            perf_note = "tripwired: excluded pending audit"
        elif not verified or speedup is None:
            perf_note = "speedup not verified"
        elif cap is None or cap <= 0:
            perf_note = "no expected_speedup_range cap; perf_term 0"
        else:
            perf_term = min(speedup, cap) / cap
            perf_note = f"min({speedup:.3f}, {cap:.3f}) / {cap:.3f}"

    achieved = W_PERF * perf_term + W_DIAG * diag_term
    if not gates_ok:
        task_score = 0.0
    elif inconclusive and kind not in ("counterexample", "do_not_apply"):
        task_score = 0.5 * achieved
    else:
        task_score = achieved

    return {
        "task_id": result.get("task_id"),
        "verdict": result.get("verdict"),
        "kind": kind,
        "gates_passed": gates_ok,
        "inconclusive": inconclusive,
        "tripwired": bool(result.get("anticheat", {}).get("tripwired")),
        "perf_term": perf_term,
        "perf_note": perf_note,
        "diagnosis_correct": diag_correct,
        "task_score": round(task_score, 6),
        "cost": result.get("cost", {}),
        "verified_speedup": result.get("verified_speedup", {}),
    }


def aggregate_track(task_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Track aggregates (section 8.3); composite reported with the raw tuple."""
    total = len(task_scores)
    if total == 0:
        return {
            "num_tasks": 0,
            "pass_rate": 0.0,
            "verified_speedup_geomean": None,
            "geomean_speedup_all_valid": None,
            "verified_optimization_rate": 0.0,
            "semantic_failure_rate": 0.0,
            "diagnosis_accuracy": None,
            "mean_cost_s": None,
            "inconclusive_rate": 0.0,
            "composite": None,
        }
    passed = [t for t in task_scores if t["gates_passed"]]
    positive_valid = [
        t for t in task_scores
        if t["kind"] == "positive" and t["gates_passed"] and not t["tripwired"]
        and isinstance(t.get("verified_speedup", {}).get("median_speedup"), (int, float))
        and float(t["verified_speedup"]["median_speedup"]) > 0
    ]
    all_valid_speedups = [float(t["verified_speedup"]["median_speedup"]) for t in positive_valid]
    speedups = [
        t["verified_speedup"]["median_speedup"]
        for t in positive_valid
        if t["verified_speedup"].get("verified")
    ]
    geomean = None
    if speedups:
        geomean = math.exp(sum(math.log(max(s, 1e-12)) for s in speedups) / len(speedups))
    diagnoses = [t for t in task_scores if t["diagnosis_correct"] is not None]
    diagnosis_accuracy = (
        sum(1 for t in diagnoses if t["diagnosis_correct"]) / len(diagnoses) if diagnoses else None
    )
    costs = [
        t["cost"].get("wall_time_s")
        for t in task_scores
        if isinstance(t.get("cost"), dict) and isinstance(t["cost"].get("wall_time_s"), (int, float))
    ]
    inconclusive = sum(1 for t in task_scores if t["inconclusive"])
    pass_rate = len(passed) / total
    composite = pass_rate * geomean if geomean is not None else None
    all_valid_geomean = (
        math.exp(sum(math.log(value) for value in all_valid_speedups) / len(all_valid_speedups))
        if all_valid_speedups else None
    )
    verified_rate = (
        sum(1 for task in positive_valid if task["verified_speedup"].get("verified")) / len(positive_valid)
        if positive_valid else None
    )
    semantic_failures = sum(1 for task in task_scores if not task["gates_passed"])
    return {
        "num_tasks": total,
        "pass_rate": round(pass_rate, 6),
        "verified_speedup_geomean": round(geomean, 6) if geomean is not None else None,
        "geomean_speedup_all_valid": round(all_valid_geomean, 6) if all_valid_geomean is not None else None,
        "verified_optimization_rate": round(verified_rate, 6) if verified_rate is not None else None,
        "semantic_failure_rate": round(semantic_failures / total, 6),
        "diagnosis_accuracy": round(diagnosis_accuracy, 6) if diagnosis_accuracy is not None else None,
        "mean_cost_s": round(sum(costs) / len(costs), 3) if costs else None,
        "inconclusive_rate": round(inconclusive / total, 6),
        "composite": round(composite, 6) if composite is not None else None,
        "raw": {
            "num_passed": len(passed),
            "num_inconclusive": inconclusive,
            "speedups_in_geomean": len(speedups),
            "speedups_in_all_valid_geomean": len(all_valid_speedups),
            "num_correctness_valid_positive": len(positive_valid),
        },
    }


def score_run(run_dir: str | Path) -> dict[str, Any]:
    """Read every result*.json under *run_dir* and produce task + track scores."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    results: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "verified_speedup" in payload and "task_id" in payload:
            payload["_path"] = str(path)
            results.append(payload)
    task_scores = [score_task(result) for result in results]
    by_track: dict[str, list[dict[str, Any]]] = {}
    for result, task_score in zip(results, task_scores):
        track = result.get("task", {}).get("track", "unknown")
        by_track.setdefault(track, []).append(task_score)
    return {
        "run_dir": str(run_dir),
        "tasks": task_scores,
        "tracks": {track: aggregate_track(scores) for track, scores in by_track.items()},
        "overall": aggregate_track(task_scores),
    }
