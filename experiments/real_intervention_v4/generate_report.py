"""Recompute the v4 primary closeout from its frozen ledger and receipts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Mapping, Sequence

from .effect_dataset import family_cluster_bootstrap_ci, summarize_effects
from .matched_sessions import load_pair_observation
from .outcomes import atomic_json, atomic_write_text
from .policy_eval import _deploy, evaluate_policy_set


_POLICIES = (
    "NEVER", "ALWAYS", "APPLICABILITY_RULE", "HORIZON_THRESHOLD",
    "FAMILY_OR_SHAPE_HEURISTIC", "SIMPLE_SUPERVISED", "ORIGINAL_ACRE",
    "REGIME_AWARE_ACRE", "ORACLE",
)
_PAIR_FIELDS = (
    "pair_id", "family_id", "split", "effect_status", "started", "closed",
    "arm_states", "log_benefit", "steady_log_benefit",
    "cold_seconds_saved_per_pair", "classification", "context", "S0", "S1",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: json.dumps(value, sort_keys=True, allow_nan=False)
            if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        })
    atomic_write_text(path, buffer.getvalue())


def _load_pairs(root: Path, manifest: Mapping[str, Any], config: Mapping[str, Any],
                final_opened: bool) -> list[dict[str, Any]]:
    planned = list(manifest.get("planned_pairs", []))
    result = []
    for planned_pair in planned:
        split = str(planned_pair.get("split", "UNKNOWN"))
        # Pair records are per-file, so unopened final outcomes are not read at all.
        if split == "final" and not final_opened:
            result.append({
                **dict(planned_pair),
                "classification": {
                    "arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED",
                    "paired_comparability": "INCOMPLETE", "policy_evidence": "MISSING",
                    "first_failure_stage": "FINAL_NOT_OPENED", "root_cause": None,
                },
                "S0": {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"},
                "S1": {"arm_execution": "NOT_STARTED", "scientific_correctness": "UNASSESSED"},
            })
            continue
        result.append(load_pair_observation(
            planned_pair=planned_pair,
            manifest_digest=str(manifest["manifest_digest"]),
            source_commit=str(manifest["source_commit"]),
            config=config, results_root=root,
            gpu_uuid=str(manifest["selected_gpu_uuid"]),
        ))
    return result


def _write_qualification_outputs(root: Path, universe: Sequence[Mapping[str, Any]],
                                 qualifications: Sequence[Mapping[str, Any]],
                                 batches: Sequence[int]) -> None:
    by_key = {(str(row.get("candidate_id")), int(row.get("batch", -1))): row
              for row in qualifications}
    matrix = []
    registry = []
    for candidate in universe:
        candidate_id = str(candidate.get("candidate_id", ""))
        candidate_rows = []
        for batch in batches:
            row = by_key.get((candidate_id, int(batch)), {})
            candidate_rows.append(row)
            matrix.append({
                "candidate_id": candidate_id, "source": candidate.get("source"),
                "model": candidate.get("model"), "family_id": candidate.get("family_id"),
                "registered_availability": candidate.get("availability"), "batch": batch,
                "status": row.get("status", "NOT_RUN"),
                "eligibility": row.get("eligibility", "NOT_RUN"),
                "reason": row.get("reason"), "first_failure_stage": row.get("first_failure_stage"),
                "conv2d_count": row.get("conv2d_count"),
                "actual_input_shapes": row.get("actual_input_shapes"),
                "model_summary": row.get("model_summary"),
                "kernel_exposure": row.get("kernel_exposure"),
                "runtime_identity": row.get("runtime_identity"),
                "supervisor_receipt_sha256": row.get("supervisor_receipt_sha256"),
                "performance_timing_recorded": row.get("performance_timing_recorded", False),
            })
        eligible = [row.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"
                    for row in candidate_rows]
        baseline_valid = [
            row.get("qualification_complete") is True
            and isinstance(row.get("s0_cross_process_aa"), Mapping)
            and row["s0_cross_process_aa"].get("passed") is True
            for row in candidate_rows
        ]
        registry.append({
            "candidate_id": candidate_id, "source": candidate.get("source"),
            "model": candidate.get("model"), "family_id": candidate.get("family_id"),
            "registered_availability": candidate.get("availability"),
            "baseline_valid_all_batches": bool(baseline_valid) and all(baseline_valid),
            "baseline_loads_and_channels_last_correct": bool(eligible) and all(eligible),
            "baseline_batch_status": [
                (row.get("phase_summary") or {}).get("S0_AA_REPLICATE", "NOT_RUN")
                for row in candidate_rows
            ],
            "batch_eligibilities": [row.get("eligibility", "NOT_RUN") for row in candidate_rows],
        })
    matrix_fields = list(matrix[0]) if matrix else ["candidate_id", "batch", "status", "eligibility"]
    registry_fields = list(registry[0]) if registry else ["candidate_id", "family_id"]
    _atomic_csv(root / "QUALIFICATION_MATRIX.csv", matrix, matrix_fields)
    _atomic_csv(root / "MODEL_REGISTRY.csv", registry, registry_fields)


def _cohort_summary(universe: Sequence[Mapping[str, Any]],
                    qualifications: Sequence[Mapping[str, Any]],
                    manifest: Mapping[str, Any],
                    batches: Sequence[int]) -> dict[str, Any]:
    by_candidate_batch = {
        (str(row.get("candidate_id")), int(row.get("batch", -1))): row
        for row in qualifications
    }
    baseline_candidates = set()
    eligible_candidates = set()
    qualification_complete = set()
    eligibility_counts: dict[str, int] = {}
    for candidate in universe:
        candidate_id = str(candidate.get("candidate_id", ""))
        rows = [by_candidate_batch.get((candidate_id, int(batch)), {}) for batch in batches]
        if rows and all(
            row.get("qualification_complete") is True
            and isinstance(row.get("s0_cross_process_aa"), Mapping)
            and row["s0_cross_process_aa"].get("passed") is True
            for row in rows
        ):
            baseline_candidates.add(candidate_id)
        if rows and all(row.get("qualification_complete") is True for row in rows):
            qualification_complete.add(candidate_id)
        eligibilities = [str(row.get("eligibility", "NOT_RUN")) for row in rows]
        for eligibility in eligibilities:
            eligibility_counts[eligibility] = eligibility_counts.get(eligibility, 0) + 1
        if rows and all(
            row.get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"
            for row in rows
        ):
            eligible_candidates.add(candidate_id)

    baseline_families = {
        str(candidate["family_id"]) for candidate in universe
        if str(candidate.get("candidate_id", "")) in baseline_candidates
        and candidate.get("family_id") not in (None, "")
    }
    eligible_families = {
        str(candidate["family_id"]) for candidate in universe
        if str(candidate.get("candidate_id", "")) in eligible_candidates
        and candidate.get("family_id") not in (None, "")
    }
    return {
        "registered_candidates": len(universe),
        "registered_independent_families": len(manifest.get("registered_families", [])),
        "baseline_valid_candidates_all_batches": len(baseline_candidates),
        "baseline_valid_independent_families": len(baseline_families),
        "qualification_complete_candidates_all_batches": len(qualification_complete),
        "channels_last_eligible_candidates_all_batches": len(eligible_candidates),
        "channels_last_eligible_independent_families": len(eligible_families),
        "unqualified_registered_families": len(manifest.get("unqualified_families", [])),
        "registered_family_split_counts": dict(manifest.get("split_family_counts", {})),
        "eligible_family_split_counts": dict(manifest.get("eligible_family_counts_by_split", {})),
        "qualification_eligibility_counts_by_candidate_batch": eligibility_counts,
    }


def _write_pair_ledger(root: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path = root / "PAIRS.jsonl"
    if path.exists():
        return
    text = "".join(json.dumps({key: row.get(key) for key in _PAIR_FIELDS},
                              sort_keys=True, allow_nan=False) + "\n" for row in pairs)
    atomic_write_text(path, text)


def _policy_comparisons(pairs: Sequence[Mapping[str, Any]], models: Mapping[str, Any],
                        strongest: str | None) -> dict[str, Any]:
    if strongest is None:
        return {"status": "NO_CALIBRATION_SELECTED_SIMPLE_BASELINE", "comparisons": {}}
    comparisons = {}
    for split in ("development", "calibration", "final"):
        rows = [pair for pair in pairs if pair.get("split") == split]
        if split == "final" and not rows:
            continue
        family_policy: dict[str, dict[str, list[float]]] = {}
        for pair in rows:
            if pair.get("effect_status") != "valid" or not pair.get("family_id"):
                continue
            family = str(pair["family_id"])
            family_policy.setdefault(family, {})
            benefit = float(pair["log_benefit"])
            for policy in (strongest, *_POLICIES):
                deploy = _deploy(policy, pair, models)
                if deploy is None:
                    continue
                family_policy[family].setdefault(policy, []).append(benefit if deploy else 0.0)
        family_means = {
            policy: {family: sum(values) / len(values) for family, scores in family_policy.items()
                     if (values := scores.get(policy))}
            for policy in set([strongest, *_POLICIES])
        }
        baseline_values = family_means.get(strongest, {})
        split_comparisons = {}
        for policy, values in family_means.items():
            common = sorted(set(values) & set(baseline_values))
            differences = {family: values[family] - baseline_values[family] for family in common}
            split_comparisons[policy] = {
                "complete_case_family_count": len(common),
                "family_equal_mean_log_benefit_difference_vs_strongest_simple": (
                    sum(differences.values()) / len(differences) if differences else None
                ),
                "family_cluster_bootstrap_ci95": family_cluster_bootstrap_ci(
                    differences, seed=101, replicates=10_000,
                ),
                "inference_scope": "complete-case family-equal difference; missing/unsafe families excluded and separately bounded",
            }
        comparisons[split] = split_comparisons
    return {"strongest_simple_baseline": strongest, "by_split": comparisons}


def _resource_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arm_walls: list[float] = []
    child_cpu: list[float] = []
    gpu_samples = 0
    gpu_utilization: list[float] = []
    for pair in pairs:
        for arm_name in ("S0", "S1"):
            arm = pair.get(arm_name)
            if not isinstance(arm, Mapping):
                continue
            receipt = arm.get("receipt") if isinstance(arm.get("receipt"), Mapping) else arm
            for key, values in (("parent_wall_seconds", arm_walls), ("child_cpu_seconds", child_cpu)):
                value = receipt.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    values.append(float(value))
            telemetry = receipt.get("telemetry", [])
            if isinstance(telemetry, list):
                for sample in telemetry:
                    if not isinstance(sample, Mapping) or sample.get("error"):
                        continue
                    gpu_samples += 1
                    value = sample.get("utilization_percent")
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(parsed):
                        gpu_utilization.append(parsed)
            worker = receipt.get("worker_result")
            if isinstance(worker, Mapping):
                value = worker.get("child_cpu_seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    child_cpu.append(float(value))
    return {
        "arm_parent_wall_seconds_sum": sum(arm_walls) if arm_walls else None,
        "arm_parent_wall_seconds_observed_arms": len(arm_walls),
        "child_cpu_seconds_sum": sum(child_cpu) if child_cpu else None,
        "child_cpu_seconds_observed_arms": len(child_cpu),
        "gpu_telemetry_samples": gpu_samples,
        "gpu_utilization_percent_observed_mean": (
            sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else None
        ),
        "gpu_busy_seconds": None,
        "gpu_busy_seconds_reason": "sampled utilization does not establish continuous busy duration",
        "complete_study_wall_seconds": None,
        "complete_study_wall_seconds_reason": "not derivable from pair receipts alone",
        "storage_bytes": None,
        "storage_bytes_reason": "not recorded as a frozen resource measure by the runner",
    }


def _write_svg(root: Path, summary: Mapping[str, Any]) -> None:
    splits = summary.get("by_split", {})
    names = ["development", "calibration", "final"]
    width, height = 720, 260
    bar_area = 150
    bars = []
    for index, name in enumerate(names):
        info = splits.get(name, {})
        effects = info.get("family_direction_counts", {})
        positive, negative, zero = (int(effects.get(key, 0)) for key in ("positive", "negative", "zero"))
        planned = int(info.get("denominators", {}).get("planned", 0))
        total = positive + negative + zero
        x, y = 180, 40 + index * 65
        bars.append(f'<text x="20" y="{y + 17}">{name}</text>')
        if total:
            cursor = x
            for count, color in ((positive, "#338a55"), (negative, "#bd4b4b"), (zero, "#858585")):
                if count:
                    segment = bar_area * count / max(planned, total)
                    bars.append(f'<rect x="{cursor:.2f}" y="{y}" width="{segment:.2f}" height="22" fill="{color}"/>')
                    cursor += segment
        bars.append(f'<text x="{x + bar_area + 12}" y="{y + 17}">planned={planned}, observed valid families={total}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/><text x="20" y="22" font-size="16">Family mean cold log-benefit direction; non-valid planned pairs remain in denominator</text>'
        + "".join(bars)
        + '<text x="180" y="245" fill="#338a55">positive</text><text x="260" y="245" fill="#bd4b4b">negative</text><text x="340" y="245" fill="#858585">zero</text></svg>'
    )
    atomic_write_text(root / "EFFECT_DIRECTIONS.svg", svg)


def _write_text_reports(root: Path, *, config: Mapping[str, Any], manifest: Mapping[str, Any],
                        source_commit: str, gpu_uuid: str, gate: Mapping[str, Any],
                        shakedown: Mapping[str, Any], summary: Mapping[str, Any],
                        policy_rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> None:
    denominators = summary.get("by_split", {})
    acre = metrics.get("acre", {})
    acre_status = str(acre.get("status", "UNKNOWN"))
    support = acre.get("support_evidence")
    support = support if isinstance(support, Mapping) else {}
    canonical_certificate = acre.get("canonical_support_certificate_created") is True
    certificate_text = (
        "scope-bound canonical support certificate recorded"
        if canonical_certificate else "no canonical support certificate recorded"
    )
    support_text = (
        f"regime={support.get('regime', 'unknown')}, action={support.get('action', 'unknown')}, "
        f"independent calibration families={support.get('independent_families', 'unknown')}, "
        f"scope-bound={support.get('certificate_scope_bound', 'unknown')}, "
        f"certificate digest={support.get('certificate_digest', 'not recorded')}, "
        f"validation certificates={support.get('validation_certificate_count', 'unknown')}, "
        f"governance errors={support.get('governance_errors', 'unknown')}"
    )
    bridge_reason = str(acre.get("reason") or "not recorded")
    acre_claim = (
        "SUPPORTED only within the recorded canonical scope"
        if acre_status == "FITTED" and canonical_certificate
        else "NOT SUPPORTED: ACRE_NOT_FITTED; no canonical support certificate"
        if acre_status == "ACRE_NOT_FITTED"
        else f"{acre_status}; canonical support evidence is incomplete"
    )
    report_lines = [
        f"# {config.get('study_id', 'v4 primary study')} closeout", "",
        f"- Source commit: `{source_commit}`",
        f"- Experiment: `{config.get('study_id')}` / `{config.get('intervention_id')}`",
        f"- Selected GPU UUID: `{gpu_uuid}`",
        f"- Final opened: **{bool(gate.get('final_opened'))}**",
        f"- Gate: **{gate.get('status', 'UNKNOWN')}**",
        f"- Primary final gate requires ACRE certificate: **{bool(gate.get('acre_certificate_required_for_primary_final', False))}**",
        f"- ACRE: **{acre_status}**; {certificate_text}; {support_text}; reason: `{bridge_reason}`",
        "",
        "## Planned denominators and valid effects", "",
        "| Split | Planned | Started | Closed | Valid | Unsafe | Candidate invalid | Baseline invalid | Infrastructure | Noise | Unknown | Not started | Valid/planned | Independent families |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("development", "calibration", "final"):
        part = denominators.get(split, {})
        counts = part.get("denominators", {})
        report_lines.append("| " + " | ".join([
            split, *(str(counts.get(key, 0)) for key in (
                "planned", "started", "closed", "valid", "unsafe", "candidate_invalid",
                "baseline_invalid", "infra", "noise", "unknown", "unstarted",
            )),
            "—" if part.get("paired_valid_rate_over_planned") is None else f"{part['paired_valid_rate_over_planned']:.3f}",
            str(part.get("independent_family_count", 0)),
        ]) + " |")
    cohort = metrics.get("cohort", {})
    registered_split = cohort.get("registered_family_split_counts", {})
    eligible_split = cohort.get("eligible_family_split_counts", {})
    report_lines += [
        "", "## Registered and qualified cohort", "",
        f"- Candidate universe: `{cohort.get('registered_candidates', 0)}` candidates / `{cohort.get('registered_independent_families', 0)}` registered independent families.",
        f"- Baseline-valid across all frozen batches: `{cohort.get('baseline_valid_candidates_all_batches', 0)}` candidates / `{cohort.get('baseline_valid_independent_families', 0)}` families.",
        f"- Channels-last correctness-eligible across all frozen batches: `{cohort.get('channels_last_eligible_candidates_all_batches', 0)}` candidates / `{cohort.get('channels_last_eligible_independent_families', 0)}` families.",
        f"- Registered split family counts: `{registered_split}`; eligible family counts without reassignment: `{eligible_split}`.",
        f"- Qualification-ineligible registered families retained in their frozen assignment: `{cohort.get('unqualified_registered_families', 0)}`.",
        "",
        "## Observed effects", "",
        "| Split | Positive families | Negative families | Zero families | Family-equal mean log-benefit | 95% family bootstrap CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in ("development", "calibration", "final"):
        part = denominators.get(split, {})
        direction = part.get("family_direction_counts", {})
        ci = part.get("family_cluster_bootstrap_ci95_family_equal_mean_log_benefit")
        mean = part.get("family_equal_mean_log_benefit")
        report_lines.append(
            f"| {split} | {direction.get('positive', 0)} | {direction.get('negative', 0)} | {direction.get('zero', 0)} | "
            f"{mean if mean is not None else 'unknown'} | {ci if ci is not None else 'not estimable'} |"
        )
    report_lines += ["", "## Frozen-policy replay", "",
                     "| Split | Policy | Status | Coverage | Known unsafe | Missing deployed | Family-equal net log-benefit | 95% family CI |",
                     "|---|---|---|---:|---:|---:|---:|---|"]
    for row in policy_rows:
        coverage = row.get("deployment_coverage")
        effect = row.get("family_equal_mean_net_log_benefit")
        report_lines.append(
            f"| {row.get('split')} | {row.get('policy')} | {row.get('status')} | "
            f"{coverage if coverage is not None else 'unknown'} | "
            f"{row.get('known_unsafe_deployments') if row.get('known_unsafe_deployments') is not None else 'unknown'} | "
            f"{row.get('missing_deployed_outcomes') if row.get('missing_deployed_outcomes') is not None else 'unknown'} | "
            f"{effect if effect is not None else 'unknown'} | "
            f"{row.get('family_cluster_bootstrap_ci95') if row.get('family_cluster_bootstrap_ci95') is not None else 'not estimable'} |"
        )
    strongest = metrics.get("policy_comparisons", {}).get("strongest_simple_baseline")
    final_compare = metrics.get("policy_comparisons", {}).get("by_split", {}).get("final", {})
    comparison = final_compare.get(strongest, {}) if strongest else {}
    report_lines += [
        "",
        f"Calibration-selected strongest simple baseline: `{strongest or 'none selected'}`.",
        "Independent final complete-case family-equal difference versus that baseline: "
        f"`{comparison.get('family_equal_mean_log_benefit_difference_vs_strongest_simple')}`; "
        f"95% family bootstrap CI `{comparison.get('family_cluster_bootstrap_ci95')}` "
        f"across `{comparison.get('complete_case_family_count', 0)}` common families. Missing/unsafe families are excluded from this complete-case contrast and remain in the separate denominator and sensitivity bounds.",
        "", "## Policy and scientific interpretation", "",
        "Policy scores are offline replays on the same measured potential outcomes. `ORACLE` is an upper-bound scorer only. Missing deployed outcomes and unsafe deployments remain explicit; unknown outcomes are not assigned zero. "
        f"ACRE status is `{acre_status}`; {certificate_text}. Candidate-synthesis evidence is not itself a canonical support certificate.",
        "", "## Resources", "",
        f"- Child CPU seconds (sum where present): `{metrics['resources']['child_cpu_seconds_sum']}` over `{metrics['resources']['child_cpu_seconds_observed_arms']}` arms.",
        f"- Parent arm wall seconds (sum where present): `{metrics['resources']['arm_parent_wall_seconds_sum']}` over `{metrics['resources']['arm_parent_wall_seconds_observed_arms']}` arms.",
        f"- GPU sampled utilization mean: `{metrics['resources']['gpu_utilization_percent_observed_mean']}` from `{metrics['resources']['gpu_telemetry_samples']}` samples; continuous GPU busy seconds remain unknown.",
        "", "## Claim boundary", "",
        "- **Observed:** only the receipt-derived valid pairs and their family-level summaries above.",
        "- **Mechanism hypothesis:** channels-last may alter convolution kernel/layout behavior; mechanism exposure is separately recorded in S1 receipts.",
        "- **External basis:** [PyTorch channels-last memory-format tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) describes supported layout behavior; it is not local efficacy evidence.",
        f"- **Not established:** dataset accuracy, time-to-quality, {('deployable ACRE support/certificate' if not canonical_certificate else 'support outside the recorded ACRE scope')}, or costs absent from receipts.",
        "",
        "The implementation mapping and claim boundaries are also recorded in `PAPER_CLAIM_MATRIX.md`.",
        "", "The engineering shakedown is reported separately and is excluded from primary efficacy denominators.",
    ]
    atomic_write_text(root / "REPORT.md", "\n".join(report_lines) + "\n")

    missing = ["# Missingness and sensitivity", "",
               "Pair-level missingness is retained in `METRICS.json` and `FAILURE_TAXONOMY.csv`. Policy evaluator bounded utility uses `tanh(log(T_S0/T_S1)/ln(2))` with unresolved or unsafe deployed contexts represented by `[-1, 1]`; unsafe deployments remain a separate hard gate.", "",
               "No unknown CPU or GPU cost is imputed as zero. Pair/session wall time, child CPU time, sampled GPU utilization, continuous GPU busy duration, and storage are distinct quantities."]
    atomic_write_text(root / "MISSINGNESS_SENSITIVITY.md", "\n".join(missing) + "\n")
    claims = [
        "# Paper claim matrix", "",
        "| Claim | Status | Boundary |", "|---|---|---|",
        "| Channels-last eager cold/steady effect on measured materialized model inputs | Observed only where valid paired receipts exist | See split-specific family summaries and full planned denominator |",
        f"| ACRE learned applicability boundary | {acre_claim} | status=`{acre_status}`; {certificate_text}; {support_text}; reason=`{bridge_reason}` |",
        "| Dataset accuracy or time-to-quality improvement | NOT MEASURED | No such claim |",
        "| Compile/cache amortization efficacy | NOT RUN by this primary study | Separate preregistered secondary ID |",
        "| External mechanism description | Supported memory-format mechanism, not local efficacy | PyTorch channels-last documentation |",
        "", "## v3-to-v4 implementation mapping", "",
        "| v3 source | v4 counterpart |",
        "|---|---|",
        "| `run_state.py` / `arm_worker.py` | `process_supervisor.py` / `worker.py` / `matched_sessions.py` |",
        "| `paired_study.py` validity and effect rows | `outcomes.py` / `effect_dataset.py` |",
        "| `model_admission.py` | `qualification.py` / `source_registry.py` |",
        "| `policy_bench.py` and canonical ACRE routing | production freeze via `policy_bridge.py`; decisions from `core/acre/engine.py` `AcreEngine.route`, replayed by `policy_eval.py` |",
        "| `analyze_real.py` closeout | `generate_report.py` |",
    ]
    atomic_write_text(root / "PAPER_CLAIM_MATRIX.md", "\n".join(claims) + "\n")


def _bundle(root: Path) -> dict[str, Any]:
    names = (
        "REPORT.md", "METRICS.json", "ACRE_STATUS.json", "MODEL_REGISTRY.csv", "QUALIFICATION_MATRIX.csv",
        "PAIRS.jsonl", "POLICY_FREEZE.json", "POLICY_RESULTS.csv",
        "FAILURE_TAXONOMY.csv", "MISSINGNESS_SENSITIVITY.md", "PAPER_CLAIM_MATRIX.md",
        "EFFECT_DIRECTIONS.svg", "FROZEN_MANIFEST.json", "PRIMARY_LAYOUT_STUDY.json",
        "RUN_IDENTITY.json", "FINAL_GATE.json", "POWER_FEASIBILITY.json",
        "V3_AUDIT_SUMMARY.json", "PRE_EXPERIMENT_TESTS.json", "FAULT_INJECTION.json",
        "CANARY_REPORT.json", "ENGINEERING_SHAKEDOWN.json", "RUN_COMPLETE.json", "RUN_EXIT.json",
    )
    files = [(root / name, name) for name in names if (root / name).is_file()]
    raw_root = root / "raw"
    if raw_root.is_dir():
        for attempt in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            receipt = attempt / "receipt.json"
            if receipt.is_file():
                files.append((receipt, receipt.relative_to(root).as_posix()))
            events = attempt / "events"
            if events.is_dir():
                files.extend(
                    (path, path.relative_to(root).as_posix())
                    for path in sorted(events.glob("*.json")) if path.is_file()
                )
    member_rows = [{"path": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size} for path, member in files]
    member_manifest = {"schema_version": 1, "members": member_rows}
    atomic_json(root / "BUNDLE_MEMBERS_SHA256.json", member_manifest)
    bundle_path = root / "EVIDENCE_BUNDLE.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as archive:
        seen: set[str] = set()
        for path, member in files:
            if member in seen:
                raise ValueError(f"duplicate evidence bundle member: {member}")
            seen.add(member)
            archive.add(path, arcname=member, recursive=False)
        data = json.dumps(member_manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        info = tarfile.TarInfo("BUNDLE_MEMBERS_SHA256.json")
        info.size = len(data)
        import io
        archive.addfile(info, io.BytesIO(data))
    return {"bundle": bundle_path.name, "members": len(files),
            "member_manifest": "BUNDLE_MEMBERS_SHA256.json"}


def write_final_outputs(*, root: Path, config: Mapping[str, Any],
                        manifest: Mapping[str, Any], universe: Sequence[Mapping[str, Any]],
                        qualifications: Sequence[Mapping[str, Any]], gate: Mapping[str, Any],
                        shakedown: Mapping[str, Any], source_commit: str,
                        selected_gpu_uuid: str) -> dict[str, Any]:
    """Write reproducible summaries without refitting ACRE or reading unopened final outcomes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final_opened = bool(gate.get("final_opened"))
    pairs = _load_pairs(root, manifest, config, final_opened)
    summarized = summarize_effects(
        pairs, seed=int(config.get("bootstrap_seed", 101)),
        replicates=int(config.get("bootstrap_repetitions", 10_000)),
    )
    enriched_pairs = summarized["pairs"]
    _write_pair_ledger(root, enriched_pairs)
    _write_qualification_outputs(root, universe, qualifications, config.get("batches", (1, 8)))
    cohort = _cohort_summary(universe, qualifications, manifest, config.get("batches", (1, 8)))

    freeze = _read_json(root / "POLICY_FREEZE.json")
    if freeze is None:
        freeze = {
            "schema_version": 1, "status": "POLICY_SET_NOT_FROZEN",
            "acre_status": "ACRE_NOT_FITTED", "acre_rule_id": None,
            "reason": "runner did not write a pre-final policy freeze; closeout will not fit policies",
            "final_outcomes_read": False,
        }
        atomic_json(root / "POLICY_FREEZE.json", freeze)
    acre_record = _read_json(root / "ACRE_STATUS.json") or {}
    acre_status = str(freeze.get("acre_status", "ACRE_NOT_FITTED"))
    recorded_status = acre_record.get("status")
    if recorded_status is not None and str(recorded_status) != acre_status:
        raise ValueError("ACRE_STATUS and POLICY_FREEZE disagree about the frozen ACRE status")
    bridge_record = freeze.get("acre_bridge", {})
    if not isinstance(bridge_record, Mapping):
        bridge_record = {}
    details = acre_record or bridge_record
    support_evidence = details.get("support_evidence")
    support_evidence = dict(support_evidence) if isinstance(support_evidence, Mapping) else None
    canonical_certificate = details.get("canonical_support_certificate_created") is True
    pair_split = {str(row.get("pair_id")): str(row.get("split")) for row in enriched_pairs}
    for field, expected_split in (("development_pair_ids", "development"),
                                 ("calibration_pair_ids", "calibration")):
        ids = freeze.get(field, [])
        if any(pair_split.get(str(pair_id)) != expected_split for pair_id in ids):
            raise ValueError(f"policy freeze {field} crosses its frozen split")
        expected_ids = {pair_id for pair_id, split in pair_split.items() if split == expected_split}
        if freeze.get("status") != "POLICY_SET_NOT_FROZEN" and set(map(str, ids)) != expected_ids:
            raise ValueError(f"policy freeze {field} does not cover the complete planned split")
    for field, expected in (("study_id", config.get("study_id")),
                            ("intervention_id", config.get("intervention_id")),
                            ("source_commit", source_commit),
                            ("manifest_digest", manifest.get("manifest_digest"))):
        if field in freeze and freeze[field] != expected:
            raise ValueError(f"policy freeze {field} differs from the current study identity")
    if freeze.get("final_outcomes_read") is True:
        raise ValueError("policy freeze declares final outcomes were read; refusing to score it as preregistered")
    models = freeze.get("policies", {}) if isinstance(freeze.get("policies"), Mapping) else {}
    policy_rows = []
    for split in ("development", "calibration", "final"):
        split_rows = [row for row in enriched_pairs if row.get("split") == split]
        if split == "final" and not final_opened:
            split_rows = []
        scores = evaluate_policy_set(split_rows, models=models, acre_status=acre_status)
        policy_rows.extend({"split": split, **score} for score in scores)
    policy_fields = list(policy_rows[0]) if policy_rows else ["split", "policy", "status"]
    _atomic_csv(root / "POLICY_RESULTS.csv", policy_rows, policy_fields)

    failure_rows = []
    for row in enriched_pairs:
        classification = row.get("classification") or {}
        if not isinstance(classification, Mapping):
            classification = {}
        failure_rows.append({
            "pair_id": row.get("pair_id"), "family_id": row.get("family_id"),
            "split": row.get("split"), "effect_status": row.get("effect_status"),
            "planned": True, "started": row.get("started"), "closed": row.get("closed"),
            "S0_execution": (row.get("S0") or {}).get("arm_execution"),
            "S1_execution": (row.get("S1") or {}).get("arm_execution"),
            "scientific_correctness": classification.get("scientific_correctness"),
            "paired_comparability": classification.get("paired_comparability"),
            "policy_evidence": classification.get("policy_evidence"),
            "first_failure_stage": classification.get("first_failure_stage"),
            "root_cause": classification.get("root_cause"),
        })
    failure_fields = list(failure_rows[0]) if failure_rows else ["pair_id", "split", "effect_status"]
    _atomic_csv(root / "FAILURE_TAXONOMY.csv", failure_rows, failure_fields)

    strongest = freeze.get("strongest_simple_baseline")
    comparisons = _policy_comparisons(enriched_pairs, models, strongest)
    resources = _resource_summary(enriched_pairs)
    metrics = {
        "schema_version": 1,
        "study_id": config.get("study_id"), "intervention_id": config.get("intervention_id"),
        "source_commit": source_commit, "manifest_digest": manifest.get("manifest_digest"),
        "selected_gpu_uuid": selected_gpu_uuid, "final_opened": final_opened,
        "gate": dict(gate), "engineering_shakedown": {
            "separate_from_primary": True, "status": shakedown.get("status", shakedown.get("passes")),
            "planned_opportunities": shakedown.get("planned_opportunities"),
            "matched_and_correct": shakedown.get("matched_and_correct"),
            "fraction": shakedown.get("fraction"),
        },
        "planned_pair_count": len(enriched_pairs),
        "cohort": cohort,
        "by_split": summarized["by_split"],
        "policy_results": policy_rows,
        "policy_comparisons": comparisons,
        "acre": {
            "status": acre_status, "fitted": acre_status == "FITTED",
            "canonical_support_certificate_created": canonical_certificate,
            "support_evidence": support_evidence,
            "candidate_synthesis_certificate_created": details.get(
                "candidate_synthesis_certificate_created", False,
            ),
            "production_runner_bridge_invoked": bool(details.get("engine_route_invoked", False)),
            "reason": details.get("reason"),
            "interpretation": (
                "FITTED is scoped to the recorded canonical support evidence"
                if acre_status == "FITTED" and canonical_certificate else
                "ACRE_NOT_FITTED is not a learned abstention or efficacy result"
                if acre_status == "ACRE_NOT_FITTED" else
                "frozen status does not establish a canonical support certificate"
            ),
        },
        "resources": resources,
        "secondary_compile_amortization": {"status": "NOT_RUN", "can_replace_primary": False},
        "report_generated_from": "frozen manifest plus per-pair records under pairs/; unopened final records are not read",
    }
    required_splits = ["development", "calibration"] + (["final"] if final_opened else [])
    split_completion = {
        split: {
            "planned": int(summarized["by_split"].get(split, {}).get("denominators", {}).get("planned", 0)),
            "closed": int(summarized["by_split"].get(split, {}).get("denominators", {}).get("closed", 0)),
        }
        for split in required_splits
    }
    formal_complete = all(item["planned"] == item["closed"] for item in split_completion.values())
    shakedown_planned = int(shakedown.get("planned_opportunities", 0))
    shakedown_required = int(shakedown.get("required_minimum",
                                          config.get("engineering_shakedown", {}).get(
                                              "minimum_model_level_pairs", 12)))
    # A baseline-only cohort smaller than the frozen shakedown minimum is a
    # completed no-go decision; a partially executed shakedown is not.
    shakedown_complete = (
        shakedown_planned < shakedown_required
        or int(shakedown.get("closed_opportunities", -1)) == shakedown_planned
    )
    closeout_complete = formal_complete and shakedown_complete
    confirmatory_analysis_complete = closeout_complete and final_opened
    if confirmatory_analysis_complete:
        completion_reason = None
        study_status = "FINAL_ANALYSIS_COMPLETE"
    elif closeout_complete:
        completion_reason = (
            "final remained unopened because one or more preregistered qualification/power gates failed"
        )
        study_status = "UNDERPOWERED_EXPLORATORY_CLOSEOUT"
    else:
        completion_reason = (
            "one or more opened primary pairs lack two terminal arm receipts"
            if not formal_complete else "engineering shakedown is only partially closed"
        )
        study_status = "INCOMPLETE"
    metrics["analysis_complete"] = confirmatory_analysis_complete
    metrics["confirmatory_analysis_complete"] = confirmatory_analysis_complete
    metrics["exploratory_closeout_complete"] = closeout_complete
    metrics["study_status"] = study_status
    metrics["analysis_completion_reason"] = completion_reason
    metrics["opened_split_completion"] = split_completion
    metrics["engineering_shakedown_complete"] = shakedown_complete
    atomic_json(root / "METRICS.json", metrics)
    _write_svg(root, summarized)
    _write_text_reports(root, config=config, manifest=manifest, source_commit=source_commit,
                        gpu_uuid=selected_gpu_uuid, gate=gate, shakedown=shakedown,
                        summary=summarized, policy_rows=policy_rows, metrics=metrics)
    bundle = _bundle(root)
    return {
        "analysis_complete": confirmatory_analysis_complete,
        "confirmatory_analysis_complete": confirmatory_analysis_complete,
        "closeout_complete": closeout_complete,
        "study_status": study_status,
        "analysis_completion_reason": completion_reason,
        "planned_pairs": len(enriched_pairs),
        "valid_pairs": sum(value.get("denominators", {}).get("valid", 0)
                            for value in summarized["by_split"].values()),
        "final_opened": final_opened,
        "acre_status": acre_status,
        "bundle": bundle,
        "outputs": [
            "REPORT.md", "METRICS.json", "ACRE_STATUS.json", "MODEL_REGISTRY.csv", "QUALIFICATION_MATRIX.csv",
            "PAIRS.jsonl", "POLICY_FREEZE.json", "POLICY_RESULTS.csv",
            "FAILURE_TAXONOMY.csv", "MISSINGNESS_SENSITIVITY.md",
            "PAPER_CLAIM_MATRIX.md", "EFFECT_DIRECTIONS.svg", "EVIDENCE_BUNDLE.tar.gz",
            "BUNDLE_MEMBERS_SHA256.json",
        ],
    }
