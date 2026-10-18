from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_BUNDLE_SUFFIXES = {".json", ".jsonl", ".log", ".stderr", ".err", ".md", ".csv", ".txt", ".diff"}
_EXCLUDED_PARTS = {"cache", "caches", "__pycache__", "weights", "checkpoints", "checkpoint", "models"}
_EXCLUDED_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".engine", ".so", ".dll"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _is_bundle_evidence(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return (not (lowered_parts & _EXCLUDED_PARTS)
            and path.suffix.lower() not in _EXCLUDED_SUFFIXES
            and path.name not in {"EVIDENCE_BUNDLE_INDEX.json", "MEMBERS_SHA256.json"})


def _markdown(metrics: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]], policies: Sequence[Mapping[str, Any]], extra: Mapping[str, Any]) -> str:
    overall = metrics.get("overall_effects", {})
    denom = metrics.get("failure_denominator", {})
    release = extra.get("release", {})
    identity = extra.get("identity", {})
    status = extra.get("status", "UNSPECIFIED")
    ci = overall.get("family_bootstrap_95_ci")
    ci_text = (f"[{ci['lower']:.6g}, {ci['upper']:.6g}]" if isinstance(ci, Mapping)
               and isinstance(ci.get("lower"), (int, float)) and isinstance(ci.get("upper"), (int, float))
               else "not inferentially estimable" if isinstance(ci, Mapping) else "not estimable")
    lines = [
        "# EvoSPE v3 real intervention results", "",
        f"**Status:** `{status}`  ",
        f"**Intervention:** `{extra.get('selected_intervention', 'unknown')}`; canary {extra.get('canary_passed', 'unknown')}/3  ",
        f"**Identity:** commit `{identity.get('source_commit', 'unknown')}`, config `{identity.get('config_digest', 'unknown')}`, Python `{identity.get('python', 'unknown')}`, GPU `{identity.get('gpu_uuid', 'unknown')}", "",
        "## Paired evidence", "",
        f"Pairs: {len(pairs)} total; {denom.get('valid_pairs', 0)} valid; {denom.get('candidate_failure_pairs', 0)} candidate failures; {denom.get('baseline_invalid_pairs', 0)} baseline invalid; {denom.get('platform_invalid_pairs', 0)} platform invalid; {denom.get('session_noise_unresolved_pairs', 0)} unresolved.", "",
        f"For valid, correct pairs, median speedup is {overall.get('median_speedup')}; context equal-weighted mean log benefit is {overall.get('context_equal_weighted_mean_log_benefit')}; family equal-weighted mean is {overall.get('family_equal_weighted_mean_log_benefit')} (family bootstrap 95% CI {ci_text}; {overall.get('n_families', 0)} independent families).", "",
        "Torchvision rows are `REAL_MODEL_MATERIALIZED_INPUT`; TorchBench rows are `TORCHBENCH_MINIATURE_INPUT`. Both measure real PyTorch model computation paths, but neither establishes dataset accuracy, scientific time-to-quality, or automatic Skill generation efficacy.", "",
        "## Policy evaluation", "",
        "Each policy is scored by split. Candidate execution failures remain in the denominator and an attempted failed deployment is unsafe. Missing deployed outcomes make total net seconds unknown; no finite penalty or synthetic duration is assigned.", "",
        "| Policy | Split | Coverage | Harm / all | Harm / deployed | Unsafe | Missing | Net seconds vs NEVER | Gate seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in policies:
        net = item.get("total_net_seconds_vs_never")
        role = " (scoring only)" if item.get("role") == "scoring_only" else ""
        lines.append(f"| {item.get('policy')}{role} | {item.get('split')} | {item.get('coverage')} | {item.get('harm_unconditional_count')} | {item.get('conditional_harm_rate')} | {item.get('unsafe_deployment_count')} | {item.get('missing_count')} | {net if net is not None else 'unknown'} | {item.get('gate_overhead_seconds')} |")
    acre_status = extra.get("policy_freeze", {}).get("acre", {}).get("status", "ACRE status in POLICY_FREEZE.json")
    lines.extend(["", f"ACRE fit status: `{acre_status}`. If no governed rule store/support certificate was available, ACRE abstains and is reported as not fitted.", "",
                  "## Release and resources", "",
                  f"Final opened: {bool(extra.get('final_opened', False))}; release: `{release.get('status', status)}`; valid development families: {release.get('valid_development_families', 'not supplied')}; valid calibration families: {release.get('valid_calibration_families', 'not supplied')}; admitted families: {release.get('admitted_families', extra.get('baseline_valid', 'not supplied'))}.", "",
                  f"Resources: `{json.dumps(extra.get('resources', {}), sort_keys=True, allow_nan=False)}`.", "",
                  "See `METRICS.json`, `PAIRS.jsonl`, and `POLICY_RESULTS.csv` for machine-readable results and full receipts.", ""])
    return "\n".join(lines)


def _claim_matrix(extra: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    release = extra.get("release", {})
    n = metrics.get("overall_effects", {}).get("n_families", 0)
    qualified = bool(release.get("qualified", extra.get("final_opened", False)))
    label = "Qualified only for the exact evaluated population and frozen intervention." if qualified else "Exploratory only; the preregistered release threshold was not met. No selective or global efficacy claim is supported."
    return ("# Paper claim matrix\n\n"
            f"| Claim | Evidence identity | Sample | Status / limit |\n|---|---|---:|---|\n"
            f"| Fixed-intervention system performance on real model computation paths | `{extra.get('identity', {}).get('source_commit', 'unknown')}` / `{extra.get('identity', {}).get('config_digest', 'unknown')}` | {metrics.get('overall_effects', {}).get('n_pairs', 0)} valid pairs; {n} independent families | {label} No dataset accuracy, scientific time-to-quality, or automatic Skill-generation claim. |\n")


def _collect_bundle_files(root: Path, extra: Mapping[str, Any]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _BUNDLE_SUFFIXES and _is_bundle_evidence(path.relative_to(root)):
            selected[path.relative_to(root).as_posix()] = path
    explicit = extra.get("bundle_files", extra.get("evidence_files", []))
    if isinstance(explicit, (str, Path)):
        explicit = [explicit]
    for raw in explicit:
        path = Path(raw).resolve()
        try:
            name = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            name = Path("external") / path.name
            name = name.as_posix()
        if path.is_file() and _is_bundle_evidence(Path(name)):
            if name in selected and selected[name].resolve() != path:
                raise ValueError(f"duplicate evidence bundle member name: {name}")
            selected[name] = path
    return dict(sorted(selected.items()))


def _write_bundle(root: Path, extra: Mapping[str, Any]) -> None:
    files = _collect_bundle_files(root, extra)
    members = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    index = {"schema": "evospe_v3_evidence_bundle_index_v1", "members": members,
             "excluded": "cache directories and model/checkpoint/weight binary artifacts"}
    index_bytes = _json_bytes(index)
    (root / "EVIDENCE_BUNDLE_INDEX.json").write_bytes(index_bytes)
    (root / "MEMBERS_SHA256.json").write_bytes(_json_bytes(members))
    with tarfile.open(root / "EVIDENCE_BUNDLE.tar.gz", "w:gz") as bundle:
        for name, path in files.items():
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(root / "EVIDENCE_BUNDLE_INDEX.json", arcname="EVIDENCE_BUNDLE_INDEX.json", recursive=False)
        bundle.add(root / "MEMBERS_SHA256.json", arcname="MEMBERS_SHA256.json", recursive=False)


def write_report(root: Path, metrics: dict[str, Any], pairs: list[dict[str, Any]], policy_results: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report_extra = dict(extra)
    freeze_path = root / "POLICY_FREEZE.json"
    if freeze_path.is_file():
        report_extra["policy_freeze"] = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload = dict(metrics)
    payload["run_metadata"] = report_extra
    (root / "METRICS.json").write_bytes(_json_bytes(payload))
    with (root / "PAIRS.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

    fields = ("policy", "role", "split", "n_pairs", "coverage", "deployed", "harm_unconditional_count", "harm_unconditional_rate",
              "harm_deployed_count", "conditional_harm_rate", "unsafe_deployment_count", "candidate_failure_count",
              "candidate_correct_count", "missing_count", "missed_benefit_seconds", "observed_net_seconds_vs_never",
              "total_net_seconds_vs_never", "net_after_gate_seconds", "gate_overhead_seconds", "family_bootstrap_95_ci_seconds")
    with (root / "POLICY_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in policy_results:
            writer.writerow({**row, "family_bootstrap_95_ci_seconds": json.dumps(row.get("family_bootstrap_95_ci_seconds"), sort_keys=True, allow_nan=False)})

    (root / "REPORT.md").write_text(_markdown(metrics, pairs, policy_results, report_extra), encoding="utf-8")
    (root / "PAPER_CLAIM_MATRIX.md").write_text(_claim_matrix(report_extra, metrics), encoding="utf-8")
    _write_bundle(root, report_extra)
