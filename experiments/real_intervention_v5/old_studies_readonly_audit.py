"""Read-only denominator audit for a frozen real-intervention V4 study root."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _load_pairs(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"line {line_number} is not a JSON object")
                    rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return rows, None


def _load_qualifications(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream)), None
    except OSError as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _value(value: Any, source: str, verified: bool = True) -> dict[str, Any]:
    return {"value": value, "status": "verified" if verified else "not_verified", "source": source}


def _read_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    value, error = _load_json(path)
    if error is None and not isinstance(value, dict):
        return None, "ValueError: expected a JSON object"
    return value, error


def _arm_raw_verification(
    root: Path,
    pair_id: str,
    arm_name: str,
    arm: Mapping[str, Any],
    selected_gpu_uuid: Any,
) -> dict[str, Any]:
    embedded_events = arm.get("events")
    embedded_events = embedded_events if isinstance(embedded_events, list) else []
    started = [
        event for event in embedded_events
        if isinstance(event, Mapping) and event.get("kind") == "started"
    ]
    if not started:
        return {}

    raw_name = f"primary-{pair_id}-{arm_name}"
    raw_dir = root / "raw" / raw_name
    source_commits = set()
    gpu_uuids = set()
    embedded_identity_events = []
    for event in embedded_events:
        identity = event.get("identity") if isinstance(event, Mapping) else None
        if isinstance(identity, Mapping):
            embedded_identity_events.append(identity)
            if identity.get("source_commit"):
                source_commits.add(str(identity["source_commit"]))
            if identity.get("gpu_uuid"):
                gpu_uuids.add(str(identity["gpu_uuid"]))

    checks: dict[str, bool | str] = {}
    expected_attempt_id = raw_name
    arm_identity = arm.get("identity")
    checks["embedded_identity_consistent"] = bool(
        len(source_commits) == 1
        and len(gpu_uuids) == 1
        and embedded_identity_events
        and isinstance(arm_identity, Mapping)
        and arm_identity in embedded_identity_events
        and arm.get("source_commit") in source_commits
        and arm.get("runtime_gpu_uuid") in gpu_uuids
        and all(
            identity.get("attempt_id") == expected_attempt_id
            and identity.get("pair_id") == pair_id
            and identity.get("arm") == arm_name
            and identity.get("source_commit") in source_commits
            and identity.get("gpu_uuid") in gpu_uuids
            for identity in embedded_identity_events
        )
    )
    checks["raw_receipt_exists"] = (raw_dir / "receipt.json").is_file()
    checks["raw_result_exists"] = (raw_dir / "result.json").is_file()

    raw_events = []
    raw_event_error = None
    event_dir = raw_dir / "events"
    if event_dir.is_dir():
        for event_path in sorted(event_dir.glob("*.json")):
            event, error = _read_object(event_path)
            if error:
                raw_event_error = f"{event_path.name}: {error}"
                break
            raw_events.append(event)
    else:
        raw_event_error = "events directory missing"
    raw_identity_events = [
        event.get("identity") for event in raw_events
        if isinstance(event.get("identity"), Mapping)
    ]
    checks["raw_event_identities_match_embedded"] = (
        all(identity in embedded_identity_events for identity in raw_identity_events)
        and all(identity in raw_identity_events for identity in embedded_identity_events)
        if raw_event_error is None else False
    )
    raw_commits = {str(identity.get("source_commit")) for identity in raw_identity_events if identity.get("source_commit")}
    raw_gpus = {str(identity.get("gpu_uuid")) for identity in raw_identity_events if identity.get("gpu_uuid")}
    checks["event_source_commit_matches"] = raw_commits == source_commits and len(source_commits) == 1
    checks["event_gpu_uuid_matches"] = raw_gpus == gpu_uuids and len(gpu_uuids) == 1
    checks["selected_gpu_uuid_matches_events"] = (
        selected_gpu_uuid is not None and str(selected_gpu_uuid) in gpu_uuids
    )

    embedded_terminal = next(
        (event for event in embedded_events if isinstance(event, Mapping) and event.get("kind") == "terminal"),
        None,
    )
    raw_terminal = next((event for event in raw_events if event.get("kind") == "terminal"), None)
    checks["raw_terminal_event_exists"] = raw_terminal is not None
    terminal_fields = ("kind", "identity", "arm_execution", "process_returncode", "sequence", "receipt_sha256")
    checks["terminal_event_matches"] = bool(
        embedded_terminal is not None
        and raw_terminal is not None
        and all(
            field in embedded_terminal
            and field in raw_terminal
            and embedded_terminal[field] == raw_terminal[field]
            for field in terminal_fields
        )
    )

    raw_receipt, receipt_error = _read_object(raw_dir / "receipt.json")
    raw_result, result_error = _read_object(raw_dir / "result.json")
    embedded_worker = arm.get("worker_result")
    raw_receipt_worker = raw_receipt.get("worker_result") if raw_receipt else None
    worker_fields = (
        "arm_execution", "scientific_correctness", "weights_sha256",
        "inputs_sha256", "reference_sha256",
    )
    worker_checks = {}
    for field in worker_fields:
        embedded_value = embedded_worker.get(field) if isinstance(embedded_worker, Mapping) else None
        raw_value = raw_result.get(field) if raw_result else None
        receipt_value = raw_receipt_worker.get(field) if isinstance(raw_receipt_worker, Mapping) else None
        worker_checks[field] = (
            embedded_value is not None
            and raw_value is not None
            and receipt_value is not None
            and embedded_value == raw_value == receipt_value
        )
    checks["worker_fields_match"] = all(worker_checks.values())
    checks["arm_execution_matches_worker_result"] = bool(
        isinstance(embedded_worker, Mapping)
        and arm.get("arm_execution") is not None
        and arm.get("arm_execution") == embedded_worker.get("arm_execution")
    )
    checks["raw_worker_result_matches_embedded"] = bool(
        isinstance(embedded_worker, Mapping)
        and isinstance(raw_receipt_worker, Mapping)
        and embedded_worker == raw_receipt_worker
    )
    checks["raw_worker_result_matches_result_file"] = bool(
        isinstance(raw_receipt_worker, Mapping)
        and isinstance(raw_result, Mapping)
        and raw_receipt_worker == raw_result
    )
    raw_receipt_identity = raw_receipt.get("identity") if raw_receipt else None
    checks["raw_receipt_identity_matches_events"] = bool(
        isinstance(raw_receipt_identity, Mapping)
        and raw_receipt_identity in raw_identity_events
    )
    checks["raw_source_commit_matches_events"] = bool(
        raw_receipt
        and raw_receipt.get("source_commit") in source_commits
        and isinstance(raw_receipt_identity, Mapping)
        and raw_receipt_identity.get("source_commit") in source_commits
    )
    checks["raw_gpu_uuid_matches_events"] = bool(
        raw_receipt
        and raw_receipt.get("runtime_gpu_uuid") in gpu_uuids
        and isinstance(raw_receipt_identity, Mapping)
        and raw_receipt_identity.get("gpu_uuid") in gpu_uuids
    )
    verified = all(value is True for value in checks.values())
    return {
        "pair_id": pair_id,
        "arm": arm_name,
        "attempt_id": expected_attempt_id,
        "status": "verified" if verified else "not_verified",
        "source_commit": next(iter(source_commits)) if len(source_commits) == 1 else None,
        "gpu_uuid": next(iter(gpu_uuids)) if len(gpu_uuids) == 1 else None,
        "raw_path": str(raw_dir),
        "checks": checks,
        "worker_field_checks": worker_checks,
        "errors": [error for error in (raw_event_error, receipt_error, result_error) if error],
    }


def _audit_raw_receipts(root: Path, pairs: list[dict[str, Any]], pairs_error: str | None) -> dict[str, Any]:
    if pairs_error is not None:
        summary = {
            "status": "not_verified",
            "reason": "PAIRS.jsonl unavailable or invalid",
            "started_arms": None,
            "verified_arms": None,
            "not_verified_arms": None,
            "source_commit_counts": {},
            "gpu_uuid_counts": {},
            "failed_check_counts": {},
        }
        return {
            "status": "not_verified",
            "reason": "PAIRS.jsonl unavailable or invalid",
            "started_arms": None,
            "verified_arms": None,
            "not_verified_arms": None,
            "source_commit_counts": {},
            "gpu_uuid_counts": {},
            "summary": summary,
            "arms": [],
        }
    arm_results = []
    started_pairs_missing_arm_events = 0
    for pair in pairs:
        pair_id = str(pair.get("pair_id") or "")
        pair_arm_count = 0
        for arm_name in ("S0", "S1"):
            arm = pair.get(arm_name)
            if not isinstance(arm, Mapping):
                continue
            result = _arm_raw_verification(
                root, pair_id, arm_name, arm, pair.get("selected_gpu_uuid")
            )
            if result:
                pair_arm_count += 1
                arm_results.append(result)
        classification = pair.get("classification")
        if (
            isinstance(classification, Mapping)
            and classification.get("arm_execution") == "COMPLETE"
            and pair_arm_count < 2
        ):
            started_pairs_missing_arm_events += 1
    verified_count = sum(row["status"] == "verified" for row in arm_results)
    source_counts = Counter(row["source_commit"] or "not_verified" for row in arm_results)
    gpu_counts = Counter(row["gpu_uuid"] or "not_verified" for row in arm_results)
    missing_arm_event = started_pairs_missing_arm_events > 0
    failed_checks = Counter(
        check_name
        for arm_result in arm_results
        for check_name, passed in arm_result["checks"].items()
        if passed is not True
    )
    summary = {
        "status": (
            "verified"
            if arm_results and verified_count == len(arm_results) and not missing_arm_event
            else "not_verified"
        ),
        "started_arms": len(arm_results),
        "verified_arms": verified_count,
        "not_verified_arms": len(arm_results) - verified_count,
        "started_pairs_missing_arm_start_events": started_pairs_missing_arm_events,
        "source_commit_counts": dict(source_counts),
        "gpu_uuid_counts": dict(gpu_counts),
        "failed_check_counts": dict(failed_checks),
    }
    return {
        "status": (
            "verified"
            if arm_results and verified_count == len(arm_results) and not missing_arm_event
            else "not_verified"
        ),
        "started_arms": len(arm_results),
        "verified_arms": verified_count,
        "not_verified_arms": len(arm_results) - verified_count,
        "started_pairs_missing_arm_start_events": started_pairs_missing_arm_events,
        "source_commit_counts": dict(source_counts),
        "gpu_uuid_counts": dict(gpu_counts),
        "summary": summary,
        "arms": arm_results,
    }


def audit_v4(root: Path) -> dict[str, Any]:
    """Parse frozen V4 metrics, pair receipts and qualification rows without writes."""
    root = Path(root)
    metrics, metrics_error = _load_json(root / "METRICS.json")
    pairs, pairs_error = _load_pairs(root / "PAIRS.jsonl")
    qualifications, qualifications_error = _load_qualifications(
        root / "QUALIFICATION_MATRIX.csv"
    )
    registry, registry_error = _load_qualifications(root / "MODEL_REGISTRY.csv")
    manifest, manifest_error = _load_json(root / "FROZEN_MANIFEST.json")

    errors = {
        name: error
        for name, error in (
            ("metrics", metrics_error),
            ("pairs", pairs_error),
            ("qualifications", qualifications_error),
            ("registry", registry_error),
            ("manifest", manifest_error),
        )
        if error is not None
    }
    if metrics_error is None and not isinstance(metrics, Mapping):
        metrics_error = "ValueError: METRICS.json must contain an object"
        errors["metrics"] = metrics_error
        metrics = None
    if manifest_error is None and not isinstance(manifest, Mapping):
        manifest_error = "ValueError: FROZEN_MANIFEST.json must contain an object"
        errors["manifest"] = manifest_error
        manifest = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(root),
        "read_only": True,
        "errors": errors,
        "counts": {
            name: _value(None, "source unavailable", verified=False)
            for name in (
                "registered_families", "baseline_valid_families",
                "s1_correctness_eligible_families", "planned_pairs", "started_pairs",
                "closed_pairs", "valid_pairs", "unsafe_pairs", "infrastructure_pairs",
                "baseline_invalid_pairs", "candidate_invalid_pairs",
                "exposure_confounded_pairs", "telemetry_unknown_pairs",
                "not_started_pairs", "ineligible_pairs", "pair_rows",
                "started_pairs_from_rows", "closed_pairs_from_rows",
                "valid_pairs_from_rows", "unsafe_pairs_from_rows",
                "infrastructure_pairs_from_rows", "baseline_invalid_pairs_from_rows",
                "candidate_invalid_pairs_from_rows", "exposure_confounded_pairs_from_rows",
                "telemetry_unknown_pairs_from_rows", "not_started_pairs_from_rows",
                "ineligible_pairs_from_rows", "registered_families_from_manifest",
                "baseline_valid_families_from_registry",
            )
        },
        "denominator_checks": {},
    }

    if metrics_error is None:
        cohort = metrics.get("cohort", {})
        denominator_totals: Counter[str] = Counter()
        for split in ("development", "calibration", "final"):
            denominator_totals.update(
                metrics.get("by_split", {}).get(split, {}).get("denominators", {})
            )
        result["counts"].update({
            "registered_families": _value(
                cohort.get("registered_independent_families"), "METRICS.json:cohort", verified=False
            ),
            "baseline_valid_families": _value(
                cohort.get("baseline_valid_independent_families"), "METRICS.json:cohort", verified=False
            ),
            "s1_correctness_eligible_families": _value(
                cohort.get("channels_last_eligible_independent_families"), "METRICS.json:cohort", verified=False
            ),
            "planned_pairs": _value(denominator_totals.get("planned"), "METRICS.json:by_split", verified=False),
            "started_pairs": _value(denominator_totals.get("started"), "METRICS.json:by_split", verified=False),
            "closed_pairs": _value(denominator_totals.get("closed"), "METRICS.json:by_split", verified=False),
            "valid_pairs": _value(denominator_totals.get("valid"), "METRICS.json:by_split", verified=False),
            "unsafe_pairs": _value(denominator_totals.get("unsafe"), "METRICS.json:by_split", verified=False),
            "infrastructure_pairs": _value(denominator_totals.get("infra"), "METRICS.json:by_split", verified=False),
            "baseline_invalid_pairs": _value(denominator_totals.get("baseline_invalid"), "METRICS.json:by_split", verified=False),
            "candidate_invalid_pairs": _value(denominator_totals.get("candidate_invalid"), "METRICS.json:by_split", verified=False),
            "exposure_confounded_pairs": _value(denominator_totals.get("noise"), "METRICS.json:by_split", verified=False),
            "telemetry_unknown_pairs": _value(denominator_totals.get("unknown"), "METRICS.json:by_split", verified=False),
            "not_started_pairs": _value(denominator_totals.get("unstarted"), "METRICS.json:by_split", verified=False),
            "ineligible_pairs": _value(denominator_totals.get("ineligible"), "METRICS.json:by_split", verified=False),
        })
        result["metrics_reported_denominators"] = dict(denominator_totals)

    if pairs_error is None:
        pair_ids = [str(row.get("pair_id") or "") for row in pairs]
        unique_ids = len(pair_ids) == len(set(pair_ids)) and all(pair_ids)
        result["counts"]["pair_rows"] = _value(len(pairs), "PAIRS.jsonl")
        result["denominator_checks"]["pair_ids_unique_and_present"] = unique_ids
        classifications = []
        for row in pairs:
            classification = row.get("classification")
            if isinstance(classification, Mapping):
                arm_execution = classification.get("arm_execution")
                comparability = classification.get("paired_comparability")
                if arm_execution and comparability:
                    classifications.append((str(arm_execution), str(comparability)))
        classification_counts = Counter(classifications)
        result["pair_classification_counts"] = {
            f"{execution}/{comparability}": count
            for (execution, comparability), count in sorted(classification_counts.items())
        }
        all_classified = len(classifications) == len(pairs)
        complete_pairs = sum(execution == "COMPLETE" for execution, _ in classifications)
        result["counts"]["started_pairs_from_rows"] = _value(
            complete_pairs if all_classified else None,
            "PAIRS.jsonl:classification.arm_execution",
            verified=all_classified,
        )
        result["counts"]["closed_pairs_from_rows"] = _value(
            complete_pairs if all_classified else None,
            "PAIRS.jsonl:classification.arm_execution",
            verified=all_classified,
        )
        class_to_category = {
            ("COMPLETE", "MATCHED"): "valid_pairs_from_rows",
            ("COMPLETE", "CONFOUNDED"): "exposure_confounded_pairs_from_rows",
            ("COMPLETE", "UNKNOWN"): "telemetry_unknown_pairs_from_rows",
            ("NOT_STARTED", "INCOMPLETE"): "not_started_pairs_from_rows",
        }
        mapped = all_classified and all(key in class_to_category for key in classifications)
        row_category_keys = (
            "valid_pairs_from_rows", "unsafe_pairs_from_rows",
            "infrastructure_pairs_from_rows", "baseline_invalid_pairs_from_rows",
            "candidate_invalid_pairs_from_rows", "exposure_confounded_pairs_from_rows",
            "telemetry_unknown_pairs_from_rows", "not_started_pairs_from_rows",
            "ineligible_pairs_from_rows",
        )
        row_category_counts = Counter()
        for key in classifications:
            if key in class_to_category:
                row_category_counts[class_to_category[key]] += 1
        for key in row_category_keys:
            result["counts"][key] = _value(
                row_category_counts[key] if mapped else None,
                "PAIRS.jsonl:classification",
                verified=mapped,
            )
        result["denominator_checks"]["pair_outcome_categories_close_to_pair_rows"] = (
            mapped and sum(row_category_counts.values()) == len(pairs)
        )
        if metrics_error is None:
            for row_count, metric_count in (
                ("pair_rows", "planned_pairs"),
                ("started_pairs_from_rows", "started_pairs"),
                ("closed_pairs_from_rows", "closed_pairs"),
                ("valid_pairs_from_rows", "valid_pairs"),
                ("unsafe_pairs_from_rows", "unsafe_pairs"),
                ("infrastructure_pairs_from_rows", "infrastructure_pairs"),
                ("baseline_invalid_pairs_from_rows", "baseline_invalid_pairs"),
                ("candidate_invalid_pairs_from_rows", "candidate_invalid_pairs"),
                ("exposure_confounded_pairs_from_rows", "exposure_confounded_pairs"),
                ("telemetry_unknown_pairs_from_rows", "telemetry_unknown_pairs"),
                ("not_started_pairs_from_rows", "not_started_pairs"),
                ("ineligible_pairs_from_rows", "ineligible_pairs"),
            ):
                observed = result["counts"].get(row_count, {}).get("value")
                reported = result["counts"].get(metric_count, {}).get("value")
                result["denominator_checks"][metric_count] = (
                    observed == reported if observed is not None and reported is not None else "not_verified"
                )
                result["counts"][metric_count]["status"] = (
                    "verified" if result["denominator_checks"][metric_count] is True else "not_verified"
                )
            categories = (
                "valid", "unsafe", "infra", "baseline_invalid", "candidate_invalid",
                "noise", "unknown", "unstarted", "ineligible",
            )
            result["denominator_checks"]["reported_outcome_categories_close_to_planned"] = (
                sum(denominator_totals.get(category, 0) for category in categories)
                == denominator_totals.get("planned")
            )

    if qualifications_error is None:
        by_family_candidate: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in qualifications:
            family = row.get("family_id", "")
            candidate = row.get("candidate_id", "")
            batch = row.get("batch", "")
            if family and candidate and batch:
                by_family_candidate[family].setdefault(candidate, {})[batch] = row.get("eligibility", "")
        eligible_families = set()
        for family, candidates in by_family_candidate.items():
            for batches in candidates.values():
                if {"1", "8"} <= batches.keys():
                    if all(
                        batches[batch] == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"
                        for batch in ("1", "8")
                    ):
                        eligible_families.add(family)
        result["qualification_row_count"] = len(qualifications)
        result["qualification_candidate_count"] = len({
            row.get("candidate_id", "") for row in qualifications if row.get("candidate_id")
        })
        result["qualification_family_counts_from_rows"] = {
            "baseline_valid_families": "not_verified_from_QUALIFICATION_MATRIX.csv",
            "s1_correctness_eligible_families": len(eligible_families),
            "status": "verified_from_candidate_rows_requiring_batches_1_and_8",
        }
        if metrics_error is None:
            result["denominator_checks"]["s1_correctness_eligible_families"] = (
                len(eligible_families)
                == result["counts"]["s1_correctness_eligible_families"]["value"]
            )
            result["counts"]["s1_correctness_eligible_families"]["status"] = (
                "verified"
                if result["denominator_checks"]["s1_correctness_eligible_families"]
                else "not_verified"
            )

    if registry_error is None:
        baseline_families = {
            row.get("family_id", "") for row in registry
            if row.get("family_id") and row.get("baseline_valid_all_batches", "").lower() == "true"
        }
        result["counts"]["baseline_valid_families_from_registry"] = _value(
            len(baseline_families), "MODEL_REGISTRY.csv:baseline_valid_all_batches"
        )
        result["registry_candidate_count"] = len({
            row.get("candidate_id", "") for row in registry if row.get("candidate_id")
        })
        if metrics_error is None:
            matches = len(baseline_families) == result["counts"]["baseline_valid_families"]["value"]
            result["denominator_checks"]["baseline_valid_families"] = matches
            result["counts"]["baseline_valid_families"]["status"] = (
                "verified" if matches else "not_verified"
            )
    else:
        result["counts"]["baseline_valid_families_from_registry"] = _value(
            None, "MODEL_REGISTRY.csv:baseline_valid_all_batches", verified=False
        )

    if manifest_error is None:
        registered = manifest.get("registered_families")
        if isinstance(registered, list):
            manifest_count = len(set(map(str, registered)))
            result["counts"]["registered_families_from_manifest"] = _value(
                manifest_count, "FROZEN_MANIFEST.json:registered_families"
            )
            if metrics_error is None:
                matches = (
                    manifest_count == result["counts"]["registered_families"]["value"]
                )
                result["denominator_checks"]["registered_families"] = matches
                result["counts"]["registered_families"]["status"] = (
                    "verified" if matches else "not_verified"
                )
        else:
            result["counts"]["registered_families_from_manifest"] = _value(
                None, "FROZEN_MANIFEST.json:registered_families", verified=False
            )
    else:
        result["counts"]["registered_families_from_manifest"] = _value(
            None, "FROZEN_MANIFEST.json:registered_families", verified=False
        )

    result["raw_receipt_verification"] = _audit_raw_receipts(root, pairs, pairs_error)
    return result
