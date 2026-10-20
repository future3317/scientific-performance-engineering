"""Frozen v4 study identity and family-level cohort construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from .outcomes import atomic_json, canonical_digest


def resolve_server_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.home() / path).resolve()


def read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("study config must be a JSON object")
    required = {
        "study_id", "intervention_id", "torchbench_root", "torchbench_commit",
        "batches", "horizons", "sessions_per_context", "split", "acre",
        "shape", "correctness", "arm_timeout_seconds", "cleanup_grace_seconds",
        "term_grace_seconds", "watchdog_heartbeat_seconds", "gpu_sample_seconds",
        "wall_cap_seconds",
        "disk_cap_bytes", "concurrency", "gpu_workers", "cpu_threads",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"study config missing fields: {missing}")
    if value["intervention_id"] != "CHANNELS_LAST_EAGER_LAYOUT_V1":
        raise ValueError("v4 primary intervention is frozen as channels-last eager layout")
    if value["concurrency"] != 1 or value["gpu_workers"] != 1:
        raise ValueError("the shared-GPU v4 resource contract permits one worker")
    if any(int(item) < 1 for item in value["batches"] + value["horizons"]):
        raise ValueError("batch and horizon values must be positive")
    if int(value["sessions_per_context"]) != 2:
        raise ValueError("v4 requires the frozen AB/BA pair")
    if (float(value["arm_timeout_seconds"]) <= 0
            or float(value["cleanup_grace_seconds"]) <= 0
            or not 0 <= float(value["term_grace_seconds"]) <= float(value["cleanup_grace_seconds"])):
        raise ValueError("invalid frozen arm deadline and cleanup grace")
    if len(value["shape"]) != 3 or any(int(item) < 1 for item in value["shape"]):
        raise ValueError("channels-last contexts require a frozen C,H,W shape")
    split = value["split"]
    split_minima = (
        int(split["minimum_development_families"]),
        int(split["minimum_calibration_families"]),
        int(split["minimum_final_families"]),
    )
    if sum(split_minima) != int(split["minimum_total_families_for_confirmatory"]):
        raise ValueError("confirmatory family minimum must cover development, calibration, and final")
    if (int(split["underpowered_development_family_cap"]) != split_minima[0]
            or int(split["underpowered_calibration_family_cap"]) != split_minima[1]):
        raise ValueError("underpowered assignment must preserve the frozen development and calibration caps")
    acre = value["acre"]
    if (int(acre["minimum_development_families"]) != split_minima[0]
            or int(acre["minimum_calibration_deploy_families"]) != split_minima[1]
            or int(acre["minimum_calibration_support_families"])
            + int(acre["minimum_calibration_non_support_families"]) > split_minima[1]):
        raise ValueError("ACRE evidence minima conflict with the frozen independent-family split")
    return value


def family_split(family_ids: Sequence[str], *, seed: int,
                 split_config: Mapping[str, Any]) -> dict[str, str]:
    """Assign the registered family roster before any qualification outcome exists."""
    families = sorted({str(item) for item in family_ids if str(item)})
    if len(families) != len(family_ids):
        raise ValueError("registered independent family ids must be unique and non-empty")
    rng = random.Random(seed)
    rng.shuffle(families)
    n = len(families)
    minimum_cal = int(split_config["minimum_calibration_families"])
    minimum_final = int(split_config["minimum_final_families"])
    minimum_dev = int(split_config["minimum_development_families"])
    minimum_total = int(split_config["minimum_total_families_for_confirmatory"])
    if n >= minimum_total:
        cal_n = minimum_cal
        final_n = minimum_final
        dev_n = n - cal_n - final_n
        if dev_n < minimum_dev:
            raise ValueError("confirmatory roster cannot satisfy the frozen family minima")
    else:
        # Preserve enough development and calibration families to attempt the
        # preregistered rule/region checks. The residual assignment is sealed
        # as final but cannot be opened while the confirmatory minima fail.
        dev_n = min(n, int(split_config["underpowered_development_family_cap"]))
        cal_n = min(n - dev_n, int(split_config["underpowered_calibration_family_cap"]))
        final_n = n - dev_n - cal_n
    result = {family: "development" for family in families[:dev_n]}
    result.update({family: "calibration" for family in families[dev_n:dev_n + cal_n]})
    result.update({family: "final" for family in families[dev_n + cal_n:]})
    if set(result) != set(family_ids):
        raise AssertionError("family split lost or duplicated an eligible family")
    return result


def registered_family_ids(universe: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return independent lineages from the full candidate roster, not qualifiers."""
    candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in universe]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every registered candidate must have an identity")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("registered candidate identities must be unique")
    return sorted({
        str(candidate["family_id"])
        for candidate in universe
        if candidate.get("family_id") not in (None, "")
    })


def split_family_counts(split_by_family: Mapping[str, str]) -> dict[str, int]:
    return {
        split: sum(value == split for value in split_by_family.values())
        for split in ("development", "calibration", "final")
    }


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("context id has no path-safe characters")
    return result


def planned_contexts(eligible_rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any],
                     split_by_family: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(eligible_rows, key=lambda item: str(item["candidate_id"])):
        family = str(row["family_id"])
        split = split_by_family[family]
        for batch in config["batches"]:
            qualification = row["qualification_by_batch"][str(batch)]
            for horizon in config["horizons"]:
                context_id = _safe_id(f"{row['candidate_id']}-B{batch}-H{horizon}")
                context = {
                    **dict(row), **dict(qualification.get("model_summary") or {}),
                    "family_id": family, "split": split,
                    "batch": int(batch), "horizon": int(horizon),
                    "shape": list(row.get("input_shape", config["shape"])),
                    "context_id": context_id,
                    "qualification": dict(qualification),
                    "qualification_digest": canonical_digest(dict(qualification)),
                    "expected_weights_sha256": qualification.get("weights_sha256"),
                    "expected_inputs_sha256": qualification.get("inputs_sha256"),
                    "reference_sha256": qualification.get("reference_sha256"),
                    "reference_path": (qualification.get("model_summary") or {}).get("reference_path"),
                    "torchbench_root": str(resolve_server_path(config["torchbench_root"])),
                }
                context.pop("qualification_by_batch", None)
                rows.append(context)
    return rows


def planned_pairs(contexts: Sequence[Mapping[str, Any]], *, sessions_per_context: int,
                  seed: int) -> list[dict[str, Any]]:
    result = []
    for context in contexts:
        for session in range(sessions_per_context):
            pair_id = _safe_id(f"{context['context_id']}-session-{session}")
            order = ["S0", "S1"] if (session + int(context["order_bit"])) % 2 == 0 else ["S1", "S0"]
            result.append({
                "pair_id": pair_id, "family_id": str(context["family_id"]),
                "split": str(context["split"]), "context_id": str(context["context_id"]),
                "session": session, "order": order, "context": dict(context),
            })
    ids = [row["pair_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("planned pair ids are not unique")
    return result


def build_frozen_manifest(*, config: Mapping[str, Any], source_commit: str,
                          config_path: Path, selected_gpu_uuid: str,
                          runtime_identity: Mapping[str, Any],
                          universe: Sequence[Mapping[str, Any]],
                          qualifications: Sequence[Mapping[str, Any]],
                          frozen_family_split: Mapping[str, str]) -> dict[str, Any]:
    """Bind qualification outcomes to an already-frozen full-roster split."""
    registered_families = registered_family_ids(universe)
    if set(frozen_family_split) != set(registered_families):
        raise ValueError("frozen family split must include every registered lineage exactly once")
    if any(value not in {"development", "calibration", "final"}
           for value in frozen_family_split.values()):
        raise ValueError("frozen family split contains an unknown partition")
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in qualifications:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    eligible_rows = []
    for candidate in universe:
        family = str(candidate.get("family_id") or "")
        if not family:
            continue
        results = by_candidate.get(str(candidate["candidate_id"]), [])
        by_batch = {str(row["batch"]): row for row in results}
        required_batches = {str(batch) for batch in config["batches"]}
        if (set(by_batch) == required_batches and all(
            by_batch[batch].get("eligibility") == "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY"
            for batch in required_batches
        )):
            first = by_batch[str(config["batches"][0])]
            eligible_rows.append({
                **dict(candidate), **dict(first["model_summary"]),
                "qualification_by_batch": by_batch,
            })
    eligible_families = sorted({
        str(row["family_id"]) for row in eligible_rows if row.get("family_id")
    })
    contexts = planned_contexts(
        eligible_rows, config=config, split_by_family=frozen_family_split,
    )
    rng = random.Random(int(config["seed"]))
    for context in contexts:
        context["order_bit"] = rng.randrange(2)
    pairs = planned_pairs(contexts, sessions_per_context=int(config["sessions_per_context"]),
                          seed=int(config["seed"]))
    split_counts = split_family_counts(frozen_family_split)
    eligible_split_counts = split_family_counts({
        family: frozen_family_split[family] for family in eligible_families
    })
    eligibility_by_id = {
        str(candidate["candidate_id"]): {
            str(row["batch"]): str(row.get("eligibility", "UNKNOWN"))
            for row in by_candidate.get(str(candidate["candidate_id"]), [])
        }
        for candidate in universe
    }
    manifest = {
        "schema_version": 1, "study_id": config["study_id"],
        "intervention_id": config["intervention_id"],
        "source_commit": source_commit,
        "config": dict(config), "config_path_name": config_path.name,
        "config_digest": canonical_digest(dict(config)),
        "selected_gpu_uuid": selected_gpu_uuid,
        "runtime_identity": dict(runtime_identity),
        "source_universe_count": len(universe),
        "registered_families": registered_families,
        "qualification_count": len(qualifications),
        "eligible_candidate_count": len(eligible_rows),
        "eligibility_roster": [
            {
                **{key: candidate.get(key) for key in (
                    "candidate_id", "source", "model", "family_id", "availability",
                )},
                "assigned_split": frozen_family_split.get(str(candidate.get("family_id"))),
                "qualification_by_batch": eligibility_by_id[str(candidate["candidate_id"])],
            }
            for candidate in universe
        ],
        "eligible_families": eligible_families,
        "unqualified_families": sorted(set(registered_families) - set(eligible_families)),
        "family_split": dict(frozen_family_split),
        "split_family_counts": split_counts,
        "eligible_family_counts_by_split": eligible_split_counts,
        "confirmatory_split_eligible": (
            len(eligible_families) >= int(config["split"]["minimum_total_families_for_confirmatory"])
            and eligible_split_counts["development"] >= int(config["split"]["minimum_development_families"])
            and eligible_split_counts["calibration"] >= int(config["split"]["minimum_calibration_families"])
            and eligible_split_counts["final"] >= int(config["split"]["minimum_final_families"])
        ),
        "contexts": contexts,
        "planned_pairs": pairs,
        "qualification_digest": canonical_digest(list(qualifications)),
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest


def write_or_verify_manifest(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(manifest):
            raise ValueError("frozen v4 manifest identity differs; refusing to rewrite it")
        return existing
    atomic_json(path, dict(manifest))
    return dict(manifest)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
