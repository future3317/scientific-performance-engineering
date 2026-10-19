"""Closeout report contracts; these tests are intended for the server suite."""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile

from experiments.real_intervention_v4.generate_report import write_final_outputs


def test_closeout_keeps_unopened_final_outcomes_unread_and_acre_unfitted(tmp_path):
    development = {"pair_id": "dev-pair", "family_id": "dev-family", "split": "development"}
    final = {"pair_id": "final-pair", "family_id": "final-family", "split": "final"}
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    (pairs / "final-pair.json").write_text("not-json-final-outcome", encoding="utf-8")

    result = write_final_outputs(
        root=tmp_path, config={"study_id": "test-study", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
                               "batches": [1], "bootstrap_seed": 101, "bootstrap_repetitions": 100},
        manifest={"manifest_digest": "frozen", "source_commit": "source-sha",
                  "selected_gpu_uuid": "GPU-test", "planned_pairs": [development, final]},
        universe=[], qualifications=[],
        gate={"status": "UNDERPOWERED_OR_INCOMPLETE", "final_opened": False},
        shakedown={"passes": True, "planned_opportunities": 12,
                   "closed_opportunities": 12, "required_minimum": 12},
        source_commit="source-sha", selected_gpu_uuid="GPU-test",
    )

    metrics = json.loads((tmp_path / "METRICS.json").read_text(encoding="utf-8"))
    assert result["analysis_complete"] is False
    assert metrics["by_split"]["final"]["denominators"]["planned"] == 1
    assert metrics["by_split"]["final"]["denominators"]["unstarted"] == 1
    assert metrics["acre"]["status"] == "ACRE_NOT_FITTED"
    assert metrics["acre"]["fitted"] is False
    assert metrics["acre"]["canonical_support_certificate_created"] is False
    assert metrics["acre"]["production_runner_bridge_invoked"] is False
    with (tmp_path / "POLICY_RESULTS.csv").open(newline="", encoding="utf-8") as stream:
        policies = list(csv.DictReader(stream))
    acre_rows = [row for row in policies if row["policy"] in {"ORIGINAL_ACRE", "REGIME_AWARE_ACRE"}]
    assert acre_rows and all(row["status"] == "ACRE_NOT_FITTED" for row in acre_rows)
    assert metrics["resources"]["gpu_busy_seconds"] is None
    assert metrics["resources"]["child_cpu_seconds_sum"] is None
    with tarfile.open(tmp_path / "EVIDENCE_BUNDLE.tar.gz", "r:gz") as archive:
        names = archive.getnames()
    assert len(names) == len(set(names))
    assert "BUNDLE_MEMBERS_SHA256.json" in names


def test_bundle_contains_receipt_bytes_and_lifecycle_events_for_pair_hashes(tmp_path):
    pair_id = "dev-pair"
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    raw = tmp_path / "raw"
    for arm in ("S0", "S1"):
        attempt = raw / f"primary-{pair_id}-{arm}"
        events = attempt / "events"
        events.mkdir(parents=True)
        receipt_bytes = json.dumps({"arm": arm}, sort_keys=True).encode("utf-8") + b"\n"
        (attempt / "receipt.json").write_bytes(receipt_bytes)
        (events / "0001-intent.json").write_text('{"kind":"intent"}\n', encoding="utf-8")
        (events / "0002-terminal.json").write_text('{"kind":"terminal"}\n', encoding="utf-8")
        (attempt / "cache").mkdir()
        (attempt / "cache" / "must-not-be-bundled").write_text("cache", encoding="utf-8")
        if arm == "S0":
            s0_digest = hashlib.sha256(receipt_bytes).hexdigest()
        else:
            s1_digest = hashlib.sha256(receipt_bytes).hexdigest()
    qualification = raw / "qual-model-B1-S1CORR"
    (qualification / "events").mkdir(parents=True)
    qualification_receipt = b'{"phase":"s1-correctness"}\n'
    (qualification / "receipt.json").write_bytes(qualification_receipt)
    (qualification / "events" / "0001-terminal.json").write_text(
        '{"kind":"terminal"}\n', encoding="utf-8",
    )
    (tmp_path / "PAIRS.jsonl").write_text(json.dumps({
        "pair_id": pair_id,
        "S0": {"raw_receipt_sha256": s0_digest},
        "S1": {"raw_receipt_sha256": s1_digest},
    }) + "\n", encoding="utf-8")

    from experiments.real_intervention_v4.generate_report import _bundle
    _bundle(tmp_path)

    with tarfile.open(tmp_path / "EVIDENCE_BUNDLE.tar.gz", "r:gz") as archive:
        members = {member.name: archive.extractfile(member).read()
                   for member in archive.getmembers() if member.isfile()}
    for arm, digest in (("S0", s0_digest), ("S1", s1_digest)):
        receipt_member = f"raw/primary-{pair_id}-{arm}/receipt.json"
        assert hashlib.sha256(members[receipt_member]).hexdigest() == digest
        assert f"raw/primary-{pair_id}-{arm}/events/0001-intent.json" in members
        assert f"raw/primary-{pair_id}-{arm}/events/0002-terminal.json" in members
    assert not any("cache" in name for name in members)
    assert hashlib.sha256(
        members["raw/qual-model-B1-S1CORR/receipt.json"]
    ).hexdigest() == hashlib.sha256(qualification_receipt).hexdigest()
    assert "raw/qual-model-B1-S1CORR/events/0001-terminal.json" in members
    assert len(members) == len(set(members))


def test_closeout_reports_frozen_canonical_acre_support_without_stale_not_fitted_claim(tmp_path):
    (tmp_path / "POLICY_FREEZE.json").write_text(json.dumps({
        "schema_version": 1, "status": "FROZEN", "acre_status": "FITTED",
        "acre_rule_id": "CHANNELS_LAST", "policies": {},
        "development_pair_ids": [], "calibration_pair_ids": [],
        "study_id": "test-study", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
        "source_commit": "source-sha", "manifest_digest": "frozen",
        "final_outcomes_read": False,
        "acre_bridge": {
            "status": "FITTED", "engine_route_invoked": True,
            "canonical_support_certificate_created": True,
            "support_evidence": {
                "regime": "SELECTIVE_IDENTIFIED", "action": "reuse",
                "independent_families": 10, "certificate_scope_bound": True,
            },
        },
    }), encoding="utf-8")
    (tmp_path / "ACRE_STATUS.json").write_text(json.dumps({
        "status": "FITTED", "engine_route_invoked": True,
        "canonical_support_certificate_created": True,
        "support_evidence": {
            "regime": "SELECTIVE_IDENTIFIED", "action": "reuse",
            "independent_families": 10, "certificate_scope_bound": True,
        },
    }), encoding="utf-8")

    write_final_outputs(
        root=tmp_path,
        config={"study_id": "test-study", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
                "batches": [1], "bootstrap_seed": 101, "bootstrap_repetitions": 100},
        manifest={"manifest_digest": "frozen", "source_commit": "source-sha",
                  "selected_gpu_uuid": "GPU-test", "planned_pairs": []},
        universe=[], qualifications=[], gate={"status": "UNDERPOWERED", "final_opened": False},
        shakedown={}, source_commit="source-sha", selected_gpu_uuid="GPU-test",
    )

    metrics = json.loads((tmp_path / "METRICS.json").read_text(encoding="utf-8"))
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    claims = (tmp_path / "PAPER_CLAIM_MATRIX.md").read_text(encoding="utf-8")
    assert metrics["acre"]["status"] == "FITTED"
    assert metrics["acre"]["canonical_support_certificate_created"] is True
    assert metrics["acre"]["support_evidence"]["independent_families"] == 10
    assert "ACRE: **FITTED**" in report
    assert "production runner has not wired" not in report
    assert "policy_bridge.py" in claims and "AcreEngine.route" in claims


def test_closeout_does_not_replace_runner_pair_ledger(tmp_path):
    original = '{"runner_owned":true}\n'
    (tmp_path / "PAIRS.jsonl").write_text(original, encoding="utf-8")
    write_final_outputs(
        root=tmp_path, config={"study_id": "empty", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
                               "batches": [1]},
        manifest={"manifest_digest": "frozen", "planned_pairs": []},
        universe=[], qualifications=[], gate={"final_opened": False}, shakedown={},
        source_commit="source-sha", selected_gpu_uuid="GPU-test",
    )
    assert (tmp_path / "PAIRS.jsonl").read_text(encoding="utf-8") == original


def test_closed_exploratory_no_go_is_not_reported_as_confirmatory_complete(tmp_path):
    final = {"pair_id": "final-pair", "family_id": "final-family", "split": "final"}
    result = write_final_outputs(
        root=tmp_path, config={"study_id": "small-roster", "intervention_id": "CHANNELS_LAST_EAGER_LAYOUT_V1",
                               "batches": [1]},
        manifest={"manifest_digest": "frozen", "planned_pairs": [final]},
        universe=[], qualifications=[],
        gate={"status": "UNDERPOWERED", "final_opened": False},
        shakedown={"passes": False, "planned_opportunities": 0,
                   "closed_opportunities": 0, "required_minimum": 12},
        source_commit="source-sha", selected_gpu_uuid="GPU-test",
    )

    metrics = json.loads((tmp_path / "METRICS.json").read_text(encoding="utf-8"))
    assert result["closeout_complete"] is True
    assert result["analysis_complete"] is False
    assert result["study_status"] == "UNDERPOWERED_EXPLORATORY_CLOSEOUT"
    assert metrics["exploratory_closeout_complete"] is True
    assert metrics["confirmatory_analysis_complete"] is False
