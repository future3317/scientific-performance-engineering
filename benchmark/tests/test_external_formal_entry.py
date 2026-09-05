from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark.formal import release_manifest, run_campaign


def test_formal_entry_accepts_external_calibration_and_release_artifacts(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    for relative in (
        "benchmark/formal",
        "benchmark/manifests",
        "benchmark/calibration",
        "references",
    ):
        (repo_root / relative).mkdir(parents=True)
    population = repo_root / "benchmark/manifests/population.json"
    population.write_text(json.dumps({
        "status": "materialized_frozen", "sealed_total": 35,
        "primary_population": "sealed-35", "slots": [],
    }), encoding="utf-8")
    claims = repo_root / "CLAIMS.yaml"
    protocol = repo_root / "references/protocol.md"
    campaign_config = repo_root / "benchmark/formal/campaign.yaml"
    agent_policy = repo_root / "benchmark/formal/agent_backbones.yaml"
    calibration_protocol = repo_root / "benchmark/calibration/calibration_protocol.json"
    for path in (claims, protocol, campaign_config, agent_policy, calibration_protocol):
        path.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    report = external / "population_report.json"
    pilot = external / "pilot_calibration.json"
    approval = external / "calibration_approval.json"
    release = external / "formal_release.json"
    report.write_text("{}\n", encoding="utf-8")
    pilot.write_text(json.dumps({"artifact_digest": "external-pilot"}), encoding="utf-8")
    approval.write_text(json.dumps({"approved": True, "pilot_calibration_digest": "external-pilot"}), encoding="utf-8")
    release.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(run_campaign, "validate_materialized_manifest", lambda *_args: [])
    monkeypatch.setattr(run_campaign, "validate_calibration_approval", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(run_campaign, "validate_formal_release", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": ""})(),
    )

    run_campaign._validate_formal_entry(
        repo_root, population, campaign_config, claims, protocol,
        calibration_approval_path=approval,
        pilot_calibration_path=pilot,
        population_report_path=report,
        formal_release_manifest_path=release,
    )



def test_formal_release_accepts_ready_external_status(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "benchmark/formal").mkdir(parents=True)
    claims = repo_root / "CLAIMS.yaml"
    protocol = repo_root / "protocol.md"
    campaign = repo_root / "campaign.yaml"
    agent_policy = repo_root / "benchmark/formal/agent_backbones.yaml"
    for path in (claims, protocol, campaign, agent_policy):
        path.write_text("{}\n", encoding="utf-8")
    population = tmp_path / "population.json"
    approval = tmp_path / "approval.json"
    population.write_text("{}\n", encoding="utf-8")
    approval.write_text(json.dumps({"approval_digest": "approval"}), encoding="utf-8")
    manifest = release_manifest.build_release_manifest(
        repo_root=repo_root, population={}, approval={"benchmark_revision": "dce4d19", "approval_digest": "approval"},
        contamination={}, claims_path=claims, protocol_path=protocol,
        campaign_config=campaign, executor_image="sha256:executor",
    )
    manifest["status"] = "ready"
    manifest["release_digest"] = release_manifest.digest_json(
        {key: value for key, value in manifest.items() if key != "release_digest"}
    )
    assert release_manifest.validate_formal_release(
        manifest, repo_root=repo_root, population_manifest=population,
        approval_path=approval, claims_path=claims, protocol_path=protocol,
        campaign_config=campaign,
    ) == []


def test_formal_entry_separates_statistical_protocol_and_analysis_plan(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    for relative in (
        "benchmark/formal",
        "benchmark/manifests",
        "benchmark/calibration",
        "references",
    ):
        (repo_root / relative).mkdir(parents=True)
    population = repo_root / "benchmark/manifests/population.json"
    population.write_text(json.dumps({
        "status": "materialized_frozen", "sealed_total": 35,
        "primary_population": "sealed-35", "slots": [],
    }), encoding="utf-8")
    claims = repo_root / "CLAIMS.yaml"
    statistical_protocol = repo_root / "references/STATISTICAL_PROTOCOL.md"
    analysis_plan = tmp_path / "external_analysis_plan.json"
    campaign_config = repo_root / "benchmark/formal/campaign.yaml"
    agent_policy = repo_root / "benchmark/formal/agent_backbones.yaml"
    calibration_protocol = repo_root / "benchmark/calibration/calibration_protocol.json"
    for path in (claims, statistical_protocol, campaign_config, agent_policy, calibration_protocol):
        path.write_text("{}\n", encoding="utf-8")
    analysis_payload = {
        "status": "frozen", "sealed_outcomes_observed": False,
    }
    analysis_payload["analysis_plan_digest"] = release_manifest.digest_json(analysis_payload)
    analysis_plan.write_text(json.dumps(analysis_payload), encoding="utf-8")

    external = tmp_path / "external"
    external.mkdir()
    report = external / "population_report.json"
    pilot = external / "pilot_calibration.json"
    approval = external / "calibration_approval.json"
    release = external / "formal_release.json"
    report.write_text("{}\n", encoding="utf-8")
    pilot.write_text(json.dumps({"artifact_digest": "external-pilot"}), encoding="utf-8")
    approval.write_text(json.dumps({
        "approval_digest": "approval",
        "approved": True, "pilot_calibration_digest": "external-pilot",
    }), encoding="utf-8")
    manifest = release_manifest.build_release_manifest(
        repo_root=repo_root, population=json.loads(population.read_text()),
        approval={"benchmark_revision": "dce4d19", "approval_digest": "approval"},
        contamination={}, claims_path=claims, protocol_path=statistical_protocol,
        campaign_config=campaign_config, executor_image="sha256:executor",
    )
    manifest["status"] = "ready"
    manifest["analysis_plan_digest"] = analysis_payload["analysis_plan_digest"]
    manifest["release_digest"] = release_manifest.digest_json(
        {key: value for key, value in manifest.items() if key != "release_digest"}
    )
    release.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(run_campaign, "validate_materialized_manifest", lambda *_args: [])
    monkeypatch.setattr(run_campaign, "validate_calibration_approval", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": ""})(),
    )
    run_campaign._validate_formal_entry(
        repo_root, population, campaign_config, claims, analysis_plan,
        calibration_approval_path=approval,
        pilot_calibration_path=pilot,
        population_report_path=report,
        formal_release_manifest_path=release,
    )
