"""Materialize the committed trusted-host DeepSeek controller contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.formal import attest, host_agent


CAMPAIGN_BUDGET = {
    "tokens": 81_920,
    "tool_calls": 24,
    "wall_time_s": 900,
    "context_tokens": 32_768,
}
PRICING_REVISION = "unpriced-provider-probe-2026-08-30"
CREDENTIAL_MARKERS = ("api_key", "token", "secret", "password", "credential")


def canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def build_runtime_manifest(controller_root: Path) -> dict[str, Any]:
    root = controller_root.resolve()
    source = root / "benchmark" / "formal" / "host_agent.py"
    if not source.is_file():
        raise ValueError("controller root is missing benchmark/formal/host_agent.py")
    return {
        "schema_version": 1,
        "runtime_attestation_kind": "trusted_host_python_runtime_manifest",
        "controller_commit": _git(root, "rev-parse", "HEAD"),
        "controller_source": "benchmark/formal/host_agent.py",
        "controller_source_digest": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
    }


def build_contract(
    runtime: dict[str, Any],
    provider_probe: dict[str, Any],
    *,
    benchmark_revision: str,
) -> dict[str, Any]:
    credential_fields = [
        key for key in provider_probe
        if any(marker in str(key).lower() for marker in CREDENTIAL_MARKERS)
    ]
    if credential_fields:
        raise ValueError("provider probe contains credential fields")
    digest = provider_probe.get("provider_model_list_response_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("provider_model_list_response_digest must be a 64-hex digest")
    if provider_probe.get("status") != "pass":
        raise ValueError("provider probe status must be pass")
    if provider_probe.get("model_id") != "deepseek-v4-flash" or provider_probe.get("returned_model") != "deepseek-v4-flash":
        raise ValueError("provider probe model identity must be deepseek-v4-flash")
    commit = runtime.get("controller_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise ValueError("runtime controller_commit must be a commit digest")
    if not re.fullmatch(r"[0-9a-f]{40}", benchmark_revision):
        raise ValueError("benchmark_revision must be a full 40-hex commit digest")
    runtime_digest = canonical_digest(runtime)
    agent_config = host_agent.build_agent_attestation(
        model_list_digest=digest,
        agent_code_commit=commit,
        container_digest=runtime_digest,
        pricing_revision=PRICING_REVISION,
    )
    agent_config["tool_versions"] = {}
    return {
        "schema_version": 1,
        "status": "ready_for_nonsealed_probe",
        "provider": "deepseek",
        "model_id": "deepseek-v4-flash",
        "benchmark_revision": benchmark_revision,
        "campaign_budget": dict(CAMPAIGN_BUDGET),
        "runtime_attestation_kind": "trusted_host_python_runtime_manifest",
        "container_digest_semantics": "runtime_manifest_digest",
        "container_digest": runtime_digest,
        "pricing_revision": PRICING_REVISION,
        "runtime_manifest": runtime,
        "agent_config": agent_config,
    }


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("container_digest") != canonical_digest(contract.get("runtime_manifest", {})):
        errors.append("container_digest does not bind runtime_manifest")
    if contract.get("campaign_budget") != CAMPAIGN_BUDGET:
        errors.append("campaign_budget does not match frozen controller budget")
    benchmark_revision = contract.get("benchmark_revision")
    if not isinstance(benchmark_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", benchmark_revision):
        errors.append("benchmark_revision must be a full 40-hex commit digest")
    manifest = {
        "schema_version": 1,
        "population_id": "SPE-EvoBench-v1.0-50",
        "experiment_id": "host-controller-contract-validation",
        "benchmark_revision": benchmark_revision,
        "skill_view_digest": "a" * 64,
        "task_manifest_digest": "b" * 64,
        "agent_model_id": "deepseek-v4-flash",
        "agent_config": contract.get("agent_config"),
        "condition": "D",
        "context_mode": "reset",
        "task_order": ["PUBLIC-CONTRACT-PROBE"],
        "worker_isolation": {"mode": "external_namespace_executor", "network_mode": "none", "mount_allowlist": ["task"]},
        "outer_trial_id": "outer-000",
        "budgets": dict(CAMPAIGN_BUDGET),
        "hardware_fingerprint": {},
        "software_fingerprint": {},
        "torch_version": None,
        "cuda_version": None,
        "slot_id": "PUBLIC-CONTRACT-PROBE",
        "visibility": "public_dev",
    }
    errors.extend(attest.validate_experiment(manifest))
    return errors


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--provider-probe", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if _git(args.controller_root, "status", "--porcelain"):
        raise ValueError("controller root must be clean before materialization")
    runtime = build_runtime_manifest(args.controller_root)
    probe = json.loads(args.provider_probe.read_text(encoding="utf-8"))
    contract = build_contract(runtime, probe, benchmark_revision=args.benchmark_revision)
    errors = validate_contract(contract)
    if errors:
        raise ValueError("invalid host controller contract: " + "; ".join(errors))
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "runtime_manifest.json", runtime)
    _write_json(args.out / "host_controller_contract.json", contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
