"""Portable Agent Skills packs for a typed EvoSPE candidate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .contract import SkillContract, validate_skill_text


@dataclass(frozen=True)
class SkillCertificateKey:
    body_sha256: str
    contract_sha256: str
    agent_revision: str
    model_identity: str
    verifier_revision: str
    task_population: str

    @property
    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def certificate_key(
    contract: SkillContract,
    *,
    agent_revision: str,
    model_identity: str,
    verifier_revision: str,
    task_population: str,
) -> SkillCertificateKey:
    values = (agent_revision, model_identity, verifier_revision, task_population)
    if any(not value.strip() for value in values):
        raise ValueError("certificate identity requires agent, model, verifier, and task population")
    return SkillCertificateKey(
        contract.body_digest,
        contract.digest,
        agent_revision,
        model_identity,
        verifier_revision,
        task_population,
    )


def _pack_name(contract: SkillContract) -> tuple[str, str]:
    name = re.sub(r"[^a-z0-9]+", "-", contract.skill_id.casefold()).strip("-")
    if not name or len(name) > 64 or "--" in name:
        raise ValueError("skill ID cannot be represented as an Agent Skills name")
    description = f"{contract.intent.strip()} ({contract.task_family})"
    if not description or len(description) > 1024 or "\n" in description:
        raise ValueError("skill description must be one line of at most 1024 characters")
    return name, description


def export_skill_pack(
    contract: SkillContract,
    destination: str | Path,
    *,
    agent_revision: str,
    model_identity: str,
    verifier_revision: str,
    task_population: str,
) -> Path:
    """Create a new SKILL.md plus typed contract and provenance manifest."""
    if contract.source_split not in {"train", "train_dev", "train_inner"}:
        raise ValueError("only train-derived skill candidates may be exported")
    if contract.status != "candidate":
        raise ValueError("only candidate skills may be exported")
    errors = validate_skill_text(contract, max_characters=1800)
    if errors:
        raise ValueError("skill pack rejected: " + ",".join(errors))
    name, description = _pack_name(contract)
    key = certificate_key(
        contract,
        agent_revision=agent_revision,
        model_identity=model_identity,
        verifier_revision=verifier_revision,
        task_population=task_population,
    )
    if contract.model_identity and contract.model_identity != model_identity:
        raise ValueError("pack model identity differs from the candidate provenance")
    skill_md = (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n\n"
        f"# {description}\n\n"
        f"{contract.body}\n"
    )
    contract_text = json.dumps(contract.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    skill_sha = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
    contract_sha = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "skill_id": contract.skill_id,
        "version": contract.version,
        "status": "candidate",
        "source_split": contract.source_split,
        "source_task_ids": list(contract.source_task_ids),
        "source_template_ids": list(contract.source_template_ids),
        "source_cluster_ids": list(contract.source_cluster_ids),
        "source_trace_hashes": list(contract.source_trace_hashes),
        "train_split_manifest_sha256": contract.train_split_manifest_sha256,
        "generator_provider": contract.generator_provider,
        "model_identity": model_identity,
        "agent_revision": agent_revision,
        "verifier_revision": verifier_revision,
        "task_population": task_population,
        "skill_body_sha256": contract.body_digest,
        "skill_md_sha256": skill_sha,
        "contract_sha256": contract.digest,
        "contract_file_sha256": contract_sha,
        "proposal_prompt_sha256": contract.proposal_prompt_sha256,
        "proposal_output_sha256": contract.proposal_output_sha256,
        "certificate_key": key.__dict__,
        "certificate_key_sha256": key.digest,
        "scripts_allowed": False,
    }
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for relative, content in (
        ("SKILL.md", skill_md),
        ("contract.json", contract_text),
        ("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"),
    ):
        with (target / relative).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    read_skill_pack(target)
    return target


def read_skill_pack(directory: str | Path) -> SkillContract:
    """Read and verify a generated pack without relying on custom frontmatter."""
    root = Path(directory)
    skill_md = (root / "SKILL.md").read_text(encoding="utf-8")
    contract_text = (root / "contract.json").read_text(encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    contract = SkillContract.from_dict(json.loads(contract_text))
    name, description = _pack_name(contract)
    expected_header = (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n\n"
    )
    if not skill_md.startswith(expected_header) or not skill_md.endswith(contract.body + "\n"):
        raise ValueError("SKILL.md frontmatter or rendered body does not match contract")
    checks = {
        "skill_md_sha256": hashlib.sha256(skill_md.encode("utf-8")).hexdigest(),
        "contract_file_sha256": hashlib.sha256(contract_text.encode("utf-8")).hexdigest(),
        "contract_sha256": contract.digest,
        "skill_body_sha256": contract.body_digest,
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError("skill pack manifest digest mismatch")
    key = certificate_key(
        contract,
        agent_revision=manifest.get("agent_revision", ""),
        model_identity=manifest.get("model_identity", ""),
        verifier_revision=manifest.get("verifier_revision", ""),
        task_population=manifest.get("task_population", ""),
    )
    if manifest.get("certificate_key") != key.__dict__ or manifest.get("certificate_key_sha256") != key.digest:
        raise ValueError("skill certificate key does not match current content and execution scope")
    if manifest.get("source_split") not in {"train", "train_dev", "train_inner"} or manifest.get("scripts_allowed") is not False:
        raise ValueError("skill pack provenance or execution policy is invalid")
    for key_name, expected in (
        ("source_task_ids", list(contract.source_task_ids)),
        ("source_template_ids", list(contract.source_template_ids)),
        ("source_cluster_ids", list(contract.source_cluster_ids)),
        ("source_trace_hashes", list(contract.source_trace_hashes)),
        ("train_split_manifest_sha256", contract.train_split_manifest_sha256),
    ):
        if manifest.get(key_name) != expected:
            raise ValueError("skill pack source provenance does not match its contract")
    if contract.model_identity and manifest.get("model_identity") != contract.model_identity:
        raise ValueError("skill pack model identity does not match its contract")
    return contract
