from __future__ import annotations

import json

from scripts.audit_public_skillsbench import audit_public_snapshot


def test_public_aggregate_without_category_metadata_is_not_used_as_external_evidence(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps([{
        "task": "t1", "modelShort": "m1", "harness": "h1", "family": "f1",
        "condition": "With Skills", "score": 100.0, "trials": 3,
    }]))
    report = audit_public_snapshot(path)
    assert report["status"] == "metadata_insufficient"
    assert report["provider_calls"] == 0
    assert report["claim_boundary"] == "not external validity evidence"
