from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.real_intervention_v4 import cache_lifecycle
from experiments.real_intervention_v4.cache_lifecycle import (
    cache_artifact_digest,
    cache_usage,
    create_cache_scope,
    tree_digest,
)


def test_empty_cache_scope_is_private_unique_and_clean_after_probe(tmp_path):
    attempt = tmp_path / "raw" / "attempt-1"
    attempt.mkdir(parents=True)
    other_attempt = tmp_path / "raw" / "attempt-2"
    other_attempt.mkdir(parents=True)

    record = create_cache_scope(attempt)
    other_record = create_cache_scope(other_attempt)

    assert record["cache_state"] == "EMPTY"
    assert set(record["environment"]) == {
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "TMPDIR", "XDG_CACHE_HOME",
    }
    assert all(item["owner_uid"] == os.getuid() for item in record["directories"].values())
    assert all(not any(Path(path).iterdir()) for path in record["environment"].values())
    assert all(Path(path).is_relative_to(Path(record["root"]))
               for path in record["environment"].values())
    assert record["root"] != other_record["root"]
    assert set(record["environment"].values()).isdisjoint(other_record["environment"].values())

    resumed = create_cache_scope(attempt)
    assert resumed == record


def test_prefilled_cache_digest_describes_materialized_worker_tree(tmp_path):
    source = tmp_path / "source-cache"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "artifact.bin").write_bytes(b"compiled-artifact")
    sources = {"TRITON_CACHE_DIR": source}
    expected, source_manifest = cache_artifact_digest(sources)
    attempt = tmp_path / "raw" / "attempt-2"
    attempt.mkdir(parents=True)

    record = create_cache_scope(
        attempt, mode="PREFILLED", prefilled=sources,
        expected_artifact_digest=expected, acquisition_cost_seconds=12.5,
    )

    materialized = Path(record["environment"]["TRITON_CACHE_DIR"])
    assert record["cache_state"] == "PREFILLED"
    assert record["source_cache_digests"] == source_manifest
    assert record["materialized_cache_digests"] == source_manifest
    assert tree_digest(materialized) == source_manifest["TRITON_CACHE_DIR"]
    assert record["acquisition_cost_seconds"] == 12.5
    assert record["directories"]["TRITON_CACHE_DIR"]["writable_directory_count"] == 2


def test_prefilled_readonly_nested_directories_are_made_writable(tmp_path):
    source = tmp_path / "source-cache"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "artifact.bin").write_bytes(b"compiled-artifact")
    nested.chmod(0o500)
    sources = {"TRITON_CACHE_DIR": source}
    expected, _ = cache_artifact_digest(sources)
    attempt = tmp_path / "raw" / "read-only-nested"
    attempt.mkdir(parents=True)

    record = create_cache_scope(attempt, mode="PREFILLED", prefilled=sources,
                                expected_artifact_digest=expected)

    materialized = Path(record["environment"]["TRITON_CACHE_DIR"]) / "nested"
    assert materialized.stat().st_mode & 0o200
    probe = materialized / "new-cache-file"
    probe.write_bytes(b"writable")
    assert probe.read_bytes() == b"writable"


def test_prefilled_source_digest_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source-cache"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"changed")
    attempt = tmp_path / "raw" / "attempt-3"
    attempt.mkdir(parents=True)

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        create_cache_scope(
            attempt, mode="PREFILLED", prefilled={"TRITON_CACHE_DIR": source},
            expected_artifact_digest="frozen-but-not-matching",
        )


def test_cache_permission_denial_is_reported_at_the_write_boundary(tmp_path):
    attempt = tmp_path / "raw" / "permission-denied"
    attempt.mkdir(parents=True)

    with patch.object(Path, "open", side_effect=PermissionError("injected cache denial")):
        with pytest.raises(PermissionError, match="injected cache denial"):
            create_cache_scope(attempt)


def test_failed_cache_record_publication_can_be_reinitialized(tmp_path):
    attempt = tmp_path / "raw" / "recoverable-init"
    attempt.mkdir(parents=True)
    real_atomic_json = cache_lifecycle.atomic_json

    def fail_scope_record_once(path, value):
        if Path(path).name == "CACHE_SCOPE.json":
            raise OSError("injected scope publication failure")
        return real_atomic_json(path, value)

    with patch("experiments.real_intervention_v4.cache_lifecycle.atomic_json",
               side_effect=fail_scope_record_once):
        with pytest.raises(OSError, match="injected scope publication failure"):
            create_cache_scope(attempt)

    assert (attempt / "cache").is_dir()
    assert not (attempt / "CACHE_SCOPE.json").exists()
    record = create_cache_scope(attempt)

    assert record["cache_state"] == "EMPTY"
    assert (attempt / "CACHE_SCOPE.json").is_file()


def test_transiently_missing_cache_entry_is_unknown_not_zero(tmp_path):
    root = tmp_path / "cache-tree"
    disappearing = root / "triton" / "temporary"
    disappearing.mkdir(parents=True)
    (disappearing / "artifact.bin").write_bytes(b"in-flight")
    real_scandir = os.scandir

    def disappear_once(path):
        if Path(path) == disappearing:
            raise FileNotFoundError("injected rename/removal race")
        return real_scandir(path)

    with patch("experiments.real_intervention_v4.cache_lifecycle.os.scandir",
               side_effect=disappear_once):
        usage = cache_usage(root)

    assert usage["status"] == "UNKNOWN"
    assert usage["complete"] is False
    assert usage["bytes"] is None
    assert usage["missing_entries_during_scan"] == 1
