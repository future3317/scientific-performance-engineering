import pytest

from experiments.real_intervention_v2.receipts import resume_receipt, write_pair
from experiments.real_intervention_v2.arm_journal import complete_arm, start_arm


def pair(context_id="ctx-1", identity="run-sha"):
    return {"context_id": context_id, "session_id": "session-1", "identity": identity,
            "arms": {"eager": {"status": "valid"}, "compile": {"status": "valid"}}}


def test_receipt_is_written_once_and_resumed_by_identity(tmp_path):
    path = write_pair(tmp_path, pair(), storage_cap=10000)
    assert resume_receipt(tmp_path, "ctx-1", "run-sha", "session-1")["context_id"] == "ctx-1"
    with pytest.raises(FileExistsError):
        write_pair(tmp_path, pair(), storage_cap=10000)
    assert path.exists()


def test_receipt_resume_rejects_different_run_identity(tmp_path):
    write_pair(tmp_path, pair(), storage_cap=10000)
    with pytest.raises(ValueError, match="identity"):
        resume_receipt(tmp_path, "ctx-1", "other-run", "session-1")


def test_receipt_storage_cap_is_enforced(tmp_path):
    with pytest.raises(RuntimeError, match="storage cap"):
        write_pair(tmp_path, pair(), storage_cap=1)


def test_storage_cap_can_cover_arm_receipts_and_cache_files(tmp_path):
    (tmp_path / "cache.bin").write_bytes(b"12345")
    with pytest.raises(RuntimeError, match="storage cap"):
        write_pair(tmp_path / "pairs", pair(), storage_cap=1, storage_scope=tmp_path)


def test_started_arm_is_not_replayed_after_interruption(tmp_path):
    first = start_arm(tmp_path, "ctx-1", "session-1", "eager", "run-sha")
    assert first["status"] == "started"
    resumed = start_arm(tmp_path, "ctx-1", "session-1", "eager", "run-sha")
    assert resumed["status"] == "unknown"
    assert resumed["reason"] == "unresolved_inflight_not_replayed"


def test_completed_arm_is_reused_without_a_second_intent(tmp_path):
    start_arm(tmp_path, "ctx-1", "session-1", "compile", "run-sha")
    complete_arm(tmp_path, "ctx-1", "session-1", "compile", "run-sha",
                 {"status": "valid", "total_job_s": 2.0})
    resumed = start_arm(tmp_path, "ctx-1", "session-1", "compile", "run-sha")
    assert resumed["status"] == "valid"
    assert resumed["total_job_s"] == 2.0


def test_sessions_have_independent_arm_receipts(tmp_path):
    start_arm(tmp_path, "ctx-1", "session-1", "eager", "run-sha")
    start_arm(tmp_path, "ctx-1", "session-2", "eager", "run-sha")
    assert len(list(tmp_path.glob("*.json"))) == 2
    write_pair(tmp_path / "pairs", pair(), storage_cap=10000)
    second = pair() | {"session_id": "session-2"}
    write_pair(tmp_path / "pairs", second, storage_cap=10000)
    assert len(list((tmp_path / "pairs").glob("*.json"))) == 2
