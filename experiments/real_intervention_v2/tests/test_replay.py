import pytest

from experiments.real_intervention_v2.contracts import digest
from experiments.real_intervention_v2.replay import freeze_policy, replay


def context(cid, split, horizon, lineage):
    return {"context_id": cid, "lineage_id": lineage, "split": split,
            "planned_horizon": horizon, "model_parameter_count": horizon * 10,
            "sessions": [{"session_id": "s1", "order": "eager_first"},
                         {"session_id": "s2", "order": "compile_first"}]}


def pair(ctx, sid, eager_s, compile_s):
    return {"context_id": ctx["context_id"], "session_id": sid, "context": ctx,
            "arms": {"eager": {"status": "valid", "total_job_s": eager_s,
                                "warm_median_s": 1.0},
                     "compile": {"status": "valid", "total_job_s": compile_s,
                                  "warm_median_s": 0.5}},
            "correctness": {"passed": True}}


def test_final_scoring_requires_pre_frozen_policy_and_counts_contexts_not_sessions():
    train_contexts = [context(f"train-{i}", "development", 1 + 2 * i, f"lineage-{i}")
                      for i in range(4)]
    final_contexts = [context("final-0", "final_untouched", 16, "final-lineage")]
    train = [pair(ctx, sid, 5.0, 4.0) for ctx in train_contexts
             for sid in ("s1", "s2")]
    final = [pair(final_contexts[0], sid, 5.0, 4.0) for sid in ("s1", "s2")]
    manifest = {"acre_rules": []}
    preview = replay(train + final, manifest)
    assert preview["final_opened"] is False
    policy = freeze_policy(train, final_contexts, manifest)
    with pytest.raises(ValueError, match="frozen policy"):
        replay(train + final, manifest, policy | {"final_decision_digest": "bad"})
    report = replay(train + final, manifest, policy)
    assert report["results"]["always"]["n"] == 1
    assert report["results"]["pairwise_oracle"]["n"] == 1


def test_policy_digest_must_match_saved_decisions():
    train_ctx = context("train", "development", 4, "lineage-1")
    final_ctx = context("final", "final_untouched", 4, "lineage-2")
    train = [pair(train_ctx, sid, 5.0, 4.0) for sid in ("s1", "s2")]
    final = [pair(final_ctx, sid, 5.0, 4.0) for sid in ("s1", "s2")]
    policy = freeze_policy(train, [final_ctx], {"acre_rules": []})
    changed = {**policy, "final_decisions": {"final": {"always": False}}}
    with pytest.raises(ValueError, match="digest mismatch"):
        replay(train + final, {"acre_rules": []}, changed)
