from __future__ import annotations

import pytest

from scripts.collect_cost_profile import summarize_samples


def test_cost_summary_excludes_warmup_and_reports_median_p90_max() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0], warmup=2, minimum_samples=3)
    assert summary["warmup_excluded"] == 2
    assert summary["n"] == 3
    assert summary["median"] == 4.0
    assert summary["p90"] == 4.8
    assert summary["max"] == 5.0


def test_cost_summary_requires_twenty_measured_samples() -> None:
    with pytest.raises(ValueError, match="20"):
        summarize_samples([1.0] * 19, warmup=0)


def test_offline_stopping_horizon_is_labeled_not_online() -> None:
    summary = summarize_samples([1.0] * 20, warmup=0, source="offline_replay")
    assert summary["stopping_horizon_source"] == "offline_replay"
    assert summary["online_stopping_horizon_claim"] is False
