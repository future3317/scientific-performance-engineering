from experiments.real_intervention_v6.eligibility_and_split import build_cohort


def test_cohort_is_family_unique_outcome_blind_and_exposure_first():
    class NoTimingAccess(dict):
        def get(self, key, default=None):
            if key in {"timing", "timings", "log_benefit", "s0_s", "s1_s"}:
                raise AssertionError(f"cohort builder accessed {key}")
            return super().get(key, default)

    snapshot = [
        NoTimingAccess({"family_id": f"f{i:02}", "eligibility": "ELIGIBLE", "split": "final",
         "log_benefit": 1000 - i, "s0_s": i, "checks": {"timing": "must not copy"}}
        )
        for i in range(17)
    ]
    snapshot += [dict(snapshot[0]), {"family_id": "failed", "eligibility": "FAILED"}]
    exposed = ["f00", "f01", "f02"]

    first = build_cohort(snapshot, exposed, "A")
    second = build_cohort(list(reversed(snapshot)), exposed, "A")

    assert first["split_counts"] == {"development": 6, "calibration": 6, "final": 5}
    assert [row["family_id"] for row in first["rows"]] == [row["family_id"] for row in second["rows"]]
    assert len({row["family_id"] for row in first["rows"]}) == 17
    assert all(row["split"] == "development" for row in first["rows"] if row["family_id"] in exposed)
    assert all("log_benefit" not in row and "checks" not in row for row in first["rows"])
    assert first["sample_feasibility"]["independent_pilot"]["feasible"] is True
    assert first["sample_feasibility"]["original_acre_certification"]["status"] == "INFEASIBLE"


def test_b16_split_and_small_cohort_infeasibility_are_explicit():
    snapshot = [{"family_id": f"b{i}", "eligibility": "ELIGIBLE"} for i in range(16)]
    cohort = build_cohort(snapshot, [], "B")
    assert cohort["split_counts"] == {"development": 6, "calibration": 6, "final": 4}

    small = build_cohort(snapshot[:5], [], "pilot")
    assert sum(small["split_counts"].values()) == 5
    assert small["sample_feasibility"]["independent_pilot"]["status"] == "INFEASIBLE"
