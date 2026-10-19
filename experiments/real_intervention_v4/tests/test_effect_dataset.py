import unittest

from experiments.real_intervention_v4.effect_dataset import summarize_effects


def pair(pair_id, family, s0, s1, *, split="final", policy="ELIGIBLE",
         comparability="MATCHED", correctness="PASS", execution="COMPLETE"):
    events = [] if execution == "NOT_STARTED" else [{"kind": "started"}]
    receipt_verified = False
    if execution in {"COMPLETE", "FAILED", "TIMEOUT"}:
        events.append({"kind": "terminal"})
        receipt_verified = True
    receipt = lambda seconds: {
        "arm_execution": execution,
        "scientific_correctness": correctness,
        "receipt_verified": receipt_verified,
        "events": events,
        "runtime_gpu_uuid": "GPU-1",
        "inputs_sha256": "input",
        "weights_sha256": "weights",
        "source_commit": "source",
        "config_digest": "config",
        "context_digest": "context",
        "raw_receipt_sha256": "a" * 64,
        "timing": {
            "cold_seconds": seconds,
            "steady_seconds": seconds / 2,
            "prepare_seconds": 0.1,
            "forwards_seconds": seconds / 2,
        },
        "correctness_summary": {},
        "telemetry": {},
    }
    return {
        "pair_id": pair_id,
        "family_id": family,
        "split": split,
        "context": {"pre_execution_feature": "value"},
        "classification": {
            "arm_execution": "COMPLETE" if execution == "COMPLETE" else execution,
            "scientific_correctness": correctness,
            "paired_comparability": comparability,
            "policy_evidence": policy,
        },
        "S0": receipt(s0),
        "S1": receipt(s1),
    }


class EffectDatasetTests(unittest.TestCase):
    def test_repeated_contexts_do_not_inflate_family_weight_or_n(self):
        records = [
            pair("a1", "A", 2, 1), pair("a2", "A", 2, 1),
            pair("a3", "A", 2, 1), pair("b1", "B", 1, 2),
        ]
        result = summarize_effects(records, replicates=200)
        final = result["by_split"]["final"]
        self.assertEqual(final["independent_family_count"], 2)
        self.assertEqual(final["valid_pair_count"], 4)
        self.assertAlmostEqual(final["family_equal_mean_log_benefit"], 0.0)
        self.assertEqual(final["denominators"]["planned"], 4)
        self.assertIsNone(final["family_cluster_bootstrap_ci95_family_equal_mean_log_benefit"])

    def test_direction_and_cluster_interval_are_deterministic(self):
        records = [pair("a", "A", 2, 1), pair("b", "B", 3, 1)]
        first = summarize_effects(records, replicates=500)
        second = summarize_effects(records, replicates=500)
        final = first["by_split"]["final"]
        self.assertEqual(final["family_direction_counts"],
                         {"positive": 2, "negative": 0, "zero": 0})
        self.assertEqual(final["family_cluster_bootstrap_ci95_family_equal_mean_log_benefit"],
                         second["by_split"]["final"][
                             "family_cluster_bootstrap_ci95_family_equal_mean_log_benefit"])
        self.assertEqual(final["leave_one_family_out_mean_log_benefit"],
                         {"A": 1.0986122886681098, "B": 0.6931471805599453})

    def test_unknown_and_unsafe_stay_in_denominators_not_effects(self):
        unknown = pair("unknown", "U", 2, 1, execution="UNKNOWN", policy="MISSING",
                       comparability="UNKNOWN", correctness="UNASSESSED")
        unsafe = pair("unsafe", "X", 2, 1, policy="UNSAFE", correctness="FAIL")
        infra = pair("infra", "I", 2, 1, execution="TIMEOUT", policy="MISSING",
                     comparability="INCOMPLETE", correctness="UNASSESSED")
        unstarted = pair("not-started", "N", 2, 1, execution="NOT_STARTED",
                         policy="MISSING", comparability="INCOMPLETE",
                         correctness="UNASSESSED")
        result = summarize_effects([unknown, unsafe, infra, unstarted], replicates=50)
        final = result["by_split"]["final"]
        self.assertEqual([row["effect_status"] for row in result["pairs"]],
                         ["unknown", "unsafe", "infra", "unstarted"])
        self.assertEqual(final["denominators"], {
            "planned": 4, "started": 3, "closed": 2, "valid": 0,
            "unsafe": 1, "infra": 1, "unknown": 1, "unstarted": 1,
            "ineligible": 0,
        })
        self.assertIsNone(final["family_equal_mean_log_benefit"])
        self.assertTrue(all(row["log_benefit"] is None for row in result["pairs"]))

    def test_receipt_digest_is_required_for_effect_eligibility(self):
        record = pair("missing-digest", "A", 2, 1)
        record["S1"].pop("raw_receipt_sha256")

        result = summarize_effects([record], replicates=50)

        self.assertEqual(result["pairs"][0]["effect_status"], "unknown")
        self.assertIsNone(result["pairs"][0]["log_benefit"])
        self.assertEqual(result["by_split"]["final"]["denominators"]["unknown"], 1)

    def test_unknown_without_a_durable_started_event_is_not_counted_as_started(self):
        record = pair("prelaunch-unknown", "A", 2, 1, execution="UNKNOWN",
                      policy="MISSING", comparability="INCOMPLETE",
                      correctness="UNASSESSED")
        record["S0"]["events"] = []
        record["S1"]["events"] = []

        result = summarize_effects([record], replicates=50)

        final = result["by_split"]["final"]
        self.assertEqual(final["denominators"]["started"], 0)
        self.assertEqual(final["denominators"]["closed"], 0)
        self.assertEqual(final["denominators"]["unknown"], 1)

    def test_baseline_correctness_failure_is_not_counted_as_candidate_harm(self):
        record = pair("baseline-unsafe", "A", 2, 1, correctness="FAIL", policy="MISSING")
        record["S0"]["scientific_correctness"] = "FAIL"
        record["S1"]["scientific_correctness"] = "PASS"
        record["classification"]["scientific_correctness"] = "FAIL"

        result = summarize_effects([record], replicates=50)

        self.assertEqual(result["pairs"][0]["effect_status"], "baseline_invalid")
        self.assertEqual(result["by_split"]["final"]["denominators"]["baseline_invalid"], 1)
        self.assertEqual(result["by_split"]["final"]["denominators"]["unsafe"], 0)

    def test_single_arm_infrastructure_failure_is_classified_as_infra(self):
        for arm in ("S0", "S1"):
            with self.subTest(arm=arm):
                record = pair(f"infra-{arm}", "A", 2, 1)
                record[arm]["infrastructure_classification"] = "INFRA"
                record[arm]["arm_execution"] = "FAILED"
                record[arm]["scientific_correctness"] = "UNASSESSED"
                record["classification"].update(
                    arm_execution="FAILED", scientific_correctness="UNASSESSED"
                )

                result = summarize_effects([record], replicates=50)

                self.assertEqual(result["pairs"][0]["effect_status"], "infra")
                self.assertEqual(result["by_split"]["final"]["denominators"]["infra"], 1)

    def test_known_candidate_correctness_harm_precedes_infrastructure(self):
        record = pair("unsafe-infra", "A", 2, 1, correctness="FAIL", policy="UNSAFE")
        record["S1"]["infrastructure_classification"] = "INFRA"

        result = summarize_effects([record], replicates=50)

        self.assertEqual(result["pairs"][0]["effect_status"], "unsafe")

    def test_one_family_never_produces_a_cross_family_interval(self):
        result = summarize_effects([pair("a", "A", 2, 1)], replicates=50)

        final = result["by_split"]["final"]
        self.assertEqual(final["independent_family_count"], 1)
        self.assertIsNone(final["family_cluster_bootstrap_ci95_family_equal_mean_log_benefit"])

    def test_a_family_cannot_appear_in_multiple_frozen_splits(self):
        records = [pair("dev", "A", 2, 1, split="development"),
                   pair("final", "A", 2, 1, split="final")]

        with self.assertRaisesRegex(ValueError, "cross frozen splits"):
            summarize_effects(records, replicates=50)


if __name__ == "__main__":
    unittest.main()
