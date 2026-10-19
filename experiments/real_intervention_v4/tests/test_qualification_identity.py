from __future__ import annotations

import unittest

from experiments.real_intervention_v4.qualification import (
    aggregate_qualification,
    classify_qualification_failure,
    cross_process_qualification_identity_differences,
    has_cuda_kernel_events,
    has_pairable_kernel_exposure,
)


class QualificationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "candidate_id": "torchvision:resnet18",
            "source": "torchvision",
            "model": "resnet18",
            "family": "resnet",
            "family_id": "resnet",
        }

    def test_s1_compatibility_failure_is_not_baseline_invalid_or_dropped(self):
        s0_reference = {"status": "S0_REFERENCE_COMPLETE", "arm_execution": "COMPLETE"}
        s0_replicate = {"status": "VALID", "arm_execution": "COMPLETE", "eligibility": "S0_AA_PASS"}
        s1 = {
            "status": "ERROR", "eligibility": "CHANNELS_LAST_EXECUTION_FAILURE",
            "arm_execution": "FAILED", "scientific_correctness": "UNKNOWN",
            "first_failure_stage": "S1_MEMORY_FORMAT_CONVERSION",
        }

        row = aggregate_qualification(
            self.candidate, 1, s0_reference=s0_reference,
            s0_replicate=s0_replicate, s1_correctness=s1,
        )

        self.assertEqual(row["eligibility"], "CHANNELS_LAST_EXECUTION_FAILURE")
        self.assertEqual(row["failure_phase"], "S1_CORRECTNESS")
        self.assertTrue(row["candidate_denominator"])
        self.assertTrue(row["qualification_complete"])
        self.assertNotEqual(row["eligibility"], "BASELINE_INVALID")

    def test_s1_correctness_mismatch_remains_candidate_failure(self):
        row = aggregate_qualification(
            self.candidate, 8,
            s0_reference={"status": "S0_REFERENCE_COMPLETE", "arm_execution": "COMPLETE"},
            s0_replicate={"status": "VALID", "arm_execution": "COMPLETE", "eligibility": "S0_AA_PASS"},
            s1_correctness={
                "status": "NUMERIC_MISMATCH", "eligibility": "CHANNELS_LAST_CORRECTNESS_FAILURE",
                "arm_execution": "COMPLETE", "scientific_correctness": "FAIL",
            },
        )

        self.assertEqual(row["eligibility"], "CHANNELS_LAST_CORRECTNESS_FAILURE")
        self.assertTrue(row["candidate_denominator"])
        self.assertTrue(row["qualification_complete"])

    def test_s0_aa_failure_prevents_s1_from_relabeling_it(self):
        row = aggregate_qualification(
            self.candidate, 1,
            s0_reference={"status": "S0_REFERENCE_COMPLETE", "arm_execution": "COMPLETE"},
            s0_replicate={
                "status": "BASELINE_INVALID", "eligibility": "BASELINE_INVALID",
                "arm_execution": "COMPLETE", "scientific_correctness": "FAIL",
            },
            s1_correctness={
                "status": "ERROR", "eligibility": "CHANNELS_LAST_EXECUTION_FAILURE",
                "arm_execution": "FAILED",
            },
        )

        self.assertEqual(row["eligibility"], "BASELINE_INVALID")
        self.assertEqual(row["phase_summary"]["S1_CORRECTNESS"], "ERROR")
        self.assertEqual(row["failure_phase"], "S0_AA_REPLICATE")
        self.assertTrue(row["candidate_denominator"])

    def test_pairability_requires_named_cuda_events_for_both_arms(self):
        captured = {
            "cuda_kernels_executed": True,
            "cuda_kernel_event_count": 2,
            "cuda_kernel_event_names": ["aten::convolution", "conv2d_kernel"],
            "included_in_performance_timing": False,
        }
        self.assertTrue(has_cuda_kernel_events(captured))
        self.assertFalse(has_cuda_kernel_events({**captured, "cuda_kernel_event_count": 0}))
        self.assertFalse(has_cuda_kernel_events({**captured, "cuda_kernel_event_names": []}))
        self.assertFalse(has_cuda_kernel_events({**captured, "included_in_performance_timing": True}))

        exposure = {
            "active": True, "pairability_status": "PAIRABLE",
            "s0_cuda_kernel_path": captured, "s1_cuda_kernel_path": captured,
        }
        self.assertTrue(has_pairable_kernel_exposure(exposure))
        self.assertFalse(has_pairable_kernel_exposure({**exposure, "s0_cuda_kernel_path": None}))

        s1 = {
            "status": "VALID", "eligibility": "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY",
            "arm_execution": "COMPLETE", "scientific_correctness": "PASS",
            "kernel_exposure": exposure,
        }
        eligible = aggregate_qualification(
            self.candidate, 1,
            s0_reference={"status": "S0_REFERENCE_COMPLETE", "arm_execution": "COMPLETE"},
            s0_replicate={"status": "VALID", "arm_execution": "COMPLETE", "eligibility": "S0_AA_PASS"},
            s1_correctness=s1,
        )
        self.assertEqual(eligible["eligibility"], "ELIGIBLE_CHANNELS_LAST_CORRECTNESS_ONLY")

        missing_s0 = {**s1, "kernel_exposure": {
            **exposure, "s0_cuda_kernel_path": None,
        }}
        not_pairable = aggregate_qualification(
            self.candidate, 1,
            s0_reference={"status": "S0_REFERENCE_COMPLETE", "arm_execution": "COMPLETE"},
            s0_replicate={"status": "VALID", "arm_execution": "COMPLETE", "eligibility": "S0_AA_PASS"},
            s1_correctness=missing_s0,
        )
        self.assertEqual(not_pairable["eligibility"], "NO_MECHANISTIC_EXPOSURE")
        self.assertEqual(not_pairable["pairability_status"], "NO_MECHANISTIC_EXPOSURE")

    def test_s1_runtime_preflight_is_not_called_a_channels_last_failure(self):
        result = classify_qualification_failure("S1_CORRECTNESS", "RUNTIME_IDENTITY")

        self.assertEqual(result["eligibility"], "RUNTIME_PREFLIGHT_FAILURE")
        self.assertNotEqual(result["eligibility"], "BASELINE_INVALID")

    def test_s1_execution_failure_has_a_distinct_stage_classification(self):
        result = classify_qualification_failure("S1_CORRECTNESS", "S1_FORWARD")

        self.assertEqual(result["eligibility"], "CHANNELS_LAST_EXECUTION_FAILURE")
        self.assertEqual(result["scientific_correctness"], "UNKNOWN")
        self.assertNotEqual(result["eligibility"], "BASELINE_INVALID")

    def test_cross_process_identity_binds_device_source_weights_and_inputs(self):
        runtime = {
            "device_uuid": "GPU-a", "device_name": "RTX",
            "device_capability": [8, 9], "torch_version": "2.6.0",
            "torch_cuda_version": "12.6", "cudnn_version": 9,
            "packages": {"torch": "2.6.0"},
            "driver": {"gpu_uuid": "GPU-a", "driver_version": "535"},
            "python_version": "3.11",
        }
        reference = {
            "source_identity": {"source": "torchvision", "model": "resnet18"},
            "runtime_identity": runtime, "weights_sha256": "w0", "inputs_sha256": "i0",
        }

        differences = cross_process_qualification_identity_differences(
            reference,
            source_identity={"source": "torchvision", "model": "resnet18"},
            runtime_identity={**runtime, "pid": 99}, expected_gpu_uuid="GPU-a",
            weights_sha256="w0", inputs_sha256="i0",
        )
        self.assertEqual(differences, {
            "runtime": {}, "selected_gpu": {}, "source_identity_match": True,
            "weights_match": True, "inputs_match": True,
        })

        changed = cross_process_qualification_identity_differences(
            reference,
            source_identity={"source": "torchbench", "model": "resnet"},
            runtime_identity={**runtime, "device_uuid": "GPU-b"},
            expected_gpu_uuid="GPU-a", weights_sha256="w1", inputs_sha256="i1",
        )
        self.assertIn("device_uuid", changed["runtime"])
        self.assertIn("device_uuid", changed["selected_gpu"])
        self.assertFalse(changed["source_identity_match"])
        self.assertFalse(changed["weights_match"])
        self.assertFalse(changed["inputs_match"])


if __name__ == "__main__":
    unittest.main()
