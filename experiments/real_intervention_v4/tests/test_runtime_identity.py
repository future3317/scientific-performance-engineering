from __future__ import annotations

import unittest

from experiments.real_intervention_v4.runtime_identity import (
    cross_process_runtime_differences,
    selected_gpu_identity_differences,
)


def runtime():
    return {
        "device_uuid": "GPU-a",
        "device_name": "RTX",
        "device_capability": [8, 9],
        "torch_version": "2.6.0",
        "torch_cuda_version": "12.6",
        "cudnn_version": 90100,
        "packages": {"torch": "2.6.0", "torchvision": "0.21.0"},
        "driver": {
            "gpu_uuid": "GPU-a", "driver_version": "535.1",
            "device_name": "RTX", "pci_bus_id": "0000:01:00.0",
            "memory_total_mb": "24576", "power_limit_w": "450",
        },
        "python_version": "3.11.0",
        "pid": 10,
        "thread_settings": {"OMP_NUM_THREADS": "1"},
    }


class RuntimeIdentityTests(unittest.TestCase):
    def test_process_local_fields_do_not_break_cross_process_identity(self):
        reference = runtime()
        observed = runtime()
        observed["pid"] = 20
        observed["thread_settings"] = {"OMP_NUM_THREADS": "2"}

        self.assertEqual(cross_process_runtime_differences(reference, observed), {})

    def test_gpu_or_runtime_change_is_reported(self):
        reference = runtime()
        observed = runtime()
        observed["device_uuid"] = "GPU-b"
        observed["driver"] = {**observed["driver"], "gpu_uuid": "GPU-b"}

        differences = cross_process_runtime_differences(reference, observed)

        self.assertIn("device_uuid", differences)
        self.assertIn("driver", differences)

    def test_missing_runtime_identity_is_not_treated_as_a_match(self):
        differences = cross_process_runtime_differences(runtime(), {"device_uuid": "GPU-a"})

        self.assertIn("driver", differences)
        self.assertIn("packages", differences)

    def test_selected_physical_gpu_must_match_both_runtime_reports(self):
        observed = runtime()
        observed["driver"] = {**observed["driver"], "gpu_uuid": "GPU-b"}

        self.assertEqual(
            selected_gpu_identity_differences(observed, "GPU-a"),
            {"driver.gpu_uuid": {"expected": "GPU-a", "observed": "GPU-b"}},
        )


if __name__ == "__main__":
    unittest.main()
