from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.real_intervention_v4.source_registry import (
    candidate_universe,
    candidate_source_identity,
    register_candidates,
)


class SourceRegistryTests(unittest.TestCase):
    def test_cross_source_aliases_share_one_independent_family_lineage(self):
        rows = register_candidates([
            {"source": "torchvision", "model": "resnet18", "family": "ResNet"},
            {"source": "torchbench", "model": "resnet", "family": "resnet"},
        ])

        self.assertEqual([row["family_id"] for row in rows], ["resnet", "resnet"])
        self.assertEqual(len({row["family_lineage_id"] for row in rows}), 1)
        self.assertNotEqual(rows[0]["candidate_id"], rows[1]["candidate_id"])

    def test_unmapped_candidate_stays_in_registry_without_fake_family(self):
        row = register_candidates([
            {"source": "torchbench", "model": "unknown_model", "family": None},
        ])[0]

        self.assertIsNone(row["family_id"])
        self.assertIsNone(row["family_lineage_id"])
        self.assertTrue(row["candidate_id"])

    def test_source_identity_binds_model_lineage_and_declared_revisions(self):
        identity = candidate_source_identity(
            {"source": "torchbench", "model": "resnet", "family": "resnet",
             "eval_deterministic": True},
            source_revisions={"torchbench": "commit-a", "experiment": "commit-b"},
        )

        self.assertEqual(identity["candidate_id"], "torchbench:resnet")
        self.assertEqual(identity["family_lineage_id"], "resnet")
        self.assertEqual(identity["source_revisions"]["torchbench"], "commit-a")
        self.assertTrue(identity["registration"]["eval_deterministic"])

    def test_duplicate_candidate_id_is_rejected_but_shared_family_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "duplicate candidate identity"):
            register_candidates([
                {"source": "torchvision", "model": "resnet18", "family": "resnet"},
                {"source": "torchvision", "model": "resnet18", "family": "resnet"},
            ])

    def test_supplemental_timm_roster_is_pinned_and_keeps_family_lineages(self):
        from experiments.real_intervention_v3 import run_all as v3_run_all
        from experiments.real_intervention_v3 import source_adapters as v3_sources

        candidates = [
            {"source": "torchvision", "model": "resnet18", "family": "resnet",
             "availability": "available"},
        ]
        supplemental = {"timm": {
            "version": "1.0.29", "git_commit": "e617d170064e1fad8417858abc6a39489746be9b",
            "models": [
                {"model": "ghostnet_100", "family": "ghostnet", "input_shape": [3, 224, 224]},
                {"model": "hrnet_w18", "family": "hrnet", "input_shape": [3, 224, 224]},
                {"model": "dla34", "family": "dla", "input_shape": [3, 224, 224]},
            ],
        }}
        with patch.object(v3_run_all, "baseline_pool", return_value=candidates), \
             patch.object(v3_sources, "model_universe", return_value=candidates), \
             patch("experiments.real_intervention_v4.source_registry.importlib.metadata.version",
                   return_value="1.0.29"):
            rows = candidate_universe(
                "unused", source_revisions={
                    "repository_commit": "repo", "torchbench_commit": "torchbench",
                }, supplemental_sources=supplemental,
            )

        timm_rows = [row for row in rows if row["source"] == "timm"]
        self.assertEqual({row["family_id"] for row in timm_rows}, {"ghostnet", "hrnet", "dla"})
        self.assertTrue(all(row["availability"] == "available" for row in timm_rows))
        self.assertTrue(all(row["pretrained"] is False for row in timm_rows))
        self.assertTrue(all(
            row["source_identity"]["source_revisions"]["timm_git_commit"]
            == "e617d170064e1fad8417858abc6a39489746be9b"
            for row in timm_rows
        ))


if __name__ == "__main__":
    unittest.main()
