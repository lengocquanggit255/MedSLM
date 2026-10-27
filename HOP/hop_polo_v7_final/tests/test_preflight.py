"""Fast checks for the full-session contract; no model training occurs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hop_polo_v6 import clean_splits, fingerprint, select_hits, verify_input
from hop_polo_v7_final import check_config, recount
from hopgs.data import ADIntGraph
from v7_setting_b_helpers import policy_stale


class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/full.json").read_text())

    def test_stage_limits_and_setting_b(self):
        check_config(self.config)
        for stage in ("synthetic", "qa_sft", "policy_warmup", "policy", "pool_ranker"):
            self.assertEqual((self.config[stage]["epochs"],
                              self.config[stage]["patience"]), (24, 3))

    def test_only_raw_and_public_inputs(self):
        audit = verify_input(ROOT / "inputs", self.config)
        self.assertFalse(audit["prior_pharmkg_checkpoint_input"])

    def test_fixed_split_and_deferred_test_answers(self):
        graph = ADIntGraph(ROOT / "inputs/raw", seed=42, defer_test_answers=True)
        reference = json.loads((ROOT / "configs/reference_split.json").read_text())
        self.assertEqual(fingerprint(graph), reference["graph_sha256"])
        split = clean_splits(graph, seed=42)
        for name in ("train", "val", "test"):
            self.assertEqual([r["qid"] for r in split["members"][name]],
                             reference[f"{name}_qids"])
        self.assertFalse(split["test_answers_loaded"])
        self.assertTrue(all(not q.answers for q in graph.splits["test"]))

    def test_epoch_zero_and_patience(self):
        rows = [{"epoch": 0, "validation": {"mean_hits_at_1": 0.5},
                 "technical_valid": True}]
        stale = 0
        for epoch in range(1, 4):
            rows.append({"epoch": epoch,
                         "validation": {"mean_hits_at_1": 0.5},
                         "technical_valid": True})
            stale = policy_stale(rows, epoch, stale)
        self.assertEqual(select_hits(rows)["epoch"], 0)
        self.assertEqual(stale, 3)

    def test_independent_recount(self):
        row = {"qid": "q", "answers": [1, 2], "ranked_top20": [1, 3],
               "selected_nodes": [1, 3, 4], "pool_nodes": [1, 2, 3, 4]}
        values = recount([row])
        self.assertEqual(values["mean_hits_at_1"], 1.0)
        self.assertEqual(values["mean_recall_at_20"], 0.5)
        self.assertEqual(values["mean_pool_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
