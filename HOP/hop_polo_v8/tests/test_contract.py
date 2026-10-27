"""Test nhanh cho phần V8 tự thêm: config, chọn checkpoint, đếm lại, bundle, cạnh ngược, nối phiên.

PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest HOP/hop_polo_v8/tests/test_contract.py   (từ gốc repo)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOP / "src"))

from hop.audit import recount  # noqa: E402
from hop.codebook import build_codebook  # noqa: E402
from hop.common import digest  # noqa: E402
from hop.data import BUNDLE_FILES, Graph  # noqa: E402
from hop.pipeline import Clock, Pause, Pipeline, check_config, load_config, select_hits  # noqa: E402
from hop.program import EdgeIndex, PrefixSIndex, concat_ranges  # noqa: E402


def toy_bundle(root: Path, triples) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "triples.npy", np.asarray(triples, np.int64))
    nodes = sorted({t[0] for t in triples} | {t[2] for t in triples})
    (root / "nodes.jsonl").write_text("".join(json.dumps({"id": n, "name": f"n{n}", "type": "t" if n % 2 else "u",
                                                         "description": ""}) + "\n" for n in nodes))
    (root / "relations.json").write_text(json.dumps({"0": "r0", "1": "r1"}))
    (root / "qa_train.jsonl").write_text(json.dumps({"qid": "a", "question": "q a", "topics": [0], "answers": [2]}) + "\n")
    (root / "qa_val.jsonl").write_text(json.dumps({"qid": "b", "question": "q b", "topics": [1], "answers": [2]}) + "\n")
    (root / "qa_test.jsonl").write_text(json.dumps({"qid": "c", "question": "q c", "topics": []}) + "\n")
    (root / "test_answers.jsonl").write_text(json.dumps({"qid": "c", "answers": [1]}) + "\n")
    (root / "manifest.json").write_text(json.dumps({"dataset": "toy", "files": {n: digest(root / n) for n in BUNDLE_FILES}}))
    return root


class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_configs(self):
        for name in ("default", "smoke"):
            check_config(load_config(HOP / "configs" / f"{name}.json"))
        smoke = load_config(HOP / "configs" / "smoke.json")
        self.assertEqual(smoke["synthetic"]["learning_rate"], 3e-05)  # kế thừa từ default
        self.assertEqual(smoke["synthetic"]["epochs"], 1)

    def test_select_hits(self):
        rows = [{"epoch": 0, "validation": {"mean_hits_at_1": 0.5}, "technical_valid": True},
                {"epoch": 1, "validation": {"mean_hits_at_1": 0.5}, "technical_valid": True},
                {"epoch": 2, "validation": {"mean_hits_at_1": 0.9}, "technical_valid": False}]
        self.assertEqual(select_hits(rows)["epoch"], 0)

    def test_recount_no_topic(self):
        rows = [{"qid": "x", "answers": [1, 2], "ranked_top20": [1, 3], "selected_nodes": [1, 3, 4],
                 "pool_ranked": [3, 1, 4, 2]},
                {"qid": "y", "answers": [5], "ranked_top20": [], "selected_nodes": [], "pool_ranked": [],
                 "no_topic": True}]
        values = recount(rows)
        self.assertEqual(values["mean_hits_at_1"], 0.5)
        self.assertEqual(values["mean_pool_recall"], 0.5)
        rows[1].pop("no_topic")
        with self.assertRaises(AssertionError):
            recount(rows)

    def test_inverse_policy_and_bundle(self):
        root = toy_bundle(self.tmp / "b", [(0, 0, 1), (1, 0, 0), (1, 1, 2)])
        missing, always = Graph(root, inverse="missing"), Graph(root, inverse="always")
        self.assertEqual((len(missing.src), missing.inverse_added), (4, 1))
        self.assertEqual(len(always.src), 6)
        self.assertEqual([q.answers for q in missing.splits["test"]], [()])
        missing.load_test_answers()
        self.assertEqual(missing.splits["test"][0].answers, (1,))
        (root / "qa_val.jsonl").write_text(json.dumps({"qid": "b", "question": "changed", "topics": [1],
                                                        "answers": [2]}) + "\n")
        with self.assertRaises(AssertionError):
            Graph(root)

    def test_concat_ranges_and_prefix_index(self):
        np.testing.assert_array_equal(concat_ranges(np.array([5, 0, 9]), np.array([2, 0, 3])), [5, 6, 9, 10, 11])
        rng = np.random.default_rng(0)
        triples = [(int(a), int(r), int(b)) for a, r, b in zip(rng.integers(0, 30, 200), rng.integers(0, 2, 200),
                                                               rng.integers(0, 30, 200))]
        graph = Graph(toy_bundle(self.tmp / "p", triples), verify=False)
        codebook = build_codebook(rng.integers(0, 5, size=(len(graph.ids), 3)).astype(np.int16))
        index = EdgeIndex(graph)
        prefix = PrefixSIndex(graph, codebook, index)
        for frontier in [(0, 3, 7), tuple(range(len(graph.ids)))]:
            for action in range(len(graph.schemas)):
                edges = index.edges(frontier, action)
                brute = {tuple(int(x) for x in codebook.node_codes[d][:k]) for d in graph.dst[edges] for k in (1, 2, 3)}
                self.assertEqual(prefix.prefixes(frontier, action), brute)
                self.assertEqual(sorted(edges.tolist()),
                                 sorted(e for e in range(len(graph.src))
                                        if graph.src[e] in frontier and graph.schema[e] == action))

    def run_phase(self, directory: Path, pause_after: int | None):
        """Pha giả: một tham số, metric val tất định theo trọng số; dùng cả RNG numpy và torch."""
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        rng = np.random.default_rng(1)
        pipe = object.__new__(Pipeline)
        pipe.device = torch.device("cpu")
        pipe.clock = Clock(None)
        calls = {"n": 0}

        def train_one(epoch):
            if pause_after is not None and epoch == pause_after + 1:
                raise Pause("test")
            x = torch.as_tensor(rng.normal(size=(8, 3)), dtype=torch.float32) + torch.randn(8, 3)
            loss = model(x).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return {"loss": float(loss)}, 1

        def evaluate(_epoch):
            calls["n"] += 1
            value = float(torch.sigmoid(model.weight.sum()).detach())
            return {"validation": {"mean_hits_at_1": round(value, 2), "mean_recall_at_20": value},
                    "technical_valid": True}

        def save_model(path):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path / "m.pt")

        def load_model(path):
            model.load_state_dict(torch.load(path / "m.pt", weights_only=True))

        selected, history = Pipeline.train_phase(
            pipe, "toy", directory, settings={"epochs": 8, "patience": 3}, optimizer=optimizer, rng=rng,
            train_one=train_one, evaluate=evaluate, save_model=save_model, load_model=load_model,
            checkpoint_file="m.pt")
        return selected, history, model.weight.detach().clone()

    def test_resume_matches_uninterrupted(self):
        straight = self.run_phase(self.tmp / "straight", None)
        with self.assertRaises(Pause):
            self.run_phase(self.tmp / "paused", 2)
        resumed = self.run_phase(self.tmp / "paused", None)
        strip = lambda rows: [{k: v for k, v in r.items() if not k.endswith("seconds")} for r in rows]  # noqa: E731
        self.assertEqual(strip(straight[1]), strip(resumed[1]))
        self.assertEqual(straight[0]["epoch"], resumed[0]["epoch"])
        torch.testing.assert_close(straight[2], resumed[2])


if __name__ == "__main__":
    unittest.main()
