"""Test nhanh cho phần V8 tự thêm: config, chọn checkpoint, đếm lại, bundle, cạnh ngược, nối phiên,
chia chỗ pool theo độ sâu, teacher khi đáp án quá xa, đường 4 hop, action (quan hệ, chiều, loại đích).

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
from hop.data import BUNDLE_FILES, Graph, Question  # noqa: E402
from hop.labels import answer_paths, teacher_paths  # noqa: E402
from hop.pipeline import Clock, Pause, Pipeline, check_config, load_config, select_hits  # noqa: E402
from hop.program import EdgeIndex, PrefixSIndex, concat_ranges  # noqa: E402
from hop.search import cap_candidates, depth_seats  # noqa: E402


def v7_cap(evidence, routes, topics, maximum):
    """Bản sao bước cắt pool của V7 (quota cố định 48/47/5%, chỗ thừa theo điểm toàn cục), để so."""
    from collections import defaultdict
    topics = set(topics)
    by_depth = defaultdict(list)
    for node in evidence:
        if node in topics:
            continue
        best = max(routes[node].values(), key=lambda row: row[0])
        depth = min(len(row[3]) for row in routes[node].values())
        by_depth[min(depth, 3)].append((best[0], node))
    for values in by_depth.values():
        values.sort(key=lambda item: (-item[0], item[1]))
    keep = set(topics)
    remaining = maximum - len(keep)
    quotas = {1: int(remaining * 0.48), 2: int(remaining * 0.47), 3: int(remaining * 0.05)}

    def closure(node):
        found, pending = {node}, [node]
        while pending:
            best = max(routes[pending.pop()].values(), key=lambda row: row[0])
            for ancestor in best[2]:
                if ancestor not in found:
                    found.add(ancestor)
                    pending.append(ancestor)
        return found

    def consider(node):
        required = closure(node)
        if len(keep | required) <= maximum:
            keep.update(required)

    for depth in (1, 2, 3):
        for _, node in by_depth[depth][:quotas[depth]]:
            consider(node)
    for _, node in sorted(((score, node) for values in by_depth.values() for score, node in values
                           if node not in keep), key=lambda item: (-item[0], item[1])):
        consider(node)
        if len(keep) == maximum:
            break
    return keep


def random_pool(rng, counts):
    """Cây ứng viên giả: topic 0, counts[d] nút ở độ sâu d + 1, điểm đường giảm dần theo độ sâu."""
    evidence, routes = {0: None}, {0: {"t": (0.0, -2, (0,), (), (), (), True)}}
    layers, next_id = [[0]], 1
    for count in counts:
        layer = []
        for _ in range(count):
            parent = layers[-1][int(rng.integers(len(layers[-1])))]
            score, _, path, edges, *_ = next(iter(routes[parent].values()))
            node, next_id = next_id, next_id + 1
            routes[node] = {"r": (score + float(rng.uniform(-3, 0)), 0, path + (node,), edges + (node,), (), (), True)}
            evidence[node] = None
            layer.append(node)
        layers.append(layer)
    return evidence, routes


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
        for name in ("default", "smoke", "tied", "prime", "adint", "smoke_prime", "smoke_adint"):
            check_config(load_config(HOP / "configs" / f"{name}.json"))
        smoke = load_config(HOP / "configs" / "smoke.json")
        self.assertEqual(smoke["synthetic"]["learning_rate"], 3e-05)  # kế thừa từ default
        self.assertEqual(smoke["synthetic"]["epochs"], 1)
        self.assertTrue(load_config(HOP / "configs" / "default.json")["flan"]["untie_lm_head"])
        self.assertFalse(load_config(HOP / "configs" / "tied.json")["flan"]["untie_lm_head"])
        default = load_config(HOP / "configs" / "default.json")
        self.assertEqual({default[p]["epochs"] for p in ("synthetic", "qa_sft", "policy_warmup", "policy",
                                                         "pool_ranker")}, {60})
        # PharmKG: pool như V7; action (quan hệ, chiều, loại đích) chung cho cả ba bộ từ 2026-10-29
        self.assertEqual((default["pool_depth_quotas"], default["pool_last_hop"], default["graph"]["action"]),
                         ([0.48, 0.47, 0.05], False, "target"))
        self.assertEqual({load_config(HOP / "configs" / f"{n}.json")["graph"]["action"]
                          for n in ("default", "prime", "adint", "tied", "smoke", "smoke_prime", "smoke_adint")},
                         {"target"})
        prime = load_config(HOP / "configs" / "prime.json")
        self.assertEqual((prime["label_edge_cap"], prime["local_top_k"], prime["path_beam"], prime["max_pool_nodes"],
                          prime["pool_depth_quotas"], prime["pool_last_hop"]),
                         (500, 250, 60, 3000, [0.43, 0.31, 0.26], True))
        adint = load_config(HOP / "configs" / "adint.json")
        self.assertEqual((adint["max_hops"], adint["graph"], adint["pool_depth_quotas"], adint["pool_last_hop"]),
                         (4, {"inverse": "missing", "action": "target"}, [0.48, 0.34, 0.10, 0.08], True))
        self.assertEqual(load_config(HOP / "configs" / "smoke_adint.json")["max_hops"], 4)  # kế thừa từ adint
        self.assertEqual(default["selection"]["tie_break"], "earliest")
        for config in (prime, adint):
            self.assertEqual(config["selection"]["tie_break"], "recall_at_20")
            self.assertEqual({config[p]["patience"] for p in ("synthetic", "qa_sft", "policy_warmup", "policy",
                                                              "pool_ranker")}, {5})
        with self.assertRaises(ValueError):
            check_config({**adint, "pool_depth_quotas": [0.5, 0.5, 0.0]})  # thiếu quota cho hop 4

    def test_select_hits(self):
        rows = [{"epoch": 0, "validation": {"mean_hits_at_1": 0.5}, "technical_valid": True},
                {"epoch": 1, "validation": {"mean_hits_at_1": 0.5}, "technical_valid": True},
                {"epoch": 2, "validation": {"mean_hits_at_1": 0.9}, "technical_valid": False}]
        self.assertEqual(select_hits(rows)["epoch"], 0)
        # Prime/ADint: hoà Hits@1 thì Recall@20 cao hơn thắng (QA-SFT Prime: e4/e6/e7 cùng 0,066, R@20 tăng dần).
        curve = [(4, 0.066, 0.309), (5, 0.064, 0.322), (6, 0.066, 0.321), (7, 0.066, 0.328)]
        rows = [{"epoch": e, "validation": {"mean_hits_at_1": h, "mean_recall_at_20": r}, "technical_valid": True}
                for e, h, r in curve]
        self.assertEqual(select_hits(rows)["epoch"], 4)
        self.assertEqual(select_hits(rows, "recall_at_20")["epoch"], 7)
        with self.assertRaises(ValueError):
            select_hits(rows, "latest")

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

    def test_depth_seats(self):
        # PharmKG: độ sâu 1 và 3 ít ứng viên, mọi chỗ thừa về độ sâu 2 (như V7).
        self.assertEqual(depth_seats([300, 1500, 30], [0.48, 0.47, 0.05], 1199), ([575, 563, 59], [300, 867, 30]))
        # Prime: chỗ thừa của độ sâu 1 chia cho độ sâu 2 và 3 theo tỉ lệ 31 : 26.
        self.assertEqual(depth_seats([500, 5000, 5000], [0.43, 0.31, 0.26], 2999),
                         ([1289, 929, 779], [500, 1358, 1138]))

    def test_cap_candidates_matches_v7_on_v7_quotas(self):
        rng = np.random.default_rng(3)
        # Cấu hình V7 giới hạn ứng viên: độ sâu 1 <= 6 nhánh x 60 = 360, độ sâu 3 <= 30 đường beam cuối, nên chỉ
        # độ sâu 2 thiếu chỗ và luật chia tiếp trùng V7. Khi nhiều độ sâu cùng thiếu (Prime), hai luật khác nhau.
        for counts in ([300, 1500, 30], [150, 900, 25], [360, 2500, 30]):
            evidence, routes = random_pool(rng, counts)
            kept, _, _ = cap_candidates(evidence, routes, (0,), 1200, (0.48, 0.47, 0.05))
            self.assertEqual(set(kept), v7_cap(evidence, routes, (0,), 1200))
        # Prime: độ sâu 3 giữ được phần chỗ của nó thay vì nhường cho nút nông điểm cao.
        evidence, routes = random_pool(rng, [500, 2500, 6000])
        kept, _, _ = cap_candidates(evidence, routes, (0,), 3000, (0.43, 0.31, 0.26))
        depth3 = sum(len(next(iter(routes[n].values()))[3]) == 3 for n in kept)
        self.assertGreaterEqual(depth3, 1100)
        self.assertLessEqual(len(kept), 3000)

    def test_teacher_skips_far_answers_and_four_hops(self):
        # chuỗi 0 - 1 - 2 - 3 - 4 (cạnh ngược được thêm): đáp án 4 cách topic 0 đúng 4 hop
        graph = Graph(toy_bundle(self.tmp / "c", [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 0, 4)]), verify=False)
        far = Question("far", "q", (0,), (4,), "train")
        near = Question("near", "q", (0,), (2,), "train")
        self.assertIsNone(answer_paths(graph, far, 4, max_hops=3)[1])
        buckets, shortest, _, _ = answer_paths(graph, far, 4, max_hops=4)
        self.assertEqual(shortest, 4)
        _pos, path = buckets[4][0]
        self.assertEqual([int(graph.src[e]) for e in path] + [int(graph.dst[path[-1]])], [0, 1, 2, 3, 4])
        teacher, depths = teacher_paths(graph, [far, near], 3)
        self.assertEqual((teacher[0], len(teacher[1]), depths), ([], 1, {"none": 1, "2": 1}))

    def test_target_action_selects_same_edges(self):
        rng = np.random.default_rng(5)
        triples = [(int(a), int(r), int(b)) for a, r, b in zip(rng.integers(0, 40, 300), rng.integers(0, 2, 300),
                                                               rng.integers(0, 40, 300))]
        root = toy_bundle(self.tmp / "t", triples)
        schema, target = Graph(root, verify=False), Graph(root, action="target", verify=False)
        self.assertLess(len(target.schemas), len(schema.schemas))
        np.testing.assert_array_equal(schema.src, target.src)
        for node in range(len(schema.ids)):
            # cùng một nút nguồn, nhóm cạnh theo action phải trùng nhóm theo schema
            edges = schema.out_edges(node)
            by_schema = {tuple(edges[schema.schema[edges] == a].tolist()) for a in set(schema.schema[edges].tolist())}
            by_target = {tuple(edges[target.schema[edges] == a].tolist()) for a in set(target.schema[edges].tolist())}
            self.assertEqual(by_schema, by_target)

    def test_setting_b_checkpoint_same_mass_and_gradient(self):
        from hop import setting_b
        from hop.labels import mine_routes
        from hop.policy import FullPolicy, GraphTensors, question_context
        from hop.program import Hop, Program, program_to_dict
        rng = np.random.default_rng(9)
        triples = [(int(a), int(r), int(b)) for a, r, b in zip(rng.integers(0, 25, 400), rng.integers(0, 2, 400),
                                                               rng.integers(0, 25, 400))]
        graph = Graph(toy_bundle(self.tmp / "s", triples), verify=False)
        codebook = build_codebook(rng.integers(0, 4, size=(len(graph.ids), 3)).astype(np.int16))
        gt = GraphTensors(graph, codebook, torch.device("cpu"))
        torch.manual_seed(0)
        policy = FullPolicy(len(graph.ids), len(graph.schemas), codebook.size, 3)
        topic = 0
        two_hop = {int(graph.dst[b]) for a in graph.out_edges(topic) for b in graph.out_edges(int(graph.dst[a]))}
        answer = sorted(two_hop - {int(graph.dst[a]) for a in graph.out_edges(topic)} - {topic})[0]
        q = Question("q", "question about n0", (topic,), (answer,), "train")
        routes, _ = mine_routes(graph, q, answer, 256, 42)
        self.assertTrue(routes and len(routes[0].edges) == 2)
        first = routes[0].edges
        program = Program(tuple(Hop(int(graph.schema[e]), None) for e in first))
        cached = {"beams": [{"program": program_to_dict(program), "probability": 0.7},
                            {"program": program_to_dict(Program(())), "probability": 0.3}]}

        def mass_and_grads():
            policy.zero_grad()
            mass = setting_b.answer_mass(policy, gt, q, cached, question_context(gt, q), routes, mode="B", tau=0.5,
                                         path_beam=30, local_k=60, rescue_slots=6)
            mass.backward()
            return mass.detach(), [p.grad.clone() for p in policy.parameters() if p.grad is not None]

        mass_ckpt, grads_ckpt = mass_and_grads()
        original = setting_b.checkpoint
        setting_b.checkpoint = lambda fn, *args, **kwargs: fn(*args)  # tắt checkpoint
        try:
            mass_plain, grads_plain = mass_and_grads()
        finally:
            setting_b.checkpoint = original
        self.assertGreater(float(mass_plain), 0)
        torch.testing.assert_close(mass_ckpt, mass_plain, rtol=0, atol=0)
        self.assertEqual(len(grads_ckpt), len(grads_plain))
        for a, b in zip(grads_ckpt, grads_plain):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_rows_cached_partial_save_creates_directory(self):
        pipe = object.__new__(Pipeline)
        pipe.clock = Clock(None)
        questions = [Question(str(i), "q", (0,), (1,), "train") for i in range(5)]
        target = self.tmp / "new_stage" / "pools.pkl"  # thư mục stage chưa tồn tại, như ranker/ ở run đủ
        rows = Pipeline.rows_cached(pipe, target, questions, lambda q: q.qid, "pool", every=2)
        self.assertEqual(rows, [q.qid for q in questions])
        self.assertTrue(target.exists() and not target.with_suffix(".partial.pkl").exists())

    def run_phase(self, directory: Path, pause_after: int | None):
        """Pha giả: một tham số, metric val tất định theo trọng số; dùng cả RNG numpy và torch."""
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        rng = np.random.default_rng(1)
        pipe = object.__new__(Pipeline)
        pipe.device = torch.device("cpu")
        pipe.clock = Clock(None)
        pipe.config = {"selection": {"tie_break": "earliest"}}
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
