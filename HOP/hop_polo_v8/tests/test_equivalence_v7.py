"""V8 tính đúng như V7 trên PharmKG (cùng đồ thị, cùng codebook, cùng trọng số).

Chạy từ gốc repo, cần input V7 (HOP/hop_polo_v7_final/inputs, không nằm trong git), bundle PharmKG của V8
(tools/prep_inputs.py --dataset pharmkg) và bộ thư viện text đã ghim (.cache/hop_v8_pylib):

  PYTHONUTF8=1 PYTHONPATH=.cache/hop_v8_pylib .venv/Scripts/python.exe -m unittest HOP/hop_polo_v8/tests/test_equivalence_v7.py

Những gì V8 cố ý đổi so với V7 không nằm trong phép so này: cạnh ngược "missing" (ở đây dùng "always"
để có đúng đồ thị V7), bỏ mã G, bỏ cột old-score của ranker.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
V7_SRC = ROOT / "HOP/hop_polo_v7_final/src"
V7_INPUT = ROOT / "HOP/hop_polo_v7_final/inputs"
BUNDLE = ROOT / "HOP/hop_polo_v8/inputs/pharmkg"
sys.path.insert(0, str(ROOT / "HOP/hop_polo_v8/src"))
sys.path.insert(0, str(V7_SRC))

# V7
from hopgs.codebooks import build_indexes as v7_build_indexes  # noqa: E402
from hopgs.data import ADIntGraph  # noqa: E402
from hopgs.executor import Executor as V7Executor, Hop as V7Hop, Program as V7Program  # noqa: E402
from hopgs.model import ProgramCodec as V7Codec  # noqa: E402
import hop_polo_full_policy as v7_policy  # noqa: E402
import hop_polo_hybrid as v7_hybrid  # noqa: E402
import hop_polo_aligned_data as v7_aligned  # noqa: E402
import hop_polo_v2_search as v7_search  # noqa: E402
import hop_polo_v6_flan as v7_flan  # noqa: E402
import hop_polo_v6_ranker as v7_ranker  # noqa: E402
import v7_surrogate as v7_b  # noqa: E402

# V8
from hop import labels as v8_labels  # noqa: E402
from hop.codebook import build_codebook  # noqa: E402
from hop.data import Graph  # noqa: E402
from hop.policy import FullPolicy, GraphTensors, question_context, teacher_loss  # noqa: E402
from hop.program import Executor, Hop, Program, ProgramCodec, program_to_dict  # noqa: E402
from hop.ranker import collect_pool  # noqa: E402
from hop.search import pool_candidates  # noqa: E402
from hop.setting_b import answer_mass  # noqa: E402
from hop.common import word_set  # noqa: E402

DEVICE = torch.device("cpu")
# Cấu hình V7: 3 hop, quota cắt pool 48/47/5%, pool không lấy hop cuối.
V7_POOL = {"depth_quotas": (0.48, 0.47, 0.05), "last_hop": False}


def v7_hop(hop: Hop) -> V7Hop:
    return V7Hop(hop.action, -1, hop.s)


def v7_program(program: Program) -> V7Program:
    return V7Program(tuple(v7_hop(h) for h in program.hops), program.return_topics)


class EquivalenceV7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.g8 = Graph(BUNDLE, inverse="always")
        cls.g7 = ADIntGraph(V7_INPUT / "raw", seed=42)
        rng = np.random.default_rng(7)
        node_codes = rng.integers(0, 64, size=(len(cls.g8.ids), 3)).astype(np.int16)
        cls.cb8 = build_codebook(node_codes)
        cls.cb7 = v7_build_indexes(cls.g7, node_codes, np.zeros(len(cls.g7.src), np.int16))
        cls.ex8 = Executor(cls.g8, cls.cb8, 10 ** 9, prefix_index=True)
        cls.ex7 = V7Executor(cls.g7, cls.cb7, "semantic", edge_budget=10 ** 9, prefix_s_index=True)
        cls.gt = GraphTensors(cls.g8, cls.cb8, DEVICE)
        cls.degrees7 = torch.as_tensor(np.log1p(np.diff(cls.g7.indptr)) / 16, dtype=torch.float32)
        torch.manual_seed(3)
        cls.p8 = FullPolicy(len(cls.g8.ids), len(cls.g8.schemas), cls.cb8.size, 3)
        cls.p7 = v7_policy.FullPolicy(len(cls.g7.ids), len(cls.g7.schemas), int(node_codes.max()) + 1)
        cls.p7.load_state_dict(cls.p8.state_dict())
        cls.questions = random.Random(5).sample(cls.g7.splits["train"], 40)

    def test_graph_arrays(self):
        for name in ("ids", "src", "dst", "rel", "direction", "schema", "indptr", "reverse_order"):
            np.testing.assert_array_equal(getattr(self.g8, name), getattr(self.g7, name), err_msg=name)
        self.assertEqual(self.g8.schemas, self.g7.schemas)
        self.assertEqual(self.g8.texts, self.g7.texts)
        self.assertEqual(self.g8.fingerprint(), "2d0cd3464ba5253fe9eaed52267b8d7d62e3912f8bc6b17e3b3f4e63733173c5")

    def frontiers(self):
        rng = random.Random(11)
        result = []
        for q in self.questions[:20]:
            result.append(tuple(sorted(set(q.topics))))
            edges = self.g8.out_edges(q.topics[0])
            dests = sorted({int(self.g8.dst[e]) for e in edges})
            result.append(tuple(rng.sample(dests, min(len(dests), 300))))
        return result

    def test_executor_masks(self):
        for frontier in self.frontiers():
            counts7 = {a: c for (a, _g), c in self.ex7.action_counts(frontier).items()}
            self.assertEqual(self.ex8.action_counts(frontier), counts7)
            for action in list(counts7)[:5]:
                self.assertEqual(self.ex8.available_prefixes(frontier, action),
                                 self.ex7.available_prefixes(frontier, action, -1))
                np.testing.assert_array_equal(self.ex8.selected_edges(frontier, action),
                                              self.ex7.selected_edges(frontier, action, -1))

    def gold_programs(self, q, executor, codebook):
        programs = []
        for answer in q.answers[:2]:
            buckets, shortest, _, _ = v8_labels.answer_paths(self.g8, q, answer, max_hops=3)
            if shortest:
                for pos, path in buckets[shortest][:2]:
                    programs.append(v8_labels.path_program(self.g8, codebook, q, pos, path))
                    programs.append(Program(tuple(Hop(int(self.g8.schema[e]), None) for e in path)))
        return programs

    def test_answer_paths_and_execute(self):
        for q in self.questions:
            for answer in q.answers[:2]:
                self.assertEqual(v8_labels.answer_paths(self.g8, q, answer, max_hops=3)[:2],
                                 v7_aligned.answer_paths(self.g7, q, answer)[:2])
            for program in self.gold_programs(q, self.ex8, self.cb8):
                a = self.ex8.execute(q.topics, program)
                b = self.ex7.execute(q.topics, v7_program(program))
                self.assertEqual((a.candidates, a.evidence, a.edges_examined, a.executable),
                                 (b.candidates, b.evidence, b.edges_examined, b.executable))

    def test_codec_masks(self):
        from transformers import AutoTokenizer
        t8 = AutoTokenizer.from_pretrained(V7_INPUT / "base_model", local_files_only=True)
        t7 = AutoTokenizer.from_pretrained(V7_INPUT / "base_model", local_files_only=True)
        c8 = ProgramCodec(t8, self.g8, self.ex8, 3)
        c7 = V7Codec(t7, self.g7, self.ex7, max_hops=3, include_g=False)
        self.assertEqual(c8.tokens, c7.tokens)
        self.assertEqual(c8.id, c7.id)
        rng = random.Random(13)
        for q in self.questions[:15]:
            for _ in range(4):
                ids = [0]
                for _step in range(14):
                    a8, a7 = c8.allowed(ids[1:], q.topics), c7.allowed(ids[1:], q.topics)
                    self.assertEqual(a8, a7)
                    token = rng.choice(a8)
                    if token == t8.eos_token_id:
                        break
                    ids.append(token)

    def test_edge_logits(self):
        rng = random.Random(17)
        for q in self.questions[:20]:
            qctx = question_context(self.gt, q)
            overlap7 = v7_hybrid.question_overlap(self.g7, q)
            qvec = self.p8.question(qctx.words, DEVICE)
            for program in self.gold_programs(q, self.ex8, self.cb8)[:2] or [Program(())]:
                current = q.topics[0]
                hop = program.hops[0] if program.hops else Hop(self.p8.n_actions, None)
                hidden = torch.randn(96)
                probability = rng.random()
                with torch.no_grad():
                    s8 = self.p8.edge_scores(self.gt, qctx.overlap, hidden, current, hop, probability)
                    edges = self.g7.out_edges(current).tolist() + [-1]
                    logits7, fb7, match7, dst7, act7 = self.p7.edge_logits(
                        self.g7, self.cb7, qvec, hidden, current, v7_hop(hop), edges, overlap7, self.degrees7,
                        probability)
                torch.testing.assert_close(s8.logits, logits7, atol=2e-5, rtol=1e-5)
                self.assertEqual(s8.fallbacks.tolist(), fb7)
                self.assertEqual(s8.action_match.tolist(), match7)
                self.assertEqual(s8.static.dst.tolist(), dst7)
                self.assertEqual(s8.static.act.tolist(), act7)
                np.testing.assert_allclose(qctx.overlap[s8.static.dst].numpy(),
                                           np.asarray([overlap7[d] for d in dst7], np.float32))

    def test_teacher_loss(self):
        for q in self.questions[:15]:
            teacher, _ = v8_labels.teacher_paths(self.g8, [q], 3)
            for chosen in teacher[0][:2]:
                ce8, rank8 = teacher_loss(self.p8, self.gt, self.g8, self.cb8, q, chosen,
                                          question_context(self.gt, q), DEVICE)
                ce7, rank7 = v7_policy.teacher_loss(self.p7, self.g7, self.cb7, q, chosen,
                                                    v7_hybrid.hash_words(q.question), self.degrees7, DEVICE, 3)
                torch.testing.assert_close(ce8, ce7, atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(rank8, rank7, atol=2e-5, rtol=1e-5)

    def cache_for(self, q):
        programs = self.gold_programs(q, self.ex8, self.cb8)[:4] + [Program(())]
        weights = np.linspace(1.0, 0.2, len(programs))
        weights /= weights.sum()
        return {"qid": q.qid, "beams": [{"program": program_to_dict(p), "probability": float(w)}
                                        for p, w in zip(programs, weights)],
                "generation_ms": 0.0, "decoder_edges_examined": 0}

    def test_pool_and_features(self):
        names = [word_set(x) for x in self.g8.names]
        texts = [word_set(x) for x in self.g8.texts]
        rel_names = self.g8.relation_names + [f"inverse of {x}" for x in self.g8.relation_names]
        relations = [word_set(x) for x in rel_names]
        same_nodes = 0
        for q in self.questions[:12]:
            cached = self.cache_for(q)
            qctx = question_context(self.gt, q)
            nodes8, *_ = pool_candidates(self.p8, self.gt, q, cached, qctx, path_beam=30, local_k=60,
                                         rescue_slots=6, maximum=1200, **V7_POOL)
            nodes7, *_ = v7_search.pool_candidates(self.p7, self.g7, self.cb7, q, cached, self.degrees7,
                                                   path_beam=30, local_k=60, rescue_slots=6, maximum=1200)
            jaccard = len(set(nodes8) & set(nodes7)) / max(1, len(set(nodes8) | set(nodes7)))
            self.assertGreater(jaccard, 0.98, q.qid)
            same_nodes += nodes8 == nodes7
            if nodes8 != nodes7:
                continue
            row8 = collect_pool(self.p8, self.gt, self.g8, q, cached, qctx, (names, texts, relations),
                                path_beam=30, local_k=60, rescue_slots=6, max_nodes=1200, **V7_POOL)
            row7 = v7_ranker.collect_pool(self.p7, self.g7, self.cb7, q, cached, self.degrees7, names, texts, relations,
                                          path_beam=30, local_k=60, rescue_slots=6, max_nodes=1200)
            np.testing.assert_array_equal(row8.nodes, row7.nodes)
            np.testing.assert_allclose(row8.numeric.astype(np.float32), row7.numeric[:, 1:].astype(np.float32),
                                       atol=2e-3)
            np.testing.assert_array_equal(row8.relation_paths, row7.relation_paths)
            np.testing.assert_array_equal(row8.path_nodes, row7.path_nodes)
        self.assertGreaterEqual(same_nodes, 10)

    def test_setting_b_mass(self):
        checked = 0
        for q in self.questions:
            cached = self.cache_for(q)
            qctx = question_context(self.gt, q)
            for answer in q.answers[:2]:
                routes8, audit8 = v8_labels.mine_routes(self.g8, q, answer, 256, 42)
                routes7, audit7 = v7_b.mine_routes(self.g7, q, answer, 256, 42)
                self.assertEqual([(r.topic_pos, r.edges) for r in routes8], [(r.topic_pos, r.edges) for r in routes7])
                self.assertEqual(audit8, audit7)
                if not routes8 or len(routes8[0].edges) > 2:
                    continue
                for mode in ("B", "C"):
                    m8 = answer_mass(self.p8, self.gt, q, cached, qctx, routes8, mode=mode, tau=0.5,
                                     path_beam=30, local_k=60, rescue_slots=6)
                    m7 = v7_b.answer_mass(self.p7, self.g7, self.cb7, q, cached, self.degrees7, answer, routes7,
                                          mode=mode, tau=0.5, path_beam=30, local_k=60, rescue_slots=6)
                    torch.testing.assert_close(m8, m7, atol=1e-6, rtol=2e-4)
                checked += 1
        self.assertGreater(checked, 5)

    def test_executable_labels(self):
        from transformers import AutoTokenizer
        t8 = AutoTokenizer.from_pretrained(V7_INPUT / "base_model", local_files_only=True)
        t7 = AutoTokenizer.from_pretrained(V7_INPUT / "base_model", local_files_only=True)
        lab8 = Executor(self.g8, self.cb8, 200, prefix_index=False)
        lab7 = V7Executor(self.g7, self.cb7, "semantic", edge_budget=200, prefix_s_index=True)
        c8 = ProgramCodec(t8, self.g8, self.ex8, 3)
        c7 = V7Codec(t7, self.g7, self.ex7, max_hops=3, include_g=False)
        subset = self.questions[:25]
        plans8, summary8, _ = v8_labels.executable_qa_labels(self.g8, self.cb8, lab8, c8, subset, cap=200,
                                                             programs_per_answer=3, paths_per_depth=32,
                                                             max_hops=3)
        plans7, summary7, _ = v7_flan.executable_qa_labels(self.g7, self.cb7, lab7, c7, subset, cap=200,
                                                           programs_per_answer=3, paths_per_depth=32)
        self.assertEqual(plans8, plans7)
        self.assertEqual({k: v for k, v in summary8.items()}, {k: v for k, v in summary7.items()})


if __name__ == "__main__":
    unittest.main()
