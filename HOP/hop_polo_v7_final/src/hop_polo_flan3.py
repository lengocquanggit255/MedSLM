"""Train the no-G, three-hop HOP planner for the full-neighbor PoLo hybrid.

Only PharmKG QA train creates SFT labels. Validation selects the planner epoch;
test is reserved for the final hybrid evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hopgs.codebooks import build_indexes
from hopgs.data import ADIntGraph
from hopgs.executor import Executor, Hop, Program
from hopgs.model import ProgramCodec, input_text, make_synthetic_examples, train_supervised


GRAPH_HASH = "2d0cd3464ba5253fe9eaed52267b8d7d62e3912f8bc6b17e3b3f4e63733173c5"


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class GrammarCodec(ProgramCodec):
    """HOP syntax and globally existing S codes, without a graph edge lookup."""

    def __init__(self, tokenizer, graph, executor):
        super().__init__(tokenizer, graph, executor, max_hops=3, include_g=False)
        rows = executor.codes.node_codes
        self.prefixes = {length: set(map(tuple, rows[:, :length].tolist()))
                         for length in range(1, rows.shape[1] + 1)}
        self.n_actions = len(graph.schemas)

    def allowed(self, ids, topics):
        program, state, _action, _g, prefix = self.parse(ids)
        if state == "done":
            return [self.tokenizer.eos_token_id]
        if state == "invalid":
            return [self.id["<STOP>"]]
        if state == "return":
            remaining = [self.id[f"<T_{i}>"] for i in range(len(topics))
                         if i not in (program.return_topics or ())]
            return remaining + ([self.id["<END_RETURN>"]] if program.return_topics else [])
        if state == "return_end":
            return [self.id["<STOP>"]]
        if state == "start":
            choices = [self.id["<STOP>"]]
            if not program.hops and topics:
                choices.append(self.id["<RETURN_TOPIC>"])
            if len(program.hops) < self.max_hops:
                choices.append(self.id["<HOP>"])
            return choices
        if state == "action":
            return [self.id[f"<A_{i}>"] for i in range(self.n_actions)]
        if state == "s":
            return [self.id["<S_ANY>"]] + [self.id[f"<S1_{p[0]}>"]
                   for p in sorted(self.prefixes[1])]
        if state == "end":
            return [self.id["<ENDHOP>"]]
        if state in ("s1", "s2", "s3"):
            choices = [self.id["<ENDHOP>"]]
            length = len(prefix)
            if length < len(self.prefixes):
                choices += [self.id[f"<S{length + 1}_{p[-1]}>"]
                            for p in sorted(self.prefixes[length + 1]) if p[:-1] == prefix]
            return choices
        return [self.id["<STOP>"]]


def mine_balanced_paths(graph, question, max_paths=4, scan_limit=300_000):
    """Find gold paths of all depths, retaining one per depth before filling.

    Three-hop search is bounded only during *label mining*; this is audited.
    Retrieval itself scores every outgoing edge with no cap.
    """
    gold = set(question.answers)
    buckets = defaultdict(list)
    for pos, topic in enumerate(question.topics):
        if topic in gold:
            buckets[0].append((pos, ()))
    first = [(pos, int(e)) for pos, topic in enumerate(question.topics)
             for e in graph.out_edges(topic)]
    for pos, a in first:
        if int(graph.dst[a]) in gold:
            buckets[1].append((pos, (a,)))
    last = defaultdict(list)
    for answer in gold:
        for edge in graph.in_edges(answer):
            last[int(graph.src[edge])].append(int(edge))
    for pos, a in first:
        for b in last.get(int(graph.dst[a]), ())[:8]:
            buckets[2].append((pos, (a, b)))
            if len(buckets[2]) >= 128:
                break
        if len(buckets[2]) >= 128:
            break
    scanned, censored = 0, False
    for pos, a in first:
        for b in graph.out_edges(int(graph.dst[a])):
            scanned += 1
            for c in last.get(int(graph.dst[b]), ())[:4]:
                buckets[3].append((pos, (a, int(b), c)))
                if len(buckets[3]) >= 128:
                    break
            if len(buckets[3]) >= 128 or scanned >= scan_limit:
                break
        if len(buckets[3]) >= 128 or scanned >= scan_limit:
            censored = scanned >= scan_limit
            break
    degree = np.diff(graph.indptr)
    def cost(item):
        _pos, path = item
        return (sum(math.log1p(int(degree[int(graph.src[e])])) for e in path), path)
    chosen = []
    for depth in range(4):
        if buckets[depth]:
            chosen.append(min(buckets[depth], key=cost))
    all_items = sorted((item for bucket in buckets.values() for item in bucket), key=cost)
    for item in all_items:
        if len(chosen) >= max_paths:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:max_paths], scanned, censored


def path_program(graph, codes, question, topic_index, path):
    if not path:
        return Program((), (topic_index,)) if len(question.topics) > 1 else Program(())
    return Program(tuple(Hop(int(graph.schema[e]), -1,
                             tuple(int(x) for x in codes.node_codes[int(graph.dst[e])]))
                         for e in path))


def planner_validation(model, tokenizer, codec, graph, questions, device):
    """Validation-only strict execution diagnostic for choosing a Flan epoch."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for i, q in enumerate(questions, 1):
            codec.executor.begin_query(10**9)
            encoded = tokenizer(input_text(graph, q), return_tensors="pt", truncation=True,
                                max_length=256, return_token_type_ids=False).to(device)
            generated = model.generate(**encoded, num_beams=5, num_return_sequences=5,
                                       max_new_tokens=26,
                                       prefix_allowed_tokens_fn=lambda _batch, ids:
                                           codec.allowed(ids.tolist(), q.topics),
                                       return_dict_in_generate=True, output_scores=True,
                                       length_penalty=0.0, renormalize_logits=True,
                                       early_stopping=True)
            candidates = set(q.topics)
            programs = []
            for sequence in generated.sequences:
                program = codec.decoded(sequence.tolist())
                if program is None:
                    continue
                programs.append(codec.tokens_for(program))
                if program.return_topics is not None:
                    candidates.update(q.topics[p] for p in program.return_topics if p < len(q.topics))
                    continue
                frontier = set(q.topics)
                for hop in program.hops:
                    following = set()
                    for node in frontier:
                        for e in graph.out_edges(node):
                            if int(graph.schema[e]) != hop.action:
                                continue
                            dst = int(graph.dst[e])
                            if hop.s is None or tuple(int(v) for v in codec.executor.codes.node_codes[dst, :len(hop.s)]) == hop.s:
                                following.add(dst)
                    candidates.update(following)
                    frontier = following
                    if not frontier:
                        break
            gold = set(q.answers)
            hit = len(candidates & gold)
            rows.append({"qid": q.qid, "candidate_count": len(candidates),
                         "recall_set": hit / len(gold),
                         "f2_set": 5 * hit / (4 * len(gold) + len(candidates)),
                         "programs": programs,
                         "decoder_edges_examined": codec.executor.query_edges_examined})
            if i % 200 == 0:
                print(f"planner val {i}/{len(questions)}", flush=True)
    return {"queries": len(rows), "mean_recall_set": float(np.mean([r["recall_set"] for r in rows])),
            "mean_f2_set": float(np.mean([r["f2_set"] for r in rows])),
            "mean_decoder_edges_examined": float(np.mean([r["decoder_edges_examined"] for r in rows]))}, rows
