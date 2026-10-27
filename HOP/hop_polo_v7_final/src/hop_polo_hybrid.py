"""One frozen no-G HOP planner with a learned edge policy and subgraph ranker.

Reusable policy and program helpers for the from-scratch V7 final run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hopgs.codebooks import build_indexes
from hopgs.data import ADIntGraph, Question
from hopgs.evaluate import answer_distance_group
from hopgs.executor import Executor, Hop, Program
from hopgs.model import ProgramCodec, input_text


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_words(value: str, bins: int = 16384) -> list[int]:
    words = re.findall(r"[\w]+", value.casefold())[:96]
    return [int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little") % bins
            for w in words] or [0]


def program_to_dict(program: Program) -> dict:
    return {"hops": [{"action": h.action, "s": list(h.s) if h.s is not None else None}
                      for h in program.hops], "return_topics": program.return_topics}


def program_from_dict(row: dict) -> Program:
    return Program(tuple(Hop(int(h["action"]), -1,
                             tuple(h["s"]) if h["s"] is not None else None)
                         for h in row["hops"]),
                   tuple(row["return_topics"]) if row["return_topics"] is not None else None)


def mine_teacher_paths(graph: ADIntGraph, q: Question, max_per_answer: int = 4,
                       max_total: int = 8) -> list[tuple[int, ...]]:
    """Find several directed 0/1/2-hop gold paths, balancing gold answers."""
    by_answer: dict[int, list[tuple[int, ...]]] = {}
    first = [(int(e), int(graph.dst[e])) for t in q.topics for e in graph.out_edges(t)]
    first_by_destination: dict[int, list[int]] = {}
    for edge, destination in first:
        first_by_destination.setdefault(destination, []).append(edge)
    for answer in q.answers:
        paths: list[tuple[int, ...]] = [()] if answer in q.topics else []
        paths += [(edge,) for edge in first_by_destination.get(answer, ())]
        for second in graph.in_edges(answer):
            for edge in first_by_destination.get(int(graph.src[second]), ()):
                if int(graph.src[edge]) != answer:  # avoid a needless return to an already-gold topic
                    paths.append((edge, int(second)))
        paths = list(dict.fromkeys(paths))
        paths.sort(key=lambda path: (len(path),
                   sum(math.log1p(int(graph.indptr[int(graph.src[e]) + 1] -
                                      graph.indptr[int(graph.src[e])])) for e in path), path))
        by_answer[answer] = paths[:max_per_answer]
    result: list[tuple[int, ...]] = []
    for index in range(max_per_answer):
        for answer in q.answers:
            if index < len(by_answer[answer]):
                path = by_answer[answer][index]
                if path not in result:
                    result.append(path)
                if len(result) >= max_total:
                    return result
    return result


def teacher_program(graph: ADIntGraph, codes, path: tuple[int, ...]) -> Program:
    return Program(tuple(Hop(int(graph.schema[e]), -1,
                             tuple(map(int, codes.node_codes[int(graph.dst[e])]))) for e in path))


class HybridPolicy(nn.Module):
    """PoLo-like history LSTM plus one shared representation for node ranking."""

    def __init__(self, n_nodes: int, n_actions: int, n_codes: int):
        super().__init__()
        self.words = nn.Embedding(16384, 64)
        self.nodes = nn.Embedding(n_nodes, 32)
        self.actions = nn.Embedding(n_actions + 1, 32)
        self.s_codes = nn.Embedding(n_codes, 16)
        self.history = nn.LSTMCell(64 + 32 + 32 + 32 + 16, 96)
        self.edge_mlp = nn.Sequential(nn.Linear(96 + 32 + 32 + 16 + 4, 64),
                                      nn.Tanh(), nn.Linear(64, 1))
        self.node_mlp = nn.Sequential(nn.Linear(64 + 32 + 96 + 4, 64),
                                      nn.Tanh(), nn.Linear(64, 1))
        self.n_actions = n_actions

    def question(self, tokens: list[int], device: torch.device):
        return self.words(torch.as_tensor(tokens, dtype=torch.long, device=device)).mean(0)

    def code_vector(self, code: tuple[int, ...] | None, device: torch.device):
        if not code:
            return torch.zeros(16, device=device)
        return self.s_codes(torch.as_tensor(code, dtype=torch.long, device=device)).mean(0)

    def step(self, qvec, current: int, previous_action: int, hop: Hop, hidden, cell):
        x = torch.cat((qvec, self.nodes.weight[current],
                       self.actions.weight[previous_action], self.actions.weight[hop.action],
                       self.code_vector(hop.s, qvec.device)))
        return self.history(x.unsqueeze(0), (hidden.unsqueeze(0), cell.unsqueeze(0)))

    def edge_logits(self, graph, codes, qvec, hidden, hop: Hop,
                    edge_ids: list[int], word_overlap: dict[int, float],
                    degrees: torch.Tensor, program_probability: float,
                    backoff_penalty: float):
        device = qvec.device
        edges = np.asarray(edge_ids, np.int64)
        destinations = torch.as_tensor(graph.dst[edges].astype(np.int64), device=device)
        n = len(edges)
        s_codes = torch.as_tensor(codes.node_codes[graph.dst[edges]].astype(np.int64), device=device)
        destination_s = self.s_codes(s_codes).mean(1)
        fallback = []
        for row in codes.node_codes[graph.dst[edges]]:
            target = hop.s or ()
            matched = 0
            for a, b in zip(target, row):
                if int(a) != int(b):
                    break
                matched += 1
            fallback.append(len(target) - matched)
        fallback_tensor = torch.tensor(fallback, dtype=torch.float32, device=device)
        overlap = torch.tensor([word_overlap[int(x)] for x in graph.dst[edges]],
                               dtype=torch.float32, device=device)
        extras = torch.stack((fallback_tensor / 3,
                              overlap,
                              degrees[destinations],
                              torch.full((n,), program_probability, device=device)), dim=1)
        features = torch.cat((hidden.expand(n, -1), self.nodes(destinations),
                              self.actions.weight[hop.action].expand(n, -1), destination_s,
                              extras), dim=1)
        return self.edge_mlp(features).squeeze(1) - backoff_penalty * fallback_tensor, fallback

    def rank(self, qvec, nodes: list[int], evidence: dict[int, tuple[torch.Tensor, float, int, int]]):
        device = qvec.device
        ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
        h = torch.stack([evidence[n][0] for n in nodes])
        extras = torch.tensor([[max(-30.0, evidence[n][1]) / 30,
                                min(evidence[n][2], 30) / 30,
                                evidence[n][3] / 2,
                                float(evidence[n][1] == 0)] for n in nodes],
                              dtype=torch.float32, device=device)
        x = torch.cat((qvec.expand(len(nodes), -1), self.nodes(ids), h, extras), dim=1)
        return self.node_mlp(x).squeeze(1)


@dataclass
class SearchState:
    program_index: int
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    backoffs: tuple[int, ...]
    score: float
    node_scores: tuple[float, ...]
    node_hiddens: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    cell: torch.Tensor
    previous_action: int


def prune_paths(states: list[SearchState], width: int) -> list[SearchState]:
    """Keep distinct program/depth/backoff evidence before filling the path beam."""
    ordering = lambda s: (-s.score, s.program_index, s.edges)
    by_branch = {}
    for state in states:
        key = (state.program_index, len(state.edges),
               state.backoffs[-1] if state.backoffs else -1)
        if key not in by_branch or ordering(state) < ordering(by_branch[key]):
            by_branch[key] = state
    reserved = sorted(by_branch.values(), key=ordering)[:width]
    keys = {(s.program_index, s.edges) for s in reserved}
    for state in sorted(states, key=ordering):
        if len(reserved) >= width:
            break
        key = (state.program_index, state.edges)
        if key not in keys:
            reserved.append(state)
            keys.add(key)
    return sorted(reserved, key=ordering)


def generate_program_cache(model, tokenizer, codec, graph, questions, cache_path: Path,
                           cap: int, device: torch.device):
    """Frozen Flan beams; retain decoder-read edge IDs for the shared query cap."""
    rows = [json.loads(line) for line in cache_path.open()] if cache_path.exists() else []
    if [r["qid"] for r in rows] != [q.qid for q in questions[:len(rows)]]:
        rows = []
    if len(rows) == len(questions):
        return rows
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for index in range(len(rows), len(questions)):
        q = questions[index]
        codec.executor.begin_query(cap)
        encoded = tokenizer(input_text(graph, q, codec.input_style), return_tensors="pt",
                            truncation=True, max_length=256, return_token_type_ids=False).to(device)
        model.eval()
        with torch.inference_mode():
            generated = model.generate(**encoded, num_beams=5, num_return_sequences=5,
                                       max_new_tokens=26,
                                       prefix_allowed_tokens_fn=lambda _batch, ids:
                                           codec.allowed(ids.tolist(), q.topics),
                                       return_dict_in_generate=True, output_scores=True,
                                       length_penalty=0.0, renormalize_logits=True,
                                       early_stopping=True)
        probabilities = F.softmax(generated.sequences_scores.float(), dim=0).cpu().tolist()
        beams = []
        for seq, p in zip(generated.sequences, probabilities):
            program = codec.decoded(seq.tolist())
            if program is not None:
                beams.append({"program": program_to_dict(program), "probability": p})
        observed = sorted({int(edge) for posting in codec.executor._query_edge_cache.values()
                           for edge in posting})
        rows.append({"qid": q.qid, "beams": beams,
                     "decoder_edges_examined": codec.executor.query_edges_examined,
                     "observed_edges": observed})
        if (index + 1) % 100 == 0:
            print(f"cached frozen Flan programs {cache_path.stem}: {index + 1}/{len(questions)}", flush=True)
            jsonl(cache_path, rows)  # a partial file can be resumed after interruption
    jsonl(cache_path, rows)
    return rows


class QueryBudget:
    def __init__(self, graph, executor, observed: list[int], decoder_cost: int, cap: int):
        self.graph, self.executor = graph, executor
        self.observed = set(observed)
        self.decoder_cost = decoder_cost
        self.cap = cap
        self.new_cost = 0
        self.postings = {}

    def edges(self, source: int, action: int) -> list[int]:
        key = (source, action)
        if key in self.postings:
            return self.postings[key]
        group = self.executor.index.get((source, action, 0))
        selected = []
        if group is not None:
            for edge in group:
                edge = int(edge)
                if edge not in self.observed:
                    if self.decoder_cost + self.new_cost >= self.cap:
                        continue
                    self.observed.add(edge)
                    self.new_cost += 1
                selected.append(edge)
        self.postings[key] = selected
        return selected

    @property
    def cost(self):
        return self.decoder_cost + self.new_cost


class LazyOverlap(dict):
    def __init__(self, graph, q):
        super().__init__()
        self.graph = graph
        self.terms = set(re.findall(r"[\w]+", q.question.casefold()))

    def __missing__(self, node):
        value = len(self.terms.intersection(re.findall(r"[\w]+",
            self.graph.names[node].casefold()))) / max(1, len(self.terms))
        self[node] = value
        return value


def question_overlap(graph, q: Question):
    return LazyOverlap(graph, q)


def beam_search(policy, graph, codes, executor, q, cached, q_tokens, degrees,
                path_beam: int, cap: int, backoff_penalty: float, device):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    budget = QueryBudget(graph, executor, cached["observed_edges"], cached["decoder_edges_examined"], cap)
    zero = torch.zeros(96, device=device)
    beams = [(program_from_dict(b["program"]), b["probability"]) for b in cached["beams"]]
    initial = []
    for i, (program, probability) in enumerate(beams):
        positions = program.return_topics if program.return_topics is not None else tuple(range(len(q.topics)))
        for pos in positions:
            if pos < len(q.topics):
                topic = q.topics[pos]
                initial_score = math.log(max(probability, 1e-12)) - math.log(len(positions))
                initial.append(SearchState(i, (topic,), (), (),
                                           initial_score, (initial_score,), (zero,),
                                           zero, zero, policy.n_actions))
    if not initial:
        initial = [SearchState(-1, (topic,), (), (), 0.0, (0.0,), (zero,),
                               zero, zero, policy.n_actions)
                   for topic in q.topics]
    active, finished = initial, []
    for depth in range(2):
        expanded = []
        for state in active:
            program = beams[state.program_index][0] if state.program_index >= 0 else Program(())
            if depth >= len(program.hops):
                finished.append(state)
                continue
            hop = program.hops[depth]
            edges = budget.edges(state.nodes[-1], hop.action)
            if not edges:
                continue
            h, c = policy.step(qvec, state.nodes[-1], state.previous_action, hop,
                               state.hidden, state.cell)
            h, c = h.squeeze(0), c.squeeze(0)
            logits, fallbacks = policy.edge_logits(graph, codes, qvec, h, hop, edges,
                                                   overlap, degrees, beams[state.program_index][1],
                                                   backoff_penalty)
            log_probs = F.log_softmax(logits, dim=0).detach().cpu().tolist()
            for edge, logp, fallback in zip(edges, log_probs, fallbacks):
                expanded.append(SearchState(state.program_index,
                    state.nodes + (int(graph.dst[edge]),), state.edges + (edge,),
                    state.backoffs + (fallback,), state.score + logp,
                    state.node_scores + (state.score + logp,), state.node_hiddens + (h,),
                    h, c, hop.action))
        active = prune_paths(expanded, path_beam)
        if not active:
            break
    finished.extend(active)
    finished = prune_paths(finished, path_beam)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    node_paths = {topic: [] for topic in q.topics}
    paths = []
    for state in finished:
        paths.append({"program_index": state.program_index, "node_ids": [int(graph.ids[n]) for n in state.nodes],
                      "edge_ids": list(state.edges), "backoff_levels": list(state.backoffs),
                      "log_score": state.score})
        for depth, node in enumerate(state.nodes):
            node_paths.setdefault(node, []).append({"program_index": state.program_index,
                "edge_ids": list(state.edges[:depth]),
                "backoff_levels": list(state.backoffs[:depth]),
                "log_score": state.node_scores[depth]})
            old = evidence.get(node)
            count = 1 if old is None else old[2] + 1
            if old is None or state.node_scores[depth] > old[1]:
                evidence[node] = (state.node_hiddens[depth], state.node_scores[depth], count, depth)
            else:
                evidence[node] = (old[0], old[1], count, old[3])
    nodes = sorted(evidence)
    scores = policy.rank(qvec, nodes, evidence)
    order = sorted(range(len(nodes)), key=lambda i: (-float(scores[i].detach()), int(graph.ids[nodes[i]])))
    ranked = [nodes[i] for i in order]
    return ranked, paths, budget.cost, budget.decoder_cost, budget.new_cost, scores, nodes, node_paths


def teacher_loss(policy, graph, codes, executor, q, paths, q_tokens, degrees,
                 backoff_penalty, device, max_negatives=128):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    zero = torch.zeros(96, device=device)
    losses = []
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    distractors = set()
    for path in paths:
        if not path:
            continue
        program = teacher_program(graph, codes, path)
        current = int(graph.src[path[0]])
        h, c = zero, zero
        previous = policy.n_actions
        for depth, (hop, target) in enumerate(zip(program.hops, path), 1):
            posting = executor.index.get((current, hop.action, 0))
            candidates = [int(e) for e in posting] if posting is not None else []
            if target not in candidates:
                raise AssertionError("teacher edge absent from its action posting")
            if len(candidates) > max_negatives + 1:
                # Deterministic evenly spaced negatives; always keep the gold edge.
                chosen = np.linspace(0, len(candidates) - 1, max_negatives, dtype=int)
                candidates = sorted(set(candidates[i] for i in chosen) | {target})
            distractors.update(int(graph.dst[e]) for e in candidates[:32])
            h, c = policy.step(qvec, current, previous, hop, h, c)
            h, c = h.squeeze(0), c.squeeze(0)
            logits, _ = policy.edge_logits(graph, codes, qvec, h, hop, candidates,
                                          overlap, degrees, 1.0, backoff_penalty)
            losses.append(F.cross_entropy(logits.unsqueeze(0),
                         torch.tensor([candidates.index(target)], device=device)))
            current = int(graph.dst[target])
            old = evidence.get(current)
            evidence[current] = (h, 0.0, 1 if old is None else old[2] + 1, depth)
            previous = hop.action
    if not losses:
        ce = torch.zeros((), device=device)
    else:
        ce = torch.stack(losses).mean()
    # Node supervision uses QA train only, with hard neighbor distractors.
    for node in sorted(distractors):
        evidence.setdefault(node, (zero, -2.0, 1, 1))
    node_ids = sorted(evidence)
    rank_logits = policy.rank(qvec, node_ids, evidence)
    positives = torch.tensor([node in q.answers for node in node_ids], device=device)
    if positives.any() and (~positives).any():
        rank_loss = F.softplus(rank_logits[~positives][None, :] - rank_logits[positives][:, None]).mean()
    else:
        rank_loss = torch.zeros((), device=device)
    return ce, rank_loss


def sample_subgraph(policy, graph, codes, executor, q, cached, q_tokens, degrees,
                    backoff_penalty, device, trajectories=4):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    budget = QueryBudget(graph, executor, cached["observed_edges"], cached["decoder_edges_examined"], 64)
    zero = torch.zeros(96, device=device)
    beams = [(program_from_dict(b["program"]), b["probability"]) for b in cached["beams"]]
    weights = np.asarray([p for _, p in beams], dtype=np.float64)
    if len(weights):
        weights /= weights.sum()
    selected_nodes = set(q.topics)
    log_probs = []
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    for _ in range(trajectories):
        if not beams:
            break
        i = int(np.random.choice(len(beams), p=weights))
        program, probability = beams[i]
        positions = program.return_topics if program.return_topics is not None else tuple(range(len(q.topics)))
        if not positions:
            continue
        current = q.topics[positions[int(np.random.randint(len(positions)))]]
        h, c, previous = zero, zero, policy.n_actions
        path_log = math.log(max(probability, 1e-12))
        for depth, hop in enumerate(program.hops, 1):
            candidates = budget.edges(current, hop.action)
            if not candidates:
                break
            h, c = policy.step(qvec, current, previous, hop, h, c)
            h, c = h.squeeze(0), c.squeeze(0)
            logits, _ = policy.edge_logits(graph, codes, qvec, h, hop, candidates,
                                          overlap, degrees, probability, backoff_penalty)
            distribution = torch.distributions.Categorical(logits=logits)
            choice = distribution.sample()
            log_prob = distribution.log_prob(choice)
            log_probs.append(log_prob)
            path_log += float(log_prob.detach())
            current = int(graph.dst[candidates[int(choice)]])
            selected_nodes.add(current)
            old = evidence.get(current)
            if old is None or path_log > old[1]:
                evidence[current] = (h, path_log, 1 if old is None else old[2] + 1, depth)
            previous = hop.action
    gold = set(q.answers)
    tp = len(selected_nodes & gold)
    reward = 5 * tp / (4 * len(gold) + len(selected_nodes))
    node_ids = sorted(selected_nodes)
    logits = policy.rank(qvec, node_ids, evidence)
    positives = torch.tensor([node in gold for node in node_ids], device=device)
    rank_loss = (F.softplus(logits[~positives][None, :] - logits[positives][:, None]).mean()
                 if positives.any() and (~positives).any() else torch.zeros((), device=device))
    return reward, (torch.stack(log_probs).sum() if log_probs else torch.zeros((), device=device)), rank_loss


def evaluate(policy, graph, codes, executor, questions, cached_rows, tokens, degrees,
             device, cap, path_beam, backoff_penalty):
    policy.eval()
    predictions = []
    with torch.inference_mode():
        for index, (q, cached, words) in enumerate(zip(questions, cached_rows, tokens), 1):
            started = time.perf_counter()
            ranked, paths, edges, decoder, new, scores, node_ids, node_paths = beam_search(
                policy, graph, codes, executor, q, cached, words, degrees,
                path_beam, cap, backoff_penalty, device)
            selected = set(node_ids)
            gold = set(q.answers)
            top = ranked[:20]
            overlap_set = len(selected & gold)
            overlap_top = len(set(top) & gold)
            group, distances = answer_distance_group(graph, q)
            predictions.append({"qid": q.qid, "split": q.split,
                "topics": [int(graph.ids[n]) for n in q.topics],
                "answers": [int(graph.ids[n]) for n in q.answers],
                "selected_nodes": [int(graph.ids[n]) for n in node_ids],
                "ranked_nodes": [{"node_id": int(graph.ids[n]),
                                  "score": float(scores[node_ids.index(n)]),
                                  "evidence_paths": sorted(node_paths.get(n, []),
                                      key=lambda p: -p["log_score"])[:3]} for n in ranked],
                "paths": paths,
                "programs": cached["beams"],
                "distance_group": group,
                "answer_shortest_hops": {str(int(graph.ids[n])): d for n, d in distances.items()},
                "recall_set": overlap_set / len(gold),
                "precision_set": overlap_set / len(selected) if selected else 0.0,
                "f2_set": 5 * overlap_set / (4 * len(gold) + len(selected)),
                "hits_at_1": float(bool(ranked) and ranked[0] in gold),
                "recall_at_20": overlap_top / len(gold),
                "f2_at_20": 5 * overlap_top / (4 * len(gold) + len(top)) if top else 0.0,
                "gold_outside_subgraph": len(gold - selected),
                "gold_in_subgraph_below_20": len((gold & selected) - set(top)),
                "edges_examined": edges, "decoder_edges_examined": decoder,
                "policy_new_edges_examined": new,
                "latency_ms": (time.perf_counter() - started) * 1000})
            if index % 100 == 0:
                print(f"evaluated {q.split} {index}/{len(questions)}", flush=True)
    metrics = {"queries": len(predictions), "cap": cap, "program_beam": 5,
               "path_beam": path_beam, "max_hops": 2, "top_k": 20}
    for key in ("recall_set", "precision_set", "f2_set", "hits_at_1", "recall_at_20",
                "f2_at_20", "gold_outside_subgraph", "gold_in_subgraph_below_20",
                "edges_examined", "decoder_edges_examined", "policy_new_edges_examined",
                "latency_ms"):
        values = [float(row[key]) for row in predictions]
        metrics[f"mean_{key}"] = float(np.mean(values))
        if key in ("edges_examined", "latency_ms"):
            metrics[f"p95_{key}"] = float(np.percentile(values, 95))
    metrics["gold_answers_outside_subgraph_total"] = sum(r["gold_outside_subgraph"] for r in predictions)
    metrics["gold_answers_in_subgraph_below_20_total"] = sum(r["gold_in_subgraph_below_20"] for r in predictions)
    return metrics, predictions
