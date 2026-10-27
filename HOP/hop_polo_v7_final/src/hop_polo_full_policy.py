"""One HOP no-G + full-neighbor PoLo-style policy experiment on PharmKG.

The Flan planner is trained separately for three-hop HOP A/S programs. This
stage freezes it, trains one LSTM edge policy and one node ranker, then returns
the union of nodes on globally selected paths and ranks every node in it.
"""
from __future__ import annotations

import argparse
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

from hop_polo_aligned_data import MAX_HOPS
from hop_polo_flan3 import path_program
from hop_polo_hybrid import hash_words, program_to_dict, program_from_dict, question_overlap
from hopgs.codebooks import build_indexes
from hopgs.data import ADIntGraph
from hopgs.executor import Executor, Hop, Program
from hopgs.model import ProgramCodec, input_text


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


class FullPolicy(nn.Module):
    """Question-conditioned LSTM history; full-neighbor edge and node scores."""

    def __init__(self, n_nodes, n_actions, n_codes):
        super().__init__()
        self.words = nn.Embedding(16384, 64)
        self.nodes = nn.Embedding(n_nodes, 32)
        self.actions = nn.Embedding(n_actions + 1, 32)  # final ID is self-loop
        self.s_codes = nn.Embedding(n_codes, 16)
        self.history = nn.LSTMCell(64 + 32 + 32 + 32 + 16, 96)
        self.edge_mlp = nn.Sequential(nn.Linear(96 + 32 + 32 + 16 + 6, 64),
                                      nn.Tanh(), nn.Linear(64, 1))
        self.node_mlp = nn.Sequential(nn.Linear(64 + 32 + 96 + 4, 64),
                                      nn.Tanh(), nn.Linear(64, 1))
        self.n_actions = n_actions

    def question(self, tokens, device):
        return self.words(torch.as_tensor(tokens, dtype=torch.long, device=device)).mean(0)

    def code_vector(self, code, device):
        if not code:
            return torch.zeros(16, device=device)
        return self.s_codes(torch.as_tensor(code, dtype=torch.long, device=device)).mean(0)

    def step(self, qvec, current, previous_action, hop, hidden, cell):
        x = torch.cat((qvec, self.nodes.weight[current],
                       self.actions.weight[previous_action], self.actions.weight[hop.action],
                       self.code_vector(hop.s, qvec.device)))
        h, c = self.history(x.unsqueeze(0), (hidden.unsqueeze(0), cell.unsqueeze(0)))
        return h.squeeze(0), c.squeeze(0)

    def edge_logits(self, graph, codes, qvec, hidden, current, hop, edge_ids,
                    overlap, degrees, program_probability, chunk_size=4096):
        """Score every edge and the self-loop; normalize *after* concatenation.

        Chunks control tensor memory only. No edge is dropped or sampled.
        """
        device = qvec.device
        edges = np.asarray(edge_ids, np.int64)
        assert len(edges) and edges[-1] == -1
        n = len(edges)
        destinations = np.full(n, current, np.int64)
        actions = np.full(n, self.n_actions, np.int64)
        if n > 1:
            destinations[:-1] = graph.dst[edges[:-1]]
            actions[:-1] = graph.schema[edges[:-1]]
        code_rows = codes.node_codes[destinations]
        target = hop.s or ()
        matches = np.zeros(n, np.int64)
        for level, code in enumerate(target):
            matches += (matches == level) & (code_rows[:, level] == code)
        fallbacks = len(target) - matches
        action_match = actions == hop.action
        off_action = ~action_match
        overlap_values = np.asarray([overlap[int(dst)] for dst in destinations], np.float32)
        result = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dst = torch.as_tensor(destinations[start:end], dtype=torch.long, device=device)
            act = torch.as_tensor(actions[start:end], dtype=torch.long, device=device)
            code = torch.as_tensor(code_rows[start:end].astype(np.int64), device=device)
            extras = torch.stack((
                torch.as_tensor(action_match[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(matches[start:end] / 3, dtype=torch.float32, device=device),
                torch.as_tensor(fallbacks[start:end] / 3, dtype=torch.float32, device=device),
                torch.as_tensor(overlap_values[start:end], device=device),
                degrees[dst],
                torch.ones(end - start, device=device) * program_probability), dim=1)
            features = torch.cat((hidden.expand(end - start, -1), self.nodes(dst),
                                  self.actions(act), self.s_codes(code).mean(1), extras), dim=1)
            raw = self.edge_mlp(features).squeeze(1)
            # A/S are graded HOP priors, never a hard filter. The longest S
            # prefix is counted once; relaxed branches coexist with exact ones.
            # Once the HOP program ends, the PoLo policy chooses whether to
            # stop or explore another edge without an invented A target.
            prior_match = (torch.ones_like(extras[:, 0]) if hop.action == self.n_actions
                           else extras[:, 0])
            bias = (1.2 * prior_match + 1.05 * extras[:, 1]
                    - 2.1 * extras[:, 2] - 1.3 * (1 - prior_match))
            result.append(raw + bias)
        return torch.cat(result), fallbacks.tolist(), action_match.tolist(), destinations.tolist(), actions.tolist()

    def rank(self, qvec, nodes, evidence):
        device = qvec.device
        ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
        h = torch.stack([evidence[node][0] for node in nodes])
        extras = torch.tensor([[max(-30.0, evidence[n][1]) / 30,
                                min(evidence[n][2], 30) / 30,
                                evidence[n][3] / MAX_HOPS,
                                float(evidence[n][1] == 0)] for n in nodes],
                              dtype=torch.float32, device=device)
        features = torch.cat((qvec.expand(len(nodes), -1), self.nodes(ids), h, extras), dim=1)
        return self.node_mlp(features).squeeze(1)


@dataclass
class SearchState:
    program_index: int
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    backoffs: tuple[int, ...]
    action_matches: tuple[bool, ...]
    score: float
    node_scores: tuple[float, ...]
    node_hiddens: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    cell: torch.Tensor
    previous_action: int


def global_prune(states, width, duplicate_limit=2):
    # Global path scores, with at most two HOP program contexts per prefix.
    # The final beam has one copy of each evidence path.
    kept, counts = [], Counter()
    for state in sorted(states, key=lambda s: (-s.score, s.program_index, s.edges)):
        key = (state.nodes, state.edges)
        if counts[key] < duplicate_limit:
            kept.append(state)
            counts[key] += 1
        if len(kept) == width:
            break
    return kept


def program_cache(model, tokenizer, codec, graph, questions, cache_path, device,
                  length_penalty=0.0):
    rows = [json.loads(x) for x in cache_path.open()] if cache_path.exists() else []
    if [r["qid"] for r in rows] != [q.qid for q in questions[:len(rows)]]:
        rows = []
    if len(rows) == len(questions):
        return rows
    model.eval()
    for index in range(len(rows), len(questions)):
        q = questions[index]
        codec.executor.begin_query(10**9)
        started = time.perf_counter()
        encoded = tokenizer(input_text(graph, q), return_tensors="pt", truncation=True,
                            max_length=256, return_token_type_ids=False).to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, num_beams=5, num_return_sequences=5,
                                       max_new_tokens=max(26, 8 * codec.max_hops + 2),
                                       prefix_allowed_tokens_fn=lambda _batch, ids:
                                           codec.allowed(ids.tolist(), q.topics),
                                       return_dict_in_generate=True, output_scores=True,
                                       length_penalty=length_penalty, renormalize_logits=True,
                                       early_stopping=True)
        probabilities = F.softmax(generated.sequences_scores.float(), dim=0).cpu().tolist()
        beams = []
        for seq, probability in zip(generated.sequences, probabilities):
            program = codec.decoded(seq.tolist())
            if program is not None:
                beams.append({"program": program_to_dict(program), "probability": probability})
        rows.append({"qid": q.qid, "beams": beams,
                     "generation_ms": (time.perf_counter() - started) * 1000,
                     "decoder_edges_examined": codec.executor.query_edges_examined})
        if (index + 1) % 100 == 0:
            print(f"cached {cache_path.stem} {index+1}/{len(questions)}", flush=True)
            jsonl(cache_path, rows)
    jsonl(cache_path, rows)
    return rows


def target_hop(program, depth, stop_id):
    return program.hops[depth] if depth < len(program.hops) else Hop(stop_id, -1, None)


def full_beam_search(policy, graph, codes, q, cached, q_tokens, degrees,
                     path_beam, device, return_evidence=False):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    zero = torch.zeros(96, device=device)
    programs = [(program_from_dict(b["program"]), b["probability"]) for b in cached["beams"]]
    initial = []
    for i, (program, probability) in enumerate(programs):
        positions = program.return_topics if program.return_topics is not None else tuple(range(len(q.topics)))
        for pos in positions:
            if pos < len(q.topics):
                score = math.log(max(probability, 1e-12)) - math.log(len(positions))
                initial.append(SearchState(i, (q.topics[pos],), (), (), (), score,
                                           (score,), (zero,), zero, zero, policy.n_actions))
    if not initial:
        initial = [SearchState(-1, (topic,), (), (), (), 0.0, (0.0,), (zero,),
                               zero, zero, policy.n_actions) for topic in q.topics]
    active = global_prune(initial, path_beam, duplicate_limit=5)
    edges_scored = 0
    unique_edges = set()
    for depth in range(MAX_HOPS):
        expanded = []
        for state in active:
            program, probability = programs[state.program_index] if state.program_index >= 0 else (Program(()), 1.0)
            hop = target_hop(program, depth, policy.n_actions)
            current = state.nodes[-1]
            edges = graph.out_edges(current).tolist() + [-1]
            edges_scored += len(edges)
            unique_edges.update(edges[:-1])
            h, c = policy.step(qvec, current, state.previous_action, hop, state.hidden, state.cell)
            logits, backoffs, matches, destinations, actions = policy.edge_logits(
                graph, codes, qvec, h, current, hop, edges, overlap, degrees, probability)
            log_probs = F.log_softmax(logits, dim=0)
            k = min(path_beam, len(edges))
            top_values, top_indices = torch.topk(log_probs, k)
            for value, index_tensor in zip(top_values, top_indices):
                j = int(index_tensor)
                score = state.score + float(value)
                expanded.append(SearchState(state.program_index,
                    state.nodes + (destinations[j],), state.edges + (int(edges[j]),),
                    state.backoffs + (int(backoffs[j]),),
                    state.action_matches + (bool(matches[j]),), score,
                    state.node_scores + (score,), state.node_hiddens + (h,),
                    h, c, actions[j]))
        active = global_prune(expanded, path_beam,
                              duplicate_limit=1 if depth == MAX_HOPS - 1 else 2)
        if not active:
            break
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    node_paths = {topic: [] for topic in q.topics}
    paths = []
    for state in active:
        paths.append({"program_index": state.program_index,
                      "node_ids": [int(graph.ids[n]) for n in state.nodes],
                      "edge_ids": list(state.edges), "backoff_levels": list(state.backoffs),
                      "action_matches": list(state.action_matches), "log_score": state.score})
        for depth, node in enumerate(state.nodes):
            node_paths.setdefault(node, []).append({"program_index": state.program_index,
                "edge_ids": list(state.edges[:depth]),
                "backoff_levels": list(state.backoffs[:depth]),
                "action_matches": list(state.action_matches[:depth]),
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
    result = ([nodes[i] for i in order], paths, scores, nodes, node_paths,
              edges_scored, len(unique_edges))
    return result + (evidence,) if return_evidence else result


def teacher_loss(policy, graph, codes, q, teacher, q_tokens, degrees, device,
                 planner_guided_hops):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    zero = torch.zeros(96, device=device)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    if teacher is None:
        return torch.zeros((), device=device), torch.zeros((), device=device)
    pos, path = teacher
    program = path_program(graph, codes, q, pos, path)
    current = q.topics[pos]
    h = c = zero
    previous = policy.n_actions
    losses = []
    distractors = set()
    for depth in range(MAX_HOPS):
        hop = (target_hop(program, depth, policy.n_actions)
               if depth < planner_guided_hops else Hop(policy.n_actions, -1, None))
        edges = graph.out_edges(current).tolist() + [-1]
        target = path[depth] if depth < len(path) else -1
        if target not in edges:
            raise AssertionError("teacher edge absent from full neighbor list")
        h, c = policy.step(qvec, current, previous, hop, h, c)
        logits, _backoff, _match, destinations, actions = policy.edge_logits(
            graph, codes, qvec, h, current, hop, edges, overlap, degrees, 1.0)
        losses.append(F.cross_entropy(logits.unsqueeze(0),
                       torch.tensor([edges.index(target)], device=device)))
        # Hard local negatives include high-scoring wrong edges, without
        # removing any edge from the CE denominator.
        chosen = torch.topk(logits.detach(), min(32, len(edges))).indices.cpu().tolist()
        distractors.update(destinations[j] for j in chosen)
        current = destinations[edges.index(target)]
        evidence[current] = (h, 0.0, evidence.get(current, (None, None, 0, None))[2] + 1, depth + 1)
        previous = actions[edges.index(target)]
    for node in distractors:
        evidence.setdefault(node, (zero, -2.0, 1, 1))
    nodes = sorted(evidence)
    scores = policy.rank(qvec, nodes, evidence)
    positives = torch.as_tensor([node in q.answers for node in nodes], device=device)
    rank_loss = (F.softplus(scores[~positives][None, :] - scores[positives][:, None]).mean()
                 if positives.any() and (~positives).any() else torch.zeros((), device=device))
    return torch.stack(losses).mean(), rank_loss


def sample_subgraph(policy, graph, codes, q, cached, q_tokens, degrees, device, trajectories=4):
    qvec = policy.question(q_tokens, device)
    overlap = question_overlap(graph, q)
    zero = torch.zeros(96, device=device)
    programs = [(program_from_dict(b["program"]), b["probability"]) for b in cached["beams"]]
    weights = np.asarray([p for _, p in programs], np.float64)
    if len(weights):
        weights /= weights.sum()
    selected = set(q.topics)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    log_probs = []
    for _ in range(trajectories):
        if not programs:
            break
        i = int(np.random.choice(len(programs), p=weights))
        program, probability = programs[i]
        positions = program.return_topics if program.return_topics is not None else tuple(range(len(q.topics)))
        if not positions:
            continue
        current = q.topics[positions[int(np.random.randint(len(positions)))]]
        h = c = zero
        previous = policy.n_actions
        path_log = math.log(max(probability, 1e-12))
        for depth in range(MAX_HOPS):
            hop = target_hop(program, depth, policy.n_actions)
            edges = graph.out_edges(current).tolist() + [-1]
            h, c = policy.step(qvec, current, previous, hop, h, c)
            logits, _backoffs, _matches, destinations, actions = policy.edge_logits(
                graph, codes, qvec, h, current, hop, edges, overlap, degrees, probability)
            distribution = torch.distributions.Categorical(logits=logits)
            choice = distribution.sample()
            log_prob = distribution.log_prob(choice)
            log_probs.append(log_prob)
            j = int(choice)
            path_log += float(log_prob.detach())
            current = destinations[j]
            selected.add(current)
            old = evidence.get(current)
            if old is None or path_log > old[1]:
                evidence[current] = (h, path_log, 1 if old is None else old[2] + 1, depth + 1)
            previous = actions[j]
    gold = set(q.answers)
    hit = len(selected & gold)
    reward = 5 * hit / (4 * len(gold) + len(selected))
    nodes = sorted(selected)
    scores = policy.rank(qvec, nodes, evidence)
    positives = torch.as_tensor([node in gold for node in nodes], device=device)
    rank_loss = (F.softplus(scores[~positives][None, :] - scores[positives][:, None]).mean()
                 if positives.any() and (~positives).any() else torch.zeros((), device=device))
    return reward, (torch.stack(log_probs).sum() if log_probs else torch.zeros((), device=device)), rank_loss


def evaluate(policy, graph, codes, questions, cached_rows, tokens, degrees, device, path_beam):
    policy.eval()
    predictions = []
    with torch.inference_mode():
        for i, (q, cached, words) in enumerate(zip(questions, cached_rows, tokens), 1):
            started = time.perf_counter()
            ranked, paths, scores, nodes, node_paths, scored, unique = full_beam_search(
                policy, graph, codes, q, cached, words, degrees, path_beam, device)
            policy_ms = (time.perf_counter() - started) * 1000
            selected = set(nodes)
            gold = set(q.answers)
            top = ranked[:20]
            hit_set = len(selected & gold)
            hit_top = len(set(top) & gold)
            path_steps = max(1, sum(len(path["edge_ids"]) for path in paths))
            on_action_steps = sum(sum(path["action_matches"]) for path in paths)
            self_loop_steps = sum(sum(edge == -1 for edge in path["edge_ids"]) for path in paths)
            s_backoffs = []
            for path in paths:
                index = path["program_index"]
                program = (program_from_dict(cached["beams"][index]["program"])
                           if index >= 0 else Program(()))
                s_backoffs.extend(level for depth, level in enumerate(path["backoff_levels"])
                                  if depth < len(program.hops) and program.hops[depth].s is not None)
            score_by_node = {node: float(score) for node, score in zip(nodes, scores)}
            predictions.append({"qid": q.qid, "split": q.split,
                "topics": [int(graph.ids[n]) for n in q.topics],
                "answers": [int(graph.ids[n]) for n in q.answers],
                "selected_nodes": [int(graph.ids[n]) for n in nodes],
                "ranked_nodes": [{"node_id": int(graph.ids[n]), "score": score_by_node[n],
                                  "evidence_paths": sorted(node_paths.get(n, []),
                                      key=lambda p: -p["log_score"])[:3]} for n in ranked],
                "paths": paths, "programs": cached["beams"],
                "recall_set": hit_set / len(gold),
                "precision_set": hit_set / len(selected),
                "f2_set": 5 * hit_set / (4 * len(gold) + len(selected)),
                "hits_at_1": float(bool(top) and top[0] in gold),
                "recall_at_20": hit_top / len(gold),
                "f2_at_20": 5 * hit_top / (4 * len(gold) + len(top)) if top else 0.0,
                "gold_outside_subgraph": len(gold - selected),
                "gold_in_subgraph_below_20": len((gold & selected) - set(top)),
                "action_match_rate": on_action_steps / path_steps,
                "exact_s_rate": sum(level == 0 for level in s_backoffs) / max(1, len(s_backoffs)),
                "s_labeled_steps": len(s_backoffs),
                "mean_s_backoff": float(np.mean(s_backoffs)) if s_backoffs else 0.0,
                "self_loop_rate": self_loop_steps / path_steps,
                "edges_scored": scored, "unique_edges_examined": unique,
                "decoder_edges_examined": cached["decoder_edges_examined"],
                "generation_ms": cached["generation_ms"], "policy_ms": policy_ms,
                "latency_ms": cached["generation_ms"] + policy_ms})
            if i % 100 == 0:
                print(f"evaluated {q.split} {i}/{len(questions)}", flush=True)
    metrics = {"queries": len(predictions), "program_beam": 5,
               "path_beam": path_beam, "max_hops": 3, "top_k": 20,
               "edge_cap": None}
    for key in ("recall_set", "precision_set", "f2_set", "hits_at_1",
                "recall_at_20", "f2_at_20", "gold_outside_subgraph",
                "gold_in_subgraph_below_20", "action_match_rate", "exact_s_rate",
                "s_labeled_steps", "mean_s_backoff",
                "self_loop_rate", "edges_scored", "unique_edges_examined",
                "decoder_edges_examined", "generation_ms", "policy_ms", "latency_ms"):
        values = [float(row[key]) for row in predictions]
        metrics[f"mean_{key}"] = float(np.mean(values))
        if key in ("edges_scored", "unique_edges_examined", "decoder_edges_examined", "latency_ms"):
            metrics[f"p95_{key}"] = float(np.percentile(values, 95))
    metrics["gold_answers_outside_subgraph_total"] = sum(x["gold_outside_subgraph"] for x in predictions)
    metrics["gold_answers_in_subgraph_below_20_total"] = sum(x["gold_in_subgraph_below_20"] for x in predictions)
    return metrics, predictions


def select_checkpoint(history, tolerance=0.002):
    best_f2 = max(item["validation"]["mean_f2_set"] for item in history)
    eligible = [item for item in history if item["validation"]["mean_f2_set"] >= best_f2 - tolerance]
    return max(eligible, key=lambda item: (
        item["validation"]["mean_f2_at_20"],
        item["validation"]["mean_hits_at_1"],
        item["validation"]["mean_f2_set"],
        -item["validation"]["mean_edges_scored"]))
