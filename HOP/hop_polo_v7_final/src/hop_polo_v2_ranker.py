"""Fresh v2 bounded candidate pool and calibrated multi-positive ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hop_polo_hybrid import hash_words
from hop_polo_v2_search import coverage_diagnostics, pool_candidates
from hop_polo_aligned_data import MAX_HOPS


LOCAL_K = 60
PATH_BEAM = 30
POOL_DEPTHS = 2
TOP_K = 20
PATHS_PER_NODE = 3
NUMERIC_DIM = 29


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_set(value):
    return set(re.findall(r"[\w]+", value.casefold()))


@dataclass
class PoolRow:
    nodes: np.ndarray                 # graph-local IDs
    old_scores: np.ndarray            # float32, preserves epoch-0 ranking
    numeric: np.ndarray               # N x NUMERIC_DIM, float16
    relation_paths: np.ndarray        # N x 3 evidence paths x 3 hops, int16
    path_nodes: np.ndarray            # N x max_hops+1, best path node closure
    final_nodes: int
    edges_scored: int
    search_ms: float
    generation_ms: float
    decoder_edges_examined: int
    search_diagnostics: dict | None = None


@torch.inference_mode()
def collect_pool(policy, graph, codes, q, cache, degrees, name_terms,
                 text_terms, relation_terms, *, path_beam=30, local_k=60,
                 rescue_slots=6, max_nodes=1200):
    """Build the identical bounded, pre-prune pool for all QA splits."""
    started = time.perf_counter()
    nodes, evidence, routes, final_nodes, uncapped_nodes, trace = pool_candidates(
        policy, graph, codes, q, cache, degrees, path_beam=path_beam,
        local_k=local_k, rescue_slots=rescue_slots, maximum=max_nodes)
    device = next(policy.parameters()).device
    qvec = policy.question(hash_words(q.question), device)
    qterms = word_set(q.question)
    old_scores = policy.rank(qvec, nodes, evidence).cpu().numpy()
    relation_paths = np.full((len(nodes), PATHS_PER_NODE, MAX_HOPS), -1,
                             dtype=np.int16)
    path_nodes = np.full((len(nodes), MAX_HOPS + 1), -1,
                         dtype=np.int32)
    numeric = np.zeros((len(nodes), NUMERIC_DIM), dtype=np.float32)
    for index, node in enumerate(nodes):
        candidates = sorted(routes[node].values(), key=lambda row:
                            (-row[0], row[1], row[3]))
        assert candidates, (q.qid, node)
        selected = candidates[:PATHS_PER_NODE]
        best = selected[0]
        best_depth = len(best[3])
        path_nodes[index, :len(best[2])] = best[2]
        best_rel_terms = set()
        for rank, route in enumerate(selected):
            score, _program, _path, edges, backoffs, matches, _final = route
            for depth, edge in enumerate(edges):
                if edge >= 0:
                    rel_id = (int(graph.rel[edge]) +
                              len(graph.relation_names) * int(graph.direction[edge]))
                    relation_paths[index, rank, depth] = rel_id
                    if rank == 0:
                        best_rel_terms.update(relation_terms[rel_id])
            offset = 14 + 5 * rank
            numeric[index, offset:offset + 5] = (
                max(-30.0, score) / 30.0,
                len(edges) / MAX_HOPS,
                sum(matches) / max(1, len(matches)),
                sum(backoffs) / max(1, 3 * len(backoffs)),
                float(not route[6]),
            )
        numeric[index, 0] = float(old_scores[index])
        numeric[index, 1] = max(-30.0, best[0]) / 30.0
        numeric[index, 2] = best_depth / MAX_HOPS
        numeric[index, 3] = sum(best[5]) / max(1, best_depth)
        numeric[index, 4] = sum(best[4]) / max(1, 3 * best_depth)
        numeric[index, 5] = (best_depth - sum(best[5])) / MAX_HOPS
        numeric[index, 6] = float(node in final_nodes)
        numeric[index, 7] = math.log1p(len(candidates)) / 5
        numeric[index, 8] = len({route[1] for route in candidates}) / 5
        numeric[index, 9] = math.log1p(int(graph.indptr[node + 1] - graph.indptr[node])) / 16
        numeric[index, 10] = len(qterms & name_terms[node]) / max(1, len(qterms))
        numeric[index, 11] = len(qterms & text_terms[node]) / max(1, len(qterms))
        numeric[index, 12] = len(qterms & best_rel_terms) / max(1, len(qterms))
        numeric[index, 13] = float(node in q.topics)
    beam_nodes = set(q.topics)
    for state in trace.final:
        beam_nodes.update(state.nodes)
    diagnostics = coverage_diagnostics(graph, q, beam_nodes, trace)
    diagnostics.update({"uncapped_nodes": len(uncapped_nodes),
                        "beam_nodes": len(beam_nodes),
                        "pool_nodes": len(nodes),
                        "gold_in_uncapped_pool": len(set(q.answers) & uncapped_nodes),
                        "gold_in_capped_pool": len(set(q.answers) & set(nodes)),
                        "terminal_rescue_slots": trace.slots.get(-1, 0)})
    from v7_hard_audit import question_hard_audit
    diagnostics["v7_hard"] = question_hard_audit(graph, q, trace, nodes)
    return PoolRow(np.asarray(nodes, dtype=np.int32), old_scores.astype(np.float32),
                   numeric.astype(np.float16), relation_paths, path_nodes,
                   len(final_nodes), trace.edges_scored,
                   (time.perf_counter() - started) * 1000,
                   float(cache["generation_ms"]),
                   int(cache["decoder_edges_examined"]), diagnostics)


@torch.inference_mode()
def read_program_cache(path, questions):
    rows = [json.loads(line) for line in path.open()]
    if len(rows) >= len(questions) and len(questions) < 100:
        rows = rows[:len(questions)]  # bounded local smoke run
    if len(rows) != len(questions) or [row["qid"] for row in rows] != [q.qid for q in questions]:
        raise AssertionError(f"program cache QA order mismatch: {path}")
    return rows


@torch.inference_mode()
def encode_texts(model, tokenizer, values, device, batch_size=32, max_length=96,
                 individual_timing=False):
    """Frozen Flan encoder means; node/relation tables are built once."""
    encoder = model.get_encoder()
    encoder.eval()
    vectors = []
    milliseconds = []
    stride = 1 if individual_timing else batch_size
    for start in range(0, len(values), stride):
        block = values[start:start + stride]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        encoded = tokenizer(block, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
        states = encoder(input_ids=encoded.input_ids,
                         attention_mask=encoded.attention_mask).last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = F.normalize(pooled.float(), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - before) * 1000
        vectors.append(pooled.cpu().half())
        milliseconds.extend([elapsed / len(block)] * len(block))
    return torch.cat(vectors), milliseconds


class PoolRanker(nn.Module):
    """Residual old ranker plus frozen Flan text and ordered relation paths."""

    def __init__(self, node_text, relation_text, node_types):
        super().__init__()
        self.register_buffer("node_text", node_text.float())
        self.register_buffer("relation_text", relation_text.float())
        self.register_buffer("node_types", torch.as_tensor(node_types, dtype=torch.long))
        width = node_text.shape[1]
        self.q_project = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.Tanh())
        self.n_project = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.Tanh())
        self.r_project = nn.Sequential(nn.Linear(width, 48), nn.Tanh())
        self.path_project = nn.Sequential(nn.Linear(MAX_HOPS * 48, 96), nn.Tanh())
        self.type_embed = nn.Embedding(int(max(node_types)) + 1, 12)
        # q, node, path, q*node, |q-node|, q*path, type, numeric
        self.delta = nn.Sequential(nn.Linear(6 * 96 + 12 + NUMERIC_DIM, 192),
                                   nn.ReLU(), nn.Linear(192, 64), nn.ReLU(),
                                   nn.Linear(64, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.old_scale = nn.Parameter(torch.ones(()))

    def forward(self, nodes, old_scores, numeric, relation_paths, question_text):
        q = self.q_project(question_text.float().unsqueeze(0))
        n = self.n_project(self.node_text[nodes.long()])
        rel = self.r_project(self.relation_text)
        valid = relation_paths >= 0
        rid = relation_paths.long().clamp_min(0)
        r = rel[rid] * valid.unsqueeze(-1)
        path = self.path_project(r.reshape(len(nodes), PATHS_PER_NODE, -1))
        path_mask = (numeric[:, 14::5] != 0) | valid.any(-1)
        path_mask[:, 0] = True  # 0-hop topics have no relation tokens
        path_logits = numeric[:, 14::5] * 4 + (path * q.unsqueeze(1)).sum(-1) / math.sqrt(96)
        path_logits = path_logits.masked_fill(~path_mask, -1e4)
        attention = F.softmax(path_logits, dim=1)
        path = (path * attention.unsqueeze(-1)).sum(1)
        q = q.expand(len(nodes), -1)
        interaction = torch.cat((q, n, path, q * n, torch.abs(q - n), q * path,
                                 self.type_embed(self.node_types[nodes.long()]),
                                 numeric.float()), dim=1)
        return self.old_scale * old_scores.float() + self.delta(interaction).squeeze(1)


def cold_numeric_score(numeric):
    """Fixed, label-free starting score shared with early-stage proxies."""
    n = np.asarray(numeric, np.float32)
    return (1.8 * n[:, 10] + 0.8 * n[:, 11] + 0.8 * n[:, 12] +
            0.25 * n[:, 1] + 0.15 * n[:, 3] - 0.10 * n[:, 2] +
            0.10 * n[:, 13])


def standardized_old(old):
    values = np.asarray(old, np.float32)
    return np.clip((values - values.mean()) / max(float(values.std()), 1e-4),
                   -4.0, 4.0)


class PoolRankerV2(PoolRanker):
    """Train-only calibrated beam score, with a learnable residual gate."""

    def __init__(self, node_text, relation_text, node_types, old_gate):
        super().__init__(node_text, relation_text, node_types)
        self.old_scale = nn.Parameter(torch.tensor(float(old_gate)))

    def forward(self, nodes, old_scores, numeric, relation_paths, question_text):
        normalized = ((old_scores.float() - old_scores.float().mean()) /
                      old_scores.float().std(unbiased=False).clamp_min(1e-4))
        normalized = normalized.clamp(-4.0, 4.0)
        fixed = (1.8 * numeric[:, 10] + 0.8 * numeric[:, 11] +
                 0.8 * numeric[:, 12] + 0.25 * numeric[:, 1] +
                 0.15 * numeric[:, 3] - 0.10 * numeric[:, 2] +
                 0.10 * numeric[:, 13])
        features = numeric.clone()
        features[:, 0] = 0.0  # old head enters only through the explicit gate
        return fixed + super().forward(nodes, normalized, features,
                                       relation_paths, question_text)


def calibrate_old_gate(rows, questions, graph):
    """Choose the initial old-score gate using QA train labels only."""
    gates = (-0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 1.0)
    results = {}
    prepared = [(row.nodes, cold_numeric_score(row.numeric),
                 standardized_old(row.old_scores), set(q.answers), len(q.answers))
                for row, q in zip(rows, questions)]
    for gate in gates:
        h1, r20 = [], []
        for nodes, base, old, gold, count in prepared:
            scores = base + gate * old
            order = np.argsort(-scores, kind="stable")[:20]
            top = [int(nodes[j]) for j in order]
            h1.append(float(bool(top) and top[0] in gold))
            r20.append(len(set(top) & gold) / count)
        results[str(gate)] = {"mean_hits_at_1": float(np.mean(h1)),
                              "mean_recall_at_20": float(np.mean(r20))}
    baseline = results["0.0"]
    def key(gate):
        value = results[str(gate)]
        h = value["mean_hits_at_1"] / max(baseline["mean_hits_at_1"], 1e-3)
        r = value["mean_recall_at_20"] / max(baseline["mean_recall_at_20"], 1e-3)
        return (min(h, r), (h + r) / 2, -abs(gate))
    chosen = max(gates, key=key)
    return chosen, {"source": "QA train only", "chosen_gate": chosen,
                    "ablations": results}


def row_tensors(row, device):
    return (torch.as_tensor(row.nodes.astype(np.int64), device=device),
            torch.as_tensor(row.old_scores, device=device),
            torch.as_tensor(row.numeric.astype(np.float32), device=device),
            torch.as_tensor(row.relation_paths.astype(np.int64), device=device))


def training_labels(rows, questions, graph):
    """Build supervision exclusively from train QA answers."""
    labels = []
    stats = Counter()
    for row, q in zip(rows, questions):
        mask = np.isin(row.nodes, q.answers)
        positives = np.flatnonzero(mask)
        wrong = np.flatnonzero(~mask)
        stats["questions"] += 1
        stats["questions_with_positive"] += int(len(positives) > 0)
        stats["gold_in_pool"] += len(positives)
        stats["gold_total"] += len(q.answers)
        stats["candidate_nodes"] += len(row.nodes)
        if not len(positives) or not len(wrong):
            labels.append((positives, np.empty(0, dtype=np.int64)))
            continue
        # Freeze the old ranker's hard negatives, then add same-type
        # distractors. The full candidate pool still enters the listwise loss.
        old_order = wrong[np.argsort(-row.old_scores[wrong], kind="stable")]
        types = graph.types[row.nodes.astype(np.int64)]
        positive_types = set(int(x) for x in types[positives])
        same_type = [int(j) for j in old_order if int(types[j]) in positive_types]
        hard = list(dict.fromkeys(old_order[:32].tolist() + same_type[:32]))
        labels.append((positives.astype(np.int64), np.asarray(hard, dtype=np.int64)))
        stats["hard_wrong"] += len(hard)
        stats["same_type_hard_wrong"] += len(same_type[:32])
    return labels, dict(stats)


def train_epoch(ranker, rows, questions, question_text, labels, optimizer,
                device, rng, accumulation=16):
    ranker.train()
    # V7 fair ablation: every branch performs the same number of optimizer
    # updates even when its hard pool covers a different number of train QA.
    valid = [i for i, (positive, _hard) in enumerate(labels)
             if len(positive) and len(positive) < len(rows[i].nodes)]
    if not valid:
        raise RuntimeError("no train QA has a rankable positive pool candidate")
    order = []
    while len(order) < len(rows):
        order.extend(int(i) for i in rng.permutation(valid))
    order = order[:len(rows)]
    losses = []
    useful = 0
    optimizer.zero_grad(set_to_none=True)
    for position, index in enumerate(order, 1):
        index = int(index)
        positives, hard = labels[index]
        assert len(positives) and len(positives) < len(rows[index].nodes)
        nodes, old, numeric, relations = row_tensors(rows[index], device)
        values = ranker(nodes, old, numeric, relations,
                        question_text[index].to(device))
        pos = torch.as_tensor(positives, dtype=torch.long, device=device)
        wrong_mask = torch.ones(len(values), dtype=torch.bool, device=device)
        wrong_mask[pos] = False
        wrong = values[wrong_mask]
        # One positive at rank 1; multiple gold answers share this QA loss.
        top1_loss = torch.logsumexp(values, 0) - torch.logsumexp(values[pos], 0)
        # Each gold answer must beat the current top-20 wrong threshold.
        top_wrong = torch.topk(wrong, min(TOP_K, len(wrong))).values
        top20_loss = F.softplus(top_wrong[-1] - values[pos] + 0.2).mean()
        hard_indices = torch.as_tensor(hard, dtype=torch.long, device=device)
        hard_loss = F.softplus(values[hard_indices, None] - values[pos][None, :] + 0.15).mean()
        loss = 0.40 * top1_loss + 0.45 * top20_loss + 0.15 * hard_loss
        (loss / accumulation).backward()
        useful += 1
        losses.append(float(loss.detach()))
        if useful % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if position % 1000 == 0:
            print(f"ranker train {position}/{len(rows)} useful={useful}", flush=True)
    if useful % accumulation:
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": float(np.mean(losses)) if losses else None,
            "useful_queries": useful, "optimizer_updates": math.ceil(useful / accumulation)}


@torch.inference_mode()
def evaluate_ranker(ranker, rows, questions, question_text, text_ms,
                    graph, device, save_predictions=False):
    ranker.eval()
    predictions = []
    values_by_metric = defaultdict(list)
    for i, (row, q) in enumerate(zip(rows, questions)):
        nodes, old, numeric, relations = row_tensors(row, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        scores = ranker(nodes, old, numeric, relations,
                        question_text[i].to(device)).float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rank_ms = (time.perf_counter() - started) * 1000
        order = np.lexsort((graph.ids[row.nodes.astype(np.int64)], -scores))
        top_indices = order[:TOP_K]
        top_nodes = row.nodes[top_indices].astype(np.int64).tolist()
        selected = set(top_nodes) | set(q.topics)
        for index in top_indices:
            selected.update(int(node) for node in row.path_nodes[index] if node >= 0)
        pool_set = set(int(x) for x in row.nodes)
        gold = set(q.answers)
        top_set = set(top_nodes)
        ranked_selected = [int(row.nodes[j]) for j in order if int(row.nodes[j]) in selected]
        assert ranked_selected[:len(top_nodes)] == top_nodes
        hit_set = len(selected & gold)
        hit_top = len(top_set & gold)
        elapsed = row.generation_ms + row.search_ms + text_ms[i] + rank_ms
        record = {
            "recall_set": hit_set / len(gold),
            "precision_set": hit_set / len(selected),
            "f2_set": 5 * hit_set / (4 * len(gold) + len(selected)),
            "hits_at_1": float(bool(top_nodes) and top_nodes[0] in gold),
            "recall_at_20": hit_top / len(gold),
            "f2_at_20": 5 * hit_top / (4 * len(gold) + len(top_nodes)),
            "pool_recall": len(pool_set & gold) / len(gold),
            "selected_nodes_count": len(selected),
            "pool_nodes_count": len(pool_set),
            "gold_outside_pool": len(gold - pool_set),
            "gold_outside_subgraph": len(gold - selected),
            "gold_in_subgraph_below_20": len((gold & selected) - top_set),
            "edges_scored": row.edges_scored,
            "decoder_edges_examined": row.decoder_edges_examined,
            "generation_ms": row.generation_ms,
            "pool_search_ms": row.search_ms,
            "text_ms": text_ms[i],
            "rank_ms": rank_ms,
            "latency_ms": elapsed,
        }
        for key, value in record.items():
            values_by_metric[key].append(float(value))
        if save_predictions:
            score_map = {int(row.nodes[j]): float(scores[j]) for j in range(len(row.nodes))}
            path_map = {int(row.nodes[j]): [int(graph.ids[n]) for n in row.path_nodes[j] if n >= 0]
                        for j in order if int(row.nodes[j]) in selected}
            predictions.append({
                "qid": q.qid, "split": q.split,
                "topics": [int(graph.ids[x]) for x in q.topics],
                "answers": [int(graph.ids[x]) for x in q.answers],
                "pool_nodes": [int(graph.ids[x]) for x in row.nodes],
                "selected_nodes": [int(graph.ids[x]) for x in sorted(selected)],
                "ranked_nodes": [{"node_id": int(graph.ids[n]), "score": score_map[n],
                                  "best_path_nodes": path_map.get(n, [])}
                                 for n in ranked_selected],
                "ranked_top20": [int(graph.ids[x]) for x in top_nodes],
                **record,
            })
        if (i + 1) % 100 == 0:
            print(f"ranker evaluated {q.split} {i+1}/{len(rows)}", flush=True)
    metrics = {f"mean_{key}": float(np.mean(values))
               for key, values in values_by_metric.items()}
    for key in ("selected_nodes_count", "pool_nodes_count", "latency_ms",
                "edges_scored"):
        metrics[f"p95_{key}"] = float(np.percentile(values_by_metric[key], 95))
    metrics.update({"queries": len(rows), "local_top_k": LOCAL_K,
                    "path_beam": PATH_BEAM, "pool_pre_prune_depths": POOL_DEPTHS,
                    "top_k": TOP_K, "max_hops": MAX_HOPS,
                    "gold_answers_outside_pool_total": int(sum(values_by_metric["gold_outside_pool"])),
                    "gold_answers_outside_subgraph_total": int(sum(values_by_metric["gold_outside_subgraph"])),
                    "gold_answers_in_subgraph_below_20_total": int(sum(values_by_metric["gold_in_subgraph_below_20"]))})
    return metrics, predictions


def pool_summary(rows, questions):
    gold_in_pool = 0
    answer_total = 0
    queries_with_positive = 0
    sizes = []
    for row, q in zip(rows, questions):
        hit = len(set(int(x) for x in row.nodes) & set(q.answers))
        gold_in_pool += hit
        answer_total += len(q.answers)
        queries_with_positive += int(hit > 0)
        sizes.append(len(row.nodes))
    return {"queries": len(rows), "gold_in_pool": gold_in_pool,
            "gold_total": answer_total,
            "queries_with_positive": queries_with_positive,
            "mean_nodes": float(np.mean(sizes)),
            "p95_nodes": float(np.percentile(sizes, 95)),
            "mean_edges_scored": float(np.mean([r.edges_scored for r in rows]))}


