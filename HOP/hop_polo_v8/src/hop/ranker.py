"""Pool ứng viên có trần (giống nhau cho mọi split) và pool ranker = điểm lạnh cố định + Δ học được.

V7 giữ một cột đặc trưng old-score và một cổng old-score luôn bằng 0 (đầu `node_mlp` của policy chưa
từng được dùng ở bước này); V8 bỏ cả hai, nên ranker chỉ còn `cold + delta`.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audit import question_hard_audit
from .common import word_set
from .search import coverage_diagnostics, pool_candidates

TOP_K = 20
PATHS_PER_NODE = 3
# Bố cục đặc trưng số của mỗi nút trong pool.
SCORE, DEPTH, MATCH, BACKOFF, OFF_ACTION, FINAL, ROUTES, PROGRAMS, DEGREE, NAME, TEXT, RELATION, TOPIC = range(13)
ROUTE0 = 13                       # 3 route tốt nhất × 5: điểm, độ dài, tỉ lệ khớp action, mức lùi S, ngoài beam cuối
NUMERIC_DIM = ROUTE0 + 5 * PATHS_PER_NODE


@dataclass
class PoolRow:
    nodes: np.ndarray            # ID cục bộ
    numeric: np.ndarray          # [N, NUMERIC_DIM] float16
    relation_paths: np.ndarray   # [N, 3 route, max_hops] int16
    path_nodes: np.ndarray       # [N, max_hops + 1] nút trên route tốt nhất
    final_nodes: int
    edges_scored: int
    search_ms: float
    generation_ms: float
    decoder_edges_examined: int
    search_diagnostics: dict | None = None


def empty_pool(cache: dict, max_hops: int) -> PoolRow:
    """Câu không có topic: không có pool; vẫn được tính vào mẫu số (chấm bằng backfill khi qua harness)."""
    return PoolRow(np.empty(0, np.int32), np.empty((0, NUMERIC_DIM), np.float16),
                   np.empty((0, PATHS_PER_NODE, max_hops), np.int16), np.empty((0, max_hops + 1), np.int32),
                   0, 0, 0.0, float(cache.get("generation_ms", 0.0)), int(cache.get("decoder_edges_examined", 0)),
                   None)


def cold_numeric_score(numeric) -> np.ndarray:
    """Điểm cố định không dùng nhãn: proxy chọn checkpoint ở các stage trước và điểm nền của ranker."""
    n = np.asarray(numeric, np.float32)
    return (1.8 * n[:, NAME] + 0.8 * n[:, TEXT] + 0.8 * n[:, RELATION] + 0.25 * n[:, SCORE] +
            0.15 * n[:, MATCH] - 0.10 * n[:, DEPTH] + 0.10 * n[:, TOPIC])


@torch.inference_mode()
def collect_pool(policy, gt, graph, q, cache, qctx, terms, *, path_beam, local_k, rescue_slots, max_nodes,
                 depth_quotas, last_hop) -> PoolRow:
    """Dựng cùng một pool có trần (ứng viên trước khi cắt beam, xem `search`) cho mọi split QA."""
    max_hops = policy.max_hops
    if not q.topics:
        return empty_pool(cache, max_hops)
    name_terms, text_terms, relation_terms = terms
    started = time.perf_counter()
    nodes, evidence, routes, final_nodes, uncapped_nodes, trace = pool_candidates(
        policy, gt, q, cache, qctx, path_beam=path_beam, local_k=local_k, rescue_slots=rescue_slots,
        maximum=max_nodes, depth_quotas=depth_quotas, last_hop=last_hop)
    qterms = word_set(q.question)
    n_rel = len(graph.relation_names)
    relation_paths = np.full((len(nodes), PATHS_PER_NODE, max_hops), -1, dtype=np.int16)
    path_nodes = np.full((len(nodes), max_hops + 1), -1, dtype=np.int32)
    numeric = np.zeros((len(nodes), NUMERIC_DIM), dtype=np.float32)
    for index, node in enumerate(nodes):
        candidates = sorted(routes[node].values(), key=lambda row: (-row[0], row[1], row[3]))
        selected = candidates[:PATHS_PER_NODE]
        best = selected[0]
        best_depth = len(best[3])
        path_nodes[index, :len(best[2])] = best[2]
        best_rel_terms = set()
        for rank, route in enumerate(selected):
            score, _program, _path, edges, backoffs, matches, final = route
            for depth, edge in enumerate(edges):
                if edge >= 0:
                    rel_id = int(graph.rel[edge]) + n_rel * int(graph.direction[edge])
                    relation_paths[index, rank, depth] = rel_id
                    if rank == 0:
                        best_rel_terms.update(relation_terms[rel_id])
            offset = ROUTE0 + 5 * rank
            numeric[index, offset:offset + 5] = (max(-30.0, score) / 30.0, len(edges) / max_hops,
                                                 sum(matches) / max(1, len(matches)),
                                                 sum(backoffs) / max(1, 3 * len(backoffs)), float(not final))
        numeric[index, SCORE] = max(-30.0, best[0]) / 30.0
        numeric[index, DEPTH] = best_depth / max_hops
        numeric[index, MATCH] = sum(best[5]) / max(1, best_depth)
        numeric[index, BACKOFF] = sum(best[4]) / max(1, 3 * best_depth)
        numeric[index, OFF_ACTION] = (best_depth - sum(best[5])) / max_hops
        numeric[index, FINAL] = float(node in final_nodes)
        numeric[index, ROUTES] = math.log1p(len(candidates)) / 5
        numeric[index, PROGRAMS] = len({route[1] for route in candidates}) / 5
        numeric[index, DEGREE] = math.log1p(int(graph.indptr[node + 1] - graph.indptr[node])) / 16
        numeric[index, NAME] = len(qterms & name_terms[node]) / max(1, len(qterms))
        numeric[index, TEXT] = len(qterms & text_terms[node]) / max(1, len(qterms))
        numeric[index, RELATION] = len(qterms & best_rel_terms) / max(1, len(qterms))
        numeric[index, TOPIC] = float(node in q.topics)
    beam_nodes = set(q.topics)
    for state in trace.final:
        beam_nodes.update(state.nodes)
    diagnostics = None
    if q.answers:  # chẩn đoán đọc nhãn; với test chỉ chạy sau khi đáp án đã được mở khoá
        diagnostics = coverage_diagnostics(graph, q, beam_nodes, trace)
        diagnostics.update({"uncapped_nodes": len(uncapped_nodes), "beam_nodes": len(beam_nodes),
                            "pool_nodes": len(nodes), "gold_in_uncapped_pool": len(set(q.answers) & uncapped_nodes),
                            "gold_in_capped_pool": len(set(q.answers) & set(nodes)),
                            "terminal_rescue_slots": trace.slots.get(-1, 0),
                            "hard": question_hard_audit(graph, q, trace, nodes, uncapped_nodes)})
    return PoolRow(np.asarray(nodes, dtype=np.int32), numeric.astype(np.float16), relation_paths, path_nodes,
                   len(final_nodes), trace.edges_scored, (time.perf_counter() - started) * 1000,
                   float(cache["generation_ms"]), int(cache["decoder_edges_examined"]), diagnostics)


@torch.inference_mode()
def encode_texts(model, tokenizer, values, device, batch_size=32, max_length=96, individual_timing=False):
    """Trung bình encoder Flan đã đóng băng; bảng nút/quan hệ dựng một lần."""
    encoder = model.get_encoder()
    encoder.eval()
    vectors, milliseconds = [], []
    stride = 1 if individual_timing else batch_size
    for start in range(0, len(values), stride):
        block = values[start:start + stride]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        encoded = tokenizer(block, padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt").to(device)
        states = encoder(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1)
        pooled = F.normalize(((states * mask).sum(1) / mask.sum(1).clamp_min(1)).float(), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - before) * 1000
        vectors.append(pooled.cpu().half())
        milliseconds.extend([elapsed / len(block)] * len(block))
    return torch.cat(vectors), milliseconds


class PoolRanker(nn.Module):
    """Điểm lạnh cố định + Δ từ văn bản Flan (câu hỏi, nút) và đường quan hệ có thứ tự."""

    def __init__(self, node_text, relation_text, node_types, max_hops: int):
        super().__init__()
        self.max_hops = max_hops
        self.register_buffer("node_text", node_text.float())
        self.register_buffer("relation_text", relation_text.float())
        self.register_buffer("node_types", torch.as_tensor(node_types, dtype=torch.long))
        width = node_text.shape[1]
        self.q_project = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.Tanh())
        self.n_project = nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.Tanh())
        self.r_project = nn.Sequential(nn.Linear(width, 48), nn.Tanh())
        self.path_project = nn.Sequential(nn.Linear(max_hops * 48, 96), nn.Tanh())
        self.type_embed = nn.Embedding(int(max(node_types)) + 1, 12)
        # q, nút, đường, q*nút, |q-nút|, q*đường, loại, số
        self.delta = nn.Sequential(nn.Linear(6 * 96 + 12 + NUMERIC_DIM, 192), nn.ReLU(),
                                   nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, nodes, numeric, relation_paths, question_text):
        q = self.q_project(question_text.float().unsqueeze(0))
        n = self.n_project(self.node_text[nodes.long()])
        rel = self.r_project(self.relation_text)
        valid = relation_paths >= 0
        r = rel[relation_paths.long().clamp_min(0)] * valid.unsqueeze(-1)
        path = self.path_project(r.reshape(len(nodes), PATHS_PER_NODE, -1))
        route_scores = numeric[:, ROUTE0::5]
        path_mask = (route_scores != 0) | valid.any(-1)
        path_mask[:, 0] = True  # topic 0 hop không có token quan hệ
        path_logits = (route_scores * 4 + (path * q.unsqueeze(1)).sum(-1) / math.sqrt(96)).masked_fill(~path_mask, -1e4)
        path = (path * F.softmax(path_logits, dim=1).unsqueeze(-1)).sum(1)
        q = q.expand(len(nodes), -1)
        interaction = torch.cat((q, n, path, q * n, torch.abs(q - n), q * path,
                                 self.type_embed(self.node_types[nodes.long()]), numeric.float()), dim=1)
        cold = (1.8 * numeric[:, NAME] + 0.8 * numeric[:, TEXT] + 0.8 * numeric[:, RELATION] +
                0.25 * numeric[:, SCORE] + 0.15 * numeric[:, MATCH] - 0.10 * numeric[:, DEPTH] +
                0.10 * numeric[:, TOPIC])
        return cold + self.delta(interaction).squeeze(1)


def row_tensors(row: PoolRow, device):
    return (torch.as_tensor(row.nodes.astype(np.int64), device=device),
            torch.as_tensor(row.numeric.astype(np.float32), device=device),
            torch.as_tensor(row.relation_paths.astype(np.int64), device=device))


def training_labels(rows, questions, graph):
    """Đáp án dương và nhiễu khó theo điểm lạnh cố định (không dùng nhãn val/test)."""
    labels, stats = [], Counter()
    for row, q in zip(rows, questions):
        mask = np.isin(row.nodes, q.answers)
        positives, wrong = np.flatnonzero(mask), np.flatnonzero(~mask)
        stats["questions"] += 1
        stats["questions_with_positive"] += int(len(positives) > 0)
        stats["gold_in_pool"] += len(positives)
        stats["gold_total"] += len(q.answers)
        stats["candidate_nodes"] += len(row.nodes)
        if not len(positives) or not len(wrong):
            labels.append((positives, np.empty(0, dtype=np.int64)))
            continue
        base = cold_numeric_score(row.numeric)
        ordered = wrong[np.argsort(-base[wrong], kind="stable")]
        types = graph.types[row.nodes.astype(np.int64)]
        positive_types = {int(x) for x in types[positives]}
        same_type = [int(j) for j in ordered if int(types[j]) in positive_types]
        hard = list(dict.fromkeys(ordered[:32].tolist() + same_type[:32]))
        labels.append((positives.astype(np.int64), np.asarray(hard, dtype=np.int64)))
        stats["hard_wrong"] += len(hard)
        stats["same_type_hard_wrong"] += len(same_type[:32])
    return labels, {**stats, "hard_negative_source": "cold_numeric_score"}


def train_epoch(ranker, rows, question_text, labels, optimizer, device, rng, accumulation=16):
    ranker.train()
    # Mọi nhánh có cùng số bước cập nhật kể cả khi pool phủ số câu train khác nhau (như V7).
    valid = [i for i, (positive, _hard) in enumerate(labels) if len(positive) and len(positive) < len(rows[i].nodes)]
    if not valid:
        raise RuntimeError("no train QA has a rankable positive pool candidate")
    order = []
    while len(order) < len(rows):
        order.extend(int(i) for i in rng.permutation(valid))
    order = order[:len(rows)]
    losses, useful = [], 0
    optimizer.zero_grad(set_to_none=True)
    for index in order:
        positives, hard = labels[index]
        nodes, numeric, relations = row_tensors(rows[index], device)
        values = ranker(nodes, numeric, relations, question_text[index].to(device))
        pos = torch.as_tensor(positives, dtype=torch.long, device=device)
        wrong_mask = torch.ones(len(values), dtype=torch.bool, device=device)
        wrong_mask[pos] = False
        wrong = values[wrong_mask]
        top1_loss = torch.logsumexp(values, 0) - torch.logsumexp(values[pos], 0)
        top_wrong = torch.topk(wrong, min(TOP_K, len(wrong))).values
        top20_loss = F.softplus(top_wrong[-1] - values[pos] + 0.2).mean()
        hard_index = torch.as_tensor(hard, dtype=torch.long, device=device)
        hard_loss = F.softplus(values[hard_index, None] - values[pos][None, :] + 0.15).mean()
        loss = 0.40 * top1_loss + 0.45 * top20_loss + 0.15 * hard_loss
        (loss / accumulation).backward()
        useful += 1
        losses.append(float(loss.detach()))
        if useful % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if useful % accumulation:
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": float(np.mean(losses)) if losses else None, "useful_queries": useful,
            "optimizer_updates": math.ceil(useful / accumulation)}


def score_record(q, top_nodes, selected, pool_set, row, text_ms, rank_ms) -> dict:
    gold = set(q.answers)
    hit_set, hit_top = len(selected & gold), len(set(top_nodes) & gold)
    return {"recall_set": hit_set / len(gold),
            "precision_set": hit_set / len(selected) if selected else 0.0,
            "f2_set": 5 * hit_set / (4 * len(gold) + len(selected)),
            "hits_at_1": float(bool(top_nodes) and top_nodes[0] in gold),
            "recall_at_20": hit_top / len(gold),
            "f2_at_20": 5 * hit_top / (4 * len(gold) + len(top_nodes)) if top_nodes else 0.0,
            "pool_recall": len(pool_set & gold) / len(gold),
            "selected_nodes_count": len(selected), "pool_nodes_count": len(pool_set),
            "gold_outside_pool": len(gold - pool_set), "gold_outside_subgraph": len(gold - selected),
            "gold_in_subgraph_below_20": len((gold & selected) - set(top_nodes)),
            "edges_scored": row.edges_scored, "decoder_edges_examined": row.decoder_edges_examined,
            "generation_ms": row.generation_ms, "pool_search_ms": row.search_ms, "text_ms": text_ms,
            "rank_ms": rank_ms, "latency_ms": row.generation_ms + row.search_ms + text_ms + rank_ms}


@torch.inference_mode()
def evaluate_ranker(ranker, rows, questions, question_text, text_ms, graph, device, save_predictions=False):
    ranker.eval()
    predictions = []
    values_by_metric = defaultdict(list)
    for i, (row, q) in enumerate(zip(rows, questions)):
        if not len(row.nodes):
            top_nodes, selected, pool_set, order, scores, rank_ms = [], set(), set(), np.empty(0, np.int64), np.empty(0), 0.0
        else:
            nodes, numeric, relations = row_tensors(row, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            scores = ranker(nodes, numeric, relations, question_text[i].to(device)).float().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rank_ms = (time.perf_counter() - started) * 1000
            order = np.lexsort((graph.ids[row.nodes.astype(np.int64)], -scores))
            top_nodes = row.nodes[order[:TOP_K]].astype(np.int64).tolist()
            selected = set(top_nodes) | set(q.topics)
            for index in order[:TOP_K]:
                selected.update(int(node) for node in row.path_nodes[index] if node >= 0)
            pool_set = {int(x) for x in row.nodes}
        record = score_record(q, top_nodes, selected, pool_set, row, text_ms[i], rank_ms)
        for key, value in record.items():
            values_by_metric[key].append(float(value))
        if save_predictions:
            ranked_selected = [int(row.nodes[j]) for j in order if int(row.nodes[j]) in selected]
            assert ranked_selected[:len(top_nodes)] == top_nodes
            path_map = {int(row.nodes[j]): [int(graph.ids[n]) for n in row.path_nodes[j] if n >= 0]
                        for j in order if int(row.nodes[j]) in selected}
            score_map = {int(row.nodes[j]): float(scores[j]) for j in range(len(row.nodes))}
            predictions.append({
                "qid": q.qid, "split": q.split, "no_topic": not q.topics,
                "topics": [int(graph.ids[x]) for x in q.topics], "answers": [int(graph.ids[x]) for x in q.answers],
                # toàn bộ pool theo thứ tự xếp hạng (điểm giảm, hoà thì ID nhỏ trước) cho harness/backfill
                "pool_ranked": [int(graph.ids[row.nodes[j]]) for j in order],
                "pool_scores": [float(scores[j]) for j in order],
                "selected_nodes": [int(graph.ids[x]) for x in sorted(selected)],
                "ranked_nodes": [{"node_id": int(graph.ids[n]), "score": score_map[n],
                                  "best_path_nodes": path_map.get(n, [])} for n in ranked_selected],
                "ranked_top20": [int(graph.ids[x]) for x in top_nodes], **record})
    metrics = {f"mean_{key}": float(np.mean(values)) for key, values in values_by_metric.items()}
    for key in ("selected_nodes_count", "pool_nodes_count", "latency_ms", "edges_scored"):
        metrics[f"p95_{key}"] = float(np.percentile(values_by_metric[key], 95))
    metrics.update({"queries": len(rows), "top_k": TOP_K, "max_hops": ranker.max_hops,
                    "queries_without_topic": sum(not q.topics for q in questions),
                    "gold_answers_outside_pool_total": int(sum(values_by_metric["gold_outside_pool"])),
                    "gold_answers_outside_subgraph_total": int(sum(values_by_metric["gold_outside_subgraph"])),
                    "gold_answers_in_subgraph_below_20_total": int(sum(values_by_metric["gold_in_subgraph_below_20"]))})
    return metrics, predictions


def pool_summary(rows, questions) -> dict:
    gold_in_pool = answer_total = queries_with_positive = 0
    sizes = []
    for row, q in zip(rows, questions):
        hit = len({int(x) for x in row.nodes} & set(q.answers))
        gold_in_pool += hit
        answer_total += len(q.answers)
        queries_with_positive += int(hit > 0)
        sizes.append(len(row.nodes))
    return {"queries": len(rows), "gold_in_pool": gold_in_pool, "gold_total": answer_total,
            "queries_with_positive": queries_with_positive, "mean_nodes": float(np.mean(sizes)),
            "p95_nodes": float(np.percentile(sizes, 95)), "mean_edges_scored": float(np.mean([r.edges_scored for r in rows]))}
