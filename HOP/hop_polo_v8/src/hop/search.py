"""Tìm kiếm HOP/PoLo có ngân sách slot theo chương trình, và pool ứng viên trước khi cắt (độ sâu 1-2)."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .labels import MAX_HOPS
from .policy import (HIDDEN, FullPolicy, GraphTensors, QuestionContext, SearchState, global_prune,
                     initial_score, target_hop)
from .program import Hop, Program, program_from_dict


@dataclass
class SearchTrace:
    programs: list[tuple[Program, float]]
    slots: dict[int, int]
    terminal_indices: list[int]
    terminal_mass: float
    pre: list[list[SearchState]]
    final: list[SearchState]
    by_depth: list[list[SearchState]]
    edges_scored: int
    unique_edges: int
    search_ms: float


def allocate_slots(programs, width: int = 30, rescue_slots: int = 6, minimum: int = 3):
    """Giữ chỗ rescue cho topic khi có chương trình kết thúc sớm, và tối thiểu cho mỗi beam có hop."""
    indices = [i for i, (p, _) in enumerate(programs) if p.hops]
    terminal = [i for i, (p, _) in enumerate(programs) if not p.hops]
    rescue = min(rescue_slots, width) if terminal or not indices else 0
    remaining = width - rescue
    if indices and remaining < len(indices):
        raise ValueError("path beam too narrow for program coverage")
    base = min(minimum, remaining // max(1, len(indices))) if indices else 0
    slots = {i: base for i in indices}
    extra = remaining - base * len(indices)
    weights = {i: math.sqrt(max(programs[i][1], 0.0) + 0.01) for i in indices}
    order = sorted(indices, key=lambda i: (-weights[i], i))
    if order:
        for j in range(extra):
            slots[order[j % len(order)]] += 1
    if rescue:
        slots[-1] = rescue
    assert sum(slots.values()) <= width
    return slots, terminal


def rescue_probability(programs, terminal) -> float:
    return max(0.01, sum(programs[i][1] for i in terminal) * 0.1)


@torch.inference_mode()
def search(policy: FullPolicy, gt: GraphTensors, q, cached, qctx: QuestionContext, *,
           path_beam: int, local_k: int, rescue_slots: int) -> SearchTrace:
    started = time.perf_counter()
    device = gt.device
    qvec = policy.question(qctx.words, device)
    zero = torch.zeros(HIDDEN, device=device)
    programs = [(program_from_dict(b["program"]), float(b["probability"])) for b in cached["beams"]]
    slots, terminal = allocate_slots(programs, path_beam, rescue_slots)
    active = {}
    for index, count in slots.items():
        probability = rescue_probability(programs, terminal) if index == -1 else programs[index][1]
        score = initial_score(probability, len(q.topics))
        initial = [SearchState(index, (topic,), (), (), (), score, (score,), (zero,), zero, zero, policy.n_actions)
                   for topic in q.topics]
        active[index] = global_prune(initial, count, duplicate_limit=5)
    pre, by_depth = [], []
    edges_scored = 0
    scored_nodes = set()
    for depth in range(MAX_HOPS):
        expanded_all = []
        next_active = {}
        for index, states in active.items():
            # RETURN_TOPIC/STOP là kết thúc; nhánh rescue có ngân sách riêng và dừng sau hai cạnh.
            if index == -1 and depth >= 2:
                next_active[index] = states
                continue
            program, probability = (programs[index] if index >= 0
                                    else (Program(()), rescue_probability(programs, terminal)))
            hop = target_hop(program, depth, policy.n_actions) if index >= 0 else Hop(policy.n_actions, None)
            expanded = []
            for state in states:
                current = state.nodes[-1]
                h, c = policy.step(qvec, current, state.previous_action, hop, state.hidden, state.cell)
                scores = policy.edge_scores(gt, qctx.overlap, h, current, hop, probability)
                n = len(scores.logits)
                edges_scored += n
                scored_nodes.add(current)
                values, choices = torch.topk(F.log_softmax(scores.logits, dim=0), min(local_k, n))
                chosen = choices.tolist()
                dst = scores.static.dst[choices].tolist()
                act = scores.static.act[choices].tolist()
                backoff = scores.fallbacks[choices].tolist()
                match = scores.action_match[choices].tolist()
                for value, j, d, a, b, m in zip(values.tolist(), chosen, dst, act, backoff, match):
                    score = state.score + value
                    expanded.append(SearchState(
                        index, state.nodes + (d,), state.edges + (scores.edge_id(j),), state.backoffs + (b,),
                        state.action_matches + (bool(m),), score, state.node_scores + (score,),
                        state.node_hiddens + (h,), h, c, a))
            expanded_all.extend(expanded)
            next_active[index] = global_prune(expanded, slots[index],
                                              duplicate_limit=1 if depth == MAX_HOPS - 1 else 2)
        if depth < MAX_HOPS - 1:
            pre.append(expanded_all)
        active = next_active
        by_depth.append([s for states in active.values() for s in states])
    final = [s for states in active.values() for s in states]
    return SearchTrace(programs, slots, terminal, sum(programs[i][1] for i in terminal), pre, final,
                       by_depth, edges_scored, sum(gt.degree(node) for node in scored_nodes),
                       (time.perf_counter() - started) * 1000)


def _routes(trace: SearchTrace, q, device):
    zero = torch.zeros(HIDDEN, device=device)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    routes = defaultdict(dict)

    def add(state, depth, final):
        node = state.nodes[depth]
        key = (state.program_index, state.edges[:depth], state.nodes[:depth + 1])
        route = (float(state.node_scores[depth]), state.program_index, tuple(state.nodes[:depth + 1]),
                 tuple(state.edges[:depth]), tuple(state.backoffs[:depth]),
                 tuple(state.action_matches[:depth]), final)
        previous = routes[node].get(key)
        if previous is None or previous[0] < route[0]:
            routes[node][key] = route
        old = evidence.get(node)
        if old is None or route[0] > old[1]:
            evidence[node] = (state.node_hiddens[depth], route[0], 1 if old is None else old[2] + 1, depth)

    for state in trace.final:
        for depth in range(len(state.nodes)):
            add(state, depth, True)
    final_nodes = set(evidence)
    for expanded in trace.pre:
        for state in expanded:
            for depth in range(len(state.nodes)):
                add(state, depth, False)
    # Mọi topic luôn là ứng viên 0 hop, kể cả khi chương trình của nó không có trong beam Flan.
    for topic in q.topics:
        routes[topic].setdefault((-2, (), (topic,)), (0.0, -2, (topic,), (), (), (), True))
    return evidence, routes, final_nodes


def cap_candidates(evidence, routes, topics, maximum: int = 1200):
    if len(evidence) <= maximum:
        return evidence, routes, len(evidence)
    if maximum < len(topics):
        raise ValueError("pool cap smaller than topic count")
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
            current = pending.pop()
            best = max(routes[current].values(), key=lambda row: row[0])
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
    leftover = sorted(((score, node) for values in by_depth.values() for score, node in values if node not in keep),
                      key=lambda item: (-item[0], item[1]))
    for _, node in leftover:
        consider(node)
        if len(keep) == maximum:
            break
    assert len(keep) <= maximum
    return {node: evidence[node] for node in keep}, {node: routes[node] for node in keep}, len(evidence)


@torch.inference_mode()
def pool_candidates(policy, gt, q, cached, qctx, *, path_beam, local_k, rescue_slots, maximum):
    trace = search(policy, gt, q, cached, qctx, path_beam=path_beam, local_k=local_k, rescue_slots=rescue_slots)
    evidence, routes, final_nodes = _routes(trace, q, gt.device)
    uncapped_nodes = set(evidence)
    evidence, routes, _ = cap_candidates(evidence, routes, q.topics, maximum)
    return sorted(evidence), evidence, routes, final_nodes, uncapped_nodes, trace


def coverage_diagnostics(graph, q, nodes, trace: SearchTrace) -> dict:
    """Đọc nhãn chỉ để chẩn đoán; không đổi ứng viên hay huấn luyện."""
    gold = set(q.answers)
    depth_nodes = [{state.nodes[-1] for state in states} for states in trace.by_depth]
    program_nodes = defaultdict(set)
    for state in trace.final:
        program_nodes[state.program_index].update(state.nodes)
    degree = max(int(graph.indptr[t + 1] - graph.indptr[t]) for t in q.topics)
    bucket = "<=116" if degree <= 116 else ("117-498" if degree <= 498 else ">498")
    return {"gold_total": len(gold), "gold_in_candidates": len(gold & set(nodes)),
            "oracle_recall_at_20": min(20, len(gold & set(nodes))) / len(gold),
            "gold_by_depth": [len(gold & values) for values in depth_nodes],
            "candidate_by_depth": [len(values) for values in depth_nodes],
            "gold_by_program": {str(i): len(gold & values) for i, values in program_nodes.items()},
            "program_slots": {str(i): count for i, count in trace.slots.items()},
            "terminal_programs": len(trace.terminal_indices),
            "terminal_probability_mass": trace.terminal_mass,
            "topic_degree": degree, "topic_degree_bucket": bucket}
