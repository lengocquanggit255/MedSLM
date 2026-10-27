"""Terminal-aware, program-budgeted HOP/PoLo search for the fresh v2 run."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from hop_polo_full_policy import SearchState, global_prune, target_hop
from hop_polo_aligned_data import MAX_HOPS
from hop_polo_hybrid import hash_words, program_from_dict, question_overlap
from hopgs.executor import Hop, Program


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


def allocate_slots(programs, width=30, rescue_slots=6, minimum=3):
    """Reserve finite topic rescue and a minimum for each nonterminal beam."""
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


@torch.inference_mode()
def search(policy, graph, codes, q, cached, degrees, *, path_beam=30,
           local_k=60, rescue_slots=6):
    started = time.perf_counter()
    device = next(policy.parameters()).device
    qvec = policy.question(hash_words(q.question), device)
    overlap = question_overlap(graph, q)
    zero = torch.zeros(96, device=device)
    programs = [(program_from_dict(b["program"]), float(b["probability"]))
                for b in cached["beams"]]
    slots, terminal = allocate_slots(programs, path_beam, rescue_slots)
    active = {}
    for index, count in slots.items():
        if index == -1:
            probability = max(0.01, sum(programs[i][1] for i in terminal) * 0.1)
            positions = tuple(range(len(q.topics)))
        else:
            probability = programs[index][1]
            positions = tuple(range(len(q.topics)))
        initial = []
        for pos in positions:
            score = math.log(max(probability, 1e-12)) - math.log(max(1, len(positions)))
            initial.append(SearchState(index, (q.topics[pos],), (), (), (), score,
                                       (score,), (zero,), zero, zero,
                                       policy.n_actions))
        active[index] = global_prune(initial, count, duplicate_limit=5)
    pre, by_depth = [], []
    edges_scored = 0
    unique = set()
    for depth in range(MAX_HOPS):
        expanded_all = []
        next_active = {}
        for index, states in active.items():
            # Explicit RETURN_TOPIC/STOP is terminal. Its optional rescue has
            # its own small budget and stops after two edges.
            if index == -1 and depth >= 2:
                next_active[index] = states
                continue
            program, probability = (programs[index] if index >= 0
                                    else (Program(()), max(0.01, sum(programs[i][1]
                                                           for i in terminal) * 0.1)))
            hop = (target_hop(program, depth, policy.n_actions) if index >= 0
                   else Hop(policy.n_actions, -1, None))
            expanded = []
            for state in states:
                current = state.nodes[-1]
                edges = graph.out_edges(current).tolist() + [-1]
                edges_scored += len(edges)
                unique.update(edges[:-1])
                h, c = policy.step(qvec, current, state.previous_action,
                                   hop, state.hidden, state.cell)
                logits, backoffs, matches, destinations, actions = policy.edge_logits(
                    graph, codes, qvec, h, current, hop, edges,
                    overlap, degrees, probability)
                log_probs = F.log_softmax(logits, dim=0)
                choices = torch.topk(log_probs, min(local_k, len(edges))).indices.tolist()
                for j in choices:
                    score = state.score + float(log_probs[j])
                    expanded.append(SearchState(
                        index, state.nodes + (int(destinations[j]),),
                        state.edges + (int(edges[j]),),
                        state.backoffs + (int(backoffs[j]),),
                        state.action_matches + (bool(matches[j]),), score,
                        state.node_scores + (score,), state.node_hiddens + (h,),
                        h, c, int(actions[j])))
            expanded_all.extend(expanded)
            next_active[index] = global_prune(expanded, slots[index],
                                               duplicate_limit=1 if depth == MAX_HOPS - 1 else 2)
        if depth < MAX_HOPS - 1:
            pre.append(expanded_all)
        active = next_active
        by_depth.append([s for states in active.values() for s in states])
    final = [s for states in active.values() for s in states]
    return SearchTrace(programs, slots, terminal,
                       sum(programs[i][1] for i in terminal), pre, final,
                       by_depth, edges_scored, len(unique),
                       (time.perf_counter() - started) * 1000)


def _routes(trace, q, device):
    zero = torch.zeros(96, device=device)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    routes = defaultdict(dict)

    def add(state, depth, final):
        node = state.nodes[depth]
        key = (state.program_index, state.edges[:depth], state.nodes[:depth + 1])
        route = (float(state.node_scores[depth]), state.program_index,
                 tuple(state.nodes[:depth + 1]), tuple(state.edges[:depth]),
                 tuple(state.backoffs[:depth]),
                 tuple(state.action_matches[:depth]), final)
        previous = routes[node].get(key)
        if previous is None or previous[0] < route[0]:
            routes[node][key] = route
        old = evidence.get(node)
        if old is None or route[0] > old[1]:
            evidence[node] = (state.node_hiddens[depth], route[0],
                              1 if old is None else old[2] + 1, depth)

    for state in trace.final:
        for depth in range(len(state.nodes)):
            add(state, depth, True)
    final_nodes = set(evidence)
    for expanded in trace.pre:
        for state in expanded:
            for depth in range(len(state.nodes)):
                add(state, depth, False)
    # Every topic remains an explicit zero-hop candidate even if its program
    # was absent from the Flan beam. This costs no path-beam slot.
    for topic in q.topics:
        routes[topic].setdefault((-2, (), (topic,)),
                                 (0.0, -2, (topic,), (), (), (), True))
    return evidence, routes, final_nodes


def cap_candidates(evidence, routes, topics, maximum=1200):
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
    quotas = {1: int(remaining * 0.48), 2: int(remaining * 0.47),
              3: int(remaining * 0.05)}
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
    leftover = [(score, node) for values in by_depth.values()
                for score, node in values if node not in keep]
    leftover.sort(key=lambda item: (-item[0], item[1]))
    for _, node in leftover:
        consider(node)
        if len(keep) == maximum:
            break
    assert len(keep) <= maximum
    uncapped = len(evidence)
    return ({node: evidence[node] for node in keep},
            {node: routes[node] for node in keep}, uncapped)


@torch.inference_mode()
def beam_candidates(policy, graph, codes, q, cached, degrees,
                    *, path_beam=30, local_k=60, rescue_slots=6):
    trace = search(policy, graph, codes, q, cached, degrees,
                   path_beam=path_beam, local_k=local_k,
                   rescue_slots=rescue_slots)
    evidence, routes, final_nodes = _routes(trace, q, next(policy.parameters()).device)
    evidence = {node: value for node, value in evidence.items() if node in final_nodes}
    nodes = sorted(evidence)
    return nodes, evidence, trace


@torch.inference_mode()
def pool_candidates(policy, graph, codes, q, cached, degrees,
                    *, path_beam=30, local_k=60, rescue_slots=6,
                    maximum=1200):
    trace = search(policy, graph, codes, q, cached, degrees,
                   path_beam=path_beam, local_k=local_k,
                   rescue_slots=rescue_slots)
    evidence, routes, final_nodes = _routes(trace, q, next(policy.parameters()).device)
    uncapped_nodes = set(evidence)
    evidence, routes, uncapped = cap_candidates(evidence, routes, q.topics, maximum)
    return sorted(evidence), evidence, routes, final_nodes, uncapped_nodes, trace


def coverage_diagnostics(graph, q, nodes, trace):
    """Read labels for diagnostics only; never changes candidates or training."""
    gold = set(q.answers)
    depth_nodes = []
    for states in trace.by_depth:
        depth_nodes.append({state.nodes[-1] for state in states})
    program_nodes = defaultdict(set)
    for state in trace.final:
        program_nodes[state.program_index].update(state.nodes)
    degree = max(int(graph.indptr[t + 1] - graph.indptr[t]) for t in q.topics)
    bucket = "<=116" if degree <= 116 else ("117-498" if degree <= 498 else ">498")
    return {"gold_total": len(gold), "gold_in_candidates": len(gold & set(nodes)),
            "oracle_recall_at_20": min(20, len(gold & set(nodes))) / len(gold),
            "gold_by_depth": [len(gold & values) for values in depth_nodes],
            "candidate_by_depth": [len(values) for values in depth_nodes],
            "gold_by_program": {str(i): len(gold & values)
                                for i, values in program_nodes.items()},
            "program_slots": {str(i): count for i, count in trace.slots.items()},
            "terminal_programs": len(trace.terminal_indices),
            "terminal_probability_mass": trace.terminal_mass,
            "topic_degree": degree, "topic_degree_bucket": bucket}
