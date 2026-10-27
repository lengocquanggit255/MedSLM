"""Train-only, answer-balanced paths for the HOP/PoLo follow-up.

All paths are directed KG paths.  A question contributes one unit of training
weight regardless of its number of answers, programs, or path alternatives.
"""
from __future__ import annotations

import math
import os
from collections import Counter

import numpy as np

MAX_HOPS = int(os.environ.get("HOP_POLO_MAX_HOPS", "3"))


def expansion_cost(graph, path):
    return sum(math.log1p(int(graph.indptr[int(graph.src[e]) + 1]
                              - graph.indptr[int(graph.src[e])])) for e in path)


def _best(graph, items, limit):
    unique = {(int(pos), tuple(int(e) for e in path)) for pos, path in items}
    return sorted(unique, key=lambda item: (expansion_cost(graph, item[1]), item))[:limit]


def answer_paths(graph, question, answer, *, per_depth=24, three_hop_scan_limit=2_000_000):
    """Return 0/1/2-hop alternatives, and 3-hop paths only if required.

    Paths of depth two are retained even for a one-hop answer so that a
    predicted two-hop program can have a matching route to that answer.
    The three-hop scan bound applies only to mining, never retrieval.
    """
    answer = int(answer)
    buckets = {depth: [] for depth in range(MAX_HOPS + 1)}
    for pos, topic in enumerate(question.topics):
        if topic == answer:
            buckets[0].append((pos, ()))
    incoming = {}
    for edge in graph.in_edges(answer):
        incoming.setdefault(int(graph.src[edge]), []).append(int(edge))
    for edges in incoming.values():
        edges.sort(key=lambda e: (expansion_cost(graph, (e,)), e))
        del edges[per_depth:]
    first = [(pos, int(edge)) for pos, topic in enumerate(question.topics)
             for edge in graph.out_edges(topic)]
    for pos, a in first:
        mid = int(graph.dst[a])
        if mid == answer:
            buckets[1].append((pos, (a,)))
        for b in incoming.get(mid, ()):
            buckets[2].append((pos, (a, b)))
        if len(buckets[2]) > per_depth * 8:
            buckets[2] = _best(graph, buckets[2], per_depth)
    buckets[0] = _best(graph, buckets[0], per_depth)
    buckets[1] = _best(graph, buckets[1], per_depth)
    buckets[2] = _best(graph, buckets[2], per_depth)
    shortest = next((depth for depth in range(3) if buckets[depth]), None)
    scanned = 0
    censored = False
    if shortest is None:
        for pos, a in first:
            for b in graph.out_edges(int(graph.dst[a])):
                scanned += 1
                for c in incoming.get(int(graph.dst[b]), ()):
                    buckets[3].append((pos, (a, int(b), c)))
                if len(buckets[3]) > per_depth * 8:
                    buckets[3] = _best(graph, buckets[3], per_depth)
                if scanned >= three_hop_scan_limit:
                    censored = True
                    break
            if censored:
                break
        buckets[3] = _best(graph, buckets[3], per_depth)
        if buckets[3]:
            shortest = 3
    if shortest is None and MAX_HOPS > 3:
        # Long ADint routes are mined exactly.  The graph already includes
        # explicit inverse edges, so directed shortest paths match the policy.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        if not hasattr(graph, "_hop_polo_shortest_csr"):
            graph._hop_polo_shortest_csr = csr_matrix(
                (np.ones(len(graph.dst), dtype=np.int32), graph.dst, graph.indptr),
                shape=(len(graph.ids), len(graph.ids)))
        distances, predecessors = shortest_path(
            graph._hop_polo_shortest_csr, directed=True, unweighted=True,
            indices=np.asarray(question.topics, dtype=np.int32),
            return_predecessors=True)
        distances = np.atleast_2d(distances)
        predecessors = np.atleast_2d(predecessors)
        for pos in np.argsort(distances[:, answer], kind="stable"):
            depth = distances[pos, answer]
            if not np.isfinite(depth) or depth > MAX_HOPS:
                break
            nodes = [answer]
            while nodes[-1] != question.topics[pos]:
                previous = int(predecessors[pos, nodes[-1]])
                if previous < 0:
                    raise AssertionError("shortest path predecessor missing")
                nodes.append(previous)
            nodes.reverse()
            path = []
            for source, target in zip(nodes, nodes[1:]):
                candidates = [int(e) for e in graph.out_edges(source)
                              if int(graph.dst[e]) == target]
                path.append(min(candidates, key=lambda e:
                                (expansion_cost(graph, (e,)), e)))
            buckets[len(path)].append((int(pos), tuple(path)))
            shortest = len(path)
            break
    return buckets, shortest, scanned, censored


def path_alignment(graph, codes, program, pos, path):
    """Cost under a decoded Flan program, using the policy's A/S priors."""
    positions = program.return_topics
    if positions is not None and pos not in positions:
        return None
    off_action = 0
    s_backoff = 0
    for depth, edge in enumerate(path):
        if depth >= len(program.hops):
            continue  # after the program ends the policy's action prior is neutral
        hop = program.hops[depth]
        off_action += int(int(graph.schema[edge]) != hop.action)
        target = hop.s or ()
        actual = codes.node_codes[int(graph.dst[edge])]
        matching = 0
        for expected, observed in zip(target, actual):
            if expected != observed:
                break
            matching += 1
        s_backoff += len(target) - matching
    exact = (len(path) == len(program.hops) and off_action == 0 and s_backoff == 0)
    cost = (2.5 * off_action + 1.05 * s_backoff
            + 0.10 * expansion_cost(graph, path)
            + 0.35 * abs(len(path) - len(program.hops)))
    return {"cost": cost, "off_action": off_action, "s_backoff": s_backoff,
            "exact": exact}


def aligned_paths(graph, codes, program, buckets, *, max_paths=5):
    """Keep good exact, off-A, and S-backoff routes with normalized weights."""
    candidates = []
    for depth in buckets:
        for pos, path in buckets[depth]:
            alignment = path_alignment(graph, codes, program, pos, path)
            if alignment is not None:
                candidates.append({"topic_index": pos, "edges": path, **alignment})
    if not candidates:
        return []
    candidates.sort(key=lambda row: (row["cost"], row["topic_index"], row["edges"]))
    selected = []
    def take(row):
        if row is not None and row not in selected and len(selected) < max_paths:
            selected.append(row)
    take(candidates[0])
    take(next((row for row in candidates if row["exact"]), None))
    take(next((row for row in candidates if row["off_action"]), None))
    take(next((row for row in candidates if row["s_backoff"]), None))
    for row in candidates:
        take(row)
    exact = [row for row in selected if row["exact"]]
    relaxed = [row for row in selected if not row["exact"]]
    for group, mass in ((exact, 0.75 if relaxed else 1.0),
                        (relaxed, 0.25 if exact else 1.0)):
        if not group:
            continue
        values = np.exp(-np.asarray([row["cost"] for row in group]) / 2.0)
        values /= values.sum()
        for row, weight in zip(group, values):
            row["weight"] = float(mass * weight)
    assert abs(sum(row["weight"] for row in selected) - 1.0) < 1e-8
    return selected


def label_audit(graph, questions, answers_to_paths):
    counts = Counter()
    for question, answer_groups in zip(questions, answers_to_paths):
        counts["qa_train"] += 1
        counts["answers_train"] += len(question.answers)
        for answer, (_buckets, shortest, scanned, censored) in zip(question.answers, answer_groups):
            counts[f"answer_shortest_{shortest}hop"] += 1
            counts["three_hop_edges_scanned"] += scanned
            counts["three_hop_mining_censored"] += int(censored)
            if shortest is None:
                counts[f"answers_without_path_{MAX_HOPS}hop"] += 1
    return dict(counts)
