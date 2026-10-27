"""Read-only label diagnostics attached after V6's label-free hard search."""
from __future__ import annotations

from collections import Counter


def question_hard_audit(graph, q, trace, pool_nodes):
    pre1 = {s.nodes[-1] for s in trace.pre[0]}
    alive1 = {s.nodes[-1] for s in trace.by_depth[0]}
    pre2 = {s.nodes[-1] for s in trace.pre[1]}
    pool = set(pool_nodes)
    counts = Counter()
    counts["search_ms"] = trace.search_ms
    counts["pool_nodes"] = len(pool)
    counts["edges_scored"] = trace.edges_scored
    counts["terminal_branches"] = len(trace.terminal_indices)
    counts["rescue_slots"] = trace.slots.get(-1, 0)
    for depth_states in trace.pre:
        for state in depth_states:
            if state.action_matches:
                counts["action_match"] += int(state.action_matches[-1])
                counts["action_total"] += 1
            if state.backoffs:
                counts[f"s_backoff_{state.backoffs[-1]}"] += 1
    for answer in q.answers:
        if answer in q.topics:
            counts["answer_0hop"] += 1
            counts["answer_0hop_pool"] += int(answer in pool)
            continue
        first = {int(graph.dst[e]) for topic in q.topics for e in graph.out_edges(topic)}
        if answer in first:
            counts["answer_1hop"] += 1
            counts["answer_1hop_pool"] += int(answer in pool)
            counts["answer_1hop_local60"] += int(answer in pre1)
            continue
        incoming = {int(graph.src[int(graph.reverse_order[j])])
                    for j in range(int(graph.reverse_indptr[answer]),
                                   int(graph.reverse_indptr[answer + 1]))}
        gold_mid = first & incoming
        if not gold_mid:
            counts["answer_3plus_hop"] += 1
            continue
        counts["answer_2hop"] += 1
        counts["answer_2hop_pool"] += int(answer in pool)
        counts["answer_2hop_mid_survived"] += int(bool(gold_mid & alive1))
        counts["answer_2hop_edge_local60"] += int(answer in pre2 and bool(gold_mid & alive1))
        if answer not in pool:
            if not gold_mid & pre1:
                counts["miss_first_local60"] += 1
            elif not gold_mid & alive1:
                counts["miss_first_beam"] += 1
            elif answer not in pre2:
                counts["miss_second_local60"] += 1
            else:
                counts["miss_pool_cap"] += 1
    return dict(counts)


def aggregate(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    return dict(total)
