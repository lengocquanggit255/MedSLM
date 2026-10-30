"""Audit chỉ đọc nhãn (gắn sau tìm kiếm cứng không nhãn) và đếm lại metric độc lập."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .program import program_from_dict


def question_hard_audit(graph, q, trace, pool_nodes, uncapped_nodes) -> dict:
    """Đáp án 1/2 hop có lọt pool không; nếu trượt thì trượt ở bước nào (top-local_k, beam, trần pool).

    `answer_<độ sâu>_cap_drop`: đáp án có trong pool trước khi cắt nhưng bị trần pool loại, theo từng độ sâu.
    """
    pre1 = {s.nodes[-1] for s in trace.pre[0]}
    alive1 = {s.nodes[-1] for s in trace.by_depth[0]}
    pre2 = {s.nodes[-1] for s in trace.pre[1]}
    pool = set(pool_nodes)
    dropped = set(uncapped_nodes) - pool
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
    first = {int(graph.dst[e]) for topic in q.topics for e in graph.out_edges(topic)}
    for answer in q.answers:
        if answer in q.topics:
            counts["answer_0hop"] += 1
            counts["answer_0hop_pool"] += int(answer in pool)
            continue
        if answer in first:
            counts["answer_1hop"] += 1
            counts["answer_1hop_pool"] += int(answer in pool)
            counts["answer_1hop_localk"] += int(answer in pre1)
            counts["answer_1hop_cap_drop"] += int(answer in dropped)
            continue
        incoming = {int(graph.src[int(graph.reverse_order[j])])
                    for j in range(int(graph.reverse_indptr[answer]), int(graph.reverse_indptr[answer + 1]))}
        gold_mid = first & incoming
        if not gold_mid:
            counts["answer_3plus_hop"] += 1
            counts["answer_3plus_hop_pool"] += int(answer in pool)
            counts["answer_3plus_hop_cap_drop"] += int(answer in dropped)
            continue
        counts["answer_2hop"] += 1
        counts["answer_2hop_pool"] += int(answer in pool)
        counts["answer_2hop_cap_drop"] += int(answer in dropped)
        counts["answer_2hop_mid_survived"] += int(bool(gold_mid & alive1))
        counts["answer_2hop_edge_localk"] += int(answer in pre2 and bool(gold_mid & alive1))
        if answer not in pool:
            if not gold_mid & pre1:
                counts["miss_first_localk"] += 1
            elif not gold_mid & alive1:
                counts["miss_first_beam"] += 1
            elif answer not in pre2:
                counts["miss_second_localk"] += 1
            else:
                counts["miss_pool_cap"] += 1
    return dict(counts)


def aggregate(rows) -> dict:
    total = Counter()
    for row in rows:
        total.update(row)
    return dict(total)


def recount(rows) -> dict:
    """Đếm lại từ ID nút đã lưu, không dùng trường metric của evaluator."""
    values = {"hits_at_1": [], "recall_at_20": [], "pool_recall": [], "recall_set": [],
              "precision_set": [], "f2_set": []}
    for row in rows:
        gold = set(row["answers"])
        top = row["ranked_top20"]
        selected = set(row["selected_nodes"])
        pool = set(row["pool_ranked"])
        if not gold or len(top) != len(set(top)) or (not top and not row.get("no_topic")):
            raise AssertionError(f"invalid prediction: {row['qid']}")
        hit = len(gold & selected)
        values["hits_at_1"].append(float(bool(top) and top[0] in gold))
        values["recall_at_20"].append(len(gold & set(top)) / len(gold))
        values["pool_recall"].append(len(gold & pool) / len(gold))
        values["recall_set"].append(hit / len(gold))
        values["precision_set"].append(hit / len(selected) if selected else 0.0)
        values["f2_set"].append(5 * hit / (4 * len(gold) + len(selected)))
    return {f"mean_{name}": float(np.mean(series)) for name, series in values.items()}


def check_recount(metrics: dict, predictions: list[dict]) -> dict:
    independent = recount(predictions)
    for name, value in independent.items():
        if abs(metrics[name] - value) > 1e-9:
            raise AssertionError(f"metric recount differs: {name}")
    return independent


def finite_predictions(metrics, predictions, expected) -> bool:
    return (len(predictions) == expected and
            all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))) and
            all("hits_at_1" in row and np.isfinite(row["hits_at_1"]) for row in predictions))


def policy_program_mismatch(graph, codebook, questions, teacher, cache) -> dict:
    """Đường vàng train có khớp action/S của beam Flan nào không (chỉ chẩn đoán, không vào CE)."""
    mismatch = Counter()
    for q, paths_by_answer, cached in zip(questions, teacher, cache):
        for _pos, path in paths_by_answer:
            mismatch["train_answer_paths"] += 1
            if not path:
                mismatch["topic_is_answer_paths"] += 1
                continue
            action_any = compatible_s_any = full_s_any = False
            for beam in cached["beams"]:
                program = program_from_dict(beam["program"])
                if len(program.hops) < len(path):
                    continue
                action = all(int(graph.schema[edge]) == program.hops[depth].action for depth, edge in enumerate(path))
                matching_s = action and all(
                    hop.s is None or tuple(int(x) for x in codebook.node_codes[int(graph.dst[edge]), :len(hop.s)]) == hop.s
                    for hop, edge in zip(program.hops, path))
                full_s = action and all(hop.s == tuple(int(x) for x in codebook.node_codes[int(graph.dst[edge])])
                                        for hop, edge in zip(program.hops, path))
                action_any |= action
                compatible_s_any |= matching_s
                full_s_any |= full_s
            mismatch["any_beam_path_prefix_action_match"] += int(action_any)
            mismatch["any_beam_path_prefix_action_s_compatible"] += int(compatible_s_any)
            mismatch["any_beam_path_prefix_action_full_s"] += int(full_s_any)
            mismatch["any_beam_s_backoff_only"] += int(compatible_s_any and not full_s_any)
            mismatch["no_beam_action_match"] += int(not action_any)
    return dict(mismatch)
