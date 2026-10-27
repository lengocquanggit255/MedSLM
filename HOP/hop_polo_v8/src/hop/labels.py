"""Nhãn chỉ từ QA train: đường vàng, chương trình thực thi được, teacher của policy, route Setting B."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .codebook import Codebook
from .data import Graph, Question
from .program import Executor, Hop, Program, ProgramCodec, evidence_programs

MAX_HOPS = 3


def expansion_cost(graph: Graph, path) -> float:
    return sum(math.log1p(int(graph.indptr[int(graph.src[e]) + 1] - graph.indptr[int(graph.src[e])])) for e in path)


def _best(graph: Graph, items, limit: int):
    unique = {(int(pos), tuple(int(e) for e in path)) for pos, path in items}
    return sorted(unique, key=lambda item: (expansion_cost(graph, item[1]), item))[:limit]


def answer_paths(graph: Graph, question: Question, answer: int, *, per_depth: int = 24,
                 three_hop_scan_limit: int = 2_000_000):
    """Đường 0/1/2 hop; đường 3 hop chỉ khi không có đường ngắn hơn. Giới hạn quét chỉ áp cho khai thác nhãn."""
    answer = int(answer)
    buckets = {depth: [] for depth in range(MAX_HOPS + 1)}
    for pos, topic in enumerate(question.topics):
        if topic == answer:
            buckets[0].append((pos, ()))
    incoming: dict[int, list[int]] = {}
    for edge in graph.in_edges(answer):
        incoming.setdefault(int(graph.src[edge]), []).append(int(edge))
    for edges in incoming.values():
        edges.sort(key=lambda e: (expansion_cost(graph, (e,)), e))
        del edges[per_depth:]
    first = [(pos, int(edge)) for pos, topic in enumerate(question.topics) for edge in graph.out_edges(topic)]
    for pos, a in first:
        mid = int(graph.dst[a])
        if mid == answer:
            buckets[1].append((pos, (a,)))
        for b in incoming.get(mid, ()):
            buckets[2].append((pos, (a, b)))
        if len(buckets[2]) > per_depth * 8:
            buckets[2] = _best(graph, buckets[2], per_depth)
    for depth in range(3):
        buckets[depth] = _best(graph, buckets[depth], per_depth)
    shortest = next((depth for depth in range(3) if buckets[depth]), None)
    scanned, censored = 0, False
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
    return buckets, shortest, scanned, censored


def extra_three_hop_paths(graph: Graph, q: Question, answer: int, *, per_depth: int = 32,
                          scan_limit: int = 2_000_000):
    """Tìm thêm đường 3 hop khi đường ngắn hơn có nhưng vượt trần cạnh của nhãn."""
    incoming: dict[int, list[int]] = {}
    for edge in graph.in_edges(answer):
        incoming.setdefault(int(graph.src[edge]), []).append(int(edge))
    paths = set()
    scanned = 0
    censored = False
    for pos, topic in enumerate(q.topics):
        for a in graph.out_edges(topic):
            for b in graph.out_edges(int(graph.dst[a])):
                scanned += 1
                for c in incoming.get(int(graph.dst[b]), ()):
                    paths.add((pos, (int(a), int(b), c)))
                    if len(paths) > per_depth * 8:
                        paths = set(sorted(paths, key=lambda item: (expansion_cost(graph, item[1]), item))[:per_depth])
                if scanned >= scan_limit:
                    censored = True
                    break
            if censored:
                break
        if censored:
            break
    ranked = sorted(paths, key=lambda item: (expansion_cost(graph, item[1]), item))
    return ranked[:per_depth], scanned, censored


def executable_qa_labels(graph: Graph, codebook: Codebook, executor: Executor, codec: ProgramCodec,
                         train: list[Question], *, cap: int, programs_per_answer: int, paths_per_depth: int):
    """Mỗi đáp án train: tối đa `programs_per_answer` chương trình mà thực thi trong trần `cap` cạnh vẫn chứa nó."""
    assert executor.edge_budget == cap
    plans, per_answer = [], []
    audit, shortest_hops, chosen_hops = Counter(), Counter(), Counter()
    for index, q in enumerate(train, 1):
        question_plans = []
        audit["questions"] += 1
        for answer in q.answers:
            one_answer = Question(q.qid, q.question, q.topics, (answer,), q.split)
            buckets, shortest, scanned, censored = answer_paths(graph, one_answer, answer, per_depth=paths_per_depth)
            audit["answers"] += 1
            if shortest is not None:
                shortest_hops[shortest] += 1
            accepted, chosen_depth, paths_considered, extra_scanned = [], None, 0, 0
            for depth in range(shortest if shortest is not None else 0, MAX_HOPS + 1):
                paths = buckets[depth]
                if depth == 3 and not paths and shortest is not None:
                    paths, extra_scanned, extra_censored = extra_three_hop_paths(
                        graph, one_answer, answer, per_depth=paths_per_depth)
                    censored = censored or extra_censored
                if not paths:
                    continue
                paths_considered += len(paths)
                candidates = evidence_programs(graph, codebook, executor, one_answer, [path for _pos, path in paths],
                                               max_programs=max(16, programs_per_answer * 4))
                accepted = [item for item in candidates
                            if item.answer == answer and item.edges_examined <= cap and
                            answer in executor.execute(one_answer.topics, item.program).candidates]
                if accepted:
                    accepted.sort(key=lambda item: (len(item.program.hops), item.edges_examined,
                                                    item.candidate_count, codec.tokens_for(item.program)))
                    chosen_depth = depth
                    break
            chosen = accepted[:programs_per_answer]
            question_plans.append([codec.ids_for(item.program) for item in chosen])
            if chosen:
                audit["answers_with_labels"] += 1
                audit["programs"] += len(chosen)
                chosen_hops[chosen_depth] += len(chosen)
                audit["fallback_to_longer_depth"] += int(shortest is not None and chosen_depth > shortest)
            else:
                audit["answers_without_feasible_label"] += 1
            audit["three_hop_search_censored"] += int(censored)
            per_answer.append({"qid": q.qid, "answer_global": int(graph.ids[answer]),
                               "shortest_hops": shortest, "chosen_hops": chosen_depth,
                               "feasible_programs": len(chosen),
                               "program_costs": [item.edges_examined for item in chosen],
                               "program_candidate_counts": [item.candidate_count for item in chosen],
                               "program_tokens": [codec.tokens_for(item.program) for item in chosen],
                               "paths_considered": paths_considered,
                               "three_hop_search_censored": bool(censored)})
        plans.append(question_plans)
        audit["questions_with_labels"] += int(any(question_plans))
        if index % 1000 == 0:
            print(f"executable QA labels {index}/{len(train)}", flush=True)
    summary = {**audit, "shortest_hops": {str(i): shortest_hops[i] for i in range(4)},
               "label_hops": {str(i): chosen_hops[i] for i in range(4)},
               "label_edge_cap": cap, "max_hops": MAX_HOPS, "programs_per_answer": programs_per_answer,
               "sampling": "all feasible answers per labeled QA, equal QA weight"}
    return plans, summary, per_answer


def path_program(graph: Graph, codebook: Codebook, question: Question, topic_index: int, path) -> Program:
    """Chương trình mục tiêu của teacher: schema + mã S đầy đủ của nút đích từng bước."""
    if not path:
        return Program((), (topic_index,)) if len(question.topics) > 1 else Program(())
    return Program(tuple(Hop(int(graph.schema[e]), tuple(int(x) for x in codebook.node_codes[int(graph.dst[e])]))
                         for e in path))


def teacher_paths(graph: Graph, questions: list[Question]):
    """Đường ngắn nhất rẻ nhất cho mỗi đáp án; dùng chung cho warm-up và Setting B."""
    result, depths = [], Counter()
    for q in questions:
        choices = []
        for answer in q.answers:
            buckets, shortest, _, _ = answer_paths(graph, q, answer)
            if shortest is None:
                raise AssertionError(f"no <=3-hop teacher route: {q.qid}")
            choices.append(buckets[shortest][0])
            depths[str(shortest)] += 1
        result.append(choices)
    return result, dict(depths)


@dataclass(frozen=True)
class Route:
    topic_pos: int
    edges: tuple[int, ...]
    intermediate_degree: int = 0


def mine_routes(graph: Graph, q: Question, answer: int, cap: int, seed: int):
    """Giao 'topic ra' với 'đáp án vào', reservoir không lệch theo (topic, bucket log2 bậc nút giữa).

    Đường trực tiếp được ưu tiên hơn đường 2 hop; số route trước/sau cắt được ghi rõ.
    """
    if cap < 1:
        raise ValueError("route cap must be positive")
    if answer in q.topics:
        return [], {"shortest": 0, "before": 0, "after": 0, "cut": False}
    direct = [Route(pos, (int(e),)) for pos, topic in enumerate(q.topics)
              for e in graph.out_edges(topic) if int(graph.dst[e]) == answer]
    if direct:
        total_direct = len(direct)
        if total_direct > cap:
            key = hashlib.sha256(f"{seed}:{q.qid}:{answer}:direct".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
            direct = [direct[i] for i in sorted(rng.choice(total_direct, cap, replace=False))]
        return direct, {"shortest": 1, "before": total_direct, "after": len(direct), "cut": total_direct > cap,
                        "sampling": "all direct routes, uniform sample if capped"}
    incoming = defaultdict(list)
    for ix in range(int(graph.reverse_indptr[answer]), int(graph.reverse_indptr[answer + 1])):
        e = int(graph.reverse_order[ix])
        incoming[int(graph.src[e])].append(e)
    key = hashlib.sha256(f"{seed}:{q.qid}:{answer}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    groups, seen = defaultdict(list), Counter()
    degree_before = Counter()
    before = 0
    for pos, topic in enumerate(q.topics):
        for e1 in graph.out_edges(topic):
            e1 = int(e1)
            mid = int(graph.dst[e1])
            if mid in (topic, answer):
                continue  # chỉ route đơn
            degree = int(graph.indptr[mid + 1] - graph.indptr[mid])
            bucket = min(12, int(math.log2(max(1, degree))))
            group = (pos, bucket)
            for e2 in incoming.get(mid, ()):
                before += 1
                degree_before[bucket] += 1
                seen[group] += 1
                route = Route(pos, (e1, e2), degree)
                reservoir = groups[group]
                if len(reservoir) < cap:
                    reservoir.append(route)
                else:
                    j = int(rng.integers(seen[group]))
                    if j < cap:
                        reservoir[j] = route
    for routes in groups.values():
        rng.shuffle(routes)
    selected = []
    keys = sorted(groups)
    while keys and len(selected) < cap:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < cap:
                selected.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    degree_after = Counter(min(12, int(math.log2(max(1, r.intermediate_degree)))) for r in selected)
    return selected, {"shortest": 2 if before else None, "before": before, "after": len(selected),
                      "cut": before > len(selected), "degree_bucket_before": dict(degree_before),
                      "degree_bucket_after": dict(degree_after),
                      "sampling": "uniform reservoir within topic and log2 degree, round-robin groups"}


def mine_train_routes(graph: Graph, questions: list[Question], cap: int, seed: int):
    all_routes, audit = [], Counter()
    degree_before, degree_after = Counter(), Counter()
    for q in questions:
        groups, cut_q = [], False
        for answer in q.answers:
            routes, row = mine_routes(graph, q, answer, cap, seed)
            groups.append(routes)
            audit["answers"] += 1
            audit["routes_before"] += row["before"]
            audit["routes_after"] += row["after"]
            audit[f"shortest_{row['shortest']}"] += 1
            audit["answers_cut"] += int(row["cut"])
            cut_q |= row["cut"]
            degree_before.update(row.get("degree_bucket_before", {}))
            degree_after.update(row.get("degree_bucket_after", {}))
        audit["questions_cut"] += int(cut_q)
        all_routes.append(groups)
    return all_routes, {**audit, "cap_per_answer": cap, "degree_bucket_before": dict(degree_before),
                        "degree_bucket_after": dict(degree_after),
                        "sampling": "topic and logarithmic degree strata; deterministic reservoir"}
