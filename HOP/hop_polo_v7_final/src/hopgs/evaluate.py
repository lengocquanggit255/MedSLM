"""Answer aggregation and query-level metrics with explicit traversal cost."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from .data import ADIntGraph, Question
from .model import ProgramCodec, generate_beams


def answer_distance_group(graph: ADIntGraph, question: Question) -> tuple[str, dict[int, int | None]]:
    """Use directed KG edges and each gold answer's shortest distance, up to two."""
    distances: dict[int, int | None] = {answer: (0 if answer in question.topics else None)
                                        for answer in question.answers}
    def outgoing(node):
        if hasattr(graph, "out_edges"):
            return graph.out_edges(node)
        return np.flatnonzero(graph.src == node)
    first = {int(graph.dst[edge]) for topic in question.topics for edge in outgoing(topic)}
    for answer in question.answers:
        if distances[answer] is None:
            if answer in first:
                distances[answer] = 1
            elif hasattr(graph, "in_edges"):
                distances[answer] = (2 if any(int(graph.src[edge]) in first
                                             for edge in graph.in_edges(answer)) else None)
            else:
                distances[answer] = (2 if any(int(graph.src[edge]) in first
                                             for edge in np.flatnonzero(graph.dst == answer)) else None)
    reachable = [distance for distance in distances.values() if distance is not None]
    if not reachable:
        group = "none_within_2_hops"
    elif len(reachable) < len(distances):
        group = "partial_within_2_hops"
    elif len(set(reachable)) > 1:
        group = "multiple_hop_lengths"
    else:
        group = f"only_{reachable[0]}_hop"
    return group, distances


def evaluate_queries(model, tokenizer, codec: ProgramCodec, graph: ADIntGraph,
                     questions: list[Question], *, beam: int, top_k: int,
                     query_edge_budget: int, device: torch.device,
                     topic_wise: bool = False) -> tuple[dict, list[dict]]:
    records = []
    g_uses: Counter[tuple[int, int]] = Counter()
    for question in questions:
        started = time.perf_counter()
        topic_costs = []
        exhausted_before_last_topic = False
        if topic_wise:
            codec.executor.begin_query(query_edge_budget)
            tagged_beams = []
            for index, topic in enumerate(question.topics):
                before = codec.executor.query_edges_examined
                generated = generate_beams(
                    model, tokenizer, codec, graph,
                    Question(question.qid, question.question, (topic,), question.answers, question.split),
                    beam, device, query_edge_budget=query_edge_budget, reset_query=False,
                )
                used = codec.executor.query_edges_examined - before
                topic_costs.append({"original_topic_index": index, "topic_node_id": int(graph.ids[topic]),
                                    "edges_examined": used})
                tagged_beams.extend((index, topic, item) for item in generated)
                exhausted_before_last_topic |= codec.executor.query_remaining == 0 and index < len(question.topics) - 1
        else:
            tagged_beams = [(None, None, item) for item in generate_beams(
                model, tokenizer, codec, graph, question, beam, device,
                query_edge_budget=query_edge_budget,
            )]
        beams = [item for _, _, item in tagged_beams]
        scores: dict[int, float] = defaultdict(float)
        evidence: dict[int, list[dict]] = defaultdict(list)
        accepted = []
        for topic_index, topic, item in tagged_beams:
            if not item.executable or not item.candidates:
                continue
            accepted.append(item)
            share = item.probability / len(item.candidates)
            for node in item.candidates:
                scores[node] += share
                path = {"beam_probability": item.probability,
                        "program": codec.tokens_for(item.program),
                        "triples": [graph.edge_triple(edge) for edge in item.evidence[node]]}
                if topic is not None:
                    path["topic_node_id"] = int(graph.ids[topic])
                    path["original_topic_index"] = topic_index
                evidence[node].append(path)
            if codec.executor.mode == "full":
                for hop in item.program.hops:
                    g_uses[(hop.action, hop.g)] += 1
        ranked = sorted(scores, key=lambda node: (-scores[node], int(graph.ids[node])))
        inference_latency_ms = (time.perf_counter() - started) * 1000
        answer_set = set(question.answers)
        top = ranked[:top_k]
        overlap = len(answer_set.intersection(top))
        f2 = 5 * overlap / (4 * len(answer_set) + len(set(top))) if top else 0.0
        group, answer_distances = answer_distance_group(graph, question)
        hit_rank = next((i + 1 for i, node in enumerate(ranked) if node in answer_set), None)
        evidenced_answers = sum(
            any(path["triples"] for path in evidence[node]) for node in answer_set
        )
        records.append({
            "qid": question.qid, "split": question.split,
            "answers": [int(graph.ids[node]) for node in question.answers],
            "topics": [int(graph.ids[node]) for node in question.topics],
            "topic_wise": topic_wise,
            "topic_costs": topic_costs,
            "exhausted_before_last_topic": exhausted_before_last_topic,
            "distance_group": group,
            "answer_shortest_hops": {str(int(graph.ids[node])): distance
                                     for node, distance in answer_distances.items()},
            "top_k": [{"node_id": int(graph.ids[node]), "name": graph.names[node],
                       "score": scores[node], "evidence_paths": sorted(evidence[node],
                           key=lambda item: -item["beam_probability"])} for node in top],
            "candidate_count": len(ranked),
            "candidate_recall_at_k": len(answer_set.intersection(top)) / len(answer_set),
            "recall_at_20": len(answer_set.intersection(top)) / len(answer_set),
            "f2_at_20": f2,
            "candidate_answer_coverage": len(answer_set.intersection(ranked)) / len(answer_set),
            "answer_evidence_coverage": evidenced_answers / len(answer_set),
            "hits_at_1": float(bool(ranked) and ranked[0] in answer_set),
            "mrr": 1 / hit_rank if hit_rank else 0.0,
            "executable_program_rate": sum(item.executable for item in beams) / max(1, len(beams)),
            "has_executable_program": float(any(item.executable for item in beams)),
            "beams_total": len(beams),
            "beams_executable": sum(item.executable for item in beams),
            "accepted_programs": len(accepted),
            "edges_examined": codec.executor.query_edges_examined,
            "decoder_edges_examined": codec.executor.decoder_edges_examined,
            "final_execution_edges_examined": (codec.executor.query_edges_examined
                                                - codec.executor.decoder_edges_examined),
            "latency_ms": inference_latency_ms,
            "status": ("evidence_found" if any(item.program.hops for item in accepted)
                       else "topic_only" if ranked else "no_executable_program"),
        })
    metrics = {"queries": len(records), "beam": beam, "top_k": top_k,
               "edge_budget": query_edge_budget, "mode": codec.executor.mode,
               "topic_wise": topic_wise,
               "edge_count_scope": "query_postings_including_decoder_mask",
               "max_hops": codec.max_hops,
               "prefix_s_index": codec.executor.prefix_s_index is not None}
    if codec.executor.prefix_s_index is not None:
        metrics["prefix_s_index_build_seconds"] = codec.executor.prefix_s_index.build_seconds
        metrics["prefix_s_index_bytes"] = codec.executor.prefix_s_index.bytes
    for name in ("candidate_recall_at_k", "recall_at_20", "f2_at_20", "candidate_answer_coverage",
                 "answer_evidence_coverage", "hits_at_1", "mrr", "has_executable_program",
                 "candidate_count", "executable_program_rate", "edges_examined", "decoder_edges_examined",
                 "final_execution_edges_examined", "latency_ms"):
        values = [float(row[name]) for row in records]
        metrics[f"mean_{name}"] = float(np.mean(values)) if values else 0.0
        if name in ("edges_examined", "latency_ms"):
            metrics[f"p95_{name}"] = float(np.percentile(values, 95)) if values else 0.0
    metrics["evidence_found_rate"] = sum(row["status"] == "evidence_found" for row in records) / max(1, len(records))
    metrics["beam_executable_rate"] = sum(row["beams_executable"] for row in records) / max(
        1, sum(row["beams_total"] for row in records))
    metrics["exhausted_before_last_topic_count"] = sum(row["exhausted_before_last_topic"] for row in records)
    metrics["distance_groups"] = {}
    for group in ("only_0_hop", "only_1_hop", "only_2_hop", "multiple_hop_lengths",
                  "partial_within_2_hops", "none_within_2_hops"):
        subset = [row for row in records if row["distance_group"] == group]
        metrics["distance_groups"][group] = {
            "questions": len(subset),
            **{f"mean_{key}": float(np.mean([row[key] for row in subset])) if subset else 0.0
               for key in ("hits_at_1", "recall_at_20", "f2_at_20", "edges_examined")},
        }
    if codec.executor.mode == "full":
        counts_by_schema: dict[int, list[int]] = defaultdict(list)
        for (schema, code), count in g_uses.items():
            counts_by_schema[schema].append(count)
        trainable = {schema: counts for schema, counts in counts_by_schema.items()
                     if not codec.executor.codes.structural_audit.get("schema_usage", {}).get(str(schema), {}).get("fallback", True)
                     and codec.executor.codes.structural_audit["schema_usage"][str(schema)]["active_codes"] > 1}
        metrics["structural_code_use"] = {
            "active_schema_code_pairs": len(g_uses),
            "mean_dominant_code_fraction": float(np.mean([max(v) / sum(v) for v in counts_by_schema.values()])) if counts_by_schema else 0.0,
            "mean_code_entropy": float(np.mean([-sum((x / sum(v)) * math.log(x / sum(v)) for x in v) for v in counts_by_schema.values()])) if counts_by_schema else 0.0,
            "trainable_schemas_used": len(trainable),
            "trainable_schemas_using_multiple_codes": sum(len(v) > 1 for v in trainable.values()),
            "mean_dominant_code_fraction_trainable": float(np.mean([max(v) / sum(v) for v in trainable.values()])) if trainable else 0.0,
        }
    return metrics, records
