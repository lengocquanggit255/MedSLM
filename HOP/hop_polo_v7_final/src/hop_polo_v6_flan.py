"""Train-only per-answer executable QA labels for the independent v6 run."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from hop_polo_aligned_data import MAX_HOPS, answer_paths, expansion_cost
from hopgs.data import Question
from hopgs.executor import MiningResult, evidence_programs
from hopgs.model import input_text



def initialize_program_tokens(model, tokenizer, old_vocab_size, *, seed=42):
    """Program token anchor initialization applied to the original public Flan model."""
    torch.manual_seed(seed)
    anchor = tokenizer.encode("the", add_special_tokens=False)[0]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        incoming = model.get_input_embeddings().weight
        outgoing = model.get_output_embeddings().weight
        source_in = incoming[anchor].clone()
        source_out = outgoing[anchor].clone()
        incoming[old_vocab_size:] = (source_in + 0.01 *
                                      torch.randn_like(incoming[old_vocab_size:]))
        if outgoing.data_ptr() != incoming.data_ptr():
            outgoing[old_vocab_size:] = (source_out + 0.01 *
                                          torch.randn_like(outgoing[old_vocab_size:]))
    return {"anchor_token": "the", "anchor_id": anchor,
            "old_vocab_size": old_vocab_size,
            "program_token_count": len(tokenizer) - old_vocab_size,
            "input_output_tied": outgoing.data_ptr() == incoming.data_ptr()}


def train_batch(model, tokenizer, examples, optimizer, device):
    texts, labels_list = zip(*examples)
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True,
                        truncation=True, max_length=256,
                        return_token_type_ids=False).to(device)
    labels = torch.full((len(labels_list), max(map(len, labels_list))), -100,
                        dtype=torch.long, device=device)
    for i, ids in enumerate(labels_list):
        labels[i, :len(ids)] = torch.as_tensor(ids, device=device)
    loss = model(**encoded, labels=labels).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def program_distribution(cached, questions, *, train_zero_rate,
                         zero_factor=5.0, zero_floor=0.10,
                         nonzero_beam_floor=0.50):
    """Audit beam ranks and probability mass without changing predictions."""
    by_rank = [Counter() for _ in range(5)]
    mass = {str(depth): [] for depth in range(MAX_HOPS + 1)}
    top_zero = 0
    top_zero_gold = 0
    top_zero_wrong = 0
    beam_nonzero = 0
    for q, row in zip(questions, cached):
        beams = row["beams"]
        values = Counter()
        for rank, beam in enumerate(beams):
            hops = len(beam["program"]["hops"])
            by_rank[rank][str(hops)] += 1
            values[str(hops)] += float(beam["probability"])
        for depth in range(MAX_HOPS + 1):
            mass[str(depth)].append(float(values[str(depth)]))
        if beams and not beams[0]["program"]["hops"]:
            top_zero += 1
            gold = bool(set(q.topics) & set(q.answers))
            top_zero_gold += int(gold)
            top_zero_wrong += int(not gold)
        beam_nonzero += int(any(beam["program"]["hops"] for beam in beams))
    n = max(1, len(cached))
    # The ceiling is fixed from train answers, with a 10% statistical floor.
    top_zero_ceiling = max(zero_floor, min(0.25, zero_factor * train_zero_rate),
                           2 / n)
    valid = (top_zero / n <= top_zero_ceiling and
             beam_nonzero / n >= nonzero_beam_floor)
    return {"queries": len(cached), "by_rank": [dict(item) for item in by_rank],
            "mean_probability_mass": {k: float(np.mean(v)) for k, v in mass.items()},
            "top_zero_queries": top_zero, "top_zero_gold": top_zero_gold,
            "top_zero_wrong": top_zero_wrong,
            "top_zero_rate": top_zero / n,
            "beam_has_nonzero_rate": beam_nonzero / n,
            "train_zero_answer_rate": train_zero_rate,
            "top_zero_ceiling": top_zero_ceiling,
            "nonzero_beam_floor": nonzero_beam_floor,
            "collapse_guard_valid": valid}


def extra_three_hop_paths(graph, q, answer, *, per_depth=32,
                          scan_limit=2_000_000):
    """Search depth three when a shorter route exists but fails cap 200."""
    incoming = {}
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
                        paths = set(sorted(paths, key=lambda item:
                            (expansion_cost(graph, item[1]), item))[:per_depth])
                if scanned >= scan_limit:
                    censored = True
                    break
            if censored:
                break
        if censored:
            break
    ranked = sorted(paths, key=lambda item:
                    (expansion_cost(graph, item[1]), item))
    return ranked[:per_depth], scanned, censored


def executable_qa_labels(graph, codes, executor, codec, train, *,
                         cap=200, programs_per_answer=3, paths_per_depth=32):
    assert executor.edge_budget == cap
    plans = []
    audit = Counter()
    shortest_hops = Counter()
    chosen_hops = Counter()
    per_answer = []
    for index, q in enumerate(train, 1):
        question_plans = []
        audit["questions"] += 1
        for answer in q.answers:
            one_answer = Question(q.qid, q.question, q.topics,
                                  (answer,), q.split)
            buckets, shortest, scanned, censored = answer_paths(
                graph, one_answer, answer, per_depth=paths_per_depth)
            audit["answers"] += 1
            if shortest is not None:
                shortest_hops[shortest] += 1
            accepted = []
            chosen_depth = None
            paths_considered = 0
            extra_scanned = 0
            for depth in range(shortest if shortest is not None else 0, 4):
                paths = buckets[depth]
                if depth == 3 and not paths and shortest is not None:
                    paths, extra_scanned, extra_censored = extra_three_hop_paths(
                        graph, one_answer, answer, per_depth=paths_per_depth)
                    censored = censored or extra_censored
                if not paths:
                    continue
                paths_considered += len(paths)
                mined = MiningResult([path for _pos, path in paths],
                                     depth, scanned + extra_scanned, censored)
                candidates = evidence_programs(
                    graph, codes, executor, one_answer, mined,
                    max_programs=max(16, programs_per_answer * 4),
                    intermediate_prefixes="all", rank_by_cost=True)
                accepted = [item for item in candidates
                            if item.answer == answer and
                            item.edges_examined <= cap and
                            answer in executor.execute(
                                one_answer.topics, item.program).candidates]
                if accepted:
                    accepted.sort(key=lambda item: (
                        len(item.program.hops), item.edges_examined,
                        item.candidate_count, codec.tokens_for(item.program)))
                    chosen_depth = depth
                    break
            chosen = accepted[:programs_per_answer]
            question_plans.append([codec.ids_for(item.program)
                                   for item in chosen])
            if chosen:
                audit["answers_with_labels"] += 1
                audit["programs"] += len(chosen)
                chosen_hops[chosen_depth] += len(chosen)
                audit["fallback_to_longer_depth"] += int(
                    shortest is not None and chosen_depth > shortest)
            else:
                audit["answers_without_feasible_label"] += 1
            audit["three_hop_search_censored"] += int(censored)
            per_answer.append({
                "qid": q.qid, "answer_global": int(graph.ids[answer]),
                "shortest_hops": shortest, "chosen_hops": chosen_depth,
                "feasible_programs": len(chosen),
                "program_costs": [item.edges_examined for item in chosen],
                "program_candidate_counts": [item.candidate_count for item in chosen],
                "program_tokens": [codec.tokens_for(item.program)
                                   for item in chosen],
                "program_token_ids": [codec.ids_for(item.program)
                                       for item in chosen],
                "paths_considered": paths_considered,
                "three_hop_search_censored": bool(censored)})
        plans.append(question_plans)
        audit["questions_with_labels"] += int(any(question_plans))
        if index % 1000 == 0:
            print(f"v6 executable QA labels {index}/{len(train)}", flush=True)
    summary = {**audit,
               "shortest_hops": {str(i): shortest_hops[i] for i in range(4)},
               "label_hops": {str(i): chosen_hops[i] for i in range(4)},
               "label_edge_cap": cap, "max_hops": 3,
               "programs_per_answer": programs_per_answer,
               "sampling": "all feasible answers per labeled QA, equal QA weight"}
    return plans, summary, per_answer


def train_qa_batch(model, tokenizer, graph, train, plans, indices,
                   epoch, optimizer, device):
    """Every labeled QA has weight one; its labeled answers split that weight."""
    texts, labels_list, weights = [], [], []
    for index in indices:
        index = int(index)
        available = [(answer_index, programs)
                     for answer_index, programs in enumerate(plans[index])
                     if programs]
        if not available:
            raise AssertionError("unlabeled QA entered the QA SFT epoch")
        for answer_index, programs in available:
            texts.append(input_text(graph, train[index]))
            labels_list.append(programs[(epoch - 1 + index + answer_index)
                                        % len(programs)])
            weights.append(1.0 / len(available))
    encoded = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=256,
                        return_token_type_ids=False).to(device)
    labels = torch.full((len(labels_list), max(map(len, labels_list))),
                        -100, dtype=torch.long, device=device)
    for i, ids in enumerate(labels_list):
        labels[i, :len(ids)] = torch.as_tensor(ids, device=device)
    logits = model(**encoded, labels=labels).logits
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 labels.reshape(-1), ignore_index=-100,
                                 reduction="none").reshape(labels.shape)
    per_answer = token_loss.sum(1) / (labels != -100).sum(1).clamp_min(1)
    weight = torch.tensor(weights, device=device, dtype=per_answer.dtype)
    loss = (per_answer * weight).sum() / len(indices)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), len(labels_list)
