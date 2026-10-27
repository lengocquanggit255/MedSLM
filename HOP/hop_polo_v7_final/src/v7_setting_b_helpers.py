"""Training-only helpers for V7 Setting B; no legacy ablation entrypoint."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from hop_polo_aligned_data import answer_paths
from hop_polo_v2_search import pool_candidates
from hop_polo_v6 import select_hits
from v7_surrogate import mine_routes, question_path_loss


def _baseline_teacher(graph, questions):
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


def _mine_train(graph, questions, cap, seed):
    all_routes, audit = [], Counter()
    degree_before, degree_after = Counter(), Counter()
    for q in questions:
        groups = []
        cut_q = False
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
    return all_routes, {**audit, "cap_per_answer": cap,
                        "degree_bucket_before": dict(degree_before),
                        "degree_bucket_after": dict(degree_after),
                        "sampling": "topic and logarithmic degree strata; deterministic reservoir"}


def train_soft_hard_gap(policy, graph, codes, questions, cache, routes,
                        degrees, config, tau):
    limit = min(config["train_gap_audit_qas"], len(questions))
    masses, hard = [], []
    with torch.no_grad():
        for q, cached, grouped in zip(questions[:limit], cache[:limit], routes[:limit]):
            _, mean_mass, count = question_path_loss(
                policy, graph, codes, q, cached, degrees, grouped,
                mode="C", tau=tau, path_beam=config["path_beam"],
                local_k=config["local_top_k"],
                rescue_slots=config["terminal_rescue_slots"])
            if not count:
                continue
            nodes, *_ = pool_candidates(policy, graph, codes, q, cached, degrees,
                                        path_beam=config["path_beam"],
                                        local_k=config["local_top_k"],
                                        rescue_slots=config["terminal_rescue_slots"],
                                        maximum=config["max_pool_nodes"])
            eligible = [a for a, rr in zip(q.answers, grouped)
                        if a not in q.topics and rr and len(rr[0].edges) <= 2]
            if eligible:
                masses.append(float(mean_mass))
                hard.append(sum(a in nodes for a in eligible) / len(eligible))
    if not masses:
        return {"qas": 0, "mean_soft_mass": None, "hard_coverage": None,
                "difference": None}
    return {"qas": len(masses), "mean_soft_mass": float(np.mean(masses)),
            "hard_coverage": float(np.mean(hard)),
            "difference": float(np.mean(hard) - np.mean(masses)),
            "interpretation": "diagnostic only; path mass is not calibrated pool probability"}


def policy_stale(history, epoch, previous):
    if epoch == 0:
        return 0
    best = select_hits(history)
    return 0 if best is not None and best["epoch"] == epoch else previous + 1
