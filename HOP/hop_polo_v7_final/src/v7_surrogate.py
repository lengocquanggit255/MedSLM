"""Train-only path-mass surrogate sharing V6's edge policy and transitions.

The mass is not a calibrated probability of inclusion in the capped pool.
Only hard V6 search measures coverage. No validation/test labels enter this module.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from hop_polo_full_policy import target_hop
from hop_polo_hybrid import hash_words, program_from_dict, question_overlap
from hop_polo_v2_search import allocate_slots
from hopgs.executor import Hop


class _CapacityTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, tau):
        n = scores.numel()
        if k <= 0 or k >= n:
            z = torch.full_like(scores, float(k >= n))
            ctx.save_for_backward(z)
            ctx.tau = float(tau)
            return z
        if tau <= 0:
            raise ValueError("tau must be positive")
        # lambda solves sum sigmoid((s-lambda)/tau)=k. Its derivative is
        # included analytically in backward, despite the scalar search here.
        with torch.no_grad():
            lo = scores.min() - 40 * tau
            hi = scores.max() + 40 * tau
            for _ in range(80):
                mid = (lo + hi) / 2
                if torch.sigmoid((scores - mid) / tau).sum() > k:
                    lo = mid
                else:
                    hi = mid
            z = torch.sigmoid((scores - (lo + hi) / 2) / tau)
        ctx.save_for_backward(z)
        ctx.tau = float(tau)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        d = z * (1 - z)
        total = d.sum()
        if total <= 0:
            return torch.zeros_like(grad_output), None, None
        weighted = (grad_output * d).sum() / total
        return d * (grad_output - weighted) / ctx.tau, None, None


def capacity_topk(scores: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Entropy regularized capacity selector; sum(z)=k up to solver error."""
    if scores.ndim != 1:
        raise ValueError("scores must be one dimensional")
    return _CapacityTopK.apply(scores, min(max(int(k), 0), scores.numel()), tau)


@dataclass(frozen=True)
class Route:
    topic_pos: int
    edges: tuple[int, ...]
    intermediate_degree: int = 0


def mine_routes(graph, q, answer: int, cap: int, seed: int):
    """Directed topic-out / answer-in intersection with unbiased reservoirs.

    Each (topic, logarithmic intermediate-degree bucket) receives a reservoir.
    The final round-robin merge keeps topic and degree diversity; counts before
    and after are explicit. Direct paths take precedence over two-hop paths.
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
        return direct, {"shortest": 1, "before": total_direct,
                        "after": len(direct), "cut": total_direct > cap,
                        "sampling": "all direct routes, uniform sample if capped"}
    incoming = defaultdict(list)
    for ix in range(int(graph.reverse_indptr[answer]), int(graph.reverse_indptr[answer + 1])):
        e = int(graph.reverse_order[ix])
        incoming[int(graph.src[e])].append(e)
    digest = hashlib.sha256(f"{seed}:{q.qid}:{answer}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    groups, seen = defaultdict(list), Counter()
    degree_before = Counter()
    before = 0
    for pos, topic in enumerate(q.topics):
        for e1 in graph.out_edges(topic):
            e1 = int(e1)
            mid = int(graph.dst[e1])
            if mid in (topic, answer):
                continue  # simple route only
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
    degree_after = Counter(min(12, int(math.log2(max(1, r.intermediate_degree))))
                           for r in selected)
    return selected, {"shortest": 2 if before else None, "before": before,
                      "after": len(selected), "cut": before > len(selected),
                      "degree_bucket_before": dict(degree_before),
                      "degree_bucket_after": dict(degree_after),
                      "sampling": "uniform reservoir within topic and log2 degree, round-robin groups"}


def _program_branches(cached, q, path_beam, rescue_slots):
    programs = [(program_from_dict(row["program"]), float(row["probability"]))
                for row in cached["beams"]]
    slots, terminal = allocate_slots(programs, path_beam, rescue_slots)
    branches = []
    for index, count in slots.items():
        if count <= 0:
            continue
        probability = (programs[index][1] if index >= 0 else
                       max(0.01, sum(programs[i][1] for i in terminal) * 0.1))
        program = programs[index][0] if index >= 0 else None
        # V6 globally prunes tied initial topic states to the branch capacity.
        positions_list = []
        topic_counts = Counter()
        for pos, topic in enumerate(q.topics):
            if topic_counts[topic] >= 5:
                continue  # V6 initial global_prune duplicate_limit=5
            positions_list.append(pos)
            topic_counts[topic] += 1
            if len(positions_list) == count:
                break
        positions = tuple(positions_list)
        branches.append((index, program, probability, count, positions))
    return branches


def answer_mass(policy, graph, codes, q, cached, degrees, answer, routes,
                *, mode: str, tau: float, path_beam=30, local_k=60,
                rescue_slots=6):
    """Mass for one train answer. B omits selectors; C includes them.

    Gold second-step states are scored even when the hard beam would prune
    their intermediate. This function never mutates inference candidates.
    """
    if mode not in ("B", "C"):
        raise ValueError(mode)
    device = next(policy.parameters()).device
    zero = torch.zeros((), device=device)
    if not routes:
        return zero
    qvec = policy.question(hash_words(q.question), device)
    overlap = question_overlap(graph, q)
    branches = _program_branches(cached, q, path_beam, rescue_slots)
    if not branches:
        return zero
    by_pos = defaultdict(list)
    for route in routes:
        by_pos[route.topic_pos].append(route)
    all_mass = []
    hidden_zero = torch.zeros(96, device=device)
    for index, program, probability, slot, positions in branches:
        if not positions:
            continue
        initial_score = math.log(max(probability, 1e-12)) - math.log(len(q.topics))
        first = {}
        prefix_scores = []
        prefix_keys = []
        for pos in positions:
            current = q.topics[pos]
            hop = (target_hop(program, 0, policy.n_actions) if index >= 0
                   else Hop(policy.n_actions, -1, None))
            h, c = policy.step(qvec, current, policy.n_actions, hop,
                               hidden_zero, hidden_zero)
            edges = graph.out_edges(current).tolist() + [-1]
            logits, _, _, destinations, actions = policy.edge_logits(
                graph, codes, qvec, h, current, hop, edges, overlap,
                degrees, probability)
            logpi = F.log_softmax(logits, dim=0)
            first[pos] = (edges, logpi, destinations, actions, h, c)
            prefix_scores.append(logpi + initial_score)
            prefix_keys.extend((pos, int(e)) for e in edges)
        if mode == "C":
            scores = torch.cat(prefix_scores)
            # Identical prefix paths get V6's depth-1 duplicate limit of two.
            occurrences = Counter()
            eligible = []
            for pos, edge in prefix_keys:
                topic = q.topics[pos]
                destination = int(graph.dst[edge]) if edge >= 0 else topic
                hard_key = ((topic, destination), (edge,))
                occurrences[hard_key] += 1
                eligible.append(occurrences[hard_key] <= 2)
            eligible_tensor = torch.tensor(eligible, device=device, dtype=torch.bool)
            indices = eligible_tensor.nonzero().flatten()
            z_flat = torch.zeros_like(scores).scatter(
                0, indices, capacity_topk(scores[indices], min(slot, len(indices)), tau))
            z_beam = dict(zip(prefix_keys, z_flat.unbind()))
        else:
            z_beam = {}
        branch_terms = []
        for pos in positions:
            if pos not in by_pos:
                continue
            edges, logpi1, destinations, actions, h, c = first[pos]
            edge_index = {int(e): j for j, e in enumerate(edges)}
            local1 = (capacity_topk(logpi1, min(local_k, len(edges)), tau)
                      if mode == "C" else None)
            second_cache = {}
            for route in by_pos[pos]:
                e1 = route.edges[0]
                if e1 not in edge_index:
                    continue
                j1 = edge_index[e1]
                contribution = logpi1[j1].exp()
                if len(route.edges) == 1:
                    if mode == "C":
                        contribution = contribution * local1[j1]
                    branch_terms.append(contribution)
                    continue
                if e1 not in second_cache:
                    mid = int(destinations[j1])
                    hop2 = (target_hop(program, 1, policy.n_actions) if index >= 0
                            else Hop(policy.n_actions, -1, None))
                    h2, _ = policy.step(qvec, mid, int(actions[j1]), hop2, h, c)
                    edges2 = graph.out_edges(mid).tolist() + [-1]
                    logits2, _, _, _, _ = policy.edge_logits(
                        graph, codes, qvec, h2, mid, hop2, edges2,
                        overlap, degrees, probability)
                    logpi2 = F.log_softmax(logits2, dim=0)
                    z2 = (capacity_topk(logpi2, min(local_k, len(edges2)), tau)
                          if mode == "C" else None)
                    second_cache[e1] = (edges2, logpi2, z2)
                edges2, logpi2, z2 = second_cache[e1]
                e2 = route.edges[1]
                j2 = edges2.index(e2)
                contribution = contribution * logpi2[j2].exp()
                if mode == "C":
                    contribution = contribution * z_beam[(pos, e1)] * z2[j2]
                branch_terms.append(contribution)
        if branch_terms:
            all_mass.append(torch.stack(branch_terms).sum())
    # Every traversable predicted branch receives equal weight. Flan's score
    # already affects slots, initial path score, and the policy feature.
    return (torch.stack(all_mass).sum() / len(branches) if all_mass else zero)


def question_path_loss(policy, graph, codes, q, cached, degrees,
                       routes_by_answer, *, mode, tau, epsilon=1e-12,
                       path_beam=30, local_k=60, rescue_slots=6):
    losses, masses = [], []
    for answer, routes in zip(q.answers, routes_by_answer):
        if answer in q.topics or not routes or len(routes[0].edges) > 2:
            continue  # teacher CE remains active for 0/3-hop answers
        mass = answer_mass(policy, graph, codes, q, cached, degrees,
                           answer, routes, mode=mode, tau=tau,
                           path_beam=path_beam, local_k=local_k,
                           rescue_slots=rescue_slots)
        losses.append(-(mass + epsilon).log())
        masses.append(mass.detach())
    device = next(policy.parameters()).device
    return (torch.stack(losses).mean() if losses else torch.zeros((), device=device),
            torch.stack(masses).mean() if masses else torch.zeros((), device=device),
            len(losses))
