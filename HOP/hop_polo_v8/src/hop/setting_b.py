"""Setting B: khối lượng đường biên của route tới đáp án dưới chương trình Flan dự đoán (chỉ train).

Khối lượng không phải xác suất đã hiệu chỉnh của việc lọt vào pool có trần; chỉ tìm kiếm cứng mới đo
độ phủ. Mode B (huấn luyện) không có bộ chọn; mode C (chỉ chẩn đoán soft-hard gap) nhân thêm bộ chọn
top-k khả vi `capacity_topk` với nhiệt độ tau.

Khác V7 về cách tính: phần tĩnh của các cạnh ra một nút (`edge_static`) được tính một lần cho mỗi câu
hỏi rồi dùng chung cho mọi nhánh chương trình; chỉ số cạnh lấy trực tiếp từ đoạn CSR thay vì dict/list;
log-softmax ở bước hai được gradient-checkpoint khi huấn luyện (giữ bộ nhớ GPU khi nút giữa là hub).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .labels import Route
from .policy import HIDDEN, FullPolicy, GraphTensors, QuestionContext, initial_score, question_context, target_hop
from .program import Hop, program_from_dict
from .search import allocate_slots, pool_candidates, rescue_probability


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
        # lambda giải sum sigmoid((s - lambda)/tau) = k; đạo hàm theo lambda được tính giải tích ở backward.
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
    """Bộ chọn dung lượng có điều chuẩn entropy; sum(z) = k trong sai số bộ giải."""
    if scores.ndim != 1:
        raise ValueError("scores must be one dimensional")
    return _CapacityTopK.apply(scores, min(max(int(k), 0), scores.numel()), tau)


def program_branches(cached, q, path_beam: int, rescue_slots: int):
    programs = [(program_from_dict(row["program"]), float(row["probability"])) for row in cached["beams"]]
    slots, terminal = allocate_slots(programs, path_beam, rescue_slots)
    branches = []
    for index, count in slots.items():
        if count <= 0:
            continue
        probability = programs[index][1] if index >= 0 else rescue_probability(programs, terminal)
        program = programs[index][0] if index >= 0 else None
        # V6 cắt toàn cục các trạng thái topic đầu về dung lượng nhánh (global_prune duplicate_limit=5).
        positions_list, topic_counts = [], Counter()
        for pos, topic in enumerate(q.topics):
            if topic_counts[topic] >= 5:
                continue
            positions_list.append(pos)
            topic_counts[topic] += 1
            if len(positions_list) == count:
                break
        branches.append((index, program, probability, count, tuple(positions_list)))
    return branches


def answer_mass(policy: FullPolicy, gt: GraphTensors, q, cached, qctx: QuestionContext, routes: list[Route], *,
                mode: str, tau: float, path_beam: int, local_k: int, rescue_slots: int,
                static_cache: dict | None = None) -> torch.Tensor:
    """Khối lượng cho một đáp án train; chấm cả trạng thái bước hai của route vàng kể cả khi beam cứng sẽ cắt."""
    if mode not in ("B", "C"):
        raise ValueError(mode)
    device = gt.device
    zero = torch.zeros((), device=device)
    if not routes:
        return zero
    qvec = policy.question(qctx.words, device)
    branches = program_branches(cached, q, path_beam, rescue_slots)
    if not branches:
        return zero
    static_cache = {} if static_cache is None else static_cache

    def static(node):
        if node not in static_cache:
            static_cache[node] = policy.edge_static(gt, node)
        return static_cache[node]

    def second_hop(mid_static, h2, hop2, probability):
        return F.log_softmax(policy.edge_logits(gt, mid_static, qctx.overlap, h2, hop2, probability).logits, dim=0)

    by_pos = defaultdict(list)
    for route in routes:
        by_pos[route.topic_pos].append(route)
    hidden_zero = torch.zeros(HIDDEN, device=device)
    all_mass = []
    for index, program, probability, slot, positions in branches:
        if not positions:
            continue
        start_score = initial_score(probability, len(q.topics))
        hop1 = target_hop(program, 0, policy.n_actions) if index >= 0 else Hop(policy.n_actions, None)
        hop2 = target_hop(program, 1, policy.n_actions) if index >= 0 else Hop(policy.n_actions, None)
        first = {}
        for pos in positions:
            current = q.topics[pos]
            h, c = policy.step(qvec, current, policy.n_actions, hop1, hidden_zero, hidden_zero)
            scores = policy.edge_logits(gt, static(current), qctx.overlap, h, hop1, probability)
            first[pos] = (scores, F.log_softmax(scores.logits, dim=0), h, c)
        z_beam = None
        if mode == "C":
            # Tiền tố trùng hệt nhau được giới hạn hai bản ở độ sâu 1, như hard search V6.
            flat_scores, eligible, offsets, occurrences, offset = [], [], {}, Counter(), 0
            for pos in positions:
                scores, logpi, _, _ = first[pos]
                flat_scores.append(logpi + start_score)
                offsets[pos] = offset
                offset += len(logpi)
                topic = q.topics[pos]
                for j, destination in enumerate(scores.static.dst.tolist()):
                    key = ((topic, destination), (scores.edge_id(j),))
                    occurrences[key] += 1
                    eligible.append(occurrences[key] <= 2)
            flat = torch.cat(flat_scores)
            keep = torch.tensor(eligible, device=device, dtype=torch.bool).nonzero().flatten()
            z_flat = torch.zeros_like(flat).scatter(
                0, keep, capacity_topk(flat[keep], min(slot, len(keep)), tau))
            z_beam = (z_flat, offsets)
        branch_terms = []
        for pos in positions:
            if pos not in by_pos:
                continue
            scores1, logpi1, h, c = first[pos]
            start1, n1 = scores1.static.start, len(logpi1)
            local1 = capacity_topk(logpi1, min(local_k, n1), tau) if mode == "C" else None
            second = {}
            for route in by_pos[pos]:
                j1 = route.edges[0] - start1
                if not 0 <= j1 < n1 - 1:
                    continue
                contribution = logpi1[j1].exp()
                if len(route.edges) == 1:
                    if mode == "C":
                        contribution = contribution * local1[j1]
                    branch_terms.append(contribution)
                    continue
                e1 = route.edges[0]
                if e1 not in second:
                    mid = int(scores1.static.dst[j1])
                    h2, _ = policy.step(qvec, mid, int(scores1.static.act[j1]), hop2, h, c)
                    mid_static = static(mid)
                    if torch.is_grad_enabled():
                        # Nút giữa có thể là hub ~15k cạnh (Prime): kích hoạt [cạnh, 64] của mọi nhánh x cạnh đầu
                        # vượt 14 GB nếu giữ cho backward, nên tính lại khi backward (cùng phép toán, cùng gradient).
                        logpi2 = checkpoint(second_hop, mid_static, h2, hop2, probability, use_reentrant=False)
                    else:
                        logpi2 = second_hop(mid_static, h2, hop2, probability)
                    z2 = capacity_topk(logpi2, min(local_k, len(logpi2)), tau) if mode == "C" else None
                    second[e1] = (mid_static.start, logpi2, z2)
                start2, logpi2, z2 = second[e1]
                j2 = route.edges[1] - start2
                contribution = contribution * logpi2[j2].exp()
                if mode == "C":
                    z_flat, offsets = z_beam
                    contribution = contribution * z_flat[offsets[pos] + j1] * z2[j2]
                branch_terms.append(contribution)
        if branch_terms:
            all_mass.append(torch.stack(branch_terms).sum())
    # Mỗi nhánh chương trình đi được có trọng số bằng nhau; điểm Flan đã vào slot, điểm đầu và đặc trưng policy.
    return torch.stack(all_mass).sum() / len(branches) if all_mass else zero


def question_path_loss(policy, gt, q, cached, qctx, routes_by_answer, *, mode, tau, epsilon=1e-12,
                       path_beam=30, local_k=60, rescue_slots=6):
    losses, masses = [], []
    static_cache: dict = {}
    for answer, routes in zip(q.answers, routes_by_answer):
        if answer in q.topics or not routes or len(routes[0].edges) > 2:
            continue  # teacher CE vẫn phủ đáp án 0 hop / 3 hop
        mass = answer_mass(policy, gt, q, cached, qctx, routes, mode=mode, tau=tau, path_beam=path_beam,
                           local_k=local_k, rescue_slots=rescue_slots, static_cache=static_cache)
        losses.append(-(mass + epsilon).log())
        masses.append(mass.detach())
    device = gt.device
    return (torch.stack(losses).mean() if losses else torch.zeros((), device=device),
            torch.stack(masses).mean() if masses else torch.zeros((), device=device),
            len(losses))


def train_soft_hard_gap(policy, gt, questions, cache, routes, config, tau):
    """Chẩn đoán: khối lượng mềm (mode C) so với độ phủ cứng của pool trên vài câu train đầu."""
    limit = min(config["train_gap_audit_qas"], len(questions))
    masses, hard = [], []
    with torch.no_grad():
        for q, cached, grouped in zip(questions[:limit], cache[:limit], routes[:limit]):
            qctx = question_context(gt, q)
            _, mean_mass, count = question_path_loss(
                policy, gt, q, cached, qctx, grouped, mode="C", tau=tau, path_beam=config["path_beam"],
                local_k=config["local_top_k"], rescue_slots=config["terminal_rescue_slots"])
            if not count:
                continue
            nodes, *_ = pool_candidates(policy, gt, q, cached, qctx, path_beam=config["path_beam"],
                                        local_k=config["local_top_k"],
                                        rescue_slots=config["terminal_rescue_slots"],
                                        maximum=config["max_pool_nodes"], depth_quotas=config["pool_depth_quotas"],
                                        last_hop=config["pool_last_hop"])
            eligible = [a for a, rr in zip(q.answers, grouped) if a not in q.topics and rr and len(rr[0].edges) <= 2]
            if eligible:
                masses.append(float(mean_mass))
                hard.append(sum(a in nodes for a in eligible) / len(eligible))
    if not masses:
        return {"qas": 0, "mean_soft_mass": None, "hard_coverage": None, "difference": None}
    return {"qas": len(masses), "mean_soft_mass": float(np.mean(masses)), "hard_coverage": float(np.mean(hard)),
            "difference": float(np.mean(hard) - np.mean(masses)),
            "interpretation": "diagnostic only; path mass is not calibrated pool probability"}
