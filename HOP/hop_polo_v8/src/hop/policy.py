"""Policy cạnh kiểu PoLo: lịch sử LSTM theo câu hỏi, chấm MỌI cạnh ra của nút hiện tại + self-loop.

Chương trình Flan (action A, tiền tố S) là prior có trọng số cộng vào logit, không phải bộ lọc cứng.

Khác V7 về cách tính, không đổi về toán:
  - Cạnh ra của một nút là một đoạn liền trong CSR, nên đặc trưng cạnh lấy bằng slice tensor trên device
    thay vì dựng list Python cho từng cạnh.
  - Lớp Linear đầu của `edge_mlp` tách theo khối cột [hidden | nút | action | S | extras]: phần hidden
    tính một lần cho mỗi trạng thái (V7 nhân lại cho từng cạnh), phần action là bảng tra theo schema.
    Layout tham số giữ nguyên V7, nên test nạp cùng trọng số vào hai bản và so logit.
  - Đặc trưng trùng từ câu hỏi-tên nút tính một lần cho mỗi câu (ma trận thưa), thay `LazyOverlap`.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .codebook import Codebook
from .common import WORD, hash_words
from .data import Graph, Question
from .labels import path_program
from .program import Hop, Program

HIDDEN, NODE, ACTION, CODE, EXTRA = 96, 32, 32, 16, 6


class GraphTensors:
    """Mảng tĩnh của đồ thị (trên device) và ma trận token tên nút, dựng một lần."""

    def __init__(self, graph: Graph, codebook: Codebook, device: torch.device):
        self.device = device
        self.indptr = graph.indptr
        self.dst = torch.as_tensor(graph.dst.astype(np.int64), device=device)
        self.schema = torch.as_tensor(graph.schema.astype(np.int64), device=device)
        self.codes = torch.as_tensor(codebook.node_codes.astype(np.int64), device=device)
        self.degrees = torch.as_tensor(np.log1p(np.diff(graph.indptr)) / 16, dtype=torch.float32, device=device)
        vocab: dict[str, int] = {}
        rows, cols = [], []
        for node, name in enumerate(graph.names):
            for token in set(WORD.findall(name.casefold())):
                rows.append(node)
                cols.append(vocab.setdefault(token, len(vocab)))
        self.vocab = vocab
        self.name_tokens = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)),
                                         shape=(len(graph.ids), max(1, len(vocab))))

    def degree(self, node: int) -> int:
        return int(self.indptr[node + 1] - self.indptr[node])


@dataclass
class QuestionContext:
    words: list[int]
    overlap: torch.Tensor   # [n_nodes] float32: |từ câu hỏi ∩ từ tên nút| / |từ câu hỏi|


def question_context(gt: GraphTensors, q: Question) -> QuestionContext:
    terms = set(WORD.findall(q.question.casefold()))
    indicator = np.zeros(gt.name_tokens.shape[1], np.float64)
    known = [gt.vocab[t] for t in terms if t in gt.vocab]
    indicator[known] = 1.0
    overlap = (gt.name_tokens @ indicator) / max(1, len(terms))
    return QuestionContext(hash_words(q.question),
                           torch.as_tensor(overlap.astype(np.float32), device=gt.device))


@dataclass
class EdgeStatic:
    """Phần không phụ thuộc trạng thái của các cạnh ra một nút (cạnh cuối = self-loop)."""
    start: int
    dst: torch.Tensor       # [n]
    act: torch.Tensor       # [n]
    codes: torch.Tensor     # [n, levels]
    pre: torch.Tensor       # [n, 64] = W_node·nút + W_action·action + W_code·mean(S)


@dataclass
class EdgeScores:
    static: EdgeStatic
    logits: torch.Tensor        # [n]
    fallbacks: torch.Tensor     # [n] số tầng S lùi
    action_match: torch.Tensor  # [n] bool

    def edge_id(self, j: int) -> int:
        return self.static.start + j if j < len(self.logits) - 1 else -1


class FullPolicy(nn.Module):
    """Lịch sử LSTM theo câu hỏi; điểm cạnh và điểm nút trên toàn láng giềng (cùng tham số V7)."""

    def __init__(self, n_nodes: int, n_actions: int, n_codes: int, max_hops: int):
        super().__init__()
        self.max_hops = max_hops
        self.words = nn.Embedding(16384, 64)
        self.nodes = nn.Embedding(n_nodes, NODE)
        self.actions = nn.Embedding(n_actions + 1, ACTION)  # ID cuối là self-loop
        self.s_codes = nn.Embedding(n_codes, CODE)
        self.history = nn.LSTMCell(64 + NODE + ACTION + ACTION + CODE, HIDDEN)
        self.edge_mlp = nn.Sequential(nn.Linear(HIDDEN + NODE + ACTION + CODE + EXTRA, 64),
                                      nn.Tanh(), nn.Linear(64, 1))
        self.node_mlp = nn.Sequential(nn.Linear(64 + NODE + HIDDEN + 4, 64), nn.Tanh(), nn.Linear(64, 1))
        self.n_actions = n_actions

    def question(self, tokens, device):
        return self.words(torch.as_tensor(tokens, dtype=torch.long, device=device)).mean(0)

    def code_vector(self, code, device):
        if not code:
            return torch.zeros(CODE, device=device)
        return self.s_codes(torch.as_tensor(code, dtype=torch.long, device=device)).mean(0)

    def step(self, qvec, current, previous_action, hop: Hop, hidden, cell):
        x = torch.cat((qvec, self.nodes.weight[current], self.actions.weight[previous_action],
                       self.actions.weight[hop.action], self.code_vector(hop.s, qvec.device)))
        h, c = self.history(x.unsqueeze(0), (hidden.unsqueeze(0), cell.unsqueeze(0)))
        return h.squeeze(0), c.squeeze(0)

    def _blocks(self):
        w = self.edge_mlp[0].weight
        cut = np.cumsum([0, HIDDEN, NODE, ACTION, CODE, EXTRA])
        return [w[:, cut[i]:cut[i + 1]] for i in range(5)]

    def edge_static(self, gt: GraphTensors, current: int) -> EdgeStatic:
        start, end = int(gt.indptr[current]), int(gt.indptr[current + 1])
        device = gt.device
        dst = torch.cat((gt.dst[start:end], torch.tensor([current], device=device)))
        act = torch.cat((gt.schema[start:end], torch.tensor([self.n_actions], device=device)))
        codes = gt.codes[dst]
        _, w_node, w_action, w_code, _ = self._blocks()
        action_table = self.actions.weight @ w_action.t()
        pre = self.nodes(dst) @ w_node.t() + action_table[act] + self.s_codes(codes).mean(1) @ w_code.t()
        return EdgeStatic(start, dst, act, codes, pre)

    def edge_logits(self, gt: GraphTensors, static: EdgeStatic, overlap: torch.Tensor, hidden,
                    hop: Hop, program_probability: float) -> EdgeScores:
        n = len(static.dst)
        target = hop.s or ()
        matches = torch.zeros(n, dtype=torch.long, device=gt.device)
        for level, code in enumerate(target):
            matches = matches + ((matches == level) & (static.codes[:, level] == code)).long()
        fallbacks = len(target) - matches
        action_match = static.act == hop.action
        extras = torch.stack((action_match.float(), (matches.double() / 3).float(),
                              (fallbacks.double() / 3).float(), overlap[static.dst], gt.degrees[static.dst],
                              torch.full((n,), float(program_probability), device=gt.device)), dim=1)
        w_hidden, _, _, _, w_extra = self._blocks()
        pre = (static.pre + (hidden @ w_hidden.t()).unsqueeze(0) + extras @ w_extra.t()
               + self.edge_mlp[0].bias)
        raw = self.edge_mlp[2](torch.tanh(pre)).squeeze(1)
        # A/S là prior có thang bậc: tiền tố S dài nhất được tính một lần, nhánh lùi cùng tồn tại với nhánh khớp.
        # Khi chương trình đã hết hop, policy tự chọn dừng hay đi tiếp mà không có action mục tiêu.
        prior_match = torch.ones_like(extras[:, 0]) if hop.action == self.n_actions else extras[:, 0]
        bias = 1.2 * prior_match + 1.05 * extras[:, 1] - 2.1 * extras[:, 2] - 1.3 * (1 - prior_match)
        return EdgeScores(static, raw + bias, fallbacks, action_match)

    def edge_scores(self, gt: GraphTensors, overlap, hidden, current: int, hop: Hop,
                    program_probability: float) -> EdgeScores:
        return self.edge_logits(gt, self.edge_static(gt, current), overlap, hidden, hop, program_probability)

    def rank(self, qvec, nodes, evidence):
        device = qvec.device
        ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
        h = torch.stack([evidence[node][0] for node in nodes])
        extras = torch.tensor([[max(-30.0, evidence[n][1]) / 30, min(evidence[n][2], 30) / 30,
                                evidence[n][3] / self.max_hops, float(evidence[n][1] == 0)] for n in nodes],
                              dtype=torch.float32, device=device)
        features = torch.cat((qvec.expand(len(nodes), -1), self.nodes(ids), h, extras), dim=1)
        return self.node_mlp(features).squeeze(1)


@dataclass
class SearchState:
    program_index: int
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    backoffs: tuple[int, ...]
    action_matches: tuple[bool, ...]
    score: float
    node_scores: tuple[float, ...]
    node_hiddens: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    cell: torch.Tensor
    previous_action: int


def global_prune(states, width, duplicate_limit=2):
    """Điểm đường toàn cục, tối đa `duplicate_limit` ngữ cảnh chương trình cho mỗi tiền tố đường."""
    kept, counts = [], Counter()
    for state in sorted(states, key=lambda s: (-s.score, s.program_index, s.edges)):
        key = (state.nodes, state.edges)
        if counts[key] < duplicate_limit:
            kept.append(state)
            counts[key] += 1
        if len(kept) == width:
            break
    return kept


def target_hop(program: Program, depth: int, stop_id: int) -> Hop:
    return program.hops[depth] if depth < len(program.hops) else Hop(stop_id, None)


def teacher_loss(policy: FullPolicy, gt: GraphTensors, graph: Graph, codebook: Codebook, q: Question,
                 teacher, qctx: QuestionContext, device):
    """CE trên toàn bộ danh sách láng giềng theo đường teacher + rank loss nút (đáp án vs nhiễu điểm cao)."""
    qvec = policy.question(qctx.words, device)
    zero = torch.zeros(HIDDEN, device=device)
    evidence = {topic: (zero, 0.0, 1, 0) for topic in q.topics}
    if teacher is None:
        return torch.zeros((), device=device), torch.zeros((), device=device)
    pos, path = teacher
    program = path_program(graph, codebook, q, pos, path)
    current = q.topics[pos]
    h = c = zero
    previous = policy.n_actions
    losses, distractors = [], set()
    for depth in range(policy.max_hops):
        hop = target_hop(program, depth, policy.n_actions)
        start, end = int(gt.indptr[current]), int(gt.indptr[current + 1])
        target = path[depth] if depth < len(path) else -1
        if target != -1 and not start <= target < end:
            raise AssertionError("teacher edge absent from full neighbor list")
        j = target - start if target != -1 else end - start
        h, c = policy.step(qvec, current, previous, hop, h, c)
        scores = policy.edge_scores(gt, qctx.overlap, h, current, hop, 1.0)
        losses.append(F.cross_entropy(scores.logits.unsqueeze(0), torch.tensor([j], device=device)))
        # Nhiễu khó = cạnh sai điểm cao, không bỏ cạnh nào khỏi mẫu số CE.
        chosen = torch.topk(scores.logits.detach(), min(32, len(scores.logits))).indices
        distractors.update(scores.static.dst[chosen].tolist())
        current = int(scores.static.dst[j])
        evidence[current] = (h, 0.0, evidence.get(current, (None, None, 0, None))[2] + 1, depth + 1)
        previous = int(scores.static.act[j])
    for node in distractors:
        evidence.setdefault(node, (zero, -2.0, 1, 1))
    nodes = sorted(evidence)
    node_scores = policy.rank(qvec, nodes, evidence)
    positives = torch.as_tensor([node in q.answers for node in nodes], device=device)
    rank_loss = (F.softplus(node_scores[~positives][None, :] - node_scores[positives][:, None]).mean()
                 if positives.any() and (~positives).any() else torch.zeros((), device=device))
    return torch.stack(losses).mean(), rank_loss


def initial_score(probability: float, positions: int) -> float:
    return math.log(max(probability, 1e-12)) - math.log(max(1, positions))
