"""SubgraphRAG cho STaRK-Prime — cài từ paper (ICLR 2025) + ĐỐI CHIẾU code gốc.

Paper: `paper/SubgraphRAG.pdf` · Code: `github.com/Graph-COM/SubgraphRAG`

    p_theta(tau | z_tau, q) = MLP([ z_q || z_h || z_r || z_t || z_tau ])

DDE (Directional Distance Encoding), paper §3.1 + code `DDE` class:
    s_e^(0)     = one-hot: e có phải topic entity không
    s_e^(l+1)   = MEAN{ s_e'^(l)   | (e',·,e) in G }     chiều TỚI
    s_e^(r,l+1) = MEAN{ s_e'^(r,l) | (e,·,e') in G }     chiều ĐI
    z_tau = [ s_h || s_t ]

Siêu tham số LẤY TỪ CODE (paper không nêu) — `configs/retriever/webqsp.yaml`:
    num_rounds = 2 · num_reverse_rounds = 2 · topic_pe = true · lr = 1e-3 · patience = 10
MLP theo code: `Linear(in, emb) -> ReLU -> Linear(emb, 1)` — đúng 2 lớp.
Loss theo code: `binary_cross_entropy_with_logits`.

🔴 SAI LỆCH BẮT BUỘC — `pos_weight`:
    Code gốc KHÔNG có trọng số lớp. Trên WebQSP mất cân bằng ~1:1,000 nên chấp nhận được.
    Trên Prime ĐO ĐƯỢC **1:164,399** (404,012 triple/subgraph, p50 chỉ 2 triple dương).
    Không có pos_weight thì gradient từ 2 mẫu dương bị 404,010 mẫu âm nhấn chìm.
    Xem PLAN §3 phần SubgraphRAG.

TỐI ƯU then chốt (paper gọi là "parallel triple-scoring"), phân rã lớp đầu:
    W·[z_q || z_h || z_r || z_t || z_tau] = W_q·z_q + W_h·z_h + W_r·z_r + W_t·z_t + W_d·z_tau
  - z_q: 1 vector/query
  - z_h, z_t: chiếu 2,000 node MỘT lần/query rồi gather
  - z_r: chiếu 18 quan hệ MỘT lần, dùng lại toàn cục
  Mỗi triple chỉ còn cộng vector đã tính sẵn. Không có nó: 100k × 3,082 × 1,024
  = 631 GFLOP/query, không chạy nổi.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dde(n_v, src, dst, seed_mask, num_rounds=2, num_reverse_rounds=2):
    """DDE — lan truyền trung bình hai chiều từ topic entity.

    src/dst: chỉ số CỤC BỘ của cạnh (h -> t). seed_mask: (n_v,) 1.0 ở topic entity.
    Trả về (n_v, 1 + num_rounds + num_reverse_rounds).

    `MEAN` chứ không phải `SUM` — theo đúng công thức paper. Trên Prime điều này quan trọng:
    bậc trung bình trong subgraph là 202, nếu dùng SUM thì giá trị sẽ nổ theo bậc.
    """
    dev = seed_mask.device
    s0 = seed_mask.clamp(max=1.0).unsqueeze(-1)              # (n_v, 1)
    feats = [s0]
    deg_in = torch.zeros(n_v, 1, device=dev).index_add_(
        0, dst, torch.ones(len(dst), 1, device=dev)).clamp(min=1)
    deg_out = torch.zeros(n_v, 1, device=dev).index_add_(
        0, src, torch.ones(len(src), 1, device=dev)).clamp(min=1)

    x = s0
    for _ in range(num_rounds):                              # chiều TỚI: (e',·,e)
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        x = agg / deg_in
        feats.append(x)
    x = s0
    for _ in range(num_reverse_rounds):                      # chiều ĐI: (e,·,e')
        agg = torch.zeros_like(x).index_add_(0, src, x[dst])
        x = agg / deg_out
        feats.append(x)
    return torch.cat(feats, dim=-1)


class SubgraphRAGRetriever(nn.Module):
    """MLP 2 lớp chấm điểm từng triple, có phân rã lớp đầu."""

    def __init__(self, lm_dim: int, n_rel: int, emb_size: int = 1024,
                 num_rounds: int = 2, num_reverse_rounds: int = 2, topic_pe: bool = True):
        super().__init__()
        self.num_rounds, self.num_reverse_rounds = num_rounds, num_reverse_rounds
        self.topic_pe = topic_pe
        dde_dim = 1 + num_rounds + num_reverse_rounds        # đặc trưng cấu trúc mỗi node
        ent_extra = dde_dim + (1 if topic_pe else 0)

        # Phân rã lớp đầu: mỗi thành phần một phép chiếu riêng, cộng lại.
        # Tương đương một Linear trên vector nối, nhưng cho phép tính sẵn theo node/quan hệ.
        self.Wq = nn.Linear(lm_dim, emb_size)                # query (có bias)
        self.Wh = nn.Linear(lm_dim + ent_extra, emb_size, bias=False)   # head
        self.Wt = nn.Linear(lm_dim + ent_extra, emb_size, bias=False)   # tail
        self.Wr = nn.Linear(lm_dim, emb_size, bias=False)               # quan hệ
        self.out = nn.Linear(emb_size, 1)

    def entity_feat(self, node_emb, dde, seed_mask):
        """[text_emb || topic_pe || DDE] cho từng node trong subgraph."""
        parts = [node_emb]
        if self.topic_pe:
            parts.append(seed_mask.clamp(max=1.0).unsqueeze(-1))
        parts.append(dde)
        return torch.cat(parts, dim=-1)

    def forward(self, q_emb, node_emb, rel_emb, src, dst, rel, seed_mask):
        """Trả logit cho TỪNG cạnh. q_emb (lm,) · node_emb (n_v, lm) · rel_emb (R, lm)."""
        n_v = node_emb.shape[0]
        dde = compute_dde(n_v, src, dst, seed_mask,
                          self.num_rounds, self.num_reverse_rounds)
        ef = self.entity_feat(node_emb, dde, seed_mask)      # (n_v, lm + extra)
        # tính sẵn theo NODE và theo QUAN HỆ, rồi chỉ gather + cộng cho từng cạnh
        ph, pt = self.Wh(ef), self.Wt(ef)                    # (n_v, emb)
        pr = self.Wr(rel_emb)                                # (R, emb)
        pq = self.Wq(q_emb)                                  # (emb,)
        z = pq.unsqueeze(0) + ph[src] + pr[rel] + pt[dst]    # (n_e, emb)
        return self.out(F.relu(z)).squeeze(-1)               # (n_e,)


def triple_to_node_scores(n_v, src, dst, logits):
    """Chuyển điểm TRIPLE -> điểm NODE cho metric xếp hạng của STaRK.

    ⚠️ THÍCH ỨNG, paper không có: SubgraphRAG trả top-K triple rồi đưa LLM, còn STaRK chấm
    bằng metric XẾP HẠNG trên node. Lấy `max` điểm các triple kề mỗi node — một node đáng
    tin nếu nó xuất hiện trong một triple điểm cao. Dùng `max` chứ không `sum` để node bậc
    cao không được lợi chỉ vì có nhiều cạnh (Prime có bậc trung bình 202 trong subgraph).
    """
    s = torch.full((n_v,), float("-inf"), device=logits.device)
    s = s.index_reduce(0, src, logits, "amax", include_self=True)
    s = s.index_reduce(0, dst, logits, "amax", include_self=True)
    # Node không nằm trên cạnh nào (subgraph PPR có thể chứa node cô lập) -> đặt dưới sàn,
    # KHÔNG để -inf vì như vậy softmax/argsort về sau sẽ sinh NaN.
    floor = float(logits.min().item()) - 1.0
    return torch.where(torch.isinf(s), torch.full_like(s, floor), s)
