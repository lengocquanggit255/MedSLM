"""SubgraphRAG cho STaRK-Prime — cài từ paper (ICLR 2025), KHỚP code gốc retriever.

Paper: `paper/SubgraphRAG.pdf` · Code: `github.com/Graph-COM/SubgraphRAG` @ 48eb05e, thư mục retrieve/:
`src/model/retriever.py` (Retriever, DDE, PEConv), `train.py`, `configs/retriever/webqsp.yaml`.

Mô hình y hệt `Retriever` gốc (cùng tên tham số, cùng thứ tự khởi tạo -> nạp state_dict qua lại được):
    h_e     = [ emb thực thể || one-hot topic (2 chiều) || DDE ]      (topic_pe = true)
    logit   = pred([ h_q || h_e[h] || h_r || h_e[t] ]),  pred = Linear(in, emb) -> ReLU -> Linear(emb, 1)
DDE (Directional Distance Encoding), như class `DDE` gốc: lan truyền chính one-hot topic 2 chiều,
mỗi vòng = MEAN trên cạnh (PyG MessagePassing aggr='mean', node không có message = 0):
    num_rounds vòng theo chiều cạnh (h -> t), rồi num_reverse_rounds vòng theo chiều ngược (t -> h)
    => mỗi vòng 2 chiều; DDE = nối các vòng (KHÔNG gồm vòng 0 — vòng 0 là chính one-hot topic).
Siêu tham số theo `configs/retriever/webqsp.yaml`: num_rounds = 2 · num_reverse_rounds = 2 · topic_pe =
true; emb_size = số chiều embedding text (gte-large-en-v1.5: 1024), như `emb_size = q_emb.shape[-1]`.
`non_text_entity_emb` dành cho thực thể không có text (Freebase CVT); Prime không có nên không dùng,
nhưng vẫn khai báo để state_dict và thứ tự khởi tạo trùng code gốc.

TỐI ƯU (paper gọi là "parallel triple-scoring"), không đổi kết quả: tách lớp đầu theo khối cột,
    W·[h_q || h_e[h] || h_r || h_e[t]] + b = (W_q·h_q + b) + (W_h·h_e)[h] + (W_r·h_r)[r] + (W_t·h_e)[t]
chiếu mỗi node / quan hệ MỘT lần rồi gather theo cạnh, thay vì dựng vector nối cho từng cạnh
(subgraph Prime p99 ~1.8M cạnh × 4,106 chiều). W_q, W_h, W_r, W_t là các lát cột của CHÍNH
`pred[0].weight`. Kiểm số với code gốc: ritek/scripts/test_sgrag_equivalence.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def topic_one_hot(seed_mask):
    """One-hot 2 chiều [không phải topic, là topic] — `F.one_hot(topic_entity_mask.long(), 2)` gốc."""
    return F.one_hot((seed_mask > 0).long(), num_classes=2).float()


def compute_dde(n_v, src, dst, one_hot, num_rounds=2, num_reverse_rounds=2):
    """DDE gốc: list vòng lan truyền MEAN của one-hot topic -> (n_v, 2·(num_rounds + num_reverse_rounds)).

    src/dst: chỉ số CỤC BỘ của cạnh (h -> t), lấy TOÀN BỘ cạnh của subgraph như code gốc.
    """
    def mean_conv(x, frm, to):
        agg = torch.zeros_like(x).index_add_(0, to, x[frm])
        cnt = torch.zeros(n_v, 1, device=x.device, dtype=x.dtype).index_add_(
            0, to, torch.ones(len(to), 1, device=x.device, dtype=x.dtype))
        return agg / cnt.clamp(min=1)

    feats = []
    x = one_hot
    for _ in range(num_rounds):                              # edge_index = [h, t]: h -> t
        x = mean_conv(x, src, dst)
        feats.append(x)
    x = one_hot
    for _ in range(num_reverse_rounds):                      # reverse_edge_index = [t, h]
        x = mean_conv(x, dst, src)
        feats.append(x)
    return torch.cat(feats, dim=-1) if feats else one_hot.new_zeros(n_v, 0)


class SubgraphRAGRetriever(nn.Module):
    """`Retriever` gốc, chấm theo khối cột của lớp đầu."""

    def __init__(self, emb_size: int, topic_pe: bool = True, num_rounds: int = 2, num_reverse_rounds: int = 2):
        super().__init__()
        self.emb_size, self.topic_pe = emb_size, topic_pe
        self.num_rounds, self.num_reverse_rounds = num_rounds, num_reverse_rounds
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.ent_dim = emb_size + (2 if topic_pe else 0) + 2 * (num_rounds + num_reverse_rounds)
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (num_rounds + num_reverse_rounds)
        assert pred_in_size == 2 * emb_size + 2 * self.ent_dim
        self.pred = nn.Sequential(nn.Linear(pred_in_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1))

    def entity_feat(self, node_emb, one_hot, dde):
        """h_e = [emb || one-hot topic || DDE], đúng thứ tự nối của code gốc."""
        return torch.cat([node_emb] + ([one_hot] if self.topic_pe else []) + [dde], dim=1)

    def project(self, q_emb, node_emb, rel_emb, one_hot, dde):
        """Chiếu theo khối cột của pred[0]: (pq (1, emb) có bias, ph, pt (n_v, emb), pr (R, emb))."""
        E, D = self.emb_size, self.ent_dim
        W, b = self.pred[0].weight, self.pred[0].bias
        he = self.entity_feat(node_emb, one_hot, dde)
        pq = F.linear(q_emb.reshape(1, -1), W[:, :E], b)
        ph = F.linear(he, W[:, E:E + D])
        pr = F.linear(rel_emb, W[:, E + D:2 * E + D])
        pt = F.linear(he, W[:, 2 * E + D:])
        return pq, ph, pt, pr

    def score(self, proj, src, dst, rel):
        """Logit (n_e,) cho các cạnh cho trước, từ kết quả `project`."""
        pq, ph, pt, pr = proj
        return self.pred[2](F.relu(pq + ph[src] + pr[rel] + pt[dst])).squeeze(-1)

    def forward(self, q_emb, node_emb, rel_emb, src, dst, rel, one_hot, dde=None, edge_chunk=None):
        """Logit cho TỪNG cạnh. edge_chunk: chấm theo khúc để giới hạn bộ nhớ, kết quả y hệt.

        dde: DDE tính sẵn trên toàn subgraph (không có tham số học); None thì tính tại chỗ.
        """
        if dde is None:
            dde = compute_dde(node_emb.shape[0], src, dst, one_hot, self.num_rounds, self.num_reverse_rounds)
        proj = self.project(q_emb, node_emb, rel_emb, one_hot, dde)
        step = edge_chunk or max(len(src), 1)
        out = [self.score(proj, src[a:a + step], dst[a:a + step], rel[a:a + step])
               for a in range(0, len(src), step)]
        return torch.cat(out) if out else proj[0].new_zeros(0)


def chunked_bce_backward(model, q_emb, node_emb, rel_emb, src, dst, rel, one_hot, dde, y, pos_weight=None,
                         edge_chunk=200_000):
    """loss = BCE-with-logits TRUNG BÌNH trên mọi cạnh (như `F.binary_cross_entropy_with_logits` gốc),
    backward theo khúc cạnh để không giữ activation của cả subgraph (p99 ~1.8M cạnh × 1,024).

    Mỗi khúc: tổng loss / n_e, backward vào bản tách (detach) của phép chiếu theo node/quan hệ + lớp
    cuối; xong mọi khúc mới backward một lần qua phép chiếu. Gradient = gradient của loss trung bình
    trên toàn subgraph (chỉ khác thứ tự cộng số thực). Trả loss (float).
    """
    n, total = len(src), 0.0
    if n == 0:                                               # gốc: `if len(h_id_tensor) == 0: continue`
        return total
    proj = model.project(q_emb, node_emb, rel_emb, one_hot, dde)
    leaves = [p.detach().requires_grad_() for p in proj]
    for a in range(0, n, edge_chunk):
        lg = model.score(leaves, src[a:a + edge_chunk], dst[a:a + edge_chunk], rel[a:a + edge_chunk])
        part = F.binary_cross_entropy_with_logits(lg, y[a:a + edge_chunk], pos_weight=pos_weight,
                                                  reduction="sum") / n
        part.backward()
        total += float(part.item())
    torch.autograd.backward(list(proj), [lv.grad for lv in leaves])
    return total


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
