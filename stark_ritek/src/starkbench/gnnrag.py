"""GNN-RAG cho STaRK-Prime — cài lại TỪ CÔNG THỨC của paper (ACL Findings 2025).

Vì sao cài lại thay vì dùng repo gốc (`github.com/cmavro/GNN-RAG`):
  - repo đòi **torch 1.7.1+cu110 / transformers 4.6.1** (2020-2021); Kaggle có torch 2.10.
    Hạ torch trên Kaggle ĐÃ từng làm vỡ transformers (GĐ4, MoR) — xem PLAN.
  - tầng dữ liệu phải viết lại trong CẢ HAI trường hợp (repo giả định Freebase, và ta không
    lưu `subgraph.tuples` được — 100-389 GB, xem PLAN §13)
  - repo có HAI nhánh `h_v^(0)`; nhánh mặc định khi đọc lướt KHÔNG phải nhánh Appendix A mô tả

Bám sát công thức, ghi rõ số hiệu để đối chiếu:

  Eq 3   q_k = gamma_k(LM(q))            r = gamma_c(LM(r))
  Eq 7   q_k = sum_j a_kj * q_j
  Eq 8   a_kj = softmax_j( W_a( q~_k (*) q_j ) )
  Eq 9   q~_k = W_k( q_{k-1} || q_c || q_{k-1}(*)q_c || q_c - q_{k-1} )      W_k: d x 4d
  Eq 4   h_v^(l+1) = psi( h_v^(l) , ||_k sum_{(v,r,v')} p_v'^(l) * sigma(q_k (*) r) )
  Eq 5   p^(l+1) = softmax( H^(l) w )
  Eq 6   p^(l) = p^(0) nếu l == L/2                      (L chẵn; L=6 -> reset ở l=3)
  App.A  h_v^(0) = ReLU( sum_{r in N_r(v)} W_r * r )     embedding quan hệ ĐÓNG BĂNG
  App.A  P = softmax( H^(L) W )                          W: d x 1
  §4.2   loss = KL( P || phân phối đáp án )

CHỖ PAPER KHÔNG RÕ, ghi lại để không tự nhận là đã bám sát:
  - Eq 8 viết `W_a` là d x d, nhưng softmax theo j cần MỘT SỐ cho mỗi token.
    Ta cài `W_a: d -> 1` (logit vô hướng). Đây là cách đọc hợp lý duy nhất.
  - App.A `W_r` — mỗi loại quan hệ một ma trận riêng, hay dùng chung? Ta dùng CHUNG
    (1 ma trận d x d cho mọi quan hệ); 18 ma trận riêng sẽ nhiều tham số hơn hẳn.
  - `sigma` không nói rõ; ta dùng sigmoid (ký hiệu thông dụng).

TỐI ƯU QUAN TRỌNG cho Prime — làm Eq 4 rẻ đi rất nhiều:
  `sigma(q_k (*) r)` phụ thuộc LOẠI QUAN HỆ, không phụ thuộc từng cạnh. Prime có 18 loại.
  Nên thay vì vật chất hoá |E| vector d chiều, ta:
      s[v, r] = sum_{(v,r,v') in E} p_v'          # scatter |E| -> |V| x 18
      agg[v, k] = sum_r s[v, r] * msg[k, r]       # einsum với msg: K x 18 x d
  Bộ nhớ: |V| x 18 + K x 18 x d, thay vì |E| x K x d. Chênh ~1,180 lần (xem PLAN §5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================  dữ liệu
class SubgraphStore:
    """Nạp cache `.cache/gnnrag/<split>.npz` + dựng lại cạnh từ CSR TOÀN CỤC.

    Ta KHÔNG lưu `subgraph.tuples` như định dạng repo gốc: trên Prime việc đó cần
    100-389 GB cho 11,204 query (PLAN §13). Lưu node id rồi dựng lại cạnh cho cùng
    một subgraph, chỉ tốn 33 MB.
    """

    def __init__(self, npz_path: str | Path, indptr, indices, edge_type, n_nodes: int):
        d = np.load(npz_path)
        self.qids = d["qids"]
        self._node = (d["node_flat"], d["node_off"])
        self._seed = (d["seed_flat"], d["seed_off"])
        self._ans = (d["ans_flat"], d["ans_off"])
        self._ppr = d["ppr_flat"] if "ppr_flat" in d else None
        self.meta = d["meta"]
        self.indptr, self.indices, self.edge_type = indptr, indices, edge_type
        self.n_nodes = n_nodes

    def __len__(self):
        return len(self.qids)

    @staticmethod
    def _get(pair, i):
        flat, off = pair
        return flat[off[i]:off[i + 1]]

    def nodes(self, i):
        return self._get(self._node, i)

    def seeds(self, i):
        return self._get(self._seed, i)

    def answers(self, i):
        return self._get(self._ans, i)

    def ppr(self, i):
        """Điểm PPR của từng node trong subgraph, cùng thứ tự với `nodes(i)`.

        Cần cho dòng đối chứng "Dense Subgraph" (Table 4 của paper) — GNN phải vượt mốc này.
        Bản cache ĐẦU không lưu trường này (lưu node sắp theo id, mất thứ tự PPR).
        """
        if self._ppr is None:
            raise RuntimeError("cache cũ không có ppr_flat — chạy lại build_gnnrag_data.py")
        _, off = self._node
        return self._ppr[off[i]:off[i + 1]]

    def subgraph(self, i):
        """(nodes, src_local, dst_local, rel) — chỉ cạnh có CẢ HAI đầu trong subgraph.

        Trả về chỉ số CỤC BỘ trong `nodes` để tensor chỉ cỡ |V_q|, không phải 129,375.
        """
        nodes = self.nodes(i)
        if len(nodes) == 0:
            z = np.zeros(0, dtype=np.int64)
            return nodes, z, z, z
        # ánh xạ toàn cục -> cục bộ; dùng mảng thưa thay vì dict cho nhanh
        loc = np.full(self.n_nodes, -1, dtype=np.int32)
        loc[nodes] = np.arange(len(nodes), dtype=np.int32)
        src, dst, rel = [], [], []
        for li, g in enumerate(nodes):
            a, b = self.indptr[g], self.indptr[g + 1]
            nb = self.indices[a:b]
            m = loc[nb] >= 0
            if not m.any():
                continue
            dst.append(loc[nb[m]])
            rel.append(self.edge_type[a:b][m])
            src.append(np.full(int(m.sum()), li, dtype=np.int32))
        if not src:
            z = np.zeros(0, dtype=np.int64)
            return nodes, z, z, z
        return (nodes, np.concatenate(src).astype(np.int64),
                np.concatenate(dst).astype(np.int64), np.concatenate(rel).astype(np.int64))


# ===========================================================================  mô hình
class QuestionEmbeddings(nn.Module):
    """Eq 3 + 7-9: K embedding câu hỏi, chú ý ĐỘNG sang phần mới của câu hỏi."""

    def __init__(self, d: int, K: int = 3, lm_dim: int | None = None):
        super().__init__()
        self.K = K
        # ⚠️ Chiều đầu ra của LM KHÁC chiều d của GNN. MedCPT ra 768; d là siêu tham số
        # (`--entity_dim` của repo gốc) chọn tự do. Bản đầu tôi giả định hai chiều BẰNG nhau
        # -> chết ở G2b với `mat1 and mat2 shapes cannot be multiplied (1x3072 and 400x100)`.
        # G2a KHÔNG bắt được vì `fake_lm(d=D)` sinh dữ liệu giả TỰ TUÂN THEO giả định sai đó.
        lm_dim = lm_dim or d
        self.q_proj = nn.Linear(lm_dim, d) if lm_dim != d else nn.Identity()
        # Eq 9: W_k ánh xạ [q_{k-1} || q_c || q_{k-1}(*)q_c || q_c - q_{k-1}] (4d) -> d
        self.W_k = nn.Linear(4 * d, d)
        # Eq 8: paper ghi d x d, nhưng softmax theo token cần MỘT SỐ -> d -> 1
        self.W_a = nn.Linear(d, 1)

    def forward(self, tok, mask, cls):
        """tok: (B,T,lm_dim) · mask: (B,T) · cls: (B,lm_dim). -> (B,K,d)"""
        tok, cls = self.q_proj(tok), self.q_proj(cls)
        qs, q_prev = [], cls
        for _ in range(self.K):
            z = torch.cat([q_prev, cls, q_prev * cls, cls - q_prev], dim=-1)
            q_t = self.W_k(z)                                            # Eq 9
            logit = self.W_a(q_t.unsqueeze(1) * tok).squeeze(-1)         # Eq 8
            logit = logit.masked_fill(~mask.bool(), float("-inf"))
            a = torch.softmax(logit, dim=-1)
            q_k = torch.bmm(a.unsqueeze(1), tok).squeeze(1)              # Eq 7
            qs.append(q_k)
            q_prev = q_k
        return torch.stack(qs, dim=1)


class GNNRAG(nn.Module):
    """GNN truy hồi của GNN-RAG. `rel_emb` (R x d) ĐÓNG BĂNG theo Appendix A."""

    def __init__(self, rel_emb: torch.Tensor, d: int, L: int = 6, K: int = 3,
                 dropout: float = 0.1, lm_dim: int | None = None,
                 node_init: str = "relation"):
        super().__init__()
        assert L % 2 == 0, "Eq 6 đòi L chẵn (reset ở l = L/2)"
        self.L, self.K, self.d = L, K, d
        self.R = rel_emb.shape[0]
        # đóng băng: "we optimize the GNN parameters, but not the relation embeddings"
        self.register_buffer("rel_emb", rel_emb)
        self.rel_proj = nn.Linear(rel_emb.shape[1], d)   # chiếu LM-dim -> d (tham số, không
        #                                                  phải embedding quan hệ, nên được học)
        self.q_enc = QuestionEmbeddings(d, K, lm_dim=lm_dim or rel_emb.shape[1])
        # `node_init`:
        #   "relation" — App.A: h_v^(0) = ReLU(sum_{r in N_r(v)} W_r·r).  BẢN CHÍNH, đúng paper.
        #   "medcpt"   — h_v^(0) = ReLU(W·MedCPT(v)).  SAI LỆCH CÓ CHỦ Ý, để đo xem 440 giá trị
        #                khởi tạo phân biệt (PLAN §12.4) có đủ không, hay bắt buộc phải có text.
        assert node_init in ("relation", "medcpt"), f"node_init không hợp lệ: {node_init}"
        self.node_init = node_init
        self.W_r = nn.Linear(d, d, bias=False)           # App.A: h_v^(0) = ReLU(sum W_r r)
        self.node_proj = (nn.Linear(lm_dim or rel_emb.shape[1], d)
                          if node_init == "medcpt" else None)
        # Eq 4: psi ánh xạ [h_v || K thông điệp] (K+1)d -> d
        self.psi = nn.Sequential(nn.Linear((K + 1) * d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d))
        self.w_p = nn.Linear(d, 1, bias=False)           # Eq 5
        self.W_out = nn.Linear(d, 1, bias=False)         # App.A: P = softmax(H^(L) W)

    def rel_d(self):
        return self.rel_proj(self.rel_emb)               # (R, d)

    def init_nodes(self, n_v, src, rel, device, node_feat=None):
        """h_v^(0). Mặc định theo App.A; "medcpt" thì lấy thẳng embedding text của node."""
        if self.node_init == "medcpt":
            if node_feat is None:
                raise ValueError("node_init='medcpt' cần node_feat (n_v, lm_dim)")
            return F.relu(self.node_proj(node_feat))
        rd = self.W_r(self.rel_d())                              # (R, d)
        # ma trận chỉ báo node x loại-quan-hệ (nhị phân: TẬP, không đếm lặp)
        ind = torch.zeros(n_v, self.R, device=device)
        if src.numel():
            ind[src, rel] = 1.0
        return F.relu(ind @ rd)                                  # (n_v, d)

    def forward(self, tok, mask, cls, n_v, src, dst, rel, seed_mask, node_feat=None):
        """Một query. src/dst/rel: chỉ số CỤC BỘ. seed_mask: (n_v,) 1.0 ở thực thể câu hỏi."""
        dev = tok.device
        q = self.q_enc(tok, mask, cls)[0]                        # (K, d)
        rd = self.rel_d()                                        # (R, d)
        # sigma(q_k (*) r) — phụ thuộc LOẠI QUAN HỆ, tính MỘT lần: (K, R, d)
        msg = torch.sigmoid(q.unsqueeze(1) * rd.unsqueeze(0))

        h = self.init_nodes(n_v, src, rel, dev, node_feat)
        p0 = seed_mask                                           # Eq 4: p_v^(0)=1 nếu là seed
        p = p0
        for l in range(self.L):
            if l == self.L // 2:
                p = p0                                           # Eq 6
            if src.numel():
                # s[v, r] = sum_{(v,r,v')} p_v'   -> tránh vật chất hoá |E| x d
                s = torch.zeros(n_v, self.R, device=dev)
                s.index_put_((src, rel), p[dst], accumulate=True)
                agg = torch.einsum("vr,krd->vkd", s, msg).reshape(n_v, self.K * self.d)
            else:
                agg = torch.zeros(n_v, self.K * self.d, device=dev)
            h = self.psi(torch.cat([h, agg], dim=-1))            # Eq 4
            p = torch.softmax(self.w_p(h).squeeze(-1), dim=0)    # Eq 5
        return torch.log_softmax(self.W_out(h).squeeze(-1), dim=0)   # App.A (log cho KL)


def kl_loss(logp, ans_local, n_v):
    """§4.2: KL(P || phân phối đáp án). Đáp án chia đều, phần còn lại 0."""
    tgt = torch.zeros(n_v, device=logp.device)
    if len(ans_local):
        tgt[ans_local] = 1.0 / len(ans_local)
    return F.kl_div(logp, tgt, reduction="sum")
