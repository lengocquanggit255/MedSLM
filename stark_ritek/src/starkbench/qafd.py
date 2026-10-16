"""QAFD-RAG — Query-Aware Flow Diffusion (Zhou et al., ICLR 2026), thích ứng cho STaRK-Prime.

Ba biến thể trọng số cạnh (Eq. 2a/2b/2c của paper, đã đối chiếu nguyên văn):

    Mean    : w̄ = ⅓·(H(u,v) + H(u,q) + H(v,q))                  — cộng, cấu trúc & ngữ nghĩa ngang hàng
    Product : w̄ = H(u,v) · H(u,q) · H(v,q)                       — nhân hoàn toàn
    Hybrid  : w̄ = H(u,v) · (a + b·(H(u,q) + H(v,q)))             — nhân với cấu trúc, cộng bên trong

Paper: *"The multiplicative interaction in Product and Hybrid amplifies edges between
query-relevant nodes while exponentially suppressing edges to irrelevant regions"*, và
*"w̄_Hybrid yields slightly better results"*. Mặc định a=b=1, 40 seed.

=============================================================================
SAI LỆCH SO VỚI PAPER — phải ghi rõ khi báo cáo
=============================================================================
1. **Đồ thị**: paper dựng KG bằng LLM (GraphRAG-style); ta dùng thẳng PrimeKG có sẵn.
2. **h(v)**: paper embed *node identifier*; ta dùng embedding của document node (đã có từ GĐ2).
3. **α**: paper đặt Δ_v = α·Σ_u T_u cho MỖI seed. Ở đây Σ_u deg(u) = 2|E| = 16.2M nên α=50
   cho ~810M khối lượng mỗi seed — diffusion sẽ quét gần hết đồ thị 129K node. Trên đồ thị nhỏ
   của paper thì hợp lý, ở quy mô này thì không. Ta coi α là nút vặn tổng khối lượng và tune
   trên val.
4. **Thứ tự push**: Algorithm 2 chọn node dư khối lượng *ngẫu nhiên đều*; ta dùng hàng đợi FIFO
   (chuẩn của APPR). Cùng điểm bất động, nhưng tái lập được và nhanh hơn.
"""
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

VARIANTS = ("mean", "product", "hybrid")


def rbf(h: np.ndarray | float, gamma: float):
    """Nhân RBF trên vector đã L2-chuẩn hoá: exp(-γ‖e-e'‖²) = exp(-2γ(1-H)).

    Paper cho phép chọn H_sim là cosine, dot product, **hoặc RBF**. Ở đây RBF là bắt buộc
    trên thực tế: embedding của corpus này bị anisotropy nặng nên cosine chỉ trải trong dải
    hẹp (H(v,q): p10=0.640 → p90=0.728), khiến trọng số cạnh gần như đồng đều (CV≈0.05) và
    QAFD suy biến thành khuếch tán KHÔNG có nhận thức query. RBF khôi phục dải động
    (γ=20 → CV≈0.97; γ=50 → CV≈1.86), đúng cơ chế "exponentially suppressing edges to
    irrelevant regions" mà paper mô tả. γ=0 nghĩa là dùng cosine thô.
    """
    if gamma <= 0:
        return h
    return np.exp(-gamma * (1.0 - h))


def edge_weight(mode: str, h_uv: np.ndarray, h_uq: float, h_vq: np.ndarray,
                a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Trọng số cạnh cho các cạnh đi ra từ một node u. Vector hoá theo hàng xóm v."""
    if mode == "mean":
        return (h_uv + h_uq + h_vq) / 3.0
    if mode == "product":
        return h_uv * h_uq * h_vq
    if mode == "hybrid":
        return h_uv * (a + b * (h_uq + h_vq))
    raise ValueError(f"biến thể không hợp lệ: {mode} (hợp lệ: {VARIANTS})")


def diffuse(indptr, indices, h_edge, deg, seeds, s_q, mode="hybrid",
            alpha=50.0, eps=1e-4, a=1.0, b=1.0, max_push=400_000) -> Dict[int, float]:
    """Algorithm 2 — khuếch tán dòng có nhận thức query, dạng local push.

    Args:
        indptr/indices: CSR của đồ thị (vô hướng, đã lưu 2 chiều).
        h_edge: (n_edges,) H_sim(h(u),h(v)) tiền tính cho từng cạnh — KHÔNG phụ thuộc query.
        deg: (n_nodes,) bậc, dùng làm sức chứa T_v = deg(v).
        seeds: mảng node id làm nguồn.
        s_q: (n_nodes,) H_sim(h(v), h(q)) cho query hiện tại.
        alpha: hệ số khối lượng nguồn (xem sai lệch #3 ở đầu file).
        max_push: chặn trên số lần push, tránh treo trên hub.

    Returns:
        {node_id: x_v} điểm quan trọng của các node đã chạm tới.
    """
    T = deg.astype(np.float64)
    m: Dict[int, float] = {}
    x: Dict[int, float] = {}
    # Khối lượng nguồn: chia đều alpha * tổng sức chứa của SEED (không phải của cả đồ thị —
    # xem sai lệch #3), để quy mô không phụ thuộc kích thước đồ thị.
    total = alpha * float(T[seeds].sum())
    for s in seeds:
        m[int(s)] = m.get(int(s), 0.0) + total / len(seeds)

    q = deque(int(s) for s in seeds if m[int(s)] > T[int(s)])
    inq = set(q)
    pushes = 0
    while q and pushes < max_push:
        v = q.popleft()
        inq.discard(v)
        mv = m.get(v, 0.0)
        if mv <= T[v] + eps:
            continue
        lo, hi = indptr[v], indptr[v + 1]
        nbr = indices[lo:hi]
        if nbr.size == 0:
            continue
        w = edge_weight(mode, h_edge[lo:hi], float(s_q[v]), s_q[nbr], a, b)
        # Trọng số âm (cosine có thể âm) làm hỏng bảo toàn khối lượng -> kẹp về >= 0.
        np.clip(w, 0.0, None, out=w)
        wv = float(w.sum())
        if wv <= 0.0:
            continue
        excess = mv - T[v]
        x[v] = x.get(v, 0.0) + excess / wv
        m[v] = T[v]
        share = excess * (w / wv)
        for u, du in zip(nbr.tolist(), share.tolist()):
            if du <= 0.0:
                continue
            m[u] = m.get(u, 0.0) + du
            if m[u] > T[u] + eps and u not in inq:
                q.append(u)
                inq.add(u)
        pushes += 1
    return x


def precompute_edge_sim(indptr, indices, node_emb, batch=2_000_000) -> np.ndarray:
    """H_sim(h(u),h(v)) cho mọi cạnh. Không phụ thuộc query nên chỉ tính MỘT lần."""
    n = len(indptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    out = np.empty(len(indices), dtype=np.float32)
    E = node_emb.astype(np.float32)
    for s in range(0, len(indices), batch):
        e = slice(s, min(s + batch, len(indices)))
        out[e] = np.einsum("ij,ij->i", E[src[e]], E[indices[e]])
    return out
