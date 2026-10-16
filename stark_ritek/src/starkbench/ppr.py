"""Personalized PageRank cho HippoRAG 2 (thích ứng cho STaRK-Prime).

Khác QAFD-RAG ở chỗ nào: QAFD dùng flow diffusion với **trọng số cạnh phụ thuộc query**;
PPR dùng trọng số cạnh **đồng đều** nhưng **vector reset phụ thuộc query**. Trên corpus này,
trọng số cạnh phụ thuộc query đã được chứng minh là vô dụng vì embedding bị anisotropy
(H(v,q) chỉ trải 0.640→0.728, xem RESULTS.md GĐ3) — nên PPR có cơ sở để hoạt động tốt hơn.

SAI LỆCH SO VỚI PAPER — ghi rõ khi báo cáo:
1. **Đồ thị**: paper dựng open KG bằng LLM OpenIE trên passage; ta dùng thẳng PrimeKG.
2. **phrase node vs passage node**: paper phân biệt hai loại; ở PrimeKG mỗi node VỪA là thực thể
   VỪA có document nên hai loại trùng nhau, đồ thị chỉ còn một tầng.
3. **query→triple**: paper truy hồi triple rồi lấy phrase node từ triple làm seed. Embed cả
   8.1M cạnh là không khả thi, nên ta lấy seed trực tiếp từ độ tương đồng query–node.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_transition(indptr, indices, n) -> sp.csr_matrix:
    """Ma trận chuyển column-stochastic P với P[u,v] = 1/deg(v) (bước ngẫu nhiên đồng đều)."""
    deg = np.diff(indptr).astype(np.float64)
    deg[deg == 0] = 1.0
    data = np.repeat(1.0 / deg, np.diff(indptr))
    A = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    return A.T.tocsr()          # cột v = phân phối ra từ v


def ppr(P: sp.csr_matrix, reset: np.ndarray, alpha: float = 0.5,
        iters: int = 40, tol: float = 1e-8) -> np.ndarray:
    """Power iteration: x = (1-α)·P·x + α·reset.

    alpha = xác suất quay về seed. alpha lớn -> bám sát seed (local hơn).
    """
    r = reset / max(reset.sum(), 1e-12)
    x = r.copy()
    for _ in range(iters):
        x_new = (1.0 - alpha) * (P @ x) + alpha * r
        if np.abs(x_new - x).sum() < tol:
            x = x_new
            break
        x = x_new
    return x


def topk_reset(scores: np.ndarray, k: int, temp: float = 1.0) -> np.ndarray:
    """Vector reset từ top-k node giống query nhất.

    HippoRAG 2 gán xác suất reset theo điểm truy hồi của seed, không phải đồng đều —
    nên giữ nguyên trọng số (chuẩn hoá) thay vì gán 1/k.
    """
    n = scores.shape[0]
    idx = np.argpartition(-scores, min(k, n - 1))[:k]
    v = scores[idx].astype(np.float64)
    v = v - v.min() + 1e-9          # đảm bảo dương, giữ thứ tự
    if temp != 1.0:
        v = v ** (1.0 / temp)
    r = np.zeros(n, dtype=np.float64)
    r[idx] = v / v.sum()
    return r
