"""Cắt document thành chunk theo cửa sổ token, để không mất thông tin quan hệ.

Vì sao chunk thay vì cắt cụt: xem docs/PLAN.md §4.2. Đo trên corpus thật, encoder 512 token
cắt cụt sẽ mất 64% node đáp án; chunking giữ 100% và chỉ tốn ~10 phút T4x2.

Biên chunk phụ thuộc tokenizer của từng model, nên PHẢI chunk bằng đúng tokenizer của encoder
sẽ dùng — không chunk sẵn theo ký tự.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def chunk_token_ids(ids: Sequence[int], size: int, stride: int) -> List[List[int]]:
    """Cắt một dãy token id thành các cửa sổ."""
    n = len(ids)
    if n <= size:
        return [list(ids)]
    out = []
    i = 0
    while i < n:
        out.append(list(ids[i:i + size]))
        if i + size >= n:
            break
        i += stride
    return out


def build_chunks(docs: Sequence[str], tokenizer, size: int = 512, stride: int = 512,
                 batch: int = 2000, verbose: bool = True) -> Tuple[List[List[int]], np.ndarray]:
    """Tokenize + cắt chunk cho toàn bộ corpus.

    Trả về:
        chunks   : list các dãy token id (chưa có special token)
        chunk2node: mảng int32, chunk2node[i] = node id của chunk i
    """
    chunks: List[List[int]] = []
    owner: List[int] = []
    n = len(docs)
    # Trừ 2 cho [CLS]/[SEP] sẽ được thêm khi encode.
    body = size - 2
    for s in range(0, n, batch):
        part = docs[s:s + batch]
        enc = tokenizer(list(part), add_special_tokens=False, truncation=False)["input_ids"]
        for local, ids in enumerate(enc):
            for c in chunk_token_ids(ids, body, body if stride == size else stride - 2):
                chunks.append(c)
                owner.append(s + local)
        if verbose and (s + batch) % 20000 < batch:
            print(f"  chunk: {min(s+batch, n):,}/{n:,} doc -> {len(chunks):,} chunk", flush=True)
    return chunks, np.asarray(owner, dtype=np.int32)


def pool_scores(sim: np.ndarray, chunk2node: np.ndarray, n_nodes: int,
                mode: str = "top3_mean") -> np.ndarray:
    """Gộp điểm chunk -> điểm node.

    Args:
        sim: (n_queries, n_chunks) điểm tương đồng.
        chunk2node: (n_chunks,) node id của từng chunk, **đã sắp tăng dần**
            (build_chunks sinh chunk theo thứ tự node nên điều này luôn đúng).
        mode: 'max' hoặc 'top3_mean'.
            top3_mean = trung bình 3 chunk giống nhất, đúng cách STaRK dùng cho
            multi-ada-002 (giảm thiên vị document có nhiều chunk — xem PLAN §4.2e).

    Trả về: (n_queries, n_nodes), node không có chunk nào -> -inf.

    Vectorised: dùng `np.maximum.reduceat` / `np.add.reduceat` cho toàn bộ, rồi chỉ xử lý
    riêng nhóm node có >3 chunk bằng vòng lặp (trung bình 1.59 chunk/node nên nhóm này rất nhỏ).
    Vòng lặp 129K node bằng Python sẽ chậm hơn hàng trăm lần.
    """
    assert np.all(np.diff(chunk2node) >= 0), "chunk2node phải sắp tăng dần"
    nq = sim.shape[0]
    starts = np.flatnonzero(np.r_[True, chunk2node[1:] != chunk2node[:-1]])
    nodes = chunk2node[starts].astype(np.int64)
    sizes = np.diff(np.r_[starts, len(chunk2node)])

    out = np.full((nq, n_nodes), -np.inf, dtype=np.float32)
    if mode == "max":
        out[:, nodes] = np.maximum.reduceat(sim, starts, axis=1)
        return out

    # top3_mean: mặc định = mean của toàn bộ chunk (đúng khi size <= 3)
    out[:, nodes] = np.add.reduceat(sim, starts, axis=1) / sizes[None, :]
    big = np.flatnonzero(sizes > 3)
    for j in big:                      # chỉ nhóm >3 chunk, rất ít
        st = starts[j]
        block = sim[:, st:st + sizes[j]]
        out[:, nodes[j]] = np.partition(block, -3, axis=1)[:, -3:].mean(axis=1)
    return out
