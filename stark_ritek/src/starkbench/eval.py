"""Metric cho STaRK-Prime — khớp chính xác với `stark_qa.evaluator.Evaluator`.

Vì sao viết lại thay vì import thẳng stark_qa:
  - stark_qa.Evaluator tạo tensor kích thước (max(candidate_ids)+1) cho MỖI query.
    Với 2,801 query x 129,375 candidate, cách đó rất chậm. Bản này vector hoá bằng numpy.
  - Cần kiểm soát rõ ràng cách xử lý candidate KHÔNG được chấm điểm (xem `full_score`).
  - Cần xử lý tường minh cái bẫy dấu của torchmetrics (xem bên dưới).

=============================================================================
BẪY QUAN TRỌNG: torchmetrics lọc theo DẤU của điểm, không chỉ theo thứ hạng
=============================================================================
Đọc source torchmetrics 1.9:

  retrieval_hit_rate        -> KHÔNG lọc dấu. Chỉ phụ thuộc thứ hạng.        (Hit@1, Hit@5)
  retrieval_recall          -> CÓ  `torch.where(preds > 0, target, 0)`       (Recall@20)
  retrieval_reciprocal_rank -> CÓ  `torch.where(preds > 0, target, 0)`       (MRR)

Tức là **mọi node đáp án có điểm <= 0 đều bị coi là KHÔNG liên quan** khi tính
Recall@20 và MRR — kể cả khi nó đang đứng hạng 1.

Hệ quả cho benchmark này:
  - MRR không cap ở k, nên nếu TẤT CẢ đáp án có điểm <= 0 thì MRR = 0 dù xếp hạng hoàn hảo.
  - Phương pháp sinh điểm âm (cosine trên Contriever, logit của MLP/GNN như SubgraphRAG,
    GNN-RAG) sẽ bị tụt MRR/R@20 một cách giả tạo so với phương pháp sinh điểm dương (BM25).
  - Điểm đúng bằng 0 cũng bị loại, vì điều kiện là `preds > 0` (ngặt).

=> Quy tắc của benchmark: dùng `sanitize_scores()` để đưa điểm về (0, 1] bằng một phép
   biến đổi ĐƠN ĐIỆU trước khi chấm. Thứ hạng không đổi, nhưng loại bỏ hoàn toàn ảnh hưởng
   của thang điểm lên R@20/MRR, giúp các phương pháp so sánh được với nhau.
   Đặt `sanitize=False` nếu muốn tái hiện đúng hành vi gốc của stark_qa.

`scripts/test_parity.py` đối chiếu file này với stark_qa.Evaluator. Sửa file này thì phải
chạy lại parity test.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

# 4 metric gốc của STaRK + 2 metric bổ sung (xem docs/PLAN.md §3.3) + recall@10 (2026-10-15) + f2 (2026-10-27,
# thay f2@5). Mọi metric trừ f2 / f2@k đều có sẵn trong stark_qa.Evaluator -> không tự chế định nghĩa.
METRICS = ("hit@1", "hit@5", "recall@10", "recall@20", "mrr", "f2", "map@20", "rprecision")
METRICS_STARK = ("hit@1", "hit@5", "recall@20", "mrr")
# Bộ metric báo cáo (ritek/PLAN.md §3): chốt 2026-10-15, f2@5 -> f2 ngày 2026-10-27.
METRICS_REPORT = ("f2", "hit@1", "hit@5", "recall@10", "recall@20", "mrr")

# Metric áp dụng bộ lọc `preds > 0` trong torchmetrics: mrr, recall, map.
# KHÔNG lọc: hit_rate, r_precision, f2.

# F2 (yêu cầu 2026-10-13, không thuộc stark_qa): coi top-k là tập "nộp".
#   P = #đúng trong top-k / k      R = #đáp án trong top-k / #đáp án
#   F2 = 5·P·R / (4·P + R)  (= 0 khi P = R = 0); macro-average trên câu hỏi.
#   Tương đương F2 = 5·tp / (4·G + k)  (tp = #đáp án trong top-k, G = #đáp án đã khử trùng).
# Chỉ phụ thuộc thứ hạng (không lọc dấu điểm) -> tính được từ predictions top-1000 với k <= 1000.
# Trong `evaluate_one`, k = min(k, n_native): phương pháp tự trả ít hơn k node thì chỉ nộp chừng
# đó node — phần backfill KHÔNG tính vào F2 (chốt 2026-10-15).
# `f2` (2026-10-27, thay f2@5 trong bảng báo cáo): tập "nộp" = TOÀN BỘ m = n_native node phương pháp tự trả về
# (subgraph / tập ứng viên), F2 = 5·tp / (4·G + m); n_native = None (chấm mọi candidate) -> m = 129,375.
F2_KS = (5,)   # f2@k: chốt 2026-10-13 nộp top-5; còn dùng ở scripts/compare.py


def prf_at_k(hits: np.ndarray, n_gold: int, k: int):
    """hits: mảng bool theo thứ hạng (True = đáp án). Trả (P@k, R@k, F2@k) của MỘT câu hỏi."""
    c = float(np.count_nonzero(hits[:k]))
    p, r = c / k, c / n_gold
    return p, r, (5 * p * r / (4 * p + r) if p + r > 0 else 0.0)


def evaluate_one(
    scores: np.ndarray,
    answer_ids: Sequence[int],
    metrics: Iterable[str] = METRICS,
    n_native: int | None = None,
) -> Dict[str, float]:
    """Tính metric cho MỘT query, khớp hành vi torchmetrics (kể cả bộ lọc dấu).

    Args:
        scores: shape (n_candidates,) — điểm cho TOÀN BỘ candidate, chỉ số = node id.
        answer_ids: id các node đáp án.
        metrics: tập metric cần tính.
        n_native: số node do CHÍNH phương pháp trả về (đứng đầu bảng xếp hạng, trên phần
            backfill — `full_score` bảo đảm điều đó). None = phương pháp chấm đủ mọi node.
            Chỉ ảnh hưởng f2 / f2@k; các metric khác tính trên toàn bộ xếp hạng.
    """
    order = np.argsort(-scores, kind="stable")  # giảm dần, hoà -> index nhỏ trước
    gold = np.zeros(scores.shape[0], dtype=bool)
    gold[np.asarray(answer_ids, dtype=np.int64)] = True
    n_gold = int(gold.sum())

    ranked_gold = gold[order]
    # bản đã lọc dấu: đáp án có điểm <= 0 bị coi là không liên quan
    ranked_gold_pos = ranked_gold & (scores[order] > 0)

    out: Dict[str, float] = {}
    for m in metrics:
        if m == "mrr":
            hits = np.flatnonzero(ranked_gold_pos)
            out[m] = float(1.0 / (hits[0] + 1)) if hits.size else 0.0
        elif m.startswith("hit@"):
            k = int(m.split("@")[1])
            out[m] = float(ranked_gold[:k].any())  # KHÔNG lọc dấu
        elif m.startswith("recall@"):
            k = int(m.split("@")[1])
            out[m] = float(ranked_gold_pos[:k].sum() / n_gold) if n_gold else 0.0
        elif m.startswith("map@"):
            # Average Precision cắt ở top-k. Theo torchmetrics: chia cho SỐ ĐÁP ÁN TÌM
            # ĐƯỢC trong top-k, không phải tổng số đáp án.
            k = int(m.split("@")[1])
            top = ranked_gold_pos[:k]
            pos = np.flatnonzero(top) + 1  # vị trí 1-based
            out[m] = float((np.arange(1, pos.size + 1) / pos).mean()) if pos.size else 0.0
        elif m == "rprecision":
            # Precision@R với R = số đáp án của chính query đó. KHÔNG lọc dấu.
            out[m] = float(ranked_gold[:n_gold].sum() / n_gold) if n_gold else 0.0
        elif m == "f2":
            # F2 trên cả tập node phương pháp tự trả về: k = n_native (None = mọi candidate). KHÔNG lọc dấu.
            k = scores.shape[0] if n_native is None else int(n_native)
            tp = int(ranked_gold[:k].sum())
            out[m] = 5.0 * tp / (4 * n_gold + k) if n_gold + k else 0.0
        elif m.startswith("f2@"):
            # F2 = 5·tp / (4·G + k), k = min(k, n_native). KHÔNG lọc dấu (chỉ phụ thuộc thứ hạng).
            k = int(m.split("@")[1])
            if n_native is not None:
                k = min(k, int(n_native))
            tp = int(ranked_gold[:k].sum())
            out[m] = 5.0 * tp / (4 * n_gold + k) if n_gold + k else 0.0
        else:
            raise ValueError(f"metric chưa hỗ trợ: {m}")
    return out


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Đưa điểm về (0, 1] bằng phép biến đổi đơn điệu — thứ hạng giữ nguyên.

    Loại bỏ ảnh hưởng của thang/dấu điểm lên Recall@20 và MRR (xem docstring đầu file).
    Dùng thứ hạng thay vì min-max để tránh việc một outlier nén toàn bộ phần còn lại
    xuống sát 0 (điều đó vẫn hợp lệ về mặt dấu, nhưng dễ gây lỗi số học ngầm).
    """
    n = scores.shape[0]
    order = np.argsort(-scores, kind="stable")
    out = np.empty(n, dtype=np.float64)
    out[order] = np.linspace(1.0, 1.0 / (n + 1), num=n)  # hạng 1 -> 1.0, hạng cuối -> >0
    return out


def full_score(
    pred: Dict[int, float],
    n_candidates: int,
    backfill: np.ndarray | None = None,
) -> np.ndarray:
    """Bung dict {node_id: score} thưa thành vector điểm đầy đủ.

    ĐÂY LÀ ĐIỂM QUYẾT ĐỊNH TÍNH CÔNG BẰNG CỦA BENCHMARK. Xem docs/EDA.md §8.

    stark_qa gán mọi candidate không được chấm điểm bằng `min(pred) - 1` — tất cả bằng
    nhau. Với phương pháp chỉ sinh vài trăm candidate (traversal / diffusion / GNN),
    ~129k node còn lại đồng hạng ở sàn và bị phá hoà theo node id — khiến MRR/R@20 phụ
    thuộc thứ tự index thay vì chất lượng mô hình.

    Args:
        pred: điểm thưa do phương pháp sinh ra.
        n_candidates: 129,375 với STaRK-Prime.
        backfill: tuỳ chọn, shape (n_candidates,). Nếu có, candidate không được chấm điểm
            sẽ xếp theo thứ tự của `backfill` nhưng LUÔN nằm dưới mọi candidate đã chấm
            điểm. Đây là chiến lược đã khai báo, không phải hoà ngẫu nhiên.
            None -> tái hiện đúng hành vi stark_qa (sàn phẳng).
    """
    if not pred:
        raise ValueError("pred rỗng")
    ids = np.fromiter(pred.keys(), dtype=np.int64, count=len(pred))
    vals = np.fromiter(pred.values(), dtype=np.float64, count=len(pred))

    lo = float(vals.min())
    if backfill is None:
        scores = np.full(n_candidates, lo - 1.0, dtype=np.float64)
    else:
        b = np.asarray(backfill, dtype=np.float64)
        rng = b.max() - b.min()
        b = (b - b.min()) / rng if rng > 0 else np.zeros_like(b)
        scores = (lo - 2.0) + b  # thuộc [lo-2, lo-1], luôn dưới phần đã chấm điểm
    scores[ids] = vals
    return scores


def evaluate_all(
    preds: Dict[int, Dict[int, float]],
    answers: Dict[int, Sequence[int]],
    n_candidates: int,
    metrics: Iterable[str] = METRICS,
    backfill: np.ndarray | None = None,
    sanitize: bool = True,
) -> Dict[str, float]:
    """Macro-average trên toàn bộ query.

    Args:
        sanitize: True (mặc định của benchmark) -> chuẩn hoá điểm về (0,1] trước khi chấm,
            để R@20/MRR chỉ phụ thuộc thứ hạng. False -> tái hiện đúng stark_qa gốc.
    """
    metrics = list(metrics)
    acc: Dict[str, List[float]] = {m: [] for m in metrics}
    for qid, pred in preds.items():
        s = full_score(pred, n_candidates, backfill=backfill)
        if sanitize:
            s = sanitize_scores(s)
        r = evaluate_one(s, answers[qid], metrics, n_native=len(pred))
        for m in metrics:
            acc[m].append(r[m])
    return {m: float(np.mean(v)) for m, v in acc.items()}
