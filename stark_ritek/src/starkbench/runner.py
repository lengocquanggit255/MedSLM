"""Hợp đồng chung cho kết quả: nạp query, chấm điểm, ghi artefact.

Mọi phương pháp phải đi qua đây để bảng kết quả so sánh được với nhau.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .eval import METRICS, evaluate_one, sanitize_scores

TOPK_STORE = 1000  # số candidate lưu vào predictions để soi lại; metric tính trên FULL ranking


def load_queries(root: str | Path, split: str, dataset: str = "stark") -> pd.DataFrame:
    """Nạp query của một split. Trả về DataFrame [id, query, answer_ids].

    dataset="ritek": query bản RiTeK (thêm topic_ids, topic_names), bỏ qua `root` — xem
    `starkbench.ritek`.
    """
    if dataset == "ritek":
        from .ritek import load_queries as load_ritek
        return load_ritek(split)
    if dataset != "stark":
        raise ValueError(f"dataset không hợp lệ: {dataset}")
    root = Path(root)
    qa = pd.read_csv(root / "stark_qa" / "stark_qa.csv")
    qa["answer_ids"] = qa["answer_ids"].apply(eval)
    if split == "human":
        h = pd.read_csv(root / "stark_qa" / "stark_qa_human_generated_eval.csv")
        h["answer_ids"] = h["answer_ids"].apply(eval)
        h.attrs["dataset"] = "stark"
        return h
    ids = [int(x) for x in (root / "split" / f"{split}.index").read_text().split()]
    out = qa[qa["id"].isin(set(ids))].reset_index(drop=True)
    out.attrs["dataset"] = "stark"
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def _breakdown_masks(qdf: pd.DataFrame, skb) -> Dict[str, np.ndarray]:
    """Mặt nạ cho các nhóm phân tích (docs/EDA.md §4).

    Trường text khác nhau giữa hai bộ: Prime lồng trong `details`, ADint phẳng ở
    `description`. Chỉ đọc `details` thì mọi node ADint thành "không có text" và cả hai
    nhóm breakdown đều vô nghĩa (một nhóm rỗng, một nhóm là toàn bộ).
    """
    def _txt(i):
        v = skb.node_info[i]
        return str(v.get("details", "") or "") + str(v.get("description", "") or "")
    has_text = np.array([len(_txt(i)) >= 50 for i in range(skb.num_nodes)])
    ans = list(qdf["answer_ids"])
    n = len(ans)
    m = {
        "all": np.ones(n, dtype=bool),
        "answers_with_text": np.array([all(has_text[i] for i in a) for a in ans]),
        "answers_without_text": np.array([not any(has_text[i] for i in a) for a in ans]),
        "single_answer": np.array([len(a) == 1 for a in ans]),
        "multi_answer": np.array([len(a) > 1 for a in ans]),
    }
    # ADint có thêm hai nhóm PHẢI tách ra, nếu không số chính bị thổi lên bởi câu dễ:
    #   - đáp án CHÍNH LÀ topic entity (33 query, 0 hop)
    #   - câu hỏi gọi thẳng tên đáp án (train 4,7% · test 5,0% · val 3,3%)
    if "topic_ids" in qdf.columns:
        m["answer_is_topic"] = np.array(
            [bool(set(a) & set(t)) for a, t in zip(ans, qdf["topic_ids"])])
        m["answer_not_topic"] = ~m["answer_is_topic"]
    return m


def evaluate_and_save(
    scores_fn,
    qdf: pd.DataFrame,
    skb,
    method: str,
    variant: str,
    split: str,
    results_root: str | Path = "results",
    config: dict | None = None,
    metrics: Sequence[str] = METRICS,
    notes: str = "",
) -> Dict[str, float]:
    """Chạy `scores_fn(row)` cho từng query rồi ghi kết quả.

    `scores_fn(row)` trả về `np.ndarray (n_candidates,)`, hoặc `(scores, n_native)` với phương
    pháp thưa: `n_native` = số node do chính phương pháp trả về, đứng trên phần backfill
    (`eval.full_score`). n_native quyết định k của f2@k và cột `native@k`.

    Metric được tính trên **full ranking** (đủ 129,375 node). `predictions.parquet` chỉ
    lưu top-1000 để soi lại, không dùng để tính metric.

    Báo cáo cả hai biến thể metric:
      - `sanitized` : chuẩn hoá điểm về (0,1] trước khi chấm (mặc định benchmark)
      - `raw`       : đúng hành vi stark_qa gốc (điểm <= 0 bị loại khỏi recall/mrr/map)
    """
    metrics = list(metrics)
    out = Path(results_root) / method / variant / split
    out.mkdir(parents=True, exist_ok=True)

    per_query_san: List[Dict[str, float]] = []
    per_query_raw: List[Dict[str, float]] = []
    n_native_all: List[int] = []
    rows = []
    t0 = time.time()
    for _, r in qdf.iterrows():
        s = scores_fn(r)
        s, n_nat = s if isinstance(s, tuple) else (s, None)
        n_native_all.append(s.shape[0] if n_nat is None else int(n_nat))
        per_query_raw.append(evaluate_one(s, r["answer_ids"], metrics, n_native=n_nat))
        per_query_san.append(evaluate_one(sanitize_scores(s), r["answer_ids"], metrics, n_native=n_nat))
        k = min(TOPK_STORE, s.shape[0])
        # PHẢI dùng đúng cùng một phép sắp xếp như `evaluate_one`: argsort ỔN ĐỊNH trên
        # TOÀN BỘ vector điểm (hoà điểm -> node id nhỏ đứng trước).
        # Không được dùng argpartition + sort cục bộ: khi nhiều candidate hoà điểm,
        # argpartition chọn tuỳ ý trong nhóm hoà nên thứ tự lưu ra lệch khỏi thứ tự dùng
        # để tính metric. Biến thể name_only (document ngắn, hoà điểm nhiều) lệch rõ:
        # hit@1 0.0643 khi tính lại từ parquet vs 0.0607 trong metrics.json.
        top = np.argsort(-s, kind="stable")[:k]
        rows.append(pd.DataFrame({"query_id": int(r["id"]), "rank": np.arange(1, k + 1),
                                  "node_id": top.astype(np.int64), "score": s[top]}))
    elapsed = time.time() - t0

    pd.concat(rows, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    # Metric theo từng query (sanitized, full ranking) + n_native: để cắt tập con (vd test-0.1 từ
    # test) và kiểm định cặp mà không phải tính lại từ top-1000.
    pq = pd.DataFrame(per_query_san)
    pq.insert(0, "query_id", qdf["id"].astype(int).to_numpy())
    pq["n_native"] = n_native_all
    pq.to_parquet(out / "per_query.parquet", index=False)

    def agg(pq, mask=None):
        idx = np.arange(len(pq)) if mask is None else np.flatnonzero(mask)
        if len(idx) == 0:
            return {m: None for m in metrics}
        return {m: float(np.mean([pq[i][m] for i in idx])) for m in metrics}

    masks = _breakdown_masks(qdf, skb)
    res = {
        "method": method, "variant": variant, "split": split,
        "n_queries": int(len(qdf)),
        "sanitized": agg(per_query_san),
        "raw_stark_qa": agg(per_query_raw),
        "breakdown_sanitized": {k: {**agg(per_query_san, v), "n": int(v.sum())}
                                for k, v in masks.items()},
        # Tỉ lệ vị trí top-k do CHÍNH phương pháp đưa ra (phần còn lại là backfill).
        "native": {"mean_n_native": float(np.mean(n_native_all)),
                   **{f"native@{k}": float(np.mean([min(k, n) / k for n in n_native_all]))
                      for k in (5, 10, 20)}},
        "runtime_sec": round(elapsed, 2),
        "sec_per_query": round(elapsed / max(len(qdf), 1), 4),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    manifest = {
        "method": method, "variant": variant, "split": split,
        "dataset": qdf.attrs.get("dataset", "unknown"),   # loader gắn nhãn, xem load_queries
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_candidates": int(skb.num_nodes),
        "full_ranking": True,
        "topk_stored": TOPK_STORE,
        "runtime_sec": round(elapsed, 2),
        "config": config or {},
        "notes": notes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return res
