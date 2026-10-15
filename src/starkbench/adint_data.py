"""Nạp query ADint — 3 split, mỗi split một nguồn khác nhau vì dữ liệu không đồng nhất.

| split | topic entity | nguồn |
|-------|--------------|-------|
| train | gold (94,0%) | `Dataset/Adint/ADint_train_with_topic_entity.json` |
| val   | 🔴 KHÔNG có  | `adint_skb/ADint/ADint_dev_QA.json`  ← KHÔNG dùng file trong Dataset/ |
| test  | gold (94,6%) | `Dataset/Adint/ADint_test_with_topic_entity.json` |

🔴 `Dataset/Adint/Adint_dev_with_topic_entity.json` là BẢN SAO BIT-ĐỐI-BIT của
`ADint_dev_QA_instruction.json` (md5 trùng) — lỗi đóng gói của tác giả. Nó không có
`answer` lẫn `topic_entity`. File đúng nằm trong gói metadata.

Trả về DataFrame cùng schema với `starkbench.runner.load_queries`:
    id · query · answer_ids        (+ topic_ids, topic_names)
"""
from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QA_DIR = ROOT / "data/ritek/Dataset/Adint"
SKB_DIR = ROOT / "data/ritek/adint_skb/ADint"
MAX_NGRAM = 8

SRC = {
    "train": (QA_DIR / "ADint_train_with_topic_entity.json", True),
    "val":   (SKB_DIR / "ADint_dev_QA.json", False),      # không có topic entity
    "test":  (QA_DIR / "ADint_test_with_topic_entity.json", True),
}


def _jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def topics_of(x: dict) -> List[dict]:
    """Chuẩn hoá `topic_entity` — dữ liệu có 2 dạng.

    `{"topic_name":…, "topic_id":…}`  94,0% train / 94,6% test
    `"gstt1"` chuỗi TRẦN, MẤT id       6,0% train / 5,4% test
    Dạng chuỗi trả `topic_id=None` — KHÔNG im lặng bỏ qua.
    """
    te = x.get("topic_entity")
    if te is None:
        return []
    if isinstance(te, str):
        return [{"topic_name": te, "topic_id": None}]
    if isinstance(te, dict):
        return [te]
    return [d if isinstance(d, dict) else {"topic_name": d, "topic_id": None} for d in te]


def build_alias_index(skb, min_len: int = 3) -> Dict[str, List[int]]:
    """alias đã chuẩn hoá -> [node id]. Tên ADint là chuỗi đồng nghĩa ngăn bằng `|`.

    Đo trên 4.084 query có gold topic: gold nằm trong tập khớp 94,6% train / 95,2% test,
    cỡ tập khớp p50=3. Xem docs/EDA_ADINT.md §3.
    """
    idx = defaultdict(list)
    for i in range(skb.num_nodes):
        for a in str(skb.name_of(i)).split("|"):
            a = norm(a)
            if len(a) >= min_len:
                idx[a].append(i)
    return dict(idx)


def link_entities(query: str, alias_idx: Dict[str, List[int]],
                  max_ngram: int = MAX_NGRAM) -> List[int]:
    """Mọi node có alias xuất hiện như n-gram liên tiếp trong câu hỏi."""
    toks = norm(query).split()
    found: set[int] = set()
    for L in range(min(max_ngram, len(toks)), 0, -1):
        for i in range(len(toks) - L + 1):
            ids = alias_idx.get(" ".join(toks[i:i + L]))
            if ids:
                found.update(ids)
    return sorted(found)


def load_queries(split: str) -> pd.DataFrame:
    path, has_topic = SRC[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} — chạy phần tải metadata trong docs/EDA_ADINT.md §0 trước")
    rows = []
    for x in _jsonl(path):
        t = topics_of(x) if has_topic else []
        rows.append({
            "id": int(x["question_id"]),
            "query": str(x["question"]).strip().strip('"'),
            "answer_ids": [int(k) for k in x["answer"]],
            "answer_names": [str(v) for v in x["answer"].values()],
            "topic_ids": [int(d["topic_id"]) for d in t if d.get("topic_id") is not None],
            "topic_names": [str(d["topic_name"]) for d in t],
        })
    df = pd.DataFrame(rows)
    assert df["id"].is_unique, f"{split}: question_id trùng"
    return df
