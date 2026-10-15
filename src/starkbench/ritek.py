"""Query STaRK-Prime bản RiTeK — cùng lược đồ với `runner.load_queries`, thêm topic entity.

RiTeK = STaRK-Prime + `topic_entity`. Đồ thị, text node, câu hỏi, đáp án và test giống hệt bản
gốc (ritek/docs/ritek_prime_compare.json), nên SKB/corpus vẫn nạp từ data/stark/prime.

Split (chốt 2026-10-15, ritek/PLAN.md §2):
  train · val  tách từ RiTeK train: val = 10% ngẫu nhiên, seed 42 (RiTeK không phát hành val;
               STaRK val không có topic nên nhóm phương pháp cần topic không dùng được)
  test         RiTeK test (= STaRK test, 2,801 query)
  test-0.1     đúng 280 id `test-0.1` của STaRK (tập con của test)
File split ghi sẵn ở ritek/splits/<split>.index (`make_splits`) để local và Kaggle dùng chung một bộ.

Trả về DataFrame: id · query · answer_ids · topic_ids · topic_names
  topic_ids rỗng ở 45 query (27 train, 18 test) — RiTeK không gán topic cho chúng.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ritek/Dataset/prime"
SPLITS = ROOT / "ritek/splits"
STARK_SPLITS = ROOT / "data/stark/prime/split"
VAL_FRAC = 0.1
SEED = 42
_SOURCE = {"train": "train", "val": "train", "test": "test", "test-0.1": "test"}


def _read(part: str) -> pd.DataFrame:
    rows = []
    for line in open(DATA / f"Prime_{part}_QA_with_topic_entity.json", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "id": int(r["question_id"]),
            "query": r["question"],
            "answer_ids": [int(a) for a in r["answer"]],
            "topic_ids": [int(t["topic_id"]) for t in r["topic_entity"]],
            "topic_names": [t["topic_name"] for t in r["topic_entity"]],
        })
    return pd.DataFrame(rows)


def make_splits(out: Path = SPLITS) -> dict:
    """Ghi ritek/splits/{train,val,test,test-0.1}.index. Tất định (seed 42)."""
    train = np.sort(_read("train")["id"].to_numpy())
    test = _read("test")["id"].to_numpy()
    perm = np.random.default_rng(SEED).permutation(train)
    n_val = int(round(VAL_FRAC * len(train)))
    stark01 = [int(x) for x in (STARK_SPLITS / "test-0.1.index").read_text().split()]
    assert set(stark01) <= set(test.tolist()), "test-0.1 của STaRK phải nằm trong RiTeK test"
    splits = {"train": np.sort(perm[n_val:]), "val": np.sort(perm[:n_val]),
              "test": np.sort(test), "test-0.1": np.sort(stark01)}
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / f"{name}.index").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")
    return {k: len(v) for k, v in splits.items()}


def load_queries(split: str) -> pd.DataFrame:
    if split not in _SOURCE:
        raise ValueError(f"split không hợp lệ: {split} (hợp lệ: {list(_SOURCE)})")
    f = SPLITS / f"{split}.index"
    if not f.exists():
        raise FileNotFoundError(f"thiếu {f} — chạy `python ritek/scripts/make_splits.py`")
    ids = {int(x) for x in f.read_text().split()}
    df = _read(_SOURCE[split])
    out = df[df["id"].isin(ids)].reset_index(drop=True)
    assert len(out) == len(ids), f"{split}: {len(ids) - len(out)} id trong file split không có trong RiTeK"
    out.attrs["dataset"] = "ritek"
    return out
