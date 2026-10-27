"""Bổ sung metric `f2` (F2 trên CẢ tập node phương pháp tự trả về; src/starkbench/eval.py, quyết định 2026-10-27,
thay f2@5) cho các kết quả RiTeK đã có — không chạy lại phương pháp.

Chạy: python ritek/scripts/recompute_f2.py [--check-only]

f2 = 5·tp / (4·G + m), m = n_native (per_query.parquet), tp = số đáp án trong m node đó (G = số đáp án đã khử trùng).
Dựng lại tập m node (chúng đứng trên phần backfill trong xếp hạng):
  m = 129,375 (chấm mọi node) -> tp = G · m = 0 -> f2 = 0
  m ≤ 1,000                  -> top-m của predictions.parquet (cùng thứ tự ổn định với metric)
  subgraphrag/*              -> đúng subgraph PPR-2000 (ritek-sgrag-data/test.npz); kiểm |subgraph| = m từng câu
  còn lại                    -> dừng, phải chấm lại bằng harness
Kiểm trước khi ghi: cùng cách dựng tập, tính lại f2@5 từng câu phải trùng f2@5 đã lưu (sai số < 1e-12).
BM25 không xử lý ở đây: chấm lại bằng scripts/run_lexical.py (n_native = số tài liệu có điểm > 0).
Ghi: per_query.parquet cột f2 · metrics.json (sanitized, raw_stark_qa, breakdown_sanitized) · manifest.json ghi chú.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.gnnrag import SubgraphStore  # noqa: E402
from starkbench.runner import _breakdown_masks, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

RES = ROOT / "ritek/results"
SKIP = {"bm25"}                      # chấm lại bằng harness
N_CAND = 129375


def subgraph_nodes():
    ds = ROOT / "ritek/outputs/kaggle_data/ritek-sgrag-data"
    c = np.load(ds / "csr.npz")
    st = SubgraphStore(ds / "test.npz", c["indptr"], c["indices"], c["etype"].astype(np.int64), int(c["n_nodes"]))
    return {int(st.qids[i]): set(st.nodes(i).tolist()) for i in range(len(st))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    qdf = load_queries("data/stark/prime", "test", dataset="ritek")
    gold = {int(r["id"]): set(int(x) for x in r["answer_ids"]) for _, r in qdf.iterrows()}
    masks = _breakdown_masks(qdf, PrimeSKB("data/stark/prime"))
    sg = None
    for d in sorted(RES.glob("*/*/test")):
        method = d.parts[-3]
        if method in SKIP or not (d / "per_query.parquet").exists():
            continue
        pq = pd.read_parquet(d / "per_query.parquet")
        assert list(pq["query_id"]) == [int(x) for x in qdf["id"]], f"{d}: thứ tự query khác qdf"
        p = pd.read_parquet(d / "predictions.parquet").sort_values(["query_id", "rank"])
        pred = {int(q): g["node_id"].to_numpy() for q, g in p.groupby("query_id")}
        if method == "subgraphrag":
            sg = sg or subgraph_nodes()
        f2, f25 = [], []
        for q, m in zip(pq["query_id"], pq["n_native"]):
            G = gold[int(q)]
            if m == 0:
                native = []
            elif m == N_CAND:
                native = None
            elif m <= 1000:
                native = pred[int(q)][:m]
            elif method == "subgraphrag":
                native = sg[int(q)]
                assert len(native) == m, f"{d} q{q}: subgraph {len(native)} node ≠ n_native {m}"
            else:
                sys.exit(f"{d} q{q}: n_native {m} không dựng lại được -> chấm lại bằng harness")
            tp = len(G) if native is None else len(G & set(int(x) for x in native))
            f2.append(5 * tp / (4 * len(G) + m))
            k5 = min(5, m)                        # kiểm thứ tự/tập native: tái tạo f2@5 đã lưu
            f25.append(5 * len(G & set(pred[int(q)][:k5].tolist())) / (4 * len(G) + k5) if k5 else 0.0)
        f25 = np.array(f25)
        dmax = float(np.abs(f25 - pq["f2@5"].to_numpy()).max())
        if dmax > 1e-12:
            sys.exit(f"{d}: tái tạo f2@5 lệch {dmax:.2e} -> không ghi")
        f2 = np.array(f2)
        print(f"{method}/{d.parts[-2]:<34} f2 {f2.mean() * 100:7.3f}% · m tb {pq['n_native'].mean():>9,.0f} · "
              f"kiểm f2@5 {len(f25)} câu, lệch tối đa {dmax:.1e}", flush=True)
        if a.check_only:
            continue
        pq["f2"] = f2
        pq.to_parquet(d / "per_query.parquet", index=False)
        mj = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        mj["sanitized"]["f2"] = mj["raw_stark_qa"]["f2"] = float(f2.mean())   # f2 không phụ thuộc dấu điểm
        for g, mask in masks.items():
            if g in mj["breakdown_sanitized"]:
                mj["breakdown_sanitized"][g]["f2"] = float(f2[mask].mean()) if mask.any() else None
        if "all" in mj["breakdown_sanitized"]:
            mj["breakdown_sanitized"]["all"]["f2"] = float(f2.mean())
        (d / "metrics.json").write_text(json.dumps(mj, indent=2, ensure_ascii=False), encoding="utf-8")
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        man["f2_added"] = "2026-10-27 ritek/scripts/recompute_f2.py (tập n_native dựng lại, f2@5 tái tạo khớp)"
        (d / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
