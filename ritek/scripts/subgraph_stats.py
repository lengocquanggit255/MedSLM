"""Thống kê subgraph của GNN-RAG và SubgraphRAG trên RiTeK test: avg node · avg edge · avg answer recall ·
full recall rate.

Chạy: python ritek/scripts/subgraph_stats.py

Định nghĩa theo notebook QAFD tham chiếu (ref_qafd/qafd-rag.ipynb, mục 8) để mọi dòng của bảng — kể cả QAFD —
so được với nhau; gold = bản RiTeK (khác STaRK):
  answer_recall_q = |A ∩ S| / |A| · full_recall_q = 1 nếu A ⊆ S · A = tập đáp án (đã khử trùng lặp).
  avg node = |S| trung bình.
  avg edge = số cạnh có CẢ HAI đầu trong S trên đồ thị VÔ HƯỚNG đã gộp (mỗi cặp node một cạnh, như nx.Graph
  dựng từ prime_KG.txt của notebook): 8,100,498 dòng (h, r, t) -> 4,049,642 cặp, không có tự vòng.
  (SubgraphRAG thực tế chấm theo triple có hướng nhiều quan hệ — số đó khác, không dùng ở đây.)
  Mẫu số: Q_valid = query có S khác rỗng (bỏ 18 query test không có topic); in thêm bản chia cho đủ 2,801
  (query rỗng tính recall 0).
S của từng dòng:
  - SubgraphRAG: subgraph PPR-2000 từ topic RiTeK (ritek-sgrag-data/test.npz), mọi node đều được chấm.
  - GNN-RAG "đầu vào": toàn KG (129,375 node) cho mọi query có topic.
  - GNN-RAG "điểm khác 0": node có pred_dist ≠ 0 (rearev_test_scores.npy, trước backfill) — đo, không giả định.
"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.gnnrag import SubgraphStore  # noqa: E402
from starkbench.runner import load_queries  # noqa: E402

DS = ROOT / "ritek/outputs/kaggle_data/ritek-sgrag-data"
GNN_RUNS = [("GNN-RAG (MiniLM, bản chính)", "ritek-gnnrag-rearev-s6")]


def undirected_pairs(indptr, indices, n_nodes):
    """Cặp node (u ≤ v) duy nhất từ CSR có hướng — đúng tập cạnh của nx.Graph dựng từ file triple."""
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    lo, hi = np.minimum(src, indices), np.maximum(src, indices)
    key = np.unique(lo * n_nodes + hi)
    return key // n_nodes, key % n_nodes


def row(name, sets, gold, n_all, pairs, n_nodes):
    """sets: {qid: mảng node}. Cạnh = cặp vô hướng có cả hai đầu trong tập node."""
    eu, ev = pairs
    mark = np.zeros(n_nodes, dtype=bool)
    nv, ne, rec, full = [], [], [], []
    for q, nodes in sets.items():
        if len(nodes) == 0:
            continue
        nv.append(len(nodes))
        mark[nodes] = True
        ne.append(int((mark[eu] & mark[ev]).sum()))
        mark[nodes] = False
        a = gold[q]
        hit = int(np.isin(a, nodes).sum())
        rec.append(hit / len(a))
        full.append(hit == len(a))
    return dict(name=name, valid=len(nv), node=np.mean(nv), edge=np.mean(ne), rec=np.mean(rec), full=np.mean(full),
                rec_all=np.sum(rec) / n_all, full_all=np.sum(full) / n_all)


def main():
    q = load_queries("data/stark/prime", "test", dataset="ritek")
    gold = {int(r["id"]): np.unique(np.asarray(r["answer_ids"], dtype=np.int64)) for _, r in q.iterrows()}
    topic = {int(r["id"]) for _, r in q.iterrows() if r["topic_ids"]}
    N = len(q)
    c = np.load(DS / "csr.npz")
    indptr, indices, V = c["indptr"], c["indices"].astype(np.int64), int(c["n_nodes"])
    pairs = undirected_pairs(indptr, indices, V)
    st = SubgraphStore(DS / "test.npz", indptr, indices, c["etype"].astype(np.int64), V)
    sg = {int(st.qids[i]): st.nodes(i).astype(np.int64) for i in range(len(st))}
    print(f"test: {N:,} query · {len(topic):,} có topic · KG {V:,} node, {len(indices):,} cạnh có hướng "
          f"-> {len(pairs[0]):,} cặp vô hướng", flush=True)

    rows = [row("SubgraphRAG (PPR-2000 từ topic)", sg, gold, N, pairs, V)]
    allv = np.arange(V)
    for name, run in GNN_RUNS:
        d = ROOT / "ritek/outputs/kaggle" / run
        if not (d / "rearev_test_scores.npy").exists():
            print(f"  bỏ qua {name}: chưa có {run}")
            continue
        rows.append(row(f"{name} — đầu vào: toàn KG", {qq: allv for qq in topic}, gold, N, pairs, V))
        S = np.load(d / "rearev_test_scores.npy", mmap_mode="r")
        qi = np.load(d / "rearev_test_qids.npy")
        nz = {int(qq): np.flatnonzero(np.asarray(S[i]) != 0) for i, qq in enumerate(qi)}
        rows.append(row(f"{name} — điểm khác 0", nz, gold, N, pairs, V))

    print(f"\n| Phương pháp | Q_valid | avg node | avg edge | avg answer recall | full recall rate | "
          f"answer recall /{N:,} | full recall /{N:,} |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['name']} | {r['valid']:,} | {r['node']:,.0f} | {r['edge']:,.0f} | {r['rec'] * 100:.2f}% | "
              f"{r['full'] * 100:.2f}% | {r['rec_all'] * 100:.2f}% | {r['full_all'] * 100:.2f}% |")


if __name__ == "__main__":
    main()
