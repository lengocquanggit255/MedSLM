"""Thống kê subgraph của GNN-RAG và SubgraphRAG trên RiTeK test: avg node · avg edge · avg answer recall ·
full recall rate.

Chạy: python ritek/scripts/subgraph_stats.py

Định nghĩa như scripts/subgraph_stats.py (lần chạy STaRK), gold = bản RiTeK (khác STaRK):
  answer_recall_q = |A ∩ S| / |A| · full_recall_q = 1 nếu A ⊆ S · A = tập đáp án (đã khử trùng lặp).
  avg node = |S| trung bình · avg edge = số cạnh cảm sinh (cả hai đầu trong S) trên KG Prime 8,100,498 cạnh,
  đúng tập triple SubgraphRAG chấm (kiểm khớp với SubgraphStore.subgraph trên vài query).
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
GNN_RUNS = [("GNN-RAG (MiniLM, bản chính)", "ritek-gnnrag-rearev-s6"),
            ("GNN-RAG + BGE-M3", "ritek-gnnrag-bgem3-s5")]


def count_edges(nodes, indptr, indices, mark):
    """Số cạnh cảm sinh trong `nodes` (mark[v] = True với v ∈ nodes), vector hoá."""
    starts, cnt = indptr[nodes], indptr[nodes + 1] - indptr[nodes]
    tot = int(cnt.sum())
    if tot == 0:
        return 0
    base = np.repeat(starts, cnt)
    offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return int(mark[indices[base + offs]].sum())


def row(name, sets, gold, n_all, edges=None, indptr=None, indices=None, n_nodes=None):
    """sets: {qid: mảng node}; edges: số cạnh cố định (toàn KG) hoặc None để đếm cạnh cảm sinh."""
    nv, ne, rec, full = [], [], [], []
    mark = np.zeros(n_nodes or 1, dtype=bool)
    for q, nodes in sets.items():
        if len(nodes) == 0:
            continue
        nv.append(len(nodes))
        if edges is None:
            mark[nodes] = True
            ne.append(count_edges(nodes, indptr, indices, mark))
            mark[nodes] = False
        a = gold[q]
        hit = int(np.isin(a, nodes).sum())
        rec.append(hit / len(a))
        full.append(hit == len(a))
    k = len(nv)
    return dict(name=name, valid=k, node=np.mean(nv), edge=edges if edges is not None else np.mean(ne),
                rec=np.mean(rec), full=np.mean(full), rec_all=np.sum(rec) / n_all, full_all=np.sum(full) / n_all)


def main():
    q = load_queries("data/stark/prime", "test", dataset="ritek")
    gold = {int(r["id"]): np.unique(np.asarray(r["answer_ids"], dtype=np.int64)) for _, r in q.iterrows()}
    topic = {int(r["id"]) for _, r in q.iterrows() if r["topic_ids"]}
    N = len(q)
    c = np.load(DS / "csr.npz")
    indptr, indices, V = c["indptr"], c["indices"].astype(np.int64), int(c["n_nodes"])
    st = SubgraphStore(DS / "test.npz", indptr, indices, c["etype"].astype(np.int64), V)
    sg = {int(st.qids[i]): st.nodes(i).astype(np.int64) for i in range(len(st))}
    mark = np.zeros(V, dtype=bool)
    for i in [i for i in range(len(st)) if len(st.nodes(i))][:3]:      # cùng định nghĩa cạnh với model
        nodes = st.nodes(i).astype(np.int64)
        mark[nodes] = True
        assert count_edges(nodes, indptr, indices, mark) == len(st.subgraph(i)[1])
        mark[nodes] = False
    print(f"test: {N:,} query · {len(topic):,} có topic · KG {V:,} node, {len(indices):,} cạnh", flush=True)

    rows = [row("SubgraphRAG (PPR-2000 từ topic)", sg, gold, N, indptr=indptr, indices=indices, n_nodes=V)]
    allv = np.arange(V)
    for name, run in GNN_RUNS:
        d = ROOT / "ritek/outputs/kaggle" / run
        if not (d / "rearev_test_scores.npy").exists():
            print(f"  bỏ qua {name}: chưa có {run}")
            continue
        rows.append(row(f"{name} — đầu vào: toàn KG", {qq: allv for qq in topic}, gold, N, edges=len(indices)))
        S = np.load(d / "rearev_test_scores.npy", mmap_mode="r")
        qi = np.load(d / "rearev_test_qids.npy")
        nz = {int(qq): np.flatnonzero(np.asarray(S[i]) != 0) for i, qq in enumerate(qi)}
        rows.append(row(f"{name} — điểm khác 0", nz, gold, N, edges=float("nan")))

    print(f"\n| Phương pháp | Q_valid | avg node | avg edge | avg answer recall | full recall rate | "
          f"answer recall /{N:,} | full recall /{N:,} |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        e = "—" if r["edge"] != r["edge"] else f"{r['edge']:,.0f}"
        print(f"| {r['name']} | {r['valid']:,} | {r['node']:,.0f} | {e} | {r['rec'] * 100:.2f}% | {r['full'] * 100:.2f}% | "
              f"{r['rec_all'] * 100:.2f}% | {r['full_all'] * 100:.2f}% |")


if __name__ == "__main__":
    main()
