"""S1 — tính nhãn SubgraphRAG: triple nằm trên ĐƯỜNG ĐI NGẮN NHẤT giữa topic và đáp án.

Paper Eq 2: `G̃_q` = shortest paths giữa topic entities và answer entities. Ngữ nghĩa lấy ĐÚNG code gốc
`retrieve/src/dataset/retriever.py` @ Graph-COM/SubgraphRAG 48eb05e (`_extract_paths_and_score`):
  - đồ thị `nx.DiGraph`: nhiều triple cùng cặp (h, t) thì chỉ triple thêm SAU CÙNG được gán nhãn
    (mọi triple vẫn vào mô hình, chỉ nhãn bị gộp);
  - với TỪNG cặp (topic q, đáp án a): mọi đường ngắn nhất q -> a VÀ a -> q; nếu cả hai chiều có đường
    thì chỉ giữ đường có độ dài nhỏ nhất; KHÔNG giới hạn số bước;
  - `triple_scores[triple_id] = 1.` cho mọi triple trên các đường giữ lại, còn lại 0;
  - `max_path_length` = độ dài đường dài nhất; None/0 -> query bị bỏ khi train/val (`skip_no_path`).
Prime lưu mỗi cạnh theo CẢ HAI chiều, nên chiều a -> q gán dương cả triple ngược của đường q -> a.
(Bản trước 2026-10-17 chỉ lấy chiều q -> a, BFS chung mọi seed, ≤ 4 bước: thiếu đúng một nửa nhãn gốc.)
`orig_triples` tính bằng BFS (cạnh (u, v) nằm trên đường ngắn nhất x -> y khi d(x,u) + 1 + d(v,y) = d(x,y)),
cho đúng tập triple như liệt kê `nx.all_shortest_paths`; kiểm với chính code gốc trong
ritek/scripts/test_sgrag_equivalence.py.

⚠️ LƯU NHÃN THEO ID TOÀN CỤC `(h_global, rel, t_global)`, KHÔNG theo chỉ số cạnh.
Lý do: chỉ số cạnh phụ thuộc thứ tự trích subgraph; lưu theo id toàn cục thì ghép lại luôn đúng.
(Đây đúng loại bẫy đã làm hỏng MoR: file dự đoán không mang id, tôi giả định thứ tự, và sai ở
2,797/2,801 query.)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starkbench.gnnrag import SubgraphStore  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402


def orig_triples(n_v, src, dst, seeds, answers):
    """Nhãn theo code gốc trên subgraph cạnh (src[i] -> dst[i]), chỉ số cục bộ.

    Trả (tập chỉ số cạnh dương, max_path_length — None nếu không có đường nào).
    """
    if not len(src) or not seeds or not answers:
        return set(), None
    key = src.astype(np.int64) * n_v + dst
    _, last_rev = np.unique(key[::-1], return_index=True)
    keep = np.sort(len(key) - 1 - last_rev)        # DiGraph.add_edge lần lượt: cặp trùng giữ triple cuối
    us, vs = src[keep], dst[keep]
    G = sp.csr_matrix((np.ones(len(keep)), (us, vs)), shape=(n_v, n_v))
    in_g = np.zeros(n_v, dtype=bool)
    in_g[us] = in_g[vs] = True                     # node không có cạnh không thuộc DiGraph -> nx báo lỗi
    ends = sorted(set(seeds) | set(answers))
    row = {x: k for k, x in enumerate(ends)}
    d_from = shortest_path(G, unweighted=True, indices=ends)              # d(x, ·)
    d_to = shortest_path(G.T.tocsr(), unweighted=True, indices=ends)      # d(·, y)
    pos, max_len = set(), None
    for q in seeds:
        for a in answers:
            if not (in_g[q] and in_g[a]):
                continue
            L = {(q, a): d_from[row[q], a], (a, q): d_from[row[a], q]}
            ok = {k: v for k, v in L.items() if np.isfinite(v)}
            if not ok:
                continue
            lmin = min(ok.values())
            for (x, y), lxy in ok.items():
                if len(ok) == 2 and lxy != lmin:       # cả hai chiều có đường: chỉ giữ đường ngắn nhất
                    continue
                max_len = int(lxy) if max_len is None else max(max_len, int(lxy))
                if lxy > 0:
                    on = d_from[row[x], us] + 1 + d_to[row[y], vs] == lxy
                    pos.update(keep[on].tolist())
    return pos, max_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=".cache/subgraphrag")
    ap.add_argument("--subgraphs", default=".cache/gnnrag", help="thư mục chứa <split>.npz (SubgraphStore)")
    ap.add_argument("--data-root", default="data/stark/prime")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    skb = PrimeSKB(args.data_root)
    n = skb.num_nodes
    ei = skb.edge_index.numpy()
    et = skb.edge_types.numpy()
    order = np.lexsort((ei[1], ei[0]))
    src_s, dst_s, et_s = ei[0][order], ei[1][order], et[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    indptr = np.cumsum(indptr)
    print(f"đồ thị {n:,} node, {len(dst_s):,} cạnh\n", flush=True)

    for split in args.splits.split(","):
        f = out / f"labels_{split}.npz"
        if f.exists():
            print(f"{split}: đã có, bỏ qua", flush=True)
            continue
        store = SubgraphStore(f"{args.subgraphs}/{split}.npz", indptr, dst_s, et_s, n)
        qids, h_l, r_l, t_l, off = [], [], [], [], [0]
        n_path = n_seed = 0
        t0 = time.time()
        for i in range(len(store)):
            qid = int(store.qids[i])
            qids.append(qid)
            nodes = store.nodes(i)
            pos = set()
            if len(nodes):
                n_seed += 1
                loc = {g: j for j, g in enumerate(nodes.tolist())}
                ans = [loc[a] for a in store.answers(i).tolist() if a in loc]
                sd = [loc[s] for s in store.seeds(i).tolist() if s in loc]
                if ans and sd:
                    _, s_, d_, r_ = store.subgraph(i)
                    tids, _ = orig_triples(len(nodes), s_, d_, sd, ans)
                    if tids:
                        n_path += 1
                        for tid in tids:
                            # id TOÀN CỤC, không phải chỉ số cạnh
                            pos.add((int(nodes[s_[tid]]), int(r_[tid]), int(nodes[d_[tid]])))
            for (h, r, t) in sorted(pos):
                h_l.append(h)
                r_l.append(r)
                t_l.append(t)
            off.append(len(h_l))
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {split} {i+1}/{len(store)}  {el/60:.1f} phút, "
                      f"còn ~{(len(store)-i-1)*el/(i+1)/60:.0f} phút", flush=True)
        np.savez_compressed(
            f, qids=np.array(qids, dtype=np.int64),
            head=np.array(h_l, dtype=np.int32), rel=np.array(r_l, dtype=np.int8),
            tail=np.array(t_l, dtype=np.int32), off=np.array(off, dtype=np.int64))
        npos = np.diff(off)
        print(f"{split}: {len(store):,} query · {n_seed:,} có seed · "
              f"{n_path:,} CÓ ĐƯỜNG ĐI ({n_path/len(store)*100:.1f}%) · "
              f"triple dương p50={np.percentile(npos[npos>0],50):.0f} "
              f"max={npos.max()} · {(time.time()-t0)/60:.1f} phút "
              f"-> {f.name} ({f.stat().st_size/1e6:.1f} MB)\n", flush=True)

    print("SUBGRAPHRAG_LABELS_DONE", flush=True)


if __name__ == "__main__":
    main()
