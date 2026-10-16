"""S1 — tính nhãn SubgraphRAG: triple nằm trên ĐƯỜNG ĐI NGẮN NHẤT seed → đáp án.

Paper Eq 2: `G̃_q` = shortest paths giữa topic entities và answer entities.
Code gốc: `triple_scores[triple_id_list] = 1.` cho mọi triple trên path, còn lại 0.

⚠️ LƯU NHÃN THEO ID TOÀN CỤC `(h_global, rel, t_global)`, KHÔNG theo chỉ số cạnh.
Lý do: script train áp trần `MAX_EDGES` bằng **lấy mẫu ngẫu nhiên**, nên chỉ số cạnh KHÔNG
ổn định giữa lúc tính nhãn và lúc train. Lưu theo id toàn cục thì ghép lại luôn đúng, bất kể
cạnh bị lấy mẫu hay sắp xếp lại. (Đây đúng loại bẫy đã làm hỏng MoR: file dự đoán không mang
id, tôi giả định thứ tự, và sai ở 2,797/2,801 query.)
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starkbench.gnnrag import SubgraphStore  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402


def sp_triples(adj, seeds, answers, max_hop):
    """Trả về tập chỉ số cạnh nằm trên đường đi ngắn nhất từ seeds tới answers."""
    dist = {s: 0 for s in seeds}
    par = {s: [] for s in seeds}
    q = deque(seeds)
    while q:
        v = q.popleft()
        if dist[v] >= max_hop:
            continue
        for u, tid in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                par[u] = [(v, tid)]
                q.append(u)
            elif dist[u] == dist[v] + 1:
                par[u].append((v, tid))
    hit = [a for a in answers if a in dist]
    if not hit:
        return set()
    tri, seen, stack = set(), set(), list(hit)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for p, tid in par.get(v, []):
            tri.add(tid)
            stack.append(p)
    return tri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max-hop", type=int, default=4)
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
                    adj = [[] for _ in range(len(nodes))]
                    for tid, (a, b) in enumerate(zip(s_.tolist(), d_.tolist())):
                        adj[a].append((b, tid))
                    tids = sp_triples(adj, sd, ans, args.max_hop)
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
