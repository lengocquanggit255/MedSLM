"""Dựng (và upload) Kaggle dataset `ritek-sgrag-data` cho SubgraphRAG trên RiTeK (P3, ritek/PLAN.md).

Chạy: python ritek/scripts/prep_sgrag_data.py [--upload]

Giống lần chạy STaRK (scripts/build_gnnrag_data.py + build_subgraphrag_labels.py), chỉ khác seed:
  - subgraph = top-2000 PPR (alpha 0.85, 20 vòng — đúng tham số cũ), xuất phát từ **topic RiTeK**
    thay vì khớp tên. Query không có topic -> subgraph rỗng (Q8: chỉ backfill).
  - split RiTeK (train 4,547 / val 505 / test 2,801).
  - nhãn = triple trên đường đi ngắn nhất topic -> đáp án trong subgraph (≤ 4 bước), id toàn cục.
Text cho encoder (gte-large-en-v1.5, encode trong kernel): tên node, tên quan hệ, câu hỏi —
code gốc SubgraphRAG encode **tên** thực thể và quan hệ (`text_entity_list`, `relation_list`).

Nội dung (ritek/outputs/kaggle_data/ritek-sgrag-data/):
  csr.npz                     indptr · indices (dst) · etype, sắp theo (src, dst) — như build_subgraphrag_labels
  {train,val,test}.npz        định dạng SubgraphStore (starkbench.gnnrag)
  labels_{train,val,test}.npz nhãn triple (scripts/build_subgraphrag_labels.py --subgraphs)
  texts.json                  node_names · relations · queries[split] = [[id, query], ...]
  gnnrag.py, subgraphrag.py   src/starkbench (SubgraphStore, mô hình)
  subgraphrag_train.py        ritek/kaggle/sgrag/subgraphrag_train.py: kernel train của lần chạy STaRK,
                              đã sửa lỗi rò nhãn ở val/test (xem docstring file đó)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from starkbench import ritek  # noqa: E402
from starkbench.ppr import build_transition, ppr  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

SLUG = "ritek-sgrag-data"
OUT = ROOT / "ritek/outputs/kaggle_data" / SLUG
M, ALPHA, ITERS = 2000, 0.85, 20          # = scripts/build_gnnrag_data.py
SPLITS = ("train", "val", "test")


def ragged(lists):
    off = np.zeros(len(lists) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(x) for x in lists])
    flat = np.concatenate([np.asarray(x, dtype=np.int32) for x in lists]) if off[-1] else np.zeros(0, np.int32)
    return flat.astype(np.int32), off


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    skb = PrimeSKB("data/stark/prime")
    n = skb.num_nodes
    ei, et = skb.edge_index.numpy(), skb.edge_types.numpy()

    # CSR cho SubgraphStore: y hệt build_subgraphrag_labels.py
    order = np.lexsort((ei[1], ei[0]))
    src_s, dst_s, et_s = ei[0][order], ei[1][order], et[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    indptr = np.cumsum(indptr)
    rels = list(skb.rel_type_lst())
    np.savez_compressed(OUT / "csr.npz", indptr=indptr, indices=dst_s.astype(np.int32),
                        etype=et_s.astype(np.int8), n_nodes=np.int64(n), n_rel=np.int64(len(rels)))

    # Ma trận chuyển cho PPR: y hệt build_gnnrag_data.py
    A = sp.csr_matrix((np.ones(ei.shape[1], dtype=np.int8), (ei[0], ei[1])), shape=(n, n))
    A.sum_duplicates()
    P = build_transition(A.indptr, A.indices, n)
    print(f"đồ thị {n:,} node / {A.nnz:,} cạnh CSR", flush=True)

    qd = {s: ritek.load_queries(s) for s in SPLITS}
    topics = sorted({t for s in SPLITS for ts in qd[s]["topic_ids"] for t in ts})
    cache, t0 = {}, time.time()
    for i, t in enumerate(topics):
        reset = np.zeros(n, dtype=np.float64)
        reset[t] = 1.0
        x = ppr(P, reset, alpha=ALPHA, iters=ITERS)
        top = np.argpartition(-x, M)[:M]
        top = top[np.argsort(-x[top])]                     # thứ tự PPR giảm dần
        cache[t] = (top.astype(np.int32), x[top].astype(np.float32))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  PPR {i+1}/{len(topics)} topic · {el/60:.1f} phút, còn ~{(len(topics)-i-1)*el/(i+1)/60:.0f}",
                  flush=True)

    for s in SPLITS:
        df = qd[s]
        seeds, nodes, ans, pprs = [], [], [], []
        for r in df.itertuples():
            tp = sorted(int(t) for t in r.topic_ids)
            assert len(tp) <= 1, "RiTeK có đúng 1 topic/query"
            seeds.append(tp)
            ans.append(sorted(set(int(a) for a in r.answer_ids)))
            if tp:
                nodes.append(cache[tp[0]][0])
                pprs.append(cache[tp[0]][1])
            else:
                nodes.append([])
                pprs.append(np.zeros(0, dtype=np.float32))
        sf, so = ragged(seeds)
        nf, no = ragged(nodes)
        af, ao = ragged(ans)
        np.savez_compressed(OUT / f"{s}.npz", qids=df["id"].to_numpy(dtype=np.int64),
                            seed_flat=sf, seed_off=so, node_flat=nf, node_off=no, ans_flat=af, ans_off=ao,
                            ppr_flat=np.concatenate(pprs).astype(np.float32),
                            meta=np.array([M, ALPHA, ITERS, -1], dtype=np.float64))   # -1: seed = topic RiTeK
        cover = np.mean([bool(set(a) & set(x.tolist() if len(x) else [])) for a, x in zip(ans, nodes)])
        print(f"{s}: {len(df):,} query · {sum(1 for x in seeds if not x)} không topic · "
              f"đáp án nằm trong subgraph {cover*100:.1f}%", flush=True)

    subprocess.run([sys.executable, "scripts/build_subgraphrag_labels.py", "--subgraphs", str(OUT),
                    "--out", str(OUT.relative_to(ROOT))], check=True, cwd=ROOT)

    names = [str(skb.node_info[i].get("name", "")) for i in range(n)]
    json.dump({"node_names": names, "relations": rels,
               "queries": {s: [[int(a), str(b)] for a, b in zip(qd[s]["id"], qd[s]["query"])] for s in SPLITS}},
              open(OUT / "texts.json", "w", encoding="utf-8"), ensure_ascii=False)
    copy_code()
    mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {mb:.1f} MB", flush=True)


def copy_code():
    for f in ("src/starkbench/gnnrag.py", "src/starkbench/subgraphrag.py",
              "ritek/kaggle/sgrag/subgraphrag_train.py"):
        shutil.copy(ROOT / f, OUT / Path(f).name)


def upload():
    from kaggle_run import ACCOUNTS, env
    for user, key in ACCOUNTS:     # tuần tự: dataset-metadata.json dùng chung một thư mục
        (OUT / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
        e = env(user, key)
        st = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"], env=e, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
        exists = "ready" in (st.stdout or "").lower()
        cmd = (["kaggle", "datasets", "version", "-p", str(OUT), "-m", "cập nhật", "--dir-mode", "zip"] if exists
               else ["kaggle", "datasets", "create", "-p", str(OUT), "--dir-mode", "zip"])
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=3600)
        print(user, r.returncode, ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--upload-only", action="store_true")
    ap.add_argument("--code-only", action="store_true", help="chỉ chép lại code vào dataset (không dựng lại dữ liệu)")
    a = ap.parse_args()
    if a.code_only:
        copy_code()
    elif not a.upload_only:
        build()
    if a.upload or a.upload_only:
        upload()
