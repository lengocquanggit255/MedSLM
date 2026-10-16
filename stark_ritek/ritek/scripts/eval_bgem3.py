"""Chấm BGE-M3 dense (kernel ritek/kaggle/bgem3_encode) qua harness, dataset RiTeK.

Chạy: python ritek/scripts/eval_bgem3.py --run ritek/outputs/kaggle/ritek-bgem3-add-rel [--split test]

Kernel chỉ xuất embedding chunk + query. Ở đây: cosine (query, chunk) -> gộp về node bằng
`chunking.pool_scores`, cả hai cách (Q5, ritek/PLAN.md §2):
    top3_mean -> cosine_bgem3/<var>      max -> cosine_bgem3/<var>_max
Với cửa sổ 8192 hầu hết node chỉ có 1 chunk, nên hai cách chỉ khác ở số ít doc rất dài.
Mọi node đều có ≥ 1 chunk (doc rỗng vẫn thành 1 chunk) nên không cần backfill: n_native = 129,375.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.chunking import pool_scores  # noqa: E402
from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="thư mục output của kernel")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results-root", default="ritek/results")
    args = ap.parse_args()
    run = Path(args.run)

    meta = json.loads(next(run.glob("meta_bgem3_*.json")).read_text(encoding="utf-8"))
    assert not meta["smoke"], "đây là output smoke, không chấm"
    var = meta["variant"]
    E = np.load(run / f"chunkemb_bgem3_{var}.npy")
    c2n = np.load(run / f"chunk2node_bgem3_{var}.npy")
    Q = np.load(run / "qemb_bgem3.npy")
    qpos = {int(q): i for i, q in enumerate(np.load(run / "qids_all.npy"))}

    skb = PrimeSKB("data/stark/prime")
    assert np.array_equal(np.unique(c2n), np.arange(skb.num_nodes)), "có node không có chunk"
    qdf = load_queries("data/stark/prime", args.split, dataset="ritek")
    rows = np.array([qpos[int(q)] for q in qdf["id"]])
    sims = Q[rows].astype(np.float32) @ E.astype(np.float32).T          # (nq, n_chunk), cosine
    print(f"{var}/{args.split}: {len(rows)} query · {len(c2n):,} chunk "
          f"({meta['n_docs_multi_chunk']} node > 1 chunk)")

    for mode in ("top3_mean", "max"):
        S = pool_scores(sims, c2n, skb.num_nodes, mode=mode)
        pos = {int(q): i for i, q in enumerate(qdf["id"])}

        def scores_fn(r, _S=S):
            return _S[pos[int(r["id"])]].astype(np.float64), skb.num_nodes

        vname = var if mode == "top3_mean" else f"{var}_{mode}"
        res = evaluate_and_save(
            scores_fn, qdf, skb, "cosine_bgem3", vname, args.split, results_root=args.results_root,
            config={**meta, "chunk_pool": mode, "run": str(run)},
            notes=f"BGE-M3 dense, cửa sổ {meta['window']} token, chunk không chồng lấn, pooling={mode}.")
        print(f"cosine_bgem3/{vname}/{args.split}")
        for m in METRICS_REPORT:
            print(f"  {m:<12}{res['sanitized'][m]*100:>8.2f}%")


if __name__ == "__main__":
    main()
