"""Chấm SubgraphRAG trên RiTeK (kernel ritek/kaggle/sgrag) qua harness.

Chạy: python ritek/scripts/eval_sgrag.py --posw sqrt [--run ritek/outputs/kaggle/ritek-sgrag-sqrt]

- Query có topic: điểm node = max logit các triple kề (`triple_to_node_scores`, đã tính trong kernel)
  trên subgraph PPR-2000; n_native = số node subgraph (2,000).
- Node ngoài subgraph và 18 query test không có topic: backfill cosine MedCPT add_rel max, đặt
  hẳn dưới điểm thấp nhất của SubgraphRAG — giống ritek/scripts/eval_gnnrag_rearev.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

BACKFILL = "results/kaggle/stark-enc-medcpt-add-rel-full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posw", required=True, choices=["none", "sqrt"])
    ap.add_argument("--run", default=None, help="mặc định ritek/outputs/kaggle/ritek-sgrag-<posw>")
    ap.add_argument("--results-root", default="ritek/results")
    args = ap.parse_args()
    run = Path(args.run or f"ritek/outputs/kaggle/ritek-sgrag-{args.posw}")

    meta = json.loads((run / f"subgraphrag_{args.posw}_meta.json").read_text(encoding="utf-8"))
    gte = json.loads((run / "gte_meta.json").read_text(encoding="utf-8"))
    d = np.load(run / f"subgraphrag_{args.posw}_test_pred.npz")
    off = d["off"]
    P = {int(q): (d["node_flat"][off[i]:off[i + 1]].astype(np.int64), d["score_flat"][off[i]:off[i + 1]])
         for i, q in enumerate(d["qids"])}

    bq = np.load(f"{BACKFILL}/qids_test.npy")
    B = np.load(f"{BACKFILL}/scores_medcpt_add_rel_test_max.npy", mmap_mode="r")
    bpos = {int(q): i for i, q in enumerate(bq)}

    skb = PrimeSKB("data/stark/prime")
    qdf = load_queries("data/stark/prime", "test", dataset="ritek")
    with_topic = {int(r["id"]) for _, r in qdf.iterrows() if r["topic_ids"]}
    # kernel bỏ query có topic nhưng subgraph rỗng/không nhãn; với RiTeK mọi query có topic đều có subgraph
    assert set(P) == with_topic, f"thiếu dự đoán: {len(with_topic - set(P))} query có topic"
    print(f"{len(P)} query có dự đoán · {len(qdf) - len(P)} query chỉ backfill · "
          f"epoch tốt nhất {meta['best_epoch']} · val triple_recall@100 {meta['best_val_triple_recall100']*100:.2f}%")

    def scores_fn(r):
        q = int(r["id"])
        b = np.asarray(B[bpos[q]], dtype=np.float64)
        b = np.where(np.isfinite(b), b, -2.0)
        if q not in P:
            return b, 0
        nd, sc = P[q]
        sc = sc.astype(np.float64)
        out = b - b.max() + (sc.min() - 1.0)            # nền: hẳn dưới mọi điểm SubgraphRAG
        out[nd] = sc
        return out, len(nd)

    res = evaluate_and_save(
        scores_fn, qdf, skb, "subgraphrag", f"mlp_{args.posw}", "test", results_root=args.results_root,
        config={"pos_weight": args.posw, "best_epoch": meta["best_epoch"],
                "best_val_triple_recall@100": meta["best_val_triple_recall100"], "n_params": meta["n_params"],
                "emb": meta["emb"], "lr": meta["lr"], "num_rounds": 2, "num_reverse_rounds": 2, "topic_pe": True,
                "subgraph": "PPR-2000 (alpha 0.85, 20 vòng) từ topic RiTeK", "max_edges_train": 100000,
                "val_test_edges": "toàn bộ cạnh subgraph (không lấy mẫu)", "dde": "trên subgraph đầy đủ",
                "lm": gte, "backfill": f"{BACKFILL} (medcpt add_rel max)", "run": str(run), "llm_calls": 0},
        notes="SubgraphRAG retriever (không LLM): MLP chấm triple trong subgraph PPR-2000, node = max triple kề.")
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>8.2f}%")


if __name__ == "__main__":
    main()
