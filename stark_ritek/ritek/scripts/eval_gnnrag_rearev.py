"""Chấm GNN-RAG A+ (kernel ritek/kaggle/gnnrag_rearev) qua harness, dataset RiTeK.

Chạy: python ritek/scripts/eval_gnnrag_rearev.py [--run ritek/outputs/kaggle/ritek-gnnrag-rearev]

- Query CÓ topic: điểm = pred_dist của ReaRev trên TOÀN KG -> xếp hạng đầy đủ, n_native = 129,375.
- Query KHÔNG có topic (18 câu test): ReaRev gốc bỏ qua -> chỉ còn backfill, n_native = 0 (Q8,
  ritek/PLAN.md §2). Backfill = cosine MedCPT add_rel max, giống các phương pháp đồ thị trước đây.
Mẫu số luôn là đủ 2,801 query test. Số trên test-0.1 cắt từ per_query.parquet.
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
    ap.add_argument("--run", default="ritek/outputs/kaggle/ritek-gnnrag-rearev")
    ap.add_argument("--variant", default="rearev_fullkg")
    ap.add_argument("--results-root", default="ritek/results")
    args = ap.parse_args()
    run = Path(args.run)

    meta = json.loads((run / "rearev_meta.json").read_text(encoding="utf-8"))
    S = np.load(run / "rearev_test_scores.npy", mmap_mode="r")
    qids = np.load(run / "rearev_test_qids.npy")
    pos = {int(q): i for i, q in enumerate(qids)}
    bq = np.load(f"{BACKFILL}/qids_test.npy")
    B = np.load(f"{BACKFILL}/scores_medcpt_add_rel_test_max.npy", mmap_mode="r")
    bpos = {int(q): i for i, q in enumerate(bq)}

    skb = PrimeSKB("data/stark/prime")
    qdf = load_queries("data/stark/prime", "test", dataset="ritek")
    no_topic = set(int(r["id"]) for _, r in qdf.iterrows() if not r["topic_ids"])
    assert set(pos) == set(int(x) for x in qdf["id"]) - no_topic, "query có topic phải có đủ điểm ReaRev"
    print(f"{len(pos)} query có điểm ReaRev · {len(no_topic)} query không topic -> chỉ backfill")

    def scores_fn(r):
        q = int(r["id"])
        if q in pos:
            return np.asarray(S[pos[q]], dtype=np.float64), skb.num_nodes
        b = np.asarray(B[bpos[q]], dtype=np.float64)
        return np.where(np.isfinite(b), b, -2.0), 0

    res = evaluate_and_save(
        scores_fn, qdf, skb, "gnnrag", args.variant, "test", results_root=args.results_root,
        config={"impl": "A+: ReaRev gốc (cmavro/GNN-RAG) trên toàn KG, message gộp theo quan hệ",
                "gnnrag_sha": meta.get("gnnrag_sha"), "code_sha": meta.get("code_sha"),
                "best_dev_h1": meta.get("best_dev_h1"), "epochs_run": meta.get("epochs_run"),
                "stopped_by_deadline": meta.get("stopped_by_deadline"), "micro_batch": meta.get("micro_batch"),
                "backfill_no_topic": f"{BACKFILL} (medcpt add_rel max)", "run": str(run)},
        notes="GNN-RAG retriever (ReaRev) — chỉ phần GNN, không LLM. Query không topic: backfill, n_native=0.")
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>8.2f}%")


if __name__ == "__main__":
    main()
