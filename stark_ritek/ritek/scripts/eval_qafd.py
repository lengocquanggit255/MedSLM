"""Chấm QAFD-RAG retriever-only (kernel ritek/kaggle/qafd) qua harness.

Chạy: python ritek/scripts/eval_qafd.py [--run ritek/outputs/kaggle/ritek-qafd-prime] [--variant notebook_add]

Tập node QAFD tự trả về = support (x > 0), điểm = x, n_native = |support| -> `f2` tính trên đúng tập đó. Node khác và
18 câu test không topic: backfill cosine MedCPT add_rel max, đặt hẳn dưới điểm thấp nhất của QAFD (như
ritek/scripts/eval_sgrag.py) — chỉ để đủ xếp hạng theo hợp đồng harness; `f2` không dùng phần backfill.
Bảng báo cáo chỉ dùng `f2` của QAFD (quyết định D9b 2026-10-24: không báo Hit@k / Recall / MRR).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

BACKFILL = "results/kaggle/stark-enc-medcpt-add-rel-full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ritek/outputs/kaggle/ritek-qafd-prime")
    ap.add_argument("--results-root", default="ritek/results")
    ap.add_argument("--variant", default="notebook_add")
    args = ap.parse_args()
    run = Path(args.run) / "prime"
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    res = [json.loads(line) for f in sorted(run.glob("results_w*.jsonl")) for line in open(f, encoding="utf-8")]
    P = {int(r["question_id"]): (np.array([int(n) for n, _ in r["support"]], dtype=np.int64),
                                 np.array([float(x) for _, x in r["support"]], dtype=np.float64))
         for r in res if r["status"] == "success" and r["support"]}

    bq = np.load(f"{BACKFILL}/qids_test.npy")
    B = np.load(f"{BACKFILL}/scores_medcpt_add_rel_test_max.npy", mmap_mode="r")
    bpos = {int(q): i for i, q in enumerate(bq)}
    skb = PrimeSKB("data/stark/prime")
    qdf = load_queries("data/stark/prime", "test", dataset="ritek")
    print(f"{len(P)} câu có support · {len(qdf) - len(P)} câu chỉ backfill · support tb {np.mean([len(v[0]) for v in P.values()]):.1f} node")

    def scores_fn(r):
        q = int(r["id"])
        b = np.asarray(B[bpos[q]], dtype=np.float64)
        b = np.where(np.isfinite(b), b, -2.0)
        if q not in P:
            return b, 0
        nd, sc = P[q]
        out = b - b.max() + (sc.min() - 1.0)            # nền: hẳn dưới mọi điểm QAFD
        out[nd] = sc
        return out, len(nd)

    res = evaluate_and_save(
        scores_fn, qdf, skb, "qafd", args.variant, "test", results_root=args.results_root,
        config={**summ["config"], "seed": "topic entity RiTeK", "support": "node có x > 0 (không cắt)",
                "node_text": "[tên (loại), no_rel bỏ dòng tên/loại] (D5b)", "graph": "prime_KG.txt vô hướng gộp",
                "backfill": f"{BACKFILL} (medcpt add_rel max)", "run": str(run), "kernel_code_sha": summ.get("code_sha"),
                "reference": "ref_qafd/qafd-rag.ipynb", "llm_calls": 0},
        notes="QAFD-RAG retriever-only theo notebook tham chiếu; chỉ báo f2 (D9b).")
    print(f"  f2 {res['sanitized']['f2'] * 100:.3f}%")


if __name__ == "__main__":
    main()
