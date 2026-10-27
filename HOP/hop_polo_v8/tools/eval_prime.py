"""Chấm HOP V8 trên RiTeK-Prime (kernel HOP/hop_polo_v8/kaggle/hop_v8.py) qua harness.

Chạy (từ gốc repo; cần harness src/starkbench và backfill results/kaggle/stark-enc-medcpt-add-rel-full):
  python HOP/hop_polo_v8/tools/eval_prime.py --run ritek/outputs/kaggle/<slug>/hop_v8_prime [--variant v8]
Kết quả ghi vào ritek/results/hop/<variant>/test/ (metrics.json, manifest.json, per_query.parquet).

- Xếp hạng của HOP = tập `selected` (top-20 ∪ topic ∪ nút trên đường tốt nhất của top-20, theo điểm ranker)
  -> phần còn lại của pool (theo điểm ranker) -> backfill cosine MedCPT add_rel max, đặt hẳn dưới mọi điểm
  HOP (giống eval_sgrag.py / eval_gnnrag_rearev.py). Thứ tự riêng của HOP được đổi thành điểm giảm ngặt.
- n_native = |selected| (F2 trên tập selected, quyết định 2026-10-27). Top-20 nằm trọn trong selected nên
  Hit@k, R@10, R@20 không phụ thuộc cách xếp phần sau.
- 18 câu test không có topic: chỉ backfill, n_native = 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

BACKFILL = "results/kaggle/stark-enc-medcpt-add-rel-full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="thư mục output hop_v8_prime (có audit.json, test/)")
    ap.add_argument("--results-root", default="ritek/results")
    ap.add_argument("--variant", default="v8")
    args = ap.parse_args()
    run = Path(args.run)
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE":
        sys.exit(f"run chưa xong: status={audit.get('status')}")
    P = {}
    with (run / "test" / "test_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            selected = [n["node_id"] for n in row["ranked_nodes"]]
            chosen = set(selected)
            order = selected + [n for n in row["pool_ranked"] if n not in chosen]
            P[int(row["qid"])] = (np.asarray(order, np.int64), len(selected))

    bq = np.load(f"{BACKFILL}/qids_test.npy")
    B = np.load(f"{BACKFILL}/scores_medcpt_add_rel_test_max.npy", mmap_mode="r")
    bpos = {int(q): i for i, q in enumerate(bq)}
    skb = PrimeSKB("data/stark/prime")
    qdf = load_queries("data/stark/prime", "test", dataset="ritek")
    assert set(P) == {int(r["id"]) for _, r in qdf.iterrows()}, "dự đoán HOP phải phủ đủ 2,801 câu test"
    native = sum(1 for order, _ in P.values() if len(order))
    print(f"{native} câu có pool HOP · {len(P) - native} câu chỉ backfill")

    def scores_fn(r):
        q = int(r["id"])
        b = np.asarray(B[bpos[q]], dtype=np.float64)
        b = np.where(np.isfinite(b), b, -2.0)
        order, n_selected = P[q]
        if not len(order):
            return b, 0
        out = b - b.max() - 1.0                                   # nền: hẳn dưới mọi điểm HOP (>= 0)
        out[order] = np.arange(len(order), 0, -1, dtype=np.float64)  # giảm ngặt theo thứ tự HOP
        return out, n_selected

    done = audit["test"]
    res = evaluate_and_save(
        scores_fn, qdf, skb, "hop", args.variant, "test", results_root=args.results_root,
        config={"run": str(run), "checkpoint_lock": done["checkpoint_lock"],
                "hop_native_test_metrics": done["metrics"], "identity": audit["identity"],
                "selected_epochs": {k: audit[k].get("selected_epoch") for k in ("warmup", "setting_b", "ranker")},
                "ranking": "selected (ranker score) -> rest of pool (ranker score) -> backfill",
                "n_native": "|selected| = top-20 ∪ topics ∪ best-path nodes of top-20",
                "backfill": f"{BACKFILL} (medcpt add_rel max)", "llm_calls": 0},
        notes="HOP PoLo V8: Flan-T5 sinh chương trình HOP, policy cạnh toàn láng giềng, pool ranker.")
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>8.2f}%")


if __name__ == "__main__":
    main()
