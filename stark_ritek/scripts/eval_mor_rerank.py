"""Chấm file dự đoán của MoR qua harness của ta (thay vì `batch_evaluator` của MoR).

Vì sao cần: `MoR/batch_evaluator` tự cài metric, gán điểm 0 cho candidate không được chấm
(stark_qa gốc gán `min(pred)-1`) và **không** áp bộ lọc dấu của torchmetrics. Muốn đặt MoR cạnh
các phương pháp khác trong bảng thì phải chấm cùng một thước.

`mor_pred_test.pkl` là list 2,801 phần tử {q_id, pred_dict: {node_id: score}}.
Candidate không có trong pred_dict được **backfill** bằng điểm Contriever-FT (đúng như các
phương pháp thưa khác: xem docs/PLAN.md §3.2), rồi dịch xuống dưới điểm thấp nhất của MoR để
không phá thứ tự MoR đã đưa ra.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402

EMB_D = "results/kaggle/stark-mor-emb/candidate_emb.npy"
EMB_Q = "results/kaggle/stark-mor-emb/query_emb_dict.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/kaggle/stark-mor-run/mor_pred_test.pkl")
    ap.add_argument("--variant", default="full_rerank")
    ap.add_argument("--split", default="test")
    ap.add_argument("--notes", default="MoR đầy đủ: mixed traversal + structure-aware reranker.")
    ap.add_argument("--data-root", default="data/stark/prime")
    ap.add_argument("--dataset", default="stark", choices=["stark", "ritek"],
                    help="bộ query để chấm; SKB/corpus luôn nạp từ --data-root (giống hệt ở 2 bản)")
    ap.add_argument("--results-root", default=None, help="mặc định: results (stark) | ritek/results (ritek)")
    ap.add_argument("--test-pkl", default="results/kaggle/stark-mor-run/prime_test.pkl",
                    help="nguồn thứ tự q_id cho định dạng sau rerank (3 list song song)")
    ap.add_argument("--emb-d", default=EMB_D, help="embedding document Contriever-FT (.npy) để backfill; "
                    "biến thể làm giàu PHẢI dùng embedding của chính nó")
    ap.add_argument("--emb-q", default=EMB_Q)
    ap.add_argument("--order-split", default="test",
                    help="split mà --test-pkl thuộc về. Kiểm chéo thứ tự LUÔN chạy trên "
                         "split này (đủ 2,801 query) kể cả khi --split là tập con như "
                         "test-0.1 — nếu lấy gold của tập con thì 2,521 query ngoài tập "
                         "con sẽ 'lệch' giả và assert nổ oan.")
    args = ap.parse_args()
    results_root = args.results_root or ("ritek/results" if args.dataset == "ritek" else "results")

    skb = PrimeSKB(args.data_root)
    n = skb.num_nodes

    raw = pickle.load(open(args.pred, "rb"))
    reranked = isinstance(raw, dict) and "pred_ids" in raw
    if reranked:
        # Định dạng SAU rerank (scripts/run_mor_rerank_cpu.py): 3 list song song, mỗi phần tử
        # ứng với một query THEO ĐÚNG THỨ TỰ DataLoader (shuffle=False). `scores` là list các
        # list-1-phần-tử [[6.50], [6.37], ...] -> phải bóc một lớp.
        # ⚠️ Thứ tự DataLoader KHÔNG phải thứ tự của `load_queries` — đã kiểm: 2,797/2,801 lệch.
        # Thứ tự thật là thứ tự list `data` trong prime_test.pkl, mỗi mục có `q_id`.
        order = [int(e["q_id"]) for e in pickle.load(open(args.test_pkl, "rb"))["data"]]
        assert len(order) == len(raw["pred_ids"]), (
            f"lệch số query: {len(order)} vs {len(raw['pred_ids'])}")
        # Kiểm chéo bằng ans_ids: lệch thứ tự thì đáp án không khớp -> nổ ngay, không ra số sai.
        gold = {int(r["id"]): set(map(int, r["answer_ids"]))
                for _, r in load_queries(args.data_root, args.order_split).iterrows()}
        assert set(order) <= set(gold), (
            f"--order-split={args.order_split} không chứa đủ q_id của {args.test_pkl}")
        mism = sum(1 for i, q in enumerate(order)
                   if set(map(int, raw["ans_ids"][i])) != gold[q])
        assert mism == 0, f"thứ tự query KHÔNG khớp: {mism}/{len(order)} query lệch answer_ids"
        print(f"đã kiểm chéo answer_ids theo q_id của prime_test.pkl (split "
              f"{args.order_split}): {len(order)}/{len(order)} khớp")
        P = {}
        for i, q in enumerate(order):
            ids = [int(x) for x in raw["pred_ids"][i]]
            sc = [float(s[0]) if isinstance(s, (list, tuple)) else float(s)
                  for s in raw["scores"][i]]
            P[q] = dict(zip(ids, sc))
    else:
        P = {int(r["q_id"]): r["pred_dict"] for r in raw}
    sizes = [len(v) for v in P.values()]
    print(f"{len(P):,} query, candidate được MoR chấm: trung bình {np.mean(sizes):.0f}, "
          f"min {min(sizes)}, max {max(sizes)}")

    D = np.load(args.emb_d).astype(np.float32)
    Q = torch.load(args.emb_q, map_location="cpu", weights_only=False)
    Q = {int(k): v.numpy().astype(np.float32).ravel() for k, v in Q.items()}
    Dt = torch.from_numpy(D)
    print(f"backfill: Contriever-FT {D.shape}")

    def scores_fn(r):
        qid = int(r["id"])
        pd_ = P.get(qid, {})
        # nền: điểm Contriever-FT, dịch xuống HẲN dưới điểm thấp nhất của MoR
        base = (torch.from_numpy(Q[qid]).view(1, -1) @ Dt.T).squeeze(0).numpy().astype(np.float64)
        if not pd_:   # MoR không trả gì cho query này: chỉ còn backfill, không nộp node nào (f2 = 0)
            return base, 0
        ids = np.fromiter(pd_.keys(), dtype=np.int64, count=len(pd_))
        val = np.fromiter(pd_.values(), dtype=np.float64, count=len(pd_))
        base = base - base.max() + (val.min() - 1.0)
        base[ids] = val
        return base, len(pd_)

    qdf = load_queries(args.data_root, args.split, dataset=args.dataset)
    # KHÔNG lọc query theo P: query thiếu dự đoán vẫn tính vào mẫu số (chỉ còn backfill).
    missing = int((~qdf["id"].isin(P)).sum())
    print(f"chấm {len(qdf):,} query trên split {args.split} ({missing} query MoR không trả gì)")

    res = evaluate_and_save(
        scores_fn, qdf, skb, "mor", args.variant, args.split, results_root=results_root,
        config={"stage": "rerank.py (sau structure-aware reranker)" if reranked
                         else "eval.py (trước reranker)",
                "pred_file": args.pred,
                "backfill": "contriever_ft dot product, dịch xuống dưới min(MoR)",
                "emb_d": args.emb_d, "emb_q": args.emb_q, "queries_without_pred": missing},
        notes=args.notes)
    print(f"\nmor/{args.variant} — {args.split}")
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>10.2f}%")
    print("\nSố MoR tự báo (batch_evaluator): hit@1 36.09  hit@5 59.62  R@20 63.05  mrr 46.52")
    print("Số paper                        : hit@1 36.41  hit@5 60.01  R@20 63.48  mrr 46.92")


if __name__ == "__main__":
    main()
