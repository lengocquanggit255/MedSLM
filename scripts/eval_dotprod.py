"""Chấm Contriever-FT (encoder đã fine-tune của MoR) bằng dot product thô, không traversal/rerank.

Chạy:
  python scripts/eval_dotprod.py                                   # V0 (embedding gốc)
  python scripts/eval_dotprod.py --emb-d <candidate_emb.npy> --emb-q <query_emb_dict.pt> --variant <tên>

Đúng cách chấm của results/contriever_ft/dotprod (config: "raw dot product"): điểm = q · d, không
chuẩn hoá. Dùng để tách phần đóng góp của encoder khỏi traversal + reranker của MoR.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-d", default="results/kaggle/stark-mor-emb/candidate_emb.npy")
    ap.add_argument("--emb-q", default="results/kaggle/stark-mor-emb/query_emb_dict.pt")
    ap.add_argument("--variant", default="dotprod")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results-root", default=None, help="mặc định: results (stark) | ritek/results (ritek)")
    ap.add_argument("--dataset", default="stark", choices=["stark", "ritek"],
                    help="bộ query để chấm; SKB/corpus luôn nạp từ --data-root (giống hệt ở 2 bản)")
    ap.add_argument("--data-root", default="data/stark/prime")
    args = ap.parse_args()
    args.results_root = args.results_root or ("ritek/results" if args.dataset == "ritek" else "results")

    skb = PrimeSKB(args.data_root)   # node_info GỐC -> nhóm breakdown giống hệt V0
    D = torch.from_numpy(np.load(args.emb_d).astype(np.float32))
    Q = torch.load(args.emb_q, map_location="cpu", weights_only=False)
    Q = {int(k): v.numpy().astype(np.float32).ravel() for k, v in Q.items()}
    assert D.shape[0] == skb.num_nodes, D.shape
    qdf = load_queries(args.data_root, args.split, dataset=args.dataset)

    def scores_fn(r):
        return (torch.from_numpy(Q[int(r["id"])]).view(1, -1) @ D.T).squeeze(0).numpy().astype(np.float64)

    res = evaluate_and_save(scores_fn, qdf, skb, "contriever_ft", args.variant, args.split,
                            results_root=args.results_root,
                            config={"source": "MoR fine-tuned Contriever", "scoring": "raw dot product",
                                    "emb_d": args.emb_d, "emb_q": args.emb_q})
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>8.2f}%")


if __name__ == "__main__":
    main()
