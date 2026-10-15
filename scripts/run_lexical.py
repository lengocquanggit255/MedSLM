"""Baseline từ vựng: BM25 (B3 no_rel / B4 add_rel / B6 name_only) và degree-prior (B5).

Chạy:
    python scripts/run_lexical.py --method bm25 --variant no_rel   --split test-0.1
    python scripts/run_lexical.py --method bm25 --variant add_rel  --split test-0.1
    python scripts/run_lexical.py --method bm25 --variant name_only --split test-0.1
    python scripts/run_lexical.py --method degree_prior             --split test-0.1

Tham số BM25 để nguyên mặc định của `bm25s` (k1=1.5, b=0.75, method='lucene') và
tokenize mặc định — trùng với cách `stark_qa/models/bm25.py` gọi, để so được với số paper.
KHÁC stark_qa một điểm CÓ CHỦ ĐÍCH: stark_qa chỉ lấy top-100 (`retrieve(k=100)`), nên MRR họ
công bố thực chất là MRR@100. Ta dùng `get_scores` để chấm ĐỦ 129,375 candidate.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starkbench import corpus  # noqa: E402
from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bm25", "degree_prior"], required=True)
    ap.add_argument("--variant", default="no_rel", choices=list(corpus.VARIANTS))
    ap.add_argument("--split", default="test-0.1")
    ap.add_argument("--data-root", default="data/stark/prime")
    ap.add_argument("--results-root", default=None, help="mặc định: results (stark) | ritek/results (ritek)")
    ap.add_argument("--dataset", default="stark", choices=["stark", "ritek"],
                    help="bộ query để chấm; SKB/corpus luôn nạp từ --data-root (giống hệt ở 2 bản)")
    ap.add_argument("--enrich", default=None,
                    help="khoá trong starkbench.enrich.ALL_SPECS (E D P T0 T01 xGO …); bỏ trống = corpus gốc")
    args = ap.parse_args()
    args.results_root = args.results_root or ("ritek/results" if args.dataset == "ritek" else "results")

    skb = PrimeSKB(args.data_root)
    qdf = load_queries(args.data_root, args.split, dataset=args.dataset)
    # degree_prior không đọc corpus nên variant của nó luôn là "none"
    variant = "none" if args.method == "degree_prior" else args.variant
    tag = None
    if args.enrich:
        from starkbench.enrich import ALL_SPECS as BENCH_SPECS, apply, variant_name
        spec = BENCH_SPECS[args.enrich]
        tag = variant = variant_name(args.variant, spec)
        # node_info làm giàu CHỈ dùng để dựng corpus; chấm điểm dùng node_info GỐC để nhóm
        # breakdown (đáp án có/không text) giống hệt V0 -> so sánh được.
        orig = skb.node_info
        skb.node_info = apply(orig, spec)
        corpus.build(skb, args.variant, tag=tag)
        skb.node_info = orig
    print(f"{args.method}/{variant} trên split={args.split}: {len(qdf)} query, "
          f"{skb.num_nodes:,} candidate")

    if args.method == "degree_prior":
        deg = skb.degree().astype(np.float64)
        # Cùng một bảng xếp hạng cho MỌI query — đây là null model, không nhìn query.
        res = evaluate_and_save(
            lambda r: deg, qdf, skb, "degree_prior", "none", args.split,
            results_root=args.results_root,
            config={"description": "xếp hạng theo bậc node, độc lập với query"},
            notes="Null model: chứng minh riêng tính hub không mang tín hiệu retrieval.")
    else:
        import bm25s
        docs = corpus.load(args.variant, tag=tag)
        print(f"  nạp corpus {tag or args.variant}: {sum(map(len, docs))/1e6:.0f} MB")
        t = time.time()
        tok = bm25s.tokenize(docs, show_progress=False)
        retriever = bm25s.BM25()          # mặc định: k1=1.5, b=0.75, lucene
        retriever.index(tok, show_progress=False)
        print(f"  index BM25: {time.time()-t:.1f}s")

        def scores_fn(r):
            q = bm25s.tokenize(r["query"], show_progress=False, return_ids=False)[0]
            return retriever.get_scores(q).astype(np.float64)

        res = evaluate_and_save(
            scores_fn, qdf, skb, "bm25", variant, args.split,
            results_root=args.results_root,
            config={"lib": f"bm25s {bm25s.__version__}", "k1": 1.5, "b": 0.75,
                    "method": "lucene", "tokenizer": "bm25s.tokenize mặc định",
                    "n_rel": -1 if args.variant == "add_rel" else None,
                    "base_variant": args.variant, "enrich": args.enrich,
                    "enrich_spec": repr(BENCH_SPECS[args.enrich]) if args.enrich else None},
            notes="Chấm đủ 129,375 candidate (stark_qa gốc chỉ chấm top-100).")

    print(f"\n  {'metric':<12}{'sanitized':>12}{'raw stark_qa':>15}")
    for m in METRICS_REPORT:
        print(f"  {m:<12}{res['sanitized'][m]*100:>11.2f}%{res['raw_stark_qa'][m]*100:>14.2f}%")
    print(f"\n  runtime {res['runtime_sec']}s ({res['sec_per_query']}s/query)")


if __name__ == "__main__":
    main()
