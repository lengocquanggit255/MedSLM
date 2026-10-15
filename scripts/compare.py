"""Kiểm định cặp giữa hai phương pháp trên cùng tập query (PLAN §2b).

Không được kết luận "A tốt hơn B" chỉ vì số trung bình cao hơn: ở n=280, CI 95% của Hit@1
là ±5.6pp. Kiểm định cặp mạnh hơn nhiều so sánh độc lập vì loại được phương sai giữa các query.

  - Hit@1 / Hit@5  : McNemar (nhị phân theo query)
  - R@20/MRR/MAP/R-Prec : paired bootstrap 10k lần

Chạy:
    python scripts/compare.py --a bm25/add_rel --b bm25_seed/add_rel --split test-0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from starkbench.eval import F2_KS, prf_at_k  # noqa: E402
from starkbench.runner import load_queries  # noqa: E402


def per_query(pred_path, gold):
    df = pd.read_parquet(pred_path)
    out = {}
    for qid, g in df.groupby("query_id"):
        top = g.sort_values("rank")["node_id"].to_numpy()
        gs = gold[int(qid)]
        ranked = np.isin(top, list(gs))
        hit = np.flatnonzero(ranked)
        out[int(qid)] = {
            "hit@1": float(ranked[:1].any()),
            "hit@5": float(ranked[:5].any()),
            "recall@20": float(ranked[:20].sum() / len(gs)),
            "map@20": float((np.arange(1, ranked[:20].sum() + 1)
                             / (np.flatnonzero(ranked[:20]) + 1)).mean())
            if ranked[:20].any() else 0.0,
            "rprecision": float(ranked[:len(gs)].sum() / len(gs)),
            "mrr": float(1.0 / (hit[0] + 1)) if hit.size else 0.0,
        }
        for k in F2_KS:
            p, r, f2 = prf_at_k(ranked, len(gs), k)
            out[int(qid)].update({f"p@{k}": p, f"r@{k}": r, f"f2@{k}": f2})
    return out


def mcnemar(a, b):
    """Trả về (b_wins, a_wins, p) — kiểm định 2 phía, chính xác nhị thức."""
    from scipy.stats import binomtest
    n01 = int(np.sum((a == 0) & (b == 1)))   # B đúng, A sai
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n, 0.5).pvalue)


def paired_bootstrap(a, b, iters=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = b - a
    obs = d.mean()
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    # p 2 phía: tỉ lệ mẫu bootstrap đổi dấu so với quan sát
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return obs, min(p, 1.0), np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="vd bm25/add_rel")
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test-0.1")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-root", default="data/stark/prime")
    args = ap.parse_args()

    qdf = load_queries(args.data_root, args.split)
    gold = {int(r["id"]): set(int(x) for x in r["answer_ids"]) for _, r in qdf.iterrows()}
    pa = per_query(Path(args.results_root) / args.a / args.split / "predictions.parquet", gold)
    pb = per_query(Path(args.results_root) / args.b / args.split / "predictions.parquet", gold)
    qids = sorted(set(pa) & set(pb))
    assert len(qids) == len(qdf), f"lệch số query chung: {len(qids)} vs {len(qdf)}"

    print(f"A = {args.a}\nB = {args.b}\nsplit = {args.split}, n = {len(qids)}\n")
    print(f"{'metric':<12}{'A':>9}{'B':>9}{'Δ (B-A)':>11}{'p':>9}  kết luận")
    print("-" * 66)
    for m in ["hit@1", "hit@5", "recall@20", "mrr", "map@20", "rprecision"]:
        a = np.array([pa[q][m] for q in qids])
        b = np.array([pb[q][m] for q in qids])
        if m.startswith("hit@"):
            n01, n10, p = mcnemar(a, b)
            extra = f"  (B thắng {n01}, A thắng {n10} query)"
        else:
            _, p, _ = paired_bootstrap(a, b)
            extra = ""
        verdict = "KHÁC BIỆT (p<0.05)" if p < 0.05 else "không kết luận được"
        print(f"{m:<12}{a.mean()*100:>8.2f}%{b.mean()*100:>8.2f}%"
              f"{(b.mean()-a.mean())*100:>+10.2f}{p:>9.3f}  {verdict}{extra}")


if __name__ == "__main__":
    main()
