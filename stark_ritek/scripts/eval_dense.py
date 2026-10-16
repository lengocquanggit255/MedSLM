"""Chấm điểm cho kết quả encode từ Kaggle (cosine baseline B1/B2).

Kernel Kaggle chỉ xuất ma trận điểm; việc chấm metric làm ở local bằng đúng harness đã
verify parity (src/starkbench/eval.py), để mọi phương pháp dùng cùng một đường chấm.

Chạy:
    python scripts/eval_dense.py --run results/kaggle/stark-enc-medcpt-no-rel
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starkbench.eval import METRICS_REPORT  # noqa: E402
from starkbench.runner import evaluate_and_save, load_queries  # noqa: E402
from starkbench.skb import PrimeSKB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="thư mục output của kernel")
    ap.add_argument("--data-root", default="data/stark/prime")
    ap.add_argument("--results-root", default=None, help="mặc định: results (stark) | ritek/results (ritek)")
    ap.add_argument("--dataset", default="stark", choices=["stark", "ritek"],
                    help="bộ query để chấm; SKB/corpus luôn nạp từ --data-root (giống hệt ở 2 bản)")
    args = ap.parse_args()
    args.results_root = args.results_root or ("ritek/results" if args.dataset == "ritek" else "results")

    run = Path(args.run)
    meta_p = glob.glob(str(run / "meta_*.json"))
    if not meta_p:
        sys.exit(f"không thấy meta_*.json trong {run}")
    meta = json.loads(Path(meta_p[0]).read_text(encoding="utf-8"))
    enc, var, split = meta["encoder"], meta["variant"], meta["split"]

    qids = np.load(run / f"qids_{split}.npy")
    skb = PrimeSKB(args.data_root)
    qdf = load_queries(args.data_root, split, dataset=args.dataset)

    # Ánh xạ query theo ID, không tin thứ tự index của ma trận điểm.
    pos = {int(q): i for i, q in enumerate(qids)}
    assert set(pos) == set(int(x) for x in qdf["id"]), "tập query id không khớp"

    # Chấm CẢ HAI cách gộp chunk, không chốt cứng một cách:
    #   top3_mean pha loãng node nhiều chunk (chính là node đáp án — trung vị 876 token),
    #   max thì ngược lại thiên vị node nhiều chunk.
    ran = 0
    for mode in ("top3_mean", "max"):
        f = run / f"scores_{enc}_{var}_{split}_{mode}.npy"
        if not f.exists():
            continue
        scores = np.load(f)
        assert scores.shape == (len(qdf), skb.num_nodes), f"lệch shape: {scores.shape}"
        ran += 1

        def scores_fn(r, _s=scores):
            # Node không có chunk -> kernel ghi -inf; hạ về sàn hữu hạn (cosine >= -1).
            v = _s[pos[int(r["id"])]].astype(np.float64)
            return np.where(np.isfinite(v), v, -2.0)

        vname = var if mode == "top3_mean" else f"{var}_{mode}"
        res = evaluate_and_save(
            scores_fn, qdf, skb, f"cosine_{enc}", vname, split,
            results_root=args.results_root, config={**meta, "chunk_pool": mode},
            notes=(f"Kaggle T4x2, chunk 512 không chồng lấn, pooling={mode}. "
                   "Điểm = cosine (vector đã L2-normalize)."))
        print(f"\ncosine_{enc} / {vname} / {split}  (n={res['n_queries']}, "
              f"{meta['n_chunks']:,} chunk, {meta['chunks_per_doc']}/doc, pool={mode})")
        for m in METRICS_REPORT:
            print(f"  {m:<12}{res['sanitized'][m]*100:>11.2f}%")
        bd = res["breakdown_sanitized"]
        for g in ["answers_with_text", "answers_without_text"]:
            if bd[g]["n"]:
                print(f"  R@20 {g:<22}n={bd[g]['n']:<4}{bd[g]['recall@20']*100:>7.2f}%")
    if not ran:
        sys.exit(f"không thấy file scores_*_{split}_<pooling>.npy trong {run}")


if __name__ == "__main__":
    main()
