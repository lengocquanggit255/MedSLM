"""Đối chiếu src/starkbench/eval.py với stark_qa.evaluator.Evaluator gốc.

Chạy: python scripts/test_parity.py
Phải PASS trước khi tin bất kỳ con số nào của benchmark.

Không import cả package stark_qa (kéo theo tdc/gdown/...); chỉ nạp file evaluator.py
đã tải về từ sdist, vì file đó chỉ phụ thuộc torch + torchmetrics.
"""
import importlib.util
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from starkbench.eval import METRICS, evaluate_one, full_score  # noqa: E402


def load_reference(evaluator_py: Path):
    spec = importlib.util.spec_from_file_location("stark_eval_ref", evaluator_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Evaluator


def main():
    ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if ref_path is None or not ref_path.exists():
        print("Cần đường dẫn tới stark_qa/evaluator.py (từ sdist stark_qa==1.1.0).")
        print("Vd: python scripts/test_parity.py <path>/stark_qa-1.1.0/stark_qa/evaluator.py")
        sys.exit(2)

    Evaluator = load_reference(ref_path)
    rng = np.random.default_rng(0)

    # candidate_ids = arange(N) đúng như PrimeSKB (candidate_types = tất cả node type)
    n_cand = 3000  # nhỏ để reference (vốn chậm) chạy được, logic không đổi theo N
    candidate_ids = list(range(n_cand))
    ev = Evaluator(candidate_ids=candidate_ids)

    # f2 / f2@k không thuộc stark_qa -> đối chiếu riêng với công thức 5·tp/(4·G + k):
    #   f2@k: k = min(k, n_native) · f2: k = n_native (cả tập phương pháp tự trả về; None = mọi candidate)
    metrics = [m for m in METRICS if not m.startswith("f2")]
    f2s = ["f2", "f2@5"]
    n_fail = n_ref = 0
    for trial in range(300):
        # mô phỏng cả hai chế độ: chấm điểm đầy đủ, và chấm điểm thưa (top-K)
        n_scored = rng.choice([n_cand, rng.integers(1, 500)])   # 1..4 node: phủ ca f2 với m < k
        scored_ids = rng.choice(n_cand, size=int(n_scored), replace=False)
        vals = rng.normal(size=int(n_scored))
        pred = {int(i): float(v) for i, v in zip(scored_ids, vals)}

        n_ans = int(rng.integers(1, 21))
        answer_ids = rng.choice(n_cand, size=n_ans, replace=False)

        # Chỉ so với stark_qa khi mọi ngưỡng cắt (k <= 20, R <= 20) nằm trong phần ĐÃ chấm điểm.
        # Vượt ra ngoài là dải sàn `min(pred)-1` hoà nhau: torch.topk chọn tuỳ ý trong dải hoà,
        # ta chọn id nhỏ trước -> không có đáp án "đúng" để đối chiếu. Benchmark không bao giờ chấm
        # trên sàn phẳng (luôn backfill), nên vùng này nằm ngoài phạm vi parity.
        if n_scored >= 20:
            n_ref += 1
            got = evaluate_one(full_score(pred, n_cand), answer_ids, metrics)
            want = ev.evaluate(pred, torch.LongTensor(answer_ids.tolist()), metrics)
        else:
            got = want = {}

        for m in (metrics if want else []):
            if abs(got[m] - want[m]) > 1e-6:
                n_fail += 1
                if n_fail <= 10:
                    print(f"MISMATCH trial={trial} n_scored={n_scored} n_ans={n_ans} "
                          f"{m}: ours={got[m]:.6f} ref={want[m]:.6f}")

        # F2: chỉ nộp node do phương pháp trả về (n_native = số node đã chấm), xếp theo điểm giảm dần
        # (hoà -> id nhỏ trước, đúng như evaluate_one); dự đoán thưa có thể ít hơn k node.
        n_nat = len(pred) if n_scored < n_cand else None
        got_f2 = evaluate_one(full_score(pred, n_cand), answer_ids, f2s, n_native=n_nat)
        ranked = sorted(pred, key=lambda i: (-pred[i], i))
        for m in f2s:
            n_sub = len(ranked) if n_nat is not None else n_cand
            k_eff = n_sub if m == "f2" else min(int(m.split("@")[1]), len(ranked))
            tp = len(set(ranked[:k_eff]) & set(answer_ids.tolist()))
            want_f2 = 5 * tp / (4 * len(set(answer_ids.tolist())) + k_eff)
            if abs(got_f2[m] - want_f2) > 1e-12:
                n_fail += 1
                if n_fail <= 10:
                    print(f"MISMATCH trial={trial} {m}: ours={got_f2[m]:.6f} công thức={want_f2:.6f}")

    # Ca viết tay khi phương pháp trả ít node (ngẫu nhiên hiếm khi sinh ra).
    #   (metric, pred, đáp án, n_native, kỳ vọng)
    hand = [
        ("f2@5", {7: 3.0, 2: 2.0, 9: 1.0}, [2, 9, 40], 3, 5 * 2 / (4 * 3 + 3)),   # m=3, tp=2
        ("f2@5", {7: 3.0}, [7], 1, 5 * 1 / (4 * 1 + 1)),                         # m=1, đúng 1/1 -> 1.0
        ("f2@5", {7: 3.0}, [8], 0, 0.0),                                         # m=0: không nộp gì
        ("f2@5", {i: -i for i in range(10)}, [0, 6], 10, 5 * 1 / (4 * 2 + 5)),    # m>k -> k=5
        ("f2", {7: 3.0, 2: 2.0, 9: 1.0}, [2, 9, 40], 3, 5 * 2 / (4 * 3 + 3)),     # cả tập 3 node, tp=2
        ("f2", {7: 3.0}, [8], 0, 0.0),                                           # m=0
        ("f2", {i: -i for i in range(10)}, [0, 6], 10, 5 * 2 / (4 * 2 + 10)),     # cả 10 node -> tp=2
    ]
    for metric, pred, ans, m_nat, want_f2 in hand:
        got_f2 = evaluate_one(full_score(pred, 50), np.array(ans), [metric], n_native=m_nat)[metric]
        if abs(got_f2 - want_f2) > 1e-12:
            n_fail += 1
            print(f"MISMATCH ca viết tay pred={pred} ans={ans} m={m_nat}: ours={got_f2} kỳ vọng={want_f2}")

    if n_fail:
        print(f"\nFAIL: {n_fail} sai khác")
        sys.exit(1)
    print(f"PASS: {n_ref} trial x {len(metrics)} metric ({', '.join(metrics)}) "
          f"khớp stark_qa.Evaluator (sai số < 1e-6)")
    print(f"PASS: 300 trial + {len(hand)} ca viết tay x {', '.join(f2s)} khớp "
          f"5·tp/(4·G + k) với k = n_native (f2) / min(k, n_native) (f2@k) (sai số < 1e-12)")


if __name__ == "__main__":
    main()
