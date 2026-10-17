"""Kiểm đường train song song nhiều GPU của kernel GNN-RAG A+ (parallel_backward / sync_params /
make_replica) so với đường tuần tự một model. Chạy trên CPU (hai "thiết bị" đều là CPU, hai luồng):
bắt lỗi logic — chia micro-batch, trọng số loss, cộng gradient, đồng bộ trọng số, chép tensor thường
của model. Phần phụ thuộc GPU thật (thiết bị cuda:1) do check_dual kiểm mỗi đầu phiên trên Kaggle.

Chạy: python ritek/scripts/test_rearev_dual.py [--repo .cache/gnnrag_repo]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ritek/kaggle/gnnrag_rearev"))
import gnnrag_rearev as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / ".cache/gnnrag_repo"))
    gnn = K.fetch_repo(ap.parse_args().repo)
    sys.path.insert(0, gnn)
    from models.ReaRev.rearev import ReaRev
    from transformers import AutoTokenizer
    K.apply_patches()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    V, R0, E, B = 60, 5, 400, 4
    rel_names = ["ppi", "off-label use", "side effect", "parent-child", "expression present"]
    h, t, r = np.random.randint(0, V, E), np.random.randint(0, V, E), np.random.randint(0, R0, E)
    rows = [{"id": i, "query": f"which drug treats disease number {i} and also targets gene {i * 7}",
             "answer_ids": sorted(np.random.choice(V, size=1 + i, replace=False).tolist()),
             "topic_ids": [int(np.random.randint(V))]} for i in range(B)]
    args = K.make_parser().parse_args(["ReaRev", "--name", "cwq", "--lm", "sbert", "--relation_word_emb", "True",
                                       "--entity_dim", "50", "--num_iter", "2", "--num_ins", "3", "--num_gnn", "3"])
    args.use_cuda = False
    a = vars(args)
    cpu = torch.device("cpu")
    kg = K.FullKG(h, r, t, V, R0, cpu)
    tok = AutoTokenizer.from_pretrained(K.LM_NAME)
    rt, rti = K.rel_texts_for(rel_names, tok, kg.R)
    loader = K.FullKGLoader(rows, kg, tok, rt, rti)
    batch = loader.get_batch(0, B, 0.0)

    def build():
        m = ReaRev(a, V, kg.R, loader.num_word)
        m.encode_rel_texts(rt, rti)
        return m

    torch.manual_seed(0)
    main_m = build()
    torch.manual_seed(123)                                   # bản sao khởi tạo khác -> make_replica phải chép đè
    rep = K.make_replica(build, main_m, cpu)
    fails = 0

    def check(name, ok, info=""):
        nonlocal fails
        fails += not ok
        print(f"  {'OK ' if ok else 'LỆCH'} {name} {info}")

    sd0, sd1 = main_m.state_dict(), rep.state_dict()
    check("make_replica: state_dict trùng", all(torch.equal(sd0[k], sd1[k]) for k in sd0), f"({len(sd0)} tensor)")
    check("make_replica: rel_features trùng", torch.equal(main_m.rel_features, rep.rel_features)
          and torch.equal(main_m.rel_features_inv, rep.rel_features_inv))
    check("make_replica: là object riêng", rep is not main_m and rep.rel_features is not main_m.rel_features)

    workers = [(main_m, kg), (rep, kg.to(cpu))]
    for micro, tol in ((2, 0.0), (1, 1e-6)):                # micro=1: 4 micro-batch, thứ tự cộng khác chút
        worst, _, _ = K.check_dual(workers, batch, micro)
        check(f"gradient song song = tuần tự (micro {micro}, dropout tắt)", worst <= tol, f"lệch tương đối {worst:.2e}")
    # ngưỡng 1e-5 của kernel phải bắt được lỗi thật: bản sao lệch trọng số cỡ một bước Adam (lr 5e-4)
    bad = K.make_replica(build, main_m, cpu)
    with torch.no_grad():
        for p in bad.parameters():
            if p.requires_grad:
                p.add_(5e-4 * torch.randn_like(p))
    worst, _, _ = K.check_dual([(main_m, kg), (bad, kg)], batch, 2)
    check("check_dual bắt bản sao lệch trọng số cỡ 1 bước Adam (> 1e-5)", worst > 1e-5, f"lệch {worst:.2e}")

    # Vài bước Adam, dropout BẬT nhưng tắt ngẫu nhiên khác luồng bằng p=0 cho so khớp tất định.
    for m, _ in workers:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
    single = K.make_replica(build, main_m, cpu)
    for mod in single.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    opt_s = torch.optim.Adam([p for p in single.parameters() if p.requires_grad], lr=5e-4)
    opt_d = torch.optim.Adam([p for p in main_m.parameters() if p.requires_grad], lr=5e-4)
    for step in range(3):
        for ws, opt, m in (([(single, kg)], opt_s, single), (workers, opt_d, main_m)):
            for w, _ in ws:
                w.train()
            opt.zero_grad()
            out = K.parallel_backward(ws, loader.get_batch(0, B, 0.0), 2)
            torch.nn.utils.clip_grad_norm_([p for _, p in m.named_parameters()], 1.0)
            opt.step()
            K.sync_params(ws)
    d = max(float((p - q).abs().max()) for p, q in zip(single.parameters(), main_m.parameters()))
    check("3 bước Adam: trọng số 2 luồng = 1 model", d == 0.0, f"Δ max {d:.2e}")
    d = max(float((p - q).abs().max()) for p, q in zip(main_m.parameters(), rep.parameters()))
    check("sync_params: bản sao = model chính", d == 0.0, f"Δ max {d:.2e}")
    check("loss trả về theo thứ tự micro-batch", len(out) == 2 and all(x is not None for x in out))

    print("\nPASS" if not fails else f"\nFAIL: {fails} mục")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
