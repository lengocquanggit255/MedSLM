"""Đối chiếu SỐ giữa ReaRev gốc và bản vá A+ (ritek/kaggle/gnnrag_rearev/gnnrag_rearev.py).

Chạy: python ritek/scripts/test_rearev_equivalence.py [--repo .cache/gnnrag_repo]
  (repo = clone cmavro/GNN-RAG @ 28d31bc0, xem fetch_repo trong kernel; tải MiniLM từ HF lần đầu)
  LM=bgem3 python ritek/scripts/test_rearev_equivalence.py   # cùng phép kiểm với LM ablation (tải bge-m3)

Cùng MỘT model ReaRev (lm sbert, cấu hình CWQ), cùng trọng số, dropout tắt, trên KG ngẫu nhiên nhỏ
có cạnh MỘT chiều (để chiều xuôi ≠ chiều ngược, kiểm cả reason_layer_inv):
  gốc : batch dựng bằng BasicDataLoader._build_fact_mat gốc + phép tính theo từng fact (sparse.mm)
  vá  : FullKG + phép tính gộp theo quan hệ
So: loss, pred_dist, gradient MỌI tham số. Thêm: calc_f1_new và Evaluator.f1_and_hits gốc so với bản
vector hoá trên phân phối ngẫu nhiên có điểm hoà.
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
    args_cli = ap.parse_args()
    gnn = K.fetch_repo(args_cli.repo)
    K.add_lm_choice(gnn)
    sys.path.insert(0, gnn)

    from dataset_load import BasicDataLoader
    from models.ReaRev.rearev import ReaRev
    from models.base_model import BaseModel
    from modules.kg_reasoning.reasongnn import ReasonGNNLayer
    from modules.layer_init import TypeLayer
    import evaluate as ev_mod
    from transformers import AutoTokenizer

    orig = {"build_matrix": ReasonGNNLayer.build_matrix, "reason_layer": ReasonGNNLayer.reason_layer,
            "reason_layer_inv": ReasonGNNLayer.reason_layer_inv, "type_forward": TypeLayer.forward,
            "calc_f1_new": BaseModel.calc_f1_new}

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    V, R0, E, B = 60, 5, 400, 3
    rel_names = ["ppi", "off-label use", "side effect", "parent-child", "expression present"][:R0]
    h = np.random.randint(0, V, E)
    t = np.random.randint(0, V, E)
    r = np.random.randint(0, R0, E)                                       # cạnh một chiều, có trùng
    rows = []
    for i in range(B):
        rows.append({"id": i, "query": f"which drug treats disease number {i} and also targets gene {i * 7}",
                     "answer_ids": sorted(np.random.choice(V, size=1 + i, replace=False).tolist()),
                     "topic_ids": [int(np.random.randint(V))]})

    args = K.make_parser().parse_args(["ReaRev", "--name", "cwq", "--lm", K.LM, "--relation_word_emb", "True",
                         "--entity_dim", "50", "--num_iter", "2", "--num_ins", "3", "--num_gnn", "3",
                         "--linear_dropout", "0", "--lm_dropout", "0"])
    args.use_cuda = False
    a = vars(args)
    dev = torch.device("cpu")

    kg = K.FullKG(h, r, t, V, R0, dev)
    tok = AutoTokenizer.from_pretrained(K.LM_NAME)
    rt, rti = K.rel_texts_for(rel_names, tok, kg.R)
    loader = K.FullKGLoader(rows, kg, tok, rt, rti)
    batch_new = loader.get_batch(0, B, 0.0)

    # Batch gốc: cùng mảng entity/query/seed/answer, kb_adj_mat dựng bằng _build_fact_mat GỐC.
    fake = type("L", (), {})()
    fake.data_eff, fake.use_self_loop, fake.num_kb_relation, fake.max_local_entity = False, True, kg.R, V
    fake.kb_adj_mats = {i: (h.copy(), r.copy(), t.copy()) for i in range(B)}
    fake.global2local_entity_maps = {i: {v: v for v in range(V)} for i in range(B)}
    np.random.seed(1)
    edges_orig = BasicDataLoader._build_fact_mat(fake, np.arange(B), fact_dropout=0.0)
    batch_orig = (batch_new[0], batch_new[1], edges_orig) + batch_new[3:]

    torch.manual_seed(0)
    model = ReaRev(a, V, kg.R, loader.num_word)
    model.encode_rel_texts(rt, rti)
    model.eval()                                                          # tắt dropout: so tất định

    def run(batch):
        model.zero_grad()
        loss, _, pred_dist, _ = model(batch, training=False)
        loss.backward()
        grads = {n: (q.grad.clone() if q.grad is not None else None) for n, q in model.named_parameters()}
        return loss.detach(), pred_dist.detach(), grads

    lo, po, go = run(batch_orig)                                          # đường GỐC
    K.apply_patches()
    ln, pn, gn = run(batch_new)                                           # đường VÁ

    fails = 0

    def check(name, x, y, tol=1e-5, atol=1e-9):
        # kiểu allclose: |Δ| <= atol + tol·max|x|. Cần atol vì vài bias đứng trước softmax
        # (score_func.bias, ca_linear.bias) có gradient thật = 0 — hai bên chỉ còn nhiễu ~1e-12.
        nonlocal fails
        d = float((x - y).abs().max()) if x.numel() else 0.0
        rel = d / max(float(x.abs().max()), 1e-12)
        ok = d <= atol + tol * float(x.abs().max())
        fails += not ok
        print(f"  {'OK ' if ok else 'LỆCH'} {name:<48} max|Δ|={d:.2e}  tương đối={rel:.2e}")

    print(f"KG nhỏ: {V} node, {E} cạnh một chiều (+{V} self-loop), R={kg.R}; batch {B}")
    check("loss", lo.view(1), ln.view(1))
    check("pred_dist", po, pn)
    n_grad = 0
    for n in go:
        if go[n] is None and gn[n] is None:
            continue
        assert (go[n] is None) == (gn[n] is None), f"gradient {n}: một bên None"
        check(f"grad {n}", go[n], gn[n], tol=1e-4)
        n_grad += 1
    print(f"  ({n_grad} tham số có gradient)")

    # calc_f1_new: gốc vs vector hoá, trên phân phối có điểm hoà
    torch.manual_seed(3)
    dist = torch.softmax(torch.randint(0, 6, (B, V)).float(), dim=1)    # nhiều điểm hoà
    ans = torch.from_numpy(batch_new[6]).float()
    h1 = torch.ones(B)
    f_o = orig["calc_f1_new"](model, dist, ans, h1)
    f_n = K._calc_f1_new(model, dist, ans, h1)
    check("calc_f1_new (điểm hoà)", f_o, f_n, tol=1e-12)

    # Evaluator.f1_and_hits gốc vs _f1_hits_vec: ngẫu nhiên + biên (rỗng, eps không vượt)
    rng = np.random.default_rng(5)
    mism = 0
    for trial in range(2000):
        n = int(rng.integers(0, 40))
        probs = rng.integers(0, 5, size=40).astype(np.float64)
        probs = probs / probs.sum() if probs.sum() else probs
        mask = rng.random(40) < 0.7
        mask[n:] = False if trial % 7 == 0 else mask[n:]
        answers = set(rng.choice(40, size=int(rng.integers(0, 4)), replace=False).tolist())
        eps = float(rng.choice([0.5, 0.95, 2.0]))
        cand = [(int(c), float(probs[c])) for c in np.flatnonzero(mask)]
        o = ev_mod.f1_and_hits(answers, cand, {i: i for i in range(40)}, None, eps)
        v = K._f1_hits_vec(probs, mask, answers, eps)
        if any(abs(x - y) > 1e-12 for x, y in zip(o[:5], v)):
            mism += 1
    fails += mism > 0
    print(f"  {'OK ' if not mism else 'LỆCH'} Evaluator.f1_and_hits (2000 ca)  sai: {mism}")

    print("\nPASS: bản vá A+ trùng số với ReaRev gốc" if not fails else f"\nFAIL: {fails} mục lệch")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
