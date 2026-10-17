"""Đối chiếu SubgraphRAG của ta với CHÍNH code gốc (Graph-COM/SubgraphRAG @ 48eb05e, thư mục retrieve/).

Chạy: python ritek/scripts/test_sgrag_equivalence.py [--repo .cache/sgrag_repo] [--pyg .cache/pyg]
Cần torch_geometric cho `PEConv` gốc (không cài vào .venv):
    pip install --target .cache/pyg torch_geometric==2.6.1 psutil aiohttp
Repo gốc tự clone vào --repo nếu chưa có.

Kiểm:
  1. Khởi tạo: cùng seed -> state_dict trùng `Retriever` gốc (cùng tên, cùng thứ tự).
  2. DDE + logit từng triple = `Retriever.forward` gốc (đồ thị ngẫu nhiên có cạnh song song, node cô lập),
     chấm theo khúc cạnh.
  3. Gradient của `chunked_bce_backward` = backward của `F.binary_cross_entropy_with_logits` gốc trên
     toàn bộ triple (có và không có pos_weight).
  4. Nhãn `orig_triples` = `RetrieverDataset._extract_paths_and_score` gốc (networkx) trên đồ thị ngẫu nhiên
     nhiều seed / nhiều đáp án / cạnh song song / đáp án = topic / không có đường, và trên subgraph
     RiTeK thật (val, những query ít cạnh nhất).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
SHA = "48eb05e"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / ".cache/sgrag_repo"))
    ap.add_argument("--pyg", default=str(ROOT / ".cache/pyg"))
    ap.add_argument("--real", type=int, default=12, help="số query val thật để so nhãn")
    a = ap.parse_args()
    repo = Path(a.repo)
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/Graph-COM/SubgraphRAG.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", SHA], check=True)
    sys.path.insert(0, a.pyg)
    sys.path.insert(0, str(repo / "retrieve"))
    from src.dataset.retriever import RetrieverDataset          # noqa: E402  (code gốc)
    from src.model.retriever import Retriever                   # noqa: E402  (code gốc)
    from starkbench.subgraphrag import SubgraphRAGRetriever, chunked_bce_backward, compute_dde, topic_one_hot
    from build_subgraphrag_labels import orig_triples

    fails = 0

    def check(name, ok, info=""):
        nonlocal fails
        fails += not ok
        print(f"  {'OK ' if ok else 'LỆCH'} {name} {info}", flush=True)

    # ---------------- 1-3: mô hình ----------------
    E, V, R, NE = 32, 60, 5, 700
    kw = dict(num_rounds=2, num_reverse_rounds=2)
    torch.manual_seed(7)
    orig = Retriever(E, topic_pe=True, DDE_kwargs=kw).double()
    torch.manual_seed(7)
    ours = SubgraphRAGRetriever(E, topic_pe=True, **kw).double()
    so, su = orig.state_dict(), ours.state_dict()
    check("khởi tạo cùng seed: state_dict trùng gốc", list(so) == list(su) and all(torch.equal(so[k], su[k]) for k in so),
          f"({list(su)})")

    g = torch.Generator().manual_seed(0)
    h = torch.randint(0, V - 5, (NE,), generator=g)                 # 5 node cuối cô lập
    t = torch.randint(0, V - 5, (NE,), generator=g)
    h[:40], t[:40] = h[40:80], t[40:80]                             # cạnh song song (cùng h, t)
    r = torch.randint(0, R, (NE,), generator=g)
    q = torch.randn(1, E, generator=g, dtype=torch.float64)
    ent = torch.randn(V, E, generator=g, dtype=torch.float64)
    rel = torch.randn(R, E, generator=g, dtype=torch.float64)
    seed = torch.zeros(V)
    seed[[3, 17]] = 1.0
    oh = topic_one_hot(seed).double()
    oh_orig = F.one_hot(seed.long(), num_classes=2).double()        # như _assembly gốc
    check("one-hot topic = gốc", torch.equal(oh, oh_orig))

    ei, rei = torch.stack([h, t]), torch.stack([t, h])
    dde_o = torch.cat(orig.dde(oh_orig, ei, rei), dim=1)
    dde_u = compute_dde(V, h, t, oh, **kw)
    check("DDE = gốc", torch.allclose(dde_o, dde_u, atol=1e-12), f"Δ {float((dde_o - dde_u).abs().max()):.1e}")

    lo = orig(h, r, t, q, ent, 0, rel, oh_orig).reshape(-1).detach()
    for chunk in (None, 97):
        lu = ours(q[0], ent, rel, h, t, r, oh, edge_chunk=chunk).detach()
        check(f"logit = gốc (edge_chunk={chunk})", torch.allclose(lo, lu, atol=1e-10), f"Δ {float((lo - lu).abs().max()):.1e}")

    y = torch.zeros(NE, dtype=torch.float64)
    y[torch.randint(0, NE, (9,), generator=g)] = 1.0
    for pw in (None, torch.tensor(5.5, dtype=torch.float64)):
        orig.zero_grad(set_to_none=True)
        ours.zero_grad(set_to_none=True)
        lo = orig(h, r, t, q, ent, 0, rel, oh_orig)
        F.binary_cross_entropy_with_logits(lo, y.unsqueeze(-1), pos_weight=pw).backward()
        loss_u = chunked_bce_backward(ours, q[0], ent, rel, h, t, r, oh, dde_u, y, pos_weight=pw, edge_chunk=97)
        loss_o = float(F.binary_cross_entropy_with_logits(lo.detach(), y.unsqueeze(-1), pos_weight=pw))
        go = {n: p.grad for n, p in orig.named_parameters() if p.grad is not None}
        gu = {n: p.grad for n, p in ours.named_parameters() if p.grad is not None}
        # non_text_entity_emb: gốc nối 0 hàng (Prime không có thực thể thiếu text) -> gradient 0, Adam không
        # đổi nó; ta không đưa vào đồ thị -> gradient None. Hai cách cho cùng trọng số sau mỗi bước.
        extra = set(go) - set(gu)
        zero_extra = all(float(go[n].abs().max()) == 0.0 for n in extra) and extra <= {"non_text_entity_emb.weight"}
        d = max(float((go[n] - gu[n]).abs().max() / go[n].abs().max().clamp(min=1e-30)) for n in gu)
        check(f"gradient chấm theo khúc = gốc (pos_weight={None if pw is None else float(pw)})",
              set(gu) <= set(go) and zero_extra and d < 1e-10 and abs(loss_u - loss_o) < 1e-12,
              f"loss {loss_u:.6f}/{loss_o:.6f} · lệch tương đối {d:.1e} · {sorted(gu)}")

    # ---------------- 4: nhãn ----------------
    ds = object.__new__(RetrieverDataset)

    def label_orig(src, dst, rel_ids, seeds, answers):
        sample = {"h_id_list": src.tolist(), "r_id_list": rel_ids.tolist(), "t_id_list": dst.tolist(),
                  "q_entity_id_list": seeds, "a_entity_id_list": answers}
        sc, ml = ds._extract_paths_and_score(sample)
        return set(sc.nonzero().reshape(-1).tolist()), ml

    rng = np.random.default_rng(1)
    bad, n_case, n_pos = 0, 0, 0
    for case in range(400):
        nv = int(rng.integers(5, 40))
        ne = int(rng.integers(0, 4 * nv))
        src = rng.integers(0, nv, ne)
        dst = rng.integers(0, nv, ne)
        if ne > 4:                                                   # cạnh song song + cạnh ngược
            k = int(rng.integers(1, ne // 2))
            src[-k:], dst[-k:] = dst[:k], src[:k]
            src[:2], dst[:2] = src[2:4], dst[2:4]
        rel_ids = rng.integers(0, 4, ne)
        seeds = sorted(set(rng.integers(0, nv, int(rng.integers(1, 3))).tolist()))
        answers = sorted(set(rng.integers(0, nv, int(rng.integers(1, 5))).tolist()))
        if case % 7 == 0:
            answers = sorted(set(answers) | {seeds[0]})                # đáp án = topic
        po, mo = label_orig(src, dst, rel_ids, seeds, answers)
        pu, mu = orig_triples(nv, src, dst, seeds, answers)
        n_case += 1
        n_pos += len(po)
        bad += (po != pu) or (mo != mu)
    check(f"nhãn = gốc trên {n_case} đồ thị ngẫu nhiên", bad == 0, f"({bad} lệch · tổng {n_pos} triple dương)")

    if a.real:
        from starkbench.gnnrag import SubgraphStore
        DS = ROOT / "ritek/outputs/kaggle_data/ritek-sgrag-data"
        c = np.load(DS / "csr.npz")
        st = SubgraphStore(DS / "val.npz", c["indptr"], c["indices"], c["etype"].astype(np.int64), int(c["n_nodes"]))
        cand = [i for i in range(len(st)) if len(st.nodes(i))]
        sizes = {i: len(st.subgraph(i)[1]) for i in cand}
        bad, n_pos = 0, 0
        for i in sorted(cand, key=sizes.get)[:a.real]:
            nodes, s, d, rr = st.subgraph(i)
            loc = {gg: j for j, gg in enumerate(nodes.tolist())}
            sd = [loc[x] for x in st.seeds(i).tolist() if x in loc]
            an = [loc[x] for x in st.answers(i).tolist() if x in loc]
            po, mo = label_orig(s, d, rr, sd, an)
            pu, mu = orig_triples(len(nodes), s, d, sd, an)
            n_pos += len(po)
            bad += (po != pu) or (mo != mu)
        check(f"nhãn = gốc trên {a.real} subgraph val RiTeK thật", bad == 0, f"({bad} lệch · tổng {n_pos} triple dương)")

    print("\nPASS" if not fails else f"\nFAIL: {fails} mục")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
