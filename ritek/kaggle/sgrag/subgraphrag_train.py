"""S3/S4 — train SubgraphRAG trên STaRK-Prime (Kaggle T4) + suy luận test.

Siêu tham số theo CODE GỐC (`configs/retriever/webqsp.yaml`), paper không nêu:
  num_rounds=2 · num_reverse_rounds=2 · topic_pe=true · lr=1e-3 · patience=10
MLP 2 lớp, loss BCE, dừng sớm theo **triple_recall@100** trên val — đúng code gốc.

🔴 SAI LỆCH BẮT BUỘC: `pos_weight`.
   Code gốc KHÔNG có trọng số lớp (WebQSP mất cân bằng ~1:1,000). Prime ĐO ĐƯỢC 1:164,399.
   POSW:
     none    — giống hệt code gốc (đối chứng, dự kiến model đoán âm hết)
     sqrt    — sqrt(n_âm/n_dương) mỗi query
     full    — n_âm/n_dương mỗi query (cân bằng hoàn toàn)
   ⚠️ Tính THEO TỪNG QUERY, không dùng hằng số: số triple dương lệch cực mạnh
   (p50=2 nhưng max=15,302) nên một hằng số sẽ sai ở một trong hai đầu.

SMOKE=1: chỉ chạy vài epoch trên tập con, in VRAM đỉnh THẬT + giây/bước.

Bản RiTeK (ritek/kaggle/sgrag): sao từ kaggle/subgraphrag_train/subgraphrag_train.py, chỉ sửa MỘT lỗi.
🔴 Bản STaRK rò nhãn: trần MAX_EDGES lấy mẫu PHÂN TẦNG theo nhãn (giữ hết triple trên đường tới
   đáp án) và áp cả lúc val lẫn test. Subgraph trung bình ~551k cạnh nên lúc test mô hình thấy mọi
   cạnh dẫn tới đáp án nhưng chỉ ~18% cạnh còn lại. Sửa: chỉ lấy mẫu lúc TRAIN; val/test chấm
   TOÀN BỘ cạnh của subgraph (như code gốc lúc suy luận), chia khúc EDGE_CHUNK để vừa VRAM.
   Kéo theo: DDE (đặc trưng cấu trúc, không có tham số) luôn tính trên subgraph ĐẦY ĐỦ như code gốc.
   Bản STaRK tính DDE trên đồ thị đã lấy mẫu theo nhãn, nên DDE lúc train cũng mang nhãn và lệch
   phân phối với test. Lúc train, lấy mẫu giờ chỉ quyết định triple nào vào loss.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
POSW = os.environ.get("POSW", "sqrt")
EMB = int(os.environ.get("EMB_SIZE", "1024"))
LR = float(os.environ.get("LR", "1e-3"))
MAX_EPOCH = int(os.environ.get("MAX_EPOCH", "50"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))
MAX_EDGES = int(os.environ.get("MAX_EDGES", "100000"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
EDGE_CHUNK = int(os.environ.get("EDGE_CHUNK", "200000"))   # chỉ để vừa VRAM, không đổi kết quả
RNG = np.random.default_rng(42)
assert POSW in ("none", "sqrt", "full"), POSW


def find_ds(must_have):
    for depth in range(5):
        for c in sorted(glob.glob("/kaggle/input/" + "*/" * depth)):
            if os.path.exists(os.path.join(c, must_have)):
                return c.rstrip("/")
    raise FileNotFoundError(must_have)


DS = os.environ.get("DS_DIR") or find_ds("csr.npz")
sys.path.insert(0, DS)
from gnnrag import SubgraphStore  # noqa: E402
from subgraphrag import SubgraphRAGRetriever, compute_dde, triple_to_node_scores  # noqa: E402

dev = torch.device(os.environ.get("DEVICE", "cuda"))
gname = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
print(f"{gname} · POSW={POSW} emb={EMB} lr={LR} max_edges={MAX_EDGES} smoke={SMOKE}",
      flush=True)

csr = np.load(f"{DS}/csr.npz")
indptr, indices, etype = csr["indptr"], csr["indices"], csr["etype"].astype(np.int64)
n_nodes, R = int(csr["n_nodes"]), int(csr["n_rel"])
rel_emb = torch.from_numpy(np.load(f"{DS}/rel_emb.npy")).to(dev)
NODEEMB = np.load(f"{DS}/nodeemb.npy")
LM = rel_emb.shape[1]
print(f"{n_nodes:,} node · {R} quan hệ · LM dim {LM}", flush=True)

stores, qs, labs = {}, {}, {}
for sp in ("train", "val", "test"):
    stores[sp] = SubgraphStore(f"{DS}/{sp}.npz", indptr, indices, etype, n_nodes)
    z = np.load(f"{DS}/q_{sp}.npz")
    qs[sp] = (z["cls"], {int(x): i for i, x in enumerate(z["qids"])})
    L = np.load(f"{DS}/labels_{sp}.npz")
    # nhãn lưu theo ID TOÀN CỤC (h, rel, t) -> ghép lại luôn đúng dù cạnh bị lấy mẫu
    labs[sp] = ({int(q): i for i, q in enumerate(L["qids"])},
                L["head"], L["rel"].astype(np.int64), L["tail"], L["off"])
    npos = np.diff(L["off"])
    print(f"  {sp}: {len(stores[sp]):,} query · {(npos>0).sum():,} có nhãn dương", flush=True)


def usable(sp):
    """Query dùng được: có subgraph VÀ có nhãn dương (code gốc: `skip_no_path`)."""
    qpos, _, _, _, off = labs[sp]
    out = []
    for i in range(len(stores[sp])):
        if not len(stores[sp].nodes(i)):
            continue
        j = qpos.get(int(stores[sp].qids[i]))
        if j is None:
            continue
        if sp != "test" and off[j + 1] - off[j] == 0:
            continue
        out.append(i)
    return out


IDX = {sp: usable(sp) for sp in ("train", "val", "test")}
if LIMIT:
    IDX = {sp: v[:LIMIT] for sp, v in IDX.items()}
    print(f"  ⚠️ LIMIT={LIMIT} — KIỂM THỬ", flush=True)
for sp, v in IDX.items():
    print(f"  {sp} dùng được: {len(v):,}", flush=True)

CACHE = {}


def prep(sp, i):
    key = (sp, i)
    if key in CACHE:
        return CACHE[key]
    st = stores[sp]
    nodes, s, d_, r = st.subgraph(i)
    loc = {g: j for j, g in enumerate(nodes.tolist())}
    sm = np.zeros(len(nodes), dtype=np.float32)
    for g in st.seeds(i).tolist():
        if g in loc:
            sm[loc[g]] = 1.0

    # DDE trên subgraph ĐẦY ĐỦ, trước mọi lấy mẫu (không có tham số -> tính một lần, cache)
    dde = compute_dde(len(nodes), torch.from_numpy(s), torch.from_numpy(d_),
                      torch.from_numpy(sm)).numpy().astype(np.float32)

    # ---- nhãn TRƯỚC khi lấy mẫu ----
    qpos, H, RL, T, off = labs[sp]
    j = qpos[int(st.qids[i])]
    pos = {(int(a), int(b), int(c))
           for a, b, c in zip(H[off[j]:off[j+1]], RL[off[j]:off[j+1]], T[off[j]:off[j+1]])}
    y = np.zeros(len(s), dtype=np.float32)
    if pos:
        gh, gt = nodes[s], nodes[d_]
        for k in range(len(s)):
            if (int(gh[k]), int(r[k]), int(gt[k])) in pos:
                y[k] = 1.0

    # ---- lấy mẫu PHÂN TẦNG: GIỮ HẾT cạnh dương, chỉ lấy mẫu cạnh âm ----
    # ⚠️ Bản đầu lấy mẫu ĐỀU trên mọi cạnh. Với |E|=404k, p50 chỉ 2 cạnh dương:
    #     MAX_EDGES=100,000 -> 56.6% khả năng MẤT HẾT nhãn dương
    #     MAX_EDGES= 20,000 -> 90.3%
    # Model sẽ học từ dữ liệu gần như không có nhãn dương, ra kết quả tệ, mà triệu chứng
    # duy nhất là "phương pháp không hiệu quả trên Prime" — dễ bị báo cáo nhầm thành
    # phát hiện. Bắt được nhờ chạy thử local với LIMIT=12 (val recall = 0.00%).
    # CHỈ lúc train: lấy mẫu theo nhãn ở val/test là rò đáp án (xem docstring).
    if sp == "train" and len(s) > MAX_EDGES:
        pi = np.flatnonzero(y > 0)
        ni = np.flatnonzero(y == 0)
        n_neg = max(MAX_EDGES - len(pi), 0)
        if n_neg < len(ni):
            ni = RNG.choice(ni, n_neg, replace=False)
        keep = np.concatenate([pi, ni])
        s, d_, r, y = s[keep], d_[keep], r[keep], y[keep]
    b = dict(nodes=nodes, src=s.astype(np.int32), dst=d_.astype(np.int32),
             rel=r.astype(np.int32), sm=sm, dde=dde, y=y, qi=qs[sp][1][int(st.qids[i])],
             ans=np.array(sorted(set(st.answers(i).tolist()) & set(nodes.tolist())),
                          dtype=np.int64))
    if sp != "test" and len(CACHE) < 6000:
        CACHE[key] = b
    return b


def to_gpu(sp, b):
    cls, _ = qs[sp]
    return (torch.from_numpy(cls[b["qi"]].astype(np.float32)).to(dev),
            torch.from_numpy(NODEEMB[b["nodes"]].astype(np.float32)).to(dev),
            torch.from_numpy(b["src"].astype(np.int64)).to(dev),
            torch.from_numpy(b["dst"].astype(np.int64)).to(dev),
            torch.from_numpy(b["rel"].astype(np.int64)).to(dev),
            torch.from_numpy(b["sm"]).to(dev),
            torch.from_numpy(b["y"]).to(dev),
            torch.from_numpy(b["dde"]).to(dev))


model = SubgraphRAGRetriever(LM, R, emb_size=EMB).to(dev)
npar = sum(p.numel() for p in model.parameters())
print(f"\nmô hình {npar/1e6:.2f}M tham số\n", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)


def loss_of(logits, y):
    npos_ = float(y.sum().item())
    if POSW == "none" or npos_ == 0:
        pw = None
    else:
        ratio = (len(y) - npos_) / npos_
        pw = torch.tensor(ratio ** 0.5 if POSW == "sqrt" else ratio, device=dev)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


@torch.no_grad()
def evaluate(sp, idxs):
    """triple_recall@100 — ĐÚNG chỉ số dừng sớm của code gốc."""
    model.eval()
    rec, n = 0.0, 0
    for i in idxs:
        b = prep(sp, i)
        if b["y"].sum() == 0:
            continue
        q, ne, s, d_, r, sm, y, dde = to_gpu(sp, b)
        lg = model(q, ne, rel_emb, s, d_, r, sm, edge_chunk=EDGE_CHUNK, dde=dde)
        top = torch.topk(lg, min(100, len(lg))).indices
        rec += float(y[top].sum().item()) / float(y.sum().item())
        n += 1
    model.train()
    return rec / max(n, 1), n


best, best_ep, hist = -1.0, -1, []
t0 = time.time()
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
for ep in range(MAX_EPOCH):
    model.train()
    RNG.shuffle(IDX["train"])
    te, tot, nb = time.time(), 0.0, 0
    for i in IDX["train"]:
        opt.zero_grad()
        b = prep("train", i)
        q, ne, s, d_, r, sm, y, dde = to_gpu("train", b)
        lg = model(q, ne, rel_emb, s, d_, r, sm, dde=dde)
        ls = loss_of(lg, y)
        ls.backward()
        opt.step()
        tot += ls.item()
        nb += 1
    vr, vn = evaluate("val", IDX["val"])
    sec = time.time() - te
    hist.append(dict(ep=ep, val_triple_recall100=round(vr, 4), loss=round(tot / nb, 4),
                     sec=round(sec, 1)))
    star = ""
    if vr > best:
        best, best_ep = vr, ep
        torch.save(model.state_dict(), f"{OUT}/subgraphrag_{POSW}_best.pt")
        star = " *"
    peak = torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0
    print(f"  epoch {ep:>2}: loss {tot/nb:8.4f} · val triple_recall@100 {vr*100:6.2f}% "
          f"({sec:5.1f}s, VRAM đỉnh {peak:.2f} GB){star}", flush=True)
    if SMOKE and ep >= 2:
        print("  SMOKE: dừng sau 3 epoch", flush=True)
        break
    if ep - best_ep >= PATIENCE:
        print(f"  dừng sớm: {PATIENCE} epoch không cải thiện", flush=True)
        break

print(f"\ntrain {(time.time()-t0)/60:.1f} phút · tốt nhất epoch {best_ep} "
      f"val triple_recall@100 {best*100:.2f}%", flush=True)

if not SMOKE:
    model.load_state_dict(torch.load(f"{OUT}/subgraphrag_{POSW}_best.pt"))
    model.eval()
    qid_l, node_l, score_l, off = [], [], [], [0]
    with torch.no_grad():
        for i in IDX["test"]:
            b = prep("test", i)
            q, ne, s, d_, r, sm, _y, dde = to_gpu("test", b)
            lg = model(q, ne, rel_emb, s, d_, r, sm, edge_chunk=EDGE_CHUNK, dde=dde)
            ns = triple_to_node_scores(len(b["nodes"]), s, d_, lg)
            qid_l.append(int(stores["test"].qids[i]))
            node_l.append(b["nodes"].astype(np.int32))
            score_l.append(ns.float().cpu().numpy().astype(np.float32))
            off.append(off[-1] + len(b["nodes"]))
    np.savez_compressed(f"{OUT}/subgraphrag_{POSW}_test_pred.npz",
                        qids=np.array(qid_l, dtype=np.int64),
                        node_flat=np.concatenate(node_l),
                        score_flat=np.concatenate(score_l),
                        off=np.array(off, dtype=np.int64))
    print(f"đã lưu dự đoán {len(qid_l):,} query test", flush=True)

json.dump(dict(posw=POSW, emb=EMB, lr=LR, best_epoch=best_ep,
               best_val_triple_recall100=best, history=hist, n_params=npar),
          open(f"{OUT}/subgraphrag_{POSW}_meta.json", "w"), indent=2)
print("SUBGRAPHRAG_DONE", flush=True)
