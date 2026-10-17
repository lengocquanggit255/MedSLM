"""Train SubgraphRAG retriever trên RiTeK (Kaggle T4) + suy luận test — vòng train theo CODE GỐC.

Code gốc: github.com/Graph-COM/SubgraphRAG @ 48eb05e, retrieve/train.py + configs/retriever/webqsp.yaml.
Bản này (2026-10-17) sửa mọi chỗ lệch so với code gốc mà bản 2026-10-16 còn giữ:
  1. Nhãn: đường ngắn nhất cả hai chiều topic <-> đáp án theo từng cặp, như `_extract_paths_and_score`
     gốc (scripts/build_subgraphrag_labels.py). Bản cũ chỉ chiều topic -> đáp án: thiếu một nửa nhãn.
  2. Train trên TOÀN BỘ triple của subgraph như gốc. Bản cũ lấy mẫu 100k cạnh/subgraph (giữ hết cạnh
     dương) nên tỉ lệ dương/âm trong loss lệch ~6 lần. Subgraph p99 ~1.8M cạnh không vừa VRAM nếu giữ
     activation của cả subgraph -> backward theo khúc cạnh (`chunked_bce_backward`), cùng gradient.
  3. Mô hình = `Retriever` gốc: one-hot topic 2 chiều, DDE lan truyền cả 2 kênh, một Linear đầu duy
     nhất (src/starkbench/subgraphrag.py; ritek/scripts/test_sgrag_equivalence.py so với code gốc).
  4. Vòng train y hệt train.py gốc: set_seed(42) + cudnn deterministic; mỗi epoch ĐÁNH GIÁ val TRƯỚC rồi
     mới train (epoch 0 chấm mô hình chưa train); best khởi đầu 0, lưu khi val triple_recall@100 tăng
     ngặt; dừng khi 10 epoch liền không tăng; num_epochs 10000; batch 1 query, thứ tự xáo bằng
     RandomSampler như DataLoader(shuffle=True); bỏ query train/val không có đường (`skip_no_path`).
Thích ứng còn lại (gốc không có, cần cho benchmark):
  - POSW=sqrt: pos_weight = sqrt(n_âm/n_dương) mỗi query (Prime mất cân bằng ~1:150,000, gốc không có
    trọng số lớp). POSW=none = đúng code gốc.
  - Test: điểm node = max logit các triple kề (`triple_to_node_scores`), chấm qua harness với backfill.
  - Chạy nhiều phiên Kaggle (12h/phiên): sau MỖI epoch lưu state (model + Adam + epoch + best + đếm
    patience + lịch sử + RNG); RESUME=1 nạp lại -> nối tiếp y như chạy liền
    (ritek/scripts/test_sgrag_resume.py). Hết hạn phiên (DEADLINE_H) thì dừng trước epoch kế, vẫn suy
    luận test bằng checkpoint tốt nhất (kết quả tạm).

Env: POSW none|sqrt · LR 1e-3 · MAX_EPOCH 10000 · PATIENCE 10 · EDGE_CHUNK 200000 (chỉ để vừa VRAM)
· DEADLINE_H 11.5 (tính từ SESSION_T0) · TEST_RESERVE_H 0.5 · RESUME 0|1 (RESUME_DIR) · SESSION_EPOCHS (chỉ để kiểm resume)
· LIMIT n (smoke: n query mỗi split + subgraph lớn nhất của train) · SMOKE 1 (in thời gian, VRAM, ước giờ/epoch).
"""
import glob
import json
import os
import queue
import random
import shutil
import sys
import threading
import time

import numpy as np
import torch

OUT = os.environ.get("OUT_DIR", "/kaggle/working")
POSW = os.environ.get("POSW", "none")
LR = float(os.environ.get("LR", "1e-3"))
MAX_EPOCH = int(os.environ.get("MAX_EPOCH", "10000"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
EDGE_CHUNK = int(os.environ.get("EDGE_CHUNK", "200000"))
DEADLINE_H = float(os.environ.get("DEADLINE_H", "11.5"))
TEST_RESERVE_H = float(os.environ.get("TEST_RESERVE_H", "0.5"))
RESUME = os.environ.get("RESUME", "0") == "1"
SESSION_EPOCHS = int(os.environ.get("SESSION_EPOCHS", "0"))
SEED = 42                                       # configs/retriever/webqsp.yaml: env.seed
T_START = float(os.environ.get("SESSION_T0", time.time()))   # sgrag.py đặt lúc phiên bắt đầu (trước encode)
assert POSW in ("none", "sqrt"), POSW


def set_seed(seed):
    """= retrieve/src/setup.py gốc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_ds(must_have):
    for depth in range(5):
        for c in sorted(glob.glob("/kaggle/input/" + "*/" * depth)):
            if os.path.exists(os.path.join(c, must_have)):
                return c.rstrip("/")
    raise FileNotFoundError(must_have)


def log(*a):
    print(*a, flush=True)


DS = os.environ.get("DS_DIR") or find_ds("csr.npz")
sys.path.insert(0, DS)
from gnnrag import SubgraphStore  # noqa: E402
from subgraphrag import (SubgraphRAGRetriever, chunked_bce_backward, compute_dde,  # noqa: E402
                         topic_one_hot, triple_to_node_scores)

set_seed(SEED)
dev = torch.device(os.environ.get("DEVICE", "cuda"))
gname = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
log(f"{gname} · POSW={POSW} lr={LR} patience={PATIENCE} edge_chunk={EDGE_CHUNK} smoke={SMOKE} limit={LIMIT}")

csr = np.load(f"{DS}/csr.npz")
indptr, indices, etype = csr["indptr"], csr["indices"], csr["etype"].astype(np.int64)
n_nodes, R = int(csr["n_nodes"]), int(csr["n_rel"])
rel_emb = torch.from_numpy(np.load(f"{DS}/rel_emb.npy")).to(dev)
NODEEMB = np.load(f"{DS}/nodeemb.npy")
EMB = rel_emb.shape[1]                          # gốc: emb_size = q_emb.shape[-1]
log(f"{n_nodes:,} node · {R} quan hệ · emb {EMB}")

stores, qs, labs = {}, {}, {}
for sp in ("train", "val", "test"):
    stores[sp] = SubgraphStore(f"{DS}/{sp}.npz", indptr, indices, etype, n_nodes)
    z = np.load(f"{DS}/q_{sp}.npz")
    qs[sp] = (z["cls"], {int(x): i for i, x in enumerate(z["qids"])})
    L = np.load(f"{DS}/labels_{sp}.npz")
    # nhãn lưu theo ID TOÀN CỤC (h, rel, t); khoá int64 = (h·R + rel)·N + t để ghép bằng np.isin
    key = (L["head"].astype(np.int64) * R + L["rel"].astype(np.int64)) * n_nodes + L["tail"].astype(np.int64)
    labs[sp] = ({int(q): i for i, q in enumerate(L["qids"])}, key, L["off"])
    log(f"  {sp}: {len(stores[sp]):,} query · {(np.diff(L['off']) > 0).sum():,} có nhãn dương")


def usable(sp):
    """Query dùng được: có subgraph; train/val còn phải có triple dương (gốc: `skip_no_path`)."""
    qpos, _, off = labs[sp]
    out = []
    for i in range(len(stores[sp])):
        if not len(stores[sp].nodes(i)):
            continue
        j = qpos.get(int(stores[sp].qids[i]))
        if j is None or (sp != "test" and off[j + 1] == off[j]):
            continue
        out.append(i)
    return out


IDX = {sp: usable(sp) for sp in ("train", "val", "test")}
if LIMIT:
    big = max(IDX["train"][::max(1, len(IDX["train"]) // 300)],
              key=lambda i: int(np.sum(indptr[stores["train"].nodes(i) + 1] - indptr[stores["train"].nodes(i)])))
    IDX = {sp: v[:LIMIT] for sp, v in IDX.items()}
    IDX["train"] = IDX["train"] + ([big] if big not in IDX["train"] else [])   # kiểm VRAM với subgraph rất lớn
    log(f"  ⚠️ LIMIT={LIMIT} — KIỂM THỬ (thêm query train {big} có subgraph lớn)")
for sp, v in IDX.items():
    log(f"  {sp} dùng được: {len(v):,}")


def prep(sp, i):
    """Một query: toàn bộ cạnh subgraph, one-hot topic, DDE (trên toàn subgraph), nhãn từng cạnh."""
    st = stores[sp]
    nodes, s, d_, r = st.subgraph(i)
    loc = np.full(n_nodes, -1, dtype=np.int64)
    loc[nodes] = np.arange(len(nodes))
    sm = np.zeros(len(nodes), dtype=np.float32)
    sd = loc[st.seeds(i)]
    sm[sd[sd >= 0]] = 1.0
    oh = topic_one_hot(torch.from_numpy(sm))
    st_, dt_ = torch.from_numpy(s), torch.from_numpy(d_)
    dde = compute_dde(len(nodes), st_, dt_, oh)
    qpos, key, off = labs[sp]
    j = qpos[int(st.qids[i])]
    ekey = (nodes[s].astype(np.int64) * R + r) * n_nodes + nodes[d_]
    y = np.isin(ekey, key[off[j]:off[j + 1]]).astype(np.float32)
    an = loc[st.answers(i)]
    return dict(nodes=nodes, src=st_, dst=dt_, rel=torch.from_numpy(r), oh=oh, dde=dde,
                y=torch.from_numpy(y), qi=qs[sp][1][int(st.qids[i])], ans=an[an >= 0])


def prefetch(sp, idxs, depth=3):
    """prep() ở luồng nền (trích subgraph ~0.1s/query trên CPU) trong khi GPU chấm query trước."""
    qu = queue.Queue(maxsize=depth)

    def work():
        try:
            for i in idxs:
                qu.put((i, prep(sp, i)))
        except Exception as e:  # noqa: BLE001 — trả lỗi về luồng chính
            qu.put((None, e))
        qu.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := qu.get()) is not None:
        if item[0] is None:
            raise item[1]
        yield item


def to_gpu(sp, b):
    cls, _ = qs[sp]
    g = lambda x: x.to(dev, non_blocking=True)  # noqa: E731
    return (torch.from_numpy(cls[b["qi"]].astype(np.float32)).to(dev),
            torch.from_numpy(NODEEMB[b["nodes"]].astype(np.float32)).to(dev),
            g(b["src"]), g(b["dst"]), g(b["rel"]), g(b["oh"]), g(b["dde"]), g(b["y"]))


model = SubgraphRAGRetriever(EMB, topic_pe=True, num_rounds=2, num_reverse_rounds=2).to(dev)
npar = sum(p.numel() for p in model.parameters())
log(f"\nmô hình {npar/1e6:.2f}M tham số\n")
opt = torch.optim.Adam(model.parameters(), lr=LR)


@torch.no_grad()
def evaluate(sp, idxs, k=100):
    """triple_recall@100 (chỉ số dừng sớm gốc) + ans_recall@100, đúng eval_epoch gốc."""
    model.eval()
    tr, ar, t0 = [], [], time.time()
    for _, b in prefetch(sp, idxs):
        if b["y"].sum() == 0:
            continue
        q, ne, s, d_, r, oh, dde, y = to_gpu(sp, b)
        lg = model(q, ne, rel_emb, s, d_, r, oh, dde=dde, edge_chunk=EDGE_CHUNK)
        top = torch.topk(lg, min(k, len(lg))).indices
        tr.append(float(y[top].sum().item()) / float(y.sum().item()))
        if len(b["ans"]):
            ent = torch.zeros(len(b["nodes"]), device=dev)
            ent[s[top]] = 1.0
            ent[d_[top]] = 1.0
            ar.append(float(ent[torch.from_numpy(b["ans"]).to(dev)].mean().item()))
    model.train()
    return float(np.mean(tr)) if tr else 0.0, float(np.mean(ar)) if ar else 0.0, len(tr), time.time() - t0


def pos_weight(y):
    npos = float(y.sum().item())
    if POSW == "none" or npos == 0:
        return None
    return torch.tensor(((len(y) - npos) / npos) ** 0.5, device=dev)


def train_epoch():
    """= train_epoch gốc: batch 1 query, xáo bằng RandomSampler (DataLoader shuffle=True dùng nó)."""
    model.train()
    order = [IDX["train"][k] for k in torch.utils.data.RandomSampler(range(len(IDX["train"])))]
    tot, n_e, t0, steps = 0.0, 0, time.time(), []
    for _, b in prefetch("train", order):
        ts = time.time()
        q, ne, s, d_, r, oh, dde, y = to_gpu("train", b)
        opt.zero_grad()
        tot += chunked_bce_backward(model, q, ne, rel_emb, s, d_, r, oh, dde, y, pos_weight=pos_weight(y),
                                    edge_chunk=EDGE_CHUNK)
        opt.step()
        if SMOKE:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            steps.append((len(s), time.time() - ts))
        n_e += len(s)
    return tot / max(len(order), 1), time.time() - t0, n_e, steps


def save_state(epoch, best, best_ep, n_pat, hist):
    st = dict(model=model.state_dict(), optim=opt.state_dict(), epoch=epoch, best=best, best_ep=best_ep,
              n_pat=n_pat, hist=hist, rng=(random.getstate(), np.random.get_state(), torch.get_rng_state(),
                                           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    torch.save(st, f"{OUT}/subgraphrag_{POSW}_state.pt.tmp")
    os.replace(f"{OUT}/subgraphrag_{POSW}_state.pt.tmp", f"{OUT}/subgraphrag_{POSW}_state.pt")


# ---- chạy tiếp từ phiên trước ----
start_ep, best, best_ep, n_pat, hist, resume_from = 0, 0.0, -1, 0, [], None
if RESUME:
    rd = os.environ.get("RESUME_DIR")
    if not rd:
        hs = sorted(glob.glob(f"/kaggle/input/**/subgraphrag_{POSW}_state.pt", recursive=True))
        rd = os.path.dirname(hs[0]) if hs else None
    assert rd and os.path.exists(f"{rd}/subgraphrag_{POSW}_state.pt"), "RESUME=1 nhưng không thấy state phiên trước"
    st = torch.load(f"{rd}/subgraphrag_{POSW}_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["optim"])
    start_ep, best, best_ep, n_pat, hist = st["epoch"], st["best"], st["best_ep"], st["n_pat"], st["hist"]
    random.setstate(st["rng"][0])
    np.random.set_state(st["rng"][1])
    torch.set_rng_state(st["rng"][2])
    if st["rng"][3] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"][3])
    if os.path.exists(f"{rd}/subgraphrag_{POSW}_best.pt") and os.path.realpath(rd) != os.path.realpath(OUT):
        shutil.copy(f"{rd}/subgraphrag_{POSW}_best.pt", f"{OUT}/subgraphrag_{POSW}_best.pt")
    resume_from = rd
    log(f"CHẠY TIẾP từ {rd}: epoch {start_ep} · best {best:.4f} (epoch {best_ep}) · patience {n_pat}/{PATIENCE}")
    assert os.environ.get("RESUME_EPOCH") in (None, str(start_ep)), \
        f"state phiên trước ở epoch {start_ep}, kỳ vọng {os.environ.get('RESUME_EPOCH')}"

# ---- vòng train y hệt train.py gốc ----
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
done, stopped_by_deadline, n_sess, ep_h, smoke_steps = False, False, 0, None, []
ep = start_ep
while True:
    if ep >= MAX_EPOCH:
        done = True
        break
    spent = (time.time() - T_START) / 3600
    if ep_h is not None and spent + 1.2 * ep_h + TEST_RESERVE_H > DEADLINE_H:
        stopped_by_deadline = True
        log(f"  hết hạn phiên: {spent:.2f}h + epoch ~{ep_h:.2f}h > {DEADLINE_H}h -> dừng trước epoch {ep}")
        break
    te = time.time()
    n_pat += 1
    vr, va, vn, vsec = evaluate("val", IDX["val"])
    star = ""
    if vr > best:
        n_pat, best, best_ep = 0, vr, ep
        torch.save(model.state_dict(), f"{OUT}/subgraphrag_{POSW}_best.pt")
        star = " *"
    loss, tsec, n_e, steps = train_epoch()
    smoke_steps += steps
    hist.append(dict(ep=ep, val_triple_recall100=round(vr, 4), val_ans_recall100=round(va, 4), loss=round(loss, 6),
                     val_sec=round(vsec, 1), train_sec=round(tsec, 1), train_edges=n_e))
    ep += 1
    n_sess += 1
    save_state(ep, best, best_ep, n_pat, hist)
    ep_h = (time.time() - te) / 3600
    peak = torch.cuda.max_memory_allocated() / 1024**3 if dev.type == "cuda" else 0
    log(f"  epoch {ep - 1:>2}: val (trước train) triple_recall@100 {vr*100:6.2f}% ans_recall@100 {va*100:6.2f}%{star}"
        f" · loss {loss:.5f} · val {vsec:.0f}s + train {tsec:.0f}s ({n_e/1e6:.1f}M cạnh) · VRAM đỉnh {peak:.2f} GB")
    if n_pat == PATIENCE:
        done = True
        log(f"  dừng sớm: {PATIENCE} epoch không cải thiện")
        break
    if SESSION_EPOCHS and n_sess >= SESSION_EPOCHS:
        break
train_h = (time.time() - T_START) / 3600
log(f"\nphiên này {n_sess} epoch · tổng {ep} epoch · tốt nhất epoch {best_ep} val triple_recall@100 {best*100:.2f}%"
    f" · {'XONG' if done else 'CHƯA XONG'}")

smoke_info = None
if SMOKE and smoke_steps:
    ne_ = np.array([x[0] for x in smoke_steps], dtype=np.float64)
    sec_ = np.array([x[1] for x in smoke_steps])
    per_m = sec_.sum() / ne_.sum() * 1e6
    full_tr = usable("train")
    est_edges = np.mean([len(stores["train"].subgraph(i)[1]) for i in full_tr[::max(1, len(full_tr) // 200)]])
    est_train_h = per_m * est_edges * len(full_tr) / 1e6 / 3600
    smoke_info = dict(train_sec_per_million_edges=round(per_m, 3), max_edges_step=int(ne_.max()),
                      sec_max_step=round(float(sec_[ne_.argmax()]), 2), mean_edges_full_train=round(float(est_edges)),
                      n_train_full=len(full_tr), est_train_hours_per_epoch=round(est_train_h, 2))
    log(f"SMOKE: {per_m:.2f}s / 1M cạnh (fwd+bwd) · bước lớn nhất {int(ne_.max()):,} cạnh {sec_[ne_.argmax()]:.1f}s"
        f" · train đủ {len(full_tr):,} query × ~{est_edges:,.0f} cạnh -> ~{est_train_h:.2f}h/epoch (chưa tính val)")

# ---- suy luận test bằng checkpoint tốt nhất ----
ck = f"{OUT}/subgraphrag_{POSW}_best.pt"
if os.path.exists(ck):
    model.load_state_dict(torch.load(ck, map_location=dev))
model.eval()
qid_l, node_l, score_l, off, t0 = [], [], [], [0], time.time()
with torch.no_grad():
    for i, b in prefetch("test", IDX["test"]):
        q, ne, s, d_, r, oh, dde, _y = to_gpu("test", b)
        lg = model(q, ne, rel_emb, s, d_, r, oh, dde=dde, edge_chunk=EDGE_CHUNK)
        ns = triple_to_node_scores(len(b["nodes"]), s, d_, lg)
        qid_l.append(int(stores["test"].qids[i]))
        node_l.append(b["nodes"].astype(np.int32))
        score_l.append(ns.float().cpu().numpy().astype(np.float32))
        off.append(off[-1] + len(b["nodes"]))
np.savez_compressed(f"{OUT}/subgraphrag_{POSW}_test_pred.npz", qids=np.array(qid_l, dtype=np.int64),
                    node_flat=np.concatenate(node_l), score_flat=np.concatenate(score_l),
                    off=np.array(off, dtype=np.int64))
test_sec = time.time() - t0
log(f"đã lưu dự đoán {len(qid_l):,} query test ({test_sec/60:.1f} phút)")

json.dump(dict(posw=POSW, emb=EMB, lr=LR, best_epoch=best_ep, best_val_triple_recall100=best, history=hist,
               n_params=npar, epochs_run=ep, done=done, stopped_by_deadline=stopped_by_deadline,
               start_epoch_this_session=start_ep, resume_from=resume_from, seed=SEED, patience=PATIENCE,
               edge_chunk=EDGE_CHUNK, limit=LIMIT, smoke=smoke_info, test_sec=round(test_sec, 1),
               session_hours=round((time.time() - T_START) / 3600, 3), train_hours=round(train_h, 3),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2) if dev.type == "cuda" else None,
               code_sha=os.environ.get("CODE_SHA"),
               protocol=dict(labels="đường ngắn nhất 2 chiều topic<->đáp án theo từng cặp, như code gốc",
                             train_edges="toàn bộ cạnh subgraph", model="Retriever gốc (one-hot 2 chiều, DDE 2 kênh)",
                             loop="train.py gốc: val trước train, patience 10, seed 42")),
          open(f"{OUT}/subgraphrag_{POSW}_meta.json", "w"), indent=2, ensure_ascii=False)
log("SUBGRAPHRAG_DONE")
