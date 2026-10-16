"""P2 (ritek/PLAN.md) — encode corpus STaRK-Prime bằng BGE-M3 (dense) trên Kaggle T4x2.

Quyết định Q5: chỉ dense (CLS rồi L2-normalize, đúng `BGEM3FlagModel` của FlagEmbedding),
cửa sổ 8192 token. Doc dài hơn cửa sổ thì cắt chunk 8190 token + <s></s>, không chồng lấn
(`starkbench.chunking.build_chunks`). Điểm node (max / top3_mean) tính ở local.

Kernel chỉ xuất embedding, không xuất ma trận điểm: embedding dùng lại được cho P5
(MoR + BGE-M3) và nhẹ hơn nhiều (265 MB thay vì 0.7 GB cho mỗi cách gộp).

Input (Kaggle dataset `stark-prime-corpus`): docs/{no_rel,add_rel}.bin + .offsets.npy,
qa/stark_qa.csv (mọi query STaRK; RiTeK train/test là tập con, cùng id và cùng text),
src/starkbench/chunking.py.

Biến môi trường: VAR (no_rel|add_rel) · SMOKE (1 = vài trăm doc) · WINDOW (8192) ·
TOK_BUDGET (token/batch/GPU, tính cả padding) · CODE_SHA.

Output (/kaggle/working):
    chunkemb_bgem3_<var>.npy     (n_chunk, 1024) fp16, đã L2-normalize
    chunk2node_bgem3_<var>.npy   (n_chunk,) int32, sắp tăng dần
    qemb_bgem3.npy, qids_all.npy mọi query trong stark_qa.csv
    meta_bgem3_<var>.json
"""
import copy
import glob
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import torch

MODEL = "BAAI/bge-m3"
OUT = "/kaggle/working"
N_NODES = 129375


def log(*a):
    print(*a, flush=True)


def find_dataset_root(slug="stark-prime-corpus"):
    # Kaggle mount dataset ở /kaggle/input/datasets/<owner>/<slug>, không phải /kaggle/input/<slug>.
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/*/{slug}", f"/kaggle/input/*/*/{slug}"):
        for c in sorted(glob.glob(p)):
            if os.path.isdir(os.path.join(c, "docs")):
                return c
    raise FileNotFoundError(f"không thấy dataset '{slug}': {glob.glob('/kaggle/input/*/*')}")


def load_corpus(root, variant):
    offs = np.load(f"{root}/docs/{variant}.offsets.npy")
    raw = open(f"{root}/docs/{variant}.bin", "rb").read()
    return [raw[offs[i]:offs[i + 1]].decode("utf-8") for i in range(len(offs) - 1)]


def make_batches(lengths, budget):
    """Chia chỉ số chunk thành batch, dài trước. Mỗi batch: số chuỗi × độ dài lớn nhất ≤ budget."""
    order = np.argsort(-np.asarray(lengths), kind="stable")
    out, i = [], 0
    while i < len(order):
        L = int(lengths[order[i]]) + 2
        n = max(1, budget // L)
        out.append(order[i:i + n])
        i += n
    return out


def encode(models, tk, seqs, batches, dim, tag):
    """Encode `seqs` (list token id, chưa có special token) -> (len(seqs), dim) fp16, L2-normalize.

    Mỗi GPU một bản model và một luồng, cùng lấy batch từ một hàng đợi: GPU nào rảnh thì lấy tiếp.
    Không dùng DataParallel vì nó chép lại toàn bộ trọng số sang GPU phụ ở mỗi lần forward."""
    out = np.empty((len(seqs), dim), dtype=np.float16)
    q = queue.Queue()
    for b in batches:
        q.put(b)
    n_tok = sum(len(s) + 2 for s in seqs)
    done = {"tok": 0, "batch": 0}
    errors = []
    lock = threading.Lock()
    t0 = time.time()

    def worker(dev, model):
        torch.cuda.set_device(dev)
        try:
            while True:
                try:
                    b = q.get_nowait()
                except queue.Empty:
                    return
                L = max(len(seqs[j]) for j in b) + 2
                ids = np.full((len(b), L), tk.pad_token_id, dtype=np.int64)
                att = np.zeros((len(b), L), dtype=np.int64)
                for r, j in enumerate(b):
                    s = [tk.cls_token_id] + list(seqs[j]) + [tk.sep_token_id]
                    ids[r, :len(s)] = s
                    att[r, :len(s)] = 1
                with torch.inference_mode():
                    h = model(input_ids=torch.from_numpy(ids).to(dev),
                              attention_mask=torch.from_numpy(att).to(dev)).last_hidden_state[:, 0]
                    v = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
                out[b] = v.astype(np.float16)
                with lock:
                    done["tok"] += int(att.sum())
                    done["batch"] += 1
                    if done["batch"] % 200 == 0:
                        el = time.time() - t0
                        rate = done["tok"] / max(el, 1e-9)
                        log(f"  [{tag}] {done['tok']/1e6:.1f}M/{n_tok/1e6:.1f}M token "
                            f"({rate:,.0f} tok/s, còn ~{(n_tok-done['tok'])/max(rate,1e-9)/60:.1f} phút)")
        except Exception as e:  # noqa: BLE001 — chuyển lỗi về luồng chính
            errors.append(e)

    th = [threading.Thread(target=worker, args=(m.device, m)) for m in models]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise errors[0]
    sec = time.time() - t0
    log(f"  [{tag}] xong {len(seqs):,} chuỗi, {n_tok/1e6:.1f}M token, {sec/60:.1f} phút "
        f"({n_tok/max(sec,1e-9):,.0f} tok/s)")
    return out, sec, n_tok


def check_padding(models, tk, seqs, dim):
    """Cùng một chuỗi encode một mình và encode trong batch có padding phải cho cùng vector.

    Bắt lỗi mask/padding (vd attention bỏ qua mask) trước khi tốn hàng giờ GPU."""
    idx = list(range(len(seqs)))
    alone = np.stack([encode(models[:1], tk, seqs, [np.array([i])], dim, "check")[0][i] for i in idx])
    batch = encode(models[:1], tk, seqs, [np.array(idx)], dim, "check")[0]
    cos = (alone.astype(np.float32) * batch.astype(np.float32)).sum(1)
    log(f"kiểm padding: cos(một mình, trong batch) min = {cos.min():.5f}")
    assert cos.min() > 0.999, f"padding làm đổi embedding: {cos}"
    return float(cos.min())


def main():
    var = os.environ.get("VAR", "no_rel")
    smoke = os.environ.get("SMOKE", "0") == "1"
    window = int(os.environ.get("WINDOW", "8192"))
    budget = int(os.environ.get("TOK_BUDGET", "16384"))
    from transformers import AutoModel, AutoTokenizer
    import transformers

    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "cần GPU"
    log(f"GPU: {n_gpu} x {torch.cuda.get_device_name(0)} · torch {torch.__version__} · "
        f"transformers {transformers.__version__}")
    root = find_dataset_root()
    sys.path.insert(0, os.path.join(root, "src"))
    from starkbench.chunking import build_chunks

    tk = AutoTokenizer.from_pretrained(MODEL)
    # T4 (Turing) không có bf16 -> fp16, như FlagEmbedding mặc định (use_fp16=True).
    # sdpa trên sm75 dùng kernel memory-efficient: bộ nhớ attention tuyến tính theo độ dài.
    try:
        base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa")
        attn = "sdpa"
    except (ValueError, ImportError) as e:
        log(f"sdpa không dùng được ({e}) -> eager")
        base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16)
        attn = "eager"
    base.eval()
    dim = base.config.hidden_size
    assert base.config.max_position_embeddings - 2 >= window, base.config.max_position_embeddings
    models = [base.to("cuda:0")]
    for d in range(1, n_gpu):
        models.append(copy.deepcopy(base).to(f"cuda:{d}"))
    revision = getattr(base.config, "_commit_hash", None)
    log(f"{MODEL} @ {revision} · attn={attn} · dim={dim} · cửa sổ {window} · budget {budget} token/batch")

    docs = load_corpus(root, var)
    assert len(docs) == N_NODES
    if smoke:
        # Doc dài nhất (đường đi chunk > 1 và attention 8192) + một mẫu ngẫu nhiên.
        ln = np.array([len(d) for d in docs])
        keep = np.union1d(np.argsort(-ln)[:40], np.random.default_rng(0).choice(N_NODES, 400, replace=False))
        docs_used = [docs[i] for i in keep]
    else:
        keep = np.arange(N_NODES)
        docs_used = docs
    del docs

    t0 = time.time()
    chunks, owner = build_chunks(docs_used, tk, size=window, stride=window)
    chunk2node = keep[owner].astype(np.int32)
    lens = np.array([len(c) for c in chunks])
    n_multi = int((np.bincount(owner) > 1).sum())
    log(f"{var}: {len(docs_used):,} doc -> {len(chunks):,} chunk · {lens.sum()/1e6:.1f}M token · "
        f"{n_multi} doc > 1 chunk · chunk dài nhất {lens.max()} · tokenize {time.time()-t0:.0f}s")
    del docs_used

    short = [i for i in np.argsort(lens)[:6]] + [int(np.argmax(lens))]
    pad_cos = check_padding(models, tk, [chunks[i][:1024] for i in short], dim)

    emb, enc_sec, n_tok = encode(models, tk, chunks, make_batches(lens, budget), dim, var)
    assert np.isfinite(emb.astype(np.float32)).all(), "embedding có NaN/inf (tràn fp16?)"
    np.save(f"{OUT}/chunkemb_bgem3_{var}.npy", emb)
    np.save(f"{OUT}/chunk2node_bgem3_{var}.npy", chunk2node)
    del chunks, emb

    import pandas as pd
    qa = pd.read_csv(f"{root}/qa/stark_qa.csv")
    if smoke:
        qa = qa.head(64)
    qtok = tk(qa["query"].tolist(), add_special_tokens=False, truncation=True, max_length=window - 2)["input_ids"]
    qemb, _, _ = encode(models, tk, qtok, make_batches([len(x) for x in qtok], budget), dim, "query")
    assert np.isfinite(qemb.astype(np.float32)).all()
    np.save(f"{OUT}/qemb_bgem3.npy", qemb)
    np.save(f"{OUT}/qids_all.npy", qa["id"].to_numpy(dtype=np.int64))

    json.dump({
        "encoder": "bgem3", "model": MODEL, "revision": revision, "variant": var, "smoke": smoke,
        "pooling": "CLS + L2 (dense BGE-M3)", "window": window, "chunk_overlap": 0, "dtype": "fp16",
        "attn": attn, "n_docs": int(len(keep)), "n_chunks": int(len(chunk2node)), "n_docs_multi_chunk": n_multi,
        "n_tokens": int(n_tok), "encode_sec": round(enc_sec, 1), "tok_per_sec": round(n_tok / enc_sec),
        "n_queries": int(len(qa)), "padding_check_min_cos": pad_cos, "n_gpu": n_gpu,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "code_sha": os.environ.get("CODE_SHA"),
    }, open(f"{OUT}/meta_bgem3_{var}.json", "w"), indent=2)
    log("\nENCODE_DONE")


if __name__ == "__main__":
    main()
