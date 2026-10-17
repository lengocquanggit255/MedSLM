"""P3 (ritek/PLAN.md) — SubgraphRAG trên RiTeK: encode gte-large-en-v1.5 rồi train + suy luận.

Hai bước trong một kernel:
  1. Encode bằng `Alibaba-NLP/gte-large-en-v1.5`, đúng `GTELargeEN` của code gốc
     (retrieve/src/model/text_encoders/gte_large_en.py @ Graph-COM/SubgraphRAG 48eb05e):
     CLS của last_hidden_state, L2-normalize, max_length 8192, fp32, trust_remote_code.
     Text = tên node, tên quan hệ, câu hỏi (code gốc encode tên thực thể/quan hệ).
     Ghi ra rel_emb.npy · nodeemb.npy · q_{split}.npz (trường `cls`) — đúng định dạng
     kernel train đọc vào.
     Chạy ở TIẾN TRÌNH CON với transformers 4.x cài riêng. Code remote `new-impl` của gte KHÔNG
     chạy được với transformers 5.x có sẵn trên Kaggle (smoke 2026-10-16: config không nạp đúng,
     CUDA device-side assert ở token_type_embeddings). Code gốc ghim 4.43.2
     (retrieve/requirements/gte_large_en_v1-5.txt), nhưng nó cần tokenizers 0.19 — không có wheel
     cho Python 3.13 của Kaggle (smoke thứ hai: build từ source lỗi). Dùng 4.45.2, bản 4.x gần
     nhất ăn tokenizers 0.20.3 (có wheel cp313).
  2. Chạy kernel train `subgraphrag_train.py` (có trong dataset; vòng train theo code gốc, xem
     docstring file đó) trên subgraph PPR-2000 xuất phát từ topic RiTeK. Biến POSW/RESUME/...
     truyền thẳng qua os.environ; SESSION_T0 = lúc phiên bắt đầu, để hạn phiên tính cả thời gian encode.

Dataset: `ritek-sgrag-data` (ritek/scripts/prep_sgrag_data.py).
Smoke: SMOKE=1 LIMIT=<n> MAX_EPOCH=2 — encode đủ (để đo giờ), train + suy luận test trên n query.
"""
import glob
import json
import os
import runpy
import subprocess
import sys
import time

MODEL = "Alibaba-NLP/gte-large-en-v1.5"
WORK = "/kaggle/temp/sgrag_ds"
TF_DIR = "/kaggle/temp/tf4"
# transformers 4.45.2 cần tokenizers 0.20 và huggingface_hub < 1.0 (Kaggle có bản 1.x).
# --no-deps: chỉ ba gói này vào thư mục riêng, không che numpy/torch của hệ thống.
# --only-binary: thiếu wheel thì báo lỗi ngay, không thử build Rust.
PINS = ["transformers==4.45.2", "tokenizers==0.20.3", "huggingface_hub==0.24.7"]

ENCODE = r'''
import json, sys, time
import numpy as np, torch, transformers
from transformers import AutoModel, AutoTokenizer
MODEL, ds, work = sys.argv[1:4]


def log(*a):
    print(*a, flush=True)


tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
try:
    # đúng tham số code gốc; memory-efficient attention cần xformers
    md, info = AutoModel.from_pretrained(MODEL, trust_remote_code=True, unpad_inputs=True,
                                         use_memory_efficient_attention=True, output_loading_info=True)
    mode = "unpad + xformers (như code gốc)"
except Exception as e:  # noqa: BLE001
    log(f"không nạp được với xformers ({type(e).__name__}: {str(e)[:200]}) -> attention thường")
    md, info = AutoModel.from_pretrained(MODEL, trust_remote_code=True, output_loading_info=True)
    mode = "attention thường (thiếu xformers; cùng phép tính)"
# Trọng số thiếu = bị khởi tạo ngẫu nhiên -> embedding rác mà vẫn "chạy được". Phải chặn.
assert not info["missing_keys"] and not info["mismatched_keys"], info
md = md.eval().cuda()
rev = getattr(md.config, "_commit_hash", None)
log(f"{MODEL} @ {rev} · {mode} · transformers {transformers.__version__}")


@torch.no_grad()
def embed(texts, budget=32768):
    """GTELargeEN.embed theo lô: CLS, L2-normalize, fp32. Sắp theo độ dài chỉ để đỡ padding."""
    out = np.zeros((len(texts), md.config.hidden_size), dtype=np.float32)
    lens = np.array([len(t) for t in texts])
    order = np.argsort(-lens, kind="stable")
    i, t0 = 0, time.time()
    while i < len(order):
        L = min(8192, int(lens[order[i]]) // 2 + 8)          # ước lượng token từ ký tự, chỉ để chia lô
        bs = max(1, min(1024, budget // L))
        idx = order[i:i + bs]
        b = tk([texts[j] for j in idx], max_length=8192, padding=True, truncation=True,
               return_tensors="pt").to("cuda")
        h = md(**b).last_hidden_state[:, 0]
        out[idx] = torch.nn.functional.normalize(h, p=2, dim=1).float().cpu().numpy()
        i += bs
    log(f"  embed {len(texts):,} text trong {time.time()-t0:.0f}s")
    return out


# Kiểm nhanh model có hiểu nghĩa không (ví dụ trong model card gte): câu hỏi phải gần đáp án hơn.
e = embed(["what is the capital of China?", "how to implement quick sort in python?",
           "Beijing", "sorting algorithms"])
s = e[:2] @ e[2:].T
log(f"kiểm model card: {np.round(s * 100, 2).tolist()}")
assert s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0], s

texts = json.load(open(f"{ds}/texts.json", encoding="utf-8"))
t0 = time.time()
np.save(f"{work}/rel_emb.npy", embed(texts["relations"]))
np.save(f"{work}/nodeemb.npy", embed(texts["node_names"]))
for sp, rows in texts["queries"].items():
    np.savez(f"{work}/q_{sp}.npz", cls=embed([q for _, q in rows]),
             qids=np.array([i for i, _ in rows], dtype=np.int64))
sec = time.time() - t0
json.dump({"model": MODEL, "revision": rev, "attention": mode, "encode_sec": round(sec, 1),
           "text": "tên node / tên quan hệ / câu hỏi", "pooling": "CLS + L2", "dtype": "fp32",
           "transformers": transformers.__version__, "model_card_check": (s * 100).round(2).tolist()},
          open("/kaggle/working/gte_meta.json", "w"), indent=2)
log(f"encode xong {sec/60:.1f} phút")
'''


def log(*a):
    print(*a, flush=True)


def find_ds():
    for depth in range(5):
        for c in sorted(glob.glob("/kaggle/input/" + "*/" * depth)):
            if os.path.exists(os.path.join(c, "texts.json")) and os.path.exists(os.path.join(c, "csr.npz")):
                return c.rstrip("/")
    raise FileNotFoundError("không thấy dataset ritek-sgrag-data")


def main():
    os.environ.setdefault("SESSION_T0", str(time.time()))
    ds = find_ds()
    log(f"dataset = {ds}")
    os.makedirs(WORK, exist_ok=True)
    for f in os.listdir(ds):
        if f != "texts.json":
            dst = os.path.join(WORK, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join(ds, f), dst)

    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary", ":all:",
                        "--target", TF_DIR, *PINS])
    assert r.returncode == 0, f"cài {PINS} lỗi"
    open("/kaggle/temp/encode_gte.py", "w", encoding="utf-8").write(ENCODE)
    env = dict(os.environ, PYTHONPATH=TF_DIR, TOKENIZERS_PARALLELISM="false")
    r = subprocess.run([sys.executable, "-u", "/kaggle/temp/encode_gte.py", MODEL, ds, WORK], env=env)
    assert r.returncode == 0, f"encode gte lỗi returncode={r.returncode}"
    m = json.load(open("/kaggle/working/gte_meta.json"))
    m["code_sha"] = os.environ.get("CODE_SHA")
    json.dump(m, open("/kaggle/working/gte_meta.json", "w"), indent=2)

    os.environ["DS_DIR"] = WORK
    runpy.run_path(f"{WORK}/subgraphrag_train.py", run_name="__main__")


if __name__ == "__main__":
    main()
