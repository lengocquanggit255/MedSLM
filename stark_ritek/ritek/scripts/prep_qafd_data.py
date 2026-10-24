"""Dựng (và upload) Kaggle dataset `ritek-qafd-data` cho QAFD-RAG retriever-only trên RiTeK (kernel ritek/kaggle/qafd).

Chạy: python ritek/scripts/prep_qafd_data.py [--upload minhduonngg]

Theo notebook tham chiếu ref_qafd/qafd-rag.ipynb (PharmKG). Mỗi bộ một thư mục, cùng một định dạng:
  triples.npz     h · r · t (int64) đúng THỨ TỰ DÒNG của file KG RiTeK — notebook dựng đồ thị theo thứ tự này
                  (thứ tự node = lần xuất hiện đầu tiên), và thứ tự node ảnh hưởng tới bước chọn node ngẫu nhiên.
  node_info.json  {id: {name, type, description}} — thứ notebook đọc từ node_info.pkl.
  test.jsonl      file test RiTeK `*_with_topic_entity.json`, chép nguyên.

pharmkg/ (bước S0: tái tạo số của notebook): Pharm_KG.txt + node_info.pkl của gói metadata RiTeK trên HuggingFace
  (Graph_node_edge_info/pharmKG-*.zip, data/ritek/pharmkg_skb/) — name/type/description GIỮ NGUYÊN (kể cả nhãn loại
  lẫn hoa thường), như notebook.
prime/: prime_KG.txt; name/type từ node_info.pkl của gói metadata Prime RiTeK (data/ritek/prime_skb/, = node_info
  STaRK). node_info Prime KHÔNG có `description` (chỉ có `details`) -> quyết định D5(b) 2026-10-24: description =
  văn bản `no_rel` của STaRK (starkbench.corpus) bỏ hai dòng `- name:` và `- type:` (giữ `- source:` + `- details:`).
"""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "ritek/outputs/kaggle_data/ritek-qafd-data"
SLUG = "ritek-qafd-data"


def read_triples(path):
    a = np.loadtxt(path, dtype=np.int64)
    assert a.ndim == 2 and a.shape[1] == 3, a.shape
    return a


def write(ds, triples, node_info, test_src):
    d = OUT / ds
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "triples.npz", h=triples[:, 0], r=triples[:, 1], t=triples[:, 2])
    (d / "node_info.json").write_text(json.dumps(node_info, ensure_ascii=False), encoding="utf-8")
    shutil.copy(test_src, d / "test.jsonl")
    nodes = np.unique(triples[:, [0, 2]])
    assert all(str(n) in node_info for n in nodes.tolist()), "node trong KG thiếu node_info"
    print(f"{ds}: {len(triples):,} triple · {len(nodes):,} node · node_info {len(node_info):,}", flush=True)


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    # --- PharmKG: y như notebook ---
    ni = pickle.load(open(ROOT / "data/ritek/pharmkg_skb/pharmKG/node_info.pkl", "rb"))
    write("pharmkg", read_triples(ROOT / "data/ritek/Dataset/pharmKG/Pharm_KG.txt"),
          {str(k): {"name": v.get("name"), "type": v.get("type"), "description": v.get("description")}
           for k, v in ni.items()},
          ROOT / "data/ritek/Dataset/pharmKG/pharmKG_test_QA_with_topic_entity.json")
    # --- Prime ---
    from starkbench import corpus
    ni = pickle.load(open(ROOT / "data/ritek/prime_skb/prime/node_info.pkl", "rb"))
    docs = corpus.load("no_rel")
    assert len(docs) == len(ni) == 129375
    info = {}
    for i in range(len(docs)):
        lines = docs[i].split("\n")
        assert lines[0] == f"- name: {ni[i]['name']}" and lines[1] == f"- type: {ni[i]['type']}", (i, lines[:2])
        info[str(i)] = {"name": ni[i]["name"], "type": ni[i]["type"], "description": "\n".join(lines[2:]).strip()}
    write("prime", read_triples(ROOT / "data/ritek/Dataset/prime/prime_KG.txt"), info,
          ROOT / "data/ritek/Dataset/prime/Prime_test_QA_with_topic_entity.json")


def upload(user):
    from kaggle_run import ACCOUNTS, env
    key = dict(ACCOUNTS)[user]
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    e = env(user, key)

    def kag(*args, timeout=3600):
        r = subprocess.run(["kaggle", *args], env=e, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    exists = "ready" in kag("datasets", "status", f"{user}/{SLUG}", timeout=180).lower()
    out = kag("datasets", "version", "-p", str(OUT), "-m", "cập nhật", "--dir-mode", "zip") if exists \
        else kag("datasets", "create", "-p", str(OUT), "--dir-mode", "zip")
    print(f"upload {SLUG} -> {user}: {out.strip().splitlines()[-1:] if out.strip() else ''}", flush=True)
    for _ in range(60):
        time.sleep(30)
        if "ready" in kag("datasets", "status", f"{user}/{SLUG}", timeout=180).lower():
            print("dataset sẵn sàng", flush=True)
            return
    sys.exit(f"{SLUG} trên {user} chưa sẵn sàng sau 30 phút")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", default=None, help="account Kaggle để upload (vd minhduonngg)")
    a = ap.parse_args()
    build()
    if a.upload:
        upload(a.upload)


if __name__ == "__main__":
    main()
