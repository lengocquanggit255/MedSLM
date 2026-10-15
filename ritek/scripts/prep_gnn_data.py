"""Dựng (và upload) Kaggle dataset `ritek-gnn-data` cho kernel GNN-RAG A+ (ritek/kaggle/gnnrag_rearev).

Chạy: python ritek/scripts/prep_gnn_data.py [--upload]

Nội dung (ritek/outputs/kaggle_data/ritek-gnn-data/):
  edges.npz          src, dst (int32) · rel (int8) · num_nodes — đúng edge_index/edge_types của STaRK
                     (= prime_KG.txt của RiTeK, đã kiểm trùng thứ tự ở ritek/docs/ritek_prime_compare.json)
  relations.json     18 tên quan hệ theo id (edge_type_dict)
  Prime_{train,test}_QA_with_topic_entity.json   QA RiTeK nguyên bản
  splits/*.index     ritek/splits (val = 10% train, seed 42)
--upload: tạo/lên version dataset private trên MỌI account trong env/.env (kaggle_run.ACCOUNTS).
"""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SLUG = "ritek-gnn-data"
OUT = ROOT / "ritek/outputs/kaggle_data" / SLUG


def build():
    P = ROOT / "data/stark/prime/processed"
    ei = torch.load(P / "edge_index.pt", weights_only=False).numpy()
    et = torch.load(P / "edge_types.pt", weights_only=False).numpy()
    etd = pickle.load(open(P / "edge_type_dict.pkl", "rb"))
    n = int(torch.load(P / "node_types.pt", weights_only=False).numel())
    assert ei.max() < n and et.max() < 128
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "splits").mkdir(parents=True)
    np.savez_compressed(OUT / "edges.npz", src=ei[0].astype(np.int32), dst=ei[1].astype(np.int32),
                        rel=et.astype(np.int8), num_nodes=np.int64(n))
    json.dump([etd[i] for i in range(len(etd))], open(OUT / "relations.json", "w", encoding="utf-8"))
    for part in ("train", "test"):
        f = f"Prime_{part}_QA_with_topic_entity.json"
        shutil.copy(ROOT / "data/ritek/Dataset/prime" / f, OUT / f)
    for f in (ROOT / "ritek/splits").glob("*.index"):
        shutil.copy(f, OUT / "splits" / f.name)
    mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {ei.shape[1]:,} cạnh · {n:,} node · {len(etd)} quan hệ · {mb:.1f} MB")


def upload():
    from kaggle_run import ACCOUNTS, env

    def one(acc):
        user, key = acc
        (OUT / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
        e = env(user, key)
        st = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"], env=e, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
        exists = "ready" in (st.stdout or "").lower()
        cmd = (["kaggle", "datasets", "version", "-p", str(OUT), "-m", "cập nhật", "--dir-mode", "zip"] if exists
               else ["kaggle", "datasets", "create", "-p", str(OUT), "--dir-mode", "zip"])
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=3600)
        return user, r.returncode, ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:]

    # Tuần tự: dataset-metadata.json dùng chung một thư mục, ghi song song sẽ đè id của nhau.
    for acc in ACCOUNTS:
        print(one(acc), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    build()
    if a.upload:
        upload()
