"""Dựng (và upload) Kaggle dataset `ritek-mor-data` cho P5 (ritek/kaggle/mor_ritek).

Chạy (sau khi P2 BGE-M3 no_rel xong): python ritek/scripts/prep_mor_data.py [--upload]

Nội dung (ritek/outputs/kaggle_data/ritek-mor-data/):
  ritek_mor_data.txt          mốc để kernel dò thư mục
  splits/*.index              split RiTeK (ritek/splits)
  contriever/candidate_emb.npy, query_emb_dict.pt
                              Contriever-FT của MoR — ĐÚNG file mà P1 dùng (results/kaggle/stark-mor-emb)
  bgem3/nodeemb_no_rel.npy    (129375, 1024) fp16: BGE-M3 dense của doc no_rel (như Contriever của
                              MoR dùng doc no_rel). Node nhiều chunk: trung bình chunk rồi L2-normalize.
  bgem3/qemb.npy, qids_all.npy
Ngoài dataset: ritek/outputs/bgem3_query_emb_dict.pt — dict query cho backfill ở
scripts/eval_mor_rerank.py --emb-q.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SLUG = "ritek-mor-data"
OUT = ROOT / "ritek/outputs/kaggle_data" / SLUG
CONTRIEVER = ROOT / "results/kaggle/stark-mor-emb"
BGEM3 = ROOT / "ritek/outputs/kaggle/ritek-bgem3-no-rel"
N = 129375


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "splits").mkdir(parents=True)
    (OUT / "contriever").mkdir()
    (OUT / "bgem3").mkdir()
    (OUT / "ritek_mor_data.txt").write_text("ritek-mor-data (ritek/scripts/prep_mor_data.py)\n", encoding="utf-8")
    for f in (ROOT / "ritek/splits").glob("*.index"):
        shutil.copy(f, OUT / "splits" / f.name)
    for f in ("candidate_emb.npy", "query_emb_dict.pt"):
        shutil.copy(CONTRIEVER / f, OUT / "contriever" / f)

    meta = json.loads((BGEM3 / "meta_bgem3_no_rel.json").read_text(encoding="utf-8"))
    assert meta["variant"] == "no_rel" and not meta["smoke"], meta
    E = np.load(BGEM3 / "chunkemb_bgem3_no_rel.npy").astype(np.float32)
    c2n = np.load(BGEM3 / "chunk2node_bgem3_no_rel.npy")
    assert np.array_equal(np.unique(c2n), np.arange(N)) and np.all(np.diff(c2n) >= 0)
    starts = np.flatnonzero(np.r_[True, c2n[1:] != c2n[:-1]])
    node = np.add.reduceat(E, starts, axis=0)
    node /= np.maximum(np.linalg.norm(node, axis=1, keepdims=True), 1e-9)
    np.save(OUT / "bgem3/nodeemb_no_rel.npy", node.astype(np.float16))
    qe, qi = np.load(BGEM3 / "qemb_bgem3.npy"), np.load(BGEM3 / "qids_all.npy")
    np.save(OUT / "bgem3/qemb.npy", qe)
    np.save(OUT / "bgem3/qids_all.npy", qi)
    torch.save({int(q): torch.from_numpy(qe[i].astype(np.float32)) for i, q in enumerate(qi)},
               ROOT / "ritek/outputs/bgem3_query_emb_dict.pt")
    print(f"BGE-M3 no_rel: {len(E):,} chunk -> {N:,} node · {len(qi):,} query", flush=True)
    mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"{OUT}: {mb:.0f} MB", flush=True)


def upload():
    from kaggle_run import ACCOUNTS, env
    for user, key in ACCOUNTS:     # tuần tự: dataset-metadata.json dùng chung một thư mục
        (OUT / "dataset-metadata.json").write_text(json.dumps(
            {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
        e = env(user, key)
        st = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"], env=e, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
        exists = "ready" in (st.stdout or "").lower()
        cmd = (["kaggle", "datasets", "version", "-p", str(OUT), "-m", "cập nhật", "--dir-mode", "zip"] if exists
               else ["kaggle", "datasets", "create", "-p", str(OUT), "--dir-mode", "zip"])
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=7200)
        print(user, r.returncode, ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    build()
    if a.upload:
        upload()
