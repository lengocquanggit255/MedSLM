"""Kiểm chạy tiếp (RESUME) của kernel SubgraphRAG RiTeK: 2 epoch + nạp state + 2 epoch phải trùng 4 epoch
chạy liền (cùng lịch sử val/loss, cùng trọng số, cùng checkpoint tốt nhất, cùng dự đoán test).

Chạy: python ritek/scripts/test_sgrag_resume.py [--limit 3]
CPU, subgraph + nhãn THẬT (ritek/outputs/kaggle_data/ritek-sgrag-data), embedding text giả 16 chiều
(chỉ kiểm cơ chế lưu/nạp, không kiểm chất lượng). LIMIT còn thêm một query train có subgraph rất lớn,
EDGE_CHUNK nhỏ để đi qua đường backward theo khúc.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
SRC_DS = ROOT / "ritek/outputs/kaggle_data/ritek-sgrag-data"
KERNEL = ROOT / "ritek/kaggle/sgrag/subgraphrag_train.py"


def fake_ds(d: Path, labels: Path, dim=16):
    d.mkdir()
    for f in ("csr.npz", "train.npz", "val.npz", "test.npz"):
        shutil.copy(SRC_DS / f, d / f)
    for sp in ("train", "val", "test"):
        shutil.copy(labels / f"labels_{sp}.npz", d / f"labels_{sp}.npz")
    for f in ("src/starkbench/gnnrag.py", "src/starkbench/subgraphrag.py"):
        shutil.copy(ROOT / f, d / Path(f).name)
    rng = np.random.default_rng(0)
    c = np.load(SRC_DS / "csr.npz")
    norm = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)  # noqa: E731
    np.save(d / "rel_emb.npy", norm(rng.standard_normal((int(c["n_rel"]), dim))))
    np.save(d / "nodeemb.npy", norm(rng.standard_normal((int(c["n_nodes"]), dim))))
    for sp in ("train", "val", "test"):
        q = np.load(SRC_DS / f"{sp}.npz")["qids"]
        np.savez(d / f"q_{sp}.npz", cls=norm(rng.standard_normal((len(q), dim))), qids=q)


def run(ds, out, limit, **env):
    out.mkdir()
    e = dict(os.environ, DS_DIR=str(ds), OUT_DIR=str(out), DEVICE="cpu", CUDA_VISIBLE_DEVICES="", LIMIT=str(limit),
             MAX_EPOCH="4", POSW="sqrt", EDGE_CHUNK="50000", PYTHONUTF8="1",
             OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", **env)   # BLAS 1 luồng: phép cộng tất định
    r = subprocess.run([sys.executable, str(KERNEL)], env=e, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode:
        sys.exit(f"kernel lỗi ({env}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return json.load(open(out / "subgraphrag_sqrt_meta.json", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--labels", default=str(SRC_DS), help="thư mục labels_<split>.npz")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="sgrag_resume_"))
    ds = tmp / "ds"
    fake_ds(ds, Path(a.labels))
    A, B1, B2 = tmp / "A", tmp / "B1", tmp / "B2"
    ma = run(ds, A, a.limit)
    m1 = run(ds, B1, a.limit, SESSION_EPOCHS="2")
    assert m1["epochs_run"] == 2 and not m1["done"], m1
    m2 = run(ds, B2, a.limit, RESUME="1", RESUME_DIR=str(B1), RESUME_EPOCH="2")
    assert m2["start_epoch_this_session"] == 2 and m2["epochs_run"] == 4 and m2["done"], m2

    keys = ("ep", "val_triple_recall100", "val_ans_recall100", "loss", "train_edges")
    ha = [{k: h[k] for k in keys} for h in ma["history"]]
    hb = [{k: h[k] for k in keys} for h in m2["history"]]
    print("liền:", [(h["ep"], h["val_triple_recall100"], h["loss"]) for h in ha])
    print("nối: ", [(h["ep"], h["val_triple_recall100"], h["loss"]) for h in hb])
    assert ha == hb, "lịch sử khác"
    assert (ma["best_epoch"], ma["best_val_triple_recall100"]) == (m2["best_epoch"], m2["best_val_triple_recall100"])
    sa = torch.load(A / "subgraphrag_sqrt_state.pt", weights_only=False)
    sb = torch.load(B2 / "subgraphrag_sqrt_state.pt", weights_only=False)
    d_w = max(float((sa["model"][k] - sb["model"][k]).abs().max()) for k in sa["model"])
    ba = torch.load(A / "subgraphrag_sqrt_best.pt")
    bb = torch.load(B2 / "subgraphrag_sqrt_best.pt")
    d_b = max(float((ba[k] - bb[k]).abs().max()) for k in ba)
    pa = np.load(A / "subgraphrag_sqrt_test_pred.npz")
    pb = np.load(B2 / "subgraphrag_sqrt_test_pred.npz")
    same_pred = all(np.array_equal(pa[k], pb[k]) for k in pa.files)
    print(f"trọng số cuối Δ {d_w:.3g} · checkpoint tốt nhất Δ {d_b:.3g} · dự đoán test trùng {same_pred} · "
          f"cạnh train/epoch {ha[0]['train_edges']:,}")
    assert d_w == 0.0 and d_b == 0.0 and same_pred
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
