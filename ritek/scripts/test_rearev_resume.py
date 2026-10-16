"""Kiểm cơ chế chạy tiếp (RESUME) của kernel GNN-RAG A+: 2 epoch + nạp state + 2 epoch phải trùng
4 epoch chạy liền (cùng trọng số, cùng loss từng epoch). CPU, toàn KG thật, vài query.

Chạy: python ritek/scripts/test_rearev_resume.py [--smoke 4]
Cần: ritek/outputs/kaggle_data/ritek-gnn-data (ritek/scripts/prep_gnn_data.py), clone repo gốc ở
.cache/gnnrag_repo (kernel tự clone nếu thiếu).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "ritek/kaggle/gnnrag_rearev/gnnrag_rearev.py"


def run(out, smoke, **env):
    e = dict(os.environ, OUT_DIR=str(out), DS_DIR=str(ROOT / "ritek/outputs/kaggle_data/ritek-gnn-data"),
             REPO_DIR=str(ROOT / ".cache/gnnrag_repo"), SMOKE=str(smoke), NUM_EPOCH="4", BATCH="2",
             TEST_BATCH="2", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="", **env)
    r = subprocess.run([sys.executable, str(KERNEL)], env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        sys.exit(f"kernel lỗi ({env}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return json.load(open(out / "rearev_meta.json", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=4, help="số query mỗi split")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="rearev_resume_"))
    A, B1, B2 = tmp / "A", tmp / "B1", tmp / "B2"
    for d in (A, B1, B2):
        d.mkdir()
    run(A, a.smoke)
    m1 = run(B1, a.smoke, SESSION_EPOCHS="2")
    assert m1["epochs_run"] == 2 and not m1["done"], m1
    m2 = run(B2, a.smoke, RESUME="1", RESUME_DIR=str(B1))
    assert m2["resume_mode"].startswith("full") and m2["start_epoch_this_session"] == 2 and m2["done"], m2

    ha = json.load(open(A / "rearev_history.json"))
    hb = json.load(open(B2 / "rearev_history.json"))
    assert [h["epoch"] for h in hb] == [1, 2, 3, 4], hb
    for x, y in zip(ha, hb):
        assert x["loss"] == y["loss"] and x.get("dev_h1") == y.get("dev_h1"), (x, y)
    sa = torch.load(A / "rearev_state.pt", weights_only=False)["model"]
    sb = torch.load(B2 / "rearev_state.pt", weights_only=False)["model"]
    diff = max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa)
    print(f"loss từng epoch: liền {[round(h['loss'], 6) for h in ha]}")
    print(f"                 nối  {[round(h['loss'], 6) for h in hb]}")
    print(f"trọng số cuối: Δ max = {diff:.3g} trên {len(sa)} tensor")
    assert diff == 0.0, "chạy nối KHÁC chạy liền"

    # nối từ phiên kiểu cũ (chỉ có rearev-final.ckpt): phải chạy được, ghi rõ chế độ
    (B1 / "rearev_state.pt").unlink()
    (tmp / "B3").mkdir()
    m3 = run(tmp / "B3", a.smoke, RESUME="1", RESUME_DIR=str(B1))
    assert m3["resume_mode"].startswith("weights_only") and m3["done"], m3
    print(f"nối từ phiên cũ (không state): {m3['resume_mode']} · epoch {m3['start_epoch_this_session']}→{m3['epochs_run']}")
    print("PASS")


if __name__ == "__main__":
    main()
