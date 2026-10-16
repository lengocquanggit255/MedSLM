"""Đẩy phiên kế tiếp của GNN-RAG A+ khi phiên trước xong (chuỗi nhiều phiên, ritek/PLAN.md P4).

Chạy: python ritek/scripts/gnn_next_session.py --prev ritek-gnnrag-rearev-s2 --next ritek-gnnrag-rearev-s3 \
          --account minhduonngg --code-sha <sha>

Chờ `--prev` (cùng account) xong, đọc output đã tải về ritek/outputs/kaggle/<prev>/rearev_meta.json:
  - phiên trước lỗi, hoặc đã đủ epoch -> dừng, không đẩy gì.
  - còn epoch -> đẩy `--next` với RESUME=1, gắn output phiên trước làm input (job["kernels"]),
    DEADLINE_H = min(11, quota còn lại - 0.8h) để không bị Kaggle cắt vì hết quota giữa epoch.
Rồi chạy dispatcher cho phiên mới (chờ xong, tải output về).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_run import ACCOUNTS, quota_left, status_of  # noqa: E402

MIN_HOURS = 2.0      # ít hơn thế thì không đáng mở phiên (tải KG + suy luận test đã ~0.4h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--next", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()
    key = dict(ACCOUNTS)[a.account]

    while (st := status_of(a.account, key, a.prev)) in ("RUNNING", "QUEUED", "UNKNOWN"):
        time.sleep(a.poll)
    meta_f = ROOT / "ritek/outputs/kaggle" / a.prev / "rearev_meta.json"
    for _ in range(30):                 # dispatcher của phiên trước tải output về sau khi COMPLETE
        if meta_f.exists():
            break
        time.sleep(60)
    if st != "COMPLETE" or not meta_f.exists():
        sys.exit(f"{a.prev}: {st}, meta {'có' if meta_f.exists() else 'KHÔNG có'} -> dừng chuỗi, cần xem tay")
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    print(f"{a.prev}: epoch {meta['epochs_run']}/{meta['num_epoch']} · best dev H1 {meta['best_dev_h1']:.4f} · "
          f"2 GPU {meta.get('dual_gpu')}", flush=True)
    if meta["done"]:
        print("đã đủ epoch -> không đẩy phiên mới", flush=True)
        return
    q = quota_left(a.account, key)
    deadline = min(11.0, q - 0.8)
    if deadline < MIN_HOURS:
        sys.exit(f"{a.account} chỉ còn {q:.2f}h quota -> không đẩy, cần chuyển account")
    job = [{"slug": a.next, "code": "ritek/kaggle/gnnrag_rearev/gnnrag_rearev.py",
            "args": {"NUM_EPOCH": str(meta["num_epoch"]), "BATCH": "8", "TEST_BATCH": "4",
                     "DEADLINE_H": f"{deadline:.2f}", "RESUME": "1", "CODE_SHA": a.code_sha},
            "datasets": ["ritek-gnn-data"], "kernels": [a.prev], "accounts": [a.account], "timeout": 43200}]
    jf = ROOT / "ritek/kaggle/jobs" / f"{a.next.replace('ritek-', '').replace('-', '_')}.json"
    jf.write_text(json.dumps(job, indent=1), encoding="utf-8")
    print(f"quota {q:.2f}h -> DEADLINE_H {deadline:.2f} · đẩy {a.next} ({jf.name})", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_run.py"), "--jobs", str(jf), "--outdir",
                    "ritek/outputs/kaggle", "--workdir", ".cache/kaggle_jobs", "--poll", "600"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
