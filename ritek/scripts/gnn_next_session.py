"""Đẩy phiên kế tiếp của GNN-RAG A+ khi phiên trước xong (chuỗi nhiều phiên, ritek/PLAN.md P4).

Chạy: python ritek/scripts/gnn_next_session.py --prev ritek-gnnrag-rearev-s2 --prev-account minhduonngg \
          --next ritek-gnnrag-rearev-s3 --account minhduonngg --code-sha <sha>

Chờ `--prev` xong, đọc output đã tải về ritek/outputs/kaggle/<prev>/rearev_meta.json:
  - phiên trước lỗi, hoặc đã đủ epoch -> dừng, không đẩy gì.
  - còn epoch -> đẩy `--next` (RESUME=1, RESUME_EPOCH = epoch phiên trước đã xong).
Hạn phiên cố định DEADLINE_H = 11.5: Kaggle cho phiên đã bắt đầu chạy hết 12h dù quota cạn giữa
chừng, nên chỉ cần account còn quota > 0 lúc đẩy. Kernel dừng train ở cuối epoch để còn chỗ suy luận
test, nên phiên xong trước mốc 12h.
Nguồn state: cùng account -> gắn thẳng output kernel trước (job["kernels"]); khác account -> upload
state + checkpoint (không kèm điểm test 1.4 GB) thành dataset `ritek-gnn-resume` của account mới.
Rồi chạy dispatcher cho phiên mới (chờ xong, tải output về).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_run import ACCOUNTS, env, kag, quota_left, status_of  # noqa: E402

DEADLINE_H = 11.5
RESUME_DS = "ritek-gnn-resume"
STATE_FILES = ("rearev_state.pt", "rearev_history.json", "rearev-h1.ckpt", "rearev-f1.ckpt",
               "rearev-final.ckpt", "rearev_meta.json")


def upload_resume(src: Path, user: str, key: str, epoch: int):
    """State phiên trước -> dataset private `ritek-gnn-resume` của account `user`; chờ tới khi sẵn sàng."""
    d = ROOT / "ritek/outputs/kaggle_data" / RESUME_DS
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in STATE_FILES:
        if (src / f).exists():
            shutil.copy(src / f, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": RESUME_DS, "id": f"{user}/{RESUME_DS}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    e = env(user, key)

    def kag(*args, timeout=3600):
        r = subprocess.run(["kaggle", *args], env=e, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    exists = "ready" in kag("datasets", "status", f"{user}/{RESUME_DS}", timeout=180).lower()
    out = kag("datasets", "version", "-p", str(d), "-m", f"state epoch {epoch}", "--dir-mode", "zip") if exists \
        else kag("datasets", "create", "-p", str(d), "--dir-mode", "zip")
    print(f"upload {RESUME_DS} -> {user}: {out.strip().splitlines()[-1:] if out.strip() else ''}", flush=True)
    for _ in range(60):                 # kernel đẩy lúc dataset chưa xử lý xong sẽ gắn bản cũ
        time.sleep(30)
        if "ready" in kag("datasets", "status", f"{user}/{RESUME_DS}", timeout=180).lower():
            return
    sys.exit(f"{RESUME_DS} trên {user} chưa sẵn sàng sau 30 phút")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--prev-account", required=True)
    ap.add_argument("--next", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()
    accs = dict(ACCOUNTS)
    pkey, key = accs[a.prev_account], accs[a.account]

    while (st := status_of(a.prev_account, pkey, a.prev)) in ("RUNNING", "QUEUED", "UNKNOWN"):
        time.sleep(a.poll)
    out_dir = ROOT / "ritek/outputs/kaggle" / a.prev
    meta_f = out_dir / "rearev_meta.json"
    have = lambda: meta_f.exists() and (out_dir / "rearev_state.pt").exists()  # noqa: E731
    for _ in range(5):                  # dispatcher của phiên trước (nếu còn) tải output về sau khi COMPLETE
        if have():
            break
        time.sleep(60)
    if st == "COMPLETE" and not have():  # không ai tải (dispatcher đã dừng) -> tự tải
        out_dir.mkdir(parents=True, exist_ok=True)
        kag(a.prev_account, pkey, "kernels", "output", f"{a.prev_account}/{a.prev}", "-p", str(out_dir), timeout=3600)
        print(f"đã tự tải output {a.prev} -> {out_dir}", flush=True)
    if st != "COMPLETE" or not meta_f.exists():
        sys.exit(f"{a.prev}: {st}, meta {'có' if meta_f.exists() else 'KHÔNG có'} -> dừng chuỗi, cần xem tay")
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    epoch = int(meta["epochs_run"])
    print(f"{a.prev}: epoch {epoch}/{meta['num_epoch']} · best dev H1 {meta['best_dev_h1']:.4f} · "
          f"2 GPU {meta.get('dual_gpu')}", flush=True)
    if meta["done"]:
        print("đã đủ epoch -> không đẩy phiên mới", flush=True)
        return
    q = quota_left(a.account, key)
    if not q > 0.1:
        sys.exit(f"{a.account} còn {q:.2f}h quota -> không đẩy, chọn account khác")

    job = {"slug": a.next, "code": "ritek/kaggle/gnnrag_rearev/gnnrag_rearev.py",
           "args": {"NUM_EPOCH": str(meta["num_epoch"]), "BATCH": "8", "TEST_BATCH": "4",
                    "DEADLINE_H": str(DEADLINE_H), "RESUME": "1", "RESUME_EPOCH": str(epoch),
                    "CODE_SHA": a.code_sha},
           "datasets": ["ritek-gnn-data"], "accounts": [a.account], "timeout": 43200}
    if a.account == a.prev_account:
        job["kernels"] = [a.prev]
    else:
        upload_resume(out_dir, a.account, key, epoch)
        job["datasets"].append(RESUME_DS)
    jf = ROOT / "ritek/kaggle/jobs" / f"{a.next.replace('ritek-', '').replace('-', '_')}.json"
    jf.write_text(json.dumps([job], indent=1), encoding="utf-8")
    print(f"{a.account} quota {q:.2f}h · DEADLINE_H {DEADLINE_H} · đẩy {a.next} từ epoch {epoch} ({jf.name})",
          flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_run.py"), "--jobs", str(jf), "--outdir",
                    "ritek/outputs/kaggle", "--workdir", ".cache/kaggle_jobs", "--poll", "600"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
