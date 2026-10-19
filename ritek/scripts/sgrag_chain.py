"""Chạy SubgraphRAG RiTeK (kernel ritek/kaggle/sgrag) nhiều phiên trên MỘT account, tự nối phiên (ritek/PLAN.md P3).

Chạy: python ritek/scripts/sgrag_chain.py --posw none --account minhduonngg --slug ritek-sgrag2-none \
          --code-sha <sha> [--from 1]

Phiên k = kernel `<slug>-s<k>`. Phiên 1 train từ đầu; phiên k > 1 gắn output phiên k-1 (job["kernels"], cùng
account) với RESUME=1, RESUME_EPOCH = epochs_run của phiên trước. Mỗi phiên: kiểm quota > 0.1h, ghi job spec
ritek/kaggle/jobs/<slug>_s<k>.json, chạy dispatcher (chờ xong, tải output về ritek/outputs/kaggle/<slug>-s<k>),
đọc subgraphrag_<posw>_meta.json: `done` (hết patience) -> dừng; chưa xong -> phiên kế. Phiên lỗi hoặc thiếu
meta -> dừng, xem tay.
Hạn phiên DEADLINE_H 11.5 như GNN (ritek/scripts/gnn_next_session.py): Kaggle cho phiên đã bắt đầu chạy hết
12h dù quota cạn giữa chừng, nên chỉ cần quota > 0 lúc đẩy.
Máy ngủ giữa chừng (dispatcher chết): chạy lại với --from <k+1>, k = phiên đang chạy dở -> script chờ phiên k
xong trên Kaggle, tự tải output rồi nối tiếp.
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
from kaggle_run import ACCOUNTS, BUSY, bad_files, download_output, quota_left, status_of  # noqa: E402

DEADLINE_H = 11.5
OUTDIR = ROOT / "ritek/outputs/kaggle"


def prev_meta(user, key, slug, posw, poll):
    """Meta của phiên `slug`: chờ phiên xong trên Kaggle, tải output nếu máy chưa có. None nếu phiên lỗi."""
    out = OUTDIR / slug
    meta_f = out / f"subgraphrag_{posw}_meta.json"
    if not (meta_f.exists() and (out / f"subgraphrag_{posw}_state.pt").exists() and not bad_files(out)):
        while (st := status_of(user, key, slug)) in (*BUSY, "UNKNOWN"):
            time.sleep(poll)
        if st != "COMPLETE":
            print(f"{slug}: {st}", flush=True)
            return None
        if not download_output(user, key, slug, out, timeout=3600):
            print(f"{slug}: tải output vẫn ra trang lỗi", flush=True)
            return None
        print(f"đã tự tải output {slug} -> {out}", flush=True)
    return json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posw", required=True, choices=["none", "sqrt"])
    ap.add_argument("--account", required=True)
    ap.add_argument("--slug", required=True, help="tiền tố; phiên k = <slug>-s<k>")
    ap.add_argument("--code-sha", required=True)
    ap.add_argument("--from", dest="start", type=int, default=1, help="phiên bắt đầu đẩy (phiên trước đã/đang chạy)")
    ap.add_argument("--max-sessions", type=int, default=6)
    ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()
    key = dict(ACCOUNTS)[a.account]

    for k in range(a.start, a.max_sessions + 1):
        slug = f"{a.slug}-s{k}"
        args = {"POSW": a.posw, "DEADLINE_H": str(DEADLINE_H), "CODE_SHA": a.code_sha}
        job = {"slug": slug, "code": "ritek/kaggle/sgrag/sgrag.py", "args": args,
               "datasets": ["ritek-sgrag-data"], "accounts": [a.account], "timeout": 43200}
        if k > 1:
            prev = f"{a.slug}-s{k - 1}"
            meta = prev_meta(a.account, key, prev, a.posw, a.poll)
            if meta is None:
                sys.exit(f"{prev}: lỗi hoặc thiếu meta -> dừng chuỗi, cần xem tay")
            print(f"{prev}: epoch {meta['epochs_run']} · tốt nhất epoch {meta['best_epoch']} val triple_recall@100 "
                  f"{meta['best_val_triple_recall100']*100:.2f}% · {'XONG' if meta['done'] else 'chưa xong'}",
                  flush=True)
            if meta["done"]:
                return
            args.update(RESUME="1", RESUME_EPOCH=str(meta["epochs_run"]))
            job["kernels"] = [prev]
        q = quota_left(a.account, key)
        if not q > 0.1:
            sys.exit(f"{a.account} còn {q:.2f}h quota -> không đẩy {slug}")
        jf = ROOT / "ritek/kaggle/jobs" / f"{slug.replace('ritek-', '').replace('-', '_')}.json"
        jf.write_text(json.dumps([job], indent=1), encoding="utf-8")
        print(f"{a.account} quota {q:.2f}h · đẩy {slug} ({jf.name})", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/kaggle_run.py"), "--jobs", str(jf), "--outdir",
                        str(OUTDIR.relative_to(ROOT)), "--workdir", ".cache/kaggle_jobs", "--poll", str(a.poll)],
                       cwd=ROOT, check=True)
    sys.exit(f"hết {a.max_sessions} phiên mà chưa xong")


if __name__ == "__main__":
    main()
