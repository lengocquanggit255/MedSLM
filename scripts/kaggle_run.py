"""Điều phối job GPU trên nhiều account Kaggle.

Ràng buộc đã xác minh (xem docs/PLAN.md §8):
  - Mỗi account chỉ chạy được **1 notebook GPU cùng lúc** => 1 slot/account.
  - Quota 30h GPU/tuần/account.
  - `machine_shape: "NvidiaTeslaT4"` trong kernel-metadata.json mới được T4x2;
    thiếu nó sẽ nhận P100 (sm_60) mà torch cài sẵn KHÔNG hỗ trợ.
  - Kaggle CLI đọc file code bằng codec mặc định của hệ -> phải đặt PYTHONUTF8=1.
  - Job có `"gpu": false` chạy phiên CPU (không tính quota GPU); vẫn giữ 1 job/account như job GPU.
  - Account + key nằm trong env/.env (xem `load_accounts`), không hardcode trong code.

Dùng:
    python scripts/kaggle_run.py --jobs jobs.json          # chạy và chờ
    python scripts/kaggle_run.py --status                  # xem slot đang rảnh
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def load_accounts(path: Path = ROOT / "env" / ".env"):
    """Account Kaggle từ env/.env: mỗi cặp KAGGLE_USERNAME_<X> / KAGGLE_KEY_<X> là một slot GPU,
    theo thứ tự trong file. Bỏ một account khỏi pool = comment cặp đó trong .env."""
    from dotenv import dotenv_values
    v = dotenv_values(path)
    accs = []
    for k, user in v.items():
        if not k.startswith("KAGGLE_USERNAME_") or not user:
            continue
        key = v.get("KAGGLE_KEY_" + k[len("KAGGLE_USERNAME_"):])
        if not key:
            raise RuntimeError(f"{path}: có {k}={user} nhưng thiếu KAGGLE_KEY tương ứng")
        accs.append((user, key))
    if not accs:
        raise RuntimeError(f"{path}: không có account Kaggle nào (KAGGLE_USERNAME_<X> / KAGGLE_KEY_<X>), "
                           "xem env/.env.example")
    return accs


ACCOUNTS = load_accounts()
BUSY = ("RUNNING", "QUEUED")
# Số lần LIÊN TIẾP đọc trạng thái thất bại trước khi bỏ theo dõi một job.
# Với poll 60s thì 5 lần ≈ 5 phút chịu đựng trục trặc API — rẻ hơn nhiều so với bỏ rơi
# một kernel đang chạy (đã xảy ra 2 lần: job vẫn RUNNING mà dispatcher đã tuyên bố hỏng).
MAX_UNKNOWN = 5


def env(user, key):
    e = dict(os.environ)
    e.update({"KAGGLE_USERNAME": user, "KAGGLE_KEY": key, "PYTHONUTF8": "1"})
    return e


def kag(user, key, *args, timeout=600):
    """Gọi kaggle CLI. KHÔNG để TimeoutExpired thoát ra ngoài — một kernel cũ treo
    (vd notebook nháp chưa từng chạy) sẽ làm sập cả dispatcher."""
    try:
        r = subprocess.run(["kaggle", *args], env=env(user, key), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except Exception as e:  # lỗi mạng nhất thời
        return 1, f"EXC {type(e).__name__}: {e}"


def status_of(user, key, slug, tries=3):
    """Trạng thái kernel. Thử lại vài lần trước khi chịu thua.

    `UNKNOWN` KHÔNG phải trạng thái kết thúc — nó chỉ có nghĩa là lần gọi API này không đọc
    được (rate limit, mạng chập, CLI quá hạn). Đã gặp: dispatcher nhận UNKNOWN rồi tuyên bố
    job thất bại và thoát, trong khi kernel vẫn đang RUNNING trên Kaggle. Xem thêm vòng lặp
    thu hoạch ở main(): phải UNKNOWN nhiều lần LIÊN TIẾP mới coi là hỏng.
    """
    for a in range(tries):
        _, out = kag(user, key, "kernels", "status", f"{user}/{slug}", timeout=240)
        for tok in ("RUNNING", "QUEUED", "COMPLETE", "ERROR",
                    "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            if tok in out:
                return tok
        if a < tries - 1:
            time.sleep(15 * (a + 1))
    return "UNKNOWN"


def bad_files(dest):
    """File hỏng trong `dest`. API trả 502 khi tải output mà kaggle CLI vẫn ghi phần thân lỗi thành file
    (~332 byte), để lại file 0 byte, hoặc đứt giữa chừng (IncompleteRead) — đã gặp 2026-10-19. Kiểm:
    trang lỗi HTML / rỗng; .pt .ckpt .npz (đều là zip) phải qua CRC; .json phải parse được."""
    bad = []
    for f in Path(dest).iterdir():
        if not f.is_file():
            continue
        if f.stat().st_size < 4096:
            head = f.read_bytes()[:64].lstrip().lower()
            if not head or head.startswith((b"<html", b"<!doctype html")):
                bad.append(f.name)
                continue
        try:
            if f.suffix in (".pt", ".ckpt", ".npz"):
                with zipfile.ZipFile(f) as z:
                    assert z.testzip() is None
            elif f.suffix == ".json":
                json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bad.append(f.name)
    return bad


def download_output(user, key, slug, dest, tries=5, timeout=1800):
    """Tải output kernel về `dest`; file nào là trang lỗi thì xoá rồi tải lại. True nếu tải sạch."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for a in range(tries):
        rc, out = kag(user, key, "kernels", "output", f"{user}/{slug}", "-p", str(dest), timeout=timeout)
        bad = bad_files(dest)
        if rc == 0 and not bad:
            return True
        print(f"  [{user}] {slug}: CLI rc={rc}, {len(bad)} file hỏng ({', '.join(bad[:3])}) · "
              f"{out.strip().splitlines()[-1][:120] if out.strip() else ''} -> tải lại ({a + 1}/{tries})", flush=True)
        for n in bad:
            (dest / n).unlink()
        time.sleep(60 * (a + 1))
    return False


def account_busy(user, key, prefix=("stark-", "ritek-")):
    """True nếu account đang có kernel GPU chạy (Kaggle chỉ cho 1 cùng lúc).

    CHỈ kiểm các kernel do benchmark này tạo (`prefix`). Trước đây quét toàn bộ kernel
    của account: vừa chậm, vừa từng làm sập dispatcher khi một notebook nháp cũ
    (`vm14k-split`) trả status treo 240s.
    """
    _, out = kag(user, key, "kernels", "list", "--mine", "--page-size", "50", timeout=300)
    slugs = []
    for line in out.splitlines()[2:]:
        p = line.split()
        if p and "/" in p[0]:
            sl = p[0].split("/", 1)[1]
            if sl.startswith(prefix):
                slugs.append(sl)
    for s in slugs:
        if status_of(user, key, s) in BUSY:
            return True, s
    return False, None


def quota_left(user, key):
    code = ("import kaggle,sys;q=kaggle.api.quota_view().gpu_quota;"
            "print((q.total_time_allowed-q.time_used).total_seconds()/3600)")
    try:   # lỗi mạng/quá hạn nhất thời -> nan (người gọi coi như "chưa đọc được", thử lại), không làm sập chuỗi
        r = subprocess.run([sys.executable, "-c", code], env=env(user, key),
                           capture_output=True, text=True, timeout=300)
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return float("nan")


def push(user, key, job, workdir: Path):
    """Dựng thư mục kernel rồi push. `job` gồm: slug, code, args(dict), datasets, kernels (tuỳ chọn)."""
    d = workdir / job["slug"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = ROOT / job["code"]
    body = src.read_text(encoding="utf-8")
    # Chèn tham số qua biến môi trường ở đầu script (kernel không nhận argv).
    inject = "import os\n" + "".join(
        f'os.environ["{k}"] = "{v}"\n' for k, v in job.get("args", {}).items())
    (d / src.name).write_text(inject + body, encoding="utf-8")
    meta = {
        "id": f"{user}/{job['slug']}", "title": job["slug"],
        "code_file": src.name, "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",          # BẮT BUỘC để được T4x2
        "dataset_sources": [f"{user}/{s}" for s in job.get("datasets", [])],
        # output của kernel khác (cùng account) gắn làm input, vd phiên trước của một job chạy nhiều phiên
        "competition_sources": [], "kernel_sources": [f"{user}/{s}" for s in job.get("kernels", [])],
        "model_sources": [],
    }
    if job.get("gpu", True) is False:              # job chỉ dùng CPU: không tính vào quota GPU tuần
        meta["enable_gpu"] = "false"
        del meta["machine_shape"]
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    code, out = kag(user, key, "kernels", "push", "-p", str(d),
                    # 43200s = 12h, giới hạn tối đa của phiên Kaggle. Đặt thấp hơn chỉ tổ bị cắt giữa chừng
                    # mà vẫn mất trọn số giờ đã chạy (đã mất 8.7h vì đặt 8h).
                    "-t", str(job.get("timeout", 43200)), timeout=900)
    ok = "successfully pushed" in out
    return ok, out.strip().splitlines()[-1] if out.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", help="file JSON chứa danh sách job")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workdir", default=".cache/kaggle_jobs")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--outdir", default="results/kaggle")
    args = ap.parse_args()

    if args.status or not args.jobs:
        print(f"{'account':<18}{'GPU còn (h)':>13}  {'trạng thái':<12}")
        for u, k in ACCOUNTS:
            busy, slug = account_busy(u, k)
            print(f"{u:<18}{quota_left(u,k):>13.2f}  "
                  f"{('ĐANG CHẠY: '+slug) if busy else 'rảnh':<12}")
        return

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    active = {}             # (user,key) -> job
    unknown_streak = {}     # slug -> số lần LIÊN TIẾP không đọc được trạng thái
    done, failed = [], []

    print(f"{len(pending)} job, {len(ACCOUNTS)} slot GPU (1 job/account cùng lúc)\n")
    while pending or active:
        # phát job vào slot rảnh
        for u, k in ACCOUNTS:
            if not pending:
                break
            if any(a[0] == u for a in active):
                continue
            # Job có thể ghim account: dataset riêng chỉ tồn tại trên một số account.
            # (Đã gặp: mor-package chỉ có trên quangminh2401 -> job chạy ở minhlon192 fail.)
            allowed = pending[0].get("accounts")
            if allowed and u not in allowed:
                continue
            busy, slug = account_busy(u, k)
            if busy:
                print(f"  [{u}] đang chạy {slug}, bỏ qua")
                continue
            job = pending.pop(0)
            ok, msg = push(u, k, job, workdir)
            if ok:
                active[(u, k)] = job
                print(f"  [{u}] ▶ {job['slug']}")
            else:
                print(f"  [{u}] ✗ push lỗi {job['slug']}: {msg[:160]}")
                failed.append((job, msg))
        if not active:
            if pending:
                time.sleep(args.poll)
                continue
            break
        time.sleep(args.poll)
        # thu hoạch
        for (u, k), job in list(active.items()):
            st = status_of(u, k, job["slug"])
            if st in BUSY:
                unknown_streak.pop(job["slug"], None)
                continue
            if st == "UNKNOWN":
                # Không bỏ job chỉ vì một lần đọc hỏng. Kernel 12h mà bị bỏ rơi thì vẫn
                # chạy trên Kaggle và vẫn tiêu quota, nhưng ta mất kết quả và mất slot.
                n = unknown_streak.get(job["slug"], 0) + 1
                unknown_streak[job["slug"]] = n
                print(f"  [{u}] ? {job['slug']}: không đọc được trạng thái ({n}/{MAX_UNKNOWN})")
                if n < MAX_UNKNOWN:
                    continue
                print(f"  [{u}] ✗ {job['slug']}: {MAX_UNKNOWN} lần liên tiếp không đọc được "
                      f"-> bỏ theo dõi (kernel có thể VẪN đang chạy trên Kaggle)")
            del active[(u, k)]
            unknown_streak.pop(job["slug"], None)
            if st == "COMPLETE":
                dest = outdir / job["slug"]
                if not download_output(u, k, job["slug"], dest):
                    print(f"  [{u}] ✗ {job['slug']}: COMPLETE nhưng tải output vẫn ra trang lỗi")
                    failed.append((job, "tải output lỗi"))
                    continue
                print(f"  [{u}] ✔ {job['slug']} -> {dest}")
                done.append(job)
            else:
                print(f"  [{u}] ✗ {job['slug']} kết thúc với {st}")
                download_output(u, k, job["slug"], outdir / job["slug"], tries=2, timeout=900)
                failed.append((job, st))

    print(f"\nXong: {len(done)} thành công, {len(failed)} thất bại")
    for job, why in failed:
        print(f"  ✗ {job['slug']}: {str(why)[:200]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
