"""Chặn RAM cho các job chạy trên server dùng chung (server196).

Vì sao cần module riêng thay vì một dòng `ulimit`: đã thử 3 cơ chế, chỉ 1 cái dùng được
cho MỌI loại job.

  1. `systemd-run --user --scope -p MemoryMax=80G`
     -> CHẤP NHẬN LỆNH NHƯNG KHÔNG CÓ HIỆU LỰC. server196 chạy cgroup **v1**
        (/sys/fs/cgroup/cgroup.controllers không tồn tại) và memory controller không được
        delegate cho user scope. Kiểm chứng: cấp phát 600MB dưới cap 200M -> thành công.
  2. `ulimit -v` (RLIMIT_AS)
     -> CÓ hiệu lực (MemoryError đúng ngưỡng) NHƯNG phá job đa luồng: torch + OpenMP
        reserve rất nhiều địa chỉ ảo, chạy bench Contriever dưới cap 80GiB thì chết ngay
        `libgomp: Thread creation failed`. Chỉ dùng được cho job numpy/scipy đơn luồng.
     -> RLIMIT_AS đếm bộ nhớ ẢO, không phải bộ nhớ THỰC. Đó là lý do nó vừa quá chặt
        (với torch) vừa không phản ánh đúng thứ ta muốn giới hạn.
  3. Watchdog đọc VmRSS  <- CÁI NÀY DÙNG
     -> Đo đúng bộ nhớ thực, cộng cả tiến trình con, không cần quyền root, không phá
        OpenMP. Nhược điểm: không chặn tức thời như cgroup mà phát hiện sau <=2 giây;
        chấp nhận được vì ta canh ở 80GB trong khi server còn ~187GB trống.

Dùng:
    from starkbench.memcap import install
    install(hard_gb=80.0)          # vượt -> dừng cả cây tiến trình
    install(hard_gb=80.0, soft_gb=64.0, on_soft=my_flush)   # 64GB -> gọi callback nhả bộ nhớ

Không có tác dụng ngoài Linux (Windows chỉ in cảnh báo rồi bỏ qua).
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

__all__ = ["rss_gb", "tree_rss_gb", "install", "set_address_space_limit"]

_PAGE = 1024.0 ** 3


def _read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm", "rb") as f:
            # statm: size resident shared ... (đơn vị: page)
            return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except (OSError, IndexError, ValueError):
        return 0


def _children(pid: int) -> list[int]:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/children", "rb") as f:
                    out += [int(x) for x in f.read().split()]
            except OSError:
                pass
    except OSError:
        pass
    return out


def rss_gb() -> float:
    """RAM thực của riêng tiến trình hiện tại."""
    return _read_rss_kb(os.getpid()) / 1024.0 ** 2


def tree_rss_gb(root: int | None = None) -> float:
    """RAM thực của cả cây tiến trình (tiến trình này + mọi con cháu).

    Cần cộng cây vì DataLoader/multiprocessing của torch và `joblib` đẻ tiến trình con;
    nhìn mỗi tiến trình cha sẽ báo thiếu rất nhiều.
    """
    root = os.getpid() if root is None else root
    total, seen, stack = 0, set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        total += _read_rss_kb(p)
        stack += _children(p)
    return total / 1024.0 ** 2


def set_address_space_limit(gb: float) -> bool:
    """RLIMIT_AS — CHỈ dùng cho job numpy/scipy đơn luồng.

    ĐỪNG gọi cho job có torch/OpenMP: xem ghi chú (2) ở đầu file.
    Trả về True nếu đặt được.
    """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * _PAGE), int(gb * _PAGE)))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[memcap] không đặt được RLIMIT_AS: {e}", file=sys.stderr)
        return False


def install(hard_gb: float = 80.0, soft_gb: float | None = None, poll: float = 2.0,
            on_soft=None, verbose_every: float = 300.0) -> threading.Thread | None:
    """Bật watchdog nền. Vượt `hard_gb` -> giết cả cây tiến trình rồi thoát mã 137.

    soft_gb : ngưỡng cảnh báo sớm; gọi `on_soft()` một lần (ví dụ để xả cache).
    poll    : chu kỳ kiểm tra (giây).
    """
    if not sys.platform.startswith("linux"):
        print(f"[memcap] bỏ qua: chỉ hỗ trợ Linux (nền tảng={sys.platform})", file=sys.stderr)
        return None

    root = os.getpid()
    state = {"soft_fired": False, "last_log": 0.0, "peak": 0.0}

    def loop():
        while True:
            used = tree_rss_gb(root)
            state["peak"] = max(state["peak"], used)
            now = time.time()
            if verbose_every and now - state["last_log"] >= verbose_every:
                state["last_log"] = now
                print(f"[memcap] RAM cây tiến trình {used:.1f} / {hard_gb:.0f} GB "
                      f"(đỉnh {state['peak']:.1f})", flush=True)
            if soft_gb and used >= soft_gb and not state["soft_fired"]:
                state["soft_fired"] = True
                print(f"[memcap] CẢNH BÁO vượt ngưỡng mềm {soft_gb:.0f} GB "
                      f"(đang dùng {used:.1f} GB)", flush=True)
                if on_soft:
                    try:
                        on_soft()
                    except Exception as e:  # noqa: BLE001
                        print(f"[memcap] on_soft lỗi: {e}", flush=True)
            if used >= hard_gb:
                print(f"[memcap] DỪNG KHẨN: {used:.1f} GB >= trần {hard_gb:.0f} GB. "
                      f"Giết cây tiến trình.", file=sys.stderr, flush=True)
                for p in set(_children(root)):
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
                os._exit(137)
            time.sleep(poll)

    t = threading.Thread(target=loop, name="memcap", daemon=True)
    t.start()
    print(f"[memcap] watchdog bật: trần cứng {hard_gb:.0f} GB"
          + (f", ngưỡng mềm {soft_gb:.0f} GB" if soft_gb else "")
          + f", chu kỳ {poll}s", flush=True)
    return t
