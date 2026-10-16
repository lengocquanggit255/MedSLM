"""Đối chiếu src/starkbench/skb.py với PrimeSKB THẬT của stark_qa v1.1.0.

Nạp `stark_qa/skb/prime.py` bằng cách stub các module nặng không cần thiết
(`tdc`, `gdown`) — chúng chỉ dùng cho đường tải/ xử lý raw, mà ta đã có `processed/`.

Chạy:
    python scripts/verify_docs.py <path>/stark_qa-1.1.0

Phải PASS trước khi tin bất kỳ document / kết quả BM25 / embedding nào.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _AnyAttr(types.ModuleType):
    """Module giả: mọi thuộc tính trả về một object rỗng thay vì AttributeError."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


def _stub(name):
    parts = name.split(".")
    for i in range(len(parts)):
        full = ".".join(parts[: i + 1])
        if full not in sys.modules:
            m = _AnyAttr(full)
            sys.modules[full] = m
            if i:
                setattr(sys.modules[".".join(parts[:i])], parts[i], m)


def import_ref_skb(pkg_root):
    """Import PrimeSKB gốc, tự stub mọi module thiếu.

    Các module bị stub (ogb, tdc, gdown, ...) chỉ được dùng ở nhánh TẢI/XỬ LÝ raw.
    Ta đã có `processed/` nên nhánh đó không chạy — stub là an toàn.
    In ra danh sách đã stub để minh bạch.
    """
    sys.path.insert(0, str(pkg_root))
    stubbed = []
    for _ in range(25):
        try:
            from stark_qa.skb.prime import PrimeSKB
            if stubbed:
                print(f"  đã stub (chỉ dùng cho nhánh raw, không chạy): {', '.join(stubbed)}")
            return PrimeSKB
        except ModuleNotFoundError as e:
            if not e.name:
                raise
            stubbed.append(e.name)
            _stub(e.name)
    raise RuntimeError("stub quá nhiều lần, dừng")


def main():
    if len(sys.argv) < 2:
        print("Cần đường dẫn tới thư mục stark_qa-1.1.0 đã giải nén.")
        sys.exit(2)
    pkg_root = Path(sys.argv[1])
    if not (pkg_root / "stark_qa" / "skb" / "prime.py").exists():
        print(f"Không thấy stark_qa/skb/prime.py trong {pkg_root}")
        sys.exit(2)

    RefSKB = import_ref_skb(pkg_root)

    from starkbench.skb import PrimeSKB as OurSKB  # noqa: E402

    root = "data/stark/prime"
    print("Nạp PrimeSKB gốc (chậm — nó dựng torch.sparse cho 18 relation type)...")
    ref = RefSKB(root=root, download_processed=True)
    print("Nạp bản của ta...")
    ours = OurSKB(root=root)

    # --- kiểm tra bất biến cơ bản -------------------------------------------
    assert ours.num_nodes == len(ref.node_info), "lệch số node"
    assert ours.candidate_ids == ref.candidate_ids, "lệch candidate_ids"
    print(f"  num_nodes={ours.num_nodes}  candidate_ids khớp ({len(ref.candidate_ids)})")

    rng = np.random.default_rng(0)
    # mẫu: node ngẫu nhiên + node bậc cao nhất (hub — dễ lộ lỗi nhất)
    deg = ours.degree()
    sample = list(rng.choice(ours.num_nodes, size=250, replace=False))
    sample += list(np.argsort(-deg)[:15])          # 15 hub lớn nhất
    sample += list(np.argsort(deg)[:10])           # 10 node bậc thấp nhất
    sample = sorted(set(int(i) for i in sample))

    fails = {"no_rel": 0, "add_rel": 0, "compact": 0, "neighbors": 0}
    for i in sample:
        # 1) hàng xóm theo từng relation type
        for et in ours.rel_type_lst():
            a = list(ours.get_neighbor_nodes(i, et))
            b = list(ref.get_neighbor_nodes(i, et))
            if a != b:
                fails["neighbors"] += 1
                if fails["neighbors"] <= 3:
                    print(f"  MISMATCH neighbors node={i} rel={et}: ours={len(a)} ref={len(b)}")
                break
        # 2) document không quan hệ
        if ours.get_doc_info(i, add_rel=False) != ref.get_doc_info(i, add_rel=False):
            fails["no_rel"] += 1
            if fails["no_rel"] <= 2:
                o = ours.get_doc_info(i, add_rel=False); r = ref.get_doc_info(i, add_rel=False)
                print(f"  MISMATCH no_rel node={i}\n    ours[:200]={o[:200]!r}\n    ref [:200]={r[:200]!r}")
        # 3) document có quan hệ, n_rel=-1 (cấu hình benchmark)
        if ours.get_doc_info(i, add_rel=True, n_rel=-1) != ref.get_doc_info(i, add_rel=True, n_rel=-1):
            fails["add_rel"] += 1
            if fails["add_rel"] <= 2:
                o = ours.get_doc_info(i, add_rel=True, n_rel=-1)
                r = ref.get_doc_info(i, add_rel=True, n_rel=-1)
                k = next((j for j in range(min(len(o), len(r))) if o[j] != r[j]), min(len(o), len(r)))
                print(f"  MISMATCH add_rel node={i} lệch tại ký tự {k}")
                print(f"    ours={o[max(0,k-80):k+80]!r}")
                print(f"    ref ={r[max(0,k-80):k+80]!r}")
        # 4) compact
        if ours.get_doc_info(i, add_rel=True, n_rel=-1, compact=True) != \
           ref.get_doc_info(i, add_rel=True, n_rel=-1, compact=True):
            fails["compact"] += 1

    n = len(sample)
    print(f"\nĐã đối chiếu {n} node (gồm 15 hub lớn nhất, 10 node bậc thấp nhất):")
    for k, v in fails.items():
        print(f"  {k:<10}: {'OK' if v == 0 else f'{v} LỆCH'}")
    if any(fails.values()):
        print("\nFAIL")
        sys.exit(1)
    print("\nPASS: document dựng ra khớp chính xác PrimeSKB của stark_qa v1.1.0")


if __name__ == "__main__":
    main()
