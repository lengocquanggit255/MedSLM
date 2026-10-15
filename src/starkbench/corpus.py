"""Dựng và cache corpus document.

Lưu dạng 1 file .bin (utf-8 nối liền) + 1 file offsets .npy, thay vì jsonl/pickle:
corpus `add_rel` nặng ~333MB, cách này nạp bằng memory-map nên gần như tức thì và
không tốn RAM cho phần không đọc tới.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import numpy as np

VARIANTS = ("no_rel", "add_rel", "name_only")


def variant_builder(skb, variant: str, n_rel: int = -1, seed: int = 42) -> Callable[[int], str]:
    """Trả về hàm dựng document cho một biến thể."""
    if variant == "no_rel":
        return lambda i: skb.get_doc_info(i, add_rel=False)
    if variant == "add_rel":
        rng = np.random.default_rng(seed)
        return lambda i: skb.get_doc_info(i, add_rel=True, n_rel=n_rel, rng=rng)
    if variant == "name_only":
        return skb.get_name_only_doc
    raise ValueError(f"biến thể không hợp lệ: {variant} (hợp lệ: {VARIANTS})")


def _is_enriched(skb) -> bool:
    return any(k.startswith("ext_") for v in skb.node_info.values() for k in (v.get("details") or {}))


def _tag(variant: str, n_rel: int, tag: str | None) -> str:
    base = variant if n_rel < 0 else f"{variant}_nrel{n_rel}"
    return base if tag is None else (tag if n_rel < 0 else f"{tag}_nrel{n_rel}")


def build(skb, variant: str, cache_dir: str | Path = ".cache/docs",
          n_rel: int = -1, seed: int = 42, verbose: bool = True, tag: str | None = None) -> Path:
    """Dựng corpus và ghi cache. Trả về đường dẫn file .bin.

    `tag` đặt tên cache. Với node_info đã làm giàu (starkbench.enrich.apply) PHẢI truyền
    tag=enrich.variant_name(...): cache chỉ đặt tên theo `variant` thì sẽ trùng file của corpus
    gốc (vd .cache/docs/no_rel.bin đã tồn tại) và hàm sẽ âm thầm trả về corpus GỐC.
    """
    if tag is None and _is_enriched(skb):
        raise ValueError("node_info đã làm giàu (có ext_*): phải truyền tag=variant_name(...), "
                         "nếu không sẽ dùng nhầm cache của corpus gốc")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = _tag(variant, n_rel, tag)
    bin_path = cache / f"{tag}.bin"
    off_path = cache / f"{tag}.offsets.npy"
    if bin_path.exists() and off_path.exists():
        if verbose:
            print(f"  [cache] {tag}: đã có, bỏ qua")
        return bin_path

    fn = variant_builder(skb, variant, n_rel=n_rel, seed=seed)
    n = skb.num_nodes
    offs = np.zeros(n + 1, dtype=np.int64)
    with open(bin_path, "wb") as f:
        pos = 0
        for i in range(n):
            b = fn(i).encode("utf-8")
            f.write(b)
            pos += len(b)
            offs[i + 1] = pos
            if verbose and (i + 1) % 20000 == 0:
                print(f"  {tag}: {i+1:,}/{n:,} node, {pos/1e6:.0f} MB")
    np.save(off_path, offs)
    if verbose:
        print(f"  [xong] {tag}: {pos/1e6:.1f} MB, trung bình {pos/n:.0f} byte/doc")
    return bin_path


def load(variant: str, cache_dir: str | Path = ".cache/docs", n_rel: int = -1,
         tag: str | None = None) -> List[str]:
    """Nạp toàn bộ corpus thành list[str]. Corpus làm giàu: truyền đúng `tag` đã dùng khi build."""
    cache = Path(cache_dir)
    tag = _tag(variant, n_rel, tag)
    offs = np.load(cache / f"{tag}.offsets.npy")
    raw = (cache / f"{tag}.bin").read_bytes()
    return [raw[offs[i]:offs[i + 1]].decode("utf-8") for i in range(len(offs) - 1)]
