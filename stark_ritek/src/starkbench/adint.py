"""SKB của RiTeK-ADint — cùng giao diện `PrimeSKB` để dùng lại nguyên harness.

Nguồn: `data/ritek/adint_skb/ADint/` (tải từ HuggingFace `Graph_node_edge_info/`, bản trong
`data/ritek/Dataset/Adint/` KHÔNG có text node — xem docs/EDA_ADINT.md §0).

Bốn khác biệt so với PrimeSKB, mỗi cái đều buộc phải xử lý:

1. **`node_type_dict` NGƯỢC CHIỀU.** Prime lưu `{id: tên}`, ADint lưu `{tên: id}`.
   Không đảo thì `node_type_of()` trả rác.

2. **21,9% dòng cạnh trùng lặp** (303.859/1.385.264). Prime không có. Phải khử trước khi
   dựng CSR, nếu không mọi phép đếm bậc và mọi document `add_rel` đều phình sai.

3. **Đồ thị KHÔNG đối xứng hoàn toàn** — chỉ 56,4% cặp (h,t) có cạnh ngược. Prime lưu sẵn
   cả hai chiều nên `_build_adj` của nó không cần đối xứng hoá. Ở đây phải chọn, và lựa
   chọn ảnh hưởng trực tiếp tới trần của traversal ⇒ `symmetrize` là tham số TƯỜNG MINH,
   mặc định True, và con số của cả hai lựa chọn được ghi trong EDA.

4. **`node_info` phẳng**: `{name, description, source, type}`, không có `details` lồng nhau.
   `name` là danh sách đồng nghĩa ngăn bằng `|` ở 10,2% node (`TUBB3|AP4B1|BHLHE23`).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from .skb import compact_text


class AdintSKB:
    """SKB của RiTeK-ADint. Giao diện khớp `PrimeSKB` (xem src/starkbench/skb.py)."""

    def __init__(self, root: str | Path = "data/ritek/adint_skb/ADint",
                 symmetrize: bool = True):
        p = Path(root)
        self.symmetrize = symmetrize
        self.node_info: Dict[int, dict] = pickle.load(open(p / "node_info.pkl", "rb"))
        self.edge_type_dict: Dict[int, str] = pickle.load(open(p / "edge_type_dict.pkl", "rb"))

        raw_nt = pickle.load(open(p / "node_type_dict.pkl", "rb"))
        # ⚠️ ADint lưu {tên: id}; PrimeSKB mong {id: tên}. Đảo, và kiểm chứ không giả định.
        if raw_nt and isinstance(next(iter(raw_nt.keys())), str):
            self.node_type_dict = {v: k for k, v in raw_nt.items()}
        else:
            self.node_type_dict = raw_nt
        assert all(isinstance(k, (int, np.integer)) for k in self.node_type_dict), \
            "node_type_dict phải là {id: tên} sau khi đảo"

        self.node_types: torch.Tensor = torch.load(p / "node_types.pt", weights_only=False)
        ei: torch.Tensor = torch.load(p / "edge_index.pt", weights_only=False)
        et: torch.Tensor = torch.load(p / "edge_types.pt", weights_only=False)
        self.num_nodes = int(self.node_types.numel())

        src, dst, rel = ei[0].numpy(), ei[1].numpy(), et.numpy()
        if symmetrize:
            src, dst, rel = (np.concatenate([src, dst]), np.concatenate([dst, src]),
                             np.concatenate([rel, rel]))
        # khử trùng lặp (h, rel, t) — 21,9% dòng gốc là bản sao
        key = (src.astype(np.int64) * self.num_nodes + dst.astype(np.int64)) \
            * (int(rel.max()) + 1) + rel.astype(np.int64)
        _, keep = np.unique(key, return_index=True)
        keep.sort()
        self.n_raw_edges = len(src)
        src, dst, rel = src[keep], dst[keep], rel[keep]
        self.edge_index = torch.from_numpy(np.stack([src, dst]))
        self.edge_types = torch.from_numpy(rel)

        # ADint không giới hạn candidate (Prime có `candidate_types`) -> mọi node là ứng viên
        self.candidate_ids: List[int] = list(range(self.num_nodes))
        self._names = np.array(
            [str(self.node_info[i].get("name", "") or "") for i in range(self.num_nodes)],
            dtype=object)
        self._build_adj()

    def _build_adj(self) -> None:
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        et = self.edge_types.numpy()
        n = self.num_nodes
        self._adj_by_type: Dict[str, sp.csr_matrix] = {}
        for tid, tname in self.edge_type_dict.items():
            m = et == tid
            a = sp.csr_matrix(
                (np.ones(int(m.sum()), dtype=np.int8), (src[m], dst[m])), shape=(n, n))
            a.sum_duplicates()
            self._adj_by_type[tname] = a

    # ------------------------------------------------------------------ tra cứu
    def node_type_of(self, idx: int) -> str:
        return self.node_type_dict[int(self.node_types[idx])]

    def name_of(self, idx: int) -> str:
        return self._names[idx]

    def aliases_of(self, idx: int) -> List[str]:
        """Tên ADint là danh sách đồng nghĩa ngăn bằng `|` (10,2% node)."""
        return [a.strip() for a in str(self._names[idx]).split("|") if a.strip()]

    def rel_type_lst(self) -> List[str]:
        return list(self.edge_type_dict.values())

    def get_neighbor_nodes(self, idx: int, edge_type: str) -> np.ndarray:
        a = self._adj_by_type[edge_type]
        return a.indices[a.indptr[idx]:a.indptr[idx + 1]]

    # ------------------------------------------------------------- dựng document
    def get_rel_info(self, idx: int, rel_types: Sequence[str] | None = None,
                     n_rel: int = -1, rng: np.random.Generator | None = None) -> str:
        """Cùng ĐỊNH DẠNG với `PrimeSKB.get_rel_info` để document hai bộ so được với nhau.

        Khác một điểm CÓ CHỦ Ý: ở đây dùng `sorted(set(...))`. Bản Prime phải bắt chước đúng
        thứ tự lặp của `set` CPython vì cần trùng khít byte với stark_qa gốc. ADint không có
        bản gốc nào để trùng khít, nên chọn thứ tự XÁC ĐỊNH cho dễ tái lập.
        """
        doc = ""
        rel_types = self.rel_type_lst() if rel_types is None else rel_types
        for edge_t in rel_types:
            node_ids = self.get_neighbor_nodes(idx, edge_t)
            if len(node_ids) == 0:
                continue
            doc += f"\n  {edge_t.replace(' ', '_')}: " + "{"
            ntypes = self.node_types[node_ids].numpy()
            for node_type in sorted(set(ntypes.tolist())):
                doc += f"{self.node_type_dict[node_type]}: "
                ids_t = node_ids[ntypes == node_type]
                if n_rel > 0 and len(ids_t) > n_rel:
                    r = rng if rng is not None else np.random.default_rng(42)
                    ids_t = ids_t[r.permutation(len(ids_t))[:n_rel]]
                doc += "(" + ", ".join(self._names[i] for i in ids_t) + "),"
            doc += "}"
        return "- relations:" + doc if doc else doc

    def get_doc_info(self, idx: int, add_rel: bool = False, compact: bool = False,
                     n_rel: int = -1, rng: np.random.Generator | None = None,
                     drop_desc: bool = False, trunc: int | None = None,
                     keep_desc: np.ndarray | None = None) -> str:
        """`node_info` của ADint phẳng: name / type / source / description.

        Hai cần gạt của Tầng 3, ĐỘC LẬP nhau — đo hai thứ khác nhau:

        `trunc=K`      giữ **K từ đầu** của MỌI description (K=0 ⇒ bỏ hẳn).
                       Cần gạt về ĐỘ DÀI. Đồng đều trên mọi node, không cần chiến lược
                       lấy mẫu. Đo được: cắt còn 20 từ giữ 55,0% số từ nhưng vẫn giữ
                       20,36/25,75 = 79% chồng lấp từ vựng với câu hỏi.

        `keep_desc`    mảng bool (n,) — node nào được GIỮ description.
                       Cần gạt về ĐỘ PHỦ. Cần chiến lược lấy mẫu (ngẫu nhiên / thiên vị
                       node đáp án / ngược lại) vì node đáp án có description 84,44% so
                       với mặt bằng 34,59%.

        `drop_desc=True` là lối tắt của `trunc=0`.
        """
        info = self.node_info[idx]
        doc = f"- name: {info.get('name')}\n"
        doc += f"- type: {info.get('type')}\n"
        src = str(info.get("source", "") or "").strip()
        if src:
            doc += f"- source: {src}\n"
        drop = drop_desc or (trunc is not None and trunc <= 0) \
            or (keep_desc is not None and not bool(keep_desc[idx]))
        if not drop:
            d = str(info.get("description", "") or "").strip()
            if d and trunc is not None:
                d = " ".join(d.split()[:trunc])
            if d:
                doc += f"- details:\n  - description: {d}\n"
        if add_rel:
            doc += self.get_rel_info(idx, n_rel=n_rel, rng=rng)
        if compact:
            doc = compact_text(doc)
        return doc

    def get_name_only_doc(self, idx: int) -> str:
        info = self.node_info[idx]
        return f"- name: {info.get('name')}\n- type: {info.get('type')}\n"

    def degree(self) -> np.ndarray:
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        return (np.bincount(src, minlength=self.num_nodes)
                + np.bincount(dst, minlength=self.num_nodes))

    def has_description(self) -> np.ndarray:
        """Mảng bool — dùng cho đối chứng âm A5 và cho Tầng 3."""
        return np.array([bool(str(self.node_info[i].get("description", "") or "").strip())
                         for i in range(self.num_nodes)])
