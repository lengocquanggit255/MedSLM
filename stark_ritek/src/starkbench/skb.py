"""Nạp STaRK-Prime SKB từ file cục bộ và dựng document.

Vì sao không dùng thẳng `stark_qa.skb.PrimeSKB`:
  - `stark_qa/skb/prime.py` import `from tdc.resource import PrimeKG` ở cấp module
    => bắt buộc cài PyTDC (rất nặng) dù ta chỉ đọc dữ liệu đã xử lý sẵn.
  - PrimeSKB dựng `sparse_adj_by_type` bằng `torch.sparse` cho cả 18 relation type trên
    8.1M cạnh => tốn RAM và chậm. Bản này dùng CSR của scipy, tra hàng xóm nhanh hơn nhiều.

ĐỔI LẠI: `scripts/verify_docs.py` đối chiếu output của file này với PrimeSKB THẬT
(nạp bằng cách stub module `tdc`) trên mẫu ngẫu nhiên. Phải PASS trước khi tin kết quả.

Định dạng document tái lập chính xác `PrimeSKB.get_doc_info()` của stark_qa v1.1.0.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
import torch

# Thứ tự này là hợp đồng của dataset — trùng PrimeSKB.NODE_TYPES / RELATION_TYPES.
NODE_TYPES = [
    "disease", "gene/protein", "molecular_function", "drug", "pathway",
    "anatomy", "effect/phenotype", "biological_process", "cellular_component", "exposure",
]
RELATION_TYPES = [
    "ppi", "carrier", "enzyme", "target", "transporter", "contraindication",
    "indication", "off-label use", "synergistic interaction", "associated with",
    "parent-child", "phenotype absent", "phenotype present", "side effect",
    "interacts with", "linked to", "expression present", "expression absent",
]

# Chú giải riêng cho gene/protein — sao y stark_qa, kể cả thứ tự key.
GENE_PROTEIN_TEXT_EXPLAIN = {
    "name": "gene name",
    "type_of_gene": "gene types",
    "alias": "other gene names",
    "other_names": "extended other gene names",
    "genomic_pos": "genomic position",
    "generif": "PubMed text",
    "interpro": "protein family and classification information",
    "summary": "protein summary text",
}


def clean_dict(dictionary: dict, remove_values=("", "nan")) -> dict:
    """Sao y `stark_qa.tools.process_text.clean_dict`.

    PrimeSKB gọi hàm này lên `node_info` NGAY SAU khi nạp, nên phải áp dụng y hệt,
    nếu không document sẽ lệch.
    """
    new = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            new[k] = clean_dict(v, remove_values)
        elif str(v) in remove_values:
            pass
        else:
            new[k] = v
    return new


def compact_text(text: str) -> str:
    """Sao y `stark_qa.tools.process_text.compact_text`."""
    import re
    text = text.replace("\n", ". ").replace("\r", "")
    text = text.replace("- ", "")
    text = text.replace(": .", ":").replace(":.", ":")
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(".. ", ". ")
    return text


class PrimeSKB:
    """SKB của STaRK-Prime, nạp từ `<root>/processed/`."""

    def __init__(self, root: str | Path = "data/stark/prime"):
        p = Path(root) / "processed"
        self.node_info: Dict[int, dict] = clean_dict(
            pickle.load(open(p / "node_info.pkl", "rb")))
        self.node_type_dict: Dict[int, str] = pickle.load(open(p / "node_type_dict.pkl", "rb"))
        self.edge_type_dict: Dict[int, str] = pickle.load(open(p / "edge_type_dict.pkl", "rb"))
        self.node_types: torch.Tensor = torch.load(p / "node_types.pt", weights_only=False)
        self.edge_index: torch.Tensor = torch.load(p / "edge_index.pt", weights_only=False)
        self.edge_types: torch.Tensor = torch.load(p / "edge_types.pt", weights_only=False)

        self.num_nodes = int(self.node_types.numel())
        # PrimeSKB: candidate_types = NODE_TYPES => candidate_ids = mọi node, đã sort.
        self.candidate_ids: List[int] = list(range(self.num_nodes))

        self._names = np.array(
            [str(self.node_info[i].get("name", "")) for i in range(self.num_nodes)], dtype=object)
        self._build_adj()

    def _build_adj(self) -> None:
        """CSR cho từng relation type.

        PrimeSKB dùng `sparse_adj_by_type[t][idx].coalesce().indices()`, mà coalesce của
        COO vừa **khử trùng lặp** vừa **sắp xếp tăng dần**. CSR + np.unique cho kết quả
        y hệt nhưng nhanh hơn nhiều.

        Dữ liệu đã lưu cả hai chiều (đã kiểm chứng đối xứng), nên không cần to_undirected.
        """
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

    def rel_type_lst(self) -> List[str]:
        return list(self.edge_type_dict.values())

    def get_neighbor_nodes(self, idx: int, edge_type: str) -> np.ndarray:
        """Hàng xóm của `idx` theo `edge_type`, đã unique + sort tăng dần."""
        a = self._adj_by_type[edge_type]
        return a.indices[a.indptr[idx]:a.indptr[idx + 1]]

    # ------------------------------------------------------------- dựng document
    def get_rel_info(self, idx: int, rel_types: Sequence[str] | None = None,
                     n_rel: int = -1, rng: np.random.Generator | None = None) -> str:
        """Sao y `PrimeSKB.get_rel_info`.

        n_rel > 0: lấy mẫu ngẫu nhiên n_rel hàng xóm mỗi (relation_type, node_type).
        stark_qa dùng `torch.randperm`; ta dùng Generator riêng để tái lập được.
        Mặc định benchmark là n_rel=-1 (giữ hết) — xem docs/PLAN.md §4.2.
        """
        doc = ""
        rel_types = self.rel_type_lst() if rel_types is None else rel_types
        for edge_t in rel_types:
            node_ids = self.get_neighbor_nodes(idx, edge_t)
            if len(node_ids) == 0:
                continue
            doc += f"\n  {edge_t.replace(' ', '_')}: " + "{"
            ntypes = self.node_types[node_ids].numpy()
            # ⚠️ PHẢI dùng đúng `set(...)` của CPython, KHÔNG được sorted().
            # stark_qa lặp `for node_type in set(node_types.tolist())`. Thứ tự lặp của set
            # CPython không phải thứ tự tăng dần: phần tử int nằm ở slot `hash(i) % table_size`
            # (hash(i)==i với int nhỏ). Với bảng cỡ 8, id 8 -> slot 0 nên đứng TRƯỚC id 2.
            # Dùng sorted() sẽ sai thứ tự node type trong mọi document có nhiều loại hàng xóm
            # (scripts/verify_docs.py bắt được: 36/275 node lệch).
            # Dựng set từ đúng list và đúng thứ tự chèn như bản gốc => thứ tự lặp trùng khớp.
            for node_type in set(ntypes.tolist()):
                doc += f"{self.node_type_dict[node_type]}: "
                ids_t = node_ids[ntypes == node_type]
                if n_rel > 0 and len(ids_t) > n_rel:
                    r = rng if rng is not None else np.random.default_rng(42)
                    ids_t = ids_t[r.permutation(len(ids_t))[:n_rel]]
                doc += "(" + ", ".join(self._names[i] for i in ids_t) + "),"
            doc += "}"
        return "- relations:" + doc if doc else doc

    def get_doc_info(self, idx: int, add_rel: bool = False, compact: bool = False,
                     n_rel: int = -1, rng: np.random.Generator | None = None) -> str:
        """Sao y `PrimeSKB.get_doc_info`."""
        info = self.node_info[idx]
        doc = f"- name: {info.get('name')}\n"
        doc += f"- type: {info.get('type')}\n"
        doc += f"- source: {info.get('source')}\n"

        feature_text = "- details:\n"
        feature_cnt = 0
        ntype = info.get("type")
        if "details" in info:
            for key, value in info["details"].items():
                if str(value) in ["", "nan"] or key.startswith("_") or "_id" in key:
                    continue
                if ntype == "gene/protein" and key in GENE_PROTEIN_TEXT_EXPLAIN:
                    if "interpro" in key:
                        if isinstance(value, dict):
                            value = [value]
                        value = [v["desc"] for v in value]
                    if "generif" in key:
                        value = "; ".join([v["text"] for v in value])
                        value = " ".join(value.split(" ")[:50000])
                    if "genomic_pos" in key:
                        if isinstance(value, list):
                            value = value[0]
                    feature_text += f"  - {key} ({GENE_PROTEIN_TEXT_EXPLAIN[key]}): {value}\n"
                else:
                    feature_text += f"  - {key}: {value}\n"
                feature_cnt += 1
        if feature_cnt == 0:
            feature_text = ""
        doc += feature_text

        if add_rel:
            doc += self.get_rel_info(idx, n_rel=n_rel, rng=rng)
        if compact:
            doc = compact_text(doc)
        return doc

    def get_name_only_doc(self, idx: int) -> str:
        """Biến thể B6: chỉ name + type, bỏ hẳn `details`.

        Dùng để tách bạch: `details` thực sự đóng góp bao nhiêu so với chỉ cái tên.
        Không phải định dạng của stark_qa — đây là baseline do ta thêm.
        """
        info = self.node_info[idx]
        return f"- name: {info.get('name')}\n- type: {info.get('type')}\n"

    def degree(self) -> np.ndarray:
        """Bậc vô hướng của mọi node (đếm trên edge_index đã lưu 2 chiều)."""
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()
        return (np.bincount(src, minlength=self.num_nodes)
                + np.bincount(dst, minlength=self.num_nodes))
