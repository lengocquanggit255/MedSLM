"""Áp lớp phủ text làm giàu (data/enrich/enrich.parquet) lên BẢN SAO node_info.

Dữ liệu gốc không bao giờ bị sửa: hàm trả về dict mới, chỉ các node được làm giàu mới bị sao chép.
Text thêm vào nằm ở key riêng trong `details`, nên `get_doc_info` in ra
  - ext_definition: ...
  - ext_synonyms: ...
và có thể phân biệt hoàn toàn với field gốc.

Ví dụ:
    spec = EnrichSpec(tiers=(0, 1), fields=("definition", "synonyms"))
    skb.node_info = apply(skb.node_info, spec)
    tag = variant_name("add_rel", spec)   # "add_rel+def+syn[T0,T1]"
    corpus.build(skb, "add_rel", tag=tag) # BẮT BUỘC truyền tag, nếu không sẽ báo lỗi (xem corpus.py)

Field: "definition" | "synonyms" (EXACT) | "synonyms_related" (RELATED, chỉ cho ablation).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .ontology import norm

DEFAULT = Path(__file__).resolve().parents[2] / "data/enrich/enrich.parquet"
_TIER_ORDER = {0: 0, 1: 1, 2: 2}


@dataclass(frozen=True)
class EnrichSpec:
    tiers: Sequence[int] = (0,)
    fields: Sequence[str] = ("definition",)
    sources: Optional[Sequence[str]] = None    # tiền tố nguồn, vd ("GO", "PUBCHEM"); None = tất cả
    exclude_sources: Sequence[str] = ()        # vd ("PUBCHEM:DrugBank",)
    only_missing: bool = False                 # chỉ node gốc KHÔNG có mô tả NL
    require_g3: bool = True                    # chỉ lấy dòng accepted (G3 + đối chiếu ID chéo)
    max_def_chars: int = 1000                  # ngân sách độ dài ext_definition / node
    max_syn: int = 15                          # số synonym tối đa / node
    # Đối chứng: xáo ext_definition giữa các node CÙNG loại; synonym giữ nguyên (xáo synonym = gán
    # tên gọi của node khác, gây hại chủ động, sẽ phóng đại khoảng cách thật-vs-placebo).
    placebo: bool = False
    exclude_types: Sequence[str] = ()          # vd ("molecular_function",): loại node không làm giàu
    # Bỏ synonym trùng TÊN của một node khác ("HIP" của gene HHIP trùng node anatomy "hip"). Audit
    # 2026-10-06: 3,078 node bị gắn synonym kiểu này.
    drop_name_collisions: bool = True
    # Bỏ ext_definition mà >= 80% token đã có trong `details` gốc của node. Audit 2026-10-06:
    # 97% định nghĩa MONDO, 76% NCIT của disease đã nằm sẵn trong details (PrimeKG lấy từ chính
    # các nguồn đó) -> thêm lại chỉ nhân đôi tần suất từ (biến gây nhiễu cho BM25), không thêm tin.
    drop_base_dup: bool = True
    # T2 (drug ghép theo TÊN) chỉ giữ dòng đã đối chiếu ID chéo khớp (`t2_xcheck == "agree"`).
    # Dòng không kiểm được ("unknown": toàn bộ NCIT-T2, và MeSH/ChEBI-T2 của drug không có ChEBI
    # ghép theo DrugBank ID) bị bỏ theo mặc định — quyết định 2026-10-06. False = giữ cả (ablation).
    t2_verified_only: bool = True
    seed: int = 42


# 3 biến thể benchmark đã chốt (docs/PLAN_ENRICH.md §10.9). V0 = corpus gốc, đã chạy.
#   E − V0 = tác động tổng · E − D = phần của synonym · E − P = phần của NỘI DUNG định nghĩa
BENCH_SPECS = {
    "E": EnrichSpec(tiers=(0, 1, 2), fields=("definition", "synonyms")),
    "D": EnrichSpec(tiers=(0, 1, 2), fields=("definition",)),
    "P": EnrichSpec(tiers=(0, 1, 2), fields=("definition", "synonyms"), placebo=True),
}

# Ablation (2026-10-08), chỉ chạy BM25 + MedCPT zero-shot:
#   tầng   : T0, T01 (so với E = T0–T2)
#   loại   : leave-one-out — E bỏ làm giàu của MỘT nhóm loại node ⇒ E − x<G> = đóng góp của nhóm G
NODE_GROUPS = {
    "GO": ("biological_process", "molecular_function", "cellular_component"),
    "PHENO": ("effect/phenotype",),
    "ANAT": ("anatomy",),
    "EXPO": ("exposure",),
    "DIS": ("disease",),
    "DRUG": ("drug",),
    "GENE": ("gene/protein",),
}
_DS = ("definition", "synonyms")
ABLATION_SPECS = {
    "T0": EnrichSpec(tiers=(0,), fields=_DS),
    "T01": EnrichSpec(tiers=(0, 1), fields=_DS),
    **{f"x{g}": EnrichSpec(tiers=(0, 1, 2), fields=_DS, exclude_types=t) for g, t in NODE_GROUPS.items()},
}
# Biến thể "clean" (2026-10-13): chỉ node CHƯA có mô tả gốc, chỉ T0 + T1. So trực tiếp với V0.
CLEAN_SPECS = {"C": EnrichSpec(tiers=(0, 1), fields=_DS, only_missing=True)}
ALL_SPECS = {**BENCH_SPECS, **ABLATION_SPECS, **CLEAN_SPECS}

_FIELD_TAG = {"definition": "def", "synonyms": "syn", "synonyms_related": "rsyn"}
_TYPE_TAG = {"molecular_function": "mf", "biological_process": "bp", "cellular_component": "cc",
             "effect/phenotype": "pheno", "gene/protein": "gene"}


def variant_name(base: str, spec: EnrichSpec) -> str:
    f = "+".join(_FIELD_TAG[x] for x in spec.fields)
    tag = f"{base}+{'placebo-' if spec.placebo else ''}{f}[{','.join(f'T{t}' for t in spec.tiers)}"
    # không dùng ":" — tag còn làm tên file cache (corpus.build), Windows cấm ":" trong tên file
    if spec.sources:
        tag += "@" + ",".join(s.lower() for s in spec.sources)
    if spec.exclude_sources:
        tag += "@-" + ",-".join(s.lower().replace(":", "~") for s in spec.exclude_sources)
    tag += "]"
    if spec.only_missing:
        tag += "-missing"
    if not spec.require_g3:
        tag += "-nog3"
    if not spec.drop_base_dup:
        tag += "-keepdup"
    if 2 in spec.tiers and not spec.t2_verified_only:
        tag += "-t2all"
    if spec.exclude_types:
        tag += "-x" + ",".join(_TYPE_TAG.get(t, t.replace("/", "-")) for t in spec.exclude_types)
    if not spec.drop_name_collisions:
        tag += "-keepcoll"
    return tag


def _toks(s) -> set:
    return set(norm(str(s)).split())


def base_dup_mask(rows: pd.DataFrame, node_info: dict, thr: float = 0.8) -> pd.Series:
    """True nếu >= thr token của text đã có trong details GỐC của node (bỏ qua key ext_*)."""
    cache = {}

    def base(i):
        if i not in cache:
            d = node_info[i].get("details") or {}
            cache[i] = _toks(" ".join(str(v) for k, v in d.items() if not str(k).startswith("ext_")))
        return cache[i]

    def dup(r):
        t = _toks(r.text)
        return bool(t) and len(t & base(r.node_idx)) / len(t) >= thr
    return pd.Series([dup(r) for r in rows.itertuples()], index=rows.index, dtype=bool)


def select(spec: EnrichSpec, path: Path | str = DEFAULT) -> pd.DataFrame:
    df = pd.read_parquet(path)
    m = df.tier.isin(spec.tiers) & df.field.isin(spec.fields)
    if spec.sources:
        m &= df.source.str.split(":").str[0].isin(spec.sources)
    for s in spec.exclude_sources:
        m &= ~df.source.str.startswith(s)
    if spec.only_missing:
        m &= ~df.base_has_nl
    if spec.require_g3:  # `accepted` = G3 ∧ không mâu thuẫn ID chéo (enrich_verify.py)
        m &= df["accepted"] if "accepted" in df else df.get("g3_pass", True)
    if spec.t2_verified_only:
        m &= (df.tier != 2) | (df["t2_xcheck"] == "agree")
    if spec.exclude_types:
        m &= ~df.node_type.isin(spec.exclude_types)
    return df[m].copy()


def _compose(rows: pd.DataFrame, spec: EnrichSpec, keep_syn=None) -> dict:
    """node_idx -> {'ext_definition': str, 'ext_synonyms': str}. Ưu tiên tầng thấp (tin cậy hơn);
    synonym EXACT đứng trước RELATED. keep_syn(i, s) -> False thì bỏ synonym s của node i."""
    rows = rows.assign(_o=rows.tier.map(_TIER_ORDER), _f=(rows.field == "synonyms_related").astype(int))
    rows = rows.sort_values(["node_idx", "_f", "_o", "source"])
    out = {}
    defs = rows[rows.field == "definition"]
    for i, g in defs.groupby("node_idx", sort=False):
        parts, n = [], 0
        for t in dict.fromkeys(g.text):
            if n + len(t) > spec.max_def_chars and parts:
                break
            parts.append(t)
            n += len(t)
        out.setdefault(i, {})["ext_definition"] = " ".join(parts)[: spec.max_def_chars]
    syns = rows[rows.field.str.startswith("synonyms")]
    for i, g in syns.groupby("node_idx", sort=False):
        syn, seen = [], set()
        for t in g.text:
            for s in t.split("; "):
                if s and s.lower() not in seen and (keep_syn is None or keep_syn(i, s)):
                    seen.add(s.lower())
                    syn.append(s)
        if syn:
            out.setdefault(i, {})["ext_synonyms"] = "; ".join(syn[: spec.max_syn])
    return out


def name_collision_filter(node_info: dict):
    """keep_syn(i, s): False nếu s (chuẩn hóa) là TÊN của một node khác i."""
    owners = {}
    for i, v in node_info.items():
        owners.setdefault(norm(v.get("name", "")), set()).add(i)
    return lambda i, s: not (owners.get(norm(s), set()) - {i})


def apply(node_info: dict, spec: EnrichSpec, path: Path | str = DEFAULT) -> dict:
    rows = select(spec, path)
    if spec.drop_base_dup:
        d = rows[(rows.field == "definition") & rows.base_has_nl]
        rows = rows.drop(d.index[base_dup_mask(d, node_info)])
    keep = name_collision_filter(node_info) if spec.drop_name_collisions else None
    add = _compose(rows, spec, keep)
    if spec.placebo:  # chỉ xáo ext_definition giữa node CÙNG loại; tập node & phân phối độ dài giữ nguyên
        rng = np.random.default_rng(spec.seed)
        by_type = {}
        for i, ext in add.items():
            if "ext_definition" in ext:
                by_type.setdefault(node_info[i]["type"], []).append(i)
        defs = {i: add[i]["ext_definition"] for ids in by_type.values() for i in ids}
        for ids in by_type.values():
            perm = rng.permutation(len(ids))
            for a, b in zip(ids, perm):
                add[a] = {**add[a], "ext_definition": defs[ids[b]]}
    new = dict(node_info)  # shallow: node không đổi dùng chung object gốc (chỉ đọc)
    for i, ext in add.items():
        info = copy.deepcopy(node_info[i])
        info.setdefault("details", {})
        info["details"].update(ext)
        new[i] = info
    return new
