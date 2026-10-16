"""Parser ontology cho phase làm giàu text: OBO, NCIT FLAT, MeSH XML.

Mọi parser trả về dict[id -> Term] cùng một dạng, để bước ghép (scripts/enrich_build.py) không
cần biết nguồn. Đọc theo dòng / iterparse nên RAM chỉ tỉ lệ với số term, không với kích thước
file (MeSH supp2021.xml là 719 MB).
"""
from __future__ import annotations

import gzip
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree.ElementTree import iterparse


@dataclass
class Term:
    id: str
    label: str = ""
    definition: str = ""
    synonyms: List[Tuple[str, str]] = field(default_factory=list)  # (text, scope)
    obsolete: bool = False
    xrefs: List[Tuple[str, str]] = field(default_factory=list)     # (curie, source annotation)
    alt_ids: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def norm(s: str) -> str:
    """Chuẩn hóa nhãn để so khớp: NFKC, thường, bỏ dấu câu, gộp khoảng trắng."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return " ".join(s.split())


def strip_obsolete(s: str) -> str:
    return re.sub(r"^\s*(obsolete[.:]?\s*)", "", s, flags=re.I).strip()


def _open(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


_Q = r'"((?:[^"\\]|\\.)*)"'
_DEF = re.compile(r"^def:\s*" + _Q)
_SYN = re.compile(r"^synonym:\s*" + _Q + r"\s*(\w+)?")
_XREF = re.compile(r"^xref:\s*(\S+)(?:\s*\{(.*)\})?")


def _unq(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", " ").strip()


def parse_obo(path) -> Dict[str, Term]:
    terms: Dict[str, Term] = {}
    cur: Term | None = None
    in_term = False
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    terms[cur.id] = cur
                cur, in_term = None, line.strip() == "[Term]"
                continue
            if not in_term or not line or line.startswith("!"):
                continue
            k, _, v = line.partition(": ")
            if k == "id":
                cur = Term(id=v.strip())
            elif cur is None:
                continue
            elif k == "name":
                cur.label = v.strip()
            elif k == "def":
                m = _DEF.match(line)
                if m:
                    cur.definition = _unq(m.group(1))
            elif k == "synonym":
                m = _SYN.match(line)
                if m:
                    cur.synonyms.append((_unq(m.group(1)), (m.group(2) or "").upper()))
            elif k == "is_obsolete":
                cur.obsolete = v.strip() == "true"
            elif k == "xref":
                m = _XREF.match(line)
                if m:
                    cur.xrefs.append((m.group(1), m.group(2) or ""))
            elif k == "alt_id":
                cur.alt_ids.append(v.strip())
            elif k in ("replaced_by", "consider"):
                cur.extra.setdefault(k, []).append(v.strip())
    if cur is not None:
        terms[cur.id] = cur
    return terms


def parse_ncit_flat(path) -> Dict[str, Term]:
    """NCIT FLAT: code, IRI, parents, synonyms(|), definition, display name, status, semantic type, subset."""
    terms: Dict[str, Term] = {}
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        with io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace") as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 8:
                    continue
                syns = [s for s in c[3].split("|") if s]
                t = Term(id=f"NCIT:{c[0]}", label=c[5] or (syns[0] if syns else ""),
                         definition=c[4].strip(),
                         synonyms=[(s, "EXACT") for s in syns],
                         obsolete="Obsolete" in c[6] or "Retired" in c[6],
                         extra={"semantic_types": [s for s in c[7].split("|") if s]})
                terms[t.id] = t
    return terms


_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# Mảnh Note của MeSH supplementary là ghi chú biên mục, không phải định nghĩa
# ("RN given refers to …", "was MH 1965-92", "use X to search Y 1966-92", "structure in first source")
_MESH_EDITORIAL = re.compile(
    r"\bRN given\b|\bwas (MH|see)\b|\bwere see\b|\bto search\b|\bfirst source\b|\bstructure given\b"
    r"|\bsee also\b|\bindex(ed)? (under|as)\b|\b(19|20)\d\d-(\d\d)\b|\bheading\b|\bRN\b", re.I)


def clean_mesh_note(s: str) -> str:
    """Bỏ các mảnh (tách theo ';') là ghi chú biên mục; giữ phần mô tả."""
    parts = [p.strip() for p in str(s).split(";")]
    return "; ".join(p for p in parts if p and not _MESH_EDITORIAL.search(p))


def _txt(el, path):
    x = el.find(path)
    return (x.text or "").strip() if x is not None and x.text else ""


def parse_mesh_xml(path) -> Dict[str, Term]:
    """MeSH descriptor (desc*.xml) hoặc supplementary (supp*.xml); id dạng MESH:D012345 / MESH:C012345."""
    terms: Dict[str, Term] = {}
    for _, el in iterparse(str(path), events=("end",)):
        if el.tag == "DescriptorRecord":
            ui = _txt(el, "DescriptorUI")
            t = Term(id=f"MESH:{ui}", label=_txt(el, "DescriptorName/String"))
            for c in el.iterfind("ConceptList/Concept"):
                if c.get("PreferredConceptYN") == "Y":
                    t.definition = _txt(c, "ScopeNote")
            t.extra["tree"] = [x.text for x in el.iterfind("TreeNumberList/TreeNumber")]
        elif el.tag == "SupplementalRecord":
            ui = _txt(el, "SupplementalRecordUI")
            t = Term(id=f"MESH:{ui}", label=_txt(el, "SupplementalRecordName/String"),
                     definition=clean_mesh_note(_txt(el, "Note")))
            t.extra["raw_note"] = _txt(el, "Note")
            t.extra["mapped_to"] = [x.text.lstrip("*") for x in
                                    el.iterfind("HeadingMappedToList/HeadingMappedTo/DescriptorReferredTo/DescriptorUI")]
        else:
            continue
        # số đăng ký CAS (RegistryNumber có thể là CAS, UNII hoặc "0") -> dùng để đối chiếu chéo ChEBI
        rn = " ".join(x.text or "" for x in el.iter() if x.tag in ("RegistryNumber", "RelatedRegistryNumber"))
        t.extra["cas"] = sorted(set(_CAS.findall(rn)))
        seen = {norm(t.label)}
        for s in el.iterfind("ConceptList/Concept/TermList/Term/String"):
            if s.text and norm(s.text) not in seen:
                seen.add(norm(s.text))
                t.synonyms.append((s.text.strip(), "EXACT"))
        terms[t.id] = t
        el.clear()
    return terms


def load(path) -> Dict[str, Term]:
    p = str(path)
    if p.endswith(".obo") or p.endswith(".obo.gz"):
        return parse_obo(path)
    if p.endswith(".FLAT.zip"):
        return parse_ncit_flat(path)
    if p.endswith(".xml"):
        return parse_mesh_xml(path)
    raise ValueError(f"không biết định dạng: {p}")
