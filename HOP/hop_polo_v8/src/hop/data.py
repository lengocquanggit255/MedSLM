"""Đồ thị có hướng + câu hỏi, đọc từ bundle chuẩn hoá do `tools/prep_inputs.py` dựng.

Bundle (một thư mục cho mỗi bộ dữ liệu):
  manifest.json        sha256 của mọi file dưới đây, số đếm, chính sách split
  triples.npy          int64 [M, 3] = (head, relation, tail), ID gốc của bộ dữ liệu
  nodes.jsonl          {id, name, type, description}
  relations.json       {relation_id: tên}
  qa_train.jsonl, qa_val.jsonl   {qid, question, topics: [ID], answers: [ID]}
  qa_test.jsonl        {qid, question, topics: [ID]}  -- KHÔNG có đáp án
  test_answers.jsonl   {qid, answers: [ID]}           -- chỉ đọc sau khi khoá checkpoint

Khác V7: cạnh ngược chỉ được thêm cho triple chưa có chiều ngược trong KG (`inverse="missing"`).
PrimeKG lưu sẵn cả hai chiều cho 100% triple, nên V7 nhân đôi số cạnh mà không thêm đường đi nào.
`inverse="always"` tái tạo đúng đồ thị V7 (dùng cho test tương đương).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import digest, read_jsonl

BUNDLE_FILES = ("triples.npy", "nodes.jsonl", "relations.json", "qa_train.jsonl",
                "qa_val.jsonl", "qa_test.jsonl", "test_answers.jsonl")


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    topics: tuple[int, ...]   # ID cục bộ
    answers: tuple[int, ...]  # ID cục bộ; rỗng với test trước khi mở khoá
    split: str


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != set(BUNDLE_FILES):
        raise AssertionError(f"bundle files differ: {set(manifest['files']) ^ set(BUNDLE_FILES)}")
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise AssertionError(f"bundle hash mismatch: {name}")
    return manifest


class Graph:
    def __init__(self, root: str | Path, *, inverse: str = "missing", verify: bool = True):
        if inverse not in ("missing", "always"):
            raise ValueError(inverse)
        root = Path(root)
        self.root = root
        self.manifest = verify_bundle(root) if verify else json.loads(
            (root / "manifest.json").read_text(encoding="utf-8"))
        nodes = {int(row["id"]): row for row in read_jsonl(root / "nodes.jsonl")}
        rel_meta = {int(k): str(v) for k, v in
                    json.loads((root / "relations.json").read_text(encoding="utf-8")).items()}
        triples = np.load(root / "triples.npy")
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples.npy must be [M, 3]")
        self.raw_edges = len(triples)
        triples = np.unique(triples.astype(np.int64), axis=0)
        self.unique_edges = len(triples)
        self.ids = np.asarray(sorted(set(nodes) | set(triples[:, 0].tolist()) | set(triples[:, 2].tolist())),
                              dtype=np.int64)
        missing = set(self.ids.tolist()) - set(nodes)
        if missing:
            raise ValueError(f"{len(missing)} graph nodes lack node metadata")
        self.id_to_local = {int(v): i for i, v in enumerate(self.ids)}
        self.names = [str(nodes[int(v)].get("name") or v) for v in self.ids]
        type_names = [str(nodes[int(v)].get("type") or "unknown") for v in self.ids]
        self.type_names = sorted(set(type_names))
        type_lookup = {name: i for i, name in enumerate(self.type_names)}
        self.types = np.asarray([type_lookup[name] for name in type_names], dtype=np.int16)
        # Cùng định dạng văn bản nút như V7: "tên | loại | mô tả[:512]".
        self.texts = [" | ".join(x for x in (self.names[i], type_names[i],
                                             str(nodes[int(v)].get("description") or "")[:512]) if x)
                      for i, v in enumerate(self.ids)]
        rel_ids = np.unique(triples[:, 1])
        if any(int(r) not in rel_meta for r in rel_ids):
            raise ValueError("relation missing from relations.json")
        self.relation_ids = [int(r) for r in rel_ids]
        self.relation_names = [rel_meta[r] for r in self.relation_ids]
        rel_local = np.searchsorted(rel_ids, triples[:, 1]).astype(np.int16)
        head = np.searchsorted(self.ids, triples[:, 0]).astype(np.int32)
        tail = np.searchsorted(self.ids, triples[:, 2]).astype(np.int32)
        if inverse == "always":
            add = np.ones(len(head), dtype=bool)
        else:
            n, r = np.int64(len(self.ids)), np.int64(len(rel_ids))
            forward_key = (head.astype(np.int64) * r + rel_local) * n + tail
            reverse_key = (tail.astype(np.int64) * r + rel_local) * n + head
            add = ~np.isin(reverse_key, forward_key)
        self.inverse_policy = inverse
        self.inverse_added = int(add.sum())
        self.src = np.concatenate((head, tail[add]))
        self.dst = np.concatenate((tail, head[add]))
        self.rel = np.concatenate((rel_local, rel_local[add]))
        self.direction = np.concatenate((np.zeros(len(head), np.int8), np.ones(int(add.sum()), np.int8)))
        order = np.lexsort((self.dst, self.direction, self.rel, self.src))
        for key in ("src", "dst", "rel", "direction"):
            setattr(self, key, getattr(self, key)[order])
        self.indptr = np.zeros(len(self.ids) + 1, np.int64)
        np.add.at(self.indptr, self.src + 1, 1)
        np.cumsum(self.indptr, out=self.indptr)
        self.reverse_order = np.argsort(self.dst, kind="stable").astype(np.int32)
        self.reverse_indptr = np.zeros(len(self.ids) + 1, np.int64)
        np.add.at(self.reverse_indptr, self.dst + 1, 1)
        np.cumsum(self.reverse_indptr, out=self.reverse_indptr)
        n_types = len(self.type_names)
        packed = (((self.types[self.src].astype(np.int64) * len(rel_ids) + self.rel) * 2 + self.direction)
                  * n_types + self.types[self.dst])
        keys, inverse_index = np.unique(packed, return_inverse=True)
        self.schema = inverse_index.astype(np.int32)
        self.schemas = []
        for key in keys:
            dest_type = int(key % n_types)
            key //= n_types
            direction = int(key % 2)
            key //= 2
            relation = int(key % len(rel_ids))
            self.schemas.append((int(key // len(rel_ids)), relation, direction, dest_type))
        self.skipped: list[dict] = []
        self.splits = {"train": self._questions(read_jsonl(root / "qa_train.jsonl"), "train", True),
                       "val": self._questions(read_jsonl(root / "qa_val.jsonl"), "val", True),
                       "test": self._questions(read_jsonl(root / "qa_test.jsonl"), "test", False)}
        self._test_answers_loaded = False

    def _local(self, values) -> tuple[int, ...]:
        return tuple(sorted({self.id_to_local[int(v)] for v in values if int(v) in self.id_to_local}))

    def _questions(self, rows: list[dict], split: str, labelled: bool) -> list[Question]:
        good = []
        for row in rows:
            if not labelled and "answers" in row:
                raise AssertionError("test answers must not be in qa_test.jsonl")
            topics = self._local(row.get("topics") or ())
            answers = self._local(row.get("answers") or ()) if labelled else ()
            if labelled and not (topics and answers):
                # prep đã lọc; câu nào lọt tới đây là lỗi bundle.
                raise AssertionError(f"{split} question {row['qid']} lacks topics or answers")
            if not topics:
                self.skipped.append({"split": split, "qid": str(row["qid"]), "reason": "no topic"})
            good.append(Question(str(row["qid"]), str(row["question"]).strip(), topics, answers, split))
        return good

    def load_test_answers(self) -> None:
        """Chỉ gọi sau khi mọi checkpoint đã được chọn, nạp lại và khoá hash."""
        if self._test_answers_loaded:
            raise RuntimeError("test answers already loaded")
        answers = {str(r["qid"]): r["answers"] for r in read_jsonl(self.root / "test_answers.jsonl")}
        old = self.splits["test"]
        missing = [q.qid for q in old if q.qid not in answers]
        if missing:
            raise AssertionError(f"{len(missing)} test questions have no answer row")
        self.splits["test"] = [Question(q.qid, q.question, q.topics, self._local(answers[q.qid]), "test")
                               for q in old]
        if any(not q.answers for q in self.splits["test"]):
            raise AssertionError("a test question has no answer inside the graph")
        self._test_answers_loaded = True

    def limit_splits(self, limits: dict) -> None:
        """Chỉ cho smoke run: giữ N câu đầu mỗi split (thứ tự bundle)."""
        for split, count in limits.items():
            self.splits[split] = self.splits[split][:int(count)]

    def out_edges(self, node: int) -> np.ndarray:
        return np.arange(self.indptr[node], self.indptr[node + 1], dtype=np.int32)

    def in_edges(self, node: int) -> np.ndarray:
        return self.reverse_order[self.reverse_indptr[node]:self.reverse_indptr[node + 1]]

    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for array in (self.ids, self.src, self.dst, self.rel, self.direction):
            h.update(array.tobytes())
        for value in self.texts:
            h.update(value.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def audit(self) -> dict:
        return {"nodes": len(self.ids), "triples_raw": self.raw_edges, "triples_unique": self.unique_edges,
                "inverse_policy": self.inverse_policy, "inverse_edges_added": self.inverse_added,
                "directed_edges": len(self.src), "relations": len(self.relation_ids),
                "schemas": len(self.schemas),
                "split_sizes": {k: len(v) for k, v in self.splits.items()},
                "questions_without_topic": {k: sum(not q.topics for q in v) for k, v in self.splits.items()},
                "multi_topic_questions": {k: sum(len(q.topics) > 1 for q in v) for k, v in self.splits.items()},
                "skipped": self.skipped}
