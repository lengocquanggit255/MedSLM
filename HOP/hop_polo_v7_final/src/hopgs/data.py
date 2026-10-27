"""Lossless directed ADInt graph and QA loading.

The older RiTeK adapter rejects ambiguous names and retains duplicate triples.
This adapter deliberately does neither: a textual topic can seed several nodes,
and each original triple occurs once before its explicitly labelled inverse.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bagrel.ritek import read_basic_pickle, read_records


def normal_name(value: object) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).split())


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    topics: tuple[int, ...]  # local node IDs; ambiguous names retain every match
    answers: tuple[int, ...]
    split: str


class ADIntGraph:
    def __init__(self, root: str | Path, seed: int = 42, val_fraction: float = 0.1,
                 defer_test_answers: bool = False):
        root = Path(root)
        from bagrel.ritek import discover_adint_files
        files = discover_adint_files(root)
        meta = {int(k): v for k, v in read_basic_pickle(files["node_info_path"]).items()}
        rel_meta = {int(k): str(v) for k, v in read_basic_pickle(files["edge_type_path"]).items()}
        triples = np.loadtxt(files["kg_path"], dtype=np.int64, ndmin=2)
        if triples.shape[1] != 3:
            raise ValueError("ADint_KG.txt must contain head, relation, tail")
        self.raw_edges = len(triples)
        triples = np.unique(triples, axis=0)
        self.unique_edges = len(triples)
        self.ids = np.asarray(sorted(set(meta) | set(triples[:, 0]) | set(triples[:, 2])), dtype=np.int64)
        missing = set(self.ids) - set(meta)
        if missing:
            raise ValueError(f"{len(missing)} graph nodes lack node_info")
        self.id_to_local = {int(v): i for i, v in enumerate(self.ids)}
        self.names = [str(meta[int(v)].get("name") or v) for v in self.ids]
        type_names = [str(meta[int(v)].get("type") or meta[int(v)].get("semantic_type") or "unknown") for v in self.ids]
        self.type_names = sorted(set(type_names))
        type_lookup = {name: i for i, name in enumerate(self.type_names)}
        self.types = np.asarray([type_lookup[name] for name in type_names], dtype=np.int16)
        self.texts = [
            " | ".join(x for x in (self.names[i], type_names[i], str(meta[int(v)].get("description") or "")[:512]) if x)
            for i, v in enumerate(self.ids)
        ]
        rel_ids = np.unique(triples[:, 1]).tolist()
        if any(int(r) not in rel_meta for r in rel_ids):
            raise ValueError("relation missing from edge_type_dict")
        self.relation_ids = [int(r) for r in rel_ids]
        self.relation_names = [rel_meta[r] for r in self.relation_ids]
        rel_local = np.searchsorted(np.asarray(rel_ids), triples[:, 1]).astype(np.int16)
        head = np.searchsorted(self.ids, triples[:, 0]).astype(np.int32)
        tail = np.searchsorted(self.ids, triples[:, 2]).astype(np.int32)
        self.src = np.concatenate((head, tail))
        self.dst = np.concatenate((tail, head))
        self.rel = np.concatenate((rel_local, rel_local))
        self.direction = np.concatenate((np.zeros(len(head), np.int8), np.ones(len(head), np.int8)))
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
        packed = (((self.types[self.src].astype(np.int64) * len(rel_ids) + self.rel) * 2 + self.direction) * n_types + self.types[self.dst])
        keys, inverse = np.unique(packed, return_inverse=True)
        self.schema = inverse.astype(np.int32)
        self.schemas = []
        for key in keys:
            dest_type = int(key % n_types)
            key //= n_types
            direction = int(key % 2)
            key //= 2
            relation = int(key % len(rel_ids))
            source_type = int(key // len(rel_ids))
            self.schemas.append((source_type, relation, direction, dest_type))
        self.name_index: dict[str, set[int]] = defaultdict(set)
        for local, global_id in enumerate(self.ids):
            info = meta[int(global_id)]
            for value in (info.get("name"), info.get("alias"), info.get("aliases"), info.get("synonyms")):
                for alias in value if isinstance(value, (list, tuple, set)) else [value]:
                    if not alias:
                        continue
                    # ADInt node names frequently encode aliases as "A|B|C".
                    # Gene/protein suffixes are labels, not part of a symbol.
                    for part in str(alias).split("|"):
                        candidates = [part]
                        for suffix in (" gene", " protein", " wt allele"):
                            if part.casefold().endswith(suffix):
                                candidates.append(part[:-len(suffix)])
                        for candidate in candidates:
                            self.name_index[normal_name(candidate)].add(local)
        train_path = files["qa_path"]
        test_files = list(root.rglob("ADint_test_with_topic_entity.json"))
        if len(test_files) != 1:
            raise FileNotFoundError("expected one official ADInt test file")
        raw_train = read_records(train_path)
        # v2 reads only the question/topic fields of official test until every
        # checkpoint is locked. The streaming parser discards answer events.
        raw_test = (self._read_test_questions_only(test_files[0])
                    if defer_test_answers else read_records(test_files[0]))
        self._deferred_test_path = test_files[0] if defer_test_answers else None
        # Topic names supplied with explicit IDs are aliases, but never override
        # other nodes bearing the same name.
        for row in raw_train:
            topic = row.get("topic_entity")
            for item in topic if isinstance(topic, list) else [topic]:
                if isinstance(item, dict) and item.get("topic_id") is not None and item.get("topic_name"):
                    local = self.id_to_local.get(int(item["topic_id"]))
                    if local is not None:
                        self.name_index[normal_name(item["topic_name"])].add(local)
        train, self.skipped = self._questions(raw_train, "train")
        test, test_skipped = (self._questions_without_answers(raw_test, "test")
                              if defer_test_answers else self._questions(raw_test, "test"))
        self.skipped += test_skipped
        if len(train) < 2:
            raise ValueError("need at least two usable train QA records")
        rng = np.random.default_rng(seed)
        val_count = min(len(train) - 1, max(1, round(val_fraction * len(train))))
        validation_ids = set(rng.choice(len(train), val_count, replace=False).tolist())
        self.splits = {
            "train": [q for i, q in enumerate(train) if i not in validation_ids],
            "val": [Question(q.qid, q.question, q.topics, q.answers, "val") for i, q in enumerate(train) if i in validation_ids],
            "test": test,
        }

    @staticmethod
    def _read_test_questions_only(path: Path) -> list[dict]:
        """Decode only top-level QA metadata; skip answer JSON without decoding it."""
        import json

        def skip_value(line: str, index: int) -> int:
            """Advance over an unwanted JSON value without materializing it."""
            opening = line[index]
            if opening not in '[{"':
                while index < len(line) and line[index] not in ',}]':
                    index += 1
                return index
            stack = []
            quoted = False
            escaped = False
            while index < len(line):
                char = line[index]
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                        if not stack:
                            return index + 1
                elif char == '"':
                    quoted = True
                elif char in '[{':
                    stack.append(']' if char == '[' else '}')
                elif char in ']}':
                    if not stack or stack.pop() != char:
                        raise ValueError("unbalanced JSON while skipping test field")
                    if not stack:
                        return index + 1
                index += 1
            raise ValueError("unterminated JSON while skipping test field")

        decoder = json.JSONDecoder()
        rows = []
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = {}
                position = line.index("{") + 1
                while True:
                    while position < len(line) and line[position].isspace():
                        position += 1
                    if line[position] == "}":
                        break
                    key, position = decoder.raw_decode(line, position)
                    if not isinstance(key, str):
                        raise ValueError("non-string test metadata key")
                    while line[position].isspace():
                        position += 1
                    if line[position] != ":":
                        raise ValueError("malformed test metadata object")
                    position += 1
                    while line[position].isspace():
                        position += 1
                    if key in ("question_id", "question", "topic_entity"):
                        row[key], position = decoder.raw_decode(line, position)
                    else:
                        position = skip_value(line, position)
                    while line[position].isspace():
                        position += 1
                    if line[position] == ",":
                        position += 1
                    elif line[position] == "}":
                        break
                    else:
                        raise ValueError("malformed test metadata separator")
                if "answer" in row:
                    raise AssertionError("test answer entered metadata parser")
                rows.append(row)
        return rows

    def _questions_without_answers(self, rows: list[dict], split: str):
        good, skipped = [], []
        for i, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            topics = self._topics(row.get("topic_entity"))
            if not question or not topics:
                skipped.append({"split": split, "row": i,
                                "reason": "empty question or unresolved topic"})
                continue
            good.append(Question(str(row.get("question_id", i)), question,
                                 topics, (), split))
        return good, skipped

    def load_test_answers(self) -> None:
        """Call only after all validation decisions and reloads are complete."""
        if self._deferred_test_path is None:
            raise RuntimeError("test answers were not deferred")
        from bagrel.ritek import read_records
        decoded, skipped = self._questions(read_records(self._deferred_test_path),
                                           "test")
        old = self.splits["test"]
        assert len(decoded) == len(old)
        assert all((a.qid, a.question, a.topics) == (b.qid, b.question, b.topics)
                   for a, b in zip(decoded, old))
        self.splits["test"] = decoded
        self.skipped += skipped
        self._deferred_test_path = None

    def _topics(self, value: object) -> tuple[int, ...]:
        found: set[int] = set()
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and item.get("topic_id") is not None:
                local = self.id_to_local.get(int(item["topic_id"]))
                if local is not None:
                    found.add(local)
            elif isinstance(item, int):
                local = self.id_to_local.get(item)
                if local is not None:
                    found.add(local)
            else:
                name = item.get("topic_name") if isinstance(item, dict) else item
                if name:
                    found.update(self.name_index.get(normal_name(name), ()))
        return tuple(sorted(found))

    def _questions(self, rows: list[dict], split: str) -> tuple[list[Question], list[dict]]:
        good, skipped = [], []
        for i, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            topics = self._topics(row.get("topic_entity"))
            answer_map = row.get("answer") or {}
            answers = tuple(sorted({self.id_to_local[int(k)] for k in answer_map if int(k) in self.id_to_local})) if isinstance(answer_map, dict) else ()
            if not (question and topics and answers):
                reason = ("empty question" if not question else
                          f"unresolved topic: {row.get('topic_entity')}" if not topics else
                          "answer node absent from KG")
                skipped.append({"split": split, "row": i, "reason": reason})
                continue
            good.append(Question(str(row.get("question_id", i)), question, topics, answers, split))
        return good, skipped

    def out_edges(self, node: int) -> np.ndarray:
        return np.arange(self.indptr[node], self.indptr[node + 1], dtype=np.int32)

    def in_edges(self, node: int) -> np.ndarray:
        return self.reverse_order[self.reverse_indptr[node]:self.reverse_indptr[node + 1]]

    def edge_triple(self, edge: int) -> dict:
        src, dst = int(self.src[edge]), int(self.dst[edge])
        return {"head": int(self.ids[src]), "relation": self.relation_names[int(self.rel[edge])],
                "direction": "inverse" if self.direction[edge] else "forward", "tail": int(self.ids[dst])}

    def audit(self) -> dict:
        return {"nodes": len(self.ids), "triples_raw": self.raw_edges, "triples_unique": self.unique_edges,
                "directed_edges": len(self.src), "relations": len(self.relation_ids), "schemas": len(self.schemas),
                "split_sizes": {k: len(v) for k, v in self.splits.items()}, "ambiguous_topics": {
                    k: sum(len(q.topics) > 1 for q in rows) for k, rows in self.splits.items()},
                "skipped": self.skipped}
