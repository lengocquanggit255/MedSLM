from __future__ import annotations

import json
import pickle
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BasicUnpickler(pickle.Unpickler):
    """Only allow primitive containers in the dataset metadata pickles."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"unsupported object in metadata pickle: {module}.{name}"
        )


def read_basic_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return BasicUnpickler(handle).load()


def read_records(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        if "question" in value or "context" in value:
            value = [value]
        else:
            value = next(
                (value[key] for key in ("data", "records", "questions") if key in value),
                None,
            )
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must be a JSON list or JSONL objects")
    return value


def _normalize_name(value: str) -> str:
    return " ".join(
        re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).split()
    )


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)


@dataclass(frozen=True)
class RitekRecord:
    qid: int
    raw_qid: str
    question: str
    seeds: np.ndarray
    answers: np.ndarray


class RitekSplit:
    def __init__(
        self,
        data: "RitekData",
        name: str,
        records: list[RitekRecord],
        query_embeddings: np.ndarray,
    ):
        self.data = data
        self.name = name
        self.records = records
        self.qids = np.asarray([record.qid for record in records], dtype=np.int64)
        self.cls = np.asarray(query_embeddings, dtype=np.float32)
        # Compatibility attributes; the HQA retrieval path only consumes cls.
        self.tok = np.zeros((len(records), 1, 1), dtype=np.int8)
        self.mask = np.ones((len(records), 1), dtype=np.int8)
        self._subgraphs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def nodes(self, index: int) -> np.ndarray:
        if index not in self._subgraphs:
            self._subgraphs[index] = self.data.query_subgraph(
                self.records[index].seeds, self.cls[index]
            )
        return self._subgraphs[index][0]

    def ppr(self, index: int) -> np.ndarray:
        self.nodes(index)
        return self._subgraphs[index][1]

    def seeds(self, index: int) -> np.ndarray:
        return self.records[index].seeds

    def answers(self, index: int) -> np.ndarray:
        return self.records[index].answers

    def query_embedding_index(self, index: int) -> int:
        return int(index)

    def usable_indices(
        self, require_answer_in_subgraph: bool = True, limit: int = 0
    ) -> list[int]:
        result: list[int] = []
        for index in range(len(self)):
            if require_answer_in_subgraph and not np.intersect1d(
                self.nodes(index), self.answers(index), assume_unique=False
            ).size:
                continue
            result.append(index)
            if limit and len(result) >= limit:
                break
        return result


class RitekData:
    """Adapter from the ADInt/PharmKG notebook files to the BAG-RL API."""

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        edge_type: np.ndarray,
        node_embeddings: np.ndarray,
        relation_embeddings: np.ndarray,
        records: list[RitekRecord],
        query_embeddings: np.ndarray,
        global_ids: np.ndarray,
        relation_names: list[str],
        max_subgraph_nodes: int,
        max_subgraph_hops: int,
        seed: int,
        explicit_split_ids: dict[str, np.ndarray] | None = None,
        node_type_ids: np.ndarray | None = None,
        node_type_names: list[str] | None = None,
    ):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.edge_type = np.asarray(edge_type, dtype=np.int16)
        self.node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
        self.relation_embeddings = np.asarray(relation_embeddings, dtype=np.float32)
        self.global_ids = np.asarray(global_ids, dtype=np.int64)
        self.relation_names = relation_names
        self.n_nodes = len(self.global_ids)
        self.n_relations = len(relation_names)
        self.node_type_ids = (
            np.asarray(node_type_ids, dtype=np.int32)
            if node_type_ids is not None
            else np.zeros(self.n_nodes, dtype=np.int32)
        )
        self.node_type_names = node_type_names or ["unknown"]
        if len(self.node_type_ids) != self.n_nodes:
            raise ValueError("node_type_ids length does not match node count")
        self.max_subgraph_nodes = int(max_subgraph_nodes)
        self.max_subgraph_hops = int(max_subgraph_hops)
        self.seed = int(seed)
        self.data_dir = Path(".")
        self._semantic_score_cache: dict[tuple[str, int], np.ndarray] = {}

        if explicit_split_ids is None:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(records))
            n_test = max(1, round(0.10 * len(records))) if len(records) >= 3 else 0
            n_val = max(1, round(0.10 * len(records))) if len(records) >= 3 else 0
            split_ids = {
                "test": order[:n_test],
                "val": order[n_test : n_test + n_val],
                "train": order[n_test + n_val :],
            }
            if not len(split_ids["train"]):
                split_ids["train"] = order
        else:
            if set(explicit_split_ids) != {"train", "val", "test"}:
                raise ValueError("explicit_split_ids must contain train, val, and test")
            split_ids = {
                name: np.asarray(ids, dtype=np.int64)
                for name, ids in explicit_split_ids.items()
            }
            all_ids = np.concatenate(list(split_ids.values()))
            if (
                len(all_ids) != len(records)
                or len(np.unique(all_ids)) != len(records)
                or np.any(all_ids < 0)
                or np.any(all_ids >= len(records))
            ):
                raise ValueError("explicit split indices must partition the records")
        self.splits: dict[str, RitekSplit] = {}
        for name, ids in split_ids.items():
            selected = [records[int(index)] for index in ids]
            embeddings = query_embeddings[np.asarray(ids, dtype=np.int64)]
            self.splits[name] = RitekSplit(self, name, selected, embeddings)

    def split(self, name: str) -> RitekSplit:
        return self.splits[name]

    def query_semantic_scores(self, split: str, query_index: int) -> np.ndarray:
        key = (split, int(query_index))
        if key not in self._semantic_score_cache:
            store = self.split(split)
            nodes = store.nodes(query_index)
            query = store.cls[query_index]
            self._semantic_score_cache[key] = np.asarray(
                self.node_embeddings[nodes] @ query, dtype=np.float32
            )
        return self._semantic_score_cache[key]

    def query_subgraph(
        self, seeds: np.ndarray, query_embedding: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query-aware best-first expansion, capped like a PPR-2000 graph."""

        import heapq

        seeds = np.unique(np.asarray(seeds, dtype=np.int64))
        visited = set(int(node) for node in seeds)
        nodes = [int(node) for node in seeds]
        depths = [0 for _ in seeds]
        expanded: set[int] = set()
        heap: list[tuple[float, int, int]] = []

        def expand(source: int, depth: int) -> None:
            if source in expanded or depth >= self.max_subgraph_hops:
                return
            expanded.add(source)
            start, stop = int(self.indptr[source]), int(self.indptr[source + 1])
            destinations = self.indices[start:stop]
            if not len(destinations):
                return
            scores = self.node_embeddings[destinations] @ query_embedding
            for destination, score in zip(destinations, scores):
                destination = int(destination)
                priority = 0.75 * float(score) + 0.25 / (depth + 2)
                heapq.heappush(heap, (-priority, destination, depth + 1))

        for seed in nodes:
            expand(seed, 0)
        while heap and len(nodes) < self.max_subgraph_nodes:
            _negative_score, node, depth = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            nodes.append(node)
            depths.append(depth)
            expand(node, depth)
        depth_array = np.asarray(depths, dtype=np.float32)
        pseudo_ppr = 1.0 / (1.0 + depth_array)
        return np.asarray(nodes, dtype=np.int32), pseudo_ppr


def _metadata_text(global_id: int, metadata: dict) -> str:
    fields = [str(metadata.get("name") or global_id)]
    for key in ("type", "semantic_type", "category", "label"):
        value = metadata.get(key)
        if value and str(value) not in fields:
            fields.append(str(value))
    description = str(metadata.get("description") or "").strip()
    if description:
        # Long UMLS definitions make character n-gram matrices unnecessarily
        # large. The leading 512 characters retain the useful definition while
        # keeping preprocessing bounded on Kaggle.
        fields.append(description[:512])
    return " | ".join(fields)


def _text_embeddings(
    node_texts: list[str],
    relation_texts: list[str],
    questions: list[str],
    dimension: int,
    seed: int,
    cache_dir: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_path = cache_dir / "ritek_node_embeddings.npy"
    relation_path = cache_dir / "ritek_relation_embeddings.npy"
    query_path = cache_dir / "ritek_query_embeddings.npy"
    expected = ((len(node_texts), dimension), (len(relation_texts), dimension), (len(questions), dimension))
    if not force and all(path.exists() for path in (node_path, relation_path, query_path)):
        node = np.load(node_path, mmap_mode="r")
        relation = np.load(relation_path)
        query = np.load(query_path)
        if (node.shape, relation.shape, query.shape) == expected:
            return node, relation, query

    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the local text embedding baseline") from exc

    n_features = max(2_048, dimension * 16)
    vectorizer = HashingVectorizer(
        n_features=n_features,
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        alternate_sign=False,
        norm="l2",
    )
    node_sparse = vectorizer.transform(node_texts)
    max_components = min(dimension, node_sparse.shape[0] - 1, node_sparse.shape[1] - 1)
    if max_components < 2:
        raise ValueError("not enough nodes/features to build text embeddings")
    svd = TruncatedSVD(n_components=max_components, n_iter=5, random_state=seed)
    node = svd.fit_transform(node_sparse).astype(np.float32)
    relation = svd.transform(vectorizer.transform(relation_texts)).astype(np.float32)
    query = svd.transform(vectorizer.transform(questions)).astype(np.float32)
    if max_components < dimension:
        pad = dimension - max_components
        node = np.pad(node, ((0, 0), (0, pad)))
        relation = np.pad(relation, ((0, 0), (0, pad)))
        query = np.pad(query, ((0, 0), (0, pad)))
    node, relation, query = _unit_rows(node), _unit_rows(relation), _unit_rows(query)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(node_path, node)
    np.save(relation_path, relation)
    np.save(query_path, query)
    return np.load(node_path, mmap_mode="r"), relation, query


def load_ritek_dataset(
    kg_path: str | Path,
    node_info_path: str | Path,
    edge_type_path: str | Path,
    qa_path: str | Path,
    cache_dir: str | Path,
    embedding_dim: int = 128,
    add_inverse_edges: bool = True,
    max_subgraph_nodes: int = 2_000,
    max_subgraph_hops: int = 6,
    seed: int = 42,
    force_embeddings: bool = False,
    split_qa_paths: dict[str, str | Path] | None = None,
    fair_directed_graph: bool = False,
) -> tuple[RitekData, dict[str, Any]]:
    """Load the exact raw schema used by the linked ADInt Kaggle notebook."""

    kg_path, node_info_path = Path(kg_path), Path(node_info_path)
    edge_type_path, qa_path = Path(edge_type_path), Path(qa_path)
    metadata = {int(key): value for key, value in read_basic_pickle(node_info_path).items()}
    relation_metadata = {
        int(key): str(value) for key, value in read_basic_pickle(edge_type_path).items()
    }
    triples = np.loadtxt(kg_path, dtype=np.int64, ndmin=2)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"{kg_path} must contain head relation tail integer triples")
    raw_edges = len(triples)
    if fair_directed_graph:
        triples = np.unique(triples, axis=0)
    global_ids = np.asarray(
        sorted(set(metadata).union(triples[:, 0].tolist(), triples[:, 2].tolist())),
        dtype=np.int64,
    )
    if any(int(node) not in metadata for node in global_ids):
        raise ValueError("the graph contains nodes missing from node_info")
    relation_ids = sorted(set(int(value) for value in triples[:, 1]))
    if any(relation not in relation_metadata for relation in relation_ids):
        raise ValueError("the graph contains relations missing from edge_type_dict")
    relation_to_local = {relation: index for index, relation in enumerate(relation_ids)}
    source = np.searchsorted(global_ids, triples[:, 0]).astype(np.int64)
    destination = np.searchsorted(global_ids, triples[:, 2]).astype(np.int64)
    relation = np.asarray(
        [relation_to_local[int(value)] for value in triples[:, 1]], dtype=np.int64
    )
    relation_names = [relation_metadata[value].replace("_", " ") for value in relation_ids]
    original_edges = len(source)
    if add_inverse_edges:
        source, destination = (
            np.concatenate([source, destination]),
            np.concatenate([destination, source]),
        )
        relation = np.concatenate([relation, relation + len(relation_ids)])
        relation_names += [f"inverse {name}" for name in relation_names]
    order = np.lexsort((destination, relation, source))
    source, destination, relation = source[order], destination[order], relation[order]
    indptr = np.zeros(len(global_ids) + 1, dtype=np.int64)
    np.add.at(indptr, source + 1, 1)
    np.cumsum(indptr, out=indptr)

    name_index: dict[str, set[int]] = defaultdict(set)
    id_to_local = {int(value): index for index, value in enumerate(global_ids)}
    for global_id, value in metadata.items():
        fields = ("name", "alias", "aliases", "synonyms") if fair_directed_graph else ("name",)
        for field in fields:
            raw = value.get(field)
            aliases = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for alias in aliases:
                if not alias:
                    continue
                parts = str(alias).split("|") if fair_directed_graph else [alias]
                for part in parts:
                    names = [part]
                    if fair_directed_graph:
                        for suffix in (" gene", " protein", " wt allele"):
                            if part.casefold().endswith(suffix):
                                names.append(part[:-len(suffix)])
                    for name in names:
                        name_index[_normalize_name(name)].add(global_id)
    if split_qa_paths is None:
        split_raw_records = {"train": read_records(qa_path)}
    else:
        if set(split_qa_paths) not in ({"train", "test"}, {"train", "val", "test"}):
            raise ValueError("split_qa_paths must contain train and test, with optional val")
        split_raw_records = {
            split: read_records(path) for split, path in split_qa_paths.items()
        }
    raw_records = [
        row for split_rows in split_raw_records.values() for row in split_rows
    ]
    alias_records = split_raw_records["train"] if fair_directed_graph else raw_records
    for row in alias_records:
        topic = row.get("topic_entity")
        candidates = topic if isinstance(topic, list) else [topic]
        for item in candidates:
            if isinstance(item, dict) and item.get("topic_id") is not None and item.get(
                "topic_name"
            ):
                name_index[_normalize_name(item["topic_name"])].add(int(item["topic_id"]))

    def resolve_topic(value: Any) -> np.ndarray:
        items = value if isinstance(value, list) else [value]
        resolved: set[int] = set()
        for item in items:
            if isinstance(item, dict) and item.get("topic_id") is not None:
                candidates = {int(item["topic_id"])}
            elif isinstance(item, (int, np.integer)):
                candidates = {int(item)}
            elif isinstance(item, str):
                candidates = name_index.get(_normalize_name(item), set())
            elif isinstance(item, dict) and item.get("topic_name"):
                candidates = name_index.get(_normalize_name(item["topic_name"]), set())
            else:
                raise ValueError("invalid topic_entity")
            if not candidates or (not fair_directed_graph and len(candidates) != 1):
                raise ValueError("topic name is not resolvable" if fair_directed_graph else "topic name is not uniquely resolvable")
            for global_id in candidates:
                if global_id not in id_to_local:
                    raise ValueError("topic node is outside the graph")
                resolved.add(id_to_local[global_id])
        if not resolved:
            raise ValueError("empty topic_entity")
        return np.asarray(sorted(resolved), dtype=np.int32)

    records: list[RitekRecord] = []
    skipped: list[dict[str, Any]] = []
    explicit_split_ids: dict[str, np.ndarray] | None = {} if split_qa_paths else None
    row_index = 0
    for split_name, split_rows in split_raw_records.items():
        split_start = len(records)
        for row in split_rows:
            try:
                question = str(row.get("question", "")).strip()
                if not question:
                    raise ValueError("empty question")
                answer = row.get("answer")
                if not isinstance(answer, dict) or not answer:
                    raise ValueError("answer must be a non-empty {node_id: name} object")
                answers = np.asarray(
                    sorted({id_to_local[int(value)] for value in answer if int(value) in id_to_local}),
                    dtype=np.int32,
                )
                if not len(answers):
                    raise ValueError("no answer node occurs in the graph")
                records.append(
                    RitekRecord(
                        qid=len(records),
                        raw_qid=str(row.get("question_id", row_index)),
                        question=question,
                        seeds=resolve_topic(row.get("topic_entity")),
                        answers=answers,
                    )
                )
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                skipped.append({"split": split_name, "row": row_index, "reason": str(exc)})
            row_index += 1
        if explicit_split_ids is not None:
            explicit_split_ids[split_name] = np.arange(
                split_start, len(records), dtype=np.int64
            )
    validation_source = None
    if explicit_split_ids is not None and "val" not in explicit_split_ids:
        train_ids = explicit_split_ids["train"]
        if len(train_ids) < 2:
            raise ValueError("cannot hold out validation examples from fewer than two usable training questions")
        n_val = min(max(1, round(0.10 * len(train_ids))), len(train_ids) - 1)
        # Match ADIntGraph's holdout exactly for paired HOP/MoR comparisons.
        val_ids = np.random.default_rng(seed).choice(train_ids, n_val, replace=False)
        explicit_split_ids["val"] = np.sort(val_ids)
        explicit_split_ids["train"] = train_ids[~np.isin(train_ids, val_ids)]
        validation_source = f"seeded 10% holdout of usable training QA (seed={seed})"
    elif explicit_split_ids is not None:
        validation_source = str(split_qa_paths["val"])
    if len(records) < 3:
        raise ValueError("fewer than three usable QA records")
    if explicit_split_ids is not None:
        empty = [name for name, ids in explicit_split_ids.items() if not len(ids)]
        if empty:
            reasons = [item for item in skipped if item["split"] in empty][:5]
            raise ValueError(f"QA split(s) {empty} have no usable records; sample rejections: {reasons}")

    node_texts = [_metadata_text(int(node), metadata[int(node)]) for node in global_ids]
    type_labels = [
        str(
            metadata[int(node)].get("type")
            or metadata[int(node)].get("node_type")
            or metadata[int(node)].get("semantic_type")
            or metadata[int(node)].get("category")
            or "unknown"
        )
        for node in global_ids
    ]
    node_type_names = sorted(set(type_labels))
    type_to_id = {name: index for index, name in enumerate(node_type_names)}
    node_type_ids = np.asarray([type_to_id[name] for name in type_labels], dtype=np.int32)
    questions = [record.question for record in records]
    node_embedding, relation_embedding, query_embedding = _text_embeddings(
        node_texts,
        relation_names,
        questions,
        embedding_dim,
        seed,
        Path(cache_dir),
        force_embeddings,
    )
    data = RitekData(
        indptr,
        destination.astype(np.int32),
        relation.astype(np.int16),
        node_embedding,
        relation_embedding,
        records,
        query_embedding,
        global_ids,
        relation_names,
        max_subgraph_nodes,
        max_subgraph_hops,
        seed,
        explicit_split_ids=explicit_split_ids,
        node_type_ids=node_type_ids,
        node_type_names=node_type_names,
    )
    audit = {
        "kg_path": str(kg_path),
        "node_info_path": str(node_info_path),
        "edge_type_path": str(edge_type_path),
        "qa_path": str(qa_path),
        "n_nodes": data.n_nodes,
        "original_edges": original_edges,
        "raw_edges": raw_edges,
        "fair_directed_graph": fair_directed_graph,
        "retrieval_edges": len(data.indices),
        "n_relations": data.n_relations,
        "n_node_types": len(node_type_names),
        "add_inverse_edges": add_inverse_edges,
        "raw_questions": len(raw_records),
        "usable_questions": len(records),
        "skipped_questions": len(skipped),
        "embedding_backend": "word_unigram_bigram_hashing_svd",
        "embedding_dim": embedding_dim,
        "max_subgraph_nodes": max_subgraph_nodes,
        "max_subgraph_hops": max_subgraph_hops,
        "split_sizes": {key: len(value) for key, value in data.splits.items()},
        "split_sources": (
            {**{name: str(path) for name, path in split_qa_paths.items()},
             "val": validation_source}
            if split_qa_paths is not None else None
        ),
        "skipped_examples": skipped[:20],
    }
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "ritek_data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return data, audit


def discover_adint_files(root: str | Path) -> dict[str, Path]:
    """Resolve the four files shown in the linked Kaggle notebook."""

    root = Path(root)
    patterns = {
        "kg_path": r"ADint_KG(?:\s*\(\d+\))*\.txt",
        "node_info_path": r"node_info(?:\s*\(\d+\))*\.pkl",
        "edge_type_path": r"edge_type_dict(?:\s*\(\d+\))*\.pkl",
        "qa_path": r"ADint_train_with_topic_entity(?:\s*\(\d+\))*\.json(?:l)?",
    }
    result: dict[str, Path] = {}
    files = [path for path in root.rglob("*") if path.is_file()]
    for key, pattern in patterns.items():
        matches = [path for path in files if re.fullmatch(pattern, path.name, re.IGNORECASE)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one {pattern} under {root}, found {len(matches)}"
            )
        result[key] = matches[0]
    return result
