"""GS-Quant residual semantic VQ and within-schema structural edge VQ."""
from __future__ import annotations

import sys
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer

from .data import ADIntGraph

# Import the actual vendored GS-Quant implementation, rather than a new VQ.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "GS-Quant"))
from codebook.rq import ResidualVectorQuantizer  # noqa: E402


def text_embeddings(texts: list[str], backend: str, model_name: str, dim: int, seed: int, batch_size: int) -> np.ndarray:
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        result = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
        result = np.asarray(result, np.float32)
        if result.shape[1] > dim:
            result = TruncatedSVD(n_components=dim, n_iter=5,
                                  random_state=seed).fit_transform(result).astype(np.float32)
        elif result.shape[1] < dim:
            raise ValueError(f"embedding model returned {result.shape[1]} dimensions, expected at least {dim}")
        return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)
    if backend != "hash-svd":
        raise ValueError(f"unknown text embedding backend: {backend}")
    vectorizer = HashingVectorizer(n_features=max(4096, dim * 16), ngram_range=(1, 2),
                                  alternate_sign=False, norm="l2")
    sparse = vectorizer.transform(texts)
    n_components = min(dim, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if n_components < 2:
        raise ValueError("need at least three node texts")
    result = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=5).fit_transform(sparse).astype(np.float32)
    if n_components < dim:
        result = np.pad(result, ((0, 0), (0, dim - n_components)))
    return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)


def fit_rvq(features: np.ndarray, levels: int, size: int, steps: int, batch_size: int,
            seed: int, device: torch.device) -> tuple[ResidualVectorQuantizer, np.ndarray]:
    if not len(features):
        raise ValueError("empty feature matrix")
    size = min(size, len(features))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ResidualVectorQuantizer(levels, size, features.shape[1], 0.25).to(device)
    sample = np.asarray(features, np.float32)
    residual = sample.copy()
    for level, layer in enumerate(model.vq_layers):
        km = MiniBatchKMeans(n_clusters=size, random_state=seed + level,
                             batch_size=min(max(32, size * 4), len(sample)), n_init=1, max_iter=30)
        km.fit(residual)
        centers = km.cluster_centers_.astype(np.float32)
        with torch.no_grad():
            layer.embedding.weight.copy_(torch.from_numpy(centers).to(device))
        residual -= centers[km.predict(residual)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        ids = rng.integers(len(sample), size=min(batch_size, len(sample)))
        x = torch.as_tensor(sample[ids], device=device)
        quantized, vq_loss, _ = model(x)
        loss = vq_loss + (quantized.sum(1) - x).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    codes = np.empty((len(sample), levels), np.int16)
    with torch.inference_mode():
        for start in range(0, len(sample), batch_size):
            x = torch.as_tensor(sample[start:start + batch_size], device=device)
            codes[start:start + len(x)] = model(x)[2].cpu().numpy().astype(np.int16)
    return model, codes


@dataclass
class Codebooks:
    node_codes: np.ndarray
    edge_codes: np.ndarray
    prefix_index: dict[tuple[int, ...], np.ndarray]
    edge_index: "EdgeIndex"
    semantic_audit: dict
    structural_audit: dict

    def prefix_nodes(self, prefix: tuple[int, ...]) -> np.ndarray:
        return self.prefix_index.get(prefix, np.empty(0, np.int32))


class EdgeIndex:
    """Compact sorted index for (source, action, code) -> real edge IDs."""

    def __init__(self, graph: ADIntGraph, actions: np.ndarray, codes: np.ndarray):
        self.actions = int(actions.max()) + 1
        self.code_count = int(codes.max()) + 1
        self.source_indptr = graph.indptr
        packed = ((graph.src.astype(np.int64) * self.actions + actions) * self.code_count + codes)
        self.order = np.argsort(packed, kind="stable").astype(np.int32)
        self.keys = packed[self.order]

    def get(self, key: tuple[int, int, int]) -> np.ndarray | None:
        source, action, code = key
        packed = (source * self.actions + action) * self.code_count + code
        start, stop = int(self.source_indptr[source]), int(self.source_indptr[source + 1])
        view = self.keys[start:stop]
        left = start + int(np.searchsorted(view, packed, side="left"))
        right = start + int(np.searchsorted(view, packed, side="right"))
        return self.order[left:right] if right > left else None

    def groups(self, source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, stop = int(self.source_indptr[source]), int(self.source_indptr[source + 1])
        if start == stop:
            empty = np.empty(0, np.int32)
            return empty, empty, empty
        keys, counts = np.unique(self.keys[start:stop], return_counts=True)
        local = keys - source * self.actions * self.code_count
        return (local // self.code_count).astype(np.int32), (local % self.code_count).astype(np.int32), counts.astype(np.int32)


def build_indexes(graph: ADIntGraph, node_codes: np.ndarray, edge_codes: np.ndarray,
                  semantic_audit: dict | None = None, structural_audit: dict | None = None) -> Codebooks:
    prefixes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for node, codes in enumerate(node_codes):
        for length in range(1, len(codes) + 1):
            prefixes[tuple(int(v) for v in codes[:length])].append(node)
    return Codebooks(node_codes, edge_codes,
                     {k: np.asarray(v, np.int32) for k, v in prefixes.items()},
                     EdgeIndex(graph, graph.schema, edge_codes),
                     semantic_audit or {}, structural_audit or {})


def structural_profiles(graph: ADIntGraph) -> tuple[np.ndarray, np.ndarray]:
    """Directed typed-relation neighborhood counts at each endpoint."""
    n, r = len(graph.ids), len(graph.relation_ids)
    counts = np.zeros((n, 2 * r), np.float32)
    np.add.at(counts, (graph.src, graph.rel + graph.direction * r), 1)
    degrees = counts.sum(1).astype(np.float32)
    counts /= np.maximum(1, degrees[:, None])
    return counts, np.log1p(degrees)


def edge_features(graph: ADIntGraph, edge_ids: np.ndarray, profiles: np.ndarray, log_degree: np.ndarray) -> np.ndarray:
    source, dest = graph.src[edge_ids], graph.dst[edge_ids]
    return np.concatenate((profiles[source], profiles[dest],
                           log_degree[source, None] / 16, log_degree[dest, None] / 16), axis=1).astype(np.float32)


def train_codebooks(graph: ADIntGraph, output: Path, *, backend: str = "sentence-transformers",
                    model_name: str = "sentence-transformers/all-mpnet-base-v2", dim: int = 128,
                    s_levels: int = 3, s_size: int = 64, s_steps: int = 300,
                    g_size: int = 8, g_min_edges: int = 256, g_steps: int = 30,
                    fit_sample: int = 30_000, batch_size: int = 2048,
                    embedding_batch_size: int = 32, seed: int = 42,
                    device: str = "cpu") -> Codebooks:
    output.mkdir(parents=True, exist_ok=True)
    target = torch.device(device)
    embeddings_path = output / "node_embeddings.npy"
    config_path = output / "embedding_config.json"
    embedding_config = {"backend": backend, "model_name": model_name if backend == "sentence-transformers" else None,
                        "dim": dim, "nodes": len(graph.ids)}
    if embeddings_path.exists() and config_path.exists() and json.loads(config_path.read_text()) == embedding_config:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if len(embeddings) != len(graph.ids):
            raise ValueError("cached node embeddings belong to another graph")
    else:
        embeddings = text_embeddings(graph.texts, backend, model_name, dim, seed, embedding_batch_size)
        np.save(embeddings_path, embeddings)
        config_path.write_text(json.dumps(embedding_config, indent=2))
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(embeddings), min(fit_sample, len(embeddings)), replace=False)
    print(f"semantic VQ: fitting {len(sample_ids)} nodes, {s_levels} levels × {s_size} codes", flush=True)
    s_model, _ = fit_rvq(np.asarray(embeddings[sample_ids], np.float32), s_levels, s_size,
                         s_steps, batch_size, seed, target)
    node_codes = np.empty((len(embeddings), s_levels), np.int16)
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            x = torch.as_tensor(np.asarray(embeddings[start:start + batch_size], np.float32), device=target)
            node_codes[start:start + len(x)] = s_model(x)[2].cpu().numpy().astype(np.int16)
    torch.save(s_model.cpu().state_dict(), output / "semantic_rvq.pt")
    np.save(output / "node_codes.npy", node_codes)
    semantic_audit = {"backend": backend, "model": model_name if backend == "sentence-transformers" else "hash-svd",
                      "active_codes": [int(np.unique(node_codes[:, i]).size) for i in range(s_levels)]}

    profiles, log_degree = structural_profiles(graph)
    edge_codes = np.zeros(len(graph.src), np.int16)
    schema_order = np.argsort(graph.schema, kind="stable")
    schema_counts = np.bincount(graph.schema, minlength=len(graph.schemas))
    schema_start = np.concatenate(([0], np.cumsum(schema_counts)))
    models = {}
    usage = {}
    print(f"structural VQ: {len(schema_counts)} schemas, {int((schema_counts >= g_min_edges).sum())} trainable", flush=True)
    for schema_id, count in enumerate(schema_counts):
        if count < g_min_edges:
            usage[str(schema_id)] = {"edges": int(count), "active_codes": 1, "fallback": True}
            continue
        ids = schema_order[schema_start[schema_id]:schema_start[schema_id + 1]]
        fit_ids = ids[rng.choice(len(ids), min(fit_sample, len(ids)), replace=False)]
        model, _ = fit_rvq(edge_features(graph, fit_ids, profiles, log_degree), 1, g_size,
                           g_steps, batch_size, seed + schema_id + 1, target)
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start + batch_size]
                x = torch.as_tensor(edge_features(graph, batch_ids, profiles, log_degree), device=target)
                edge_codes[batch_ids] = model(x)[2][:, 0].cpu().numpy().astype(np.int16)
        models[str(schema_id)] = {k: v.cpu() for k, v in model.state_dict().items()}
        usage[str(schema_id)] = {"edges": int(count), "active_codes": int(np.unique(edge_codes[ids]).size), "fallback": False}
        if len(models) % 100 == 0:
            print(f"structural VQ: trained {len(models)} schemas", flush=True)
    torch.save(models, output / "structural_rvq_by_schema.pt")
    np.save(output / "edge_codes.npy", edge_codes)
    structural_audit = {"schema_usage": usage, "trained_schemas": len(models),
                        "fallback_schemas": len(graph.schemas) - len(models)}
    return build_indexes(graph, node_codes, edge_codes, semantic_audit, structural_audit)
