"""Codebook ngữ nghĩa S: mpnet -> SVD -> RVQ (GS-Quant) nhiều tầng, cùng chỉ mục tiền tố.

V7 còn huấn luyện mã cấu trúc G cho từng schema; Setting B không dùng G ở bất kỳ bước nào
(executor `semantic`, decoder `include_g=False`), nên V8 bỏ hẳn phần đó.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from .common import digest
from .third_party.gs_quant.rq import ResidualVectorQuantizer


def text_embeddings(texts: list[str], model_dir: str, dim: int, seed: int, batch_size: int,
                    device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_dir, device=device)
    result = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    result = np.asarray(result, np.float32)
    if result.shape[1] > dim:
        result = TruncatedSVD(n_components=dim, n_iter=5, random_state=seed).fit_transform(result).astype(np.float32)
    elif result.shape[1] < dim:
        raise ValueError(f"embedding model returned {result.shape[1]} dimensions, expected at least {dim}")
    return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)


def fit_rvq(features: np.ndarray, levels: int, size: int, steps: int, batch_size: int,
            seed: int, device: torch.device) -> ResidualVectorQuantizer:
    """Khởi tạo từng tầng bằng MiniBatchKMeans trên phần dư, rồi tinh chỉnh `steps` bước Adam (như V7)."""
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
    return model.eval()


@dataclass
class Codebook:
    node_codes: np.ndarray                       # [n_nodes, levels] int16
    prefix_index: dict[tuple[int, ...], np.ndarray]

    @property
    def levels(self) -> int:
        return int(self.node_codes.shape[1])

    @property
    def size(self) -> int:
        return int(self.node_codes.max()) + 1

    def prefix_nodes(self, prefix: tuple[int, ...]) -> np.ndarray:
        return self.prefix_index.get(prefix, np.empty(0, np.int32))


def build_codebook(node_codes: np.ndarray) -> Codebook:
    prefixes: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for node, codes in enumerate(node_codes.tolist()):
        for length in range(1, len(codes) + 1):
            prefixes[tuple(codes[:length])].append(node)
    return Codebook(node_codes, {k: np.asarray(v, np.int32) for k, v in prefixes.items()})


def train_codebook(graph, output: Path, settings: dict, *, model_dir: str, seed: int,
                   device: str) -> tuple[Codebook, dict]:
    output.mkdir(parents=True, exist_ok=True)
    target = torch.device(device)
    embeddings = text_embeddings(graph.texts, model_dir, settings["embedding_dim"], seed,
                                 settings["embedding_batch_size"], device)
    np.save(output / "node_embeddings.npy", embeddings)
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(embeddings), min(settings["fit_sample"], len(embeddings)), replace=False)
    model = fit_rvq(embeddings[sample_ids], settings["semantic_levels"], settings["semantic_codes"],
                    settings["semantic_steps"], settings["batch_size"], seed, target)
    node_codes = np.empty((len(embeddings), settings["semantic_levels"]), np.int16)
    with torch.inference_mode():
        for start in range(0, len(embeddings), settings["batch_size"]):
            x = torch.as_tensor(embeddings[start:start + settings["batch_size"]], device=target)
            node_codes[start:start + len(x)] = model(x)[2].cpu().numpy().astype(np.int16)
    torch.save(model.cpu().state_dict(), output / "semantic_rvq.pt")
    np.save(output / "node_codes.npy", node_codes)
    audit = {"embedding_dim": int(embeddings.shape[1]), "fit_sample": int(len(sample_ids)),
             "active_codes": [int(np.unique(node_codes[:, i]).size) for i in range(node_codes.shape[1])],
             "unique_full_codes": int(len({tuple(row) for row in node_codes.tolist()})),
             "node_codes_sha256": digest(output / "node_codes.npy"),
             "node_embeddings_sha256": digest(output / "node_embeddings.npy"),
             "semantic_rvq_sha256": digest(output / "semantic_rvq.pt")}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return build_codebook(node_codes), audit


def load_codebook(output: Path) -> Codebook:
    return build_codebook(np.load(output / "node_codes.npy"))
