"""Cỡ nhóm cạnh cạnh tranh với cạnh vàng ở nút cha của đáp án, theo tầng prior A/S (căn cứ chọn `local_top_k`).

python HOP/hop_polo_v8/tools/hub_groups.py prime --embeddings <codebook/node_embeddings.npy>   (từ gốc repo)
python HOP/hop_polo_v8/tools/hub_groups.py pharmkg --codes <codebook/node_codes.npy>
python HOP/hop_polo_v8/tools/hub_groups.py adint                                    (chưa có codebook: chỉ G0)

Trên đường teacher (ngắn nhất, rẻ nhất) của mỗi đáp án train trong mẫu `--sample` câu (seed 42), đếm số cạnh ra của
nút cha cùng tầng prior với cạnh vàng: G0 = cùng action, Gk = cùng action và cùng k tầng đầu mã S của nút đích. Khi
chương trình Flan đúng tới tầng k, cạnh vàng chỉ còn cạnh tranh trong nhóm Gk. `--embeddings` fit lại RVQ đúng cấu
hình codebook của `configs/default.json` (dùng khi chỉ có embedding, vd từ run smoke fit trên ít mẫu). Ghi kết quả vào
`docs/hub_groups.json` (gộp theo bộ).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOP / "src"))

from hop import labels  # noqa: E402
from hop.codebook import fit_rvq  # noqa: E402
from hop.data import Graph  # noqa: E402
from hop.pipeline import load_config  # noqa: E402

CONFIGS = {"pharmkg": "default.json", "prime": "prime.json", "adint": "adint.json"}
THRESHOLDS = (5, 10, 30, 60, 120, 250, 500, 1000)


def pct(values) -> dict:
    values = np.asarray(values, np.float64)
    if not len(values):
        return {}
    out = {f"p{p}": round(float(np.percentile(values, p)), 1) for p in (25, 50, 75, 90, 99)}
    out.update(mean=round(float(values.mean()), 1), max=int(values.max()), n=int(len(values)))
    out["share_le"] = {str(t): round(float((values <= t).mean()), 3) for t in THRESHOLDS}
    return out


def refit_codes(embeddings: np.ndarray) -> np.ndarray:
    settings = load_config(HOP / "configs" / "default.json")["codebook"]
    ids = np.random.default_rng(42).choice(len(embeddings), min(settings["fit_sample"], len(embeddings)), replace=False)
    model = fit_rvq(embeddings[ids], settings["semantic_levels"], settings["semantic_codes"],
                    settings["semantic_steps"], settings["batch_size"], 42, torch.device("cpu"))
    codes = np.empty((len(embeddings), settings["semantic_levels"]), np.int16)
    with torch.inference_mode():
        for start in range(0, len(embeddings), 8192):
            codes[start:start + 8192] = model(torch.as_tensor(embeddings[start:start + 8192]))[2].numpy()
    return codes


def groups(graph: Graph, codes, parent: int, edge: int) -> dict:
    start, end = int(graph.indptr[parent]), int(graph.indptr[parent + 1])
    match = graph.schema[start:end] == graph.schema[edge]
    out = {"degree": end - start, "G0": int(match.sum())}
    if codes is not None:
        target = codes[int(graph.dst[edge])]
        destination = codes[graph.dst[start:end]]
        for level in range(codes.shape[1]):
            match &= destination[:, level] == target[level]
            out[f"G{level + 1}"] = int(match.sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(CONFIGS))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--codes", type=Path)
    source.add_argument("--embeddings", type=Path)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--out", type=Path, default=HOP / "docs" / "hub_groups.json")
    args = parser.parse_args()
    started = time.perf_counter()
    max_hops = load_config(HOP / "configs" / CONFIGS[args.dataset])["max_hops"]
    graph = Graph(HOP / "inputs" / args.dataset)
    codes = (np.load(args.codes) if args.codes else
             refit_codes(np.load(args.embeddings)) if args.embeddings else None)
    if codes is not None and len(codes) != len(graph.ids):
        raise ValueError("codes/embeddings do not match the graph")
    sample = random.Random(42).sample(graph.splits["train"], min(args.sample, len(graph.splits["train"])))
    last_hop = {depth: [] for depth in range(1, max_hops + 1)}
    first_hop = {depth: [] for depth in range(2, max_hops + 1)}
    depths = Counter()
    for q in sample:
        for answer in q.answers:
            buckets, shortest, _, _ = labels.answer_paths(graph, q, answer, max_hops=max_hops)
            depths[str(shortest)] += 1
            if not shortest:
                continue
            _pos, path = buckets[shortest][0]
            last_hop[shortest].append(groups(graph, codes, int(graph.src[path[-1]]), path[-1]))
            if shortest >= 2:
                first_hop[shortest].append(groups(graph, codes, int(graph.src[path[0]]), path[0]))
    report = {"questions": len(sample), "max_hops": max_hops, "answers_by_shortest_depth": dict(depths),
              "codes": "given" if args.codes else "refit" if args.embeddings else None,
              "unique_full_codes": None if codes is None else len({tuple(r) for r in codes.tolist()}),
              "last_hop_parent": {str(d): {k: pct([r[k] for r in rows]) for k in (rows[0] if rows else {})}
                                  for d, rows in last_hop.items()},
              "first_hop_for_deeper_answers": {str(d): {k: pct([r[k] for r in rows]) for k in (rows[0] if rows else {})}
                                               for d, rows in first_hop.items()},
              "seconds": round(time.perf_counter() - started, 1)}
    merged = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    merged[args.dataset] = report
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
