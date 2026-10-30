"""Thống kê dữ liệu gắn với từng siêu tham số HOP (giá trị V7, chỉnh trên PharmKG) cho PharmKG / Prime / ADint.

python HOP/hop_polo_v8/tools/hparam_stats.py [--sample 500]   (từ gốc repo; ghi HOP/hop_polo_v8/docs/hparam_stats.json)

Chỉ đọc bundle `inputs/<bộ>/` (train/val, không đụng đáp án test). Khoảng cách hop chính xác (BFS toàn đồ
thị) tính trên mọi câu train + val; các đại lượng theo đường vàng tính trên một mẫu cố định `--sample` câu
train (seed 42) vì khai thác đường 3 hop trên Prime tốn tới ~0,3 s/đáp án.
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOP / "src"))

from hop import labels  # noqa: E402
from hop.data import Graph  # noqa: E402
from hop.pipeline import load_config  # noqa: E402

CONFIGS = {"pharmkg": "default.json", "prime": "prime.json", "adint": "adint.json"}


def pct(values, points=(50, 90, 99)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    out = {f"p{p}": round(float(np.percentile(values, p)), 1) for p in points}
    out.update(mean=round(float(values.mean()), 1), max=round(float(values.max()), 1), n=int(len(values)))
    return out


def share(values, threshold) -> float:
    values = np.asarray(values)
    return round(float((values <= threshold).mean()), 3) if len(values) else None


def exact_hops(graph, questions) -> dict:
    """Khoảng cách ngắn nhất thật (BFS toàn đồ thị, không giới hạn hop) topic -> đáp án; nhiều topic lấy min."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    n = len(graph.ids)
    adjacency = csr_matrix((np.ones(len(graph.dst), np.int8), graph.dst, graph.indptr), shape=(n, n))
    topics = sorted({t for q in questions for t in q.topics})
    distance = {}
    for start in range(0, len(topics), 256):
        block = topics[start:start + 256]
        for t, row in zip(block, shortest_path(adjacency, directed=True, unweighted=True, indices=block)):
            distance[t] = row
    per_answer, per_question = Counter(), Counter()
    for q in questions:
        hops = [min(distance[t][a] for t in q.topics) for a in q.answers]
        per_answer.update("inf" if np.isinf(h) else str(int(h)) for h in hops)
        worst = max(hops)
        per_question["inf" if np.isinf(worst) else str(int(worst))] += 1

    def shares(counter):
        total = sum(counter.values())
        return {k: round(counter[k] / total, 4) for k in sorted(counter, key=lambda k: (k == "inf", int(k) if k != "inf" else 0))}

    finite = [int(k) for k in per_answer if k != "inf"]
    return {"answers": sum(per_answer.values()), "max_finite_hops": max(finite),
            "per_answer": shares(per_answer), "per_question_all_answers": shares(per_question)}


def dataset_stats(name: str, sample_size: int) -> dict:
    started = time.perf_counter()
    max_hops = load_config(HOP / "configs" / CONFIGS[name])["max_hops"]
    graph = Graph(HOP / "inputs" / name)
    degree = np.diff(graph.indptr)
    train, val = graph.splits["train"], graph.splits["val"]
    labelled = train + val
    result = {
        "graph": {"nodes": len(graph.ids), "directed_edges": int(len(graph.src)), "triples": graph.unique_edges,
                  "inverse_edges_added": graph.inverse_added, "relations": len(graph.relation_ids),
                  "schemas": len(graph.schemas), "node_types": len(graph.type_names),
                  "actions_relation_direction_target_type": len({s[1:] for s in graph.schemas}),
                  "out_degree": pct(degree, (50, 90, 99, 99.9))},
        "qa": {"train": len(train), "val": len(val), "test": len(graph.splits["test"]),
               "updates_per_epoch_batch16": int(np.ceil(len(train) / 16)),
               "answers_per_question": pct([len(q.answers) for q in labelled]),
               "share_questions_more_than_20_answers": round(float(np.mean([len(q.answers) > 20 for q in labelled])), 3),
               "topics_per_question": pct([len(q.topics) for q in labelled]),
               "topic_out_degree": pct([degree[t] for q in train for t in q.topics], (50, 90, 99))},
    }
    sample = random.Random(42).sample(train, min(sample_size, len(train)))
    hops, first_group, mid_degree, routes_2hop, direct_routes = Counter(), [], [], [], []
    for q in sample:
        first_by_schema = Counter(int(graph.schema[e]) for t in set(q.topics) for e in graph.out_edges(t))
        for answer in q.answers:
            buckets, shortest, _, _ = labels.answer_paths(graph, q, answer, max_hops=max_hops)
            hops[str(shortest)] += 1
            if shortest in (1, 2):
                _pos, path = buckets[shortest][0]
                # chi phí hop đầu khi thực thi nhãn = số cạnh từ mọi topic có cùng schema với cạnh vàng
                first_group.append(first_by_schema[int(graph.schema[path[0]])])
                if shortest == 2:
                    mid_degree.append(int(degree[int(graph.dst[path[0]])]))
            _routes, row = labels.mine_routes(graph, q, answer, 256, 42)
            if row["shortest"] == 1:
                direct_routes.append(row["before"])
            elif row["shortest"] == 2:
                routes_2hop.append(row["before"])
    total = sum(hops.values())
    result["answers_sample"] = {
        "questions": len(sample), "answers": total, "max_hops": max_hops,
        "shortest_hops_share": {k: round(v / total, 3) for k, v in sorted(hops.items())},
        "first_hop_group_edges": pct(first_group),
        "first_hop_group_share_within": {str(c): share(first_group, c) for c in (200, 500, 1000)},
        "two_hop_mid_out_degree": pct(mid_degree),
        "two_hop_routes_per_answer": pct(routes_2hop),
        "two_hop_routes_share_within_256": share(routes_2hop, 256),
        "direct_routes_per_answer": pct(direct_routes),
    }
    result["exact_hops_train_val"] = exact_hops(graph, labelled)
    result["synthetic_coverage_at_10000"] = {"nodes": round(min(1.0, 10000 / len(graph.ids)), 3),
                                             "edges": round(min(1.0, 10000 / len(graph.src)), 4)}
    result["codebook_3x64"] = {"full_codes_available": 64 ** 3, "nodes_per_full_code_if_uniform":
                               round(len(graph.ids) / 64 ** 3, 3)}
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--datasets", nargs="+", default=["pharmkg", "prime", "adint"])
    parser.add_argument("--out", type=Path, default=HOP / "docs" / "hparam_stats.json")
    args = parser.parse_args()
    stats = {}
    for name in args.datasets:
        stats[name] = dataset_stats(name, args.sample)
        print(name, json.dumps(stats[name], ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sample_questions": args.sample, "seed": 42, "datasets": stats},
                                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
