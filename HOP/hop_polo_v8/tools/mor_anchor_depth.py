"""RiTeK-Prime: độ sâu đáp án tính từ topic RiTeK so với từ thực thể Restriction của planner MoR (khớp tên chính xác).

python HOP/hop_polo_v8/tools/mor_anchor_depth.py [--splits train val]      (chạy từ gốc repo; không dùng test)

Neo MoR = mọi nút có tên trùng (không phân biệt hoa thường) với một tên trong rg.Restriction, đúng loại đã ghi.
Độ sâu = BFS không giới hạn trên đồ thị V8 (Prime lưu sẵn hai chiều). Tách theo "tên topic có trong câu hỏi không".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

HOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOP / "src"))
sys.path.insert(0, str(HOP / "tools"))
from compare_mor_paths import MOR_CACHE, PlainUnpickler, TypePaths  # noqa: E402
from hop.data import Graph  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def bucket(d) -> str:
    return "inf" if np.isinf(d) else ">3" if d > 3 else str(int(d))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    graph = Graph(HOP / "inputs" / "prime", verify=False)
    tp = TypePaths(graph)
    mor = {str(r["q_id"]): r for r in PlainUnpickler(MOR_CACHE.open("rb")).load()}
    questions = [q for split in args.splits for q in graph.splits[split]]
    anchors = {}
    for q in questions:
        restriction = (mor[q.qid].get("rg") or {}).get("Restriction") or {}
        nodes = set()
        for type_name in restriction:
            if type_name in tp.type_id:
                nodes.update(tp.anchors(restriction, tp.type_id[type_name]))
        anchors[q.qid] = sorted(nodes)
    n = len(graph.ids)
    adjacency = csr_matrix((np.ones(len(graph.dst), np.int8), graph.dst, graph.indptr), shape=(n, n))
    sources = sorted({t for q in questions for t in q.topics} | {a for v in anchors.values() for a in v})
    needed = defaultdict(set)
    for q in questions:
        for s in list(q.topics) + anchors[q.qid]:
            needed[s].update(q.answers)
    dist: dict[tuple[int, int], float] = {}
    for start in range(0, len(sources), 256):
        block = sources[start:start + 256]
        rows = shortest_path(adjacency, directed=True, unweighted=True, indices=block)
        for s, row in zip(block, rows):
            for a in needed[s]:
                dist[(s, a)] = row[a]
    stat = defaultdict(Counter)
    for q in questions:
        named = any(len(graph.names[t]) >= 3 and graph.names[t].casefold() in q.question.casefold() for t in q.topics)
        group = "topic_named" if named else "topic_not_named"
        s = stat[group]
        s["questions"] += 1
        s["questions_with_mor_anchor"] += int(bool(anchors[q.qid]))
        s["topic_among_mor_anchors"] += int(any(t in anchors[q.qid] for t in q.topics))
        for a in q.answers:
            d_topic = min(dist[(t, a)] for t in q.topics)
            d_anchor = min((dist[(x, a)] for x in anchors[q.qid]), default=np.inf)
            s["answers"] += 1
            s[f"topic_{bucket(d_topic)}"] += 1
            s[f"anchor_{bucket(d_anchor)}"] += 1
            s[f"best_{bucket(min(d_topic, d_anchor))}"] += 1
            if d_topic >= 2:
                s["deep_from_topic"] += 1
                s["deep_from_topic_but_anchor_le1"] += int(d_anchor <= 1)
    out = {}
    for group, s in stat.items():
        a = s["answers"]
        dist_of = lambda prefix: {k: round(100 * s[f"{prefix}_{k}"] / a, 1)  # noqa: E731
                                  for k in ("0", "1", "2", "3", ">3", "inf") if s[f"{prefix}_{k}"]}
        out[group] = {"questions": s["questions"], "answers": a,
                      "questions_with_mor_anchor_%": round(100 * s["questions_with_mor_anchor"] / s["questions"], 1),
                      "topic_among_mor_anchors_%": round(100 * s["topic_among_mor_anchors"] / s["questions"], 1),
                      "depth_from_topic_%": dist_of("topic"), "depth_from_mor_anchor_%": dist_of("anchor"),
                      "depth_from_topic_or_anchor_%": dist_of("best"),
                      "deep_from_topic_but_<=1_from_anchor_%": round(
                          100 * s["deep_from_topic_but_anchor_le1"] / max(1, s["deep_from_topic"]), 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
