"""So path do planner LLM của MoR sinh (cache công bố) với đường BFS mà HOP dùng làm nhãn, trên RiTeK-Prime.

python HOP/hop_polo_v8/tools/compare_mor_paths.py [--splits train val] [--out HOP/hop_polo_v8/docs/mor_vs_bfs.json]
(chạy từ gốc repo; chỉ train/val của RiTeK, đều nằm trong train STaRK nên có trong cache train của MoR; không dùng test)

Nguồn MoR: external/MoR/cache/prime/path/train_20250222.pkl — mỗi câu có `rg.Metapath` (chuỗi
[loại, quan hệ, loại, ...]) và `rg.Restriction` ({loại: [tên thực thể]}), mở bằng unpickler chỉ cho kiểu dữ liệu cơ bản.

Đo theo từng đáp án (độ sâu BFS = đường ngắn nhất từ topic RiTeK, <= 3 hop như HOP):
  - metapath hợp lệ: mọi loại/quan hệ có trong Prime và mỗi hop là một schema có thật;
  - neo: thực thể Restriction khớp tên (không phân biệt hoa thường) trong đúng loại; topic RiTeK có nằm trong neo không;
  - MoR từ topic: metapath bắt đầu ở loại của topic, thực thi từ topic theo (loại, quan hệ, loại) -> có tới đáp án không,
    dài bao nhiêu, chuỗi loại có trùng một đường BFS ngắn nhất không;
  - MoR từ mọi neo: như trên nhưng xuất phát từ các thực thể Restriction của chính metapath;
  - cỡ tập nút mà một metapath (từ topic) chạm tới, so với đường BFS rẻ nhất ở cùng mức loại (độ chọn lọc).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOP = Path(__file__).resolve().parents[1]
ROOT = HOP.parents[1]
sys.path.insert(0, str(HOP / "src"))

from hop import labels  # noqa: E402
from hop.data import Graph  # noqa: E402
from hop.program import EdgeIndex  # noqa: E402

MOR_CACHE = ROOT / "external/MoR/cache/prime/path/train_20250222.pkl"
MAX_FRONTIER = 200_000


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in ("set", "frozenset", "tuple", "list", "dict"):
            return getattr(__import__("builtins"), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


class TypePaths:
    """Thực thi chuỗi (loại, quan hệ, loại) trên đồ thị HOP (mọi chiều có cùng tên quan hệ)."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.index = EdgeIndex(graph)
        self.type_id = {t: i for i, t in enumerate(graph.type_names)}
        self.rel_id = {r: i for i, r in enumerate(graph.relation_names)}
        self.schemas = defaultdict(list)
        for sid, (src, rel, _direction, dst) in enumerate(graph.schemas):
            self.schemas[(src, rel, dst)].append(sid)
        self.by_name = defaultdict(list)
        for node, (name, t) in enumerate(zip(graph.names, graph.types.tolist())):
            self.by_name[(t, name.strip().casefold())].append(node)

    def parse(self, metapath) -> tuple[tuple[int, int, int], ...] | None:
        if not isinstance(metapath, list) or len(metapath) < 3 or len(metapath) % 2 == 0:
            return None
        types, rels = metapath[0::2], metapath[1::2]
        if any(t not in self.type_id for t in types) or any(r not in self.rel_id for r in rels):
            return None
        hops = tuple((self.type_id[types[i]], self.rel_id[rels[i]], self.type_id[types[i + 1]]) for i in range(len(rels)))
        return hops if all(h in self.schemas for h in hops) else None

    def execute(self, start, hops) -> set[int]:
        frontier = tuple(sorted(set(start)))
        for hop in hops:
            edges = np.concatenate([self.index.edges(frontier, sid) for sid in self.schemas[hop]] or [np.empty(0)])
            frontier = tuple(np.unique(self.graph.dst[edges.astype(np.int64)]).tolist()) if len(edges) else ()
            if not frontier or len(frontier) > MAX_FRONTIER:
                return set(frontier)
        return set(frontier)

    def anchors(self, restriction: dict, type_index: int) -> list[int]:
        names = restriction.get(self.graph.type_names[type_index]) or []
        if isinstance(names, str):
            names = [names]
        return sorted({n for name in names if isinstance(name, str)
                       for n in self.by_name.get((type_index, name.strip().casefold()), ())})

    def edge_hop(self, edge: int) -> tuple[int, int, int]:
        src, rel, _direction, dst = self.graph.schemas[int(self.graph.schema[edge])]
        return (src, rel, dst)


def pct(part, whole) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--out", type=Path, default=HOP / "docs" / "mor_vs_bfs.json")
    args = parser.parse_args()
    started = time.perf_counter()
    graph = Graph(HOP / "inputs" / "prime", verify=False)
    tp = TypePaths(graph)
    mor = {str(r["q_id"]): r for r in PlainUnpickler(MOR_CACHE.open("rb")).load()}
    questions = [q for split in args.splits for q in graph.splits[split]]
    if any(q.qid not in mor for q in questions):
        raise SystemExit("có câu RiTeK không nằm trong cache MoR")

    qstat, astat = Counter(), defaultdict(Counter)
    size_mor, size_bfs = [], []
    for q in questions:
        rg = mor[q.qid].get("rg") or {}
        restriction = rg.get("Restriction") or {}
        raw = rg.get("Metapath") or []
        parsed = [tp.parse(m) for m in raw]
        valid = [h for h in parsed if h]
        topic = q.topics[0]
        topic_type = int(graph.types[topic])
        qstat["questions"] += 1
        qstat["metapaths"] += len(raw)
        qstat["metapaths_valid"] += len(valid)
        qstat["questions_with_valid_metapath"] += int(bool(valid))
        anchor_sets = [tp.anchors(restriction, h[0][0]) for h in valid]
        qstat["topic_named_in_restriction"] += int(topic in tp.anchors(restriction, topic_type))
        from_topic = [h for h in valid if h[0][0] == topic_type]
        qstat["questions_with_metapath_from_topic_type"] += int(bool(from_topic))
        reach_topic = [(h, tp.execute([topic], h)) for h in from_topic]
        reach_any = [(h, tp.execute(a, h)) for h, a in zip(valid, anchor_sets) if a]
        for h, reached in reach_topic:
            size_mor.append(len(reached))
        for answer in q.answers:
            buckets, shortest, _, _ = labels.answer_paths(graph, q, answer, max_hops=3)
            key = str(shortest)
            a = astat[key]
            a["answers"] += 1
            bfs_seqs = {tuple(tp.edge_hop(e) for e in path) for _pos, path in buckets[shortest]} if shortest else set()
            a["bfs_ambiguous"] += int(len(bfs_seqs) > 1)
            hits_topic = [h for h, reached in reach_topic if answer in reached]
            hits_any = [h for h, reached in reach_any if answer in reached]
            a["mor_from_topic_reaches"] += int(bool(hits_topic))
            a["mor_any_anchor_reaches"] += int(bool(hits_any))
            if hits_any and not hits_topic:
                a["reached_only_from_other_anchor"] += 1
                a["reached_only_from_other_anchor_1hop"] += int(min(len(h) for h in hits_any) == 1)
            if shortest:
                a["mor_from_topic_same_seq_as_bfs"] += int(any(h in bfs_seqs for h in from_topic))
                if hits_topic:
                    best = min(len(h) for h in hits_topic)
                    a["mor_reach_len_eq_bfs"] += int(best == shortest)
                    a["mor_reach_len_gt_bfs"] += int(best > shortest)
                    if len(bfs_seqs) > 1:
                        a["ambiguous_mor_picks_a_bfs_seq"] += int(any(h in bfs_seqs for h in hits_topic))
                        a["ambiguous_reached"] += 1
                if from_topic:
                    lengths = {len(h) for h in from_topic}
                    a["mor_topic_paths_all_shorter_than_bfs"] += int(max(lengths) < shortest)
                if bfs_seqs and shortest <= 2:
                    _pos, path = buckets[shortest][0]
                    size_bfs.append(len(tp.execute([topic], tuple(tp.edge_hop(e) for e in path))))
    by_depth = {}
    for depth, a in sorted(astat.items()):
        n = a["answers"]
        by_depth[depth] = {
            "answers": n,
            "bfs_ambiguous_%": pct(a["bfs_ambiguous"], n),
            "mor_from_topic_reaches_%": pct(a["mor_from_topic_reaches"], n),
            "mor_any_anchor_reaches_%": pct(a["mor_any_anchor_reaches"], n),
            "reached_only_from_other_anchor_%": pct(a["reached_only_from_other_anchor"], n),
            "  of_which_1hop_%": pct(a["reached_only_from_other_anchor_1hop"], a["reached_only_from_other_anchor"]),
            "mor_from_topic_same_seq_as_bfs_%": pct(a["mor_from_topic_same_seq_as_bfs"], n),
            "when_reached_len_eq_bfs_%": pct(a["mor_reach_len_eq_bfs"], a["mor_from_topic_reaches"]),
            "when_reached_len_gt_bfs_%": pct(a["mor_reach_len_gt_bfs"], a["mor_from_topic_reaches"]),
            "mor_topic_paths_all_shorter_than_bfs_%": pct(a["mor_topic_paths_all_shorter_than_bfs"], n),
            "ambiguous_and_reached_mor_picks_bfs_seq_%": pct(a["ambiguous_mor_picks_a_bfs_seq"], a["ambiguous_reached"]),
        }
    q = qstat
    report = {
        "splits": args.splits, "questions": q["questions"],
        "metapaths_per_question": round(q["metapaths"] / q["questions"], 2),
        "metapaths_valid_%": pct(q["metapaths_valid"], q["metapaths"]),
        "questions_with_valid_metapath_%": pct(q["questions_with_valid_metapath"], q["questions"]),
        "questions_with_metapath_from_topic_type_%": pct(q["questions_with_metapath_from_topic_type"], q["questions"]),
        "topic_named_in_restriction_%": pct(q["topic_named_in_restriction"], q["questions"]),
        "by_bfs_depth": by_depth,
        "reached_set_size_from_topic": {
            "mor_metapath_p50_p90": [float(np.percentile(size_mor, 50)), float(np.percentile(size_mor, 90))] if size_mor else None,
            "bfs_cheapest_typelevel_p50_p90": [float(np.percentile(size_bfs, 50)), float(np.percentile(size_bfs, 90))] if size_bfs else None},
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
