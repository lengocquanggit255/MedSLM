"""Chương trình HOP: ngữ pháp token, executor trên đồ thị, giải mã có ràng buộc, nhãn chương trình.

Một hop = (action = schema cạnh, tiền tố mã S của nút đích hoặc S_ANY). V8 chỉ còn mode
`semantic` của V7 (không mã G). Các phép toán trên frontier (đếm action, lấy cạnh, liệt kê tiền tố S)
được vector hoá theo nhóm (nguồn, action) thay vì lặp Python từng nút như V7: frontier sau một hop
trên PrimeKG có thể tới hàng nghìn nút.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from .codebook import Codebook
from .data import Graph, Question


@dataclass(frozen=True)
class Hop:
    action: int                         # schema ID; n_actions = self-loop / dừng
    s: tuple[int, ...] | None = None    # None = S_ANY


@dataclass(frozen=True)
class Program:
    hops: tuple[Hop, ...]
    # Vị trí trong danh sách topic của câu hỏi; None = ngữ pháp HOP (kể cả STOP rỗng = trả mọi topic).
    return_topics: tuple[int, ...] | None = None


def program_to_dict(program: Program) -> dict:
    return {"hops": [{"action": h.action, "s": list(h.s) if h.s is not None else None} for h in program.hops],
            "return_topics": program.return_topics}


def program_from_dict(row: dict) -> Program:
    return Program(tuple(Hop(int(h["action"]), tuple(h["s"]) if h["s"] is not None else None)
                         for h in row["hops"]),
                   tuple(row["return_topics"]) if row["return_topics"] is not None else None)


@dataclass
class Execution:
    candidates: tuple[int, ...]
    evidence: dict[int, tuple[int, ...]]
    edges_examined: int
    executable: bool
    budget_exceeded: bool = False


def concat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Nối các đoạn [start, start + count) theo thứ tự, không lặp Python."""
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    shift = np.asarray(starts, np.int64) - np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.repeat(shift, counts) + np.arange(total)


class EdgeIndex:
    """(nguồn, action) -> ID cạnh thật, sắp theo khoá (nguồn·A + action), ổn định trong nhóm."""

    def __init__(self, graph: Graph):
        self.n_actions = len(graph.schemas)
        packed = graph.src.astype(np.int64) * self.n_actions + graph.schema
        self.order = np.argsort(packed, kind="stable").astype(np.int32)
        sorted_keys = packed[self.order]
        self.group_keys, self.group_start, self.group_count = np.unique(
            sorted_keys, return_index=True, return_counts=True)
        self.group_action = (self.group_keys % self.n_actions).astype(np.int64)
        self.group_indptr = np.searchsorted(self.group_keys // self.n_actions,
                                            np.arange(len(graph.ids) + 1))

    def groups_of(self, frontier: tuple[int, ...]) -> np.ndarray:
        nodes = np.asarray(frontier, np.int64)
        lo = self.group_indptr[nodes]
        return concat_ranges(lo, self.group_indptr[nodes + 1] - lo)

    def edges(self, frontier: tuple[int, ...], action: int) -> np.ndarray:
        if not frontier:
            return np.empty(0, np.int32)
        keys = np.asarray(frontier, np.int64) * self.n_actions + action
        pos = np.minimum(np.searchsorted(self.group_keys, keys), len(self.group_keys) - 1)
        hit = pos[self.group_keys[pos] == keys]
        return self.order[concat_ranges(self.group_start[hit], self.group_count[hit])]


class PrefixSIndex:
    """(nguồn, action, tiền tố S của nút đích) có tồn tại không — chỉ dùng cho mặt nạ giải mã.

    Chỉ mục chỉ thay việc liệt kê tiền tố; thực thi một hop vẫn lấy và tính chi phí cạnh thật.
    """

    def __init__(self, graph: Graph, codebook: Codebook, index: EdgeIndex):
        self.radix = codebook.size
        self.levels = codebook.levels
        self.n_actions = index.n_actions
        group_keys = (graph.src.astype(np.int64) * self.n_actions + graph.schema)[index.order]
        destination = codebook.node_codes[graph.dst[index.order]].astype(np.int64)
        self.keys = []
        encoded = np.zeros(len(group_keys), np.int64)
        for level in range(self.levels):
            encoded = encoded * self.radix + destination[:, level]
            scale = self.radix ** (level + 1)
            if len(group_keys) and int(group_keys.max()) > (np.iinfo(np.int64).max - scale) // scale:
                raise OverflowError("prefix S index key exceeds int64")
            self.keys.append(np.unique(group_keys * scale + encoded))

    def prefixes(self, frontier: tuple[int, ...], action: int) -> set[tuple[int, ...]]:
        groups = np.asarray(frontier, np.int64) * self.n_actions + action
        result: set[tuple[int, ...]] = set()
        for level, keys in enumerate(self.keys, 1):
            scale = self.radix ** level
            lo = np.searchsorted(keys, groups * scale, side="left")
            hi = np.searchsorted(keys, (groups + 1) * scale, side="left")
            values = np.unique(keys[concat_ranges(lo, hi - lo)] % scale)
            digits = np.empty((len(values), level), np.int64)
            for position in range(level - 1, -1, -1):
                digits[:, position] = values % self.radix
                values = values // self.radix
            result.update(map(tuple, digits.tolist()))
        return result


class Executor:
    def __init__(self, graph: Graph, codebook: Codebook, edge_budget: int, prefix_index: bool = True,
                 index: EdgeIndex | None = None):
        self.graph, self.codebook, self.edge_budget = graph, codebook, edge_budget
        self.index = index or EdgeIndex(graph)
        self.prefix_s_index = PrefixSIndex(graph, codebook, self.index) if prefix_index else None
        self.query_edges_examined = 0
        self._query_budget: int | None = None
        self._query_edges: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
        self._fresh_caches()

    def _fresh_caches(self) -> None:
        # Cache theo từng instance (V7 dùng lru_cache cấp lớp, xoá cache của mọi executor cùng lúc).
        self.action_counts = lru_cache(maxsize=512)(self._action_counts)
        self.feasible_actions = lru_cache(maxsize=512)(self._feasible_actions)
        self.available_prefixes = lru_cache(maxsize=256)(self._available_prefixes)
        self.execute = lru_cache(maxsize=256)(self._execute)

    def begin_query(self, edge_budget: int | None = None) -> None:
        """Bắt đầu một câu truy vấn có đo đếm: đếm số cạnh lấy ra khi tính mặt nạ và khi thực thi."""
        self.query_edges_examined = 0
        self._query_budget = self.edge_budget if edge_budget is None else edge_budget
        self._query_edges.clear()
        self._fresh_caches()

    @property
    def query_remaining(self) -> int:
        if self._query_budget is None:
            return self.edge_budget
        return max(0, self._query_budget - self.query_edges_examined)

    def hop_for_edge(self, edge: int, prefix: tuple[int, ...] | None) -> Hop:
        return Hop(int(self.graph.schema[edge]), prefix)

    def selected_edges(self, frontier: tuple[int, ...], action: int) -> np.ndarray:
        key = (frontier, action)
        if self._query_budget is not None and key in self._query_edges:
            return self._query_edges[key]
        edges = self.index.edges(frontier, action)
        if self._query_budget is not None:
            edges = edges[:self.query_remaining]
            self.query_edges_examined += len(edges)
            self._query_edges[key] = edges
        return edges

    def _action_counts(self, frontier: tuple[int, ...]) -> dict[int, int]:
        if not frontier:
            return {}
        groups = self.index.groups_of(frontier)
        counts = np.bincount(self.index.group_action[groups], weights=self.index.group_count[groups],
                             minlength=self.index.n_actions)
        nonzero = np.flatnonzero(counts)
        return dict(zip(nonzero.tolist(), counts[nonzero].astype(np.int64).tolist()))

    def _feasible_actions(self, frontier: tuple[int, ...], remaining_budget: int) -> tuple[int, ...]:
        return tuple(sorted(a for a, c in self.action_counts(frontier).items() if 0 < c <= remaining_budget))

    def _available_prefixes(self, frontier: tuple[int, ...], action: int) -> set[tuple[int, ...]]:
        if self.prefix_s_index is not None:
            return self.prefix_s_index.prefixes(frontier, action)
        result = set()
        for dest in np.unique(self.graph.dst[self.selected_edges(frontier, action)]):
            row = self.codebook.node_codes[dest]
            result.update(tuple(int(x) for x in row[:length]) for length in range(1, len(row) + 1))
        return result

    def _execute(self, topics: tuple[int, ...], program: Program) -> Execution:
        if program.return_topics is not None:
            if program.hops or not program.return_topics or any(
                    index < 0 or index >= len(topics) for index in program.return_topics):
                return Execution((), {}, 0, False)
            selected = tuple(sorted({topics[index] for index in program.return_topics}))
            return Execution(selected, {node: () for node in selected}, 0, True)
        frontier = tuple(sorted(set(topics)))
        paths: dict[int, tuple[int, ...]] = {node: () for node in frontier}
        scanned = 0
        for hop in program.hops:
            edges = self.selected_edges(frontier, hop.action)
            if scanned + len(edges) > self.edge_budget:
                return Execution((), {}, scanned, False, True)
            scanned += len(edges)
            if hop.s is not None:
                allowed = self.codebook.prefix_nodes(hop.s)
                if not len(allowed):
                    return Execution((), {}, scanned, False)
                dest = self.graph.dst[edges]
                positions = np.searchsorted(allowed, dest)
                match = (positions < len(allowed)) & (allowed[np.minimum(positions, len(allowed) - 1)] == dest)
                edges = edges[match]
            if not len(edges):
                return Execution((), {}, scanned, False)
            # Cạnh đầu tiên tới mỗi nút đích (theo thứ tự cạnh) làm bằng chứng, như V7.
            destinations, first = np.unique(self.graph.dst[edges], return_index=True)
            sources = self.graph.src[edges[first]]
            paths = {int(d): paths[int(s)] + (int(e),)
                     for d, s, e in zip(destinations.tolist(), sources.tolist(), edges[first].tolist())}
            frontier = tuple(sorted(paths))
        return Execution(frontier, paths, scanned, True)


class ProgramCodec:
    """Token chương trình (thêm vào tokenizer Flan) và mặt nạ token hợp lệ theo đồ thị."""

    def __init__(self, tokenizer, graph: Graph, executor: Executor, max_hops: int):
        self.tokenizer, self.graph, self.executor, self.max_hops = tokenizer, graph, executor, max_hops
        self.levels = executor.codebook.levels
        self.max_topics = max((len(q.topics) for split in graph.splits.values() for q in split), default=1)
        self.tokens = ["<HOP>", "<ENDHOP>", "<STOP>", "<RETURN_TOPIC>", "<END_RETURN>",
                       "<S_ANY>", "<READ_S>", "<READ_G>"]
        self.tokens += [f"<T_{i}>" for i in range(self.max_topics)]
        self.tokens += [f"<A_{i}>" for i in range(len(graph.schemas))]
        self.tokens += [f"<S{level}_{i}>" for level in range(1, self.levels + 1)
                        for i in range(executor.codebook.size)]
        tokenizer.add_special_tokens({"additional_special_tokens": self.tokens})
        self.id = {token: tokenizer.convert_tokens_to_ids(token) for token in self.tokens}
        self.name = {v: k for k, v in self.id.items()}
        if len(self.name) != len(self.id):
            raise ValueError("program tokens must each have a unique token ID")

    def tokens_for(self, program: Program) -> list[str]:
        if program.return_topics is not None:
            return ["<RETURN_TOPIC>", *(f"<T_{i}>" for i in program.return_topics), "<END_RETURN>", "<STOP>"]
        result = []
        for hop in program.hops:
            result += ["<HOP>", f"<A_{hop.action}>"]
            result += ["<S_ANY>"] if hop.s is None else [f"<S{i + 1}_{code}>" for i, code in enumerate(hop.s)]
            result += ["<ENDHOP>"]
        return result + ["<STOP>"]

    def ids_for(self, program: Program) -> list[int]:
        return [self.id[token] for token in self.tokens_for(program)] + [self.tokenizer.eos_token_id]

    def parse(self, ids: list[int]) -> tuple[Program, str, int | None, tuple[int, ...] | None]:
        """Trả các hop đã hoàn chỉnh và trạng thái ngữ pháp của hop đang dở."""
        hops: list[Hop] = []
        returned: list[int] = []
        state, action, prefix = "start", None, None

        def stop(final_state):
            return Program(tuple(hops), tuple(returned) if returned else None), final_state, action, prefix

        for token_id in ids:
            token = self.name.get(int(token_id))
            if token is None:
                continue  # decoder start / pad
            if state == "start" and token == "<HOP>":
                state = "action"
            elif state == "start" and not hops and token == "<RETURN_TOPIC>":
                state = "return"
            elif state == "return" and token.startswith("<T_"):
                index = int(token[3:-1])
                if index in returned:
                    return Program(tuple(hops)), "invalid", action, prefix
                returned.append(index)
            elif state == "return" and returned and token == "<END_RETURN>":
                state = "return_end"
            elif state in ("return_end", "start") and token == "<STOP>":
                state = "done"
            elif state == "action" and token.startswith("<A_"):
                action, state = int(token[3:-1]), "s"
            elif state == "s" and token == "<S_ANY>":
                prefix, state = None, "end"
            elif state == "s" and token.startswith("<S1_"):
                prefix, state = (int(token[4:-1]),), "s1"
            elif state in ("s1", "s2") and token.startswith(f"<S{len(prefix) + 1}_"):
                prefix = prefix + (int(token[4:-1]),)
                state = f"s{len(prefix)}"
            elif state in ("end", "s1", "s2", "s3") and token == "<ENDHOP>":
                hops.append(Hop(action, prefix))
                state, action, prefix = "start", None, None
            else:
                return stop("invalid")
        return stop(state)

    def allowed(self, ids: list[int], topics: tuple[int, ...]) -> list[int]:
        program, state, action, prefix = self.parse(ids)
        if state == "done":
            return [self.tokenizer.eos_token_id]
        if state == "invalid":
            return [self.id["<STOP>"]]
        if state == "return":
            remaining = [self.id[f"<T_{i}>"] for i in range(len(topics)) if i not in (program.return_topics or ())]
            return remaining + ([self.id["<END_RETURN>"]] if program.return_topics else [])
        if state == "return_end":
            return [self.id["<STOP>"]]
        executor = self.executor
        current = executor.execute(topics, program)
        if not current.executable or current.budget_exceeded:
            return [self.id["<STOP>"]]
        frontier = current.candidates
        feasible = executor.feasible_actions(
            frontier, min(executor.edge_budget - current.edges_examined, executor.query_remaining))
        if state == "start":
            options = [self.id["<STOP>"]]
            if not program.hops and topics:
                options.append(self.id["<RETURN_TOPIC>"])
            if len(program.hops) < self.max_hops and feasible:
                options.append(self.id["<HOP>"])
            return options
        if state == "action":
            return [self.id[f"<A_{a}>"] for a in feasible] or [self.id["<STOP>"]]
        if state in ("s", "s1", "s2", "s3", "end"):
            count = executor.action_counts(frontier).get(action, 0)
            if (current.edges_examined + count > executor.edge_budget or
                    count > executor.query_remaining or not count):
                return [self.id["<S_ANY>"]]
            prefixes = executor.available_prefixes(frontier, action)
            if state == "s":
                return [self.id["<S_ANY>"]] + [self.id[f"<S1_{p[0]}>"] for p in sorted(prefixes) if len(p) == 1]
            if state == "end":
                return [self.id["<ENDHOP>"]]
            choices = [self.id["<ENDHOP>"]]
            if len(prefix) < self.levels:
                choices += [self.id[f"<S{len(p)}_{p[-1]}>"] for p in sorted(prefixes)
                            if len(p) == len(prefix) + 1 and p[:len(prefix)] == prefix]
            return choices
        return [self.id["<STOP>"]]

    def decoded(self, ids: list[int]) -> Program | None:
        program, state, _, _ = self.parse(ids)
        return program if state == "done" else None


def input_text(graph: Graph, question: Question) -> str:
    names = "; ".join(f"topic {index}: {graph.names[node]}" for index, node in enumerate(question.topics))
    return f"question: {question.question} {names}"


def make_synthetic_examples(graph: Graph, codebook: Codebook, codec: ProgramCodec,
                            n_nodes: int, n_edges: int, seed: int) -> list[tuple[str, list[int]]]:
    """Ví dụ chỉ sinh từ KG: đọc mã S của nút (READ_S) và đọc schema của một chuyển cạnh (READ_G)."""
    import random
    rng = np.random.default_rng(seed)
    rows = []
    for node in rng.choice(len(graph.ids), min(n_nodes, len(graph.ids)), replace=False):
        tokens = ["<READ_S>"] + [f"<S{i + 1}_{int(v)}>" for i, v in enumerate(codebook.node_codes[node])] + ["<STOP>"]
        rows.append((f"encode node: {graph.texts[node]}",
                     [codec.id[x] for x in tokens] + [codec.tokenizer.eos_token_id]))
    # Mỗi action giải mã được phải xuất hiện ít nhất một lần.
    _, representatives = np.unique(graph.schema, return_index=True)
    extra_count = max(0, min(n_edges, len(graph.src)) - len(representatives))
    extras = rng.choice(len(graph.src), extra_count, replace=False) if extra_count else np.empty(0, np.int64)
    neighborhoods: dict[int, str] = {}

    def neighborhood(node: int) -> str:
        if node not in neighborhoods:
            edges = graph.out_edges(node)
            labels = graph.rel[edges].astype(np.int32) * 2 + graph.direction[edges]
            values, counts = np.unique(labels, return_counts=True)
            leading = sorted(zip(values, counts), key=lambda pair: (-pair[1], pair[0]))[:8]
            neighborhoods[node] = ", ".join(
                f"{graph.relation_names[int(label) // 2]} {'inverse' if int(label) % 2 else 'forward'} {int(count)}"
                for label, count in leading)
        return neighborhoods[node]

    for edge in np.concatenate((representatives, extras)):
        source, dest = int(graph.src[edge]), int(graph.dst[edge])
        schema = graph.schemas[int(graph.schema[edge])]
        relation = graph.relation_names[schema[1]] + (" inverse" if schema[2] else " forward")
        text = (f"encode transition: {graph.type_names[schema[0]]} {relation} {graph.type_names[schema[3]]}; "
                f"source {graph.names[source]}; destination {graph.names[dest]}; "
                f"source neighborhood {neighborhood(source)}; destination neighborhood {neighborhood(dest)}")
        tokens = ["<READ_G>", f"<A_{int(graph.schema[edge])}>", "<STOP>"]
        rows.append((text, [codec.id[x] for x in tokens] + [codec.tokenizer.eos_token_id]))
    random.Random(seed).shuffle(rows)
    return rows


@dataclass
class EvidenceProgram:
    program: Program
    answer: int
    candidate_count: int
    edges_examined: int


def evidence_programs(graph: Graph, codebook: Codebook, executor: Executor, question: Question,
                      paths: list[tuple[int, ...]], max_programs: int) -> list[EvidenceProgram]:
    """Mọi chương trình (mọi lựa chọn tiền tố S) tái tạo một đường vàng, xếp theo chi phí thực thi.

    Giữ đúng biến thể V7 mà nhãn QA dùng: tiền tố trung gian "all", xếp theo chi phí.
    """
    possible: list[EvidenceProgram] = []
    gold = set(question.answers)
    for path in paths:
        if not path:
            gold_positions = tuple(i for i, topic in enumerate(question.topics) if topic in gold)
            if gold_positions:
                if len(question.topics) == 1:
                    possible.append(EvidenceProgram(Program(()), question.topics[0], 1, 0))
                possible.append(EvidenceProgram(Program((), gold_positions), question.topics[gold_positions[0]],
                                                len({question.topics[i] for i in gold_positions}), 0))
                if len(gold_positions) == len(question.topics) and len(question.topics) > 1:
                    possible.append(EvidenceProgram(Program(()), question.topics[0], len(set(question.topics)), 0))
            continue
        destinations = [int(graph.dst[e]) for e in path]
        answer = destinations[-1]
        choices = []
        for step, dest in enumerate(destinations):
            row = codebook.node_codes[dest]
            prefixes = [tuple(int(x) for x in row[:i]) for i in range(1, len(row) + 1)]
            choices.append([None, *prefixes] if step < len(path) - 1 else prefixes)
        for prefix_choice in product(*choices):
            program = Program(tuple(executor.hop_for_edge(edge, prefix) for edge, prefix in zip(path, prefix_choice)))
            result = executor.execute(question.topics, program)
            if answer in result.candidates:
                possible.append(EvidenceProgram(program, answer, len(result.candidates), result.edges_examined))
    unique: dict[Program, EvidenceProgram] = {}
    for item in sorted(possible, key=lambda x: (x.edges_examined, x.candidate_count, len(x.program.hops))):
        unique.setdefault(item.program, item)
    ranked = list(unique.values())
    representatives: dict[int, EvidenceProgram] = {}
    for item in ranked:
        representatives.setdefault(item.answer, item)
    chosen = list(representatives.values())[:max_programs]
    # Đáp án lẫn 0-hop và có cạnh: giữ chỗ cho cả hai loại trước khi điền theo thứ hạng.
    if max_programs >= 2 and any(not item.program.hops for item in ranked) and any(item.program.hops for item in ranked):
        zero_item = next(item for item in ranked if not item.program.hops)
        edge_item = next(item for item in ranked if item.program.hops)
        chosen = ([zero_item, edge_item] + [item for item in chosen if item not in (zero_item, edge_item)])[:max_programs]
    seen_actions = {tuple(hop.action for hop in item.program.hops) for item in chosen}
    for item in ranked:
        if len(chosen) >= max_programs:
            break
        actions = tuple(hop.action for hop in item.program.hops)
        if item not in chosen and actions not in seen_actions:
            chosen.append(item)
            seen_actions.add(actions)
    chosen += [item for item in ranked if item not in chosen][:max(0, max_programs - len(chosen))]
    return chosen
