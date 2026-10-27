"""HOP program semantics, evidence paths, constrained actions, and path mining."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from collections import Counter
import time

import numpy as np

from .codebooks import Codebooks, EdgeIndex
from .data import ADIntGraph, Question


@dataclass(frozen=True)
class Hop:
    action: int  # rel*2+direction for relation mode, schema ID otherwise
    g: int = -1  # -1 means any learned structural code
    s: tuple[int, ...] | None = None  # None means S_ANY


@dataclass(frozen=True)
class Program:
    hops: tuple[Hop, ...]
    # Positions in the question's complete topic list. None denotes the HOP
    # grammar (including a lone STOP, which returns every topic).
    return_topics: tuple[int, ...] | None = None


@dataclass
class Execution:
    candidates: tuple[int, ...]
    evidence: dict[int, tuple[int, ...]]
    edges_examined: int
    executable: bool
    budget_exceeded: bool = False


class PrefixSIndex:
    """Offline (source, action, G, S-prefix) membership for decoder masks.

    The index changes only prefix enumeration. Executing a completed hop still
    fetches and charges the actual edge postings through ``selected_edges``.
    """

    def __init__(self, graph: ADIntGraph, codes: Codebooks, edge_index: EdgeIndex):
        started = time.perf_counter()
        self.radix = int(codes.node_codes.max()) + 1
        self.levels = codes.node_codes.shape[1]
        self.actions = edge_index.actions
        self.code_count = edge_index.code_count
        destination_codes = codes.node_codes[graph.dst[edge_index.order]]
        group_keys = edge_index.keys
        self.keys = []
        encoded = np.zeros(len(group_keys), np.int64)
        for level in range(self.levels):
            encoded = encoded * self.radix + destination_codes[:, level]
            scale = self.radix ** (level + 1)
            if len(group_keys) and int(group_keys[-1]) > (np.iinfo(np.int64).max - scale) // scale:
                raise OverflowError("prefix S index key exceeds int64")
            self.keys.append(np.unique(group_keys * scale + encoded))
        self.build_seconds = time.perf_counter() - started
        self.bytes = sum(values.nbytes for values in self.keys)

    def prefixes(self, frontier: tuple[int, ...], action: int, code: int) -> set[tuple[int, ...]]:
        result: set[tuple[int, ...]] = set()
        for node in frontier:
            group = (node * self.actions + action) * self.code_count + code
            for level, keys in enumerate(self.keys, 1):
                scale = self.radix ** level
                left = int(np.searchsorted(keys, group * scale, side="left"))
                right = int(np.searchsorted(keys, (group + 1) * scale, side="left"))
                for packed in keys[left:right]:
                    value = int(packed % scale)
                    prefix = [0] * level
                    for position in range(level - 1, -1, -1):
                        prefix[position] = value % self.radix
                        value //= self.radix
                    result.add(tuple(prefix))
        return result


class Executor:
    def __init__(self, graph: ADIntGraph, codes: Codebooks, mode: str = "full", edge_budget: int = 50_000,
                 prefix_s_index: bool = False):
        if mode not in ("relation", "schema", "semantic", "full"):
            raise ValueError(mode)
        self.graph, self.codes, self.mode, self.edge_budget = graph, codes, mode, edge_budget
        if mode == "full":
            self.index = codes.edge_index
        else:
            actions = graph.rel.astype(np.int32) * 2 + graph.direction if mode == "relation" else graph.schema
            self.index = EdgeIndex(graph, actions, np.zeros(len(graph.src), np.int8))
        self.prefix_s_index = PrefixSIndex(graph, codes, self.index) if prefix_s_index and mode in ("semantic", "full") else None
        self.query_edges_examined = 0
        self.decoder_edges_examined = 0
        self._query_budget: int | None = None
        self._query_edge_cache: dict[tuple[tuple[int, ...], int, int], np.ndarray] = {}

    def begin_query(self, edge_budget: int | None = None) -> None:
        """Start a measured inference query; count postings fetched by masking and execution."""
        self.query_edges_examined = 0
        self.decoder_edges_examined = 0
        self._query_budget = self.edge_budget if edge_budget is None else edge_budget
        self._query_edge_cache.clear()
        self.action_counts.cache_clear()
        self.available_actions.cache_clear()
        self.feasible_actions.cache_clear()
        self.available_prefixes.cache_clear()
        self.execute.cache_clear()

    @property
    def query_remaining(self) -> int:
        return max(0, self._query_budget - self.query_edges_examined) if self._query_budget is not None else self.edge_budget

    def action_for_edge(self, edge: int) -> int:
        g = self.graph
        return (int(g.rel[edge]) * 2 + int(g.direction[edge])) if self.mode == "relation" else int(g.schema[edge])

    def hop_for_edge(self, edge: int, prefix: tuple[int, ...] | None) -> Hop:
        return Hop(self.action_for_edge(edge), int(self.codes.edge_codes[edge]) if self.mode == "full" else -1,
                   prefix if self.mode not in ("relation", "schema") else None)

    def selected_edges(self, frontier: tuple[int, ...], action: int, code: int) -> np.ndarray:
        key = (frontier, action, code)
        if self._query_budget is not None and key in self._query_edge_cache:
            return self._query_edge_cache[key]
        chunks = []
        for node in frontier:
            if self._query_budget is not None and self.query_remaining == 0:
                break
            candidate = self.index.get((node, action, code if self.mode == "full" else 0))
            if candidate is not None:
                if self._query_budget is not None:
                    candidate = candidate[:self.query_remaining]
                    self.query_edges_examined += len(candidate)
                chunks.append(candidate)
        edges = np.concatenate(chunks) if chunks else np.empty(0, np.int32)
        if self._query_budget is not None:
            self._query_edge_cache[key] = edges
        return edges

    @lru_cache(maxsize=512)
    def action_counts(self, frontier: tuple[int, ...]) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        for node in frontier:
            actions, codes, values = self.index.groups(node)
            for action, code, value in zip(actions, codes, values):
                counts[(int(action), int(code) if self.mode == "full" else -1)] += int(value)
        return counts

    @lru_cache(maxsize=512)
    def available_actions(self, frontier: tuple[int, ...]) -> dict[int, set[int]]:
        options: dict[int, set[int]] = {}
        for action, code in self.action_counts(frontier):
            options.setdefault(action, set()).add(code)
        return options

    @lru_cache(maxsize=512)
    def feasible_actions(self, frontier: tuple[int, ...], remaining_budget: int) -> dict[int, set[int]]:
        options: dict[int, set[int]] = {}
        for (action, code), count in self.action_counts(frontier).items():
            if 0 < count <= remaining_budget:
                options.setdefault(action, set()).add(code)
        return options

    @lru_cache(maxsize=256)
    def available_prefixes(self, frontier: tuple[int, ...], action: int, code: int) -> set[tuple[int, ...]]:
        if self.prefix_s_index is not None:
            return self.prefix_s_index.prefixes(frontier, action, code if self.mode == "full" else 0)
        edges = self.selected_edges(frontier, action, code)
        result = set()
        if self.mode in ("relation", "schema"):
            return result
        for dest in np.unique(self.graph.dst[edges]):
            row = self.codes.node_codes[dest]
            result.update(tuple(int(x) for x in row[:length]) for length in range(1, len(row) + 1))
        return result

    @lru_cache(maxsize=256)
    def execute(self, topics: tuple[int, ...], program: Program) -> Execution:
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
            edges = self.selected_edges(frontier, hop.action, hop.g)
            if scanned + len(edges) > self.edge_budget:
                return Execution((), {}, scanned, False, True)
            scanned += len(edges)
            if hop.s is not None:
                allowed = self.codes.prefix_nodes(hop.s)
                if not len(allowed):
                    return Execution((), {}, scanned, False)
                dest = self.graph.dst[edges]
                positions = np.searchsorted(allowed, dest)
                match = (positions < len(allowed)) & (allowed[np.minimum(positions, len(allowed) - 1)] == dest)
                edges = edges[match]
            next_paths: dict[int, tuple[int, ...]] = {}
            for edge in edges:
                source, dest = int(self.graph.src[edge]), int(self.graph.dst[edge])
                if source in paths and dest not in next_paths:
                    next_paths[dest] = paths[source] + (int(edge),)
            if not next_paths:
                return Execution((), {}, scanned, False)
            paths = next_paths
            frontier = tuple(sorted(paths))
        return Execution(frontier, paths, scanned, True)


@dataclass
class MiningResult:
    paths: list[tuple[int, ...]]
    min_hops: int | None
    edges_scanned: int
    censored: bool


def mine_short_paths(graph: ADIntGraph, topics: tuple[int, ...], answers: tuple[int, ...],
                     max_hops: int = 3, max_paths: int = 8, scan_budget: int = 1_000_000) -> MiningResult:
    """Bidirectional search; never silently labels a censored search unreachable."""
    if len(answers) > 1:
        per_answer = max(1, max_paths // len(answers))
        separate = [mine_short_paths(graph, topics, (answer,), max_hops, per_answer,
                                     scan_budget) for answer in answers]
        paths = sorted((path for result in separate for path in result.paths), key=lambda path: (len(path), path))
        minimum = min((result.min_hops for result in separate if result.min_hops is not None), default=None)
        return MiningResult(paths[:max_paths], minimum,
                            sum(result.edges_scanned for result in separate),
                            any(result.censored for result in separate))
    targets = set(answers)
    zero = [()] if targets.intersection(topics) else []
    if targets.issubset(topics):
        return MiningResult(zero, 0, 0, False)
    scanned = 0
    first = []
    for topic in topics:
        first.extend(int(e) for e in graph.out_edges(topic))
    scanned += len(first)
    if scanned > scan_budget:
        return MiningResult(zero, 0 if zero else None, scanned, True)
    paths = (zero + [(e,) for e in first if int(graph.dst[e]) in targets])[:max_paths]
    minimum = 0 if zero else 1 if paths else None
    if max_hops == 1 or len(paths) >= max_paths:
        return MiningResult(paths, minimum, scanned, False)
    last_by_source: dict[int, list[int]] = {}
    for answer in targets:
        for edge in graph.in_edges(answer):
            last_by_source.setdefault(int(graph.src[edge]), []).append(int(edge))
            scanned += 1
    if scanned > scan_budget:
        return MiningResult(paths, minimum, scanned, True)
    for a in first:
        for b in last_by_source.get(int(graph.dst[a]), ()):
            paths.append((a, b))
            if len(paths) >= max_paths:
                return MiningResult(paths, minimum or 2, scanned, False)
    if len(paths) and minimum is None:
        minimum = 2
    if max_hops == 2:
        return MiningResult(paths, minimum, scanned, False)
    # Two-sided meet: topic --a--> u --b--> v --c--> answer.
    # This finds genuine directed edge paths, including explicitly inverse edges.
    for a in first:
        for b in graph.out_edges(int(graph.dst[a])):
            scanned += 1
            for c in last_by_source.get(int(graph.dst[b]), ()):
                paths.append((a, int(b), c))
                if len(paths) >= max_paths:
                    return MiningResult(paths, minimum or 3, scanned, False)
            if scanned > scan_budget:
                return MiningResult(paths, minimum or (3 if paths else None), scanned, True)
    return MiningResult(paths, minimum or (3 if paths else None), scanned, False)


def mine_cost_aware_paths(graph: ADIntGraph, codes: Codebooks, executor: Executor, question: Question,
                          max_paths: int = 8, scan_budget: int = 1_000_000,
                          intermediate_prefixes: str = "all") -> MiningResult:
    """Rank all one/two-hop gold paths by executable cost before keeping paths.

    This is label mining, not a decoder simulation: it charges the postings of
    the selected program, but cannot predict postings spent on wrong beams.
    """
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    zero = [()] if set(question.topics) & set(question.answers) else []
    if set(question.answers).issubset(question.topics):
        return MiningResult(zero, 0, 0, False)
    first = [int(edge) for topic in question.topics for edge in graph.out_edges(topic)]
    scanned = len(first)
    if scanned > scan_budget:
        return MiningResult(zero, 0 if zero else None, scanned, True)
    targets = set(question.answers)
    direct = [(edge,) for edge in first if int(graph.dst[edge]) in targets]
    last_by_source: dict[int, list[int]] = {}
    for answer in question.answers:
        for edge in graph.in_edges(answer):
            last_by_source.setdefault(int(graph.src[edge]), []).append(int(edge))
            scanned += 1
    if scanned > scan_budget:
        return MiningResult(zero, 0 if zero else 1 if direct else None, scanned, True)

    initial = tuple(sorted(set(question.topics)))
    first_counts = executor.action_counts(initial)
    seen: set[tuple[int, ...]] = set()
    feasible: list[tuple[tuple[int, ...], EvidenceProgram]] = []
    found_two = False
    execution_cache: dict[Program, Execution] = {}

    def consider(path: tuple[int, ...]) -> None:
        if path in seen:
            return
        seen.add(path)
        first_hop = executor.hop_for_edge(path[0], None)
        if first_counts[(first_hop.action, first_hop.g)] > executor.edge_budget:
            return
        programs = evidence_programs(graph, codes, executor, question,
                                     MiningResult([path], len(path), scanned, False),
                                     max_programs=1, intermediate_prefixes=intermediate_prefixes,
                                     rank_by_cost=True, execution_cache=execution_cache)
        if programs:
            feasible.append((path, programs[0]))

    for path in direct:
        consider(path)
    for first_edge in first:
        matching = last_by_source.get(int(graph.dst[first_edge]), ())
        if matching:
            found_two = True
        for last_edge in matching:
            consider((first_edge, last_edge))
    minimum = 0 if zero else 1 if direct else 2 if found_two else None
    feasible.sort(key=lambda pair: (pair[1].edges_examined, pair[1].candidate_count,
                                    len(pair[0]), pair[0]))
    chosen: list[tuple[int, ...]] = list(zero)
    chosen_set: set[tuple[int, ...]] = set(zero)
    answers_seen: set[int] = set()
    actions_seen: set[tuple[int, ...]] = set()
    for priority in ("answer", "action", "cost"):
        for path, item in feasible:
            actions = tuple(hop.action for hop in item.program.hops)
            if path in chosen_set or (priority == "answer" and item.answer in answers_seen) or (
                    priority == "action" and actions in actions_seen):
                continue
            chosen.append(path)
            chosen_set.add(path)
            answers_seen.add(item.answer)
            actions_seen.add(actions)
            if len(chosen) == max_paths:
                return MiningResult(chosen, minimum, scanned, False)
    return MiningResult(chosen, minimum, scanned, False)


@dataclass
class EvidenceProgram:
    program: Program
    answer: int
    candidate_count: int
    edges_examined: int


def evidence_programs(graph: ADIntGraph, codes: Codebooks, executor: Executor, question: Question,
                      mined: MiningResult, max_programs: int = 4,
                      intermediate_prefixes: str = "s1", rank_by_cost: bool = False,
                      execution_cache: dict[Program, Execution] | None = None) -> list[EvidenceProgram]:
    if intermediate_prefixes not in ("s1", "all"):
        raise ValueError(intermediate_prefixes)
    possible: list[EvidenceProgram] = []
    for path in mined.paths:
        if not path:
            gold_positions = tuple(index for index, topic in enumerate(question.topics)
                                   if topic in set(question.answers))
            if gold_positions:
                if len(question.topics) == 1:
                    possible.append(EvidenceProgram(Program(()), question.topics[0], 1, 0))
                program = Program((), gold_positions)
                possible.append(EvidenceProgram(program, question.topics[gold_positions[0]],
                                                len(set(question.topics[index] for index in gold_positions)), 0))
                if len(gold_positions) == len(question.topics) and len(question.topics) > 1:
                    possible.append(EvidenceProgram(Program(()), question.topics[0],
                                                    len(set(question.topics)), 0))
            continue
        destinations = [int(graph.dst[e]) for e in path]
        answer = destinations[-1]
        choices = []
        for step, dest in enumerate(destinations):
            row = codes.node_codes[dest]
            prefixes = [tuple(int(x) for x in row[:i]) for i in range(1, len(row) + 1)]
            if step < len(path) - 1:
                choices.append([None, prefixes[0]] if intermediate_prefixes == "s1" else [None, *prefixes])
            else:
                choices.append(prefixes)
        for prefix_choice in product(*choices):
            program = Program(tuple(executor.hop_for_edge(edge, prefix) for edge, prefix in zip(path, prefix_choice)))
            if execution_cache is None:
                result = executor.execute(question.topics, program)
            else:
                result = execution_cache.get(program)
                if result is None:
                    result = executor.execute(question.topics, program)
                    execution_cache[program] = result
            if answer in result.candidates:
                possible.append(EvidenceProgram(program, answer, len(result.candidates), result.edges_examined))
    unique: dict[Program, EvidenceProgram] = {}
    key = ((lambda x: (x.edges_examined, x.candidate_count, len(x.program.hops))) if rank_by_cost else
           (lambda x: (len(x.program.hops), x.candidate_count, x.edges_examined)))
    for item in sorted(possible, key=key):
        unique.setdefault(item.program, item)
    ranked = list(unique.values())
    representatives: dict[int, EvidenceProgram] = {}
    for item in ranked:
        representatives.setdefault(item.answer, item)
    chosen = list(representatives.values())[:max_programs]
    # Mixed gold answers need both a zero-edge and an edge-bearing target.
    # Reserve these before filling the remaining slots by the usual ranking.
    if max_programs >= 2 and any(not item.program.hops for item in ranked) and any(item.program.hops for item in ranked):
        zero_item = next(item for item in ranked if not item.program.hops)
        edge_item = next(item for item in ranked if item.program.hops)
        chosen = [zero_item, edge_item] + [item for item in chosen if item not in (zero_item, edge_item)]
        chosen = chosen[:max_programs]
    if rank_by_cost:
        seen_actions = {tuple(hop.action for hop in item.program.hops) for item in chosen}
        for item in ranked:
            actions = tuple(hop.action for hop in item.program.hops)
            if len(chosen) >= max_programs:
                break
            if item not in chosen and actions not in seen_actions:
                chosen.append(item)
                seen_actions.add(actions)
    chosen += [item for item in ranked if item not in chosen][:max(0, max_programs - len(chosen))]
    return chosen
