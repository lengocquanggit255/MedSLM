"""One pretrained encoder-decoder trained to emit executable HOP programs."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from .data import ADIntGraph, Question
from .executor import EvidenceProgram, Executor, Hop, Program


@dataclass
class BeamResult:
    program: Program
    probability: float
    candidates: tuple[int, ...]
    evidence: dict[int, tuple[int, ...]]
    edges_examined: int
    executable: bool


class ProgramCodec:
    def __init__(self, tokenizer, graph: ADIntGraph, executor: Executor, max_hops: int = 3,
                 include_g: bool = True, input_style: str = "minimal"):
        self.tokenizer, self.graph, self.executor, self.max_hops = tokenizer, graph, executor, max_hops
        if input_style not in ("minimal", "tagged"):
            raise ValueError(input_style)
        if not include_g and executor.mode == "full":
            raise ValueError("the full mode requires G codes")
        self.include_g, self.input_style = include_g, input_style
        n_actions = len(graph.relation_ids) * 2 if executor.mode == "relation" else len(graph.schemas)
        s_size = int(executor.codes.node_codes.max()) + 1
        g_size = int(executor.codes.edge_codes.max()) + 1
        self.max_topics = max((len(question.topics) for split in getattr(graph, "splits", {}).values()
                               for question in split), default=1)
        self.tokens = ["<HOP>", "<ENDHOP>", "<STOP>", "<RETURN_TOPIC>", "<END_RETURN>"]
        if include_g:
            self.tokens.append("<G_ANY>")
        self.tokens += ["<S_ANY>", "<READ_S>", "<READ_G>"]
        self.tokens += [f"<T_{i}>" for i in range(self.max_topics)]
        self.tokens += [f"<A_{i}>" for i in range(n_actions)]
        if include_g:
            self.tokens += [f"<G_{i}>" for i in range(g_size)]
        self.tokens += [f"<S{level}_{i}>" for level in range(1, executor.codes.node_codes.shape[1] + 1) for i in range(s_size)]
        tokenizer.add_special_tokens({"additional_special_tokens": self.tokens})
        self.id = {token: tokenizer.convert_tokens_to_ids(token) for token in self.tokens}
        self.name = {v: k for k, v in self.id.items()}
        if len(self.name) != len(self.id):
            raise ValueError("program tokens must each have a unique token ID")

    def tokens_for(self, program: Program) -> list[str]:
        if program.return_topics is not None:
            return ["<RETURN_TOPIC>", *(f"<T_{i}>" for i in program.return_topics),
                    "<END_RETURN>", "<STOP>"]
        result = []
        for hop in program.hops:
            result += ["<HOP>", f"<A_{hop.action}>"]
            if self.include_g:
                result += [f"<G_{hop.g}>" if hop.g >= 0 else "<G_ANY>"]
            result += ["<S_ANY>"] if hop.s is None else [f"<S{i + 1}_{code}>" for i, code in enumerate(hop.s)]
            result += ["<ENDHOP>"]
        result.append("<STOP>")
        return result

    def ids_for(self, program: Program) -> list[int]:
        return [self.id[token] for token in self.tokens_for(program)] + [self.tokenizer.eos_token_id]

    def parse(self, ids: list[int]) -> tuple[Program, str, int | None, int | None, tuple[int, ...] | None]:
        """Return completed hops and the grammar state of an unfinished hop."""
        hops = []
        returned = []
        state, action, g_code, prefix = "start", None, None, None
        for token_id in ids:
            token = self.name.get(int(token_id))
            if token is None:
                continue  # decoder start / pad token
            if state == "start" and token == "<HOP>":
                state = "action"
            elif state == "start" and not hops and token == "<RETURN_TOPIC>":
                state = "return"
            elif state == "return" and token.startswith("<T_"):
                index = int(token[3:-1])
                if index in returned:
                    return Program(tuple(hops)), "invalid", action, g_code, prefix
                returned.append(index)
            elif state == "return" and returned and token == "<END_RETURN>":
                state = "return_end"
            elif state == "return_end" and token == "<STOP>":
                state = "done"
            elif state == "start" and token == "<STOP>":
                state = "done"
            elif state == "action" and token.startswith("<A_"):
                action, state = int(token[3:-1]), "g" if self.include_g else "s"
                if not self.include_g:
                    g_code = -1
            elif state == "g" and (token == "<G_ANY>" or token.startswith("<G_")):
                g_code = -1 if token == "<G_ANY>" else int(token[3:-1])
                state = "s"
            elif state == "s" and token == "<S_ANY>":
                prefix, state = None, "end"
            elif state == "s" and token.startswith("<S1_"):
                prefix, state = (int(token[4:-1]),), "s1"
            elif state in ("s1", "s2") and token.startswith(f"<S{len(prefix) + 1}_"):
                prefix = prefix + (int(token[4:-1]),)
                state = f"s{len(prefix)}"
            elif state in ("end", "s1", "s2", "s3") and token == "<ENDHOP>":
                hops.append(Hop(action, g_code, prefix))
                state, action, g_code, prefix = "start", None, None, None
            else:
                return Program(tuple(hops), tuple(returned) if returned else None), "invalid", action, g_code, prefix
        return Program(tuple(hops), tuple(returned) if returned else None), state, action, g_code, prefix

    def allowed(self, ids: list[int], topics: tuple[int, ...]) -> list[int]:
        program, state, action, code, prefix = self.parse(ids)
        if state == "done":
            return [self.tokenizer.eos_token_id]
        if state == "invalid":
            return [self.id["<STOP>"]]
        if state == "return":
            remaining = [self.id[f"<T_{i}>"] for i in range(len(topics))
                         if i not in (program.return_topics or ())]
            return remaining + ([self.id["<END_RETURN>"]] if program.return_topics else [])
        if state == "return_end":
            return [self.id["<STOP>"]]
        current = self.executor.execute(topics, program)
        if not current.executable or current.budget_exceeded:
            return [self.id["<STOP>"]]
        frontier = current.candidates
        feasible = self.executor.feasible_actions(
            frontier, min(self.executor.edge_budget - current.edges_examined, self.executor.query_remaining)
        )
        if state == "start":
            options = [self.id["<STOP>"]]
            if not program.hops and topics:
                options.append(self.id["<RETURN_TOPIC>"])
            if len(program.hops) < self.max_hops and feasible:
                options.append(self.id["<HOP>"])
            return options
        if state == "action":
            return [self.id[f"<A_{a}>"] for a in sorted(feasible)] or [self.id["<STOP>"]]
        if state == "g":
            codes = feasible.get(action, set())
            return [self.id[f"<G_{g}>"] if g >= 0 else self.id["<G_ANY>"] for g in sorted(codes)] or [self.id["<G_ANY>"]]
        if state in ("s", "s1", "s2", "s3", "end"):
            if self.executor.prefix_s_index is not None:
                count = self.executor.action_counts(frontier).get((action, code), 0)
                if (current.edges_examined + count > self.executor.edge_budget or
                        count > self.executor.query_remaining or not count):
                    return [self.id["<S_ANY>"]]
            else:
                count = len(self.executor.selected_edges(frontier, action, code))
                if current.edges_examined + count > self.executor.edge_budget or not count:
                    return [self.id["<S_ANY>"]]
            prefixes = self.executor.available_prefixes(frontier, action, code)
            if state == "s":
                choices = [self.id["<S_ANY>"]]
                if self.executor.mode not in ("relation", "schema"):
                    choices += [self.id[f"<S1_{p[0]}>"] for p in sorted(prefixes) if len(p) == 1]
                return choices
            if state == "end":
                return [self.id["<ENDHOP>"]]
            choices = [self.id["<ENDHOP>"]]
            if len(prefix) < self.executor.codes.node_codes.shape[1]:
                choices += [self.id[f"<S{len(p)}_{p[-1]}>"] for p in sorted(prefixes)
                            if len(p) == len(prefix) + 1 and p[:len(prefix)] == prefix]
            return choices
        return [self.id["<STOP>"]]

    def decoded(self, ids: list[int]) -> Program | None:
        program, state, _, _, _ = self.parse(ids)
        return program if state == "done" else None


def input_text(graph: ADIntGraph, question: Question, style: str = "minimal") -> str:
    if style == "minimal":
        names = "; ".join(f"topic {index}: {graph.names[node]}"
                          for index, node in enumerate(question.topics))
        return f"question: {question.question} {names}"
    if style == "tagged":
        names = " ".join(f"<TOPIC>{index}: {graph.names[node]}</TOPIC>"
                         for index, node in enumerate(question.topics))
        return f"<QUESTION>{question.question}</QUESTION> {names}"
    raise ValueError(style)


def make_synthetic_examples(graph: ADIntGraph, executor: Executor, codec: ProgramCodec,
                            n_nodes: int, n_edges: int, seed: int,
                            cover_all_actions: bool = True) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(seed)
    rows = []
    node_ids = rng.choice(len(graph.ids), min(n_nodes, len(graph.ids)), replace=False)
    for node in node_ids:
        codes = executor.codes.node_codes[node]
        tokens = ["<READ_S>"] + [f"<S{i + 1}_{int(v)}>" for i, v in enumerate(codes)] + ["<STOP>"]
        rows.append((f"encode node: {graph.texts[node]}", [codec.id[x] for x in tokens] + [codec.tokenizer.eos_token_id]))
    if cover_all_actions:
        # Every decodable action must appear at least once in full training.
        if executor.mode == "relation":
            keys = graph.rel.astype(np.int64) * 2 + graph.direction
        elif executor.mode == "full":
            keys = graph.schema.astype(np.int64) * (int(executor.codes.edge_codes.max()) + 1) + executor.codes.edge_codes
        else:
            keys = graph.schema
        _, representatives = np.unique(keys, return_index=True)
        extra_count = max(0, min(n_edges, len(graph.src)) - len(representatives))
        extras = rng.choice(len(graph.src), extra_count, replace=False) if extra_count else np.empty(0, np.int64)
        edge_ids = np.concatenate((representatives, extras))
    else:
        edge_ids = rng.choice(len(graph.src), min(n_edges, len(graph.src)), replace=False)
    @lru_cache(maxsize=8192)
    def neighborhood(node: int) -> str:
        edges = graph.out_edges(node)
        labels = graph.rel[edges].astype(np.int32) * 2 + graph.direction[edges]
        values, counts = np.unique(labels, return_counts=True)
        leading = sorted(zip(values, counts), key=lambda pair: (-pair[1], pair[0]))[:8]
        return ", ".join(f"{graph.relation_names[int(label) // 2]} {'inverse' if int(label) % 2 else 'forward'} {int(count)}"
                         for label, count in leading)
    for edge in edge_ids:
        source, dest = int(graph.src[edge]), int(graph.dst[edge])
        schema = graph.schemas[int(graph.schema[edge])]
        relation = graph.relation_names[schema[1]] + (" inverse" if schema[2] else " forward")
        text = (f"encode transition: {graph.type_names[schema[0]]} {relation} {graph.type_names[schema[3]]}; "
                f"source {graph.names[source]}; destination {graph.names[dest]}; "
                f"source neighborhood {neighborhood(source)}; destination neighborhood {neighborhood(dest)}")
        action = executor.action_for_edge(int(edge))
        g_code = int(executor.codes.edge_codes[edge]) if executor.mode == "full" else -1
        tokens = ["<READ_G>", f"<A_{action}>"]
        if codec.include_g:
            tokens.append(f"<G_{g_code}>" if g_code >= 0 else "<G_ANY>")
        tokens.append("<STOP>")
        rows.append((text, [codec.id[x] for x in tokens] + [codec.tokenizer.eos_token_id]))
    random.Random(seed).shuffle(rows)
    return rows


def train_supervised(model, tokenizer, rows: list[tuple[str, list[int]]], *, epochs: int,
                     batch_size: int, learning_rate: float, device: torch.device, seed: int,
                     row_groups: list[tuple[str, int]] | None = None,
                     sampling: str = "program") -> list[float]:
    if sampling not in ("program", "question", "hop_balanced"):
        raise ValueError(sampling)
    if sampling != "program" and (row_groups is None or len(row_groups) != len(rows)):
        raise ValueError("one (question ID, minimum hops) group is required per SFT row")
    groups: dict[str, list[int]] = {}
    group_hops: dict[str, int] = {}
    if row_groups is not None:
        for index, (qid, hops) in enumerate(row_groups):
            groups.setdefault(qid, []).append(index)
            if qid in group_hops and group_hops[qid] != hops:
                raise ValueError(f"inconsistent minimum hops for question {qid}")
            group_hops[qid] = hops
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        if sampling == "program":
            epoch_indices = rng.permutation(len(rows))
        else:
            qids = list(groups)
            if sampling == "hop_balanced":
                short = [qid for qid in qids if group_hops[qid] < 2]
                two_hop = [qid for qid in qids if group_hops[qid] == 2]
                if short and two_hop:
                    smaller = short if len(short) < len(two_hop) else two_hop
                    qids = short + two_hop + list(rng.choice(smaller, size=abs(len(short) - len(two_hop)),
                                                               replace=True))
            epoch_indices = np.asarray([rng.choice(groups[qid]) for qid in qids], dtype=np.int64)
            rng.shuffle(epoch_indices)
        for batch_indices in np.array_split(epoch_indices, max(1, math.ceil(len(epoch_indices) / batch_size))):
            batch = [rows[int(i)] for i in batch_indices]
            encoded = tokenizer([x[0] for x in batch], return_tensors="pt", padding=True, truncation=True,
                                max_length=256, return_token_type_ids=False).to(device)
            width = max(len(x[1]) for x in batch)
            labels = torch.full((len(batch), width), -100, dtype=torch.long, device=device)
            for i, (_, target) in enumerate(batch):
                labels[i, :len(target)] = torch.tensor(target, device=device)
            loss = model(**encoded, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print(f"supervised epoch {epoch + 1}/{epochs}: examples={len(epoch_indices)} "
              f"labels={len(rows)} sampling={sampling} loss={history[-1]:.4f}", flush=True)
    return history


def generate_beams(model, tokenizer, codec: ProgramCodec, graph: ADIntGraph, question: Question,
                   beam: int, device: torch.device, query_edge_budget: int | None = None,
                   reset_query: bool = True) -> list[BeamResult]:
    if reset_query:
        codec.executor.begin_query(query_edge_budget)
    before_decode = codec.executor.query_edges_examined
    encoded = tokenizer(input_text(graph, question, codec.input_style), return_tensors="pt", truncation=True,
                        max_length=256, return_token_type_ids=False).to(device)
    model.eval()
    with torch.inference_mode():
        # Keep the token cap fixed at the three-hop training/evaluation value
        # when ablating max_hops=2. Otherwise a second generation limit changes.
        token_limit_hops = max(3, codec.max_hops)
        generated = model.generate(**encoded, num_beams=beam, num_return_sequences=beam,
                                   max_new_tokens=max(token_limit_hops * (5 + codec.executor.codes.node_codes.shape[1]) + 2,
                                                      codec.max_topics + 4),
                                   prefix_allowed_tokens_fn=lambda _batch, ids: codec.allowed(ids.tolist(), question.topics),
                                   return_dict_in_generate=True, output_scores=True,
                                   length_penalty=0.0, renormalize_logits=True,
                                   early_stopping=True)
    codec.executor.decoder_edges_examined += codec.executor.query_edges_examined - before_decode
    scores = generated.sequences_scores.float()
    probabilities = F.softmax(scores, dim=0).cpu().tolist()
    results = []
    for sequence, probability in zip(generated.sequences, probabilities):
        program = codec.decoded(sequence.tolist())
        if program is None:
            results.append(BeamResult(Program(()), float(probability), (), {}, 0, False))
            continue
        execution = codec.executor.execute(question.topics, program)
        results.append(BeamResult(program, float(probability), execution.candidates,
                                  execution.evidence, execution.edges_examined, execution.executable))
    return results


def train_answer_reward(model, tokenizer, codec: ProgramCodec, graph: ADIntGraph,
                        questions: list[Question], *, beam: int, epochs: int,
                        learning_rate: float, device: torch.device) -> list[float]:
    """Beam policy update from executed answer reward; no extra planner/reranker."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        rewards_seen = []
        for question_index, question in enumerate(questions, 1):
            beams = generate_beams(model, tokenizer, codec, graph, question, beam, device)
            if not beams:
                continue
            rewards = []
            for item in beams:
                overlap = len(set(item.candidates) & set(question.answers))
                reward = overlap / len(question.answers) - 0.025 * math.log1p(len(item.candidates))
                if not item.executable:
                    reward -= 0.25
                rewards.append(reward)
            rewards_seen.extend(rewards)
            input_ids = tokenizer(input_text(graph, question), return_tensors="pt", truncation=True,
                                  max_length=256, return_token_type_ids=False).to(device)
            targets = [codec.ids_for(item.program) for item in beams]
            labels = torch.full((len(targets), max(map(len, targets))), -100, dtype=torch.long, device=device)
            for i, target in enumerate(targets):
                labels[i, :len(target)] = torch.tensor(target, device=device)
            model.train()
            outputs = model(input_ids=input_ids.input_ids.repeat(len(targets), 1),
                            attention_mask=input_ids.attention_mask.repeat(len(targets), 1), labels=labels)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            safe_labels = labels.clamp_min(0)
            log_likelihood = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != -100
            log_likelihood = (log_likelihood * mask).sum(1) / mask.sum(1).clamp_min(1)
            # Fixed baseline keeps an all-positive or all-negative beam useful.
            advantage = torch.tensor(rewards, dtype=torch.float32, device=device) - 0.25
            loss = -(advantage.detach() * log_likelihood).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if question_index % 250 == 0 or question_index == len(questions):
                print(f"answer reward epoch {epoch + 1}/{epochs}: "
                      f"questions={question_index}/{len(questions)}", flush=True)
        history.append(float(np.mean(rewards_seen)) if rewards_seen else 0.0)
        print(f"answer reward epoch {epoch + 1}/{epochs}: mean={history[-1]:.4f}", flush=True)
    return history
