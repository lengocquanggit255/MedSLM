#!/usr/bin/env python3
"""One fresh PharmKG HOP PoLo v6 session with Hits@1-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from hop_polo_aligned_data import answer_paths
from hop_polo_full_policy import FullPolicy, program_cache as upstream_program_cache, teacher_loss
from hop_polo_hybrid import hash_words, program_from_dict
from hop_polo_v6_flan import (executable_qa_labels, initialize_program_tokens,
                             program_distribution, train_batch, train_qa_batch)
from hop_polo_v2_ranker import (cold_numeric_score, encode_texts,
                                evaluate_ranker, pool_summary, train_epoch,
                                word_set)
from hop_polo_v6_ranker import PoolRankerV6, collect_pool, training_labels
from hop_polo_v2_search import beam_candidates, coverage_diagnostics
from hopgs.codebooks import train_codebooks
from hopgs.data import ADIntGraph, Question
from hopgs.executor import Executor
from hopgs.model import ProgramCodec, input_text, make_synthetic_examples


RAW_FILES = {"ADint_KG.txt", "ADint_train_with_topic_entity.json",
             "ADint_test_with_topic_entity.json", "edge_type_dict.pkl",
             "node_info.pkl", "node_type_dict.pkl"}
FLAN_FILES = {"config.json", "generation_config.json", "model.safetensors",
              "special_tokens_map.json", "spiece.model", "tokenizer.json",
              "tokenizer_config.json"}
EMBED_FILES = {"config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
               "modules.json", "1_Pooling/config.json",
               "config_sentence_transformers.json"}


def program_cache(model, tokenizer, codec, graph, questions, cache_path, device):
    # Length-normalized generation corrects the built-in bias for short
    # RETURN_TOPIC/STOP programs. All milestones and final inference use it.
    return upstream_program_cache(model, tokenizer, codec, graph, questions,
                                  cache_path, device, length_penalty=2.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fingerprint(graph):
    h = hashlib.sha256()
    for array in (graph.ids, graph.src, graph.dst, graph.rel, graph.direction):
        h.update(array.tobytes())
    for value in graph.texts:
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def verify_input(root, config):
    manifest = json.loads((root / "input_manifest.json").read_text())
    assert manifest["flan_repo"] == config["flan_repo"]
    assert manifest["flan_revision"] == config["flan_revision"]
    assert manifest["embedding_repo"] == config["embedding_repo"]
    assert manifest["embedding_revision"] == config["embedding_revision"]
    allowed = ({f"raw/{name}" for name in RAW_FILES} |
               {f"base_model/{name}" for name in FLAN_FILES} |
               {f"embedding_model/{name}" for name in EMBED_FILES})
    actual = {str(path.relative_to(root)): digest(path) for path in root.rglob("*")
              if path.is_file() and path.name not in
              ("input_manifest.json", "dataset-metadata.json",
               "raw.zip", "base_model.zip", "embedding_model.zip")}
    assert set(actual) == allowed, f"unexpected/missing input files: {set(actual) ^ allowed}"
    assert actual == manifest["files"], "input hash mismatch"
    assert actual["base_model/model.safetensors"] == config["flan_model_sha256"]
    assert actual["embedding_model/model.safetensors"] == config["embedding_model_sha256"]
    return {"input_manifest_sha256": digest(root / "input_manifest.json"),
            "files": sorted(actual), "prior_pharmkg_checkpoint_input": False,
            "original_flan_model_sha256": config["flan_model_sha256"],
            "public_embedding_model_sha256": config["embedding_model_sha256"]}


def clean_splits(graph, seed=42):
    """Group exact question/topic pairs; test rows have no loaded answers."""
    original = graph.splits["train"] + graph.splits["val"]
    assert all(not q.answers for q in graph.splits["test"])
    test_keys = {(q.question.casefold().strip(), q.topics)
                 for q in graph.splits["test"]}
    groups, removed = {}, []
    for q in original:
        key = (q.question.casefold().strip(), q.topics)
        if key in test_keys:
            removed.append(q.qid)
        else:
            groups.setdefault(key, []).append(q)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    target = round(sum(map(len, groups.values())) * 0.10)
    val_keys, count = set(), 0
    for key in keys:
        if count >= target:
            break
        val_keys.add(key)
        count += len(groups[key])
    def tagged(q, name):
        return Question(q.qid, q.question, q.topics, q.answers, name)
    graph.splits["train"] = [tagged(q, "train") for key in keys if key not in val_keys
                             for q in groups[key]]
    graph.splits["val"] = [tagged(q, "val") for key in keys if key in val_keys
                           for q in groups[key]]
    all_keys = {name: {(q.question.casefold().strip(), q.topics) for q in rows}
                for name, rows in graph.splits.items()}
    all_texts = {name: {q.question.casefold().strip() for q in rows}
                 for name, rows in graph.splits.items()}
    overlaps = {f"{a}_{b}": {"question_topic": len(all_keys[a] & all_keys[b]),
                              "question_only": len(all_texts[a] & all_texts[b])}
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(not x["question_topic"] and not x["question_only"]
               for x in overlaps.values())
    def split_row(q):
        key = json.dumps([q.question.casefold().strip(),
                          [int(graph.ids[node]) for node in q.topics]],
                         ensure_ascii=False, separators=(",", ":"))
        return {"qid": q.qid, "question_topic_sha256":
                hashlib.sha256(key.encode()).hexdigest()}
    return {"seed": seed, "removed_test_duplicates": removed,
            "overlap": overlaps,
            "split_sizes": {name: len(rows) for name, rows in graph.splits.items()},
            "members": {name: [split_row(q) for q in rows]
                        for name, rows in graph.splits.items()},
            "test_answers_loaded": False}


def select_hits(rows):
    """Highest validation Hits@1, then earliest epoch; no soft guardrails."""
    valid = [row for row in rows if row.get("technical_valid", False)]
    return max(valid, key=lambda row: (row["validation"]["mean_hits_at_1"],
                                       -row["epoch"])) if valid else None


def select_phase_hits(rows, phase):
    return select_hits([row for row in rows if row["phase"] == phase])


def finite_predictions(metrics, predictions, expected):
    return (len(predictions) == expected and
            all(np.isfinite(value) for value in metrics.values()
                if isinstance(value, (int, float))) and
            all("hits_at_1" in row and np.isfinite(row["hits_at_1"])
                for row in predictions))


def proxy_validation(policy, graph, codes, questions, cache, degrees,
                     name_terms, text_terms, relation_terms, config):
    """Frozen numeric selector on the predicted-program pre-prune pool."""
    policy.eval()
    degrees = degrees.to(next(policy.parameters()).device)
    records = []
    diagnostics = []
    options = {"path_beam": config["path_beam"],
               "local_k": config["local_top_k"],
               "rescue_slots": config["terminal_rescue_slots"],
               "max_nodes": config["max_pool_nodes"]}
    with torch.inference_mode():
        for q, beam in zip(questions, cache):
            row = collect_pool(policy, graph, codes, q, beam, degrees,
                               name_terms, text_terms, relation_terms,
                               **options)
            scores = cold_numeric_score(row.numeric)
            order = np.lexsort((graph.ids[row.nodes.astype(np.int64)], -scores))
            top = [int(row.nodes[j]) for j in order[:20]]
            selected = set(top) | set(q.topics)
            for j in order[:20]:
                selected.update(int(node) for node in row.path_nodes[j] if node >= 0)
            gold = set(q.answers)
            hit_set = len(gold & selected)
            hit_top = len(gold & set(top))
            records.append({"qid": q.qid, "split": q.split,
                            "answers": [int(graph.ids[n]) for n in q.answers],
                            "topics": [int(graph.ids[n]) for n in q.topics],
                            "pool_nodes": [int(graph.ids[n]) for n in row.nodes],
                            "selected_nodes": [int(graph.ids[n]) for n in sorted(selected)],
                            "ranked_top20": [int(graph.ids[n]) for n in top],
                            "hits_at_1": float(bool(top) and top[0] in gold),
                            "recall_at_20": hit_top / len(gold),
                            "recall_set": hit_set / len(gold),
                            "precision_set": hit_set / len(selected),
                            "f2_set": 5 * hit_set / (4 * len(gold) + len(selected)),
                            "selected_nodes_count": len(selected),
                            "latency_ms": row.generation_ms + row.search_ms})
            diagnostics.append(row.search_diagnostics)
    metrics = {"queries": len(records)}
    for key in ("hits_at_1", "recall_at_20", "recall_set", "precision_set",
                "f2_set", "selected_nodes_count", "latency_ms"):
        metrics[f"mean_{key}"] = float(np.mean([row[key] for row in records]))
    metrics["p95_selected_nodes_count"] = float(np.percentile(
        [row["selected_nodes_count"] for row in records], 95))
    summary = {"mean_beam_oracle_recall_at_20": float(np.mean([
                   x["oracle_recall_at_20"] for x in diagnostics])),
               "mean_pool_recall_drop_from_cap": float(np.mean([
                   (x["gold_in_uncapped_pool"] - x["gold_in_capped_pool"]) /
                   x["gold_total"] for x in diagnostics])),
               "mean_beam_nodes": float(np.mean([x["beam_nodes"] for x in diagnostics])),
               "mean_pool_nodes": float(np.mean([x["pool_nodes"] for x in diagnostics])),
               "mean_terminal_mass": float(np.mean([
                   x["terminal_probability_mass"] for x in diagnostics])),
               "depth_gold": [sum(x["gold_by_depth"][d] for x in diagnostics)
                              for d in range(config["max_hops"])],
               "depth_candidates": [sum(x["candidate_by_depth"][d]
                                        for x in diagnostics) for d in range(config["max_hops"])],
               "topic_degree_buckets": {bucket: {"queries": sum(
                   x["topic_degree_bucket"] == bucket for x in diagnostics),
                   "beam_gold": sum(x["gold_in_candidates"] for x in diagnostics
                                    if x["topic_degree_bucket"] == bucket),
                   "gold_total": sum(x["gold_total"] for x in diagnostics
                                     if x["topic_degree_bucket"] == bucket)}
                   for bucket in ("<=116", "117-498", ">498")},
               "program_slots": [x["program_slots"] for x in diagnostics],
               "per_question": diagnostics}
    return metrics, records, summary


def run_flan(graph, codes, train, val, proxy_policy, proxy_context,
             config, output, input_root, device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    executor = Executor(graph, codes, "semantic", edge_budget=10**9,
                        prefix_s_index=True)
    label_executor = Executor(graph, codes, "semantic",
                              edge_budget=config["label_edge_cap"],
                              prefix_s_index=True)
    base = input_root / "base_model"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    old_vocab = len(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(base,
                                                  local_files_only=True).to(device)
    codec = ProgramCodec(tokenizer, graph, executor, max_hops=config["max_hops"],
                         include_g=False)
    token_audit = initialize_program_tokens(model, tokenizer, old_vocab,
                                            seed=config["seed"])
    model.to(device)
    qa_plans, label_audit, per_answer = executable_qa_labels(
        graph, codes, label_executor, codec, train,
        cap=config["label_edge_cap"],
        programs_per_answer=config["qa_sft"]["programs_per_answer"],
        paths_per_depth=config["qa_sft"]["paths_per_depth"])
    dump(output / "label_audits/flan_qa_sft.json", label_audit)
    jsonl(output / "label_audits/flan_qa_per_answer.jsonl", per_answer)
    synthetic = make_synthetic_examples(
        graph, executor, codec, config["synthetic"]["nodes"],
        config["synthetic"]["edges"], config["seed"])
    synthetic_count = len(synthetic)
    read_s = sum(ids[0] == codec.id["<READ_S>"] for _text, ids in synthetic)
    read_g = sum(ids[0] == codec.id["<READ_G>"] for _text, ids in synthetic)
    assert read_s + read_g == synthetic_count
    assert all(codec.id["<HOP>"] not in ids for _text, ids in synthetic)
    dump(output / "label_audits/flan_synthetic.json",
         {"examples": synthetic_count, "source": "KG only",
          "read_s_examples": read_s, "read_g_examples": read_g,
          "hop_program_format_added": False})
    train_zero_rate = label_audit["shortest_hops"]["0"] / max(1, label_audit["answers"])
    epoch_dir = output / "flan_epoch_checkpoints"
    cache_dir = output / "flan/val_program_cache_by_epoch"
    epoch_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    history = []
    total_updates = 0
    names, texts, relations, degrees = proxy_context

    def evaluate(phase, phase_epoch, train_loss, train_examples=0,
                 train_answers=0):
        index = len(history)
        saved = epoch_dir / f"epoch_{index:02d}"
        model.save_pretrained(saved, safe_serialization=True)
        tokenizer.save_pretrained(saved)
        cached = program_cache(model, tokenizer, codec, graph, val,
                               cache_dir / f"epoch_{index:02d}.jsonl", device)
        metrics, predictions, diagnostics = proxy_validation(
            proxy_policy, graph, codes, val, cached, degrees,
            names, texts, relations, config)
        distribution = program_distribution(
            cached, val, train_zero_rate=train_zero_rate,
            zero_factor=config["program_audit"]["flan_top_zero_factor"],
            zero_floor=config["program_audit"]["flan_top_zero_floor"],
            nonzero_beam_floor=config["program_audit"]["flan_nonzero_beam_floor"])
        technical = finite_predictions(metrics, predictions, len(val))
        row = {"epoch": index, "phase": phase, "phase_epoch": phase_epoch,
               "updates": total_updates, "train_loss": train_loss,
               "train_examples": train_examples,
               "train_answer_examples": train_answers,
               "validation": metrics, "program_audit": distribution,
               "candidate_diagnostics": diagnostics,
               "technical_valid": technical,
               "proxy": config["proxy_evaluation"]}
        history.append(row)
        jsonl(output / f"flan/val_predictions_epoch_{index:02d}.jsonl",
              predictions)
        dump(output / "flan_epoch_history.json", history)
        selected = select_phase_hits(history, phase)
        print(f"v6 Flan {phase} epoch {phase_epoch}: "
              f"H1={metrics['mean_hits_at_1']:.4f} "
              f"R20={metrics['mean_recall_at_20']:.4f} "
              f"top0={distribution['top_zero_rate']:.3f} "
              f"valid={technical} best={selected['epoch'] if selected else None}",
              flush=True)
        return row

    syn = config["synthetic"]
    evaluate("synthetic", 0, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=syn["learning_rate"])
    rng = np.random.default_rng(config["seed"] + 11)
    stale = 0
    for epoch in range(1, syn["epochs"] + 1):
        order = rng.permutation(len(synthetic))
        losses = []
        for start in range(0, len(order), syn["batch_size"]):
            examples = [synthetic[int(i)] for i in order[start:start + syn["batch_size"]]]
            model.train()
            losses.append(train_batch(model, tokenizer, examples, optimizer, device))
            total_updates += 1
        assert len(losses) == math.ceil(len(synthetic) / syn["batch_size"])
        row = evaluate("synthetic", epoch, float(np.mean(losses)), len(order))
        best = select_phase_hits(history, "synthetic")
        stale = 0 if best is not None and best["epoch"] == row["epoch"] else stale + 1
        if stale >= syn["patience"]:
            break
    selected_synthetic = select_phase_hits(history, "synthetic")
    if selected_synthetic is None:
        raise RuntimeError("no technically valid synthetic validation checkpoint")
    del optimizer, synthetic, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        epoch_dir / f"epoch_{selected_synthetic['epoch']:02d}",
        local_files_only=True).to(device)

    qa = config["qa_sft"]
    usable = np.asarray([i for i, answers in enumerate(qa_plans)
                         if any(answers)], dtype=np.int64)
    if not len(usable):
        raise RuntimeError("no train QA has a cap-200 executable program")
    evaluate("qa_sft", 0, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=qa["learning_rate"])
    rng = np.random.default_rng(config["seed"] + 12)
    stale = 0
    for epoch in range(1, qa["epochs"] + 1):
        order = rng.permutation(usable)
        losses = []
        answer_examples = 0
        for start in range(0, len(order), qa["batch_size"]):
            model.train()
            loss, answers = train_qa_batch(
                model, tokenizer, graph, train, qa_plans,
                order[start:start + qa["batch_size"]], epoch,
                optimizer, device)
            losses.append(loss)
            answer_examples += answers
            total_updates += 1
        assert len(losses) == math.ceil(len(usable) / qa["batch_size"])
        assert answer_examples == label_audit["answers_with_labels"]
        row = evaluate("qa_sft", epoch, float(np.mean(losses)),
                       len(order), answer_examples)
        best = select_phase_hits(history, "qa_sft")
        stale = 0 if best is not None and best["epoch"] == row["epoch"] else stale + 1
        if stale >= qa["patience"]:
            break
    selected = select_phase_hits(history, "qa_sft")
    if selected is None:
        raise RuntimeError("no technically valid QA SFT validation checkpoint")
    del optimizer, qa_plans, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_dir = output / "checkpoint_best/flan_no_g"
    shutil.copytree(epoch_dir / f"epoch_{selected['epoch']:02d}", best_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        best_dir, local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = output / "program_cache"
    cache.mkdir(exist_ok=True)
    shutil.copy2(cache_dir / f"epoch_{selected['epoch']:02d}.jsonl",
                 cache / "val.jsonl")
    val_cache = [json.loads(line) for line in (cache / "val.jsonl").open()]
    train_cache = program_cache(model, tokenizer, codec, graph, train,
                                cache / "train.jsonl", device)
    train_distribution = program_distribution(
        train_cache, train, train_zero_rate=train_zero_rate,
        zero_factor=config["program_audit"]["flan_top_zero_factor"],
        zero_floor=config["program_audit"]["flan_top_zero_floor"],
        nonzero_beam_floor=config["program_audit"]["flan_nonzero_beam_floor"])
    dump(cache / "train_distribution.json", train_distribution)
    dump(cache / "val_distribution.json", selected["program_audit"])
    return model, tokenizer, codec, train_cache, val_cache, {
        "selected_epoch": selected["epoch"],
        "selected_phase_epoch": selected["phase_epoch"],
        "selected_updates": selected["updates"],
        "synthetic_selected_epoch": selected_synthetic["epoch"],
        "synthetic_selected_phase_epoch": selected_synthetic["phase_epoch"],
        "model_sha256": digest(best_dir / "model.safetensors"),
        "token_initialization": token_audit,
        "synthetic_examples": synthetic_count,
        "selected_program_audit": selected["program_audit"],
        "train_program_audit": train_distribution,
        "proxy_evaluation": config["proxy_evaluation"],
        "qa_label_audit": label_audit}


def run_policy(graph, codes, train, val, train_cache, val_cache,
               policy, proxy_context, config, output, device):
    names, texts, relations, degrees_cpu = proxy_context
    degrees_gpu = degrees_cpu.to(device)
    words = [hash_words(q.question) for q in train]
    epoch_dir = output / "policy_warmup_epoch_checkpoints"
    epoch_dir.mkdir()
    teacher_paths = []
    for q in train:
        choices = []
        for answer in q.answers:
            buckets, shortest, _scanned, _censored = answer_paths(
                graph, q, answer)
            if shortest is None:
                raise AssertionError(f"warm-up QA has no <=3-hop path: {q.qid}")
            choices.append(buckets[shortest][0])
        teacher_paths.append(choices)
    settings = config["policy_warmup"]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"])
    rng = np.random.default_rng(config["seed"] + 21)
    history = []
    updates = 0

    def evaluate(epoch, loss, train_examples):
        checkpoint = epoch_dir / f"epoch_{epoch:02d}.pt"
        torch.save(policy.state_dict(), checkpoint)
        metrics, predictions, diagnostics = proxy_validation(
            policy, graph, codes, val, val_cache, degrees_cpu,
            names, texts, relations, config)
        technical = finite_predictions(metrics, predictions, len(val))
        row = {"epoch": epoch, "updates": updates, "train_loss": loss,
               "train_examples": train_examples,
               "validation": metrics, "candidate_diagnostics": diagnostics,
               "technical_valid": technical,
               "proxy": config["proxy_evaluation"]}
        history.append(row)
        jsonl(output / f"policy_warmup/val_predictions_epoch_{epoch:02d}.jsonl",
              predictions)
        dump(output / "policy_warmup_epoch_history.json", history)
        best = select_hits(history)
        print(f"v6 policy warm-up epoch {epoch}: "
              f"H1={metrics['mean_hits_at_1']:.4f} "
              f"R20={metrics['mean_recall_at_20']:.4f} "
              f"best={best['epoch'] if best else None}", flush=True)
        return row

    evaluate(0, None, 0)
    stale = 0
    for epoch in range(1, settings["epochs"] + 1):
        policy.train()
        losses = []
        order = rng.permutation(len(train))
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for pos, index in enumerate(order, 1):
            index = int(index)
            q = train[index]
            teacher = teacher_paths[index][(epoch - 1 + index) %
                                           len(teacher_paths[index])]
            ce, rank = teacher_loss(policy, graph, codes, q, teacher,
                                    words[index], degrees_gpu, device, 3)
            loss = ce + 0.15 * rank
            (loss / 16).backward()
            pending += 1
            losses.append(float(loss.detach()))
            if pending == 16 or pos == len(order):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                updates += 1
        assert len(losses) == len(train)
        row = evaluate(epoch, float(np.mean(losses)), len(order))
        best = select_hits(history)
        stale = 0 if best is not None and best["epoch"] == epoch else stale + 1
        if stale >= settings["patience"]:
            break
    selected = select_hits(history)
    if selected is None:
        raise RuntimeError("no technically valid policy warm-up validation checkpoint")
    best_path = output / "checkpoint_best/policy.pt"
    best_path.parent.mkdir(exist_ok=True)
    shutil.copy2(epoch_dir / f"epoch_{selected['epoch']:02d}.pt", best_path)
    policy.load_state_dict(torch.load(best_path, map_location=device,
                                      weights_only=True))
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    # Train gold paths are used only for this diagnostic, never for aligned CE.
    mismatch = Counter()
    for q, paths_by_answer, cached in zip(train, teacher_paths, train_cache):
        for _pos, path in paths_by_answer:
            mismatch["train_answer_paths"] += 1
            if not path:
                mismatch["topic_is_answer_paths"] += 1
                continue
            action_any = False
            compatible_s_any = False
            full_s_any = False
            for beam in cached["beams"]:
                program = program_from_dict(beam["program"])
                if len(program.hops) < len(path):
                    continue
                action = all(int(graph.schema[edge]) == program.hops[depth].action
                             for depth, edge in enumerate(path))
                matching_s = action and all(
                    hop.s is None or tuple(int(x) for x in codes.node_codes[
                        int(graph.dst[edge]), :len(hop.s)]) == hop.s
                    for hop, edge in zip(program.hops, path))
                full_s = action and all(
                    hop.s == tuple(int(x) for x in codes.node_codes[
                        int(graph.dst[edge])])
                    for hop, edge in zip(program.hops, path))
                action_any |= action
                compatible_s_any |= matching_s
                full_s_any |= full_s
            mismatch["any_beam_path_prefix_action_match"] += int(action_any)
            mismatch["any_beam_path_prefix_action_s_compatible"] += int(compatible_s_any)
            mismatch["any_beam_path_prefix_action_full_s"] += int(full_s_any)
            mismatch["any_beam_s_backoff_only"] += int(compatible_s_any and
                                                       not full_s_any)
            mismatch["no_beam_action_match"] += int(not action_any)
    dump(output / "policy_program_mismatch.json", dict(mismatch))
    del optimizer, teacher_paths
    return policy, {"selected_epoch": selected["epoch"],
                    "selected_updates": selected["updates"],
                    "checkpoint_sha256": digest(best_path),
                    "program_mismatch_audit": dict(mismatch),
                    "aligned_training": False}


def make_pool_rows(policy, graph, codes, questions, cache, degrees,
                   name_terms, text_terms, relation_terms, config):
    rows = []
    degrees = degrees.to(next(policy.parameters()).device)
    for index, (q, beam) in enumerate(zip(questions, cache), 1):
        rows.append(collect_pool(
            policy, graph, codes, q, beam, degrees,
            name_terms, text_terms, relation_terms,
            path_beam=config["path_beam"], local_k=config["local_top_k"],
            rescue_slots=config["terminal_rescue_slots"],
            max_nodes=config["max_pool_nodes"]))
        if index % 250 == 0:
            print(f"v6 pool {q.split} {index}/{len(questions)}", flush=True)
    return rows


def pool_candidate_audit(rows, questions):
    diagnostics = [row.search_diagnostics for row in rows]
    return {"queries": len(rows),
            "mean_pool_recall_drop_from_cap": float(np.mean([
                (x["gold_in_uncapped_pool"] - x["gold_in_capped_pool"]) /
                x["gold_total"] for x in diagnostics])),
            "gold_in_uncapped_pool": sum(x["gold_in_uncapped_pool"] for x in diagnostics),
            "gold_in_capped_pool": sum(x["gold_in_capped_pool"] for x in diagnostics),
            "gold_in_final_beam": sum(x["gold_in_candidates"] for x in diagnostics),
            "gold_total": sum(x["gold_total"] for x in diagnostics),
            "mean_beam_oracle_recall_at_20": float(np.mean([
                x["oracle_recall_at_20"] for x in diagnostics])),
            "mean_pool_nodes": float(np.mean([len(row.nodes) for row in rows])),
            "mean_final_beam_nodes": float(np.mean([
                x["beam_nodes"] for x in diagnostics])),
            "mean_pool_search_ms": float(np.mean([row.search_ms for row in rows])),
            "gold_by_depth": [sum(x["gold_by_depth"][depth]
                                  for x in diagnostics) for depth in range(3)],
            "candidate_by_depth": [sum(x["candidate_by_depth"][depth]
                                       for x in diagnostics) for depth in range(3)],
            "mean_uncapped_nodes": float(np.mean([
                x["uncapped_nodes"] for x in diagnostics])),
            "terminal_queries": sum(x["terminal_programs"] > 0 for x in diagnostics)}


def run_pool_ranker(policy, flan, tokenizer, graph, codes, train, val,
                    train_cache, val_cache, proxy_context, config, output,
                    device):
    names, texts, relation_terms, degrees = proxy_context
    pools = {}
    for name, questions, cache in (("train", train, train_cache),
                                   ("val", val, val_cache)):
        rows = make_pool_rows(policy, graph, codes, questions, cache,
                              degrees, names, texts, relation_terms, config)
        pools[name] = rows
        dump(output / f"{name}_pool_summary.json", pool_summary(rows, questions))
        dump(output / f"{name}_pool_candidate_audit.json",
             pool_candidate_audit(rows, questions))
    train_labels, train_label_stats = training_labels(pools["train"], train, graph)
    dump(output / "label_audits/pool_ranker.json", train_label_stats)
    node_text, _ = encode_texts(flan, tokenizer, graph.texts, device,
                                max_length=96)
    relation_names = (graph.relation_names +
                      [f"inverse of {name}" for name in graph.relation_names])
    relation_text, _ = encode_texts(flan, tokenizer, relation_names,
                                    device, max_length=48)
    qtext, qms = {}, {}
    for name, questions in (("train", train), ("val", val)):
        qtext[name], qms[name] = encode_texts(
            flan, tokenizer, [input_text(graph, q) for q in questions],
            device, max_length=128, individual_timing=(name == "val"))
    ranker = PoolRankerV6(node_text, relation_text, graph.types).to(device)
    settings = config["pool_ranker"]
    optimizer = torch.optim.AdamW(ranker.parameters(),
                                   lr=settings["learning_rate"],
                                   weight_decay=1e-3)
    rng = np.random.default_rng(config["seed"] + 41)
    epoch_dir = output / "pool_ranker_epoch_checkpoints"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    history = []
    stale = 0
    pool_diag = pool_candidate_audit(pools["val"], val)
    for epoch in range(settings["epochs"] + 1):
        trained = None
        if epoch:
            trained = train_epoch(ranker, pools["train"], train,
                                  qtext["train"], train_labels,
                                  optimizer, device, rng)
        metrics, predictions = evaluate_ranker(
            ranker, pools["val"], val, qtext["val"], qms["val"],
            graph, device, save_predictions=True)
        row = {"epoch": epoch, "train": trained, "validation": metrics,
               "candidate_diagnostics": pool_diag,
               "technical_valid": finite_predictions(metrics, predictions,
                                                     len(val)),
               "old_gate": float(ranker.old_scale)}
        assert row["old_gate"] == 0.0 and not ranker.old_scale.requires_grad
        history.append(row)
        torch.save(ranker.state_dict(), epoch_dir / f"epoch_{epoch:02d}.pt")
        jsonl(output / f"pool_ranker/val_predictions_epoch_{epoch:02d}.jsonl",
              predictions)
        dump(output / "pool_ranker_epoch_history.json", history)
        best = select_hits(history)
        print(f"v6 pool ranker epoch {epoch}: "
              f"H1={metrics['mean_hits_at_1']:.4f} "
              f"R20={metrics['mean_recall_at_20']:.4f} "
              f"old_gate={float(ranker.old_scale):.3f} "
              f"selected={best['epoch'] if best else None}", flush=True)
        if epoch:
            stale = 0 if best is not None and best["epoch"] == epoch else stale + 1
            if stale >= settings["patience"]:
                break
    selected = select_hits(history)
    if selected is None:
        raise RuntimeError("no technically valid pool ranker validation checkpoint")
    ranker_best = output / "checkpoint_best/pool_ranker.pt"
    ranker_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(epoch_dir / f"epoch_{selected['epoch']:02d}.pt",
                 ranker_best)
    ranker.load_state_dict(torch.load(ranker_best, map_location=device,
                                      weights_only=True))
    val_metrics, val_predictions = evaluate_ranker(
        ranker, pools["val"], val, qtext["val"], qms["val"],
        graph, device, save_predictions=True)
    for key in ("mean_hits_at_1", "mean_recall_at_20"):
        assert abs(val_metrics[key] - selected["validation"][key]) < 1e-9
    dump(output / "val_metrics.json", val_metrics)
    jsonl(output / "val_predictions.jsonl", val_predictions)
    assert float(ranker.old_scale) == 0.0
    return ranker, {"selected_epoch": selected["epoch"],
                    "checkpoint_sha256": digest(ranker_best),
                    "train_label_stats": train_label_stats,
                    "old_gate_fixed": 0.0,
                    "old_scores_used": False,
                    "pool_candidate_audit": pool_diag}, node_text
