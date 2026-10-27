#!/usr/bin/env python3
"""One-session PharmKG HOP PoLo V7 Setting B, trained from public weights."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from hop_polo_full_policy import FullPolicy, teacher_loss
from hop_polo_hybrid import hash_words
from hop_polo_v2_ranker import encode_texts, evaluate_ranker, pool_summary, word_set
from hop_polo_v6 import (clean_splits, digest, dump, fingerprint, jsonl,
                         finite_predictions, make_pool_rows, pool_candidate_audit,
                         program_cache, proxy_validation, run_flan, run_policy,
                         run_pool_ranker, select_hits, verify_input)
from hop_polo_v6_flan import program_distribution
from hop_polo_v6_ranker import PoolRankerV6
from v7_setting_b_helpers import (_baseline_teacher, _mine_train, policy_stale,
                                  train_soft_hard_gap)
from hopgs.codebooks import train_codebooks
from hopgs.data import ADIntGraph
from hopgs.model import input_text
from v7_hard_audit import aggregate
from v7_surrogate import question_path_loss


def check_config(config):
    if config["protocol"] != "pharmkg-hop-polo-v7-final-from-scratch-setting-b":
        raise ValueError("wrong full-pipeline protocol")
    if config["seed"] != 42 or config["max_hops"] != 3:
        raise ValueError("Setting B requires seed 42 and three hops")
    if config["label_edge_cap"] != 200 or config["inference_edge_cap"] is not None:
        raise ValueError("label/inference cap differs from Setting B")
    if config["path_beam"] != 30 or config["local_top_k"] != 60:
        raise ValueError("Setting B beam/local budget differs")
    if not config["prefix_s_index"] or not config["no_g"]:
        raise ValueError("A/S grammar differs")
    if config["program_beam"] != 5 or config["decoder_length_penalty"] != 2.0:
        raise ValueError("Flan decoder differs")
    for stage in ("synthetic", "qa_sft", "policy_warmup", "policy", "pool_ranker"):
        if config[stage]["epochs"] != 24 or config[stage]["patience"] != 3:
            raise ValueError(f"{stage} must have max 24 update epochs, patience 3")
    if len(config["temperature_by_epoch"]) < 24:
        raise ValueError("Setting B temperature schedule is too short")


def recount(rows):
    """Independent count from saved node IDs, without evaluator metric fields."""
    values = {"hits_at_1": [], "recall_at_20": [], "pool_recall": [],
              "recall_set": [], "precision_set": [], "f2_set": []}
    for row in rows:
        gold = set(row["answers"])
        top = row["ranked_top20"]
        selected = set(row["selected_nodes"])
        pool = set(row["pool_nodes"])
        if not gold or not top or len(top) != len(set(top)):
            raise AssertionError(f"invalid prediction: {row['qid']}")
        hit = len(gold & selected)
        values["hits_at_1"].append(float(top[0] in gold))
        values["recall_at_20"].append(len(gold & set(top)) / len(gold))
        values["pool_recall"].append(len(gold & pool) / len(gold))
        values["recall_set"].append(hit / len(gold))
        values["precision_set"].append(hit / len(selected))
        values["f2_set"].append(5 * hit / (4 * len(gold) + len(selected)))
    return {f"mean_{name}": float(np.mean(series)) for name, series in values.items()}


def train_setting_b(graph, codes, train, val, train_cache, val_cache,
                    proxy_context, config, output, warmup_checkpoint, device):
    """V7 branch B loss/selection, initialized only from this session's warm-up."""
    names, texts, relations, degrees_cpu = proxy_context
    seed = config["seed"]
    torch.manual_seed(seed)
    policy = FullPolicy(len(graph.ids), len(graph.schemas),
                        int(codes.node_codes.max()) + 1).to(device)
    policy.load_state_dict(torch.load(warmup_checkpoint, map_location=device,
                                      weights_only=True))
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    teacher, teacher_depths = _baseline_teacher(graph, train)
    routes, route_audit = _mine_train(
        graph, train, config["route_cap_per_answer"], seed)
    dump(output / "B/label_audits/teacher_depths.json", teacher_depths)
    dump(output / "B/label_audits/train_route_mining.json", route_audit)
    words = [hash_words(q.question) for q in train]
    degrees = degrees_cpu.to(device)
    settings = config["policy"]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings["learning_rate"])
    rng = np.random.default_rng(seed + 21)
    history = []
    updates = stale = 0
    out = output / "B"
    for epoch in range(settings["epochs"] + 1):
        component = Counter()
        if epoch:
            policy.train()
            order = rng.permutation(len(train))
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            tau = config["temperature_by_epoch"][epoch - 1]
            for pos, index in enumerate(order, 1):
                index = int(index)
                q = train[index]
                chosen = teacher[index][(epoch - 1 + index) % len(teacher[index])]
                ce, node_rank = teacher_loss(policy, graph, codes, q, chosen,
                                             words[index], degrees, device, 3)
                marginal, soft_mass, eligible = question_path_loss(
                    policy, graph, codes, q, train_cache[index], degrees,
                    routes[index], mode="B", tau=tau,
                    path_beam=config["path_beam"], local_k=config["local_top_k"],
                    rescue_slots=config["terminal_rescue_slots"])
                loss = (config["loss"]["teacher_ce"] * ce +
                        config["loss"]["node_rank"] * node_rank +
                        config["loss"]["path"] * marginal)
                (loss / settings["batch_size"]).backward()
                pending += 1
                component["path_eligible_answers"] += eligible
                for name, value in (("teacher_ce", ce), ("node_rank", node_rank),
                                    ("marginal", marginal), ("soft_mass", soft_mass),
                                    ("total", loss)):
                    component[name] += float(value.detach())
                if pending == settings["batch_size"] or pos == len(order):
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    if not math.isfinite(float(norm)):
                        raise FloatingPointError("nonfinite Setting B policy gradient")
                    component["grad_norm_sum"] += float(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    updates += 1
            for name in ("teacher_ce", "node_rank", "marginal", "soft_mass", "total"):
                component[name] /= len(train)
            component["grad_norm_mean"] = component.pop("grad_norm_sum") / math.ceil(
                len(train) / settings["batch_size"])
        policy.eval()
        checkpoint = out / f"checkpoints/policy_epoch_{epoch:02d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), checkpoint)
        metrics, predictions, diagnostics = proxy_validation(
            policy, graph, codes, val, val_cache, degrees_cpu,
            names, texts, relations, config)
        hard = aggregate([x["v7_hard"] for x in diagnostics["per_question"]])
        gap = train_soft_hard_gap(
            policy, graph, codes, train, train_cache, routes, degrees, config,
            config["temperature_by_epoch"][max(0, epoch - 1)])
        row = {"epoch": epoch, "updates": updates,
               "temperature": config["temperature_by_epoch"][epoch - 1] if epoch else None,
               "loss": dict(component), "validation": metrics,
               "hard_coverage": hard, "train_soft_hard_gap": gap,
               "candidate_diagnostics": diagnostics,
               "technical_valid": finite_predictions(metrics, predictions, len(val)),
               "proxy": config["proxy_evaluation"]}
        history.append(row)
        jsonl(out / f"predictions/val_proxy_epoch_{epoch:02d}.jsonl", predictions)
        dump(out / "history/policy.json", history)
        best = select_hits(history)
        print(f"V7 final B policy epoch {epoch}: H1={metrics['mean_hits_at_1']:.4f} "
              f"R20={metrics['mean_recall_at_20']:.4f} "
              f"best={best['epoch'] if best else None}", flush=True)
        stale = policy_stale(history, epoch, stale)
        if epoch and stale >= settings["patience"]:
            break
    selected = select_hits(history)
    if selected is None:
        raise RuntimeError("no technically valid Setting B policy checkpoint")
    best_path = out / "checkpoints/policy_best.pt"
    shutil.copy2(out / f"checkpoints/policy_epoch_{selected['epoch']:02d}.pt",
                 best_path)
    policy.load_state_dict(torch.load(best_path, map_location=device,
                                      weights_only=True))
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    coverage_max = max(history, key=lambda x:
                       x["hard_coverage"].get("answer_2hop_pool", 0))
    audit = {"selected_epoch": selected["epoch"], "selected_updates": selected["updates"],
             "checkpoint_sha256": digest(best_path),
             "warmup_input_sha256": digest(warmup_checkpoint),
             "highest_2hop_pool_epoch": coverage_max["epoch"],
             "selected_2hop_pool": selected["hard_coverage"].get("answer_2hop_pool", 0),
             "highest_2hop_pool": coverage_max["hard_coverage"].get("answer_2hop_pool", 0),
             "last_epoch": epoch, "last_stale_count": stale,
             "train_route_mining": route_audit,
             "checkpoint_rule": config["selection"]}
    dump(out / "audit/policy_selection.json", audit)
    return policy, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-split", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    check_config(config)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("full run requires an empty output directory")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    input_audit = verify_input(args.input_root, config)
    shutil.copy2(args.input_root / "input_manifest.json", output / "input_manifest.json")
    shutil.copy2(args.config, output / "config.json")
    shutil.copy2(args.reference_split, output / "reference_split.json")
    if args.source_manifest is not None:
        source_root = args.source_manifest.parent
        source_files = json.loads(args.source_manifest.read_text())
        if not source_files or any(
            digest(source_root / name) != expected
            for name, expected in source_files.items()
        ):
            raise AssertionError("source manifest does not match executable source")
        shutil.copy2(args.source_manifest, output / "source_manifest.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    graph = ADIntGraph(args.input_root / "raw", seed=seed, defer_test_answers=True)
    graph_sha = fingerprint(graph)
    split = clean_splits(graph, seed=seed)
    reference = json.loads(args.reference_split.read_text())
    if graph_sha != reference["graph_sha256"]:
        raise AssertionError("raw graph differs from fixed Setting B split")
    for part in ("train", "val", "test"):
        if [r["qid"] for r in split["members"][part]] != reference[f"{part}_qids"]:
            raise AssertionError(f"{part} qid order differs from Setting B")
    dump(output / "split_manifest.json", split)
    train, val = graph.splits["train"], graph.splits["val"]
    audit = {"status": "RUNNING", "seed": seed, "graph_sha256": graph_sha,
             "input_lineage": input_audit,
             "test_answers_loaded_before_checkpoint_lock": False,
             "training_labels_split": "QA train only",
             "checkpoint_selection_split": "validation only",
             "stages": ["fresh codebook", "Flan synthetic", "Flan QA SFT",
                        "policy warm-up", "Setting B policy", "pool ranker",
                        "locked test once"]}
    dump(output / "audit.json", audit)
    started = time.perf_counter()
    times = {}
    mark = time.perf_counter()
    b = config["codebook"]
    codes = train_codebooks(
        graph, output / "codebooks", backend=b["backend"],
        model_name=str(args.input_root / "embedding_model"),
        dim=b["embedding_dim"], s_levels=b["semantic_levels"],
        s_size=b["semantic_codes"], s_steps=b["semantic_steps"],
        g_size=b["structural_codes"], g_min_edges=b["structural_min_edges"],
        g_steps=b["structural_steps"], fit_sample=b["fit_sample"],
        batch_size=b["batch_size"],
        embedding_batch_size=b["embedding_batch_size"],
        seed=seed, device=args.device)
    code_dir = output / "codebooks"
    actual_dim = int(np.load(code_dir / "node_embeddings.npy", mmap_mode="r").shape[1])
    if actual_dim != b["embedding_dim"]:
        raise AssertionError("codebook embedding dimension differs")
    code_manifest = {"graph_sha256": graph_sha,
                     "embedding_model_sha256": config["embedding_model_sha256"],
                     "embedding_revision": config["embedding_revision"],
                     "settings": b, "actual_embedding_dim": actual_dim,
                     "semantic": codes.semantic_audit,
                     "structural": codes.structural_audit,
                     "node_codes_sha256": digest(code_dir / "node_codes.npy"),
                     "edge_codes_sha256": digest(code_dir / "edge_codes.npy"),
                     "node_embeddings_sha256": digest(code_dir / "node_embeddings.npy")}
    dump(code_dir / "manifest.json", code_manifest)
    audit["codebook"] = code_manifest
    dump(output / "audit.json", audit)
    times["codebook_seconds"] = time.perf_counter() - mark
    names = [word_set(x) for x in graph.names]
    texts = [word_set(x) for x in graph.texts]
    relation_names = graph.relation_names + [f"inverse of {x}" for x in graph.relation_names]
    relation_terms = [word_set(x) for x in relation_names]
    degrees = torch.as_tensor(np.log1p(np.diff(graph.indptr)) / 16, dtype=torch.float32)
    proxy_context = names, texts, relation_terms, degrees
    proxy_policy = FullPolicy(len(graph.ids), len(graph.schemas),
                              int(codes.node_codes.max()) + 1).to(device)
    mark = time.perf_counter()
    flan, tokenizer, codec, train_cache, val_cache, flan_audit = run_flan(
        graph, codes, train, val, proxy_policy, proxy_context, config,
        output, args.input_root, device)
    audit["flan"] = flan_audit
    dump(output / "audit.json", audit)
    times["flan_seconds"] = time.perf_counter() - mark
    mark = time.perf_counter()
    warmup_policy, warmup_audit = run_policy(
        graph, codes, train, val, train_cache, val_cache,
        proxy_policy, proxy_context, config, output, device)
    audit["policy_warmup"] = warmup_audit
    dump(output / "audit.json", audit)
    times["warmup_seconds"] = time.perf_counter() - mark
    del warmup_policy
    mark = time.perf_counter()
    policy, setting_b_audit = train_setting_b(
        graph, codes, train, val, train_cache, val_cache, proxy_context,
        config, output, output / "checkpoint_best/policy.pt", device)
    audit["setting_b_policy"] = setting_b_audit
    dump(output / "audit.json", audit)
    times["setting_b_seconds"] = time.perf_counter() - mark
    mark = time.perf_counter()
    torch.manual_seed(seed + 41)
    ranker, ranker_audit, _ = run_pool_ranker(
        policy, flan, tokenizer, graph, codes, train, val,
        train_cache, val_cache, proxy_context, config, output / "B", device)
    selected_val_predictions = [json.loads(line) for line in
                                (output / "B/val_predictions.jsonl").open()]
    selected_val_metrics = json.loads((output / "B/val_metrics.json").read_text())
    val_recount = recount(selected_val_predictions)
    for name, value in val_recount.items():
        if abs(selected_val_metrics[name] - value) > 1e-9:
            raise AssertionError(f"validation metric recount differs: {name}")
    dump(output / "B/val_independent_recount.json", val_recount)
    audit["pool_ranker"] = ranker_audit
    dump(output / "audit.json", audit)
    times["ranker_seconds"] = time.perf_counter() - mark
    # All checkpoint decisions are finished. Recreate the modules and reload
    # the exact selected bytes before touching official test answers.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    del flan, ranker, policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    flan_dir = output / "checkpoint_best/flan_no_g"
    policy_file = output / "B/checkpoints/policy_best.pt"
    ranker_file = output / "B/checkpoint_best/pool_ranker.pt"
    flan = AutoModelForSeq2SeqLM.from_pretrained(
        flan_dir, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(flan_dir, local_files_only=True)
    policy = FullPolicy(len(graph.ids), len(graph.schemas),
                        int(codes.node_codes.max()) + 1).to(device)
    policy.load_state_dict(torch.load(policy_file, map_location=device,
                                      weights_only=True))
    policy.eval()
    rank_state = torch.load(ranker_file, map_location="cpu", weights_only=True)
    ranker = PoolRankerV6(rank_state["node_text"], rank_state["relation_text"],
                          graph.types).to(device)
    ranker.load_state_dict(rank_state)
    ranker.eval()
    if float(ranker.old_scale) != 0.0:
        raise AssertionError("Setting B old score gate must be zero")
    locked = {"flan_sha256": digest(flan_dir / "model.safetensors"),
              "warmup_policy_sha256": digest(output / "checkpoint_best/policy.pt"),
              "setting_b_policy_sha256": digest(policy_file),
              "pool_ranker_sha256": digest(ranker_file),
              "flan_epoch": flan_audit["selected_epoch"],
              "warmup_epoch": warmup_audit["selected_epoch"],
              "setting_b_epoch": setting_b_audit["selected_epoch"],
              "ranker_epoch": ranker_audit["selected_epoch"],
              "test_answers_loaded": False}
    if (locked["flan_sha256"] != flan_audit["model_sha256"] or
        locked["warmup_policy_sha256"] != warmup_audit["checkpoint_sha256"] or
        locked["setting_b_policy_sha256"] != setting_b_audit["checkpoint_sha256"] or
        locked["pool_ranker_sha256"] != ranker_audit["checkpoint_sha256"]):
        raise AssertionError("selected checkpoint hash changed before test")
    dump(output / "checkpoint_lock.json", locked)
    # Program generation uses only question/topic. Answer fields stay deferred.
    mark = time.perf_counter()
    test_unlabelled = graph.splits["test"]
    test_cache = program_cache(flan, tokenizer, codec, graph, test_unlabelled,
                               output / "program_cache/test.jsonl", device)
    graph.load_test_answers()
    test = graph.splits["test"]
    if [q.qid for q in test] != reference["test_qids"]:
        raise AssertionError("test qid order changed after answer load")
    dump(output / "program_cache/test_distribution.json",
         program_distribution(test_cache, test,
             train_zero_rate=flan_audit["selected_program_audit"]["train_zero_answer_rate"],
             zero_factor=config["program_audit"]["flan_top_zero_factor"],
             zero_floor=config["program_audit"]["flan_top_zero_floor"],
             nonzero_beam_floor=config["program_audit"]["flan_nonzero_beam_floor"]))
    test_rows = make_pool_rows(policy, graph, codes, test, test_cache,
                               degrees, names, texts, relation_terms, config)
    dump(output / "test_pool_summary.json", pool_summary(test_rows, test))
    dump(output / "test_pool_candidate_audit.json",
         pool_candidate_audit(test_rows, test))
    qtext, text_ms = encode_texts(
        flan, tokenizer, [input_text(graph, q) for q in test], device,
        max_length=128, individual_timing=True)
    test_metrics, predictions = evaluate_ranker(
        ranker, test_rows, test, qtext, text_ms, graph, device,
        save_predictions=True)
    jsonl(output / "test_predictions.jsonl", predictions)
    dump(output / "test_metrics.json", test_metrics)
    independent = recount(predictions)
    for name, value in independent.items():
        if abs(test_metrics[name] - value) > 1e-9:
            raise AssertionError(f"test metric recount differs: {name}")
    dump(output / "test_independent_recount.json", independent)
    times["test_seconds"] = time.perf_counter() - mark
    audit.update({"status": "COMPLETE", "checkpoint_lock": locked,
                  "test_evaluation_count": 1,
                  "test_after_all_checkpoints_reloaded": True,
                  "test_predictions_sha256": digest(output / "test_predictions.jsonl"),
                  "program_cache_sha256": {name: digest(output / f"program_cache/{name}.jsonl")
                                            for name in ("train", "val", "test")},
                  "stage_seconds": times,
                  "total_seconds": time.perf_counter() - started})
    dump(output / "audit.json", audit)
    print(f"V7 final COMPLETE: test H1={test_metrics['mean_hits_at_1']:.4f} "
          f"R20={test_metrics['mean_recall_at_20']:.4f}", flush=True)


if __name__ == "__main__":
    main()
