#!/usr/bin/env python3
"""Test-only replay of old Setting B weights through V7 final inference code.

This is deliberately separate from the from-scratch training entrypoint.
It verifies architecture and inference compatibility; it never trains or
selects checkpoints, and it needs the old codebook paired with those weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "hop_polo_v7"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=OLD / "inputs")
    p.add_argument("--checkpoint-root", type=Path,
                   default=OLD / "setting_b_remote/hop_polo_v7/B")
    p.add_argument("--reference-output", type=Path,
                   default=OLD / "setting_b_test/download/setting_b_test")
    p.add_argument("--output", type=Path, default=ROOT / "replay_setting_b")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    source = ROOT
    source_manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in source_manifest.items():
        assert sha(source / name) == expected, name
    sys.path.insert(0, str(source / "src"))
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from hopgs.data import ADIntGraph
    from hopgs.codebooks import build_indexes
    from hopgs.model import input_text
    from hop_polo_full_policy import FullPolicy
    from hop_polo_v6_ranker import PoolRankerV6
    from hop_polo_v6 import clean_splits, fingerprint, make_pool_rows, jsonl, dump
    from hop_polo_v2_ranker import encode_texts, evaluate_ranker, word_set

    inputs = args.input_root
    selected = args.checkpoint_root
    lineage = json.loads((args.reference_output / "lineage.json").read_text())
    manifest = json.loads((inputs / "inputs_manifest.json").read_text())
    config = json.loads((source / "configs/full.json").read_text())
    assert config["path_beam"] == 30 and config["local_top_k"] == 60
    assert config["max_pool_nodes"] == 1200 and config["terminal_rescue_slots"] == 6
    graph = ADIntGraph(inputs / "raw", seed=42, defer_test_answers=True)
    assert fingerprint(graph) == manifest["graph_sha256"]
    split = clean_splits(graph, seed=42)
    for part in ("train", "val", "test"):
        assert [x["qid"] for x in split["members"][part]] == manifest[f"{part}_qids"]
    cache = [json.loads(line) for line in (inputs / "program_cache/test.jsonl").open()]
    assert [x["qid"] for x in cache] == [q.qid for q in graph.splits["test"]]
    codes = build_indexes(graph, np.load(inputs / "codebooks/node_codes.npy"),
                          np.load(inputs / "codebooks/edge_codes.npy"))
    policy_file = selected / "checkpoints/policy_best.pt"
    ranker_file = selected / "checkpoint_best/pool_ranker.pt"
    flan_dir = inputs / "checkpoint_best/flan_no_g"
    assert sha(policy_file) == lineage["policy_sha256"]
    assert sha(ranker_file) == lineage["ranker_sha256"]
    assert sha(flan_dir / "model.safetensors") == lineage["flan_sha256"]
    policy = FullPolicy(len(graph.ids), len(graph.schemas),
                        int(codes.node_codes.max()) + 1).cpu()
    policy.load_state_dict(torch.load(policy_file, map_location="cpu", weights_only=True))
    policy.eval()
    state = torch.load(ranker_file, map_location="cpu", weights_only=True)
    ranker = PoolRankerV6(state["node_text"], state["relation_text"], graph.types)
    ranker.load_state_dict(state)
    ranker.eval()
    assert float(ranker.old_scale) == 0.0
    flan = AutoModelForSeq2SeqLM.from_pretrained(flan_dir, local_files_only=True).cpu().eval()
    # Transformers 4.53 expects this same list under additional_special_tokens.
    # tokenizer.json and all token IDs remain byte-for-byte unchanged.
    tokenizer_dir = args.output / "tokenizer_compat"
    if not tokenizer_dir.exists():
        tokenizer_dir.mkdir()
        for path in flan_dir.iterdir():
            if path.name != "model.safetensors":
                shutil.copy2(path, tokenizer_dir / path.name)
        cfg_path = tokenizer_dir / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["additional_special_tokens"] = cfg.pop("extra_special_tokens")
        cfg_path.write_text(json.dumps(cfg))
    assert sha(tokenizer_dir / "tokenizer.json") == sha(flan_dir / "tokenizer.json")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                               use_fast=True)
    assert tokenizer.convert_tokens_to_ids("<HOP>") == 32100
    graph.load_test_answers()
    test = graph.splits["test"][:args.limit or None]
    cached = cache[:len(test)]
    names = [word_set(x) for x in graph.names]
    texts = [word_set(x) for x in graph.texts]
    relation_names = graph.relation_names + [f"inverse of {x}" for x in graph.relation_names]
    relation_terms = [word_set(x) for x in relation_names]
    degrees = torch.as_tensor(np.log1p(np.diff(graph.indptr)) / 16, dtype=torch.float32)
    started = time.perf_counter()
    rows = make_pool_rows(policy, graph, codes, test, cached, degrees,
                          names, texts, relation_terms, config)
    pool_seconds = time.perf_counter() - started
    question_text, text_ms = encode_texts(
        flan, tokenizer, [input_text(graph, q) for q in test], torch.device("cpu"),
        max_length=128, individual_timing=True)
    metrics, predictions = evaluate_ranker(
        ranker, rows, test, question_text, text_ms, graph,
        torch.device("cpu"), save_predictions=True)
    dump(args.output / "test_metrics.json", metrics)
    jsonl(args.output / "test_predictions.jsonl", predictions)
    original = {row["qid"]: row for row in map(json.loads,
        (args.reference_output / "test_predictions.jsonl").open())}
    comparison = {"queries": len(predictions),
                  "top1_identical": sum(row["ranked_top20"][0] ==
                      original[row["qid"]]["ranked_top20"][0] for row in predictions),
                  "top20_identical": sum(row["ranked_top20"] ==
                      original[row["qid"]]["ranked_top20"] for row in predictions),
                  "pool_identical": sum(row["pool_nodes"] ==
                      original[row["qid"]]["pool_nodes"] for row in predictions),
                  "selected_identical": sum(row["selected_nodes"] ==
                      original[row["qid"]]["selected_nodes"] for row in predictions),
                  "ranked_node_id_order_identical": sum(
                      [x["node_id"] for x in row["ranked_nodes"]] ==
                      [x["node_id"] for x in original[row["qid"]]["ranked_nodes"]]
                      for row in predictions),
                  "pool_seconds": pool_seconds,
                  "total_seconds": time.perf_counter() - started,
                  "source_manifest_sha256": sha(source / "source_manifest.json"),
                  "policy_sha256": sha(policy_file), "ranker_sha256": sha(ranker_file),
                  "flan_sha256": sha(flan_dir / "model.safetensors"),
                  "tokenizer_json_unchanged": True,
                  "codebase": "hop_polo_v7_final/src",
                  "training_checkpoint_input": False,
                  "runtime": "CPU, Transformers 4.53.2 tokenizer metadata adaptation"}
    dump(args.output / "comparison.json", comparison)
    print(json.dumps({"hits_at_1": metrics["mean_hits_at_1"],
                      "recall_at_20": metrics["mean_recall_at_20"],
                      **comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
