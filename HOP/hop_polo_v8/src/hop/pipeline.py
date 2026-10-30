"""Điều phối HOP V8 cho một bộ dữ liệu: codebook -> Flan -> chương trình -> warm-up -> Setting B -> ranker -> test.

Quy tắc chọn checkpoint giữ đúng V7: mỗi pha tối đa `epochs` epoch cập nhật, epoch 0 cũng được chấm và
được chọn, chọn Hits@1 val cao nhất (proxy điểm lạnh với Flan/policy), hoà thì lấy epoch sớm hơn, dừng sau
`patience` epoch liên tiếp không tăng ngặt. Test chỉ chấm đúng một lần, sau khi mọi checkpoint đã được nạp
lại từ byte trên đĩa và khoá hash.

Bỏ so với V7 (không đổi kết quả): epoch 0 của QA-SFT, warm-up và Setting B trùng hệt dòng đã chọn của pha
trước (cùng trọng số, cùng chương trình val, cùng policy proxy) nên được chép lại thay vì chấm lại; chỉ giữ
checkpoint `best` và `last` của mỗi pha (hash của mọi epoch vẫn nằm trong lịch sử).

Nối phiên: mỗi stage ghi `done.json` khi xong; các pha có vòng epoch lưu trạng thái đầy đủ (trọng số,
optimizer, RNG, lịch sử) sau mỗi epoch; các bước dài theo từng câu (nhãn, giải mã, pool) lưu từng phần.
Gần hạn phiên (`--deadline-hours`), pipeline dừng ở ranh giới an toàn với trạng thái PAUSED; phiên sau chạy
lại cùng lệnh với `--resume` trên bản sao output cũ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .audit import aggregate, check_recount, finite_predictions, policy_program_mismatch
from .codebook import load_codebook, train_codebook
from .common import digest, dump, jsonl, read_jsonl, seed_all, word_set
from .data import Graph
from .flan import decode_programs, initialize_program_tokens, program_distribution, train_batch, train_qa_batch
from .labels import SUPPORTED_HOPS, executable_qa_labels, mine_train_routes, teacher_paths
from .policy import FullPolicy, GraphTensors, question_context, teacher_loss
from .program import EdgeIndex, Executor, ProgramCodec, input_text, make_synthetic_examples
from .ranker import (PoolRanker, collect_pool, cold_numeric_score, encode_texts, evaluate_ranker,
                     pool_summary, train_epoch, training_labels)
from .setting_b import question_path_loss, train_soft_hard_gap

SEED_OFFSET = {"codebook": 0, "policy_init": 1, "flan_tokens": 2, "flan_synthetic": 11, "flan_qa": 12,
               "warmup": 21, "setting_b": 31, "ranker": 41}
PHASES = ("synthetic", "qa_sft", "policy_warmup", "policy", "pool_ranker")


class Pause(Exception):
    """Hết giờ phiên ở một ranh giới an toàn; trạng thái đã được lưu."""


class Clock:
    def __init__(self, hours: float | None):
        self.deadline = time.time() + hours * 3600 if hours else None

    def need(self, seconds: float, where: str) -> None:
        if self.deadline is not None and time.time() + seconds > self.deadline:
            raise Pause(where)


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    return merge(load_config(path.parent / parent), config) if parent else config


def check_config(config: dict) -> None:
    if config["protocol"] != "hop-polo-v8":
        raise ValueError("wrong protocol")
    if config["max_hops"] not in SUPPORTED_HOPS:
        raise ValueError(f"max_hops must be one of {SUPPORTED_HOPS}")
    if config["graph"]["inverse"] not in ("missing", "always"):
        raise ValueError("graph.inverse must be missing|always")
    if config["graph"]["action"] not in ("schema", "target"):
        raise ValueError("graph.action must be schema|target")
    quotas = config["pool_depth_quotas"]
    if len(quotas) != config["max_hops"] or min(quotas) < 0 or sum(quotas) > 1 + 1e-9:
        raise ValueError("pool_depth_quotas: one non-negative share per hop, sum <= 1")
    if not isinstance(config["pool_last_hop"], bool):
        raise ValueError("pool_last_hop must be true|false")
    for phase in PHASES:
        if config[phase]["epochs"] < 1 or config[phase]["patience"] < 1:
            raise ValueError(f"{phase} needs >= 1 epoch and patience")
    if len(config["temperature_by_epoch"]) < config["policy"]["epochs"]:
        raise ValueError("temperature schedule shorter than Setting B epochs")
    if config["path_beam"] - config["terminal_rescue_slots"] < config["program_beam"]:
        raise ValueError("path beam too narrow for program coverage")
    if not isinstance(config["flan"]["untie_lm_head"], bool):
        raise ValueError("flan.untie_lm_head must be true|false")
    if config["selection"]["tie_break"] not in TIE_BREAKS:
        raise ValueError(f"selection.tie_break must be one of {TIE_BREAKS}")


def runtime_versions(device: torch.device) -> dict:
    """Phiên bản thực chạy (vd hành vi buộc lm_head phụ thuộc phiên bản transformers)."""
    from importlib import metadata
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "sentence-transformers",
                    "sentencepiece", "numpy", "scipy", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["cuda_device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return versions


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


TIE_BREAKS = ("earliest", "recall_at_20")


def select_hits(rows, tie_break: str = "earliest"):
    """Hits@1 val cao nhất; chỉ xét dòng hợp lệ kỹ thuật. Hoà Hits@1: `earliest` (V7) lấy epoch sớm hơn,
    `recall_at_20` lấy Recall@20 cao hơn rồi mới tới epoch sớm hơn. Early stopping dùng cùng thứ tự này."""
    valid = [row for row in rows if row.get("technical_valid", False)]
    if tie_break not in TIE_BREAKS:
        raise ValueError(tie_break)

    def key(row):
        v = row["validation"]
        return v["mean_hits_at_1"], v["mean_recall_at_20"] if tie_break == "recall_at_20" else 0.0, -row["epoch"]
    return max(valid, key=key) if valid else None


def with_teacher(order, teacher) -> list[int]:
    """Thứ tự câu train của một epoch, bỏ câu không có teacher; RNG vẫn rút như khi đủ câu."""
    return [int(index) for index in order if teacher[int(index)]]


def merge_label_summaries(parts: list[dict]) -> dict:
    summary = dict(parts[0])
    for part in parts[1:]:
        for key, value in part.items():
            if isinstance(value, int) and not isinstance(value, bool) and key not in ("label_edge_cap", "max_hops",
                                                                                     "programs_per_answer"):
                summary[key] = summary.get(key, 0) + value
            elif key in ("shortest_hops", "label_hops"):
                summary[key] = {k: summary[key].get(k, 0) + v for k, v in value.items()}
    return summary


class Pipeline:
    def __init__(self, args, config: dict):
        self.args, self.config = args, config
        self.out: Path = args.output
        self.device = torch.device(args.device)
        self.clock = Clock(args.deadline_hours)
        self.seed = config["seed"]
        self.audit_path = self.out / "audit.json"
        self.audit = json.loads(self.audit_path.read_text()) if self.audit_path.exists() else {}

    # ---------- trạng thái chung ----------
    def save_audit(self, **values) -> None:
        self.audit.update(values)
        dump(self.audit_path, self.audit)

    def done(self, stage: str) -> dict | None:
        path = self.out / stage / "done.json"
        return json.loads(path.read_text()) if path.exists() else None

    def mark_done(self, stage: str, payload: dict) -> None:
        dump(self.out / stage / "done.json", payload)
        self.save_audit(**{stage: payload})

    def rows_cached(self, path: Path, questions, build, label: str, every: int = 200):
        """Dựng từng câu, lưu từng phần mỗi `every` câu; dừng an toàn khi còn dưới 15 phút."""
        qids = [q.qid for q in questions]
        if path.exists():
            saved = pickle.loads(path.read_bytes())
            if saved["qids"] != qids:
                raise AssertionError(f"cached rows differ from questions: {path}")
            return saved["rows"]
        partial = path.with_suffix(".partial.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if partial.exists():
            saved = pickle.loads(partial.read_bytes())
            if saved["qids"] == qids[:len(saved["rows"])]:
                rows = saved["rows"]
        for index in range(len(rows), len(questions)):
            rows.append(build(questions[index]))
            if (index + 1) % every == 0 and index + 1 < len(questions):
                partial.write_bytes(pickle.dumps({"qids": qids[:len(rows)], "rows": rows}))
                print(f"{label} {index + 1}/{len(questions)}", flush=True)
                self.clock.need(900, f"{label} {index + 1}/{len(questions)}")
        path.write_bytes(pickle.dumps({"qids": qids, "rows": rows}))
        partial.unlink(missing_ok=True)
        return rows

    def train_phase(self, name: str, directory: Path, *, settings: dict, optimizer, rng, train_one, evaluate,
                    save_model, load_model, checkpoint_file: str, epoch0: dict | None = None):
        """Vòng epoch V7, nối phiên được. Trả (dòng đã chọn, lịch sử); model được nạp lại checkpoint đã chọn."""
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state.json"
        tie_break = self.config["selection"]["tie_break"]

        def checkpoint(row, history):
            last = directory / "last"
            shutil.rmtree(last, ignore_errors=True)
            save_model(last)
            row["checkpoint_sha256"] = digest(last / checkpoint_file)
            best = select_hits(history + [row], tie_break)
            if best is row:
                shutil.rmtree(directory / "best", ignore_errors=True)
                shutil.copytree(last, directory / "best")

        def persist(history, stale, updates, finished):
            torch.save(optimizer.state_dict(), directory / "last_optimizer.pt")
            torch.save({"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                       directory / "last_rng.pt")
            dump(state_path, {"phase": name, "history": history, "stale": stale, "updates": updates,
                              "finished": finished, "numpy_rng": rng.bit_generator.state})
            dump(directory / "history.json", history)

        if state_path.exists():
            state = json.loads(state_path.read_text())
            load_model(directory / "last")
            optimizer.load_state_dict(torch.load(directory / "last_optimizer.pt", map_location=self.device,
                                                 weights_only=False))
            saved_rng = torch.load(directory / "last_rng.pt", weights_only=False)
            torch.set_rng_state(saved_rng["torch"])
            if saved_rng["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved_rng["cuda"])
            rng.bit_generator.state = state["numpy_rng"]
            history, stale, updates, finished = state["history"], state["stale"], state["updates"], state["finished"]
            print(f"{name}: resume after epoch {history[-1]['epoch']}", flush=True)
        else:
            started = time.perf_counter()
            row = dict(epoch0) if epoch0 is not None else evaluate(0)
            row.update(epoch=0, updates=0, seconds=time.perf_counter() - started)
            history, stale, updates, finished = [], 0, 0, False
            checkpoint(row, history)
            history.append(row)
            persist(history, stale, updates, finished)
        epoch = history[-1]["epoch"]
        while not finished:
            if epoch >= settings["epochs"] or stale >= settings["patience"]:
                finished = True
                persist(history, stale, updates, finished)
                break
            durations = [row["seconds"] for row in history if row["epoch"] > 0]
            self.clock.need((max(durations[-1:] or [0]) * 1.15) + 600, f"{name} epoch {epoch + 1}")
            started = time.perf_counter()
            epoch += 1
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            info, steps = train_one(epoch)
            updates += steps
            trained = time.perf_counter()
            row = evaluate(epoch)
            row.update(epoch=epoch, updates=updates, train=info, seconds=time.perf_counter() - started,
                       train_seconds=trained - started, eval_seconds=time.perf_counter() - trained,
                       cuda_peak_gb=(torch.cuda.max_memory_allocated(self.device) / 1e9
                                     if self.device.type == "cuda" else None))
            checkpoint(row, history)
            history.append(row)
            best = select_hits(history, tie_break)
            stale = 0 if best is not None and best["epoch"] == epoch else stale + 1
            v = row["validation"]
            print(f"{name} epoch {epoch}: H1={v['mean_hits_at_1']:.4f} R20={v['mean_recall_at_20']:.4f} "
                  f"best={best['epoch'] if best else None} stale={stale}", flush=True)
            persist(history, stale, updates, finished)
        selected = select_hits(history, tie_break)
        if selected is None:
            raise RuntimeError(f"no technically valid {name} checkpoint")
        if digest(directory / "best" / checkpoint_file) != selected["checkpoint_sha256"]:
            raise AssertionError(f"{name}: best checkpoint differs from selected epoch")
        load_model(directory / "best")
        return selected, history

    # ---------- chuẩn bị ----------
    def prepare(self) -> None:
        args, config = self.args, self.config
        bundle_manifest = digest(args.input / "manifest.json")
        models = {}
        for name, directory in (("flan", args.flan_dir), ("embedding", args.embedding_dir)):
            actual = digest(directory / "model.safetensors")
            if actual != config["models"][name]["model_sha256"]:
                raise AssertionError(f"{name} model hash differs from config")
            models[name] = actual
        source = digest(args.source_manifest) if args.source_manifest else None
        identity = {"config_sha256": canonical_sha(config), "bundle_manifest_sha256": bundle_manifest,
                    "models": models, "source_manifest_sha256": source}
        run_path = self.out / "run.json"
        if run_path.exists():
            if not args.resume:
                raise RuntimeError("output directory already holds a run; pass --resume to continue it")
            previous = json.loads(run_path.read_text())
            if previous["identity"] != identity:
                raise AssertionError("resume: config, bundle, model or source differs from the paused run")
        elif self.out.exists() and any(self.out.iterdir()):
            raise RuntimeError("a new run needs an empty output directory")
        else:
            dump(run_path, {"identity": identity, "config": config, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.graph = Graph(args.input, inverse=config["graph"]["inverse"], action=config["graph"]["action"])
        if config.get("limits"):
            self.graph.limit_splits(config["limits"])
        self.train, self.val = self.graph.splits["train"], self.graph.splits["val"]
        if any(not q.topics for q in self.train + self.val):
            raise AssertionError("train/val questions need topics")
        self.save_audit(status="RUNNING", identity=identity, environment=runtime_versions(self.device),
                        graph=self.graph.audit(),
                        graph_fingerprint=self.graph.fingerprint(), test_answers_loaded=False,
                        training_labels_split="QA train only", checkpoint_selection_split="validation only")

    # ---------- stage 1: codebook ----------
    def stage_codebook(self) -> None:
        directory = self.out / "codebook"
        if not self.done("codebook"):
            seed_all(self.seed + SEED_OFFSET["codebook"])
            _, audit = train_codebook(self.graph, directory, self.config["codebook"], model_dir=str(self.args.embedding_dir),
                                      seed=self.seed, device=self.args.device)
            self.mark_done("codebook", audit)
        self.codebook = load_codebook(directory)
        if digest(directory / "node_codes.npy") != self.done("codebook")["node_codes_sha256"]:
            raise AssertionError("codebook changed on disk")
        graph, config = self.graph, self.config
        self.gt = GraphTensors(graph, self.codebook, self.device)
        relation_names = graph.relation_names + [f"inverse of {x}" for x in graph.relation_names]
        self.relation_names = relation_names
        self.terms = ([word_set(x) for x in graph.names], [word_set(x) for x in graph.texts],
                      [word_set(x) for x in relation_names])
        index = EdgeIndex(graph)
        self.decode_executor = Executor(graph, self.codebook, 10 ** 9, prefix_index=True, index=index)
        self.label_executor = Executor(graph, self.codebook, config["label_edge_cap"], prefix_index=False, index=index)
        init = self.out / "policy_init.pt"
        if not init.exists():
            seed_all(self.seed + SEED_OFFSET["policy_init"])
            torch.save(self.new_policy().state_dict(), init)
        self.proxy_policy = self.load_policy(init)

    def new_policy(self) -> FullPolicy:
        return FullPolicy(len(self.graph.ids), len(self.graph.schemas), self.codebook.size,
                          self.config["max_hops"]).to(self.device)

    def load_policy(self, path: Path) -> FullPolicy:
        policy = self.new_policy()
        policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return policy.eval()

    def pool_options(self) -> dict:
        c = self.config
        return {"path_beam": c["path_beam"], "local_k": c["local_top_k"], "rescue_slots": c["terminal_rescue_slots"],
                "max_nodes": c["max_pool_nodes"], "depth_quotas": c["pool_depth_quotas"], "last_hop": c["pool_last_hop"]}

    def pool_row(self, policy, q, cache):
        return collect_pool(policy, self.gt, self.graph, q, cache, question_context(self.gt, q), self.terms,
                            **self.pool_options())

    def proxy_validation(self, policy, questions, cache):
        """Bộ chọn số cố định trên pool trước khi cắt dựng từ chương trình dự đoán (như V7)."""
        policy.eval()
        graph = self.graph
        records, diagnostics = [], []
        with torch.inference_mode():
            for q, beam in zip(questions, cache):
                row = self.pool_row(policy, q, beam)
                scores = cold_numeric_score(row.numeric)
                order = np.lexsort((graph.ids[row.nodes.astype(np.int64)], -scores))
                top = [int(row.nodes[j]) for j in order[:20]]
                selected = set(top) | set(q.topics)
                for j in order[:20]:
                    selected.update(int(node) for node in row.path_nodes[j] if node >= 0)
                gold = set(q.answers)
                hit_set, hit_top = len(gold & selected), len(gold & set(top))
                records.append({"qid": q.qid, "split": q.split, "answers": [int(graph.ids[n]) for n in q.answers],
                                "topics": [int(graph.ids[n]) for n in q.topics],
                                "pool_ranked": [int(graph.ids[row.nodes[j]]) for j in order],
                                "selected_nodes": [int(graph.ids[n]) for n in sorted(selected)],
                                "ranked_top20": [int(graph.ids[n]) for n in top],
                                "hits_at_1": float(bool(top) and top[0] in gold), "recall_at_20": hit_top / len(gold),
                                "recall_set": hit_set / len(gold), "precision_set": hit_set / len(selected),
                                "f2_set": 5 * hit_set / (4 * len(gold) + len(selected)),
                                "selected_nodes_count": len(selected),
                                "latency_ms": row.generation_ms + row.search_ms})
                diagnostics.append(row.search_diagnostics)
        metrics = {"queries": len(records)}
        for key in ("hits_at_1", "recall_at_20", "recall_set", "precision_set", "f2_set", "selected_nodes_count",
                    "latency_ms"):
            metrics[f"mean_{key}"] = float(np.mean([row[key] for row in records]))
        metrics["p95_selected_nodes_count"] = float(np.percentile([r["selected_nodes_count"] for r in records], 95))
        summary = {"mean_beam_oracle_recall_at_20": float(np.mean([x["oracle_recall_at_20"] for x in diagnostics])),
                   "mean_pool_recall_drop_from_cap": float(np.mean([
                       (x["gold_in_uncapped_pool"] - x["gold_in_capped_pool"]) / x["gold_total"] for x in diagnostics])),
                   # số đáp án bị trần pool loại, theo độ sâu (để chỉnh pool_depth_quotas bằng số đo trên val)
                   "pool_cap_drop_by_depth": {depth: sum(x["hard"].get(f"answer_{depth}_cap_drop", 0) for x in diagnostics)
                                              for depth in ("1hop", "2hop", "3plus_hop")},
                   "mean_beam_nodes": float(np.mean([x["beam_nodes"] for x in diagnostics])),
                   "mean_pool_nodes": float(np.mean([x["pool_nodes"] for x in diagnostics])),
                   "mean_terminal_mass": float(np.mean([x["terminal_probability_mass"] for x in diagnostics])),
                   "depth_gold": [sum(x["gold_by_depth"][d] for x in diagnostics) for d in range(self.config["max_hops"])],
                   "depth_candidates": [sum(x["candidate_by_depth"][d] for x in diagnostics) for d in range(self.config["max_hops"])],
                   "topic_degree_buckets": {bucket: {
                       "queries": sum(x["topic_degree_bucket"] == bucket for x in diagnostics),
                       "beam_gold": sum(x["gold_in_candidates"] for x in diagnostics if x["topic_degree_bucket"] == bucket),
                       "gold_total": sum(x["gold_total"] for x in diagnostics if x["topic_degree_bucket"] == bucket)}
                       for bucket in ("<=116", "117-498", ">498")},
                   "program_slots": [x["program_slots"] for x in diagnostics],
                   "per_question": diagnostics}
        return metrics, records, summary

    # ---------- stage 2: Flan ----------
    def load_flan(self, directory: Path):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(directory, local_files_only=True).to(self.device)
        return model, tokenizer

    def qa_labels(self, codec, directory: Path):
        """Nhãn chương trình QA, lưu theo khúc 250 câu để nối phiên trên PrimeKG."""
        config = self.config
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "qa_labels.pkl"
        if final.exists():
            return pickle.loads(final.read_bytes())
        parts, step = [], 250
        for start in range(0, len(self.train), step):
            part_path = directory / f"qa_labels_{start:06d}.pkl"
            if part_path.exists():
                parts.append(pickle.loads(part_path.read_bytes()))
                continue
            self.clock.need(900, f"qa labels {start}/{len(self.train)}")
            part = executable_qa_labels(self.graph, self.codebook, self.label_executor, codec,
                                        self.train[start:start + step], cap=config["label_edge_cap"],
                                        programs_per_answer=config["qa_sft"]["programs_per_answer"],
                                        paths_per_depth=config["qa_sft"]["paths_per_depth"],
                                        max_hops=config["max_hops"])
            part_path.write_bytes(pickle.dumps(part))
            parts.append(part)
            print(f"qa labels {min(start + step, len(self.train))}/{len(self.train)}", flush=True)
        plans = [plan for part in parts for plan in part[0]]
        summary = merge_label_summaries([part[1] for part in parts])
        per_answer = [row for part in parts for row in part[2]]
        result = (plans, summary, per_answer)
        final.write_bytes(pickle.dumps(result))
        for path in directory.glob("qa_labels_*.pkl"):
            path.unlink()
        dump(directory / "qa_label_audit.json", summary)
        jsonl(directory / "qa_label_per_answer.jsonl", per_answer)
        return result

    def stage_flan(self) -> None:
        if self.done("flan"):
            return
        from transformers import AutoModelForSeq2SeqLM
        config, graph, device = self.config, self.graph, self.device
        root = self.out / "flan"
        base = root / "base"
        if not (base / "model.safetensors").exists():
            model, tokenizer = self.load_flan(self.args.flan_dir)
            old_vocab = len(tokenizer)
            ProgramCodec(tokenizer, graph, self.decode_executor, config["max_hops"])
            token_audit = initialize_program_tokens(model, tokenizer, old_vocab, seed=self.seed + SEED_OFFSET["flan_tokens"],
                                                    untie_lm_head=config["flan"]["untie_lm_head"])
            model.save_pretrained(base, safe_serialization=True)
            tokenizer.save_pretrained(base)
            dump(root / "token_initialization.json", token_audit)
            del model
        model, tokenizer = self.load_flan(base)
        codec = ProgramCodec(tokenizer, graph, self.decode_executor, config["max_hops"])
        plans, label_audit, _ = self.qa_labels(codec, root / "labels")
        train_zero_rate = label_audit["shortest_hops"]["0"] / max(1, label_audit["answers"])
        audit_cfg = config["program_audit"]
        synthetic = make_synthetic_examples(graph, self.codebook, codec, config["synthetic"]["nodes"],
                                            config["synthetic"]["edges"], self.seed)
        dump(root / "synthetic_audit.json",
             {"examples": len(synthetic), "source": "KG only",
              "read_s_examples": sum(ids[0] == codec.id["<READ_S>"] for _t, ids in synthetic),
              "read_g_examples": sum(ids[0] == codec.id["<READ_G>"] for _t, ids in synthetic)})

        def evaluate_with(model_ref, phase):
            def evaluate(phase_epoch):
                tag = f"epoch_{phase_epoch:02d}"
                cached = decode_programs(model_ref(), tokenizer, codec, graph, self.val,
                                         root / phase / "val_cache" / f"{tag}.jsonl", device,
                                         beam=config["program_beam"], length_penalty=config["decoder_length_penalty"])
                metrics, predictions, diagnostics = self.proxy_validation(self.proxy_policy, self.val, cached)
                jsonl(root / phase / "val_predictions" / f"{tag}.jsonl", predictions)
                distribution = program_distribution(cached, self.val, max_hops=config["max_hops"],
                                                    train_zero_rate=train_zero_rate,
                                                    zero_factor=audit_cfg["flan_top_zero_factor"],
                                                    zero_floor=audit_cfg["flan_top_zero_floor"],
                                                    nonzero_beam_floor=audit_cfg["flan_nonzero_beam_floor"])
                return {"phase": phase, "validation": metrics, "program_audit": distribution,
                        "candidate_diagnostics": diagnostics,
                        "technical_valid": finite_predictions(metrics, predictions, len(self.val)),
                        "val_cache": f"{phase}/val_cache/{tag}.jsonl"}
            return evaluate

        holder = {"model": model}

        def save_model(path):
            holder["model"].save_pretrained(path, safe_serialization=True)
            tokenizer.save_pretrained(path)

        def load_model(path):
            # Nạp TẠI CHỖ: optimizer giữ tham chiếu tới tham số của model hiện có.
            loaded = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
            holder["model"].load_state_dict(loaded.state_dict())
            del loaded

        # 2a. synthetic: đọc mã S của nút và schema của chuyển cạnh, chỉ từ KG
        seed_all(self.seed + SEED_OFFSET["flan_synthetic"])
        syn = config["synthetic"]
        optimizer = torch.optim.AdamW(holder["model"].parameters(), lr=syn["learning_rate"])
        rng = np.random.default_rng(self.seed + SEED_OFFSET["flan_synthetic"])

        def train_synthetic(_epoch):
            order = rng.permutation(len(synthetic))
            losses = []
            for start in range(0, len(order), syn["batch_size"]):
                holder["model"].train()
                losses.append(train_batch(holder["model"], tokenizer,
                                          [synthetic[int(i)] for i in order[start:start + syn["batch_size"]]],
                                          optimizer, device))
            return {"train_loss": float(np.mean(losses)), "train_examples": len(order)}, len(losses)

        selected_syn, _ = self.train_phase("flan synthetic", root / "synthetic", settings=syn, optimizer=optimizer,
                                           rng=rng, train_one=train_synthetic,
                                           evaluate=evaluate_with(lambda: holder["model"], "synthetic"),
                                           save_model=save_model, load_model=load_model,
                                           checkpoint_file="model.safetensors")
        # 2b. QA SFT từ checkpoint synthetic đã chọn, optimizer mới; epoch 0 = dòng synthetic đã chọn
        del optimizer
        load_model(root / "synthetic" / "best")
        qa = config["qa_sft"]
        usable = np.asarray([i for i, answers in enumerate(plans) if any(answers)], dtype=np.int64)
        if not len(usable):
            raise RuntimeError("no train QA has an executable label within the edge cap")
        seed_all(self.seed + SEED_OFFSET["flan_qa"])
        optimizer = torch.optim.AdamW(holder["model"].parameters(), lr=qa["learning_rate"])
        rng = np.random.default_rng(self.seed + SEED_OFFSET["flan_qa"])

        def train_qa(epoch):
            order = rng.permutation(usable)
            losses, answers = [], 0
            for start in range(0, len(order), qa["batch_size"]):
                holder["model"].train()
                loss, count = train_qa_batch(holder["model"], tokenizer, graph, self.train, plans,
                                             order[start:start + qa["batch_size"]], epoch, optimizer, device)
                losses.append(loss)
                answers += count
            assert answers == label_audit["answers_with_labels"]
            return {"train_loss": float(np.mean(losses)), "train_examples": len(order),
                    "train_answer_examples": answers}, len(losses)

        epoch0 = {**{k: v for k, v in selected_syn.items() if k not in ("epoch", "updates", "seconds", "train",
                                                                        "checkpoint_sha256")},
                  "phase": "qa_sft", "reused_from": f"synthetic epoch {selected_syn['epoch']}"}
        selected_qa, _ = self.train_phase("flan qa_sft", root / "qa_sft", settings=qa, optimizer=optimizer, rng=rng,
                                          train_one=train_qa, evaluate=evaluate_with(lambda: holder["model"], "qa_sft"),
                                          save_model=save_model, load_model=load_model,
                                          checkpoint_file="model.safetensors", epoch0=epoch0)
        selected_dir = root / "selected"
        shutil.rmtree(selected_dir, ignore_errors=True)
        shutil.copytree(root / "qa_sft" / "best", selected_dir)
        shutil.copy2(root / selected_qa["val_cache"], self.out / "programs_val.jsonl")
        self.mark_done("flan", {"synthetic_selected_epoch": selected_syn["epoch"],
                                "qa_sft_selected_epoch": selected_qa["epoch"],
                                "selected_row_reused_from": selected_qa.get("reused_from"),
                                "model_sha256": digest(selected_dir / "model.safetensors"),
                                "val_cache": selected_qa["val_cache"], "validation": selected_qa["validation"],
                                "program_audit": selected_qa["program_audit"],
                                "train_zero_answer_rate": train_zero_rate, "qa_label_audit": label_audit,
                                "synthetic_examples": len(synthetic)})

    # ---------- stage 3: chương trình train/val từ Flan đã chọn ----------
    def stage_programs(self) -> None:
        directory = self.out / "programs"
        if not self.done("programs"):
            model, tokenizer = self.load_flan(self.out / "flan" / "selected")
            codec = ProgramCodec(tokenizer, self.graph, self.decode_executor, self.config["max_hops"])
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.out / "programs_val.jsonl", directory / "val.jsonl")
            train_cache = decode_programs(model, tokenizer, codec, self.graph, self.train, directory / "train.jsonl",
                                          self.device, beam=self.config["program_beam"],
                                          length_penalty=self.config["decoder_length_penalty"],
                                          checkpoint=lambda done: self.clock.need(900, f"decode train {done}"))
            flan_done = self.done("flan")
            a = self.config["program_audit"]
            distribution = program_distribution(train_cache, self.train, max_hops=self.config["max_hops"],
                                                train_zero_rate=flan_done["train_zero_answer_rate"],
                                                zero_factor=a["flan_top_zero_factor"], zero_floor=a["flan_top_zero_floor"],
                                                nonzero_beam_floor=a["flan_nonzero_beam_floor"])
            dump(directory / "train_distribution.json", distribution)
            self.mark_done("programs", {"train_sha256": digest(directory / "train.jsonl"),
                                        "val_sha256": digest(directory / "val.jsonl"),
                                        "train_distribution": distribution})
        self.train_cache = read_jsonl(directory / "train.jsonl")
        self.val_cache = read_jsonl(directory / "val.jsonl")
        assert [r["qid"] for r in self.train_cache] == [q.qid for q in self.train]
        assert [r["qid"] for r in self.val_cache] == [q.qid for q in self.val]

    def teacher(self):
        path = self.out / "labels" / "teacher.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        result = teacher_paths(self.graph, self.train, self.config["max_hops"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(result))
        dump(self.out / "labels" / "teacher_depths.json", result[1])
        return result

    def policy_holder(self, init: Path):
        holder = {"policy": self.load_policy(init)}
        for parameter in holder["policy"].parameters():
            parameter.requires_grad_(True)

        def save_model(path):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(holder["policy"].state_dict(), path / "policy.pt")

        def load_model(path):
            holder["policy"].load_state_dict(torch.load(path / "policy.pt", map_location=self.device, weights_only=True))
        return holder, save_model, load_model

    def policy_eval(self, holder, directory: Path, extra=None):
        def evaluate(epoch):
            metrics, predictions, diagnostics = self.proxy_validation(holder["policy"], self.val, self.val_cache)
            jsonl(directory / "val_predictions" / f"epoch_{epoch:02d}.jsonl", predictions)
            row = {"validation": metrics, "candidate_diagnostics": diagnostics,
                   "technical_valid": finite_predictions(metrics, predictions, len(self.val))}
            if extra:
                row.update(extra(epoch, row))
            return row
        return evaluate

    # ---------- stage 4: warm-up policy ----------
    def stage_warmup(self) -> None:
        if self.done("warmup"):
            return
        directory = self.out / "warmup"
        settings = self.config["policy_warmup"]
        teacher, depths = self.teacher()
        holder, save_model, load_model = self.policy_holder(self.out / "policy_init.pt")
        seed_all(self.seed + SEED_OFFSET["warmup"])
        optimizer = torch.optim.AdamW(holder["policy"].parameters(), lr=settings["learning_rate"])
        rng = np.random.default_rng(self.seed + SEED_OFFSET["warmup"])
        weight = self.config["loss"]["node_rank"]

        def train_one(epoch):
            policy = holder["policy"]
            policy.train()
            order = with_teacher(rng.permutation(len(self.train)), teacher)
            optimizer.zero_grad(set_to_none=True)
            losses, pending, steps = [], 0, 0
            for pos, index in enumerate(order, 1):
                q = self.train[index]
                chosen = teacher[index][(epoch - 1 + index) % len(teacher[index])]
                ce, rank = teacher_loss(policy, self.gt, self.graph, self.codebook, q, chosen,
                                        question_context(self.gt, q), self.device)
                loss = ce + weight * rank
                (loss / settings["batch_size"]).backward()
                pending += 1
                losses.append(float(loss.detach()))
                if pending == settings["batch_size"] or pos == len(order):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    steps += 1
            return {"train_loss": float(np.mean(losses)), "train_examples": len(order)}, steps

        flan_done = self.done("flan")
        flan_history = json.loads((self.out / "flan" / "qa_sft" / "history.json").read_text())
        flan_row = next(row for row in flan_history if row["epoch"] == flan_done["qa_sft_selected_epoch"])
        epoch0 = {"validation": flan_row["validation"], "candidate_diagnostics": flan_row["candidate_diagnostics"],
                  "technical_valid": flan_row["technical_valid"],
                  "reused_from": f"flan {flan_done['val_cache']} (same untrained policy, same val programs)"}
        selected, _ = self.train_phase("policy warm-up", directory, settings=settings, optimizer=optimizer, rng=rng,
                                       train_one=train_one, evaluate=self.policy_eval(holder, directory),
                                       save_model=save_model, load_model=load_model, checkpoint_file="policy.pt",
                                       epoch0=epoch0)
        mismatch = policy_program_mismatch(self.graph, self.codebook, self.train, teacher, self.train_cache)
        dump(directory / "policy_program_mismatch.json", mismatch)
        self.mark_done("warmup", {"selected_epoch": selected["epoch"], "selected_updates": selected["updates"],
                                  "teacher_answers_without_route": depths.get("none", 0),
                                  "train_questions_without_teacher": sum(not choices for choices in teacher),
                                  "checkpoint_sha256": digest(directory / "best" / "policy.pt"),
                                  "validation": selected["validation"], "program_mismatch_audit": mismatch})

    # ---------- stage 5: Setting B ----------
    def stage_setting_b(self) -> None:
        if self.done("setting_b"):
            return
        config = self.config
        directory = self.out / "setting_b"
        settings = config["policy"]
        teacher, depths = self.teacher()
        routes, route_audit = mine_train_routes(self.graph, self.train, config["route_cap_per_answer"], self.seed)
        dump(directory / "label_audits" / "train_route_mining.json", route_audit)
        dump(directory / "label_audits" / "teacher_depths.json", depths)
        warmup_best = self.out / "warmup" / "best" / "policy.pt"
        holder, save_model, load_model = self.policy_holder(warmup_best)
        seed_all(self.seed + SEED_OFFSET["setting_b"])
        optimizer = torch.optim.AdamW(holder["policy"].parameters(), lr=settings["learning_rate"])
        rng = np.random.default_rng(self.seed + SEED_OFFSET["setting_b"])
        temperatures = config["temperature_by_epoch"]
        w = config["loss"]

        def extra(epoch, row):
            return {"temperature": temperatures[epoch - 1] if epoch else None,
                    "hard_coverage": aggregate([x["hard"] for x in row["candidate_diagnostics"]["per_question"]]),
                    "train_soft_hard_gap": train_soft_hard_gap(holder["policy"], self.gt, self.train, self.train_cache,
                                                               routes, config, temperatures[max(0, epoch - 1)])}

        def train_one(epoch):
            policy = holder["policy"]
            policy.train()
            order = with_teacher(rng.permutation(len(self.train)), teacher)
            optimizer.zero_grad(set_to_none=True)
            component, pending, steps = Counter(), 0, 0
            tau = temperatures[epoch - 1]
            for pos, index in enumerate(order, 1):
                q = self.train[index]
                qctx = question_context(self.gt, q)
                chosen = teacher[index][(epoch - 1 + index) % len(teacher[index])]
                ce, node_rank = teacher_loss(policy, self.gt, self.graph, self.codebook, q, chosen, qctx,
                                             self.device)
                marginal, soft_mass, eligible = question_path_loss(
                    policy, self.gt, q, self.train_cache[index], qctx, routes[index], mode="B", tau=tau,
                    path_beam=config["path_beam"], local_k=config["local_top_k"],
                    rescue_slots=config["terminal_rescue_slots"])
                loss = w["teacher_ce"] * ce + w["node_rank"] * node_rank + w["path"] * marginal
                (loss / settings["batch_size"]).backward()
                pending += 1
                component["path_eligible_answers"] += eligible
                for name, value in (("teacher_ce", ce), ("node_rank", node_rank), ("marginal", marginal),
                                    ("soft_mass", soft_mass), ("total", loss)):
                    component[name] += float(value.detach())
                if pending == settings["batch_size"] or pos == len(order):
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    if not math.isfinite(float(norm)):
                        raise FloatingPointError("nonfinite Setting B policy gradient")
                    component["grad_norm_sum"] += float(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    steps += 1
            for name in ("teacher_ce", "node_rank", "marginal", "soft_mass", "total"):
                component[name] /= len(order)
            component["grad_norm_mean"] = component.pop("grad_norm_sum") / max(1, steps)
            return dict(component), steps

        warm = self.done("warmup")
        warm_history = json.loads((self.out / "warmup" / "history.json").read_text())
        warm_row = next(row for row in warm_history if row["epoch"] == warm["selected_epoch"])
        epoch0 = {"validation": warm_row["validation"], "candidate_diagnostics": warm_row["candidate_diagnostics"],
                  "technical_valid": warm_row["technical_valid"],
                  "reused_from": f"warm-up epoch {warm['selected_epoch']} (same weights, same val programs)"}
        epoch0.update(extra(0, epoch0))
        selected, history = self.train_phase("setting B", directory, settings=settings, optimizer=optimizer, rng=rng,
                                             train_one=train_one, evaluate=self.policy_eval(holder, directory, extra),
                                             save_model=save_model, load_model=load_model,
                                             checkpoint_file="policy.pt", epoch0=epoch0)
        coverage_max = max(history, key=lambda x: x["hard_coverage"].get("answer_2hop_pool", 0))
        self.mark_done("setting_b", {"selected_epoch": selected["epoch"], "selected_updates": selected["updates"],
                                     "checkpoint_sha256": digest(directory / "best" / "policy.pt"),
                                     "warmup_input_sha256": digest(warmup_best),
                                     "highest_2hop_pool_epoch": coverage_max["epoch"],
                                     "selected_2hop_pool": selected["hard_coverage"].get("answer_2hop_pool", 0),
                                     "highest_2hop_pool": coverage_max["hard_coverage"].get("answer_2hop_pool", 0),
                                     "validation": selected["validation"], "train_route_mining": route_audit,
                                     "checkpoint_rule": config["selection"]})

    # ---------- stage 6: pool + ranker ----------
    def text_tables(self, flan, tokenizer, directory: Path):
        path = directory / "text_tables.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        node_text, _ = encode_texts(flan, tokenizer, self.graph.texts, self.device, max_length=96)
        relation_text, _ = encode_texts(flan, tokenizer, self.relation_names, self.device, max_length=48)
        tables = {"node_text": node_text, "relation_text": relation_text}
        torch.save(tables, path)
        return tables

    def stage_ranker(self) -> None:
        if self.done("ranker"):
            return
        config, graph, device = self.config, self.graph, self.device
        directory = self.out / "ranker"
        policy = self.load_policy(self.out / "setting_b" / "best" / "policy.pt")
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        pools = {}
        for name, questions, cache in (("train", self.train, self.train_cache), ("val", self.val, self.val_cache)):
            by_qid = {row["qid"]: row for row in cache}
            pools[name] = self.rows_cached(directory / f"pools_{name}.pkl", questions,
                                           lambda q: self.pool_row(policy, q, by_qid[q.qid]), f"pool {name}")
            dump(directory / f"{name}_pool_summary.json", pool_summary(pools[name], questions))
        labels, label_stats = training_labels(pools["train"], self.train, graph)
        dump(directory / "label_audits" / "pool_ranker.json", label_stats)
        flan, tokenizer = self.load_flan(self.out / "flan" / "selected")
        tables = self.text_tables(flan, tokenizer, directory)
        qtext, qms = {}, {}
        for name, questions in (("train", self.train), ("val", self.val)):
            qtext[name], qms[name] = encode_texts(flan, tokenizer, [input_text(graph, q) for q in questions], device,
                                                  max_length=128, individual_timing=(name == "val"))
        del flan
        settings = config["pool_ranker"]
        seed_all(self.seed + SEED_OFFSET["ranker"])
        holder = {"ranker": PoolRanker(tables["node_text"], tables["relation_text"], graph.types,
                                       self.config["max_hops"]).to(device)}
        optimizer = torch.optim.AdamW(holder["ranker"].parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        rng = np.random.default_rng(self.seed + SEED_OFFSET["ranker"])

        def save_model(path):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(holder["ranker"].state_dict(), path / "pool_ranker.pt")

        def load_model(path):
            holder["ranker"].load_state_dict(torch.load(path / "pool_ranker.pt", map_location=device, weights_only=True))

        def train_one(_epoch):
            info = train_epoch(holder["ranker"], pools["train"], qtext["train"], labels, optimizer, device, rng,
                               accumulation=settings["accumulation"])
            return info, info["optimizer_updates"]

        def evaluate(epoch):
            metrics, predictions = evaluate_ranker(holder["ranker"], pools["val"], self.val, qtext["val"], qms["val"],
                                                   graph, device, save_predictions=True)
            jsonl(directory / "val_predictions" / f"epoch_{epoch:02d}.jsonl", predictions)
            return {"validation": metrics, "technical_valid": finite_predictions(metrics, predictions, len(self.val))}

        selected, _ = self.train_phase("pool ranker", directory, settings=settings, optimizer=optimizer, rng=rng,
                                       train_one=train_one, evaluate=evaluate, save_model=save_model,
                                       load_model=load_model, checkpoint_file="pool_ranker.pt")
        val_metrics, val_predictions = evaluate_ranker(holder["ranker"], pools["val"], self.val, qtext["val"],
                                                       qms["val"], graph, device, save_predictions=True)
        for key in ("mean_hits_at_1", "mean_recall_at_20"):
            assert abs(val_metrics[key] - selected["validation"][key]) < 1e-9
        dump(directory / "val_metrics.json", val_metrics)
        jsonl(directory / "val_predictions.jsonl", val_predictions)
        dump(directory / "val_independent_recount.json", check_recount(val_metrics, val_predictions))
        self.mark_done("ranker", {"selected_epoch": selected["epoch"],
                                  "checkpoint_sha256": digest(directory / "best" / "pool_ranker.pt"),
                                  "train_label_stats": label_stats, "validation": val_metrics})

    # ---------- stage 7: khoá checkpoint, test đúng một lần ----------
    def stage_test(self) -> None:
        if self.done("test"):
            return
        config, graph, device = self.config, self.graph, self.device
        directory = self.out / "test"
        directory.mkdir(parents=True, exist_ok=True)
        flan_dir = self.out / "flan" / "selected"
        policy_file = self.out / "setting_b" / "best" / "policy.pt"
        ranker_file = self.out / "ranker" / "best" / "pool_ranker.pt"
        # Mọi quyết định checkpoint đã xong: dựng lại module và nạp đúng byte đã chọn trước khi chạm đáp án test.
        flan, tokenizer = self.load_flan(flan_dir)
        flan.eval()
        policy = self.load_policy(policy_file)
        state = torch.load(ranker_file, map_location="cpu", weights_only=True)
        ranker = PoolRanker(state["node_text"], state["relation_text"], graph.types, config["max_hops"]).to(device)
        ranker.load_state_dict(state)
        ranker.eval()
        locked = {"flan_sha256": digest(flan_dir / "model.safetensors"),
                  "warmup_policy_sha256": digest(self.out / "warmup" / "best" / "policy.pt"),
                  "setting_b_policy_sha256": digest(policy_file), "pool_ranker_sha256": digest(ranker_file),
                  "flan_epochs": [self.done("flan")["synthetic_selected_epoch"], self.done("flan")["qa_sft_selected_epoch"]],
                  "warmup_epoch": self.done("warmup")["selected_epoch"],
                  "setting_b_epoch": self.done("setting_b")["selected_epoch"],
                  "ranker_epoch": self.done("ranker")["selected_epoch"], "test_answers_loaded": False}
        if (locked["flan_sha256"] != self.done("flan")["model_sha256"] or
                locked["warmup_policy_sha256"] != self.done("warmup")["checkpoint_sha256"] or
                locked["setting_b_policy_sha256"] != self.done("setting_b")["checkpoint_sha256"] or
                locked["pool_ranker_sha256"] != self.done("ranker")["checkpoint_sha256"]):
            raise AssertionError("selected checkpoint hash changed before test")
        dump(directory / "checkpoint_lock.json", locked)
        # Sinh chương trình chỉ dùng câu hỏi/topic; đáp án vẫn chưa được đọc.
        codec = ProgramCodec(tokenizer, graph, self.decode_executor, config["max_hops"])
        test_unlabelled = graph.splits["test"]
        test_cache = decode_programs(flan, tokenizer, codec, graph, test_unlabelled, directory / "program_cache.jsonl",
                                     device, beam=config["program_beam"], length_penalty=config["decoder_length_penalty"],
                                     checkpoint=lambda done: self.clock.need(900, f"decode test {done}"))
        graph.load_test_answers()
        self.save_audit(test_answers_loaded=True)
        test = graph.splits["test"]
        if [q.qid for q in test] != [q.qid for q in test_unlabelled]:
            raise AssertionError("test qid order changed after answer load")
        labelled = [q for q in test if q.topics]
        a = config["program_audit"]
        dump(directory / "program_distribution.json",
             program_distribution([c for c, q in zip(test_cache, test) if q.topics], labelled,
                                  max_hops=config["max_hops"], train_zero_rate=self.done("flan")["train_zero_answer_rate"],
                                  zero_factor=a["flan_top_zero_factor"], zero_floor=a["flan_top_zero_floor"],
                                  nonzero_beam_floor=a["flan_nonzero_beam_floor"]))
        by_qid = {row["qid"]: row for row in test_cache}
        rows = self.rows_cached(directory / "pools_test.pkl", test, lambda q: self.pool_row(policy, q, by_qid[q.qid]),
                                "pool test")
        dump(directory / "pool_summary.json", pool_summary(rows, test))
        qtext, text_ms = encode_texts(flan, tokenizer, [input_text(graph, q) for q in test], device, max_length=128,
                                      individual_timing=True)
        metrics, predictions = evaluate_ranker(ranker, rows, test, qtext, text_ms, graph, device, save_predictions=True)
        jsonl(directory / "test_predictions.jsonl", predictions)
        dump(directory / "test_metrics.json", metrics)
        dump(directory / "test_independent_recount.json", check_recount(metrics, predictions))
        self.mark_done("test", {"checkpoint_lock": locked, "test_evaluation_count": 1,
                                "test_after_all_checkpoints_reloaded": True,
                                "test_predictions_sha256": digest(directory / "test_predictions.jsonl"),
                                "program_cache_sha256": digest(directory / "program_cache.jsonl"),
                                "metrics": {k: metrics[k] for k in ("mean_hits_at_1", "mean_recall_at_20",
                                                                    "mean_pool_recall", "mean_f2_set", "queries")}})

    def run(self, stop_after: str | None = None) -> None:
        started = time.perf_counter()
        self.prepare()
        for stage in (self.stage_codebook, self.stage_flan, self.stage_programs, self.stage_warmup,
                      self.stage_setting_b, self.stage_ranker, self.stage_test):
            mark = time.perf_counter()
            stage()
            name = stage.__name__.removeprefix("stage_")
            times = self.audit.get("stage_seconds", {})
            times[name] = times.get(name, 0.0) + time.perf_counter() - mark
            self.save_audit(stage_seconds=times)
            if name == stop_after:
                # Chạy một phần (vd ablation chỉ stage Flan); chạy tiếp được bằng --resume không có --stop-after.
                self.save_audit(status=f"STOPPED_AFTER_{name}", session_seconds=time.perf_counter() - started)
                print(f"HOP V8 stopped after stage {name}", flush=True)
                return
        self.save_audit(status="COMPLETE", session_seconds=time.perf_counter() - started)
        m = self.done("test")["metrics"]
        print(f"HOP V8 COMPLETE: test H1={m['mean_hits_at_1']:.4f} R20={m['mean_recall_at_20']:.4f}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="bundle dir from tools/prep_inputs.py")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flan-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deadline-hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=("codebook", "flan", "programs", "warmup", "setting_b", "ranker"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    check_config(config)
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args, config)
    try:
        pipeline.run(stop_after=args.stop_after)
    except Pause as pause:
        pipeline.save_audit(status="PAUSED", paused_at=str(pause))
        print(f"HOP V8 PAUSED at {pause}; rerun with --resume on a copy of this output", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
