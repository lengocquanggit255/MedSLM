"""Dựng bundle chuẩn hoá cho HOP V8 từ dữ liệu RiTeK trong data/ritek (chạy local, từ gốc repo).

python HOP/hop_polo_v8/tools/prep_inputs.py --dataset prime|pharmkg|adint [--out HOP/hop_polo_v8/inputs]

Chính sách split (quyết định 2026-10-27):
  prime    train/val theo ritek/splits/{train,val}.index (10% RiTeK train, seed 42 — chung với mọi phương pháp
           RiTeK), test = 2,801 câu RiTeK; 18 câu test không có topic được giữ (chỉ backfill khi chấm).
  pharmkg  train/dev/test chính thức của RiTeK (dev làm val).
  adint    file dev của RiTeK không có topic -> val = 10% train theo đúng quy tắc nhóm của V7
           (nhóm theo (câu hỏi, topic), xáo khoá bằng seed 42, lấy nhóm tới khi đủ 10%).
Mọi bộ: bỏ câu train/val trùng văn bản (casefold) với test, và câu train trùng văn bản với val.
Văn bản nút Prime: node_info Prime của RiTeK = STaRK, không có `description` -> quyết định D5(b) 2026-10-24:
description = văn bản `no_rel` của STaRK (starkbench.corpus) bỏ hai dòng `- name:` và `- type:`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[3]
RITEK = ROOT / "data" / "ritek"
DATASETS = {
    "pharmkg": {"kg": "Dataset/pharmKG/Pharm_KG.txt", "skb": "pharmkg_skb/pharmKG",
                "train": "Dataset/pharmKG/pharmKG_train_QA_with_topic_entity.json",
                "val": "Dataset/pharmKG/pharmKG_dev_with_topic_entity.json",
                "test": "Dataset/pharmKG/pharmKG_test_QA_with_topic_entity.json",
                "split_policy": "official RiTeK train/dev/test (dev as val)"},
    "adint": {"kg": "Dataset/Adint/ADint_KG.txt", "skb": "adint_skb/ADint",
              "train": "Dataset/Adint/ADint_train_with_topic_entity.json", "val": None,
              "test": "Dataset/Adint/ADint_test_with_topic_entity.json",
              "split_policy": "val = 10% of RiTeK train by V7 grouping rule (seed 42); RiTeK dev has no topics"},
    "prime": {"kg": "Dataset/prime/prime_KG.txt", "skb": "prime_skb/prime",
              "train": "Dataset/prime/Prime_train_QA_with_topic_entity.json", "val": None,
              "test": "Dataset/prime/Prime_test_QA_with_topic_entity.json",
              "split_policy": "train/val from ritek/splits/{train,val}.index; test = RiTeK test"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_triples(path: Path) -> np.ndarray:
    import pandas as pd
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64, engine="c")
    if frame.shape[1] != 3:
        raise ValueError(f"{path} must have head, relation, tail")
    return frame.to_numpy(np.int64)


def node_rows(dataset: str, info: dict) -> list[dict]:
    if dataset != "prime":
        return [{"id": int(k), "name": v.get("name"), "type": v.get("type"), "description": v.get("description")}
                for k, v in sorted(info.items())]
    sys.path.insert(0, str(ROOT / "src"))
    from starkbench import corpus
    docs = corpus.load("no_rel")
    if not len(docs) == len(info) == 129375:
        raise AssertionError("Prime node_info / no_rel corpus size mismatch")
    rows = []
    for i in range(len(docs)):
        lines = docs[i].split("\n")
        if lines[0] != f"- name: {info[i]['name']}" or lines[1] != f"- type: {info[i]['type']}":
            raise AssertionError(f"no_rel document {i} does not start with name/type lines")
        rows.append({"id": i, "name": info[i]["name"], "type": info[i]["type"],
                     "description": "\n".join(lines[2:]).strip()})
    return rows


def normal_name(value) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).split())


def name_index(info: dict, train_rows: list[dict]) -> dict[str, set[int]]:
    """Chỉ mục tên của V7 (ADIntGraph): tên/alias tách theo '|', bỏ hậu tố gene/protein, cộng tên topic có ID của train."""
    index: dict[str, set[int]] = {}
    for node, meta in info.items():
        for value in (meta.get("name"), meta.get("alias"), meta.get("aliases"), meta.get("synonyms")):
            for alias in value if isinstance(value, (list, tuple, set)) else [value]:
                if not alias:
                    continue
                for part in str(alias).split("|"):
                    candidates = [part] + [part[:-len(s)] for s in (" gene", " protein", " wt allele")
                                           if part.casefold().endswith(s)]
                    for candidate in candidates:
                        index.setdefault(normal_name(candidate), set()).add(int(node))
    for row in train_rows:
        value = row.get("topic_entity")
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and item.get("topic_id") is not None and item.get("topic_name"):
                if int(item["topic_id"]) in info:
                    index.setdefault(normal_name(item["topic_name"]), set()).add(int(item["topic_id"]))
    return index


def normalize(rows: list[dict], split: str, nodes: set[int], drops: Counter, labelled: bool,
              names: dict[str, set[int]]) -> list[dict]:
    result = []
    for row in rows:
        qid, question = str(row["question_id"]), str(row.get("question") or "").strip()
        topics, unknown_topics = [], 0
        value = row.get("topic_entity") or []
        for item in value if isinstance(value, list) else [value]:  # PharmKG/ADint: dict đơn, list, hoặc tên trơn
            if isinstance(item, dict) and item.get("topic_id") not in (None, ""):
                topic = int(item["topic_id"])
                if topic in nodes:
                    topics.append(topic)
                else:
                    unknown_topics += 1
                continue
            # Không có ID: tra tên như V7; tên mơ hồ giữ MỌI nút khớp.
            name = item.get("topic_name") if isinstance(item, dict) else item
            matches = names.get(normal_name(name), set()) & nodes if name else set()
            drops[f"{split}_topic_by_name"] += 1
            drops[f"{split}_topic_by_name_unresolved"] += int(not matches)
            drops[f"{split}_topic_by_name_ambiguous"] += int(len(matches) > 1)
            topics.extend(matches)
        drops[f"{split}_topic_not_in_graph"] += unknown_topics
        answers = sorted({int(k) for k in (row.get("answer") or {})})
        kept = [a for a in answers if a in nodes]
        drops[f"{split}_answer_not_in_graph"] += len(answers) - len(kept)
        if not question:
            drops[f"{split}_question_empty"] += 1
            continue
        if not kept:
            drops[f"{split}_question_without_answer_in_graph"] += 1
            continue
        if labelled and not topics:
            drops[f"{split}_question_without_topic"] += 1
            continue
        result.append({"qid": qid, "question": question, "topics": sorted(set(topics)), "answers": kept})
    return result


def v7_grouped_val(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["question"].casefold().strip(), tuple(row["topics"])), []).append(row)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    target, count, val_keys = round(len(rows) * 0.10), 0, set()
    for key in keys:
        if count >= target:
            break
        val_keys.add(key)
        count += len(groups[key])
    return ([r for k in keys if k not in val_keys for r in groups[k]],
            [r for k in keys if k in val_keys for r in groups[k]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out", type=Path, default=ROOT / "HOP" / "hop_polo_v8" / "inputs")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    sources = {name: RITEK / spec[name] for name in ("kg", "train", "val", "test") if spec[name]}
    sources["node_info"] = RITEK / spec["skb"] / "node_info.pkl"
    sources["edge_type_dict"] = RITEK / spec["skb"] / "edge_type_dict.pkl"
    triples = read_triples(sources["kg"])
    info = {int(k): v for k, v in pickle.load(open(sources["node_info"], "rb")).items()}
    relations = {int(k): str(v) for k, v in pickle.load(open(sources["edge_type_dict"], "rb")).items()}
    nodes = set(info) | set(triples[:, 0].tolist()) | set(triples[:, 2].tolist())
    if nodes - set(info):
        raise AssertionError(f"{len(nodes - set(info))} KG nodes lack node_info")
    drops: Counter = Counter()
    raw_train = read_rows(sources["train"])
    names = name_index(info, raw_train)
    train = normalize(raw_train, "train", nodes, drops, True, names)
    test = normalize(read_rows(sources["test"]), "test", nodes, drops, False, names)
    if args.dataset == "pharmkg":
        val = normalize(read_rows(sources["val"]), "val", nodes, drops, True, names)
    elif args.dataset == "adint":
        train, val = v7_grouped_val(train)
    else:
        split_dir = ROOT / "ritek" / "splits"
        train_ids = {line.strip() for line in (split_dir / "train.index").read_text().splitlines() if line.strip()}
        val_ids = {line.strip() for line in (split_dir / "val.index").read_text().splitlines() if line.strip()}
        test_ids = [line.strip() for line in (split_dir / "test.index").read_text().splitlines() if line.strip()]
        if {q["qid"] for q in test} != set(test_ids):
            raise AssertionError("RiTeK test qids differ from ritek/splits/test.index")
        drops["train_not_in_ritek_split"] = sum(q["qid"] not in train_ids | val_ids for q in train)
        val = [q for q in train if q["qid"] in val_ids]
        train = [q for q in train if q["qid"] in train_ids]
    # Chống rò: bỏ câu train/val trùng văn bản với test, câu train trùng văn bản với val.
    test_text = {q["question"].casefold().strip() for q in test}
    val_text = {q["question"].casefold().strip() for q in val}
    before = (len(train), len(val))
    val = [q for q in val if q["question"].casefold().strip() not in test_text]
    train = [q for q in train if q["question"].casefold().strip() not in test_text | val_text]
    drops["train_removed_text_overlap"] = before[0] - len(train)
    drops["val_removed_text_overlap"] = before[1] - len(val)
    np.save(out / "triples.npy", triples)
    with (out / "nodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in node_rows(args.dataset, info):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "relations.json").write_text(json.dumps({str(k): v for k, v in sorted(relations.items())},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    for name, rows in (("qa_train", train), ("qa_val", val)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "qa_test.jsonl").open("w", encoding="utf-8") as questions, \
            (out / "test_answers.jsonl").open("w", encoding="utf-8") as answers:
        for row in test:
            questions.write(json.dumps({k: row[k] for k in ("qid", "question", "topics")}, ensure_ascii=False) + "\n")
            answers.write(json.dumps({"qid": row["qid"], "answers": row["answers"]}) + "\n")
    files = ("triples.npy", "nodes.jsonl", "relations.json", "qa_train.jsonl", "qa_val.jsonl", "qa_test.jsonl",
             "test_answers.jsonl")
    manifest = {"dataset": args.dataset, "split_policy": spec["split_policy"],
                "files": {name: sha(out / name) for name in files},
                "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in sources.items()},
                "counts": {"triples": int(len(triples)), "nodes": len(info), "relations": len(relations),
                           "train": len(train), "val": len(val), "test": len(test),
                           "test_without_topic": sum(not q["topics"] for q in test)},
                "dropped": dict(sorted(drops.items()))}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dataset", "counts", "dropped")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
