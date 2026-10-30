"""Flan-T5 sinh chương trình HOP: khởi tạo token, batch huấn luyện, giải mã có ràng buộc, audit phân bố."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import jsonl
from .program import input_text, program_to_dict


def initialize_program_tokens(model, tokenizer, old_vocab_size: int, *, seed: int, untie_lm_head: bool) -> dict:
    """Token chương trình mới = embedding của "the" + nhiễu 0.01 (đầu vào và đầu ra nếu không buộc trọng số).

    transformers 5.x ép `tie_word_embeddings=True` cho mọi T5, nên `resize_token_embeddings` thay lm_head đã
    huấn luyện riêng của Flan-T5 (v1.1) bằng chính ma trận embedding. V7 (và checkpoint Setting B cũ) chạy
    như vậy. `untie_lm_head=True` dựng lại lm_head riêng từ trọng số gốc sau khi resize.
    """
    torch.manual_seed(seed)
    anchor = tokenizer.encode("the", add_special_tokens=False)[0]
    original_head = model.get_output_embeddings().weight.detach().clone()
    untied_before = original_head.data_ptr() != model.get_input_embeddings().weight.data_ptr()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    rebuilt = False
    if untie_lm_head and untied_before and \
            model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        head = torch.nn.Linear(original_head.shape[1], len(tokenizer), bias=False,
                               device=original_head.device, dtype=original_head.dtype)
        with torch.no_grad():
            keep = min(old_vocab_size, original_head.shape[0])
            head.weight[:keep] = original_head[:keep]
        model.set_output_embeddings(head)
        rebuilt = True
    with torch.no_grad():
        incoming = model.get_input_embeddings().weight
        outgoing = model.get_output_embeddings().weight
        source_in, source_out = incoming[anchor].clone(), outgoing[anchor].clone()
        incoming[old_vocab_size:] = source_in + 0.01 * torch.randn_like(incoming[old_vocab_size:])
        if outgoing.data_ptr() != incoming.data_ptr():
            outgoing[old_vocab_size:] = source_out + 0.01 * torch.randn_like(outgoing[old_vocab_size:])
    return {"anchor_token": "the", "anchor_id": anchor, "old_vocab_size": old_vocab_size,
            "program_token_count": len(tokenizer) - old_vocab_size,
            "untie_lm_head": untie_lm_head, "lm_head_rebuilt_after_resize": rebuilt,
            "input_output_tied": outgoing.data_ptr() == incoming.data_ptr()}


def _labels(labels_list, device):
    labels = torch.full((len(labels_list), max(map(len, labels_list))), -100, dtype=torch.long, device=device)
    for i, ids in enumerate(labels_list):
        labels[i, :len(ids)] = torch.as_tensor(ids, device=device)
    return labels


def train_batch(model, tokenizer, examples, optimizer, device) -> float:
    texts, labels_list = zip(*examples)
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=256,
                        return_token_type_ids=False).to(device)
    loss = model(**encoded, labels=_labels(labels_list, device)).loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def train_qa_batch(model, tokenizer, graph, train, plans, indices, epoch, optimizer, device):
    """Mỗi câu có nhãn nặng 1; các đáp án có nhãn của nó chia đều trọng số; chương trình xoay vòng theo epoch."""
    texts, labels_list, weights = [], [], []
    for index in indices:
        index = int(index)
        available = [(answer_index, programs) for answer_index, programs in enumerate(plans[index]) if programs]
        if not available:
            raise AssertionError("unlabeled QA entered the QA SFT epoch")
        for answer_index, programs in available:
            texts.append(input_text(graph, train[index]))
            labels_list.append(programs[(epoch - 1 + index + answer_index) % len(programs)])
            weights.append(1.0 / len(available))
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256,
                        return_token_type_ids=False).to(device)
    labels = _labels(labels_list, device)
    logits = model(**encoded, labels=labels).logits
    token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
                                 reduction="none").reshape(labels.shape)
    per_answer = token_loss.sum(1) / (labels != -100).sum(1).clamp_min(1)
    loss = (per_answer * torch.tensor(weights, device=device, dtype=per_answer.dtype)).sum() / len(indices)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), len(labels_list)


def decode_programs(model, tokenizer, codec, graph, questions, cache_path: Path, device, *, beam: int,
                    length_penalty: float, checkpoint=None) -> list[dict]:
    """Beam có ràng buộc cho từng câu; nối tiếp từ file cache dở (cùng thứ tự qid).

    `checkpoint(n)` được gọi sau mỗi lần lưu từng phần (mỗi 100 câu), vd để dừng an toàn khi gần hạn phiên.
    """
    rows = [json.loads(x) for x in cache_path.open(encoding="utf-8")] if cache_path.exists() else []
    if [r["qid"] for r in rows] != [q.qid for q in questions[:len(rows)]]:
        rows = []
    model.eval()
    for index in range(len(rows), len(questions)):
        q = questions[index]
        if not q.topics:
            rows.append({"qid": q.qid, "beams": [], "generation_ms": 0.0, "decoder_edges_examined": 0,
                         "no_topic": True})
            continue
        codec.executor.begin_query(10 ** 9)
        started = time.perf_counter()
        encoded = tokenizer(input_text(graph, q), return_tensors="pt", truncation=True, max_length=256,
                            return_token_type_ids=False).to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, num_beams=beam, num_return_sequences=beam,
                                       max_new_tokens=max(26, 8 * codec.max_hops + 2),
                                       prefix_allowed_tokens_fn=lambda _batch, ids: codec.allowed(ids.tolist(), q.topics),
                                       return_dict_in_generate=True, output_scores=True,
                                       length_penalty=length_penalty, renormalize_logits=True, early_stopping=True)
        probabilities = F.softmax(generated.sequences_scores.float(), dim=0).cpu().tolist()
        beams = []
        for sequence, probability in zip(generated.sequences, probabilities):
            program = codec.decoded(sequence.tolist())
            if program is not None:
                beams.append({"program": program_to_dict(program), "probability": probability})
        rows.append({"qid": q.qid, "beams": beams, "generation_ms": (time.perf_counter() - started) * 1000,
                     "decoder_edges_examined": codec.executor.query_edges_examined})
        if (index + 1) % 100 == 0:
            print(f"decoded {cache_path.stem} {index + 1}/{len(questions)}", flush=True)
            jsonl(cache_path, rows)
            if checkpoint is not None and index + 1 < len(questions):
                checkpoint(index + 1)
    jsonl(cache_path, rows)
    return rows


def program_distribution(cached, questions, *, max_hops, train_zero_rate, zero_factor=5.0, zero_floor=0.10,
                         nonzero_beam_floor=0.50) -> dict:
    """Audit hạng beam và khối lượng xác suất theo số hop, không đổi dự đoán."""
    by_rank = [Counter() for _ in range(5)]
    mass = {str(depth): [] for depth in range(max_hops + 1)}
    top_zero = top_zero_gold = top_zero_wrong = beam_nonzero = 0
    for q, row in zip(questions, cached):
        beams = row["beams"]
        values = Counter()
        for rank, beam in enumerate(beams):
            hops = len(beam["program"]["hops"])
            by_rank[min(rank, 4)][str(hops)] += 1
            values[str(hops)] += float(beam["probability"])
        for depth in range(max_hops + 1):
            mass[str(depth)].append(float(values[str(depth)]))
        if beams and not beams[0]["program"]["hops"]:
            top_zero += 1
            gold = bool(set(q.topics) & set(q.answers))
            top_zero_gold += int(gold)
            top_zero_wrong += int(not gold)
        beam_nonzero += int(any(beam["program"]["hops"] for beam in beams))
    n = max(1, len(cached))
    # Trần cố định từ đáp án train, sàn thống kê 10%.
    top_zero_ceiling = max(zero_floor, min(0.25, zero_factor * train_zero_rate), 2 / n)
    return {"queries": len(cached), "by_rank": [dict(item) for item in by_rank],
            "mean_probability_mass": {k: float(np.mean(v)) if v else 0.0 for k, v in mass.items()},
            "top_zero_queries": top_zero, "top_zero_gold": top_zero_gold, "top_zero_wrong": top_zero_wrong,
            "top_zero_rate": top_zero / n, "beam_has_nonzero_rate": beam_nonzero / n,
            "train_zero_answer_rate": train_zero_rate, "top_zero_ceiling": top_zero_ceiling,
            "nonzero_beam_floor": nonzero_beam_floor,
            "collapse_guard_valid": top_zero / n <= top_zero_ceiling and beam_nonzero / n >= nonzero_beam_floor}
