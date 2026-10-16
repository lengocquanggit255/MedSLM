"""GNN-RAG (A+) trên TOÀN KG STaRK-Prime bản RiTeK — model + vòng train ReaRev GỐC, chỉ thay tầng dữ
liệu và cách TÍNH message (tương đương đại số). Quyết định 2026-10-16: ritek/PLAN.md §2 Q2.

Code gốc: github.com/cmavro/GNN-RAG @ 28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4, thư mục gnn/. Repo
KHÔNG có license nên không chép vào repo này: kernel clone đúng commit đó rồi vá lúc chạy
(`apply_patches`). Cấu hình theo scripts/rearev_cwq.sh của repo: entity_dim 50 · batch 8 ·
num_iter 2 · num_ins 3 · num_gnn 3 · lm sbert · relation_word_emb · eval_every 2 · num_epoch 100.

VÌ SAO PHẢI VÁ PHÉP TÍNH (không đổi kết quả):
  ReaRev tạo message cho TỪNG cạnh: fact_val [|E_batch|, d] = relu(W·r ⊙ q)·p(head) rồi sparse.mm
  cộng vào đuôi. Toàn KG Prime |E| = 8.1M (+129k self-loop) mỗi câu hỏi ⇒ ~1.6 GB mỗi tensor,
  6 tensor mỗi layer, nhân batch ⇒ OOM trên T4 dù batch 1. Message chỉ phụ thuộc (quan hệ r, câu
  hỏi) và p(head) vô hướng, nên
      Σ_{(h,r,t)} relu(W·r ⊙ q)·p_h  =  Σ_r relu(W·r ⊙ q)·S[t, r],   S[t, r] = Σ_{h:(h,r,t)} p_h
  S = M @ p, M ∈ N^{(V·R)×V} cố định ⇒ một sparse-mm mỗi layer. Chiều ngược (reason_layer_inv) và
  khởi tạo entity (TypeLayer, trọng số 1) gộp y hệt. ritek/gnnrag/test_equivalence.py đối chiếu số
  với code gốc trên KG nhỏ.

THAY ĐỔI TẦNG DỮ LIỆU (bắt buộc, không đụng model):
  1. Subgraph của mọi câu hỏi = TOÀN KG (id cục bộ = id toàn cục). Không lưu tuples.
  2. Tên quan hệ Prime tách theo khoảng trắng. Bộ tách gốc theo định dạng Freebase `a.b.c` biến cả 18
     quan hệ thành 'UNK'.
  3. max_length token quan hệ = số TOKEN dài nhất (gốc: số TỪ dài nhất). Tên Prime ≤ 2 từ ⇒ cách gốc
     cắt mọi quan hệ còn [CLS][SEP] ⇒ 18 quan hệ trùng embedding.
  4. Ứng viên gồm cả topic entity — hành vi gốc khi name == 'cwq' (cấu hình đang theo).
  5. Câu hỏi không có topic bị bỏ khỏi train/dev — như gốc (`len(line['entities']) == 0`).
THAY ĐỔI VÒNG TRAIN (không ảnh hưởng chọn mô hình — vẫn theo dev H1 như gốc):
  - Bỏ chấm test sau mỗi lần eval dev (gốc chỉ in log; trên toàn KG tốn ~4h).
  - calc_f1_new / Evaluator.evaluate: vector hoá vòng lặp Python trên 129k ứng viên — kết quả trùng.
  - Micro-batch + cộng dồn gradient khi batch 8 không vừa VRAM: loss gốc chia cho batch nên nhân lại
    hệ số len(micro)/len(batch) ⇒ cùng gradient trước clip/step.
  - Dừng train khi chạm DEADLINE_H (phiên Kaggle tối đa 12h), dùng checkpoint dev-H1 tốt nhất.
  - CHẠY TIẾP qua nhiều phiên (quyết định 2026-10-16: không giảm số epoch). Cuối mỗi phiên lưu
    rearev_state.pt = model + Adam + epoch + best_h1/f1 + RNG numpy/torch. RESUME=1 nạp lại từ
    dataset chứa output phiên trước ⇒ nối tiếp y như chạy liền (kiểm: ritek/scripts/test_rearev_resume.py).
    Phiên đầu (code 7011a10) chưa lưu state: lần nối đầu chỉ nạp trọng số rearev-final.ckpt, Adam
    khởi tạo lại (meta ghi resume_mode).

Đầu ra (/kaggle/working): rearev_test_scores.npy (n, 129375) float32 = pred_dist · rearev_test_qids.npy
· rearev_meta.json · rearev-h1.ckpt · rearev-f1.ckpt · rearev-final.ckpt · rearev_state.pt
· rearev_history.json (cộng dồn mọi phiên) · log. Query test KHÔNG có topic không được chấm (Q8: backfill).
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

GNNRAG_SHA = "28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4"
LM_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # lm == 'sbert' trong code gốc
VERY_NEG = -100000000000


# =============================================================================== repo gốc
def fetch_repo(dest):
    """Clone đúng commit, chỉ thư mục gnn/. Trả về đường dẫn gnn/."""
    if not os.path.exists(os.path.join(dest, "gnn", "train_model.py")):
        os.makedirs(dest, exist_ok=True)
        for c in (["git", "init", "-q"], ["git", "remote", "add", "origin", "https://github.com/cmavro/GNN-RAG.git"],
                  ["git", "sparse-checkout", "set", "gnn"],
                  ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", GNNRAG_SHA],
                  ["git", "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(c, cwd=dest, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
    assert head == GNNRAG_SHA, f"repo GNN-RAG ở {head}, cần {GNNRAG_SHA}"
    return os.path.join(dest, "gnn")


# ======================================================================== đồ thị toàn KG
class FullKG:
    """Toàn KG làm 'subgraph' của mọi câu hỏi — thay cho tuple kb_adj_mat của code gốc.

    Fact = cạnh (h, r, t) đã lưu + self-loop (v, R0, v) cho mọi node — đúng _build_fact_mat gốc
    (use_self_loop=True, relation id = num_kb_relation - 1, use_inverse_relation=False).
      M_fwd[(t·R + r), h] = #fact (h, r, t)   ->  S_fwd = gộp p(head) vào đuôi theo quan hệ
      M_inv[(h·R + r), t] = #fact (h, r, t)   ->  S_inv = gộp p(tail) vào đầu theo quan hệ
      C[v, r] = #fact có đuôi v + #fact có đầu v  (TypeLayer: fact2tail + fact2head, trọng số 1)
    """

    def __init__(self, src, rel, dst, num_nodes, num_base_rel, device):
        V, R0 = int(num_nodes), int(num_base_rel)
        self.V, self.R = V, R0 + 1                      # + self-loop
        ar = np.arange(V, dtype=np.int64)
        h = np.concatenate([np.asarray(src, np.int64), ar])
        t = np.concatenate([np.asarray(dst, np.int64), ar])
        r = np.concatenate([np.asarray(rel, np.int64), np.full(V, R0, np.int64)])
        self.num_fact = int(h.size)
        ones = torch.ones(h.size, dtype=torch.float32)

        def coo(rows, cols):
            m = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), ones, (V * self.R, V))
            return m.coalesce().to(device)
        self.M_fwd = coo(t * self.R + r, h)
        self.M_inv = coo(h * self.R + r, t)
        C = np.zeros(V * self.R, dtype=np.float32)
        np.add.at(C, t * self.R + r, 1.0)
        np.add.at(C, h * self.R + r, 1.0)
        self.C = torch.from_numpy(C.reshape(V, self.R)).to(device)

    def gather(self, M, dist):
        """dist (B, V) -> S (B, V, R): S[b, v, r] = Σ_{fact r nối vào v} dist[b, đầu kia]."""
        B = dist.size(0)
        S = torch.sparse.mm(M, dist.t().contiguous())       # (V·R, B)
        return S.view(self.V, self.R, B).permute(2, 0, 1)


# ============================================================ bản vá tính toán (tương đương)
def _reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
    """= ReasonGNNLayer.reason_layer gốc (sparse.mm theo từng fact), gộp theo quan hệ."""
    kg = self.edge_list
    rel = rel_linear(self.rel_features[:kg.R])
    if pos_emb is not None:
        rel = rel + pos_emb.weight[:kg.R]
    g = F.relu(rel.unsqueeze(0) * instruction.unsqueeze(1))            # (B, R, d)
    return torch.bmm(kg.gather(kg.M_fwd, curr_dist), g)                 # (B, V, d)


def _reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
    kg = self.edge_list
    rel = rel_linear(self.rel_features_inv[:kg.R])
    if pos_emb_inv is not None:
        rel = rel + pos_emb_inv.weight[:kg.R]
    g = F.relu(rel.unsqueeze(0) * instruction.unsqueeze(1))
    return torch.bmm(kg.gather(kg.M_inv, curr_dist), g)


def _build_matrix(self):
    """Gốc dựng 7 ma trận thưa (|E_batch| × B·V). Bản vá đọc thẳng FullKG trong reason_layer."""
    self.num_fact = self.edge_list.num_fact


def _type_layer_forward(self, local_entity, edge_list, rel_features):
    """= TypeLayer.forward gốc: relu(fact2tail @ W·r + fact2head @ W·r), trọng số 1 (norm_rel tắt)."""
    assert not self.norm_rel, "chỉ hỗ trợ norm_rel=False (mặc định gốc)"
    kg = edge_list
    B, V = local_entity.size()
    emb = F.relu(kg.C @ self.kb_self_linear(rel_features[:kg.R]))     # (V, d) — giống nhau mọi câu hỏi
    return emb.unsqueeze(0).expand(B, V, emb.size(1))


def _f1_hits_vec(probs, cand_mask, answers, eps):
    """= f1_and_hits gốc (sorted giảm dần, ổn định; cộng dồn tới khi > eps) — vector hoá.

    probs: float64 (V,) · cand_mask: bool (V,) — ứng viên hợp lệ · answers: tập id đáp án.
    Trả (precision, recall, f1, hits, em) đúng các nhánh của code gốc.
    """
    idx = np.flatnonzero(cand_mask)
    order = idx[np.argsort(-probs[idx], kind="stable")]
    best = order[0] if order.size else -1
    if order.size:
        over = np.flatnonzero(np.cumsum(probs[order]) > eps)
        retrieved = order[: over[0] + 1] if over.size else order
    else:
        retrieved = order
    correct = int(np.isin(retrieved, list(answers)).sum()) if answers else 0
    em = 1 if correct > 0 else 0
    if len(answers) == 0:
        return (1.0, 1.0, 1.0, 1.0, 1.0) if retrieved.size == 0 else (0.0, 1.0, 0.0, 1.0, 1.0)
    hits = float(best in answers)
    if retrieved.size == 0:
        return 1.0, 0.0, 0.0, hits, hits
    p, r = correct / retrieved.size, correct / len(answers)
    f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
    return p, r, f1, hits, em


def _calc_f1_new(self, curr_dist, dist_ans, h1_vec):
    """= BaseModel.calc_f1_new gốc (vòng lặp Python trên mọi ứng viên), vector hoá."""
    B, V = curr_dist.size()
    ignore_prob = (1 - self.eps) / V
    probs = curr_dist.detach().double().cpu().numpy()
    ans = dist_ans.detach().cpu().numpy()
    seed = self.seed_entities.detach().cpu().numpy()
    cand = self.local_entity.detach().cpu().numpy()
    out = []
    for b in range(B):
        if h1_vec[b].item() == 0.0:
            out.append(0.0)
            continue
        keep = (seed[b] <= 0) & (cand[b] != self.num_entity)
        answers = set(cand[b][keep & (ans[b] > 0)].tolist())
        out.append(_f1_hits_vec(probs[b], keep & (probs[b] >= ignore_prob), answers, self.eps)[2])
    return torch.FloatTensor(out).to(self.device)


def _evaluate(self, valid_data, test_batch_size=20, write_info=False):
    """= Evaluator.evaluate gốc (trả mean f1, hits, em), vector hoá; không ghi file .info."""
    self.model.eval()
    eps = self.eps
    f1s, hits, ems = [], [], []
    valid_data.reset_batches(is_sequential=True)
    ignore_prob = (1 - eps) / valid_data.max_local_entity
    pad = len(self.id2entity)
    for it in range(math.ceil(valid_data.num_data / test_batch_size)):
        batch = valid_data.get_batch(it, test_batch_size, fact_dropout=0.0, test=True)
        with torch.no_grad():
            _, _, pred_dist, _ = self.model(batch[:-1])
        local_entity, query_entities, _, _, _, _, _, answer_list = batch
        probs = pred_dist.double().cpu().numpy()
        for b in range(probs.shape[0]):
            keep = (query_entities[b] != 1.0) & (local_entity[b] != pad) & (probs[b] >= ignore_prob)
            _, _, f1, hit, em = _f1_hits_vec(probs[b], keep, set(answer_list[b]), eps)
            f1s.append(f1)
            hits.append(hit)
            ems.append(em)
    return np.mean(f1s), np.mean(hits), np.mean(ems)


def make_parser():
    """Parser ReaRev của repo gốc. `parsing.add_parse_args` gốc nổ NameError ở commit này
    (gọi create_parser_nutrea chưa định nghĩa) — nên dựng đúng nhánh ReaRev bằng hàm gốc."""
    import argparse
    import parsing
    p = argparse.ArgumentParser()
    parsing.create_parser_rearev(p.add_subparsers(help="Reason KGQA model").add_parser("ReaRev"))
    return p


def apply_patches():
    """Vá module của repo gốc (phải import được: gnn/ nằm trên sys.path)."""
    from modules.kg_reasoning.reasongnn import ReasonGNNLayer
    from modules.layer_init import TypeLayer
    from models.base_model import BaseModel
    from evaluate import Evaluator
    ReasonGNNLayer.build_matrix = _build_matrix
    ReasonGNNLayer.reason_layer = _reason_layer
    ReasonGNNLayer.reason_layer_inv = _reason_layer_inv
    TypeLayer.forward = _type_layer_forward
    BaseModel.calc_f1_new = _calc_f1_new
    Evaluator.evaluate = _evaluate


# ================================================================ tầng dữ liệu toàn KG
def rel_texts_for(rel_names, tokenizer, num_kb_relation):
    """Như BasicDataLoader.build_rel_words gốc (nhánh BERT) — trừ 2 điểm bắt buộc (docstring đầu file)."""
    words = [r.strip().split() for r in rel_names]                       # (2) tách khoảng trắng
    enc = [tokenizer(" ".join(w))["input_ids"] for w in words]
    enc_inv = [tokenizer(" ".join(w[::-1]))["input_ids"] for w in words]
    max_len = max(len(x) for x in enc + enc_inv)                         # (3) theo token
    pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    rt = np.full((num_kb_relation + 1, max_len), pad, dtype=int)
    rti = np.full((num_kb_relation + 1, max_len), pad, dtype=int)
    for i, (a, b) in enumerate(zip(enc, enc_inv)):
        rt[i, :len(a)] = a
        rti[i, :len(b)] = b
    return rt, rti


class FullKGLoader:
    """Thay SingleDataLoader gốc: cùng giao diện get_batch/reset_batches/thuộc tính mà Trainer,
    ReaRev và Evaluator dùng. Batch sinh tại chỗ (mảng (num_data, 129375) của gốc không vừa RAM)."""

    def __init__(self, rows, kg, tokenizer, rel_texts, rel_texts_inv):
        rows = [r for r in rows if r["topic_ids"]]                       # (5) như gốc: bỏ câu không topic
        self.rows = rows
        self.kg = kg
        self.num_data = len(rows)
        self.max_local_entity = kg.V
        self.num_kb_relation = kg.R
        self.rel_texts, self.rel_texts_inv = rel_texts, rel_texts_inv
        self.tokenizer = tokenizer
        self.num_word = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # Như gốc: độ dài câu = số TỪ dài nhất + 2 ([CLS], [SEP]), padding tới đó, cắt nếu dài hơn.
        self.max_query_word = max(len(r["query"].split(" ")) for r in rows) + 2
        self.query_texts = np.array([tokenizer(r["query"], max_length=self.max_query_word, padding="max_length",
                                               truncation=True)["input_ids"] for r in rows], dtype=int)
        self.question_id = [r["id"] for r in rows]
        self.answer_lists = np.empty(self.num_data, dtype=object)
        for i, r in enumerate(rows):
            self.answer_lists[i] = list(r["answer_ids"])
        self.batches = np.arange(self.num_data)

    def reset_batches(self, is_sequential=True):
        self.batches = np.arange(self.num_data) if is_sequential else np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        assert fact_dropout == 0, "fact_drop mặc định gốc = 0"
        ids = self.batches[batch_size * iteration: min(batch_size * (iteration + 1), self.num_data)]
        self.sample_ids = ids
        B, V = len(ids), self.kg.V
        local_entity = np.tile(np.arange(V), (B, 1))                     # (4) gồm cả topic
        query_entities = np.zeros((B, V), dtype=float)
        seed_dist = np.zeros((B, V), dtype=float)
        answer_dist = np.zeros((B, V), dtype=float)
        for b, i in enumerate(ids):
            tp = sorted(set(self.rows[i]["topic_ids"]))
            query_entities[b, tp] = 1.0
            seed_dist[b, tp] = 1.0 / len(tp)
            answer_dist[b, list(self.rows[i]["answer_ids"])] = 1.0
        out = (local_entity, query_entities, self.kg, self.query_texts[ids], seed_dist, None, answer_dist)
        return out + (self.answer_lists[ids],) if test else out

    def get_quest(self, training=False):
        return [self.rows[i]["query"] for i in self.sample_ids]


# ====================================================================== kernel (Kaggle)
def find_ds(must_have):
    for depth in range(1, 5):
        for c in sorted(glob.glob("/kaggle/input/" + "*/" * depth)):
            if os.path.exists(os.path.join(c, must_have)):
                return c.rstrip("/")
    raise FileNotFoundError(f"không thấy dataset chứa {must_have}")


def load_rows(ds, split):
    part = "train" if split in ("train", "val") else "test"
    ids = {int(x) for x in open(f"{ds}/splits/{split}.index").read().split()}
    rows = []
    for line in open(f"{ds}/Prime_{part}_QA_with_topic_entity.json", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if int(r["question_id"]) in ids:
                rows.append({"id": int(r["question_id"]), "query": r["question"],
                             "answer_ids": [int(a) for a in r["answer"]],
                             "topic_ids": [int(t["topic_id"]) for t in r["topic_entity"]]})
    assert len(rows) == len(ids), (split, len(rows), len(ids))
    return rows


def main():
    t_start = time.time()
    OUT = os.environ.get("OUT_DIR", "/kaggle/working")
    DS = os.environ.get("DS_DIR") or find_ds("edges.npz")
    SMOKE = int(os.environ.get("SMOKE", "0"))
    DEADLINE_H = float(os.environ.get("DEADLINE_H", "11.0"))
    NUM_EPOCH = int(os.environ.get("NUM_EPOCH", "100"))
    BATCH = int(os.environ.get("BATCH", "8"))
    MICRO = int(os.environ.get("MICRO", str(BATCH)))
    TEST_BATCH = int(os.environ.get("TEST_BATCH", "4"))
    RESUME = os.environ.get("RESUME", "0") == "1"
    SESSION_EPOCHS = int(os.environ.get("SESSION_EPOCHS", "0"))   # >0: dừng sau n epoch (chỉ để kiểm resume)
    os.makedirs(OUT, exist_ok=True)
    gnn_dir = fetch_repo(os.environ.get("REPO_DIR", "/kaggle/temp/GNN-RAG"))
    sys.path.insert(0, gnn_dir)
    os.chdir(gnn_dir)

    from utils import create_logger
    import train_model
    apply_patches()
    from transformers import AutoTokenizer

    args = make_parser().parse_args([
        "ReaRev", "--name", "cwq", "--data_folder", DS + "/", "--lm", "sbert", "--relation_word_emb", "True",
        "--entity_dim", "50", "--num_epoch", str(NUM_EPOCH), "--batch_size", str(BATCH), "--eval_every", "2",
        "--num_iter", "2", "--num_ins", "3", "--num_gnn", "3", "--test_batch_size", str(TEST_BATCH),
        "--checkpoint_dir", OUT + "/", "--experiment_name", "rearev"])
    args.use_cuda = torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = create_logger(args)
    dev = torch.device("cuda" if args.use_cuda else "cpu")

    z = np.load(f"{DS}/edges.npz")
    rel_names = json.load(open(f"{DS}/relations.json", encoding="utf-8"))
    kg = FullKG(z["src"], z["rel"], z["dst"], int(z["num_nodes"]), len(rel_names), dev)
    tok = AutoTokenizer.from_pretrained(LM_NAME)
    rt, rti = rel_texts_for(rel_names, tok, kg.R)
    rows = {s: load_rows(DS, s) for s in ("train", "val", "test")}
    if SMOKE:
        rows = {s: v[:SMOKE] for s, v in rows.items()}
    loaders = {s: FullKGLoader(v, kg, tok, rt, rti) for s, v in rows.items()}
    logger.info(f"KG: {kg.V:,} node · {kg.num_fact:,} fact (gồm self-loop) · R={kg.R} · "
                + " · ".join(f"{s} {l.num_data}" for s, l in loaders.items()))

    class Trainer(train_model.Trainer_KBQA):
        def load_data(self, a, tokenize):
            self.train_data, self.valid_data, self.test_data = loaders["train"], loaders["val"], loaders["test"]
            self.entity2id = {i: i for i in range(kg.V)}
            self.relation2id = {n: i for i, n in enumerate(rel_names)}
            self.word2id = {}
            self.num_word = loaders["test"].num_word
            self.num_kb_relation = kg.R
            self.num_entity = kg.V
            self.rel_texts, self.rel_texts_inv = rt, rti

        def train_epoch(self):
            """= train_epoch gốc; batch tách micro-batch, loss nhân len(micro)/len(batch)."""
            self.model.train()
            self.train_data.reset_batches(is_sequential=False)
            losses, h1_all, f1_all = [], [], []
            for it in range(math.ceil(self.train_data.num_data / self.args["batch_size"])):
                batch = self.train_data.get_batch(it, self.args["batch_size"], self.args["fact_drop"])
                B = batch[0].shape[0]
                self.optim_model.zero_grad()
                tot = 0.0
                for s in range(0, B, self.micro):
                    part = tuple(x if isinstance(x, FullKG) or x is None else x[s:s + self.micro] for x in batch)
                    loss, _, _, tp = self.model(part, training=True)
                    w = part[0].shape[0] / B
                    (loss * w).backward()
                    tot += loss.item() * w
                    h1_all.extend(tp[0])
                    f1_all.extend(tp[1])
                torch.nn.utils.clip_grad_norm_([p for _, p in self.model.named_parameters()],
                                               self.args["gradient_clip"])
                self.optim_model.step()
                losses.append(tot)
            return np.mean(losses), [0, 0], h1_all, f1_all

        def train(self, start_epoch, end_epoch):
            """= train gốc, bỏ chấm test mỗi lần eval; dừng khi chạm hạn giờ."""
            self.history = list(prev_hist)
            n_sess = 0
            for epoch in range(start_epoch, end_epoch + 1):
                st = time.time()
                loss, _, h1s, f1s = self.train_epoch()
                if self.decay_rate > 0:
                    self.scheduler.step()
                rec = {"epoch": epoch + 1, "loss": float(loss), "train_h1": float(np.mean(h1s)),
                       "train_f1": float(np.mean(f1s)), "sec": round(time.time() - st, 1)}
                self.logger.info(f"Epoch: {epoch + 1}, loss : {loss:.4f}, time: {time.time() - st}")
                if (epoch + 1) % self.args["eval_every"] == 0:
                    ev_f1, ev_h1, ev_em = self.evaluate(self.valid_data, self.test_batch_size)
                    rec.update(dev_f1=float(ev_f1), dev_h1=float(ev_h1))
                    self.logger.info(f"EVAL F1: {ev_f1:.4f}, H1: {ev_h1:.4f}, EM {ev_em:.4f}")
                    if epoch > self.warmup_epoch:
                        if ev_h1 > self.best_h1:
                            self.best_h1 = ev_h1
                            self.save_ckpt("h1")
                            self.logger.info(f"BEST EVAL H1: {ev_h1:.4f}")
                        if ev_f1 > self.best_f1:
                            self.best_f1 = ev_f1
                            self.save_ckpt("f1")
                self.history.append(rec)
                n_sess += 1
                json.dump(self.history, open(f"{OUT}/rearev_history.json", "w"), indent=1)
                spent = (time.time() - t_start) / 3600
                per_epoch = spent / n_sess
                if epoch < end_epoch and spent + 1.5 * per_epoch + 0.4 > DEADLINE_H:  # chừa chỗ suy luận test
                    self.logger.info(f"DỪNG theo hạn giờ: {spent:.2f}h sau {epoch + 1} epoch")
                    self.stopped_by_deadline = epoch + 1
                    break
                if SESSION_EPOCHS and n_sess >= SESSION_EPOCHS:
                    self.stopped_by_deadline = epoch + 1
                    break
            self.save_ckpt("final")
            self.save_state(self.history[-1]["epoch"] if self.history else 0)

        def save_state(self, epoch_done):
            torch.save({"model": self.model.state_dict(), "optim": self.optim_model.state_dict(),
                        "epoch": epoch_done, "best_h1": self.best_h1, "best_f1": self.best_f1,
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       f"{OUT}/rearev_state.pt")

    # Micro-batch: thử BATCH nguyên, OOM thì chia đôi (cùng gradient, xem train_epoch).
    micro = MICRO
    while True:
        try:
            torch.manual_seed(args.seed)
            trainer = Trainer(args=vars(args), model_name="ReaRev", logger=logger)
            trainer.micro = micro
            trainer.stopped_by_deadline = None
            if args.use_cuda:
                torch.cuda.reset_peak_memory_stats()
            b0 = loaders["train"].get_batch(0, args.batch_size, 0)
            loss, *_ = trainer.model(tuple(x if isinstance(x, FullKG) or x is None else x[:micro] for x in b0),
                                     training=True)
            loss.backward()
            trainer.optim_model.zero_grad()
            peak = torch.cuda.max_memory_allocated() / 2**30 if args.use_cuda else 0.0
            logger.info(f"micro-batch {micro}: thử 1 bước OK, đỉnh VRAM {peak:.2f} GB")
            break
        except torch.cuda.OutOfMemoryError:
            del trainer
            torch.cuda.empty_cache()
            assert micro > 1, "OOM cả với micro-batch 1"
            micro //= 2
            logger.info(f"OOM -> micro-batch {micro}")

    # ---- chạy tiếp từ phiên trước ----
    start_epoch, prev_hist, resume_mode = 0, [], None
    if RESUME:
        hs = sorted(glob.glob("/kaggle/input/**/rearev_history.json", recursive=True))
        rd = os.environ.get("RESUME_DIR") or (os.path.dirname(hs[0]) if hs else None)
        assert rd, "RESUME=1 nhưng không thấy output phiên trước (rearev_history.json)"
        prev_hist = json.load(open(f"{rd}/rearev_history.json"))
        for f in ("rearev-h1.ckpt", "rearev-f1.ckpt"):         # best của các phiên trước
            if os.path.exists(f"{rd}/{f}"):
                shutil.copy(f"{rd}/{f}", f"{OUT}/{f}")
        if os.path.exists(f"{rd}/rearev_state.pt"):
            st = torch.load(f"{rd}/rearev_state.pt", map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(st["model"])
            trainer.optim_model.load_state_dict(st["optim"])
            trainer.best_h1, trainer.best_f1, start_epoch = st["best_h1"], st["best_f1"], st["epoch"]
            np.random.set_state(st["np_rng"])
            torch.set_rng_state(st["torch_rng"])
            if st["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            resume_mode = "full (model + Adam + RNG)"
        else:
            # phiên tạo bởi code cũ (7011a10): chỉ có trọng số cuối, không có Adam/RNG
            trainer.load_ckpt(f"{rd}/rearev-final.ckpt")
            start_epoch = prev_hist[-1]["epoch"]
            dev = [h for h in prev_hist if "dev_h1" in h and h["epoch"] - 1 > trainer.warmup_epoch]
            trainer.best_h1 = max([h["dev_h1"] for h in dev], default=0.0)
            trainer.best_f1 = max([h["dev_f1"] for h in dev], default=0.0)
            np.random.seed(args.seed + start_epoch)
            resume_mode = "weights_only (Adam khởi tạo lại)"
        assert prev_hist and prev_hist[-1]["epoch"] == start_epoch, (len(prev_hist), start_epoch)
        logger.info(f"CHẠY TIẾP từ {rd}: epoch {start_epoch}/{args.num_epoch} · {resume_mode} · "
                    f"best dev H1 {trainer.best_h1:.4f}")

    t_train = time.time()
    if start_epoch < args.num_epoch:
        trainer.train(start_epoch, args.num_epoch - 1)
    else:
        trainer.history = list(prev_hist)
    train_h = (time.time() - t_train) / 3600

    # Suy luận test bằng checkpoint dev-H1 tốt nhất (đúng evaluate_best gốc, mục h1).
    ck = f"{OUT}/rearev-h1.ckpt"
    if os.path.exists(ck):
        trainer.load_ckpt(ck)
    model = trainer.model
    model.eval()
    te = loaders["test"]
    te.reset_batches(is_sequential=True)
    scores = np.zeros((te.num_data, kg.V), dtype=np.float32)
    for it in range(math.ceil(te.num_data / TEST_BATCH)):
        batch = te.get_batch(it, TEST_BATCH, 0.0, test=True)
        with torch.no_grad():
            _, _, pred_dist, _ = model(batch[:-1])
        scores[it * TEST_BATCH: it * TEST_BATCH + pred_dist.size(0)] = pred_dist.float().cpu().numpy()
    np.save(f"{OUT}/rearev_test_scores.npy", scores)
    np.save(f"{OUT}/rearev_test_qids.npy", np.array(te.question_id, dtype=np.int64))
    test_f1, test_h1, test_em = trainer.evaluate(te, TEST_BATCH)
    meta = {
        "code_sha": os.environ.get("CODE_SHA", "unknown"),   # commit repo này, dispatcher chèn vào
        "gnnrag_sha": GNNRAG_SHA, "lm": LM_NAME, "args": {k: v for k, v in vars(args).items()
                                                          if isinstance(v, (int, float, str, bool, type(None)))},
        "micro_batch": micro, "batch": args.batch_size, "smoke": SMOKE,
        "n": {s: l.num_data for s, l in loaders.items()},
        "kg": {"nodes": kg.V, "facts_incl_self_loop": kg.num_fact, "R_incl_self_loop": kg.R},
        "best_dev_h1": float(trainer.best_h1), "best_dev_f1": float(trainer.best_f1),
        "epochs_run": len(trainer.history), "stopped_by_deadline": trainer.stopped_by_deadline,
        "num_epoch": args.num_epoch, "done": bool(trainer.history) and trainer.history[-1]["epoch"] >= args.num_epoch,
        "resume_mode": resume_mode, "start_epoch_this_session": start_epoch,
        "train_hours": round(train_h, 3), "total_hours": round((time.time() - t_start) / 3600, 3),
        "own_eval_test_best_h1": {"f1": float(test_f1), "h1": float(test_h1), "em": float(test_em)},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if args.use_cuda else "cpu",
    }
    json.dump(meta, open(f"{OUT}/rearev_meta.json", "w"), indent=1)
    logger.info(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
