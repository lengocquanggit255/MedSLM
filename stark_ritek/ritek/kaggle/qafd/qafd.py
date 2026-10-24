"""QAFD-RAG retriever-only trên RiTeK — port của notebook tham chiếu ref_qafd/qafd-rag.ipynb (PharmKG) thành kernel.

Thuật toán: lớp `FlowDiffusionRetriever` CHÍNH THỨC của github.com/Tarzanagh/QAFD-RAG (ghim commit REPO_SHA; repo
không có LICENSE nên không chép vào repo này, kernel clone lúc chạy) + hai lớp con của notebook (bỏ warm-start;
trọng số cạnh dùng sim MedCPT thay HSIM) — chép nguyên văn từ notebook. Cấu hình y notebook: alpha 0.5 · confidence
0.5 · epsilon 0.01 · tối đa 500 lượt · step 0.2 · seed 42 · weight_func `add` · seed = topic entity RiTeK ·
sim = MedCPT-Query(câu hỏi) · MedCPT-Article([tên (loại), mô tả]) -> cosine -> min-max [0,1] trên toàn KG.
Metric y notebook mục 8: support = node có x > 0 (không cắt) · answer_recall/full_recall theo alias (id HOẶC tên)
· subgraph_node · subgraph_edge (cạnh của nx.Graph có cả hai đầu trong support) · elapsed.

KHÁC notebook — không đổi kết quả:
  - ghim commit repo (notebook clone HEAD lúc chạy);
  - nạp riêng src/utils/logging.py (logger "QAFD_RAG", thứ duy nhất flow_diffusion.py import từ src.utils) thay vì
    cả src/utils (khỏi cần tiktoken...);
  - cạnh chỉ giữ thuộc tính `weight` (thuật toán chỉ đọc weight; notebook lưu thêm chữ mô tả quan hệ);
  - đồ thị dựng từ triples.npz (đúng thứ tự dòng file KG) thay vì đọc file text — cùng thứ tự node/cạnh;
  - đếm subgraph_edge bằng mảng cặp node thay vì duyệt graph.edges() mỗi câu (cùng số; Prime có 4M cạnh);
  - embedding câu hỏi tính trước, TỪNG CÂU MỘT (batch 1, đúng như notebook gọi trong vòng lặp);
  - ghi kết quả từng câu ra jsonl thay vì ghi lại cả checkpoint json mỗi câu.
THÊM: recall theo id node (cột `*_id`) bên cạnh recall theo alias của notebook — bảng ritek/PLAN.md §6 dùng id,
như các dòng GNN-RAG/SubgraphRAG.

Biến môi trường: STAGES (vd "pharmkg,prime") · PRIME_LIMIT (0 = tất cả) · WORKERS (tiến trình song song, chỉ CPU)
· DEADLINE_H · EMB_DIR (thư mục output cũ có node_emb để khỏi encode lại; "auto" = tìm trong /kaggle/input) · RESUME_DIR (output cũ: bỏ câu đã chạy)
· DATA_DIR · OUT_DIR · REPO_DIR · CODE_SHA. Dataset: ritek-qafd-data (ritek/scripts/prep_qafd_data.py).
S0 (stage pharmkg): so số tổng hợp với số notebook đã in (NOTEBOOK_PHARMKG) để chứng minh port y hệt.
"""
import glob
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import types
from collections import defaultdict as _dd
from pathlib import Path

import networkx as nx
import numpy as np

REPO_URL = "https://github.com/Tarzanagh/QAFD-RAG.git"
REPO_SHA = "2a3a42c8ea4e9c416de0a7fe53c04fab50b756ef"

# ===== cấu hình notebook (ref_qafd/qafd-rag.ipynb mục 2) =====
MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ENTITY_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MAX_LEN = 512
ENTITY_MAX_LEN = 512
ENTITY_BATCH_SIZE = 64
SIM_METRIC = "cosine"
ALPHA = 0.5
MAX_ITERS = 500
EPSILON = 0.01
WEIGHT_FUNC = "add"
RANDOM_SEED = 42
CONFIDENCE = 0.5
STEP_SIZE = 0.2
QUERY_FORMAT = "auto"
TOPIC_ENTITY_INDEX = 0
# số notebook đã in cho PharmKG test, weight_func add (mục 9)
NOTEBOOK_PHARMKG = {"n_valid_queries": 1023, "avg_answer_recall": 0.737048, "full_recall_rate": 0.733138,
                    "avg_subgraph_node": 204.083089, "avg_subgraph_edge": 5548.934506}

STAGES = [s for s in os.environ.get("STAGES", "pharmkg,prime").split(",") if s]
PRIME_LIMIT = int(os.environ.get("PRIME_LIMIT", "0"))
WORKERS = int(os.environ.get("WORKERS", "1"))
DEADLINE_H = float(os.environ.get("DEADLINE_H", "11.5"))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def rss_peak_gb():
    try:
        import resource   # chỉ có trên Linux (Kaggle); chạy thử trên Windows thì bỏ qua
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except ImportError:
        return float("nan")


def find_data():
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    for c in glob.glob("/kaggle/input/**/pharmkg/triples.npz", recursive=True):
        return Path(c).parent.parent
    raise FileNotFoundError("không thấy dataset ritek-qafd-data")


def fetch_repo(dest):
    dest = Path(dest)
    if not (dest / "src" / "retrievers" / "flow_diffusion.py").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for c in (["git", "init", "-q"], ["git", "remote", "add", "origin", REPO_URL],
                  ["git", "fetch", "-q", "--depth", "1", "origin", REPO_SHA],
                  ["git", "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(c, cwd=dest, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
    assert head == REPO_SHA, f"repo QAFD-RAG ở {head}, cần {REPO_SHA}"
    return dest


# ===== import retriever chính thức (notebook mục 5; chỉ nạp src/utils/logging.py) =====
def load_official_retriever(repo_root):
    src_dir = repo_root / "src"
    for name, path in (("src", src_dir), ("src.retrievers", src_dir / "retrievers"), ("src.utils", src_dir / "utils")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    lg = load("src.utils.logging", src_dir / "utils" / "logging.py")
    sys.modules["src.utils"].logger = lg.logger
    load("src.retrievers.base", src_dir / "retrievers" / "base.py")
    flow = load("src.retrievers.flow_diffusion", src_dir / "retrievers" / "flow_diffusion.py")
    return flow.FlowDiffusionRetriever, lg.logger


class NonConvergeCounter(logging.Handler):
    """Đếm cảnh báo 'QAFD did not converge' của repo (notebook chỉ in ra)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "did not converge" in record.getMessage():
            self.count += 1


def make_classes(FlowDiffusionRetriever):
    # ===== NO-WARM-START VARIANT (chép nguyên notebook) =====
    class NoWarmStartFlowDiffusionRetriever(FlowDiffusionRetriever):
        def initialize(self, alpha: float = 50, use_node_degree: bool = True):
            for node in self.graph.nodes():
                if use_node_degree:
                    self.sink_capacity[node] = max(self.graph.degree(node), 1)
                else:
                    self.sink_capacity[node] = 1

            total_sink = sum(self.sink_capacity.values())
            if total_sink > 0:
                for node in self.sink_capacity:
                    self.sink_capacity[node] = 10.0 * self.sink_capacity[node] / total_sink
            total_sink = sum(self.sink_capacity.values())

            for node in self.graph.nodes():
                self.mass[node] = 0

            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * total_sink * confidence_boost

            # KHÔNG warm-start: x rỗng.
            self.x = _dd(float)

    # ===== MEDCPT QUERY-AWARE VARIANT (chép nguyên notebook) =====
    class MedCPTNoWarmStartFlowDiffusionRetriever(NoWarmStartFlowDiffusionRetriever):
        def __init__(self, *args, node_sim_scores=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.node_sim_scores = node_sim_scores or {}

        def get_edge_weight(self, node1: str, node2: str) -> float:
            cache_key = (node1, node2)
            if cache_key in self.edge_weights_cache:
                return self.edge_weights_cache[cache_key]

            edge_data = self.graph[node1][node2]
            original_weight = float(edge_data.get("weight", 1.0))

            if original_weight <= 0:
                self.edge_weights_cache[cache_key] = 0.0
                return 0.0

            node1_query_score = float(self.node_sim_scores.get(node1, 0.0))
            node2_query_score = float(self.node_sim_scores.get(node2, 0.0))

            if self.weight_func == "multiply":
                smart_weight = original_weight * node1_query_score * node2_query_score
            elif self.weight_func == "add":
                smart_weight = (original_weight + node1_query_score + node2_query_score) / 3.0
            else:
                query_factor = (node1_query_score + node2_query_score) / 2.0
                smart_weight = original_weight * (1.0 + query_factor * 0.5)

            self.edge_weights_cache[cache_key] = smart_weight
            return smart_weight

    return MedCPTNoWarmStartFlowDiffusionRetriever


# ===== dữ liệu (notebook mục 3 + 6) =====
def load_questions(path):
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    obj = json.loads(raw) if raw.startswith("[") else [json.loads(x) for x in raw.splitlines() if x.strip()]
    if not isinstance(obj, list):
        raise ValueError("Question JSON root phải là list")
    return obj


def build_graph(ds_dir):
    """= build_custom_kg + mục 6 của notebook: node theo lần xuất hiện đầu tiên (h rồi t, từng dòng), cạnh theo thứ
    tự dòng, weight 1.0; nx.Graph gộp cạnh lặp/ngược chiều."""
    z = np.load(ds_dir / "triples.npz")
    h, t = z["h"], z["t"]
    node_info = json.loads((ds_dir / "node_info.json").read_text(encoding="utf-8"))
    ht = np.empty(2 * len(h), dtype=np.int64)
    ht[0::2], ht[1::2] = h, t
    _, first = np.unique(ht, return_index=True)
    order = ht[np.sort(first)]
    graph = nx.Graph()
    for nid in order.tolist():
        sid = str(nid)
        info = node_info.get(sid) or {}
        graph.add_node(sid, entity_name=str(info.get("name", info.get("node_name", sid))),
                       entity_type=str(info.get("type", info.get("node_type", "entity"))),
                       description=str(info.get("description", info.get("desc", ""))), source_id=sid)
    for u, v in zip(h.tolist(), t.tolist()):
        graph.add_edge(str(u), str(v), weight=1.0)
    return graph, len(h)


class MedCPTDualEncoder:
    """Chép nguyên notebook: Query-Encoder và Article-Encoder là 2 model độc lập."""

    def __init__(self, query_model_name, entity_model_name, device, query_max_len=64, entity_max_len=512,
                 metric="cosine"):
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.query_max_len = int(query_max_len)
        self.entity_max_len = int(entity_max_len)
        self.metric = metric
        self.q_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.q_model = AutoModel.from_pretrained(query_model_name).to(device).eval()
        self.e_tokenizer = AutoTokenizer.from_pretrained(entity_model_name)
        self.e_model = AutoModel.from_pretrained(entity_model_name).to(device).eval()

    def _post(self, emb):
        import torch
        if self.metric == "cosine":
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def encode_queries(self, queries):
        import torch
        with torch.no_grad():
            if isinstance(queries, str):
                queries = [queries]
            enc = self.q_tokenizer(list(queries), truncation=True, padding=True, return_tensors="pt",
                                   max_length=self.query_max_len).to(self.device)
            out = self.q_model(**enc).last_hidden_state[:, 0, :]  # CLS
            return self._post(out).float().cpu().numpy()

    def encode_entities(self, titles, texts, batch_size=64, show_progress=True):
        import torch
        with torch.no_grad():
            n = len(titles)
            order = sorted(range(n), key=lambda i: len(titles[i]) + len(texts[i]))
            out = np.zeros((n, self.e_model.config.hidden_size), dtype=np.float32)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                enc = self.e_tokenizer([titles[i] for i in idx], [texts[i] for i in idx], truncation=True,
                                       padding=True, return_tensors="pt", max_length=self.entity_max_len).to(self.device)
                emb = self.e_model(**enc).last_hidden_state[:, 0, :]  # CLS
                out[idx] = self._post(emb).float().cpu().numpy()
                if show_progress and ((start // batch_size) % 200 == 0):
                    log(f"  encoded {min(start + batch_size, n):,}/{n:,} entities")
            return out


def minmax_scale_01(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ===== chuẩn hoá query + đánh giá (notebook mục 7, chép nguyên phần logic) =====
def _norm_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)


def normalize_topic_entity(topic_entity, query_format="auto", index=0):
    if isinstance(topic_entity, dict):
        return {"topic_id": topic_entity.get("topic_id"), "topic_name": topic_entity.get("topic_name")}
    if isinstance(topic_entity, list):
        if len(topic_entity) == 0:
            raise ValueError("topic_entity là list rỗng.")
        if index < 0 or index >= len(topic_entity):
            raise IndexError(f"TOPIC_ENTITY_INDEX={index} vượt phạm vi ({len(topic_entity)} elements).")
        selected = topic_entity[index]
        if not isinstance(selected, dict):
            raise ValueError(f"topic_entity[{index}] phải là dict, nhận được {type(selected).__name__}.")
        return {"topic_id": selected.get("topic_id"), "topic_name": selected.get("topic_name")}
    if isinstance(topic_entity, str):
        topic_name = topic_entity.strip()
        if not topic_name:
            raise ValueError("topic_entity là string rỗng.")
        return {"topic_id": None, "topic_name": topic_name}
    raise ValueError(f"topic_entity không hợp lệ: {type(topic_entity).__name__}.")


def normalize_gold_answers(rec):
    if "gold_answers" in rec:
        answers = rec.get("gold_answers", [])
        if answers is None:
            return []
        return answers if isinstance(answers, list) else [answers]
    if "answer" in rec:
        answer = rec.get("answer", {})
        if answer is None:
            return []
        if isinstance(answer, dict):
            return [{"answer_id": str(a), "answer_name": n} for a, n in answer.items()]
        return answer if isinstance(answer, list) else [answer]
    return []


def normalize_query_record(rec):
    for req_field in ["question_id", "question", "topic_entity"]:
        if req_field not in rec:
            raise KeyError(f"Thiếu trường bắt buộc: {req_field}")
    normalized = dict(rec)
    normalized["topic_entity"] = normalize_topic_entity(rec["topic_entity"], QUERY_FORMAT, TOPIC_ENTITY_INDEX)
    normalized["gold_answers"] = normalize_gold_answers(rec)
    return normalized


def resolve_topic_entity_seed(topic_entity, graph):
    topic_id = topic_entity.get("topic_id")
    topic_name = topic_entity.get("topic_name")
    for cand in [topic_id, topic_name]:
        if cand is None:
            continue
        sc = str(cand)
        for node_id, attrs in graph.nodes(data=True):
            aliases = {str(node_id), str(attrs.get("source_id", "")), str(attrs.get("entity_name", "")),
                       str(attrs.get("name", ""))}
            if sc in aliases:
                return node_id
    raise KeyError(f"Không resolve được topic_entity: topic_id={topic_id!r}, topic_name={topic_name!r}")


def _answer_aliases(answer):
    if isinstance(answer, dict):
        aliases = set()
        ans_id = answer.get("answer_id")
        ans_name = answer.get("answer_name")
        if ans_id is not None:
            aliases.update({str(ans_id), str(_norm_id(ans_id))})
        if ans_name is not None:
            aliases.update({str(ans_name), str(_norm_id(ans_name))})
        return aliases
    return {str(answer), str(_norm_id(answer))}


def node_aliases(node_id, attrs):
    vals = {str(node_id)}
    for key in ("source_id", "entity_name", "name"):
        val = attrs.get(key)
        if val is not None:
            vals.update({str(val), str(_norm_id(val))})
    return vals


def evaluate(scores, graph, gold_answers, pair_u, pair_v, node_idx):
    """= evaluate_retrieval của notebook (MAX_NODE = None) + recall theo id; cạnh đếm bằng mảng cặp node."""
    support_nodes = {node for node, score in scores.items() if float(score) > 0}
    mark = np.zeros(len(node_idx), dtype=bool)
    if support_nodes:
        mark[[node_idx[n] for n in support_nodes]] = True
    n_edges = int((mark[pair_u] & mark[pair_v]).sum())
    support_aliases = set()
    for node in support_nodes:
        support_aliases.update(node_aliases(node, dict(graph.nodes[node])))
    n_ans = len(gold_answers)
    found = sum(1 for ans in gold_answers if (_answer_aliases(ans) & support_aliases))
    found_id = sum(1 for ans in gold_answers if str(ans.get("answer_id")) in support_nodes)
    return dict(n_support_nodes=len(support_nodes), n_support_edges=n_edges, n_answers=n_ans,
                answer_recall=found / n_ans if n_ans else None,
                full_recall=int(n_ans > 0 and found == n_ans) if n_ans else None,
                answer_recall_id=found_id / n_ans if n_ans else None,
                full_recall_id=int(n_ans > 0 and found_id == n_ans) if n_ans else None)


# ===== chạy một bộ =====
_W = {}   # trạng thái dùng chung cho tiến trình con (fork)


def run_query(rec):
    g, Retr, counter = _W["graph"], _W["cls"], _W["counter"]
    qid = rec["question_id"]
    started = time.time()
    row = {"question_id": qid, "status": "error", "topic_id": rec["topic_entity"].get("topic_id")}
    try:
        seed = resolve_topic_entity_seed(rec["topic_entity"], g)
        raw = _W["node_emb"] @ _W["q_emb"][str(qid)]
        scaled = minmax_scale_01(raw)
        node_sim_scores = {nid: float(scaled[i]) for i, nid in enumerate(_W["node_ids"])}
        before = counter.count
        retriever = Retr(graph=g, source_node=seed, target_node=seed, confidence=CONFIDENCE, epsilon=EPSILON,
                         node_embeddings=None, subquery_embedding=None, node_sim_scores=node_sim_scores,
                         weight_func=None if WEIGHT_FUNC == "original" else WEIGHT_FUNC, step_size=STEP_SIZE,
                         random_seed=RANDOM_SEED)
        retriever.retrieve(alpha=ALPHA, max_iterations=MAX_ITERS)
        scores = retriever.get_node_scores()
        m = evaluate(scores, g, rec["gold_answers"], _W["pair_u"], _W["pair_v"], _W["node_idx"])
        sup = sorted(((n, float(s)) for n, s in scores.items() if float(s) > 0), key=lambda kv: -kv[1])
        row.update(status="success", selected_seed=seed, converged=counter.count == before, support=sup, **m)
    except Exception as e:  # noqa: BLE001 — như notebook: ghi lỗi, đi tiếp
        row.update(error_type=type(e).__name__, error_message=str(e))
    row["elapsed_sec"] = time.time() - started
    return row


def worker(k, recs, path):
    with open(path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(recs):
            if (time.time() - T0) / 3600 > DEADLINE_H:
                log(f"  [w{k}] hết hạn phiên sau {i} câu")
                return
            f.write(json.dumps(run_query(rec), ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                log(f"  [w{k}] {i + 1}/{len(recs)} · {(time.time() - T0) / 3600:.2f}h")


def summarize(rows, n_skipped):
    ok = [r for r in rows if r["status"] == "success"]
    mean = lambda k: float(np.mean([r[k] for r in ok if r.get(k) is not None])) if ok else None  # noqa: E731
    return {"n_valid_queries": len(ok), "n_error": len(rows) - len(ok), "n_skipped_queries": n_skipped,
            "avg_answer_recall": mean("answer_recall"), "full_recall_rate": mean("full_recall"),
            "avg_answer_recall_id": mean("answer_recall_id"), "full_recall_rate_id": mean("full_recall_id"),
            "avg_subgraph_node": mean("n_support_nodes"), "avg_subgraph_edge": mean("n_support_edges"),
            "avg_elapsed_sec": mean("elapsed_sec"), "n_not_converged": sum(not r.get("converged", True) for r in ok)}


def run_stage(ds, data, Retr, counter, device):
    import torch
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    graph, n_rel = build_graph(data / ds)
    log(f"[{ds}] đồ thị {graph.number_of_nodes():,} node · {graph.number_of_edges():,} cạnh "
        f"(từ {n_rel:,} triple) · {time.time() - t:.0f}s · RSS đỉnh {rss_peak_gb():.1f} GB")
    node_ids = list(graph.nodes())
    node_idx = {n: i for i, n in enumerate(node_ids)}
    pu, pv = zip(*((node_idx[u], node_idx[v]) for u, v in graph.edges()))
    pair_u, pair_v = np.asarray(pu, dtype=np.int64), np.asarray(pv, dtype=np.int64)

    questions = load_questions(data / ds / "test.jsonl")
    if ds == "prime" and PRIME_LIMIT:
        questions = questions[:PRIME_LIMIT]
    records, skipped = [], []
    for i, raw in enumerate(questions):
        try:
            records.append(normalize_query_record(raw))
        except Exception as e:  # noqa: BLE001
            skipped.append({"sample_index": i, "question_id": raw.get("question_id"),
                            "error_type": type(e).__name__, "error_message": str(e)})
    log(f"[{ds}] {len(questions):,} câu · hợp lệ {len(records):,} · bỏ {len(skipped):,}")

    encoder = MedCPTDualEncoder(MEDCPT_QUERY_MODEL, MEDCPT_ENTITY_MODEL, device, query_max_len=QUERY_MAX_LEN,
                                entity_max_len=ENTITY_MAX_LEN, metric=SIM_METRIC)
    emb_dir = os.environ.get("EMB_DIR")
    if emb_dir == "auto":                       # output kernel trước gắn làm input (job["kernels"])
        hits = glob.glob(f"/kaggle/input/**/{ds}/node_emb.npy", recursive=True)
        emb_dir = str(Path(hits[0]).parent.parent) if hits else None
    emb_src = Path(emb_dir) / ds if emb_dir else None
    if emb_src and (emb_src / "node_emb.npy").exists() and \
            json.loads((emb_src / "node_ids.json").read_text()) == node_ids:
        node_emb = np.load(emb_src / "node_emb.npy")
        log(f"[{ds}] dùng lại node_emb từ {emb_src}")
    else:
        t = time.time()
        titles = [f"{graph.nodes[n].get('entity_name', n)} ({graph.nodes[n].get('entity_type', 'entity')})"
                  for n in node_ids]
        descs = [str(graph.nodes[n].get("description", "")) for n in node_ids]
        node_emb = encoder.encode_entities(titles, descs, batch_size=ENTITY_BATCH_SIZE)
        log(f"[{ds}] encode {len(node_ids):,} node trong {time.time() - t:.0f}s")
    np.save(out / "node_emb.npy", node_emb)
    (out / "node_ids.json").write_text(json.dumps(node_ids))
    t = time.time()
    q_emb = {str(r["question_id"]): encoder.encode_queries(r["question"])[0] for r in records}   # batch 1 như notebook
    log(f"[{ds}] encode {len(q_emb):,} câu hỏi (từng câu) trong {time.time() - t:.0f}s")
    del encoder
    torch.cuda.empty_cache()

    done = {}
    if os.environ.get("RESUME_DIR"):
        for p in glob.glob(f"{os.environ['RESUME_DIR']}/{ds}/results_w*.jsonl"):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                done[str(r["question_id"])] = r
        log(f"[{ds}] chạy tiếp: {len(done):,} câu đã có")
    todo = [r for r in records if str(r["question_id"]) not in done]
    _W.update(graph=graph, cls=Retr, counter=counter, node_emb=node_emb, q_emb=q_emb, node_ids=node_ids,
              node_idx=node_idx, pair_u=pair_u, pair_v=pair_v)
    t = time.time()
    if WORKERS <= 1:
        worker(0, todo, out / "results_w0.jsonl")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=worker, args=(k, todo[k::WORKERS], out / f"results_w{k}.jsonl"))
                 for k in range(WORKERS)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    rows = list(done.values())
    for p in sorted(out.glob("results_w*.jsonl")):
        rows += [json.loads(line) for line in open(p, encoding="utf-8")]
    summ = summarize(rows, len(skipped))
    summ.update(stage=ds, run_sec=round(time.time() - t, 1), n_queries_run=len(todo), workers=WORKERS,
                rss_peak_gb=round(rss_peak_gb(), 2),
                config=dict(alpha=ALPHA, max_iters=MAX_ITERS, epsilon=EPSILON, weight_func=WEIGHT_FUNC,
                            random_seed=RANDOM_SEED, confidence=CONFIDENCE, step_size=STEP_SIZE, warm_start=False,
                            sim=f"medcpt_{SIM_METRIC}_minmax_01", repo_sha=REPO_SHA),
                code_sha=os.environ.get("CODE_SHA"))
    if ds == "pharmkg":
        summ["notebook"] = NOTEBOOK_PHARMKG
        summ["diff_vs_notebook"] = {k: summ[k] - v for k, v in NOTEBOOK_PHARMKG.items() if summ.get(k) is not None}
    (out / "skipped_queries.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=1))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    log(f"[{ds}] TỔNG HỢP " + json.dumps(summ, ensure_ascii=False))


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    data = find_data()
    repo = fetch_repo(os.environ.get("REPO_DIR", "/kaggle/temp/QAFD-RAG"))
    Retr0, qlogger = load_official_retriever(repo)
    counter = NonConvergeCounter()
    qlogger.setLevel(logging.WARNING)
    qlogger.addHandler(counter)
    qlogger.propagate = False
    Retr = make_classes(Retr0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"repo QAFD-RAG @ {REPO_SHA[:7]} · device {device} · STAGES {STAGES} · PRIME_LIMIT {PRIME_LIMIT} · WORKERS {WORKERS}")
    for ds in STAGES:
        run_stage(ds, data, Retr, counter, device)
    log("QAFD_DONE")


if __name__ == "__main__":
    main()
