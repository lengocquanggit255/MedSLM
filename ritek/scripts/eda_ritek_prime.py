"""So sánh STaRK-Prime gốc (data/stark/prime) với bản Prime do RiTeK phát hành (data/ritek/Dataset/prime).

Chạy: python ritek/scripts/eda_ritek_prime.py [--out ritek/docs/ritek_prime_compare.json]

RiTeK phát hành Prime gồm:
  prime_KG.txt                          head \\t rel \\t tail — ID node/quan hệ của STaRK, KHÔNG có text node
  Prime_{train,test}_QA_with_topic_entity.json   question_id · question · answer{id: tên} · topic_entity[]
  Prime_{train,test}_QA_instruction.json         cùng query dạng instruction (context/response), không ID
Không có split val.

So sánh:
  1. Đồ thị: cạnh có trùng khớp edge_index/edge_types của STaRK không.
  2. Query: question_id có thuộc STaRK không, thuộc split nào, câu hỏi/đáp án có bị sửa không.
  3. topic_entity (STaRK KHÔNG phát hành): số lượng, khớp tên node, khoảng cách tới đáp án.
  4. Query nào của STaRK bị bỏ, và nhóm bị bỏ khác nhóm giữ lại thế nào.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.skb import PrimeSKB  # noqa: E402

STARK = ROOT / "data/stark/prime"
RITEK = ROOT / "data/ritek/Dataset/prime"
STARK_SPLITS = ("train", "val", "test")
RITEK_SPLITS = ("train", "test")


def jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


# ----------------------------------------------------------------------------- 1. đồ thị
def compare_kg(skb: PrimeSKB) -> dict:
    kg = pd.read_csv(RITEK / "prime_KG.txt", sep="\t", header=None, names=["h", "r", "t"],
                     dtype=np.int64).to_numpy()
    src = skb.edge_index[0].numpy()
    dst = skb.edge_index[1].numpy()
    et = skb.edge_types.numpy()
    n = skb.num_nodes
    same_shape = kg.shape[0] == src.shape[0]
    in_order = bool(same_shape and (kg[:, 0] == src).all() and (kg[:, 1] == et).all()
                    and (kg[:, 2] == dst).all())
    # So như multiset: mã hoá (h, r, t) thành một số nguyên rồi sort.
    R = 18

    def key(h, r, t):
        return (h * R + r) * n + t
    k_r = np.sort(key(kg[:, 0], kg[:, 1], kg[:, 2]))
    k_s = np.sort(key(src, et, dst))
    only_r = np.setdiff1d(k_r, k_s, assume_unique=False)
    only_s = np.setdiff1d(k_s, k_r, assume_unique=False)
    return {
        "n_edges": {"stark": int(src.shape[0]), "ritek": int(kg.shape[0])},
        "identical_in_order": in_order,
        "identical_as_multiset": bool(k_r.shape == k_s.shape and (k_r == k_s).all()),
        "edges_only_in_ritek": int(only_r.size),
        "edges_only_in_stark": int(only_s.size),
        "ritek_rel_ids": sorted(int(x) for x in np.unique(kg[:, 1])),
        "ritek_node_id_range": [int(kg[:, [0, 2]].min()), int(kg[:, [0, 2]].max())],
        "stark_num_nodes": n,
        "ritek_has_node_text": False,   # chỉ có file cạnh; tên node chỉ xuất hiện trong QA
    }


# --------------------------------------------------------------- khoảng cách topic -> đáp án
class Hops:
    """Khoảng cách vô hướng (0,1,2,3,>3) từ topic tới từng đáp án, không BFS toàn đồ thị.

    d<=2: tra N1(t), N2(t). d==3: N2(t) giao N1(a) khác rỗng. Chỉ mở rộng 2 bước từ topic.
    """

    def __init__(self, skb: PrimeSKB):
        n = skb.num_nodes
        src = skb.edge_index[0].numpy()
        dst = skb.edge_index[1].numpy()
        a = sp.csr_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
        a.sum_duplicates()
        self.a = a
        self.n = n
        self._cache = {}

    def nb(self, i):
        return self.a.indices[self.a.indptr[i]:self.a.indptr[i + 1]]

    def _balls(self, t):
        if t not in self._cache:
            n1 = self.nb(t)
            m1 = np.zeros(self.n, dtype=bool)
            m1[n1] = True
            m2 = m1.copy()
            if n1.size:
                m2[self.a[n1].indices] = True
            m2[t] = True
            self._cache = {t: (m1, m2)}   # chỉ giữ 1 topic: mỗi mask 129k byte
        return self._cache[t]

    def dist(self, t, a):
        if a == t:
            return 0
        m1, m2 = self._balls(t)
        if m1[a]:
            return 1
        if m2[a]:
            return 2
        return 3 if m2[self.nb(a)].any() else 4   # 4 = ">3"


# ------------------------------------------------------------------------------- 2-3. query
def compare_queries(skb: PrimeSKB, hops: Hops) -> tuple[dict, dict]:
    full = pd.read_csv(STARK / "stark_qa/stark_qa.csv")
    full["answer_ids"] = full["answer_ids"].apply(eval)
    stark = {int(r.id): (r.query, [int(x) for x in r.answer_ids]) for r in full.itertuples()}
    split_of = {}
    for s in STARK_SPLITS:
        for x in (STARK / "split" / f"{s}.index").read_text().split():
            split_of[int(x)] = s

    names = skb._names
    ntype = skb.node_types.numpy()
    tname = skb.node_type_dict
    out, ritek_ids = {}, {}
    for rs in RITEK_SPLITS:
        rows = jsonl(RITEK / f"Prime_{rs}_QA_with_topic_entity.json")
        instr = jsonl(RITEK / f"Prime_{rs}_QA_instruction.json")
        qids = [int(r["question_id"]) for r in rows]
        ritek_ids[rs] = set(qids)
        c = Counter()
        cross = Counter()
        n_topics = Counter()
        topic_types = Counter()
        hop_min = Counter()          # khoảng cách topic -> đáp án GẦN nhất
        hop_all = Counter()          # khoảng cách topic -> TỪNG đáp án
        ex = {"question_changed": [], "answers_changed": [], "answer_name_mismatch": [],
              "topic_name_mismatch": [], "far_topic": []}
        for r, ins in zip(rows, instr):
            q = int(r["question_id"])
            ans = {int(k): v for k, v in r["answer"].items()}
            topics = [(int(t["topic_id"]), t["topic_name"]) for t in r["topic_entity"]]
            cross[split_of.get(q, "not_in_stark")] += 1
            # instruction file: cùng thứ tự, cùng câu hỏi, response = tên đáp án nối bằng dấu phẩy
            c["instr_same_question"] += ins["context"] == r["question"]
            # không strip: vài tên node gốc có khoảng trắng cuối ("Platelet degranulation ")
            c["instr_same_answers"] += ins["response"] == ",".join(ans.values()) + "\n"
            if q not in stark:
                continue
            sq, sa = stark[q]
            if sq == r["question"]:
                c["question_identical"] += 1
            elif " ".join(sq.split()) == " ".join(r["question"].split()):
                c["question_whitespace_only"] += 1
            else:
                c["question_changed"] += 1
                ex["question_changed"].append({"id": q, "stark": sq, "ritek": r["question"]})
            ra, sa = set(ans), set(sa)
            kind = ("answers_identical" if ra == sa else "answers_ritek_subset" if ra < sa
                    else "answers_ritek_superset" if ra > sa else "answers_overlap" if ra & sa
                    else "answers_disjoint")
            c[kind] += 1
            if kind != "answers_identical":
                ex["answers_changed"].append({"id": q, "stark": sorted(sa), "ritek": sorted(ra)})
            for a, nm in ans.items():
                ok = names[a] == nm
                c["answer_names_match"] += ok
                c["answer_names_total"] += 1
                if not ok:
                    ex["answer_name_mismatch"].append({"id": q, "node": a, "stark": names[a], "ritek": nm})
            n_topics[len(topics)] += 1
            ds = []
            for t, nm in topics:
                c["topics_total"] += 1
                if not 0 <= t < skb.num_nodes:
                    c["topic_id_invalid"] += 1
                    continue
                ok = names[t] == nm
                c["topic_names_match"] += ok
                if not ok:
                    ex["topic_name_mismatch"].append({"id": q, "node": t, "stark": names[t], "ritek": nm})
                topic_types[tname[int(ntype[t])]] += 1
                c["topic_is_answer"] += t in ans
                # topic có được nhắc NGUYÊN VĂN trong câu hỏi không (không phân biệt hoa thường)
                c["topic_name_in_question"] += nm.strip().lower() in r["question"].lower()
                ds.append([hops.dist(t, a) for a in ans])
            if ds:
                # topic tốt nhất của query (nhiều topic: lấy topic gần đáp án nhất)
                best = min(ds, key=min)
                hop_min[min(best)] += 1
                for d in best:
                    hop_all[d] += 1
                if min(best) == 4:
                    ex["far_topic"].append({"id": q, "question": r["question"], "topics": topics,
                                            "answers": list(ans.items())[:5]})
        lab = {0: "0", 1: "1", 2: "2", 3: "3", 4: ">3"}
        out[rs] = {
            "n": len(rows),
            "n_unique_ids": len(set(qids)),
            "instruction_rows": len(instr),
            "by_stark_split": dict(cross),
            **{k: int(v) for k, v in sorted(c.items())},
            "n_topics_per_query": {str(k): v for k, v in sorted(n_topics.items())},
            "topic_node_types": dict(topic_types.most_common()),
            "hop_topic_to_nearest_answer": {lab[k]: hop_min[k] for k in sorted(hop_min)},
            "hop_topic_to_each_answer": {lab[k]: hop_all[k] for k in sorted(hop_all)},
            "examples": {k: v[:5] for k, v in ex.items()},
        }
    return out, ritek_ids


# -------------------------------------------------------------------- 4. query STaRK bị bỏ
def coverage(skb: PrimeSKB, ritek_ids: dict) -> dict:
    full = pd.read_csv(STARK / "stark_qa/stark_qa.csv")
    full["answer_ids"] = full["answer_ids"].apply(eval)
    ans = {int(r.id): [int(x) for x in r.answer_ids] for r in full.itertuples()}
    ntype = skb.node_types.numpy()
    tname = skb.node_type_dict
    in_r = ritek_ids["train"] | ritek_ids["test"]
    res = {}
    for s in STARK_SPLITS + ("test-0.1",):
        ids = [int(x) for x in (STARK / "split" / f"{s}.index").read_text().split()]
        kept = [i for i in ids if i in in_r]
        drop = [i for i in ids if i not in in_r]

        def prof(group):
            if not group:
                return None
            na = np.array([len(ans[i]) for i in group])
            maj = Counter(tname[int(Counter(ntype[ans[i]].tolist()).most_common(1)[0][0])] for i in group)
            return {"n": len(group), "mean_n_answers": round(float(na.mean()), 2),
                    "single_answer_pct": pct(int((na == 1).sum()), len(group)),
                    "answer_type_pct": {k: pct(v, len(group)) for k, v in maj.most_common()}}
        res[s] = {"n": len(ids),
                  "in_ritek_train": sum(i in ritek_ids["train"] for i in ids),
                  "in_ritek_test": sum(i in ritek_ids["test"] for i in ids),
                  "in_neither": len(drop),
                  "kept_profile": prof(kept), "dropped_profile": prof(drop)}
    # Tập human có không gian ID RIÊNG (0–97, trùng số với query synthetic) -> so theo text, không theo ID.
    h = pd.read_csv(STARK / "stark_qa/stark_qa_human_generated_eval.csv")
    rq = {r["question"] for s in RITEK_SPLITS
          for r in jsonl(RITEK / f"Prime_{s}_QA_with_topic_entity.json")}
    res["human_generated_eval"] = {"n": len(h), "in_ritek_by_text": int(h["query"].isin(rq).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ritek/docs/ritek_prime_compare.json")
    args = ap.parse_args()

    skb = PrimeSKB(STARK)
    print("1. đồ thị ...", flush=True)
    kg = compare_kg(skb)
    print(json.dumps(kg, ensure_ascii=False, indent=1))
    print("2-3. query ...", flush=True)
    qs, ritek_ids = compare_queries(skb, Hops(skb))
    for s, v in qs.items():
        print(f"\n[{s}]", json.dumps({k: x for k, x in v.items() if k != "examples"},
                                     ensure_ascii=False, indent=1))
    print("\n4. độ phủ split STaRK ...", flush=True)
    cov = coverage(skb, ritek_ids)
    print(json.dumps(cov, ensure_ascii=False, indent=1))
    Path(args.out).write_text(json.dumps({"kg": kg, "queries": qs, "stark_coverage": cov},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
