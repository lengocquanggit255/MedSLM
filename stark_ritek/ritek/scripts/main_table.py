"""Bảng kết quả chính RiTeK (test đủ 2,801 query) từ ritek/results/*/*/test/metrics.json.

Chạy: python ritek/scripts/main_table.py      # in markdown, dán vào ritek/PLAN.md §6

Danh sách dưới đây là quy ước "bản nào là bản chính" (ritek/PLAN.md Q9): retriever dense/lexical dùng
corpus `add_rel` (n_rel=-1) + gộp chunk `top3_mean` như stark_qa MultiVSS; `no_rel`, `_max` vào phụ lục.
Biến thể chưa có metrics.json thì in dòng trống kèm "chưa có".
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from starkbench.eval import METRICS_REPORT  # noqa: E402

MAIN = [("BM25", "bm25/add_rel"),
        ("MedCPT", "cosine_medcpt/add_rel"),
        ("BGE-M3", "cosine_bgem3/add_rel"),
        ("Contriever-FT", "contriever_ft/dotprod"),
        ("MoR", "mor/full_rerank"),
        ("GNN-RAG (adapted: toàn KG)", "gnnrag/rearev_fullkg"),
        ("SubgraphRAG", "subgraphrag/orig_none"),
        ("ARK", "ark/main")]
ABLATION = [("MoR traversal, không rerank", "mor/traversal_no_rerank"),
            ("MoR w/o RT", "mor/wo_RT"),
            ("MoR w/o RS", "mor/wo_RS"),
            ("MoR, reranker train lại (đối chứng)", "mor/contriever_retrained_rerank"),
            ("MoR + BGE-M3", "mor/bgem3_rerank"),
            ("GNN-RAG + BGE-M3 (dừng ở epoch 42/100)", "gnnrag/rearev_fullkg_bgem3_s2"),
            ("SubgraphRAG + pos_weight sqrt", "subgraphrag/orig_sqrt")]
APPENDIX = [("BM25 no_rel", "bm25/no_rel"),
            ("MedCPT add_rel, gộp max", "cosine_medcpt/add_rel_max"),
            ("MedCPT no_rel", "cosine_medcpt/no_rel"),
            ("MedCPT no_rel, gộp max", "cosine_medcpt/no_rel_max"),
            ("BGE-M3 add_rel, gộp max", "cosine_bgem3/add_rel_max"),
            ("BGE-M3 no_rel", "cosine_bgem3/no_rel"),
            ("BGE-M3 no_rel, gộp max", "cosine_bgem3/no_rel_max")]


def table(title, rows):
    print(f"\n**{title}**\n")
    print("| Phương pháp | Biến thể | " + " | ".join(METRICS_REPORT) + " |")
    print("|---|---|" + "---:|" * len(METRICS_REPORT))
    for name, var in rows:
        f = ROOT / "ritek/results" / var / "test/metrics.json"
        if f.exists():
            s = json.loads(f.read_text(encoding="utf-8"))["sanitized"]
            cells = [f"{s[m] * 100:.2f}" for m in METRICS_REPORT]
        else:
            cells = ["chưa có"] + ["—"] * (len(METRICS_REPORT) - 1)
        print(f"| {name} | `{var}` | " + " | ".join(cells) + " |")


def main():
    table("Bản chính", MAIN)
    table("Ablation / đối chứng", ABLATION)
    table("Phụ lục: độ nhạy corpus và cách gộp chunk", APPENDIX)


if __name__ == "__main__":
    main()
