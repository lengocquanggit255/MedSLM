"""So hai run HOP V8 ở stage Flan (vd ablation lm_head): dòng đã chọn, McNemar exact trên Hits@1 val từng câu.

python HOP/hop_polo_v8/tools/compare_flan_runs.py --a <run tied>/hop_v8_pharmkg --b <run untied>/hop_v8_pharmkg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def selected(run: Path) -> dict:
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    flan = audit["flan"]
    history = json.loads((run / "flan" / "qa_sft" / "history.json").read_text(encoding="utf-8"))
    row = next(r for r in history if r["epoch"] == flan["qa_sft_selected_epoch"])
    # val_cache = "<phase>/val_cache/epoch_XX.jsonl" -> dự đoán cùng epoch nằm ở "<phase>/val_predictions/"
    phase, _, name = row["val_cache"].split("/")
    predictions = [json.loads(line) for line in (run / "flan" / phase / "val_predictions" / name).open(encoding="utf-8")]
    synthetic = json.loads((run / "flan" / "synthetic" / "history.json").read_text(encoding="utf-8"))
    flan_config = json.loads((run / "flan" / "selected" / "config.json").read_text(encoding="utf-8"))
    return {"audit": audit, "row": row, "hits": {p["qid"]: p["hits_at_1"] for p in predictions},
            "epochs": {"synthetic": len(synthetic) - 1, "qa_sft": len(history) - 1},
            "token_init": json.loads((run / "flan" / "token_initialization.json").read_text(encoding="utf-8")),
            # phiên bản thực chạy: save_pretrained ghi vào config; run mới còn có audit["environment"]
            "versions": {"transformers_in_checkpoint": flan_config.get("transformers_version"),
                         "tie_word_embeddings_in_checkpoint": flan_config.get("tie_word_embeddings"),
                         **{k: v for k, v in audit.get("environment", {}).items()
                            if k in ("torch", "transformers", "tokenizers", "sentence-transformers", "cuda_device")}}}


def summary(name: str, s: dict) -> dict:
    v, d, p = s["row"]["validation"], s["row"]["candidate_diagnostics"], s["row"]["program_audit"]
    return {"run": name, "untie_lm_head": s["token_init"].get("untie_lm_head"),
            "input_output_tied": s["token_init"]["input_output_tied"], "versions": s["versions"],
            "selected": f"synthetic e{s['audit']['flan']['synthetic_selected_epoch']} / qa_sft e{s['row']['epoch']}"
                        + (f" ({s['row']['reused_from']})" if s["row"].get("reused_from") else ""),
            "epochs_run": s["epochs"], "val_hits_at_1": round(v["mean_hits_at_1"], 4),
            "val_recall_at_20": round(v["mean_recall_at_20"], 4),
            "beam_oracle_recall_at_20": round(d["mean_beam_oracle_recall_at_20"], 4),
            "top_zero_rate": round(p["top_zero_rate"], 4), "beam_has_nonzero_rate": round(p["beam_has_nonzero_rate"], 4),
            "flan_stage_seconds": round(s["audit"]["stage_seconds"]["flan"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()
    a, b = selected(args.a), selected(args.b)
    if set(a["hits"]) != set(b["hits"]):
        raise AssertionError("the two runs were validated on different questions")
    better_a = sum(a["hits"][q] > b["hits"][q] for q in a["hits"])
    better_b = sum(b["hits"][q] > a["hits"][q] for q in a["hits"])
    p = binomtest(min(better_a, better_b), better_a + better_b, 0.5).pvalue if better_a + better_b else 1.0
    for name, s in (("a", a), ("b", b)):
        print(json.dumps(summary(name, s), ensure_ascii=False))
    print(json.dumps({"val_questions": len(a["hits"]), "only_a_hit": better_a, "only_b_hit": better_b,
                      "mcnemar_exact_p": round(p, 4),
                      "delta_hits_at_1_b_minus_a": round(b["row"]["validation"]["mean_hits_at_1"]
                                                         - a["row"]["validation"]["mean_hits_at_1"], 4)}))


if __name__ == "__main__":
    main()
