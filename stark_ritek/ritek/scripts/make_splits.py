"""Ghi ritek/splits/{train,val,test,test-0.1}.index — val = 10% RiTeK train, seed 42.

Chạy: python ritek/scripts/make_splits.py
Tất định: chạy lại cho đúng cùng file. Xem `starkbench.ritek`.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from starkbench.ritek import load_queries, make_splits  # noqa: E402

if __name__ == "__main__":
    n = make_splits()
    print(n)
    for s in n:
        q = load_queries(s)
        print(f"  {s:9} {len(q):5} query · không topic: {int((q['topic_ids'].str.len() == 0).sum())}")
