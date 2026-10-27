"""Entrypoint HOP V8. Ví dụ (từ gốc repo):

python HOP/hop_polo_v8/src/run.py --input HOP/hop_polo_v8/inputs/pharmkg --config HOP/hop_polo_v8/configs/smoke.json \
    --output <thư mục rỗng> --flan-dir <flan-t5-small> --embedding-dir <all-mpnet-base-v2> --device cpu
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hop.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
