"""Tiện ích dùng chung: hash file, ghi/đọc JSON(L), seed, tách từ."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

WORD = re.compile(r"[\w]+")


def digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_all(seed: int) -> None:
    """Seed tường minh ở đầu mỗi stage: kết quả một stage không phụ thuộc stage trước tiêu bao nhiêu RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def word_set(value: str) -> set[str]:
    return set(WORD.findall(value.casefold()))


def hash_words(value: str, bins: int = 16384) -> list[int]:
    words = WORD.findall(value.casefold())[:96]
    return [int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little") % bins
            for w in words] or [0]
