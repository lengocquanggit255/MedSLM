"""lm_head của Flan-T5 khi thêm token chương trình (transformers 5.16.1 ghim).

  PYTHONUTF8=1 PYTHONPATH=.cache/hop_v8_pylib .venv/Scripts/python.exe -m unittest HOP/hop_polo_v8/tests/test_flan_head.py

Cần flan-t5-small gốc ở HOP/hop_polo_v7_final/inputs/base_model (không nằm trong git).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "HOP/hop_polo_v7_final/inputs/base_model"
sys.path.insert(0, str(ROOT / "HOP/hop_polo_v8/src"))
sys.path.insert(0, str(ROOT / "HOP/hop_polo_v7_final/src"))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from hop.flan import initialize_program_tokens  # noqa: E402
from hop_polo_v6_flan import initialize_program_tokens as v7_initialize  # noqa: E402

TOKENS = [f"<P{i}>" for i in range(20)]


def load():
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    return model, tokenizer


def with_tokens(tokenizer) -> int:
    old = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    return old


def last_logits(model, tokenizer):
    encoded = tokenizer(["question: which drug treats asthma? topic 0: asthma"], return_tensors="pt")
    with torch.no_grad():
        return model(**encoded, decoder_input_ids=torch.tensor([[0, 1820, 3]])).logits[0, -1]


def tied(model) -> bool:
    return model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


class FlanHead(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tied_is_v7(self):
        v8, t8 = load()
        v7, t7 = load()
        audit = initialize_program_tokens(v8, t8, with_tokens(t8), seed=5, untie_lm_head=False)
        v7_initialize(v7, t7, with_tokens(t7), seed=5)
        self.assertTrue(audit["input_output_tied"])
        for (name, a), (_, b) in zip(v8.state_dict().items(), v7.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_untied_keeps_flan_head_and_reloads(self):
        original, t0 = load()
        reference = last_logits(original, t0)
        model, tokenizer = load()
        old = with_tokens(tokenizer)
        audit = initialize_program_tokens(model, tokenizer, old, seed=5, untie_lm_head=True)
        self.assertFalse(audit["input_output_tied"])
        self.assertTrue(audit["lm_head_rebuilt_after_resize"])
        self.assertFalse(tied(model))
        logits = last_logits(model, tokenizer)
        torch.testing.assert_close(logits[:old], reference[:old], rtol=0, atol=0)
        head = model.get_output_embeddings().weight.detach()
        shared = model.get_input_embeddings().weight.detach()
        anchor = tokenizer.encode("the", add_special_tokens=False)[0]
        self.assertLess(float((head[old:] - head[anchor]).abs().max()), 0.1)
        self.assertLess(float((shared[old:] - shared[anchor]).abs().max()), 0.1)
        model.save_pretrained(self.tmp)
        reloaded = AutoModelForSeq2SeqLM.from_pretrained(self.tmp).eval()
        self.assertFalse(tied(reloaded))
        torch.testing.assert_close(last_logits(reloaded, tokenizer), logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
