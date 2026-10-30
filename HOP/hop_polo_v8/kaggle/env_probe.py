"""Kernel CPU nhỏ: in phiên bản thư viện có sẵn trong image Kaggle (không cài gì, không đọc dataset).

Dùng để kiểm xem image có tự mang transformers bản nào (vd hành vi buộc lm_head của Flan-T5 ở transformers 5.x).
"""
import json
import platform
from importlib import metadata

packages = ("transformers", "tokenizers", "safetensors", "huggingface-hub", "sentence-transformers", "sentencepiece",
            "accelerate", "torch", "numpy", "scipy", "scikit-learn", "pandas")
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps({"python": platform.python_version(), "kaggle_image_preinstalled": versions}, indent=1), flush=True)
