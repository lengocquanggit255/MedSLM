#!/usr/bin/env python3
"""Launch the complete V7 final pipeline, locally or in one Kaggle session."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as packed:
        for member in packed.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with packed.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def find_mount() -> Path:
    matches = []
    for root in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
        if root.exists():
            matches.extend(p.parent for p in root.rglob("input_manifest.json")
                           if p.parent.name.startswith("pharmkg-hop-polo-v7-final"))
    matches = list(dict.fromkeys(p.resolve() for p in matches))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one V7 final dataset mount: {matches}")
    return matches[0]


def runtime_preflight(device: str) -> None:
    import numpy as np
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Keep Kaggle's working Torch/CUDA and NumPy. Only the text stack is pinned.
    exact = {"transformers": "5.16.1", "huggingface-hub": "1.29.0",
             "tokenizers": "0.23.1", "safetensors": "0.8.0"}
    install = []
    for package, expected in exact.items():
        try:
            current = metadata.version(package)
        except metadata.PackageNotFoundError:
            current = None
        if current != expected:
            install.append(f"{package}=={expected}")
    for package, pinned in (("sentence-transformers", "5.1.2"),
                            ("sentencepiece", "0.2.1")):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            install.append(f"{package}=={pinned}")
    if install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-deps", *install])
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    assert AutoModelForSeq2SeqLM and AutoTokenizer and SentenceTransformer and TruncatedSVD
    print(json.dumps({"runtime": {"torch": torch.__version__,
                                  "numpy": np.__version__,
                                  "cuda": torch.cuda.is_available(),
                                  "text_stack_installed": install}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/hop_polo_v7_final"))
    parser.add_argument("--device", default="cuda")
    args, extra = parser.parse_known_args()
    if extra and not (len(extra) == 2 and extra[0] == "-f"):
        parser.error(f"unexpected arguments: {extra}")
    mounted = args.input_root or find_mount()
    source = args.source_root or Path(__file__).resolve().parents[1]
    if not (source / "src/hop_polo_v7_final.py").is_file():
        source = mounted / "source"
    if not (source / "src/hop_polo_v7_final.py").is_file():
        source = args.output.parent / "_v7_final_source"
        if source.exists():
            raise RuntimeError("source extraction path already exists")
        extract(mounted / "source.zip", source)
    manifest_path = source / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    for name, expected in json.loads(manifest_path.read_text()).items():
        if sha(source / name) != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    if sha(Path(__file__).resolve()) != json.loads(manifest_path.read_text())["kaggle/run.py"]:
        raise AssertionError("kernel launcher differs from packaged source")
    inputs = mounted
    allowed = json.loads((mounted / "input_manifest.json").read_text())["files"]
    actual = {str(p.relative_to(mounted)) for folder in
              ("raw", "base_model", "embedding_model")
              for p in (mounted / folder).rglob("*") if p.is_file()}
    other_files = {p.name for p in mounted.iterdir() if p.is_file()} - {
        "input_manifest.json", "dataset-metadata.json"}
    if actual != set(allowed) or other_files:
        inputs = args.output.parent / "_v7_final_inputs"
        if inputs.exists():
            raise RuntimeError("input extraction path already exists")
        inputs.mkdir(parents=True)
        for folder in ("raw", "base_model", "embedding_model"):
            if (mounted / folder).is_dir():
                for name in allowed:
                    relative = Path(name)
                    if relative.parts[0] != folder:
                        continue
                    target = inputs / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mounted / relative, target)
            else:
                extract(mounted / f"{folder}.zip", inputs / folder)
        shutil.copy2(mounted / "input_manifest.json",
                     inputs / "input_manifest.json")
    runtime_preflight(args.device)
    sys.path.insert(0, str(source / "src"))
    sys.argv = [str(source / "src/hop_polo_v7_final.py"),
                "--input-root", str(inputs), "--output", str(args.output),
                "--config", str(source / "configs/full.json"),
                "--reference-split", str(source / "configs/reference_split.json"),
                "--source-manifest", str(manifest_path),
                "--device", args.device]
    runpy.run_path(str(source / "src/hop_polo_v7_final.py"), run_name="__main__")


if __name__ == "__main__":
    main()
