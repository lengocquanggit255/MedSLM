#!/usr/bin/env python3
"""Create hash-checked source and input archives; does not submit Kaggle jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive(files, root, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in files:
            z.write(file, arcname=str(file.relative_to(root)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True,
                        help="Kaggle username for dataset/kernel metadata")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kaggle/package")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.output / "dataset"
    kernel_dir = args.output / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for folder in ("src", "configs")
                     for path in (ROOT / folder).rglob("*")
                     if path.is_file() and (path.suffix in {".py", ".json"}))
    sources.extend([ROOT / "README.md", ROOT / "kaggle/run.py"])
    source_manifest = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    manifest_path = ROOT / "source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    archive(sources + [manifest_path], ROOT, dataset_dir / "source.zip")
    shutil.copy2(ROOT / "kaggle/run.py", kernel_dir / "run.py")
    if args.source_only:
        print(json.dumps({"source_manifest_sha256": sha(manifest_path),
                          "source_files": len(sources),
                          "source_zip": str(dataset_dir / "source.zip")}, indent=2))
        return
    inputs = ROOT / "inputs"
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    allowed = manifest["files"]
    if any(Path(name).parts[0] not in {"raw", "base_model", "embedding_model"}
           for name in allowed):
        raise AssertionError("trained artifact in input manifest")
    for folder in ("raw", "base_model", "embedding_model"):
        files = sorted((inputs / folder).rglob("*"))
        files = [p for p in files if p.is_file()]
        for path in files:
            if sha(path) != allowed[str(path.relative_to(inputs))]:
                raise AssertionError(f"input hash mismatch: {path}")
        archive(files, inputs / folder, dataset_dir / f"{folder}.zip")
    shutil.copy2(inputs / "input_manifest.json",
                 dataset_dir / "input_manifest.json")
    dataset = f"{args.owner}/pharmkg-hop-polo-v7-final-input"
    kernel = f"{args.owner}/pharmkg-hop-polo-v7-final-one-session"
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "PharmKG HOP PoLo V7 Final Raw Inputs and Source",
        "id": dataset, "licenses": [{"name": "unknown"}]}, indent=2) + "\n")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": "PharmKG HOP PoLo V7 Final One Session",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [dataset],
        "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    print(json.dumps({"dataset": dataset, "kernel": kernel,
                      "source_manifest_sha256": sha(manifest_path),
                      "input_manifest_sha256": sha(inputs / "input_manifest.json")},
                     indent=2))


if __name__ == "__main__":
    main()
