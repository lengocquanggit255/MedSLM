#!/usr/bin/env python3
"""Stage only raw PharmKG and pinned public models, with hash verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True,
                   help="Directory containing raw/, base_model/, embedding_model/, input_manifest.json")
    p.add_argument("--destination", type=Path,
                   default=Path(__file__).resolve().parents[1] / "inputs")
    args = p.parse_args()
    manifest = json.loads((args.source / "input_manifest.json").read_text())
    files = manifest["files"]
    if not files or any(Path(name).parts[0] not in
                        {"raw", "base_model", "embedding_model"} for name in files):
        raise AssertionError("input manifest includes a non-public or trained artifact")
    actual = {str(path.relative_to(args.source)): sha(path)
              for folder in ("raw", "base_model", "embedding_model")
              for path in (args.source / folder).rglob("*") if path.is_file()}
    if actual != files:
        raise AssertionError("source raw/public-model files differ from input manifest")
    args.destination.mkdir(parents=True, exist_ok=True)
    existing = [path for path in args.destination.iterdir()]
    if existing:
        raise RuntimeError(f"destination is not empty: {args.destination}")
    for name, expected in files.items():
        target = args.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / name, target)
        if sha(target) != expected:
            raise AssertionError(f"copied input changed: {name}")
    shutil.copy2(args.source / "input_manifest.json",
                 args.destination / "input_manifest.json")
    (args.destination.parent / "input_lineage.json").write_text(json.dumps({
        "source_input_manifest_sha256": sha(args.source / "input_manifest.json"),
        "files": len(files),
        "trained_pharmkg_artifacts": False,
        "allowed_roots": ["raw", "base_model", "embedding_model"],
    }, indent=2) + "\n")
    print(f"Staged {len(files)} verified raw/public-model files into {args.destination}")


if __name__ == "__main__":
    main()
