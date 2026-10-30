"""Kernel Kaggle cho HOP V8 (đẩy bằng scripts/kaggle_run.py; tham số qua biến môi trường).

DATASET     prime | pharmkg | adint
CONFIG      tên file configs/<CONFIG>.json trong source: default (PharmKG) | prime | adint | smoke | smoke_prime |
            smoke_adint | tied
DEADLINE_H  số giờ tối đa trước khi pipeline dừng an toàn ở trạng thái PAUSED (mặc định 11.3)
RESUME      "auto": nếu output HOP V8 của phiên trước được gắn (job `kernels`), chép sang rồi chạy tiếp
STOP_AFTER  tên stage để dừng sau nó (vd "flan" cho ablation lm_head); rỗng = chạy hết
CODE_SHA    sha256 mong đợi của source_manifest.json (khoá đúng bản code đã upload); rỗng = không kiểm
Datasets: ritek-hop-v8-code (source.zip + source_manifest.json, tools/upload_kaggle.py code)
          ritek-hop-v8-<DATASET> (bundle, tools/upload_kaggle.py data)
Model: tải từ Hugging Face đúng revision ghim trong config, kiểm sha256 của model.safetensors.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TEMP = Path("/tmp/hop_v8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight(root: Path) -> None:
    """Giữ Torch/CUDA/NumPy của Kaggle; chỉ ghim bộ thư viện text theo requirements.txt của source (như V7)."""
    exact = dict(line.split("==", 1) for line in (root / "requirements.txt").read_text().splitlines()
                 if "==" in line and not line.lstrip().startswith("#"))
    install, preinstalled = [], {}
    for package, expected in exact.items():
        try:
            current = metadata.version(package)
        except metadata.PackageNotFoundError:
            current = None
        preinstalled[package] = current
        if current != expected:
            install.append(f"{package}=={expected}")
    # In ra để log Kaggle cho biết image có sẵn bản nào và ta đã ép cài lại những gì.
    print(json.dumps({"kaggle_preinstalled": preinstalled, "forced_install": install}), flush=True)
    if install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *install])


def extract_source() -> tuple[Path, Path]:
    """Kaggle tự giải nén zip trong dataset: nhận cây đã giải nén (có source_manifest.json) hoặc source.zip."""
    root = TEMP / "src"
    shutil.rmtree(root, ignore_errors=True)
    trees = [p.parents[2] for p in INPUT.rglob("pipeline.py")
             if p.parent.name == "hop" and p.parents[1].name == "src" and (p.parents[2] / "source_manifest.json").exists()]
    archives = sorted(INPUT.rglob("source.zip"))
    if len(trees) == 1:
        shutil.copytree(trees[0], root)
    elif not trees and len(archives) == 1:
        with zipfile.ZipFile(archives[0]) as packed:
            for member in packed.infolist():
                if Path(member.filename).is_absolute() or ".." in Path(member.filename).parts:
                    raise ValueError(f"unsafe archive member: {member.filename}")
            packed.extractall(root)
    else:
        raise FileNotFoundError(f"expected one HOP V8 source tree or source.zip: trees={trees} zips={archives}")
    manifest = root / "source_manifest.json"
    for name, expected in json.loads(manifest.read_text()).items():
        if sha(root / name) != expected:
            raise AssertionError(f"source hash mismatch: {name}")
    expected = os.environ.get("CODE_SHA", "")
    if expected and sha(manifest) != expected:
        raise AssertionError("source_manifest.json differs from CODE_SHA")
    return root, manifest


def find_bundle(dataset: str) -> Path:
    found = [p.parent for p in INPUT.rglob("manifest.json")
             if json.loads(p.read_text()).get("dataset") == dataset and (p.parent / "triples.npy").exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {dataset} bundle under /kaggle/input: {found}")
    return found[0]


def fetch_model(spec: dict, dest: Path) -> Path:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(spec["repo"], revision=spec["revision"], local_dir=dest,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "1_Pooling/*"]))
    if sha(path / "model.safetensors") != spec["model_sha256"]:
        raise AssertionError(f"{spec['repo']} model hash differs from config")
    return path


def main() -> None:
    dataset = os.environ["DATASET"]
    config_name = os.environ.get("CONFIG", "default")
    root, manifest = extract_source()
    preflight(root)
    sys.path.insert(0, str(root / "src"))
    from hop.pipeline import load_config
    config_path = root / "configs" / f"{config_name}.json"
    config = load_config(config_path)
    bundle = find_bundle(dataset)
    flan = fetch_model(config["models"]["flan"], TEMP / "models" / "flan")
    embedding = fetch_model(config["models"]["embedding"], TEMP / "models" / "embedding")
    output = WORK / f"hop_v8_{dataset}"
    command = [sys.executable, str(root / "src" / "run.py"), "--input", str(bundle), "--config", str(config_path),
               "--output", str(output), "--flan-dir", str(flan), "--embedding-dir", str(embedding),
               "--source-manifest", str(manifest), "--device", "cuda",
               "--deadline-hours", os.environ.get("DEADLINE_H", "11.3")]
    if os.environ.get("STOP_AFTER"):
        command += ["--stop-after", os.environ["STOP_AFTER"]]
    if os.environ.get("RESUME", "") == "auto":
        previous = [p.parent for p in INPUT.rglob("run.json") if p.parent.name == output.name]
        if len(previous) > 1:
            raise FileNotFoundError(f"several previous outputs mounted: {previous}")
        if previous and not output.exists():
            shutil.copytree(previous[0], output)
            command.append("--resume")
            print(f"resume from {previous[0]}", flush=True)
    subprocess.check_call(command)
    audit = json.loads((output / "audit.json").read_text())
    print(json.dumps({"status": audit.get("status"), "paused_at": audit.get("paused_at"),
                      "stage_seconds": audit.get("stage_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
