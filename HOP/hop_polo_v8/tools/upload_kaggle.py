"""Đóng gói và upload dataset Kaggle cho HOP V8 (pool tài khoản của scripts/kaggle_run.py, không hardcode key).

python HOP/hop_polo_v8/tools/upload_kaggle.py code [--account <user>]                    # ritek-hop-v8-code
python HOP/hop_polo_v8/tools/upload_kaggle.py data --dataset prime [--account <user>]    # ritek-hop-v8-prime
Không có --account thì chỉ dựng thư mục staging (.cache/hop_v8_kaggle/). In sha256 của source_manifest.json:
giá trị này là CODE_SHA trong job JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOP = Path(__file__).resolve().parents[1]
ROOT = HOP.parents[1]
STAGING = ROOT / ".cache" / "hop_v8_kaggle"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_code() -> tuple[Path, str]:
    out = STAGING / "code"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    files = sorted(p for p in (HOP / "src").rglob("*.py") if "__pycache__" not in p.parts)
    files += sorted((HOP / "configs").glob("*.json")) + [HOP / "kaggle" / "hop_v8.py", HOP / "README.md",
                                                         HOP / "requirements.txt"]
    manifest = {str(p.relative_to(HOP)).replace("\\", "/"): sha(p) for p in files}
    manifest_path = out / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
        for path in files:
            packed.write(path, arcname=str(path.relative_to(HOP)).replace("\\", "/"))
        packed.write(manifest_path, arcname="source_manifest.json")
    return out, sha(manifest_path)


def build_data(dataset: str) -> Path:
    bundle = HOP / "inputs" / dataset
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha(bundle / name) != expected:
            raise AssertionError(f"bundle hash mismatch: {name}")
    out = STAGING / dataset
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for name in ["manifest.json", *manifest["files"]]:
        shutil.copy2(bundle / name, out / name)
    return out


def upload(folder: Path, slug: str, user: str) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_run import ACCOUNTS, env
    e = env(user, dict(ACCOUNTS)[user])
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")

    def kag(*args, timeout=3600):
        r = subprocess.run(["kaggle", *args], env=e, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")

    exists = "ready" in kag("datasets", "status", f"{user}/{slug}", timeout=180).lower()
    out = (kag("datasets", "version", "-p", str(folder), "-m", "cập nhật", "--dir-mode", "zip") if exists
           else kag("datasets", "create", "-p", str(folder), "--dir-mode", "zip"))
    print(f"upload {slug} -> {user}: {out.strip().splitlines()[-1:] if out.strip() else ''}", flush=True)
    for _ in range(60):
        time.sleep(30)
        if "ready" in kag("datasets", "status", f"{user}/{slug}", timeout=180).lower():
            print("dataset sẵn sàng", flush=True)
            return
    sys.exit(f"{slug} trên {user} chưa sẵn sàng sau 30 phút")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("code", "data"))
    parser.add_argument("--dataset", choices=("prime", "pharmkg", "adint"))
    parser.add_argument("--account")
    args = parser.parse_args()
    if args.kind == "code":
        folder, manifest_sha = build_code()
        print(f"CODE_SHA={manifest_sha}")
        slug = "ritek-hop-v8-code"
    else:
        if not args.dataset:
            parser.error("data needs --dataset")
        folder = build_data(args.dataset)
        slug = f"ritek-hop-v8-{args.dataset}"
    print(f"staging: {folder}")
    if args.account:
        upload(folder, slug, args.account)


if __name__ == "__main__":
    main()
