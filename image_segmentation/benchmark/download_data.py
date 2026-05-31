from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco_val2017(output_dir: str, force: bool = False) -> dict:
    root = Path(output_dir) / "coco"
    images_dir = root / "val2017"
    annotations_dir = root / "annotations"
    instances_file = annotations_dir / "instances_val2017.json"
    root.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    images_zip = root / "val2017.zip"
    annotations_zip = root / "annotations_trainval2017.zip"
    if force or not images_dir.exists():
        _download(COCO_VAL_IMAGES_URL, images_zip, force=force)
        _extract_zip(images_zip, root, force=force)
    else:
        print(f"[SKIP] images already exist: {images_dir}")
    if force or not instances_file.exists():
        _download(COCO_ANNOTATIONS_URL, annotations_zip, force=force)
        _extract_instances_file(annotations_zip, annotations_dir, force=force)
    else:
        print(f"[SKIP] annotations already exist: {instances_file}")
    return {"images_dir": str(images_dir), "annotations_file": str(instances_file)}


def _download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"[SKIP] download exists: {path}")
        return
    print(f"[INFO] downloading {url}")
    try:
        with urllib.request.urlopen(url) as response, path.open("wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\r[INFO] {path.name}: {pct:5.1f}%", end="")
            if total:
                print()
    except Exception as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc


def _extract_zip(path: Path, destination: Path, force: bool = False) -> None:
    if force and (destination / "val2017").exists():
        shutil.rmtree(destination / "val2017")
    print(f"[INFO] extracting {path} -> {destination}")
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(destination)
    except Exception as exc:
        raise RuntimeError(f"extract failed for {path}: {exc}") from exc


def _extract_instances_file(path: Path, annotations_dir: Path, force: bool = False) -> None:
    target = annotations_dir / "instances_val2017.json"
    if target.exists() and not force:
        print(f"[SKIP] annotations already extracted: {target}")
        return
    print(f"[INFO] extracting instances_val2017.json -> {annotations_dir}")
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("annotations/instances_val2017.json") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        raise RuntimeError(f"extract instances_val2017.json failed: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicitly download public benchmark data.")
    parser.add_argument("--dataset", required=True, choices=["coco_val2017"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = download_coco_val2017(args.output_dir, force=args.force)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print("[OK] COCO val2017 data ready")
    print(f"images_dir={summary['images_dir']}")
    print(f"annotations_file={summary['annotations_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

