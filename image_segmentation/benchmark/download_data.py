from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from image_segmentation.benchmark.dataset_registry import DATASET_REGISTRY, get_dataset_entry


COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LVIS_VAL_ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"


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


def download_lvis_val(output_dir: str, force: bool = False, dry_run: bool = False) -> dict:
    root = Path(output_dir) / "lvis"
    annotations_dir = root / "annotations"
    target = annotations_dir / "lvis_v1_val.json"
    zip_path = root / "lvis_v1_val.json.zip"
    if dry_run:
        print(f"[DRY-RUN] would download LVIS val annotations: {LVIS_VAL_ANNOTATIONS_URL}")
        print(f"[DRY-RUN] would write annotations_file={target}")
        size = _remote_size_bytes(LVIS_VAL_ANNOTATIONS_URL)
        print(f"[DRY-RUN] remote_file_size_bytes={size if size is not None else 'unknown'}")
        print("[INFO] LVIS images are expected at data/external/coco/val2017; run COCO download separately if needed.")
        return {"annotations_file": str(target), "images_dir": str(Path(output_dir) / "coco" / "val2017")}
    root.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    if force or not target.exists():
        _download(LVIS_VAL_ANNOTATIONS_URL, zip_path, force=force)
        _extract_lvis_file(zip_path, annotations_dir, force=force)
    else:
        print(f"[SKIP] LVIS annotations already exist: {target}")
        print("[INFO] COCO val2017 images are not downloaded by lvis_val; expected images_dir=data/external/coco/val2017")
    return {"annotations_file": str(target), "images_dir": str(Path(output_dir) / "coco" / "val2017")}


def manual_dataset_message(dataset: str, output_dir: str) -> str:
    entry = get_dataset_entry(dataset)
    root = Path(output_dir) / dataset
    lines = [
        f"[MANUAL] {entry.display_name} requires manual download or an external archive.",
        f"[LICENSE] {entry.license_note}",
        "[LAYOUT] expected directory layout:",
    ]
    for local_dir in entry.expected_local_dirs:
        parts = Path(local_dir).parts
        suffix = Path(*parts[2:]) if len(parts) > 2 else Path(dataset)
        lines.append(f"  {Path(output_dir) / suffix}")
    lines.extend(
        [
            "[NEXT] after placement, run:",
            f"  python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.{dataset}300.yaml",
            f"[INFO] no files were written under {root}",
        ]
    )
    return "\n".join(lines)


def download_open_images_subset(
    output_dir: str,
    split: str = "validation",
    max_samples: int | None = None,
    use_fiftyone: bool = False,
    dry_run: bool = False,
) -> dict:
    target = Path(output_dir) / "open_images_v7" / split
    if split != "validation":
        raise ValueError("Open Images V7 segmentations currently supports only --split validation")
    if dry_run:
        print(f"[DRY-RUN] would prepare Open Images V7 segmentation subset at {target}")
        print(f"[DRY-RUN] split={split} max_samples={max_samples} use_fiftyone={use_fiftyone}")
        return {"dataset_dir": str(target)}
    if not use_fiftyone:
        raise RuntimeError("Open Images V7 scripted subset requires --use-fiftyone true")
    try:
        import fiftyone.zoo as foz  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Open Images V7 subset download requires FiftyOne. Install it with: pip install fiftyone"
        ) from exc
    raise RuntimeError(
        "FiftyOne is available, but export wiring is intentionally conservative in this first pass; "
        "use --dry-run or manually export images/, segmentations/, metadata.json, annotations.jsonl."
    )


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


def _remote_size_bytes(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


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


def _extract_lvis_file(path: Path, annotations_dir: Path, force: bool = False) -> None:
    target = annotations_dir / "lvis_v1_val.json"
    if target.exists() and not force:
        print(f"[SKIP] LVIS annotations already extracted: {target}")
        return
    print(f"[INFO] extracting lvis_v1_val.json -> {annotations_dir}")
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("lvis_v1_val.json") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        raise RuntimeError(f"extract lvis_v1_val.json failed: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicitly download public benchmark data.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-fiftyone", default="false")
    parser.add_argument("--allow-manual-placeholder", default="false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    use_fiftyone = str(args.use_fiftyone).lower() in {"1", "true", "yes"}
    allow_manual = str(args.allow_manual_placeholder).lower() in {"1", "true", "yes"}
    entry = get_dataset_entry(args.dataset)
    print(f"[INFO] dataset={entry.dataset_name} mode={entry.download_mode}")
    print(f"[LICENSE] {entry.license_note}")
    try:
        if args.dataset == "coco_val2017":
            if args.dry_run:
                print("[DRY-RUN] would download COCO val2017 images and annotations")
                return 0
            summary = download_coco_val2017(args.output_dir, force=args.force)
            print("[OK] COCO val2017 data ready")
            print(f"images_dir={summary['images_dir']}")
            print(f"annotations_file={summary['annotations_file']}")
            return 0
        if args.dataset == "lvis_val":
            summary = download_lvis_val(args.output_dir, force=args.force, dry_run=args.dry_run)
            print("[OK] LVIS val annotations ready" if not args.dry_run else "[OK] LVIS dry run complete")
            print(f"images_dir={summary['images_dir']}")
            print(f"annotations_file={summary['annotations_file']}")
            return 0
        if args.dataset in {"dis5k", "cod10k", "camo"}:
            print(manual_dataset_message(args.dataset, args.output_dir))
            return 0 if allow_manual else 1
        if args.dataset == "open_images_v7_segmentations":
            summary = download_open_images_subset(
                args.output_dir,
                split=args.split,
                max_samples=args.max_samples,
                use_fiftyone=use_fiftyone,
                dry_run=args.dry_run,
            )
            print("[OK] Open Images V7 subset dry run complete" if args.dry_run else "[OK] Open Images V7 subset ready")
            print(f"dataset_dir={summary['dataset_dir']}")
            return 0
        raise RuntimeError(f"unsupported dataset: {args.dataset}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
