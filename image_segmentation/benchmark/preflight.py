from __future__ import annotations

import argparse
from pathlib import Path

from image_segmentation.benchmark.config import enabled_coco_config, load_config
from image_segmentation.benchmark.datasets.coco import iter_coco_manifest_samples, load_coco_json


def run_preflight(config_path: str) -> tuple[bool, list[str]]:
    lines: list[str] = []
    path = Path(config_path)
    if not path.exists():
        return False, [f"[FAIL] config not found: {path}"]
    lines.append(f"[OK] config exists: {path}")
    try:
        config = load_config(path)
    except Exception as exc:
        return False, lines + [f"[FAIL] config is not readable: {exc}"]

    coco = enabled_coco_config(config)
    if not coco:
        return False, lines + ["[FAIL] no enabled COCO dataset in config"]
    images_dir = Path(coco.get("images_dir", ""))
    annotations_file = Path(coco.get("annotations_file", ""))
    ok = True
    if images_dir.exists():
        lines.append(f"[OK] COCO images_dir exists: {images_dir}")
    else:
        ok = False
        lines.append(f"[FAIL] COCO images_dir missing: {images_dir}")
    if annotations_file.exists():
        lines.append(f"[OK] COCO annotations_file exists: {annotations_file}")
    else:
        ok = False
        lines.append(f"[FAIL] COCO annotations_file missing: {annotations_file}")
    if not ok:
        lines.append(
            "[NEXT] Run: python -m image_segmentation.benchmark.download_data "
            "--dataset coco_val2017 --output-dir data/external"
        )
        return False, lines
    try:
        data = load_coco_json(annotations_file)
        lines.append(
            f"[OK] COCO JSON readable: {len(data.get('images', []))} images, "
            f"{len(data.get('annotations', []))} annotations"
        )
        first = next(iter(iter_coco_manifest_samples(images_dir, annotations_file, "preflight", max_samples=1)))
        lines.append(f"[OK] parsed one valid image+annotation+mask: {first['sample_id']}")
    except Exception as exc:
        return False, lines + [f"[FAIL] could not parse valid COCO sample: {exc}"]
    return True, lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local benchmark data before running.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ok, lines = run_preflight(args.config)
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

