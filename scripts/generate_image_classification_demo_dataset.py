#!/usr/bin/env python3
"""Generate a deterministic synthetic image-classification demo dataset."""

from __future__ import annotations

import argparse
import json
import random
import struct
import zlib
from pathlib import Path
from typing import Iterable

SEED = 20260516
WIDTH = 320
HEIGHT = 240
EVAL_SET_ID = "image-cls-eval-v1"
DEFAULT_PRODUCT_TRAIN = 90
DEFAULT_OTHER_TRAIN = 90
DEFAULT_PRODUCT_EVAL = 15
DEFAULT_OTHER_EVAL = 15

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "demo_data" / "local-files" / "images"
EXPORT_PATH = ROOT / "demo_data" / "tasks" / "image_classification_labeled_export.json"
EVAL_MANIFEST_PATH = ROOT / "demo_data" / "tasks" / "image_classification_eval_manifest.json"
PROBES_PATH = ROOT / "demo_data" / "tasks" / "image_classification_regression_probes.json"

LEGACY_PRODUCT_TRAIN = [
    "demo_blue.png",
    "demo_blue_dark.png",
    "demo_blue_light.png",
    "demo_product_red.png",
    "demo_product_orange.png",
    "demo_product_green.png",
    "demo_product_teal.png",
    "demo_product_purple.png",
    "demo_product_neutral.png",
    "demo_product_offset_left.png",
    "demo_product_tall.png",
    "demo_product_wide.png",
]
LEGACY_OTHER_TRAIN = [
    "demo_green.png",
    "demo_green_dark.png",
    "demo_gray.png",
    "demo_other_blue.png",
    "demo_other_blue_dark.png",
    "demo_other_blue_light.png",
    "demo_other_red.png",
    "demo_other_orange.png",
    "demo_other_teal.png",
    "demo_other_purple.png",
    "demo_other_yellow.png",
    "demo_other_neutral.png",
]
LEGACY_PRODUCT_EVAL = [
    "demo_product_eval_cool.png",
    "demo_product_eval_warm.png",
    "demo_product_eval_dark.png",
    "demo_product_offset_right.png",
]
LEGACY_OTHER_EVAL = [
    "demo_other_eval_blue.png",
    "demo_other_eval_green.png",
    "demo_other_eval_dark.png",
    "demo_other_lowlight.png",
]


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack("!I", len(data))
        + tag
        + data
        + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)
        start = y * row_bytes
        raw.extend(pixels[start : start + row_bytes])
    png = signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _clamp(value: float) -> int:
    return max(0, min(255, int(value)))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return (_clamp(a[0] * (1 - ratio) + b[0] * ratio), _clamp(a[1] * (1 - ratio) + b[1] * ratio), _clamp(a[2] * (1 - ratio) + b[2] * ratio))


def _rect(canvas: bytearray, x: int, y: int, w: int, h: int, color: tuple[int, int, int], alpha: float = 1.0) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
    for py in range(y0, y1):
        for px in range(x0, x1):
            i = (py * WIDTH + px) * 3
            if alpha >= 1:
                canvas[i : i + 3] = bytes(color)
            else:
                canvas[i : i + 3] = bytes(_mix((canvas[i], canvas[i + 1], canvas[i + 2]), color, alpha))


def _canvas(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> bytearray:
    data = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        row_color = _mix(top, bottom, y / (HEIGHT - 1))
        for x in range(WIDTH):
            i = (y * WIDTH + x) * 3
            data[i : i + 3] = bytes(row_color)
    return data


def _stripes(canvas: bytearray, color: tuple[int, int, int], stride: int, alpha: float) -> None:
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if ((x + y) // stride) % 2 == 0:
                i = (y * WIDTH + x) * 3
                canvas[i : i + 3] = bytes(_mix((canvas[i], canvas[i + 1], canvas[i + 2]), color, alpha))


def render_product(rng: random.Random, variant: int) -> bytes:
    hue = variant % 6
    palettes = [
        ((34, 82, 160), (72, 138, 212), (245, 210, 64)),
        ((142, 68, 70), (194, 114, 98), (74, 128, 218)),
        ((58, 132, 92), (104, 174, 124), (250, 214, 76)),
        ((88, 78, 146), (132, 108, 178), (248, 204, 72)),
        ((50, 126, 136), (92, 168, 174), (238, 196, 86)),
        ((118, 112, 116), (156, 148, 142), (245, 194, 82)),
    ]
    top, bottom, obj = palettes[hue]
    canvas = _canvas(top, bottom)
    _stripes(canvas, _mix(top, (255, 255, 255), 0.35), 22 + (variant % 9), 0.12)
    w = 112 + (variant * 17) % 74
    h = 78 + (variant * 13) % 58
    x = 48 + (variant * 29) % max(1, WIDTH - w - 70)
    y = 44 + (variant * 19) % max(1, HEIGHT - h - 74)
    if variant % 11 == 0:
        x = 28
    if variant % 13 == 0:
        x = WIDTH - w - 32
    _rect(canvas, x + 5, y + 6, w, h, (20, 20, 22), 0.32)
    _rect(canvas, x, y, w, h, obj, 1.0)
    border = _mix(obj, (0, 0, 0), 0.45)
    _rect(canvas, x, y, w, 3, border)
    _rect(canvas, x, y + h - 3, w, 3, border)
    _rect(canvas, x, y, 3, h, border)
    _rect(canvas, x + w - 3, y, 3, h, border)
    _rect(canvas, x + 6, y + 6, max(8, w - 12), 5, (255, 255, 255), 0.35)
    if variant % 5 == 0:
        _rect(canvas, x + w // 4, y + h // 3, w // 2, h // 4, _mix(obj, (255, 255, 255), 0.25), 0.5)
    return bytes(canvas)


def render_other(rng: random.Random, variant: int) -> bytes:
    hue = variant % 6
    palettes = [
        ((62, 142, 92), (96, 170, 122), (126, 156, 136)),
        ((126, 126, 132), (154, 154, 160), (176, 176, 182)),
        ((54, 96, 150), (92, 130, 174), (124, 144, 178)),
        ((132, 86, 118), (166, 110, 138), (180, 142, 158)),
        ((142, 118, 70), (184, 156, 104), (202, 188, 134)),
        ((50, 60, 74), (80, 92, 106), (104, 116, 130)),
    ]
    top, bottom, patch = palettes[hue]
    canvas = _canvas(top, bottom)
    _stripes(canvas, _mix(patch, (0, 0, 0), 0.35), 15 + (variant % 10), 0.20)
    for idx in range(5):
        x = (variant * 37 + idx * 61) % 270
        y = (variant * 23 + idx * 43) % 190
        w = 28 + ((variant + idx) * 11) % 58
        h = 22 + ((variant + idx) * 7) % 48
        _rect(canvas, x, y, w, h, patch, 0.20 + ((variant + idx) % 5) * 0.025)
    # Low contrast center texture: deliberately not a crisp product object.
    _rect(canvas, 110, 82, 102, 74, _mix(top, bottom, 0.5), 0.26)
    return bytes(canvas)


def _image_ref(filename: str) -> str:
    return f"/data/local-files/?d=images/{filename}"


def _task(task_id: int, annotation_id: int, filename: str, label: str, split: str) -> dict:
    return {
        "id": task_id,
        "meta": {"dataset_split": split, "eval_set_id": EVAL_SET_ID},
        "data": {
            "image": _image_ref(filename),
            "caption": f"Synthetic {label.lower()} {split} sample: {filename}",
        },
        "annotations": [
            {
                "id": annotation_id,
                "was_cancelled": False,
                "result": [
                    {
                        "from_name": "image_label",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": [label]},
                    }
                ],
            }
        ],
    }


def _extend_names(legacy: list[str], prefix: str, total: int) -> list[str]:
    names = list(legacy)
    idx = 0
    while len(names) < total:
        candidate = f"{prefix}_{idx:03d}.png"
        if candidate not in names:
            names.append(candidate)
        idx += 1
    return names


def generate_dataset(
    product_train: int = DEFAULT_PRODUCT_TRAIN,
    other_train: int = DEFAULT_OTHER_TRAIN,
    product_eval: int = DEFAULT_PRODUCT_EVAL,
    other_eval: int = DEFAULT_OTHER_EVAL,
) -> dict:
    if product_train < len(LEGACY_PRODUCT_TRAIN) or other_train < len(LEGACY_OTHER_TRAIN):
        raise ValueError("train counts must preserve existing legacy train images")
    if product_eval < len(LEGACY_PRODUCT_EVAL) or other_eval < len(LEGACY_OTHER_EVAL):
        raise ValueError("eval counts must preserve existing legacy eval images")

    product_train_names = _extend_names(LEGACY_PRODUCT_TRAIN, "demo_product_train", product_train)
    other_train_names = _extend_names(LEGACY_OTHER_TRAIN, "demo_other_train", other_train)
    product_eval_names = _extend_names(LEGACY_PRODUCT_EVAL, "demo_product_eval_extra", product_eval)
    other_eval_names = _extend_names(LEGACY_OTHER_EVAL, "demo_other_eval_extra", other_eval)

    rng = random.Random(SEED)
    for idx, filename in enumerate(product_train_names + product_eval_names):
        write_png(IMAGE_DIR / filename, WIDTH, HEIGHT, render_product(rng, idx))
    for idx, filename in enumerate(other_train_names + other_eval_names):
        write_png(IMAGE_DIR / filename, WIDTH, HEIGHT, render_other(rng, idx))

    tasks = []
    task_id = 4101
    annotation_id = 5101
    for filename in product_train_names:
        tasks.append(_task(task_id, annotation_id, filename, "Product", "train"))
        task_id += 1
        annotation_id += 1
    for filename in other_train_names:
        tasks.append(_task(task_id, annotation_id, filename, "Other", "train"))
        task_id += 1
        annotation_id += 1
    for filename in product_eval_names:
        tasks.append(_task(task_id, annotation_id, filename, "Product", "eval"))
        task_id += 1
        annotation_id += 1
    for filename in other_eval_names:
        tasks.append(_task(task_id, annotation_id, filename, "Other", "eval"))
        task_id += 1
        annotation_id += 1

    eval_manifest = {
        "eval_set_id": EVAL_SET_ID,
        "description": "Fixed independent eval set for image classification. These images must not be used for training fit.",
        "images": [
            *[{"filename": name, "label": "Product"} for name in product_eval_names],
            *[{"filename": name, "label": "Other"} for name in other_eval_names],
        ],
    }
    probes = [
        {
            "id": "img_probe_product_blue",
            "category": "obvious_product",
            "image": _image_ref("demo_blue.png"),
            "expected_label": "Product",
        },
        {
            "id": "img_probe_product_blue_dark",
            "category": "obvious_product",
            "image": _image_ref("demo_blue_dark.png"),
            "expected_label": "Product",
        },
        {
            "id": "img_probe_other_green",
            "category": "obvious_other",
            "image": _image_ref("demo_green.png"),
            "expected_label": "Other",
        },
        {
            "id": "img_probe_other_green_dark",
            "category": "obvious_other",
            "image": _image_ref("demo_green_dark.png"),
            "expected_label": "Other",
        },
        {
            "id": "img_probe_other_gray",
            "category": "borderline_other",
            "image": _image_ref("demo_gray.png"),
            "expected_label": "Other",
        },
    ]

    EXPORT_PATH.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    EVAL_MANIFEST_PATH.write_text(json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROBES_PATH.write_text(json.dumps(probes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "product_train": len(product_train_names),
        "other_train": len(other_train_names),
        "product_eval": len(product_eval_names),
        "other_eval": len(other_eval_names),
        "total_images": len(product_train_names) + len(other_train_names) + len(product_eval_names) + len(other_eval_names),
        "export_path": str(EXPORT_PATH.relative_to(ROOT)),
        "eval_manifest_path": str(EVAL_MANIFEST_PATH.relative_to(ROOT)),
        "probes_path": str(PROBES_PATH.relative_to(ROOT)),
        "image_dir": str(IMAGE_DIR.relative_to(ROOT)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-train", type=int, default=DEFAULT_PRODUCT_TRAIN)
    parser.add_argument("--other-train", type=int, default=DEFAULT_OTHER_TRAIN)
    parser.add_argument("--product-eval", type=int, default=DEFAULT_PRODUCT_EVAL)
    parser.add_argument("--other-eval", type=int, default=DEFAULT_OTHER_EVAL)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_dataset(args.product_train, args.other_train, args.product_eval, args.other_eval)
    print("Generated deterministic image classification demo dataset")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
