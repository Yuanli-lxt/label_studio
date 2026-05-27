#!/usr/bin/env python3
"""
看起来像没 mask？
是 UI 透明度问题？
是 RLE 错了？
是 mask 在别的地方？
是中心框分错？
"""
"""Decode a Label Studio Brush RLE prediction and write mask/overlay PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from label_studio_converter.brush import decode_rle
from PIL import Image


class RLEOverlayError(ValueError):
    """Raised when the input cannot be decoded into a Brush mask."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError as exc:
        raise RLEOverlayError(f"input JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RLEOverlayError(f"input JSON is not valid JSON: {path}: {exc}") from exc


def _iter_result_items(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("result"), list):
                yield from _iter_result_items(item["result"])
            elif isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    for key in ("result", "results", "predictions"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("result"), list):
                    yield from _iter_result_items(item["result"])
                elif isinstance(item, dict):
                    yield item


def extract_brush_rle_result(payload: Any) -> Dict[str, Any]:
    for item in _iter_result_items(payload):
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        if item.get("type") == "brushlabels" and value.get("format") == "rle":
            if "rle" not in value:
                raise RLEOverlayError("found brushlabels result but value.rle is missing")
            return item
    raise RLEOverlayError("could not find a brushlabels result with format=rle")


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def decode_brush_mask(result: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    try:
        width = int(result["original_width"])
        height = int(result["original_height"])
    except KeyError as exc:
        raise RLEOverlayError(f"{exc.args[0]} is missing from brush result") from exc
    except (TypeError, ValueError) as exc:
        raise RLEOverlayError("original_width/original_height must be integers") from exc

    if width <= 0 or height <= 0:
        raise RLEOverlayError("original_width/original_height must be positive")

    value = result.get("value") if isinstance(result.get("value"), dict) else {}
    rle = value.get("rle")
    if not isinstance(rle, list) or not rle:
        raise RLEOverlayError("value.rle must be a non-empty list")

    try:
        decoded = decode_rle(rle)
    except Exception as exc:
        raise RLEOverlayError(f"RLE decode failed: {exc}") from exc

    expected = width * height * 4
    if int(decoded.size) != expected:
        raise RLEOverlayError(f"decoded RLE size {decoded.size} does not match width*height*4 {expected}")

    rgba = decoded.reshape((height, width, 4))
    mask = (rgba[:, :, 3] > 0) | (rgba[:, :, :3].max(axis=2) > 0)
    return mask.astype(np.uint8), width, height


def write_outputs(mask: np.ndarray, image_path: Path, out_prefix: Path) -> Tuple[Path, Path]:
    if not image_path.exists():
        raise RLEOverlayError(f"image file does not exist: {image_path}")

    mask_path = out_prefix.with_name(out_prefix.name + "_mask.png")
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask * 255, mode="L").save(mask_path)
    with Image.open(image_path) as image:
        base = image.convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        raise RLEOverlayError(
            f"image size {base.size} does not match mask size {(mask.shape[1], mask.shape[0])}"
        )

    color = Image.new("RGBA", base.size, (255, 215, 0, 0))
    alpha = Image.fromarray(mask * 120, mode="L")
    color.putalpha(alpha)
    Image.alpha_composite(base, color).save(overlay_path)
    return mask_path, overlay_path


def build_report(input_path: Path, image_path: Path, out_prefix: Path) -> Dict[str, Any]:
    payload = load_json(input_path)
    result = extract_brush_rle_result(payload)
    mask, width, height = decode_brush_mask(result)
    bbox = mask_bbox(mask)
    mask_path, overlay_path = write_outputs(mask, image_path, out_prefix)
    nonzero = int(mask.sum())
    return {
        "input": str(input_path),
        "image": str(image_path),
        "original_width": width,
        "original_height": height,
        "mask_shape": list(mask.shape),
        "nonzero": nonzero,
        "bbox": list(bbox) if bbox else None,
        "area_ratio": nonzero / float(width * height),
        "mask_png": str(mask_path),
        "overlay_png": str(overlay_path),
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"input: {report['input']}")
    print(f"image: {report['image']}")
    print(f"original_width: {report['original_width']}")
    print(f"original_height: {report['original_height']}")
    print(f"mask shape: {tuple(report['mask_shape'])}")
    print(f"nonzero pixel count: {report['nonzero']}")
    print(f"bbox: {tuple(report['bbox']) if report['bbox'] else None}")
    print(f"area ratio: {report['area_ratio']:.6f}")
    print(f"mask png: {report['mask_png']}")
    print(f"overlay png: {report['overlay_png']}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="prediction result or /predict response JSON")
    parser.add_argument("--image", required=True, type=Path, help="source image path")
    parser.add_argument("--out-prefix", required=True, type=Path, help="output path prefix")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        print_report(build_report(args.input, args.image, args.out_prefix))
    except RLEOverlayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
