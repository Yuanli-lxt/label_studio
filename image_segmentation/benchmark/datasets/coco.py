from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from image_segmentation.benchmark.metrics import mask_bbox, touches_border
from image_segmentation.benchmark.schema import coco_xywh_to_xyxy


def load_coco_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("COCO annotation file must contain a JSON object")
    return data


def iter_coco_manifest_samples(
    images_dir: str | Path,
    annotations_file: str | Path,
    benchmark_id: str,
    max_samples: int | None = None,
) -> Iterable[dict]:
    data = load_coco_json(annotations_file)
    images_by_id = {row["id"]: row for row in data.get("images", []) if isinstance(row, dict) and "id" in row}
    categories = {
        row["id"]: row.get("name", str(row["id"]))
        for row in data.get("categories", [])
        if isinstance(row, dict) and "id" in row
    }
    annotations = [row for row in data.get("annotations", []) if _valid_annotation(row)]
    instances_per_image = Counter(row.get("image_id") for row in annotations)
    count = 0
    for ann in annotations:
        image = images_by_id.get(ann.get("image_id"))
        if not image:
            continue
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        image_path = Path(images_dir) / str(image.get("file_name") or "")
        bbox_xyxy = coco_xywh_to_xyxy(ann["bbox"], width, height)
        mask = segmentation_to_mask(ann.get("segmentation"), width, height, ann.get("iscrowd", 0))
        area = int(mask.sum()) if mask is not None else int(float(ann.get("area") or 0))
        area_ratio = float(area / (width * height)) if width * height else 0.0
        gt_mask_bbox = mask_bbox(mask) if mask is not None else bbox_xyxy
        tags = difficulty_tags(
            area_ratio=area_ratio,
            bbox_xyxy=bbox_xyxy,
            gt_mask=mask,
            width=width,
            height=height,
            instances_in_image=instances_per_image.get(ann.get("image_id"), 0),
        )
        yield {
            "benchmark_id": benchmark_id,
            "dataset": "COCO",
            "sample_id": f"COCO_{image.get('id')}_{ann.get('id')}",
            "image_id": str(image.get("id")),
            "annotation_id": str(ann.get("id")),
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "category_id": str(ann.get("category_id")),
            "category_name": str(categories.get(ann.get("category_id"), ann.get("category_id"))),
            "gt_bbox_xyxy": bbox_xyxy,
            "gt_area": area,
            "gt_area_ratio": area_ratio,
            "gt_touches_border": touches_border(mask, bbox_xyxy, width, height) if mask is not None else False,
            "difficulty_tags": tags,
            "split": benchmark_id,
            "gt_segmentation": ann.get("segmentation"),
            "gt_iscrowd": int(ann.get("iscrowd") or 0),
            "metadata": {"coco_file_name": image.get("file_name")},
        }
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def segmentation_to_mask(segmentation: Any, width: int, height: int, iscrowd: int = 0) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    try:
        from pycocotools import mask as mask_utils

        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, height, width)
        else:
            raise ValueError("unsupported COCO segmentation")
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return (decoded > 0).astype(np.uint8)
    except Exception:
        if isinstance(segmentation, list):
            return _polygon_mask(segmentation, width, height)
        if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
            return _uncompressed_rle_mask(segmentation["counts"], width, height)
        raise


def difficulty_tags(
    area_ratio: float,
    bbox_xyxy: list[float],
    gt_mask: np.ndarray | None,
    width: int,
    height: int,
    instances_in_image: int,
) -> list[str]:
    tags: list[str] = []
    if area_ratio < 0.01:
        tags.append("small_object")
    elif area_ratio < 0.10:
        tags.append("medium_object")
    else:
        tags.append("large_object")
    if touches_border(gt_mask, bbox_xyxy, width, height) if gt_mask is not None else False:
        tags.append("touches_border")
    box_width = max(0.0, bbox_xyxy[2] - bbox_xyxy[0])
    box_height = max(0.0, bbox_xyxy[3] - bbox_xyxy[1])
    aspect = box_width / box_height if box_height else 0.0
    if aspect > 4.0 or (aspect > 0 and aspect < 0.25):
        tags.append("elongated_object")
    if instances_in_image >= 5:
        tags.append("crowded_or_multi_instance")
    return tags


def _valid_annotation(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and row.get("segmentation") is not None
        and isinstance(row.get("bbox"), list)
        and len(row.get("bbox")) == 4
        and float(row.get("area") or 0) > 0
    )


def _polygon_mask(polygons: list, width: int, height: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)


def _uncompressed_rle_mask(counts: list[int], width: int, height: int) -> np.ndarray:
    values: list[int] = []
    current = 0
    for count in counts:
        values.extend([current] * int(count))
        current = 1 - current
    total = width * height
    if len(values) < total:
        values.extend([0] * (total - len(values)))
    arr = np.asarray(values[:total], dtype=np.uint8)
    return arr.reshape((height, width), order="F")

