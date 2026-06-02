from __future__ import annotations

from typing import Any, Sequence


REQUIRED_MANIFEST_FIELDS = [
    "benchmark_id",
    "dataset",
    "sample_id",
    "image_id",
    "annotation_id",
    "image_path",
    "width",
    "height",
    "category_id",
    "category_name",
    "gt_bbox_xyxy",
    "gt_area",
    "gt_area_ratio",
    "gt_touches_border",
    "difficulty_tags",
    "split",
]

OPTIONAL_DATASET_FIELDS = [
    "mask_path",
    "boundary_metadata",
]


def coco_xywh_to_xyxy(bbox: Sequence[Any], width: int, height: int) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("COCO bbox must be [x, y, width, height]")
    x, y, w, h = [float(value) for value in bbox]
    return clip_xyxy([x, y, x + w, y + h], width, height)


def clip_xyxy(bbox: Sequence[Any], width: int, height: int) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has non-positive clipped area")
    return [x1, y1, x2, y2]


def validate_manifest_sample(sample: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in sample:
            errors.append(f"missing field: {field}")
    if "gt_bbox_xyxy" in sample:
        bbox = sample["gt_bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append("gt_bbox_xyxy must be a 4-item list")
    if float(sample.get("gt_area_ratio", 0.0) or 0.0) < 0.0:
        errors.append("gt_area_ratio must be non-negative")
    if not sample.get("image_path"):
        errors.append("image_path is required")
    if not sample.get("sample_id"):
        errors.append("sample_id is required")
    return errors
