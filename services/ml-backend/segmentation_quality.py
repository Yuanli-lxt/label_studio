from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


MIN_AREA_RATIO = 0.001
MAX_AREA_RATIO = 0.75
LOW_BBOX_IOU_THRESHOLD = 0.30


def evaluate_segmentation_quality(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    prompt_bbox: Optional[List[float]] = None,
    mask_bbox: Optional[List[float]] = None,
    rle_length: Optional[int] = None,
    backend_metadata: Optional[dict] = None,
) -> dict:
    width = _positive_int(image_width)
    height = _positive_int(image_height)
    binary_mask = _as_binary_mask(mask)
    valid_dimensions = width > 0 and height > 0
    expected_pixels = width * height if valid_dimensions else 0

    mask_area_px = int(binary_mask.sum()) if binary_mask is not None else 0
    valid_mask = bool(binary_mask is not None and mask_area_px > 0 and expected_pixels > 0)
    mask_area_ratio = float(mask_area_px / expected_pixels) if expected_pixels > 0 else 0.0
    mask_area_ratio = max(0.0, min(1.0, mask_area_ratio))

    normalized_prompt_bbox = _normalize_bbox(prompt_bbox, width, height)
    normalized_mask_bbox = _normalize_bbox(mask_bbox, width, height)
    bbox_iou = _bbox_iou(normalized_prompt_bbox, normalized_mask_bbox)
    touches_border = _mask_touches_border(binary_mask) if binary_mask is not None else False

    reasons: List[str] = []
    if not valid_mask:
        reasons.append("empty_or_invalid_mask")
    else:
        if mask_area_ratio < MIN_AREA_RATIO:
            reasons.append("mask_area_too_small")
        if mask_area_ratio > MAX_AREA_RATIO:
            reasons.append("mask_area_too_large")
        if touches_border:
            reasons.append("mask_touches_image_border")

    if valid_mask and mask_bbox is None:
        reasons.append("missing_mask_bbox")
    if bbox_iou is not None and bbox_iou < LOW_BBOX_IOU_THRESHOLD:
        reasons.append("low_prompt_mask_bbox_iou")
    if _has_backend_fallback_or_error(backend_metadata):
        reasons.append("backend_fallback_or_error")

    priority, score = _review_priority(reasons)
    return {
        "mask_quality": {
            "valid_mask": valid_mask,
            "mask_area_px": mask_area_px,
            "mask_area_ratio": mask_area_ratio,
            "image_width": width,
            "image_height": height,
            "prompt_bbox": normalized_prompt_bbox,
            "mask_bbox": normalized_mask_bbox,
            "bbox_iou_prompt_mask": bbox_iou,
            "mask_touches_border": bool(touches_border),
            "rle_length": _positive_int_or_none(rle_length),
        },
        "review": {
            "needs_review": bool(reasons),
            "review_priority": priority,
            "review_priority_score": score,
            "review_reason": reasons,
        },
    }


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _positive_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _as_binary_mask(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        arr = np.asarray(mask)
    except Exception:
        return None
    if not hasattr(arr, "size"):
        return _as_binary_mask_from_nested_values(arr)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    if arr.ndim >= 3:
        arr = arr[0]
    try:
        return np.isfinite(arr).astype(bool) & (arr > 0)
    except Exception:
        try:
            return arr.astype(bool)
        except Exception:
            return None


def _as_binary_mask_from_nested_values(value: Any) -> Optional[np.ndarray]:
    rows = _nested_binary_rows(value)
    if not rows:
        return None
    try:
        return np.asarray(rows).astype(bool)
    except Exception:
        return None


def _nested_binary_rows(value: Any) -> List[List[bool]]:
    if not isinstance(value, (list, tuple)):
        return [[_value_is_foreground(value)]]
    if not value:
        return []
    if all(not isinstance(item, (list, tuple)) for item in value):
        return [[_value_is_foreground(item) for item in value]]
    rows: List[List[bool]] = []
    for item in value:
        nested = _nested_binary_rows(item)
        rows.extend(nested)
    return rows


def _value_is_foreground(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return bool(value)


def _normalize_bbox(
    bbox: Optional[Sequence[Any]],
    image_width: int,
    image_height: int,
) -> Optional[List[float]]:
    if bbox is None or image_width <= 0 or image_height <= 0:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return None
    left = max(0.0, min(float(image_width - 1), left))
    right = max(0.0, min(float(image_width - 1), right))
    top = max(0.0, min(float(image_height - 1), top))
    bottom = max(0.0, min(float(image_height - 1), bottom))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _bbox_iou(
    a: Optional[Sequence[float]],
    b: Optional[Sequence[float]],
) -> Optional[float]:
    if a is None or b is None:
        return None
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection = intersection_width * intersection_height
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None
    return float(intersection / union)


def _mask_touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0 or mask.ndim != 2:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _has_backend_fallback_or_error(metadata: Optional[dict]) -> bool:
    if not isinstance(metadata, dict):
        return False
    for key in ("backend_error", "error", "exception", "traceback"):
        if metadata.get(key):
            return True
    fallback = metadata.get("fallback")
    return bool(fallback is True or (isinstance(fallback, str) and fallback.strip()))


def _review_priority(reasons: List[str]) -> Tuple[str, int]:
    if not reasons:
        return "low", 0
    high_reasons = {"empty_or_invalid_mask", "backend_fallback_or_error"}
    if any(reason in high_reasons for reason in reasons):
        return "high", 100
    return "medium", min(90, 25 * len(reasons))
