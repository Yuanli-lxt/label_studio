from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


STABLE_MEAN_IOU_THRESHOLD = 0.80
STABLE_MIN_IOU_THRESHOLD = 0.60
STABLE_DISAGREEMENT_THRESHOLD = 0.10
HIGH_MEAN_IOU_THRESHOLD = 0.90
HIGH_MIN_IOU_THRESHOLD = 0.75
HIGH_DISAGREEMENT_THRESHOLD = 0.05


def generate_bbox_prompt_variants(
    bbox: list[float],
    image_width: int,
    image_height: int,
    jitter_ratio: float = 0.05,
) -> list[dict]:
    normalized = _normalize_bbox(bbox, image_width, image_height)
    if normalized is None:
        return []

    x_min, y_min, x_max, y_max = normalized
    width = x_max - x_min
    height = y_max - y_min
    dx = width * max(0.0, float(jitter_ratio))
    dy = height * max(0.0, float(jitter_ratio))
    raw_variants = [
        ("original", [x_min, y_min, x_max, y_max]),
        ("expand", [x_min - dx, y_min - dy, x_max + dx, y_max + dy]),
        ("shrink", [x_min + dx, y_min + dy, x_max - dx, y_max - dy]),
        ("shift_left", [x_min - dx, y_min, x_max - dx, y_max]),
        ("shift_right", [x_min + dx, y_min, x_max + dx, y_max]),
        ("shift_up", [x_min, y_min - dy, x_max, y_max - dy]),
        ("shift_down", [x_min, y_min + dy, x_max, y_max + dy]),
    ]

    variants = []
    for name, candidate in raw_variants:
        clipped = _normalize_bbox(candidate, image_width, image_height)
        if clipped is not None:
            variants.append({"name": name, "bbox": clipped})
    return variants


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = _as_binary_mask(mask_a)
    b = _as_binary_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} != {b.shape}")
    a_count = int(a.sum())
    b_count = int(b.sum())
    if a_count == 0 and b_count == 0:
        return 1.0
    if a_count == 0 or b_count == 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(intersection / union) if union else 1.0


def evaluate_prompt_stability(
    masks: list[np.ndarray],
    prompt_variants: Optional[list[dict]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> dict:
    valid_masks = _valid_binary_masks(masks)
    output: Dict[str, Any] = {
        "method": "prompt_stability",
        "enabled": True,
        "num_prompt_variants": len(prompt_variants) if isinstance(prompt_variants, list) else len(masks or []),
        "num_valid_masks": len(valid_masks),
    }

    if len(valid_masks) < 2:
        output.update(
            {
                "mean_pairwise_iou": None,
                "min_pairwise_iou": None,
                "max_pairwise_iou": None,
                "disagreement_area_ratio": None,
                "stable": False,
                "stability_bucket": "unknown",
                "reason": ["insufficient_valid_masks"],
            }
        )
        return {"uncertainty": output}

    pairwise_ious = [compute_mask_iou(a, b) for a, b in combinations(valid_masks, 2)]
    mean_iou = float(np.mean(pairwise_ious))
    min_iou = float(np.min(pairwise_ious))
    max_iou = float(np.max(pairwise_ious))
    disagreement_ratio = _disagreement_area_ratio(valid_masks, image_width, image_height)

    reasons = []
    if mean_iou < STABLE_MEAN_IOU_THRESHOLD:
        reasons.append("low_mean_pairwise_iou")
    if min_iou < STABLE_MIN_IOU_THRESHOLD:
        reasons.append("low_min_pairwise_iou")
    if disagreement_ratio > STABLE_DISAGREEMENT_THRESHOLD:
        reasons.append("high_disagreement_area_ratio")

    stable = not reasons
    if (
        stable
        and mean_iou >= HIGH_MEAN_IOU_THRESHOLD
        and min_iou >= HIGH_MIN_IOU_THRESHOLD
        and disagreement_ratio <= HIGH_DISAGREEMENT_THRESHOLD
    ):
        bucket = "high"
    elif stable:
        bucket = "medium"
    else:
        bucket = "low"

    output.update(
        {
            "mean_pairwise_iou": mean_iou,
            "min_pairwise_iou": min_iou,
            "max_pairwise_iou": max_iou,
            "disagreement_area_ratio": disagreement_ratio,
            "stable": stable,
            "stability_bucket": bucket,
            "reason": reasons,
        }
    )
    return {"uncertainty": output}


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
        x_min, y_min, x_max, y_max = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return None
    x_min = max(0.0, min(float(image_width - 1), x_min))
    x_max = max(0.0, min(float(image_width - 1), x_max))
    y_min = max(0.0, min(float(image_height - 1), y_min))
    y_max = max(0.0, min(float(image_height - 1), y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def _as_binary_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    if arr.ndim >= 3:
        arr = arr[0]
    try:
        return (np.isfinite(arr) & (arr > 0)).astype(bool)
    except Exception:
        return arr.astype(bool)


def _valid_binary_masks(masks: list[np.ndarray]) -> List[np.ndarray]:
    valid = []
    for mask in masks or []:
        try:
            arr = _as_binary_mask(mask)
        except Exception:
            continue
        if arr.size > 0:
            valid.append(arr)
    if not valid:
        return []
    first_shape = valid[0].shape
    return [mask for mask in valid if mask.shape == first_shape]


def _disagreement_area_ratio(
    masks: List[np.ndarray],
    image_width: Optional[int],
    image_height: Optional[int],
) -> float:
    stack = np.stack(masks, axis=0).astype(bool)
    consensus = stack.any(axis=0)
    agreement = stack.all(axis=0)
    disagreement = np.logical_and(consensus, np.logical_not(agreement))
    total_pixels = _positive_image_pixels(image_width, image_height) or disagreement.size
    return float(disagreement.sum() / total_pixels) if total_pixels else 0.0


def _positive_image_pixels(image_width: Optional[int], image_height: Optional[int]) -> int:
    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        return 0
    return width * height if width > 0 and height > 0 else 0
