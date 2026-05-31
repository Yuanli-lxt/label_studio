from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def binary_mask_metrics(model_mask: Any, gt_mask: Any) -> dict:
    model = as_binary_mask(model_mask)
    gt = as_binary_mask(gt_mask)
    if model.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {model.shape} != {gt.shape}")
    model_area = int(model.sum())
    gt_area = int(gt.sum())
    intersection = int(np.logical_and(model, gt).sum())
    union = int(np.logical_or(model, gt).sum())
    return {
        "iou": 1.0 if union == 0 else float(intersection / union),
        "dice": 1.0 if model_area + gt_area == 0 else float((2 * intersection) / (model_area + gt_area)),
        "precision": 1.0 if model_area == 0 else float(intersection / model_area),
        "recall": 1.0 if gt_area == 0 else float(intersection / gt_area),
    }


def as_binary_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    if arr.ndim >= 3:
        arr = arr[0]
    return (arr > 0).astype(bool)


def mask_bbox(mask: Any) -> list[float] | None:
    arr = as_binary_mask(mask)
    ys, xs = np.where(arr)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def touches_border(mask: Any, bbox: Sequence[float] | None, width: int, height: int, tolerance: int = 1) -> bool:
    arr = as_binary_mask(mask)
    tol = max(0, int(tolerance))
    if arr.size:
        if arr[: tol + 1, :].any() or arr[max(0, height - tol - 1) :, :].any():
            return True
        if arr[:, : tol + 1].any() or arr[:, max(0, width - tol - 1) :].any():
            return True
    if bbox is None:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1 <= tol or y1 <= tol or x2 >= width - 1 - tol or y2 >= height - 1 - tol


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None

