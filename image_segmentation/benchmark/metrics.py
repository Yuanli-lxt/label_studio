from __future__ import annotations

import math
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


def boundary_metrics(model_mask: Any, gt_mask: Any, tolerance_px: int = 2) -> dict:
    model = as_binary_mask(model_mask)
    gt = as_binary_mask(gt_mask)
    if model.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {model.shape} != {gt.shape}")
    tol = max(0, int(tolerance_px))
    model_boundary = mask_boundary(model)
    gt_boundary = mask_boundary(gt)
    if not model_boundary.any() and not gt_boundary.any():
        return {
            "boundary_iou": None,
            "boundary_f1": None,
            "boundary_precision": None,
            "boundary_recall": None,
            "boundary_tolerance_px": tol,
            "boundary_error_area_ratio": float(np.logical_xor(model, gt).sum() / model.size) if model.size else None,
            "model_boundary_complexity": boundary_complexity(model),
            "human_boundary_complexity": boundary_complexity(gt),
            "warning": "empty_boundaries",
        }
    model_band = dilate_binary(model_boundary, tol)
    gt_band = dilate_binary(gt_boundary, tol)
    matched_model = np.logical_and(model_boundary, gt_band)
    matched_gt = np.logical_and(gt_boundary, model_band)
    precision = _safe_ratio(int(matched_model.sum()), int(model_boundary.sum()))
    recall = _safe_ratio(int(matched_gt.sum()), int(gt_boundary.sum()))
    f1 = None if precision is None or recall is None or precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    union = np.logical_or(model_band, gt_band)
    intersection = np.logical_and(model_band, gt_band)
    return {
        "boundary_iou": float(intersection.sum() / union.sum()) if union.any() else None,
        "boundary_f1": f1,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_tolerance_px": tol,
        "boundary_error_area_ratio": float(np.logical_xor(model, gt).sum() / model.size) if model.size else None,
        "model_boundary_complexity": boundary_complexity(model),
        "human_boundary_complexity": boundary_complexity(gt),
    }


def mask_boundary(mask: Any) -> np.ndarray:
    arr = as_binary_mask(mask)
    if not arr.any():
        return np.zeros(arr.shape, dtype=bool)
    return np.logical_and(arr, np.logical_not(erode_binary(arr, 1)))


def boundary_complexity(mask: Any) -> float:
    arr = as_binary_mask(mask)
    area = int(arr.sum())
    if area <= 0:
        return 0.0
    return float(mask_boundary(arr).sum() / area)


def mask_shape_metadata(mask: Any, bbox: Sequence[float] | None = None) -> dict:
    arr = as_binary_mask(mask)
    height, width = arr.shape
    area = int(arr.sum())
    bbox = list(bbox) if bbox is not None else mask_bbox(arr)
    bbox_area = 0.0
    bbox_area_ratio = 0.0
    fill_ratio = 0.0
    if bbox:
        bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        bbox_area_ratio = float(bbox_area / (width * height)) if width * height else 0.0
        fill_ratio = float(area / bbox_area) if bbox_area else 0.0
    perimeter = float(mask_boundary(arr).sum())
    return {
        "perimeter_px": perimeter,
        "perimeter_area_ratio": float(perimeter / area) if area else 0.0,
        "bbox_area_ratio": bbox_area_ratio,
        "mask_bbox_fill_ratio": fill_ratio,
        "thin_structure_score": thin_structure_score(arr),
        "connected_components": connected_components(arr),
    }


def prediction_time_boundary_shape_features(mask: Any, width: int | None = None, height: int | None = None) -> dict:
    arr = as_binary_mask(mask)
    mask_height, mask_width = arr.shape
    width = int(width or mask_width)
    height = int(height or mask_height)
    image_area = max(0, width * height)
    area = int(arr.sum())
    bbox = mask_bbox(arr)
    bbox_area = 0.0
    aspect = 0.0
    if bbox:
        box_width = max(0.0, float(bbox[2]) - float(bbox[0]))
        box_height = max(0.0, float(bbox[3]) - float(bbox[1]))
        bbox_area = box_width * box_height
        aspect = float(box_width / box_height) if box_height else 0.0
    boundary = mask_boundary(arr)
    perimeter = float(boundary.sum())
    components = component_areas(arr)
    largest = max(components) if components else 0
    return {
        "pred_area_ratio": float(area / image_area) if image_area else 0.0,
        "pred_bbox_area_ratio": float(bbox_area / image_area) if image_area else 0.0,
        "pred_extent": float(area / bbox_area) if bbox_area else 0.0,
        "pred_aspect_ratio": aspect,
        "pred_touches_border": touches_border(arr, bbox, width, height),
        "pred_perimeter_px": perimeter,
        "pred_boundary_complexity": float((perimeter * perimeter) / (4.0 * math.pi * area)) if area else 0.0,
        "pred_boundary_density": float(perimeter / math.sqrt(area)) if area else 0.0,
        "pred_component_count": len(components),
        "pred_largest_component_ratio": float(largest / area) if area else 0.0,
        "pred_hole_count": hole_count(arr),
        "pred_thinness_proxy": thin_structure_score(arr),
    }


def thin_structure_score(mask: Any) -> float:
    arr = as_binary_mask(mask)
    area = int(arr.sum())
    if area <= 0:
        return 0.0
    eroded = erode_binary(arr, 1)
    lost = area - int(eroded.sum())
    return float(lost / area)


def erode_binary(mask: Any, radius: int = 1) -> np.ndarray:
    arr = as_binary_mask(mask)
    radius = max(0, int(radius))
    if radius == 0:
        return arr.copy()
    padded = np.pad(arr, radius, mode="constant", constant_values=False)
    out = np.ones(arr.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return out


def dilate_binary(mask: Any, radius: int = 1) -> np.ndarray:
    arr = as_binary_mask(mask)
    radius = max(0, int(radius))
    if radius == 0:
        return arr.copy()
    padded = np.pad(arr, radius, mode="constant", constant_values=False)
    out = np.zeros(arr.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return out


def connected_components(mask: Any) -> int:
    return len(component_areas(mask))


def component_areas(mask: Any) -> list[int]:
    arr = as_binary_mask(mask)
    try:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=np.uint8)
        labels, count = ndimage.label(arr, structure=structure)
        if count <= 0:
            return []
        return [int((labels == idx).sum()) for idx in range(1, int(count) + 1)]
    except Exception:
        pass
    seen = np.zeros(arr.shape, dtype=bool)
    areas: list[int] = []
    height, width = arr.shape
    for y, x in zip(*np.where(arr)):
        if seen[y, x]:
            continue
        area = 0
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            area += 1
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if ny == cy and nx == cx:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        areas.append(area)
    return areas


def hole_count(mask: Any) -> int:
    arr = as_binary_mask(mask)
    if not arr.any():
        return 0
    background = np.logical_not(arr)
    try:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=np.uint8)
        labels, count = ndimage.label(background, structure=structure)
        if count <= 0:
            return 0
        border_labels = set()
        if labels.size:
            border_labels.update(int(v) for v in labels[0, :] if v)
            border_labels.update(int(v) for v in labels[-1, :] if v)
            border_labels.update(int(v) for v in labels[:, 0] if v)
            border_labels.update(int(v) for v in labels[:, -1] if v)
        return sum(1 for idx in range(1, int(count) + 1) if idx not in border_labels)
    except Exception:
        pass
    seen = np.zeros(background.shape, dtype=bool)
    height, width = background.shape
    holes = 0
    for y, x in zip(*np.where(background)):
        if seen[y, x]:
            continue
        touches_image_border = False
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            if cy == 0 or cx == 0 or cy == height - 1 or cx == width - 1:
                touches_image_border = True
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if ny == cy and nx == cx:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if not touches_image_border:
            holes += 1
    return holes


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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None
