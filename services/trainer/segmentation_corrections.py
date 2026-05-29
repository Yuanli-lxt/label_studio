from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


MAJOR_IOU_THRESHOLD = 0.50
MODERATE_IOU_THRESHOLD = 0.75
MINOR_IOU_THRESHOLD = 0.90
MODERATE_CORRECTION_AREA_RATIO = 0.15
MAJOR_CORRECTION_AREA_RATIO = 0.25
MINOR_CORRECTION_AREA_RATIO = 0.03
LARGE_ADDED_AREA_RATIO = 0.10
LARGE_REMOVED_AREA_RATIO = 0.10
LARGE_CENTROID_SHIFT_RATIO = 0.10
LOW_BBOX_IOU_THRESHOLD = 0.50


class SegmentationCorrectionError(ValueError):
    """Raised when a correction-delta record cannot be decoded."""


def decode_label_studio_brush_rle_to_mask(rle: Any, width: int, height: int) -> np.ndarray:
    if not isinstance(rle, list) or not rle:
        raise SegmentationCorrectionError("value.rle must be a non-empty list")
    width = _positive_int(width)
    height = _positive_int(height)
    if width is None or height is None:
        raise SegmentationCorrectionError("image dimensions must be positive integers")

    try:
        from label_studio_converter.brush import decode_rle

        decoded = decode_rle(rle)
    except Exception as exc:
        raise SegmentationCorrectionError(f"RLE decode failed: {exc}") from exc

    expected = width * height * 4
    if int(decoded.size) != expected:
        raise SegmentationCorrectionError(
            f"decoded RLE size {decoded.size} does not match width*height*4 {expected}"
        )
    rgba = decoded.reshape((height, width, 4))
    mask = (rgba[:, :, 3] > 0) | (rgba[:, :, :3].max(axis=2) > 0)
    return mask.astype(np.uint8)


def compute_mask_delta_metrics(
    model_mask: np.ndarray,
    human_mask: np.ndarray,
    image_width: int,
    image_height: int,
    model_bbox: Optional[list[float]] = None,
    human_bbox: Optional[list[float]] = None,
) -> dict:
    width = _positive_int(image_width) or 0
    height = _positive_int(image_height) or 0
    image_area = width * height
    diagonal = math.sqrt(float(width * width + height * height)) if image_area > 0 else 0.0

    model = _as_binary_mask(model_mask)
    human = _as_binary_mask(human_mask)
    if model.shape != human.shape:
        raise ValueError(f"mask shape mismatch: {model.shape} != {human.shape}")

    model_area = int(model.sum())
    human_area = int(human.sum())
    intersection_area = int(np.logical_and(model, human).sum())
    union_area = int(np.logical_or(model, human).sum())
    added_area = int(np.logical_and(human, np.logical_not(model)).sum())
    removed_area = int(np.logical_and(model, np.logical_not(human)).sum())
    symmetric_diff_area = added_area + removed_area

    iou = 1.0 if union_area == 0 else float(intersection_area / union_area)
    dice_denominator = model_area + human_area
    dice = 1.0 if dice_denominator == 0 else float((2 * intersection_area) / dice_denominator)
    precision = 1.0 if model_area == 0 else float(intersection_area / model_area)
    recall = 1.0 if human_area == 0 else float(intersection_area / human_area)

    normalized_model_bbox = _normalize_bbox(model_bbox, width, height)
    normalized_human_bbox = _normalize_bbox(human_bbox, width, height)
    bbox_iou = _bbox_iou(normalized_model_bbox, normalized_human_bbox)

    model_centroid = _mask_centroid(model)
    human_centroid = _mask_centroid(human)
    centroid_shift_px = None
    centroid_shift_ratio = None
    if model_centroid is not None and human_centroid is not None:
        centroid_shift_px = float(
            math.dist((model_centroid[0], model_centroid[1]), (human_centroid[0], human_centroid[1]))
        )
        centroid_shift_ratio = float(centroid_shift_px / diagonal) if diagonal > 0 else None

    added_ratio = _ratio(added_area, image_area)
    removed_ratio = _ratio(removed_area, image_area)
    correction_ratio = _ratio(symmetric_diff_area, image_area)
    human_to_model_area_ratio = None if model_area == 0 else float(human_area / model_area)

    severity, major, reasons = _correction_severity(
        model_area=model_area,
        human_area=human_area,
        iou=iou,
        correction_area_ratio=correction_ratio,
        added_area_ratio=added_ratio,
        removed_area_ratio=removed_ratio,
        centroid_shift_ratio=centroid_shift_ratio,
        bbox_iou=bbox_iou,
    )

    return {
        "model_human_iou": iou,
        "model_human_dice": dice,
        "model_human_precision": precision,
        "model_human_recall": recall,
        "model_area_px": model_area,
        "human_area_px": human_area,
        "intersection_area_px": intersection_area,
        "union_area_px": union_area,
        "added_area_px": added_area,
        "removed_area_px": removed_area,
        "symmetric_diff_area_px": symmetric_diff_area,
        "added_area_ratio": added_ratio,
        "removed_area_ratio": removed_ratio,
        "correction_area_ratio": correction_ratio,
        "human_to_model_area_ratio": human_to_model_area_ratio,
        "model_bbox": normalized_model_bbox,
        "human_bbox": normalized_human_bbox,
        "model_human_bbox_iou": bbox_iou,
        "centroid_shift_px": centroid_shift_px,
        "centroid_shift_ratio": centroid_shift_ratio,
        "major_correction": major,
        "correction_severity": severity,
        "correction_reason": reasons,
    }


def build_segmentation_correction_record(
    task: dict,
    prediction_result: dict,
    annotation_result: dict,
    image_width: int,
    image_height: int,
    extra_metadata: Optional[dict] = None,
) -> dict:
    metadata = extra_metadata if isinstance(extra_metadata, dict) else {}
    base = _base_record(task, prediction_result, annotation_result, image_width, image_height, metadata)

    if _positive_int(image_width) is None or _positive_int(image_height) is None:
        return _skipped(base, "missing_image_dimensions")
    model_rle = _result_rle(prediction_result)
    if model_rle is None:
        return _skipped(base, "missing_model_rle")
    human_rle = _result_rle(annotation_result)
    if human_rle is None:
        return _skipped(base, "missing_human_rle")

    try:
        model_mask = decode_label_studio_brush_rle_to_mask(model_rle, image_width, image_height)
    except SegmentationCorrectionError:
        return _skipped(base, "decode_model_rle_failed")
    try:
        human_mask = decode_label_studio_brush_rle_to_mask(human_rle, image_width, image_height)
    except SegmentationCorrectionError:
        return _skipped(base, "decode_human_rle_failed")

    base["human_mask_bbox"] = _mask_bbox(human_mask)
    base["delta"] = compute_mask_delta_metrics(
        model_mask,
        human_mask,
        image_width,
        image_height,
        model_bbox=base.get("model_mask_bbox"),
        human_bbox=base.get("human_mask_bbox"),
    )
    base["record_status"] = "ok"
    base["skip_reason"] = None
    return base


def build_segmentation_correction_record_from_task(
    task: dict,
    annotation: Optional[dict] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    annotation = annotation if isinstance(annotation, dict) else _first_non_cancelled_annotation(task)
    if annotation is None:
        return _skipped(_base_record(task, None, None, None, None, extra_metadata), "missing_human_annotation")

    annotation_result = _select_brush_result(annotation.get("result"), preferred=None)
    if annotation_result is None:
        return _skipped(
            _base_record(task, None, None, None, None, _annotation_metadata(annotation, extra_metadata)),
            "missing_human_rle",
        )

    prediction_container, prediction_result = _select_prediction_pair(task, annotation_result)
    if prediction_container is None or prediction_result is None:
        dims = _dimensions_from_result(annotation_result)
        return _skipped(
            _base_record(
                task,
                None,
                annotation_result,
                dims[0],
                dims[1],
                _annotation_metadata(annotation, extra_metadata),
            ),
            "missing_model_prediction",
        )

    width, height = _dimensions_from_results(prediction_result, annotation_result)
    metadata = _prediction_metadata(prediction_container, annotation, extra_metadata)
    return build_segmentation_correction_record(
        task,
        prediction_result,
        annotation_result,
        width,
        height,
        metadata,
    )


def summarize_segmentation_correction_records(records: list[dict], output_path: str) -> dict:
    rows = [row for row in records if isinstance(row, dict)]
    ok_rows = [row for row in rows if row.get("record_status") == "ok" and isinstance(row.get("delta"), dict)]
    skipped_rows = [row for row in rows if row.get("record_status") == "skipped"]
    severity_counts = {"none": 0, "minor": 0, "moderate": 0, "major": 0}
    ious = []
    correction_ratios = []
    major_count = 0
    for row in ok_rows:
        delta = row["delta"]
        severity = delta.get("correction_severity")
        if severity in severity_counts:
            severity_counts[severity] += 1
        if delta.get("major_correction"):
            major_count += 1
        if isinstance(delta.get("model_human_iou"), (int, float)):
            ious.append(float(delta["model_human_iou"]))
        if isinstance(delta.get("correction_area_ratio"), (int, float)):
            correction_ratios.append(float(delta["correction_area_ratio"]))

    return {
        "enabled": True,
        "records_total": len(rows),
        "records_ok": len(ok_rows),
        "records_skipped": len(skipped_rows),
        "major_corrections": major_count,
        "severity_counts": severity_counts,
        "mean_model_human_iou": _mean_or_none(ious),
        "mean_correction_area_ratio": _mean_or_none(correction_ratios),
        "output_path": output_path,
    }


def _base_record(
    task: Optional[dict],
    prediction_result: Optional[dict],
    annotation_result: Optional[dict],
    image_width: Any,
    image_height: Any,
    metadata: Optional[dict],
) -> dict:
    task = task if isinstance(task, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    prediction_result = prediction_result if isinstance(prediction_result, dict) else {}
    annotation_result = annotation_result if isinstance(annotation_result, dict) else {}
    prediction_meta = prediction_result.get("meta") if isinstance(prediction_result.get("meta"), dict) else {}
    value = annotation_result.get("value") if isinstance(annotation_result.get("value"), dict) else {}
    labels = value.get("brushlabels") if isinstance(value.get("brushlabels"), list) else []
    label = labels[0] if labels else metadata.get("label") or "Object"

    return {
        "task_id": task.get("id") or metadata.get("task_id"),
        "image": _task_image(task),
        "model_version": metadata.get("model_version"),
        "prediction_id": metadata.get("prediction_id"),
        "annotation_id": metadata.get("annotation_id"),
        "label": label,
        "from_name": annotation_result.get("from_name") or prediction_result.get("from_name"),
        "to_name": annotation_result.get("to_name") or prediction_result.get("to_name"),
        "image_width": _positive_int(image_width),
        "image_height": _positive_int(image_height),
        "prompt_bbox": prediction_meta.get("prompt_box") or metadata.get("prompt_bbox"),
        "model_mask_bbox": prediction_meta.get("mask_bbox") or metadata.get("model_mask_bbox"),
        "human_mask_bbox": metadata.get("human_mask_bbox"),
        "mask_quality": prediction_meta.get("mask_quality"),
        "review": prediction_meta.get("review"),
        "uncertainty": prediction_meta.get("uncertainty"),
        "delta": None,
        "record_status": "skipped",
        "skip_reason": None,
    }


def _skipped(record: Optional[dict], reason: str) -> dict:
    record = record if isinstance(record, dict) else {}
    record["record_status"] = "skipped"
    record["skip_reason"] = reason
    if "delta" not in record:
        record["delta"] = None
    return record


def _select_prediction_pair(task: dict, annotation_result: dict) -> tuple[Optional[dict], Optional[dict]]:
    predictions = task.get("predictions") if isinstance(task, dict) else None
    if not isinstance(predictions, list):
        return None, None

    candidates = []
    for pred_idx, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            continue
        result_items = prediction.get("result")
        if not isinstance(result_items, list):
            continue
        for result_idx, item in enumerate(result_items):
            if not isinstance(item, dict) or item.get("type") != "brushlabels":
                continue
            value = item.get("value") if isinstance(item.get("value"), dict) else {}
            if str(value.get("format") or "").strip().lower() != "rle":
                continue
            score = _pair_score(item, annotation_result)
            candidates.append((score, pred_idx, result_idx, prediction, item))
    if not candidates:
        return None, None
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    _, _, _, prediction, item = candidates[0]
    return prediction, item


def _select_brush_result(result_items: Any, preferred: Optional[dict]) -> Optional[dict]:
    if not isinstance(result_items, list):
        return None
    candidates = []
    for idx, item in enumerate(result_items):
        if not isinstance(item, dict) or item.get("type") != "brushlabels":
            continue
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        if str(value.get("format") or "").strip().lower() != "rle":
            continue
        score = _pair_score(item, preferred) if preferred else 0
        candidates.append((score, idx, item))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return candidates[0][2]


def _pair_score(candidate: dict, reference: Optional[dict]) -> int:
    if not isinstance(reference, dict):
        return 0
    score = 0
    if candidate.get("from_name") == reference.get("from_name"):
        score += 3
    if candidate.get("to_name") == reference.get("to_name"):
        score += 3
    if _first_label(candidate) and _first_label(candidate) == _first_label(reference):
        score += 2
    return score


def _prediction_metadata(prediction: dict, annotation: dict, extra_metadata: Optional[dict]) -> dict:
    metadata = dict(extra_metadata or {})
    metadata["prediction_id"] = prediction.get("id")
    metadata["model_version"] = prediction.get("model_version")
    metadata["annotation_id"] = annotation.get("id")
    return metadata


def _annotation_metadata(annotation: dict, extra_metadata: Optional[dict]) -> dict:
    metadata = dict(extra_metadata or {})
    metadata["annotation_id"] = annotation.get("id")
    return metadata


def _first_non_cancelled_annotation(task: dict) -> Optional[dict]:
    annotations = task.get("annotations") if isinstance(task, dict) else None
    if not isinstance(annotations, list):
        return None
    for annotation in annotations:
        if isinstance(annotation, dict) and not annotation.get("was_cancelled"):
            return annotation
    return None


def _dimensions_from_result(result: dict) -> tuple[Optional[int], Optional[int]]:
    return _positive_int(result.get("original_width")), _positive_int(result.get("original_height"))


def _dimensions_from_results(a: dict, b: dict) -> tuple[Optional[int], Optional[int]]:
    width = _positive_int(a.get("original_width")) or _positive_int(b.get("original_width"))
    height = _positive_int(a.get("original_height")) or _positive_int(b.get("original_height"))
    return width, height


def _result_rle(result: dict) -> Optional[list]:
    value = result.get("value") if isinstance(result, dict) and isinstance(result.get("value"), dict) else {}
    rle = value.get("rle")
    return rle if isinstance(rle, list) and rle else None


def _first_label(result: dict) -> Optional[str]:
    value = result.get("value") if isinstance(result, dict) and isinstance(result.get("value"), dict) else {}
    labels = value.get("brushlabels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return None


def _task_image(task: dict) -> Any:
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    return data.get("image") or data.get("image_path") or meta.get("image") or meta.get("image_path")


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_binary_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    try:
        return np.isfinite(arr).astype(bool) & (arr > 0)
    except Exception:
        return arr.astype(bool)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mask_bbox(mask: np.ndarray) -> Optional[list[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _normalize_bbox(bbox: Optional[Sequence[Any]], width: int, height: int) -> Optional[list[float]]:
    if bbox is None or width <= 0 or height <= 0:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return None
    left = max(0.0, min(float(width - 1), left))
    right = max(0.0, min(float(width - 1), right))
    top = max(0.0, min(float(height - 1), top))
    bottom = max(0.0, min(float(height - 1), bottom))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _bbox_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
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


def _correction_severity(
    model_area: int,
    human_area: int,
    iou: float,
    correction_area_ratio: float,
    added_area_ratio: float,
    removed_area_ratio: float,
    centroid_shift_ratio: Optional[float],
    bbox_iou: Optional[float],
) -> tuple[str, bool, list[str]]:
    reasons = []
    if added_area_ratio > LARGE_ADDED_AREA_RATIO:
        reasons.append("large_added_area")
    if removed_area_ratio > LARGE_REMOVED_AREA_RATIO:
        reasons.append("large_removed_area")
    if centroid_shift_ratio is not None and centroid_shift_ratio > LARGE_CENTROID_SHIFT_RATIO:
        reasons.append("large_centroid_shift")
    if bbox_iou is not None and bbox_iou < LOW_BBOX_IOU_THRESHOLD:
        reasons.append("low_bbox_iou")

    if model_area == 0 and human_area == 0:
        return "none", False, reasons
    if model_area == 0 or human_area == 0:
        return "major", True, _append_reason(reasons, "one_mask_empty")
    if iou < MAJOR_IOU_THRESHOLD:
        return "major", True, _append_reason(reasons, "low_model_human_iou")
    if iou < MODERATE_IOU_THRESHOLD or correction_area_ratio > MODERATE_CORRECTION_AREA_RATIO:
        return "moderate", correction_area_ratio > MAJOR_CORRECTION_AREA_RATIO, reasons
    if iou < MINOR_IOU_THRESHOLD or correction_area_ratio > MINOR_CORRECTION_AREA_RATIO:
        return "minor", False, reasons
    return "none", False, reasons


def _append_reason(reasons: list[str], reason: str) -> list[str]:
    if reason not in reasons:
        reasons.append(reason)
    return reasons


def _mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))
