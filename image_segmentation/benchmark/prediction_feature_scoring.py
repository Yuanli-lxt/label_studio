from __future__ import annotations

import math
from typing import Any


PREDICTION_FEATURE_NAMES = [
    "pred_area_ratio",
    "pred_bbox_area_ratio",
    "pred_extent",
    "pred_aspect_ratio",
    "pred_touches_border",
    "pred_boundary_complexity",
    "pred_boundary_density",
    "pred_component_count",
    "pred_largest_component_ratio",
    "pred_hole_count",
    "pred_thinness_proxy",
]

LEAKY_FIELD_TOKENS = {
    "boundary_metadata",
    "delta",
    "evaluation_only",
    "model_human_iou",
    "model_human_dice",
    "boundary_iou",
    "boundary_f1",
    "correction_area_ratio",
    "correction_severity",
    "major_correction",
    "severity",
}


def prediction_features(record: dict) -> dict:
    record = record if isinstance(record, dict) else {}
    features = record.get("prediction_features") if isinstance(record.get("prediction_features"), dict) else None
    if features is None:
        mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
        features = mask_quality.get("prediction_time_boundary_shape")
    if features is None:
        source = record.get("source_metadata") if isinstance(record.get("source_metadata"), dict) else {}
        features = source.get("prediction_features")
    if features is None:
        source = record.get("source_metadata") if isinstance(record.get("source_metadata"), dict) else {}
        mask_quality = source.get("mask_quality") if isinstance(source.get("mask_quality"), dict) else {}
        features = mask_quality.get("prediction_time_boundary_shape")
    source = features if isinstance(features, dict) else {}
    return {name: source.get(name) for name in PREDICTION_FEATURE_NAMES if name in source}


def boundary_shape_score_from_features(features: dict) -> float:
    if not isinstance(features, dict) or not features:
        return 0.0
    score = 0.0
    area = _num(features.get("pred_area_ratio"))
    extent = _num(features.get("pred_extent"))
    aspect = _num(features.get("pred_aspect_ratio"))
    complexity = _num(features.get("pred_boundary_complexity"))
    density = _num(features.get("pred_boundary_density"))
    components = _num(features.get("pred_component_count"))
    largest_ratio = _num(features.get("pred_largest_component_ratio"))
    holes = _num(features.get("pred_hole_count"))
    thinness = _num(features.get("pred_thinness_proxy"))
    if area > 0 and area < 0.01:
        score += 0.12
    if features.get("pred_touches_border") is True:
        score += 0.12
    if extent > 0:
        score += 0.16 * (1.0 - _clip(extent))
    if aspect > 0 and (aspect > 4.0 or aspect < 0.25):
        score += 0.12
    score += 0.18 * _clip((complexity - 1.0) / 12.0)
    score += 0.14 * _clip(density / 18.0)
    if components > 1:
        score += 0.12 * _clip((components - 1.0) / 5.0)
    if largest_ratio > 0:
        score += 0.08 * (1.0 - _clip(largest_ratio))
    if holes > 0:
        score += 0.06 * _clip(holes / 5.0)
    score += 0.18 * _clip(thinness)
    return _clip(score)


def calibrated_boundary_shape_score(record: dict) -> float:
    features = prediction_features(record)
    if not features:
        return 0.0
    parts = [
        _small_or_large(_num(features.get("pred_area_ratio")), small=0.01, large=0.45),
        _small_or_large(_num(features.get("pred_bbox_area_ratio")), small=0.015, large=0.55),
        1.0 - _clip(_num(features.get("pred_extent"))) if _num(features.get("pred_extent")) > 0 else 0.0,
        _aspect_risk(_num(features.get("pred_aspect_ratio"))),
        1.0 if features.get("pred_touches_border") is True else 0.0,
        _clip(math.log1p(max(0.0, _num(features.get("pred_boundary_complexity")) - 1.0)) / math.log1p(12.0)),
        _clip(math.log1p(max(0.0, _num(features.get("pred_boundary_density")))) / math.log1p(18.0)),
        _clip(math.log1p(max(0.0, _num(features.get("pred_component_count")) - 1.0)) / math.log1p(5.0)),
        1.0 - _clip(_num(features.get("pred_largest_component_ratio")))
        if _num(features.get("pred_largest_component_ratio")) > 0
        else 0.0,
        _clip(math.log1p(max(0.0, _num(features.get("pred_hole_count")))) / math.log1p(5.0)),
        _clip(_num(features.get("pred_thinness_proxy"))),
    ]
    weights = [0.08, 0.05, 0.12, 0.10, 0.10, 0.14, 0.12, 0.11, 0.08, 0.04, 0.16]
    return _clip(sum(value * weight for value, weight in zip(parts, weights)) / sum(weights))


def rank_boundary_shape_scores(records: list[dict]) -> list[float]:
    calibrated = [calibrated_boundary_shape_score(row) for row in records]
    if not calibrated:
        return []
    if len(set(calibrated)) <= 1:
        return [0.5] * len(calibrated)
    order = sorted(range(len(calibrated)), key=lambda idx: calibrated[idx])
    ranks = [0.5] * len(calibrated)
    denom = max(1, len(calibrated) - 1)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and calibrated[order[end]] == calibrated[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = float(average_rank / denom)
        start = end
    return ranks


def safe_prediction_feature_vector(record: dict) -> dict:
    features = prediction_features(record)
    out = {}
    for name in PREDICTION_FEATURE_NAMES:
        value = features.get(name)
        if name == "pred_touches_border":
            out[name] = 1.0 if value is True else 0.0
        else:
            out[name] = _num(value)
    out["boundary_shape_score"] = boundary_shape_score_from_features(features)
    out["boundary_shape_calibrated_score"] = calibrated_boundary_shape_score(record)
    return out


def target_label(record: dict) -> bool | None:
    if not isinstance(record, dict):
        return None
    candidates = [
        ((record.get("evaluation_only") or {}).get("delta") or {}) if isinstance(record.get("evaluation_only"), dict) else {},
        record.get("delta") if isinstance(record.get("delta"), dict) else {},
    ]
    for delta in candidates:
        if delta.get("major_correction") is True:
            return True
        if delta.get("major_correction") is False:
            return False
        severity = str(delta.get("correction_severity") or "").strip().lower()
        if severity:
            return severity == "major"
    if record.get("major_correction") is True:
        return True
    if record.get("major_correction") is False:
        return False
    return None


def assert_no_leaky_feature_names(feature_names: list[str]) -> None:
    joined = "\n".join(feature_names).lower()
    for token in LEAKY_FIELD_TOKENS:
        if token.lower() in joined:
            raise ValueError(f"GT-derived or label field is not allowed in learned fusion features: {token}")


def _small_or_large(value: float, small: float, large: float) -> float:
    if value <= 0:
        return 0.0
    if value < small:
        return _clip((small - value) / small)
    if value > large:
        return _clip((value - large) / max(1e-9, 1.0 - large))
    return 0.0


def _aspect_risk(value: float) -> float:
    if value <= 0:
        return 0.0
    if value > 1:
        return _clip((math.log(value) - math.log(2.0)) / max(1e-9, math.log(10.0) - math.log(2.0)))
    return _clip((math.log(1.0 / value) - math.log(2.0)) / max(1e-9, math.log(10.0) - math.log(2.0)))


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
