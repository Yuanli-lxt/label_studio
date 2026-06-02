from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Optional


DEFAULT_SCORE_WEIGHTS = {
    "correction_risk_score": 0.35,
    "uncertainty_score": 0.25,
    "rule_review_score": 0.20,
    "geometry_complexity_score": 0.10,
    "boundary_shape_score": 0.0,
    "diversity_score": 0.10,
}
DEFAULT_REVIEW_WEIGHT_PRESET = "current"

REVIEW_PRIORITY_LEVELS = {"low": 0.0, "medium": 0.5, "high": 1.0}
STABILITY_BUCKET_SCORES = {"unknown": 0.0, "high": 0.0, "medium": 0.5, "low": 1.0}


def load_review_candidate_records(path: str) -> list[dict]:
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            content = line.strip()
            if not content:
                continue
            try:
                row = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    return records


def score_review_candidate(
    record: dict,
    correction_risk: Optional[dict] = None,
    weights: Optional[dict] = None,
    weight_preset: Optional[str] = None,
    weight_presets_file: Optional[str] = None,
) -> dict:
    score_weights, preset_name = _resolve_score_weights(weights, weight_preset, weight_presets_file)
    correction_risk = correction_risk if isinstance(correction_risk, dict) else _record_correction_risk(record)
    components = {
        "correction_risk_score": _correction_risk_score(correction_risk),
        "uncertainty_score": _uncertainty_score(record),
        "rule_review_score": _rule_review_score(record),
        "geometry_complexity_score": _geometry_complexity_score(record),
        "boundary_shape_score": _boundary_shape_score(record),
        "diversity_score": 0.0,
    }
    reasons = _review_reasons(record, correction_risk, components)
    return _queue_item(record, correction_risk, components, score_weights, reasons, preset_name)


def build_segmentation_review_queue(
    records: list[dict],
    output_path: str,
    max_items: Optional[int] = None,
    weights: Optional[dict] = None,
    weight_preset: Optional[str] = None,
    weight_presets_file: Optional[str] = None,
) -> dict:
    rows = [row for row in records if isinstance(row, dict)]
    score_weights, preset_name = _resolve_score_weights(weights, weight_preset, weight_presets_file)
    if not rows:
        summary = _summary(
            status="skipped",
            skip_reason="no_review_candidates",
            records_total=0,
            records_scored=0,
            records_skipped=0,
            output_path=output_path,
            score_weights=score_weights,
            weight_preset=preset_name,
            items=[],
        )
        _write_queue(output_path, [])
        return {"review_queue": summary, "items": []}

    scored = []
    skipped = 0
    for index, record in enumerate(rows):
        if record.get("record_status") == "skipped" and not record.get("mask_quality") and not record.get("review"):
            skipped += 1
            continue
        item = score_review_candidate(record, weights=score_weights, weight_preset=preset_name)
        item["_input_index"] = index
        item["_diversity_bin"] = _diversity_bin(record, item)
        scored.append(item)

    selected = _apply_diversity_order(scored, score_weights, max_items)
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item.pop("_input_index", None)
        item.pop("_diversity_bin", None)

    _write_queue(output_path, selected)
    summary = _summary(
        status="generated" if selected else "skipped",
        skip_reason=None if selected else "no_review_candidates",
        records_total=len(rows),
        records_scored=len(selected),
        records_skipped=skipped + max(0, len(scored) - len(selected)),
        output_path=output_path,
        score_weights=score_weights,
        weight_preset=preset_name,
        items=selected,
    )
    return {"review_queue": summary, "items": selected}


def _apply_diversity_order(items: list[dict], weights: dict, max_items: Optional[int]) -> list[dict]:
    remaining = list(items)
    selected = []
    bin_counts: Counter[str] = Counter()
    limit = len(remaining) if max_items is None else max(0, min(int(max_items), len(remaining)))
    while remaining and len(selected) < limit:
        rescored = []
        for item in remaining:
            diversity = 1.0 / (1.0 + bin_counts[item["_diversity_bin"]])
            components = dict(item["score_components"])
            components["diversity_score"] = diversity
            priority = _weighted_score(components, weights)
            candidate = dict(item)
            candidate["score_components"] = components
            candidate["priority_score"] = priority
            candidate["priority_bucket"] = _priority_bucket(priority)
            reasons = list(candidate["review_reasons"])
            if diversity >= 0.5 and "diversity_boost" not in reasons:
                reasons.append("diversity_boost")
            candidate["review_reasons"] = reasons
            rescored.append(candidate)
        rescored.sort(
            key=lambda row: (
                -row["priority_score"],
                row.get("_input_index", 0),
                str(row.get("task_id")),
            )
        )
        chosen = rescored[0]
        selected.append(chosen)
        bin_counts[chosen["_diversity_bin"]] += 1
        remaining = [
            item
            for item in remaining
            if item.get("_input_index") != chosen.get("_input_index")
        ]
    return selected


def _queue_item(
    record: dict,
    correction_risk: dict,
    components: dict,
    weights: dict,
    reasons: list[str],
    weight_preset: str = DEFAULT_REVIEW_WEIGHT_PRESET,
) -> dict:
    priority = _weighted_score(components, weights)
    return {
        "rank": None,
        "task_id": record.get("task_id"),
        "dataset": record.get("dataset"),
        "category_name": record.get("category_name"),
        "image": record.get("image"),
        "prediction_id": record.get("prediction_id"),
        "model_version": record.get("model_version"),
        "label": record.get("label") or "Object",
        "priority_score": priority,
        "priority_bucket": _priority_bucket(priority),
        "review_weight_preset": weight_preset,
        "review_weight_weights": weights,
        "score_components": components,
        "review_reasons": reasons,
        "source_metadata": {
            "mask_quality": record.get("mask_quality"),
            "prediction_features": record.get("prediction_features"),
            "review": record.get("review"),
            "uncertainty": record.get("uncertainty"),
            "correction_risk": correction_risk or None,
        },
        "evaluation_only": {
            "delta": record.get("delta") if isinstance(record.get("delta"), dict) else None,
        },
    }


def _summary(
    status: str,
    skip_reason: Optional[str],
    records_total: int,
    records_scored: int,
    records_skipped: int,
    output_path: str,
    score_weights: dict,
    weight_preset: str,
    items: list[dict],
) -> dict:
    scores = [float(item.get("priority_score", 0.0)) for item in items]
    bucket_counts = {"high": 0, "medium": 0, "low": 0}
    for item in items:
        bucket = item.get("priority_bucket")
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
    reason_counts = Counter()
    for item in items:
        reason_counts.update(item.get("review_reasons") or [])
    return {
        "enabled": True,
        "status": status,
        "skip_reason": skip_reason,
        "records_total": records_total,
        "records_scored": records_scored,
        "records_skipped": records_skipped,
        "output_path": output_path,
        "score_weights": score_weights,
        "review_weight_preset": weight_preset,
        "priority_bucket_counts": bucket_counts,
        "mean_priority_score": (sum(scores) / len(scores)) if scores else None,
        "max_priority_score": max(scores) if scores else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "correction_risk_available": any(
            (item.get("source_metadata") or {}).get("correction_risk") for item in items
        ),
        "uncertainty_available_count": sum(
            1 for item in items if (item.get("source_metadata") or {}).get("uncertainty")
        ),
    }


def _rule_review_score(record: dict) -> float:
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    score = _clamp(_num(review.get("review_priority_score")) / 100.0)
    score = max(score, REVIEW_PRIORITY_LEVELS.get(str(review.get("review_priority") or "").lower(), 0.0))
    if review.get("needs_review") is True:
        score += 0.20
    if isinstance(review.get("review_reason"), list):
        score += min(0.20, 0.05 * len(review["review_reason"]))
    return _clamp(score)


def _uncertainty_score(record: dict) -> float:
    uncertainty = record.get("uncertainty") if isinstance(record.get("uncertainty"), dict) else {}
    if not uncertainty or uncertainty.get("enabled") is False:
        return 0.0
    score = 0.0
    mean_iou = uncertainty.get("mean_pairwise_iou")
    min_iou = uncertainty.get("min_pairwise_iou")
    disagreement = _num(uncertainty.get("disagreement_area_ratio"))
    if mean_iou is not None:
        score += 0.30 * (1.0 - _clamp(_num(mean_iou)))
    if min_iou is not None:
        score += 0.25 * (1.0 - _clamp(_num(min_iou)))
    score += 0.25 * _clamp(disagreement)
    if uncertainty.get("stable") is False:
        score += 0.10
    score += 0.10 * STABILITY_BUCKET_SCORES.get(
        str(uncertainty.get("stability_bucket") or "unknown").strip().lower(),
        0.0,
    )
    return _clamp(score)


def _correction_risk_score(correction_risk: Optional[dict]) -> float:
    risk = correction_risk.get("correction_risk") if isinstance(correction_risk, dict) else None
    if isinstance(risk, dict):
        return _clamp(_num(risk.get("risk_score")))
    return _clamp(_num(correction_risk.get("risk_score"))) if isinstance(correction_risk, dict) else 0.0


def _geometry_complexity_score(record: dict) -> float:
    mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
    area = _num(mask_quality.get("mask_area_ratio"))
    bbox_iou = mask_quality.get("bbox_iou_prompt_mask")
    score = 0.0
    if area > 0 and area < 0.01:
        score += 0.35
    elif area < 0.03:
        score += 0.20
    if area > 0.60:
        score += 0.35
    elif area > 0.35:
        score += 0.20
    if bbox_iou is not None:
        score += 0.30 * (1.0 - _clamp(_num(bbox_iou)))
    if mask_quality.get("mask_touches_border") is True:
        score += 0.20
    aspect = _bbox_aspect(record.get("model_mask_bbox"))
    if aspect > 0 and (aspect > 4.0 or aspect < 0.25):
        score += 0.15
    return _clamp(score)


def _boundary_shape_score(record: dict) -> float:
    features = _prediction_features(record)
    if not features:
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
        score += 0.16 * (1.0 - _clamp(extent))
    if aspect > 0 and (aspect > 4.0 or aspect < 0.25):
        score += 0.12
    score += 0.18 * _clamp((complexity - 1.0) / 12.0)
    score += 0.14 * _clamp(density / 18.0)
    if components > 1:
        score += 0.12 * _clamp((components - 1.0) / 5.0)
    if largest_ratio > 0:
        score += 0.08 * (1.0 - _clamp(largest_ratio))
    if holes > 0:
        score += 0.06 * _clamp(holes / 5.0)
    score += 0.18 * _clamp(thinness)
    return _clamp(score)


def _prediction_features(record: dict) -> dict:
    features = record.get("prediction_features") if isinstance(record.get("prediction_features"), dict) else None
    if features is None:
        mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
        features = mask_quality.get("prediction_time_boundary_shape")
    return features if isinstance(features, dict) else {}


def _review_reasons(record: dict, correction_risk: dict, components: dict) -> list[str]:
    reasons = []
    risk_score = components["correction_risk_score"]
    if risk_score >= 0.70:
        reasons.append("high_correction_risk")
    elif risk_score >= 0.40:
        reasons.append("medium_correction_risk")
    elif not correction_risk:
        reasons.append("missing_correction_risk_score")
        reasons.append("missing_correction_risk_model")

    uncertainty = record.get("uncertainty") if isinstance(record.get("uncertainty"), dict) else {}
    if not uncertainty:
        reasons.append("missing_uncertainty_metadata")
    else:
        if uncertainty.get("stable") is False or str(uncertainty.get("stability_bucket")).lower() == "low":
            reasons.append("unstable_prompt_response")
        if _num(uncertainty.get("disagreement_area_ratio")) >= 0.10:
            reasons.append("high_prompt_disagreement")

    mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    if review.get("needs_review") is True or components["rule_review_score"] >= 0.50:
        reasons.append("mask_quality_review_flag")
    if mask_quality.get("bbox_iou_prompt_mask") is not None and _num(mask_quality.get("bbox_iou_prompt_mask")) < 0.50:
        reasons.append("low_prompt_mask_alignment")
    if mask_quality.get("mask_touches_border") is True:
        reasons.append("mask_touches_border")
    boundary_shape_score = components.get("boundary_shape_score", 0.0)
    if boundary_shape_score >= 0.55:
        reasons.append("complex_prediction_shape")
    elif boundary_shape_score >= 0.30:
        reasons.append("moderate_prediction_shape_complexity")
    area = _num(mask_quality.get("mask_area_ratio"))
    if area > 0 and area < 0.03:
        reasons.append("small_mask")
    if area > 0.35:
        reasons.append("large_mask")
    return _dedupe(reasons)


def _diversity_bin(record: dict, item: dict) -> str:
    mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
    mask_area = _num(mask_quality.get("mask_area_ratio"))
    uncertainty = _num((record.get("uncertainty") or {}).get("disagreement_area_ratio")) if isinstance(record.get("uncertainty"), dict) else 0.0
    risk = item["score_components"]["correction_risk_score"]
    return "|".join(
        [
            _bin(mask_area, 0.03, 0.35, "small", "medium", "large"),
            _bin(uncertainty, 0.05, 0.15, "low_uncertainty", "medium_uncertainty", "high_uncertainty"),
            _bin(risk, 0.40, 0.70, "low_risk", "medium_risk", "high_risk"),
        ]
    )


def _weighted_score(components: dict, weights: dict) -> float:
    return _clamp(
        sum(_clamp(_num(components.get(key))) * _num(weight) for key, weight in weights.items())
        / max(0.000001, sum(_num(value) for value in weights.values()))
    )


def _score_weights(weights: Optional[dict]) -> dict:
    raw = dict(DEFAULT_SCORE_WEIGHTS)
    if isinstance(weights, dict):
        for key in raw:
            if key in weights:
                raw[key] = max(0.0, _num(weights[key]))
    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return {key: value / total for key, value in raw.items()}


def _resolve_score_weights(
    weights: Optional[dict],
    weight_preset: Optional[str],
    weight_presets_file: Optional[str],
) -> tuple[dict, str]:
    env_preset = os.getenv("IMAGE_SEG_REVIEW_WEIGHT_PRESET")
    env_file = os.getenv("IMAGE_SEG_REVIEW_WEIGHT_PRESETS_FILE")
    preset_name = weight_preset or env_preset or DEFAULT_REVIEW_WEIGHT_PRESET
    if weights is not None:
        return _score_weights(weights), preset_name
    if preset_name in {"", "current", DEFAULT_REVIEW_WEIGHT_PRESET}:
        return _score_weights(None), DEFAULT_REVIEW_WEIGHT_PRESET
    presets = _load_weight_presets(weight_presets_file or env_file)
    preset = presets.get(preset_name)
    if not isinstance(preset, dict):
        raise ValueError(f"unknown review weight preset: {preset_name}")
    return _score_weights(preset), preset_name


def _load_weight_presets(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("review weight presets file must contain a mapping")
    return data


def _record_correction_risk(record: dict) -> dict:
    value = record.get("correction_risk") if isinstance(record, dict) else None
    return value if isinstance(value, dict) else {}


def _write_queue(path: str, items: list[dict]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _priority_bucket(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _bbox_aspect(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        left, top, right, bottom = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return 0.0
    return (right - left) / (bottom - top) if bottom > top else 0.0


def _bin(value: float, low_threshold: float, high_threshold: float, low: str, medium: str, high: str) -> str:
    if value < low_threshold:
        return low
    if value < high_threshold:
        return medium
    return high


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
