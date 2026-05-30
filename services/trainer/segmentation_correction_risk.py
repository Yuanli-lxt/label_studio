from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - exercised through helper behavior
    joblib = None


FEATURE_NAMES = [
    "mask_quality.mask_area_ratio",
    "mask_quality.bbox_iou_prompt_mask",
    "mask_quality.mask_touches_border",
    "mask_quality.valid_mask",
    "mask_quality.rle_length",
    "mask_quality.image_width",
    "mask_quality.image_height",
    "review.needs_review",
    "review.review_priority_score",
    "review.review_priority_level",
    "review.review_reason_count",
    "uncertainty.enabled",
    "uncertainty.num_prompt_variants",
    "uncertainty.num_valid_masks",
    "uncertainty.mean_pairwise_iou",
    "uncertainty.min_pairwise_iou",
    "uncertainty.max_pairwise_iou",
    "uncertainty.disagreement_area_ratio",
    "uncertainty.stable",
    "uncertainty.stability_bucket_level",
    "uncertainty.reason_count",
    "geometry.prompt_bbox_width_ratio",
    "geometry.prompt_bbox_height_ratio",
    "geometry.prompt_bbox_area_ratio",
    "geometry.model_mask_bbox_width_ratio",
    "geometry.model_mask_bbox_height_ratio",
    "geometry.model_mask_bbox_area_ratio",
    "geometry.image_aspect_ratio",
    "geometry.model_bbox_aspect_ratio",
]

REVIEW_PRIORITY_LEVELS = {"low": 0.0, "medium": 1.0, "high": 2.0}
STABILITY_BUCKET_LEVELS = {"unknown": -1.0, "low": 0.0, "medium": 1.0, "high": 2.0}
MISSING_SKLEARN_MESSAGE = "scikit-learn is required for correction-risk training"


def load_correction_delta_records(path: str) -> list[dict]:
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


def extract_correction_risk_features(record: dict) -> dict:
    record = record if isinstance(record, dict) else {}
    mask_quality = record.get("mask_quality") if isinstance(record.get("mask_quality"), dict) else {}
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    uncertainty = record.get("uncertainty") if isinstance(record.get("uncertainty"), dict) else {}

    image_width = _num(record.get("image_width")) or _num(mask_quality.get("image_width"))
    image_height = _num(record.get("image_height")) or _num(mask_quality.get("image_height"))

    features = {
        "mask_quality.mask_area_ratio": _num(mask_quality.get("mask_area_ratio")),
        "mask_quality.bbox_iou_prompt_mask": _num(mask_quality.get("bbox_iou_prompt_mask")),
        "mask_quality.mask_touches_border": _bool(mask_quality.get("mask_touches_border")),
        "mask_quality.valid_mask": _bool(mask_quality.get("valid_mask")),
        "mask_quality.rle_length": _num(mask_quality.get("rle_length")),
        "mask_quality.image_width": _num(mask_quality.get("image_width")),
        "mask_quality.image_height": _num(mask_quality.get("image_height")),
        "review.needs_review": _bool(review.get("needs_review")),
        "review.review_priority_score": _num(review.get("review_priority_score")),
        "review.review_priority_level": REVIEW_PRIORITY_LEVELS.get(
            str(review.get("review_priority") or "").strip().lower(),
            0.0,
        ),
        "review.review_reason_count": float(len(review.get("review_reason") or []))
        if isinstance(review.get("review_reason"), list)
        else 0.0,
        "uncertainty.enabled": _bool(uncertainty.get("enabled")),
        "uncertainty.num_prompt_variants": _num(uncertainty.get("num_prompt_variants")),
        "uncertainty.num_valid_masks": _num(uncertainty.get("num_valid_masks")),
        "uncertainty.mean_pairwise_iou": _num(uncertainty.get("mean_pairwise_iou")),
        "uncertainty.min_pairwise_iou": _num(uncertainty.get("min_pairwise_iou")),
        "uncertainty.max_pairwise_iou": _num(uncertainty.get("max_pairwise_iou")),
        "uncertainty.disagreement_area_ratio": _num(uncertainty.get("disagreement_area_ratio")),
        "uncertainty.stable": _bool(uncertainty.get("stable")),
        "uncertainty.stability_bucket_level": STABILITY_BUCKET_LEVELS.get(
            str(uncertainty.get("stability_bucket") or "unknown").strip().lower(),
            -1.0,
        ),
        "uncertainty.reason_count": float(len(uncertainty.get("reason") or []))
        if isinstance(uncertainty.get("reason"), list)
        else 0.0,
        "geometry.image_aspect_ratio": _safe_div(image_width, image_height),
    }
    features.update(_bbox_features("geometry.prompt_bbox", record.get("prompt_bbox"), image_width, image_height))
    features.update(
        _bbox_features("geometry.model_mask_bbox", record.get("model_mask_bbox"), image_width, image_height)
    )
    model_width = features["geometry.model_mask_bbox_width_ratio"] * image_width if image_width else 0.0
    model_height = features["geometry.model_mask_bbox_height_ratio"] * image_height if image_height else 0.0
    features["geometry.model_bbox_aspect_ratio"] = _safe_div(model_width, model_height)
    return {name: float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}


def build_correction_risk_dataset(records: list[dict]) -> tuple[list[dict], list[int], dict]:
    rows = []
    labels = []
    skipped = 0
    for record in records if isinstance(records, list) else []:
        label = _target_label(record)
        if label is None:
            skipped += 1
            continue
        rows.append(extract_correction_risk_features(record))
        labels.append(label)

    positives = int(sum(labels))
    negatives = int(len(labels) - positives)
    summary = {
        "records_total": len(records) if isinstance(records, list) else 0,
        "usable_records": len(labels),
        "positive_records": positives,
        "negative_records": negatives,
        "skipped_records": skipped,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
    }
    return rows, labels, summary


def train_correction_risk_model(
    records: list[dict],
    output_dir: str,
    min_records: int = 8,
    min_positive: int = 2,
    min_negative: int = 2,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    feature_rows, labels, summary = build_correction_risk_dataset(records)
    paths = _artifact_paths(output_dir)
    quality_skip = _quality_skip_reason(summary, min_records, min_positive, min_negative)
    if quality_skip:
        metadata = _metadata_base(summary, paths)
        metadata["correction_risk"].update(
            {
                "status": "skipped",
                "skip_reason": quality_skip,
                "quality_rules": {
                    "min_records": min_records,
                    "min_positive": min_positive,
                    "min_negative": min_negative,
                },
            }
        )
        _write_json(paths["metadata_path"], metadata)
        return metadata

    x = _feature_matrix(feature_rows, FEATURE_NAMES)
    y = np.asarray(labels, dtype=int)
    try:
        classifier, model_type = _fit_logistic_classifier(x, y)
        probabilities = classifier.predict_proba(x)[:, 1]
        predictions = classifier.predict(x)
        metrics = _training_metrics(y, predictions, probabilities)
        _dump_classifier(classifier, model_type, paths["classifier_path"])
        _write_json(paths["feature_names_path"], FEATURE_NAMES)
        _write_training_dataset(paths["training_dataset_path"], feature_rows, labels, records)
        metadata = _metadata_base(summary, paths)
        metadata["correction_risk"].update(
            {
                "status": "trained",
                "skip_reason": None,
                "model_type": model_type,
                "target": "major_correction",
                "metrics": metrics,
                "feature_weights": _feature_weights(classifier, FEATURE_NAMES),
                "model_version": f"seg-correction-risk-v{_utc_compact()}",
            }
        )
        _write_json(paths["metadata_path"], metadata)
        return metadata
    except ImportError:
        metadata = _metadata_base(summary, paths)
        metadata["correction_risk"].update(
            {
                "status": "error",
                "skip_reason": "missing_scikit_learn",
                "error": "missing_scikit_learn",
                "message": MISSING_SKLEARN_MESSAGE,
                "model_type": "LogisticRegression",
                "target": "major_correction",
            }
        )
        _write_json(paths["metadata_path"], metadata)
        return metadata
    except Exception as exc:
        metadata = _metadata_base(summary, paths)
        metadata["correction_risk"].update(
            {
                "status": "error",
                "skip_reason": "training_failed",
                "error": str(exc),
                "model_type": "LogisticRegression",
                "target": "major_correction",
            }
        )
        _write_json(paths["metadata_path"], metadata)
        return metadata


def predict_correction_risk(record: dict, model_artifact_dir: str) -> dict:
    paths = _artifact_paths(model_artifact_dir)
    classifier = _load_classifier(paths["classifier_path"])
    metadata = _read_json(paths["metadata_path"]) or {}
    feature_names = _read_json(paths["feature_names_path"]) or FEATURE_NAMES
    features = extract_correction_risk_features(record)
    x = _feature_matrix([features], feature_names)
    if hasattr(classifier, "predict_proba"):
        risk_score = float(classifier.predict_proba(x)[0][1])
    else:
        risk_score = float(classifier.predict(x)[0])
    weights = {
        row.get("feature"): float(row.get("weight", 0.0))
        for row in (metadata.get("correction_risk", {}).get("feature_weights") or [])
        if isinstance(row, dict)
    }
    return {
        "correction_risk": {
            "model_version": metadata.get("correction_risk", {}).get("model_version"),
            "risk_score": risk_score,
            "risk_bucket": _risk_bucket(risk_score),
            "predicted_major_correction": bool(risk_score >= 0.5),
            "top_risk_features": _top_risk_features(features, weights),
        }
    }


def _metadata_base(summary: dict, paths: dict) -> dict:
    return {
        "correction_risk": {
            "enabled": True,
            "status": None,
            "skip_reason": None,
            "model_type": None,
            "target": "major_correction",
            "records_total": summary["records_total"],
            "usable_records": summary["usable_records"],
            "positive_records": summary["positive_records"],
            "negative_records": summary["negative_records"],
            "feature_count": summary["feature_count"],
            "feature_names": summary["feature_names"],
            "training_artifacts": {
                "classifier_path": paths["classifier_path"],
                "feature_names_path": paths["feature_names_path"],
                "metadata_path": paths["metadata_path"],
                "training_dataset_path": paths["training_dataset_path"],
            },
        }
    }


def _fit_logistic_classifier(x: np.ndarray, y: np.ndarray) -> tuple[Any, str]:
    LogisticRegression, *_ = _sklearn_training_dependencies()

    classifier = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    classifier.fit(x, y)
    return classifier, "LogisticRegression"


def _dump_classifier(classifier: Any, model_type: str, path: str) -> None:
    if joblib is None:
        raise ImportError("joblib")
    joblib.dump(classifier, path)


def _load_classifier(path: str) -> Any:
    if joblib is None:
        raise RuntimeError("joblib is required to load correction-risk artifacts")
    try:
        _sklearn_prediction_dependencies()
        return joblib.load(path)
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to load correction-risk artifacts") from exc


def _training_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    _, accuracy_score, average_precision_score, precision_recall_fscore_support = _sklearn_training_dependencies()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "scope": "training",
        "train_accuracy": float(accuracy_score(y_true, y_pred)),
        "train_precision": float(precision),
        "train_recall": float(recall),
        "train_f1": float(f1),
        "train_average_precision": float(average_precision_score(y_true, probabilities)),
    }


def _sklearn_training_dependencies() -> tuple[Any, Any, Any, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support

    return LogisticRegression, accuracy_score, average_precision_score, precision_recall_fscore_support


def _sklearn_prediction_dependencies() -> None:
    from sklearn.linear_model import LogisticRegression  # noqa: F401


def _quality_skip_reason(summary: dict, min_records: int, min_positive: int, min_negative: int) -> Optional[str]:
    if summary["usable_records"] < min_records:
        return "not_enough_records"
    if summary["positive_records"] < min_positive:
        return "not_enough_positive_records"
    if summary["negative_records"] < min_negative:
        return "not_enough_negative_records"
    return None


def _feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows], dtype=float)


def _feature_weights(classifier: Any, feature_names: list[str]) -> list[dict]:
    coef = getattr(classifier, "coef_", None)
    if coef is None:
        return []
    weights = [
        {"feature": name, "weight": float(weight)}
        for name, weight in zip(feature_names, np.asarray(coef)[0].tolist())
    ]
    return sorted(weights, key=lambda row: abs(row["weight"]), reverse=True)


def _write_training_dataset(path: str, feature_rows: list[dict], labels: list[int], records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        usable_idx = 0
        for record in records:
            label = _target_label(record)
            if label is None:
                continue
            row = {
                "task_id": record.get("task_id"),
                "prediction_id": record.get("prediction_id"),
                "annotation_id": record.get("annotation_id"),
                "target_major_correction": labels[usable_idx],
                "features": feature_rows[usable_idx],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            usable_idx += 1


def _target_label(record: Any) -> Optional[int]:
    if not isinstance(record, dict):
        return None
    if record.get("record_status") not in (None, "ok"):
        return None
    delta = record.get("delta") if isinstance(record.get("delta"), dict) else {}
    major = delta.get("major_correction")
    if isinstance(major, bool):
        return int(major)
    severity = delta.get("correction_severity")
    if isinstance(severity, str) and severity:
        return int(severity.strip().lower() == "major")
    return None


def _bbox_features(prefix: str, bbox: Any, image_width: float, image_height: float) -> dict:
    normalized = _bbox(bbox)
    if normalized is None:
        return {
            f"{prefix}_width_ratio": 0.0,
            f"{prefix}_height_ratio": 0.0,
            f"{prefix}_area_ratio": 0.0,
        }
    left, top, right, bottom = normalized
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    image_area = image_width * image_height if image_width and image_height else 0.0
    return {
        f"{prefix}_width_ratio": _safe_div(width, image_width),
        f"{prefix}_height_ratio": _safe_div(height, image_height),
        f"{prefix}_area_ratio": _safe_div(width * height, image_area),
    }


def _bbox(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return number


def _bool(value: Any) -> float:
    return 1.0 if value is True else 0.0


def _safe_div(numerator: Any, denominator: Any) -> float:
    numerator = _num(numerator)
    denominator = _num(denominator)
    return float(numerator / denominator) if denominator else 0.0


def _risk_bucket(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _top_risk_features(features: dict, weights: dict) -> list[dict]:
    rows = []
    for name in FEATURE_NAMES:
        value = float(features.get(name, 0.0) or 0.0)
        weight = float(weights.get(name, 0.0) or 0.0)
        rows.append({"feature": name, "value": value, "weight": weight, "contribution": value * weight})
    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return rows[:5]


def _artifact_paths(output_dir: str) -> dict:
    return {
        "classifier_path": os.path.join(output_dir, "classifier.joblib"),
        "metadata_path": os.path.join(output_dir, "metadata.json"),
        "feature_names_path": os.path.join(output_dir, "feature_names.json"),
        "training_dataset_path": os.path.join(output_dir, "training_dataset.jsonl"),
    }


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
