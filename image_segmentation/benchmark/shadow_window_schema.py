from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.prediction_feature_scoring import PREDICTION_FEATURE_NAMES, prediction_features


LEARNED_SCORE_FIELD = "learned_boundary_shape_only_score"
WINDOW_QUEUE_NAME = "live_review_queue.shadow_scored.jsonl"
FORBIDDEN_OUTPUT_KEYS = {
    "gt_mask",
    "gt_mask_path",
    "mask_path",
    "iou",
    "dice",
    "boundary_iou",
    "boundary_f1",
    "correction_delta",
    "boundary_delta",
    "major_correction_label",
    "severity_label",
    "labels",
    "evaluation_only",
    "boundary_metadata",
    "delta",
}
SAFE_METADATA_KEYS = (
    "dataset",
    "benchmark_name",
    "source",
    "prediction_source",
    "model_name",
    "project_id",
    "queue_type",
)


def classify_shadow_windows(multi_window_root: str, output_dir: str) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    windows = classify_window_root(root)
    pairwise = pairwise_window_compatibility(windows)
    rows = []
    for window in windows:
        rows.extend(window.get("safe_rows") or [])
    public_windows = [_public_window_summary(window) for window in windows]
    report = {
        "multi_window_root": str(root),
        "output_dir": str(output),
        "windows": public_windows,
        "pairwise_same_sample_overlap_matrix": pairwise["same_sample_overlap_matrix"],
        "pairwise_compatibility_matrix": pairwise["compatibility_matrix"],
        "safe_fields_only": True,
        "production_safe_classification": True,
    }
    _write_json(output / "shadow_window_classification.json", report)
    _write_jsonl(output / "window_schema_rows.jsonl", rows)
    (output / "shadow_window_classification.md").write_text(render_window_classification_markdown(report), encoding="utf-8")
    return report


def classify_window_root(root: Path) -> list[dict]:
    windows = []
    for path in sorted(root.iterdir() if root.exists() else []):
        if not path.is_dir():
            continue
        queue = _window_queue_path(path)
        if queue is None:
            continue
        windows.append(classify_shadow_window(path.name, _read_jsonl(queue), queue_path=queue))
    return windows


def classify_shadow_window(window_id: str, items: list[dict], queue_path: str | Path | None = None) -> dict:
    rows = [_safe_row_summary(window_id, row) for row in items]
    row_count = len(items)
    metadata = [_metadata(row) for row in items]
    learned_values = [_shadow_value(row, LEARNED_SCORE_FIELD) for row in items]
    learned_non_null = sum(value is not None for value in learned_values)
    missing_names = sorted({name for row in items for name in _missing_feature_names(row)})
    safe_presence_rates = _safe_feature_presence_rates(items)
    safe_feature_presence_rate = _mean(list(safe_presence_rates.values()))
    prediction_features_rate = _row_rate(items, lambda row: isinstance(row.get("prediction_features"), dict) and bool(row.get("prediction_features")))
    boundary_shape_rate = _row_rate(
        items,
        lambda row: isinstance(row.get("mask_quality"), dict)
        and isinstance((row.get("mask_quality") or {}).get("prediction_time_boundary_shape"), dict)
        and bool((row.get("mask_quality") or {}).get("prediction_time_boundary_shape")),
    )
    neutral_fallback_rate = _neutral_fallback_rate(items, learned_values)
    row_feature_presence_rates = [_row_feature_presence_rate(row) for row in items]
    feature_completeness_bucket = _feature_completeness_bucket(
        safe_feature_presence_rate,
        safe_presence_rates,
        row_feature_presence_rates,
        row_count,
    )
    schema_version = _queue_schema_version(
        row_count=row_count,
        prediction_features_rate=prediction_features_rate,
        boundary_shape_rate=boundary_shape_rate,
        safe_feature_presence_rate=safe_feature_presence_rate,
        neutral_fallback_rate=neutral_fallback_rate,
        feature_completeness_bucket=feature_completeness_bucket,
    )
    sample_ids = [_sample_key(row) for row in items]
    safe_metadata_buckets = [_safe_metadata_bucket(row) for row in items]
    return {
        "window_id": window_id,
        "queue_path": str(queue_path) if queue_path is not None else None,
        "queue_schema_version": schema_version,
        "feature_completeness_bucket": feature_completeness_bucket,
        "queue_type": _infer_queue_type(items),
        "sample_set_fingerprint": _fingerprint(sorted(sample_ids)),
        "safe_metadata_fingerprint": _fingerprint(sorted(json.dumps(bucket, sort_keys=True) for bucket in safe_metadata_buckets)),
        "feature_completeness_signature": _fingerprint(sorted(f"{name}:{rate:.6f}" for name, rate in safe_presence_rates.items())),
        "rows": row_count,
        "safe_feature_presence_rate": safe_feature_presence_rate,
        "safe_feature_presence_rates": safe_presence_rates,
        "prediction_features_presence_rate": prediction_features_rate,
        "prediction_time_boundary_shape_presence_rate": boundary_shape_rate,
        "learned_score_coverage": float(learned_non_null / row_count) if row_count else None,
        "learned_score_null_rate": float((row_count - learned_non_null) / row_count) if row_count else None,
        "neutral_fallback_rate": neutral_fallback_rate,
        "artifact_status_counts": dict(sorted(Counter(str(row.get("artifact_validation_status") or "missing") for row in metadata).items())),
        "missing_feature_names": missing_names,
        "missing_feature_name_counts": dict(sorted(Counter(name for row in items for name in _missing_feature_names(row)).items())),
        "shadow_only_all_true": all(row.get("shadow_only") is True for row in metadata) if metadata else False,
        "affects_default_ranking_all_false": all(row.get("affects_default_ranking") is False for row in metadata) if metadata else False,
        "valid_shadow_metadata": bool(metadata),
        "sample_ids_hash_inputs": sample_ids,
        "safe_rows": rows,
    }


def pairwise_window_compatibility(windows: list[dict]) -> dict:
    overlap_matrix = {}
    compatibility_matrix = {}
    by_id = {row["window_id"]: row for row in windows}
    for left_id, right_id in combinations(sorted(by_id), 2):
        left = by_id[left_id]
        right = by_id[right_id]
        key = f"{left_id}__{right_id}"
        left_ids = set(left.get("sample_ids_hash_inputs") or [])
        right_ids = set(right.get("sample_ids_hash_inputs") or [])
        shared = len(left_ids & right_ids)
        denominator = min(len(left_ids), len(right_ids)) if left_ids and right_ids else 0
        overlap = {
            "left_rows": left.get("rows"),
            "right_rows": right.get("rows"),
            "same_sample_overlap": shared,
            "same_sample_overlap_rate_min_window": float(shared / denominator) if denominator else None,
        }
        overlap_matrix[key] = overlap
        compatibility_matrix[key] = classify_pairwise_compatibility(left, right, overlap)
    return {"same_sample_overlap_matrix": overlap_matrix, "compatibility_matrix": compatibility_matrix}


def classify_pairwise_compatibility(window_a: dict, window_b: dict, overlap: dict | None = None) -> dict:
    overlap = overlap or _overlap(window_a, window_b)
    reasons = []
    if window_a.get("queue_schema_version") != window_b.get("queue_schema_version"):
        reasons.append(f"schema differs: {window_a.get('queue_schema_version')} vs {window_b.get('queue_schema_version')}")
    if not _compatible_feature_bucket(window_a.get("feature_completeness_bucket"), window_b.get("feature_completeness_bucket")):
        reasons.append(
            f"feature completeness differs: {window_a.get('feature_completeness_bucket')} vs {window_b.get('feature_completeness_bucket')}"
        )
    if not _compatible_queue_type(window_a.get("queue_type"), window_b.get("queue_type")):
        reasons.append(f"queue type differs: {window_a.get('queue_type')} vs {window_b.get('queue_type')}")
    if not (window_a.get("valid_shadow_metadata") and window_b.get("valid_shadow_metadata")):
        reasons.append("shadow metadata missing")
    if not (window_a.get("shadow_only_all_true") and window_b.get("shadow_only_all_true")):
        reasons.append("shadow_only is not true for all rows")
    if not (window_a.get("affects_default_ranking_all_false") and window_b.get("affects_default_ranking_all_false")):
        reasons.append("affects_default_ranking is not false for all rows")
    sample_set_changed = (overlap.get("same_sample_overlap") or 0) < min(window_a.get("rows") or 0, window_b.get("rows") or 0)
    direct_topk_limited = (overlap.get("same_sample_overlap") or 0) == 0
    return {
        "compatible_for_drift_comparison": not reasons,
        "reason": "same schema + compatible feature completeness bucket + compatible queue type" if not reasons else "; ".join(reasons),
        "sample_set_changed": bool(sample_set_changed),
        "same_sample_overlap": overlap.get("same_sample_overlap"),
        "same_sample_overlap_rate_min_window": overlap.get("same_sample_overlap_rate_min_window"),
        "direct_topk_comparison_limited": bool(direct_topk_limited),
    }


def render_window_classification_markdown(report: dict) -> str:
    lines = [
        "# Shadow Window Classification",
        "",
        f"- multi_window_root: {report.get('multi_window_root')}",
        f"- safe_fields_only: {report.get('safe_fields_only')}",
        f"- production_safe_classification: {report.get('production_safe_classification')}",
        "",
        "| window | schema | completeness | queue type | rows | learned coverage | null rate | safe feature rate | neutral fallback | sample fingerprint |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("windows") or []:
        lines.append(
            f"| {row.get('window_id')} | {row.get('queue_schema_version')} | {row.get('feature_completeness_bucket')} | "
            f"{row.get('queue_type')} | {row.get('rows')} | {_fmt(row.get('learned_score_coverage'))} | "
            f"{_fmt(row.get('learned_score_null_rate'))} | {_fmt(row.get('safe_feature_presence_rate'))} | "
            f"{_fmt(row.get('neutral_fallback_rate'))} | {row.get('sample_set_fingerprint')} |"
        )
    lines.extend(["", "## Missing Features", ""])
    for row in report.get("windows") or []:
        lines.append(f"- {row.get('window_id')}: {row.get('missing_feature_names')}")
    lines.extend(["", "## Pairwise Compatibility", ""])
    for key, row in (report.get("pairwise_compatibility_matrix") or {}).items():
        lines.append(
            f"- {key}: compatible={row.get('compatible_for_drift_comparison')} "
            f"sample_overlap={row.get('same_sample_overlap')} reason={row.get('reason')}"
        )
    return "\n".join(lines) + "\n"


def _safe_row_summary(window_id: str, row: dict) -> dict:
    metadata = _metadata(row)
    return {
        "window_id": window_id,
        "sample_id_hash": _fingerprint([_sample_key(row)]),
        "sample_key_type": _sample_key(row).split(":", 1)[0],
        "queue_type": _infer_queue_type([row]),
        "safe_metadata_bucket": _safe_metadata_bucket(row),
        "safe_feature_presence_rate": _row_feature_presence_rate(row),
        "missing_feature_names": _missing_feature_names(row),
        "learned_score_present": _shadow_value(row, LEARNED_SCORE_FIELD) is not None,
        "artifact_status": metadata.get("artifact_validation_status") or "missing",
        "shadow_only": metadata.get("shadow_only"),
        "affects_default_ranking": metadata.get("affects_default_ranking"),
    }


def _public_window_summary(window: dict) -> dict:
    return {
        key: value
        for key, value in window.items()
        if key not in {"sample_ids_hash_inputs", "safe_rows"}
    }


def _queue_schema_version(
    *,
    row_count: int,
    prediction_features_rate: float | None,
    boundary_shape_rate: float | None,
    safe_feature_presence_rate: float | None,
    neutral_fallback_rate: float | None,
    feature_completeness_bucket: str,
) -> str:
    if row_count <= 0:
        return "unknown"
    if _none_to_zero(prediction_features_rate) >= 0.95 and _none_to_zero(boundary_shape_rate) >= 0.95 and _none_to_zero(safe_feature_presence_rate) >= 0.95:
        return "full_prediction_features"
    if feature_completeness_bucket == "mixed":
        return "mixed"
    if _none_to_zero(boundary_shape_rate) < 0.5 and (_none_to_zero(neutral_fallback_rate) >= 0.5 or _none_to_zero(safe_feature_presence_rate) < 0.5):
        return "old_schema_fallback"
    if prediction_features_rate is None and boundary_shape_rate is None:
        return "unknown"
    return "unknown"


def _feature_completeness_bucket(
    safe_feature_presence_rate: float | None,
    feature_rates: dict[str, float],
    row_rates: list[float],
    rows: int,
) -> str:
    if rows <= 0 or safe_feature_presence_rate is None:
        return "unknown"
    rates = list(feature_rates.values())
    if rates and max(rates) - min(rates) >= 0.50 and 0.05 < safe_feature_presence_rate < 0.95:
        return "mixed"
    if row_rates and min(row_rates) < 0.50 and max(row_rates) >= 0.95 and 0.05 < safe_feature_presence_rate < 0.95:
        return "mixed"
    if safe_feature_presence_rate >= 0.95:
        return "full"
    if safe_feature_presence_rate >= 0.50:
        return "partial"
    return "missing_boundary_shape"


def _infer_queue_type(items: list[dict]) -> str:
    values = []
    for row in items:
        for key in ("queue_type", "source", "benchmark_name", "dataset", "prediction_source"):
            value = row.get(key)
            if value is not None:
                values.append(str(value).lower())
    text = " ".join(values)
    if any(token in text for token in ("benchmark_replay", "replay", "benchmark", "coco", "lvis", "cod10k", "camo", "dis5k", "open_images")):
        return "benchmark_replay"
    if any(token in text for token in ("production_like", "prod_like", "production")):
        return "production_like"
    if "live" in text:
        return "live"
    return "unknown"


def _safe_feature_presence_rates(items: list[dict]) -> dict[str, float]:
    if not items:
        return {name: 0.0 for name in PREDICTION_FEATURE_NAMES}
    counts = Counter()
    for row in items:
        features = prediction_features(row)
        for name in PREDICTION_FEATURE_NAMES:
            if _present(features.get(name)):
                counts[name] += 1
    return {name: float(counts[name] / len(items)) for name in PREDICTION_FEATURE_NAMES}


def _row_feature_presence_rate(row: dict) -> float:
    features = prediction_features(row)
    present = sum(_present(features.get(name)) for name in PREDICTION_FEATURE_NAMES)
    return float(present / len(PREDICTION_FEATURE_NAMES))


def _missing_feature_names(row: dict) -> list[str]:
    metadata = _metadata(row)
    names = metadata.get("missing_safe_features")
    if isinstance(names, list):
        return sorted(str(name) for name in names)
    features = prediction_features(row)
    return [name for name in PREDICTION_FEATURE_NAMES if not _present(features.get(name))]


def _safe_metadata_bucket(row: dict) -> dict:
    return {key: row.get(key) for key in SAFE_METADATA_KEYS if row.get(key) is not None}


def _neutral_fallback_rate(items: list[dict], learned_values: list[float | None]) -> float | None:
    if not items:
        return None
    fallback_rows = 0
    for row, learned in zip(items, learned_values):
        metadata = _metadata(row)
        missing_count = metadata.get("missing_safe_feature_count")
        missing = int(missing_count or 0) if _looks_int(missing_count) else len(_missing_feature_names(row))
        if missing > 0 and learned is not None:
            fallback_rows += 1
    return float(fallback_rows / len(items))


def _metadata(row: dict) -> dict:
    return row.get("shadow_score_metadata") if isinstance(row.get("shadow_score_metadata"), dict) else {}


def _shadow_value(row: dict, field: str) -> float | None:
    shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
    try:
        value = float(shadow.get(field))
    except (TypeError, ValueError):
        return None
    return value


def _sample_key(row: dict) -> str:
    for key in ("sample_id", "image_id", "annotation_id", "task_id", "prediction_id"):
        if row.get(key) is not None:
            return f"{key}:{row.get(key)}"
    safe = {key: row.get(key) for key in SAFE_METADATA_KEYS if row.get(key) is not None}
    safe["rank"] = row.get("rank")
    return json.dumps(safe, sort_keys=True)


def _window_queue_path(window_dir: Path) -> Path | None:
    preferred = window_dir / WINDOW_QUEUE_NAME
    if preferred.exists():
        return preferred
    matches = sorted(window_dir.glob("*.shadow_scored.jsonl"))
    return matches[0] if matches else None


def _overlap(window_a: dict, window_b: dict) -> dict:
    left_ids = set(window_a.get("sample_ids_hash_inputs") or [])
    right_ids = set(window_b.get("sample_ids_hash_inputs") or [])
    shared = len(left_ids & right_ids)
    denominator = min(len(left_ids), len(right_ids)) if left_ids and right_ids else 0
    return {
        "same_sample_overlap": shared,
        "same_sample_overlap_rate_min_window": float(shared / denominator) if denominator else None,
    }


def _compatible_feature_bucket(left: Any, right: Any) -> bool:
    if left == right:
        return True
    return {left, right} <= {"full", "partial"}


def _compatible_queue_type(left: Any, right: Any) -> bool:
    if left == right:
        return True
    return "unknown" in {left, right}


def _row_rate(items: list[dict], predicate) -> float | None:
    if not items:
        return None
    return float(sum(bool(predicate(row)) for row in items) / len(items))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _none_to_zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _looks_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _fingerprint(values: list[str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_strip_forbidden_recursive(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_strip_forbidden_recursive(row), ensure_ascii=False) + "\n")


def _strip_forbidden_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden_recursive(item)
            for key, item in value.items()
            if key not in FORBIDDEN_OUTPUT_KEYS
        }
    if isinstance(value, list):
        return [_strip_forbidden_recursive(item) for item in value]
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
