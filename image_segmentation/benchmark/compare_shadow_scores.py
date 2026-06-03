from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.prediction_feature_scoring import safe_prediction_feature_vector, target_label


SHADOW_SCORE_FIELDS = [
    "boundary_shape_calibrated_score",
    "boundary_shape_rank_score",
    "learned_boundary_shape_only_score",
    "learned_current_plus_boundary_shape_score",
    "gated_boundary_shape_score",
    "gated_current_boundary_score",
]

DEFAULT_ALERT_THRESHOLDS = {
    "learned_score_null_rate": 0.05,
    "inference_error_rate": 0.01,
    "missing_safe_feature_count_p95": 0.0,
    "queue_generation_latency_regression": 0.10,
    "shadow_scoring_latency_p95_ms": None,
    "artifact_load_latency_ms": None,
    "score_distribution_p95_shift_std": 3.0,
    "top100_jaccard_relative_change": 0.50,
    "require_all_active_artifacts_valid": True,
    "require_shadow_only_true": True,
    "require_affects_default_ranking_false": True,
}


def compare_shadow_scores(
    review_queue: str,
    output_dir: str,
    window_id: str | None = None,
    baseline: str | None = None,
    production_mode: bool = False,
    no_labels: bool = False,
    performance_context: dict | None = None,
) -> dict:
    monitoring_started = time.perf_counter()
    items = _read_jsonl(review_queue)
    current_scores = [_clip(_num(item.get("priority_score"))) for item in items]
    labels_enabled = not production_mode and not no_labels
    labels = [target_label(item) for item in items] if labels_enabled else []
    has_labels = any(label is not None for label in labels) if labels_enabled else False
    fields = {}
    disagreement_rows = []
    for field in SHADOW_SCORE_FIELDS:
        values = [_shadow_value(item, field) for item in items]
        non_null = [value for value in values if value is not None]
        field_report = {
            "non_null_count": len(non_null),
            "null_count": len(values) - len(non_null),
            "null_rate": float((len(values) - len(non_null)) / len(values)) if values else None,
            "coverage": _coverage(items, values),
            "distribution": _distribution(non_null),
            "topk_overlap": {
                "k20": _topk_overlap(current_scores, values, 20),
                "k50": _topk_overlap(current_scores, values, 50),
                "k100": _topk_overlap(current_scores, values, 100),
            },
            "correlation": {
                "spearman": _spearman(current_scores, values),
                "pearson": _pearson(current_scores, values),
            },
            "rank_delta_summary": _rank_delta_summary(current_scores, values),
        }
        if has_labels and non_null:
            labeled_pairs = [(label, 0.0 if value is None else float(value)) for label, value in zip(labels, values) if label is not None]
            label_subset = [1 if label else 0 for label, _ in labeled_pairs]
            filled = [score for _, score in labeled_pairs]
            field_report["offline_label_metrics"] = {
                "evaluation_only": True,
                "average_precision": _average_precision(label_subset, filled),
                "roc_auc": _roc_auc(label_subset, filled),
                "brier": float(np.mean([(score - label) ** 2 for score, label in zip(filled, label_subset)])) if filled else None,
                "lift_at_20": _lift_at_k(label_subset, filled, 20),
                "precision_at_20": _precision_at_k(label_subset, filled, 20),
                "bucket_major_rate": _bucket_major_rate(filled, label_subset),
                "quintile_table": _bucket_major_rate(filled, label_subset, buckets=5),
                "decile_table": _bucket_major_rate(filled, label_subset, buckets=10),
            }
        fields[field] = field_report
        disagreement_rows.extend(_disagreement_examples(items, current_scores, values, field))
    metadata = _metadata_summary(items)
    baseline_report = _read_json(Path(baseline)) if baseline else None
    performance = _performance_summary(items, baseline_report, performance_context=performance_context)
    performance["monitoring_latency_ms"] = (time.perf_counter() - monitoring_started) * 1000.0
    distribution_drift = _distribution_drift(fields, baseline_report, performance=performance)
    report = {
        "input_review_queue": review_queue,
        "window_id": window_id,
        "production_mode": bool(production_mode),
        "labels_read": bool(labels_enabled),
        "n_samples": len(items),
        "shadow_score_fields": SHADOW_SCORE_FIELDS,
        "fields": fields,
        "coverage": _queue_coverage(items),
        "distribution_drift": distribution_drift,
        "disagreement_volume": _disagreement_volume(current_scores, [_shadow_value(item, "learned_boundary_shape_only_score") for item in items]),
        "performance": performance,
        "safety_summary": metadata,
        "alert_thresholds": DEFAULT_ALERT_THRESHOLDS,
        "alerts": _alerts(
            fields,
            metadata,
            len(items),
            distribution_drift,
            performance,
            labels_read=bool(labels_enabled),
        ),
        "label_metrics_available": has_labels,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "shadow_monitoring.json", report)
    _write_jsonl(output / "shadow_disagreement_examples.jsonl", disagreement_rows)
    (output / "shadow_monitoring.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Shadow Score Monitoring",
        "",
        f"- input review queue: {report.get('input_review_queue')}",
        f"- window_id: {report.get('window_id')}",
        f"- production_mode: {report.get('production_mode')}",
        f"- labels_read: {report.get('labels_read')}",
        f"- n_samples: {report.get('n_samples')}",
        f"- label_metrics_available: {report.get('label_metrics_available')}",
        "",
        "## Coverage",
        f"- total_rows: {((report.get('coverage') or {}).get('total_rows'))}",
        f"- shadow_scores_present_count: {((report.get('coverage') or {}).get('shadow_scores_present_count'))}",
        f"- learned_score_non_null_count: {((report.get('coverage') or {}).get('learned_score_non_null_count'))}",
        f"- learned_score_null_rate: {_fmt(((report.get('coverage') or {}).get('learned_score_null_rate')))}",
        "",
        "| field | non-null | null rate | median | p95 | spearman vs current | top20 overlap | AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, row in (report.get("fields") or {}).items():
        dist = row.get("distribution") or {}
        metrics = row.get("offline_label_metrics") or {}
        lines.append(
            f"| {field} | {row.get('non_null_count')} | {_fmt(row.get('null_rate'))} | {_fmt(dist.get('median'))} | "
            f"{_fmt(dist.get('p95'))} | {_fmt((row.get('correlation') or {}).get('spearman'))} | "
            f"{_fmt(((row.get('topk_overlap') or {}).get('k20') or {}).get('jaccard'))} | {_fmt(metrics.get('average_precision'))} |"
        )
    safety = report.get("safety_summary") or {}
    lines.extend(
        [
            "",
            "## Safety",
            f"- shadow_only: {safety.get('shadow_only')}",
            f"- affects_default_ranking: {safety.get('affects_default_ranking')}",
            f"- artifact_version: {safety.get('artifact_version')}",
            f"- score_version: {safety.get('score_version')}",
            f"- artifact_validation_status_counts: {safety.get('artifact_validation_status_counts')}",
            f"- inference_error_count: {safety.get('inference_error_count')}",
            f"- leakage_guard_status: {safety.get('leakage_guard_status')}",
            "",
            "## Alerts",
            f"- triggered: {((report.get('alerts') or {}).get('triggered'))}",
            f"- checks: {((report.get('alerts') or {}).get('checks'))}",
        ]
    )
    return "\n".join(lines)


def _metadata_summary(items: list[dict]) -> dict:
    metadata = [item.get("shadow_score_metadata") for item in items if isinstance(item.get("shadow_score_metadata"), dict)]
    first = metadata[0] if metadata else {}
    validation_counts = Counter(str(row.get("artifact_validation_status") or "missing") for row in metadata)
    loaded_counts = Counter(bool(row.get("artifact_loaded")) for row in metadata)
    missing_counts = Counter(str(row.get("missing_safe_feature_count", 0)) for row in metadata)
    return {
        "shadow_only": all((row.get("shadow_only") is True) for row in metadata) if metadata else None,
        "affects_default_ranking": any((row.get("affects_default_ranking") is True) for row in metadata) if metadata else None,
        "artifact_version": first.get("artifact_version"),
        "schema_version": first.get("score_version"),
        "score_version": first.get("score_version"),
        "artifact_loaded_count": int(loaded_counts.get(True, 0)),
        "artifact_validation_status_counts": dict(sorted(validation_counts.items())),
        "missing_feature_count_distribution": dict(sorted(missing_counts.items())),
        "inference_error_count": sum(bool(row.get("inference_error")) for row in metadata),
        "leakage_guard_status": "metadata_present" if metadata else "no_shadow_metadata",
        "fail_open_count": sum(
            str(row.get("artifact_validation_status") or "") in {"missing", "load_failed", "schema_mismatch", "inference_failed"}
            for row in metadata
        ),
    }


def _queue_coverage(items: list[dict]) -> dict:
    learned = [_shadow_value(item, "learned_boundary_shape_only_score") for item in items]
    learned_non_null = sum(value is not None for value in learned)
    metadata = [item.get("shadow_score_metadata") for item in items if isinstance(item.get("shadow_score_metadata"), dict)]
    missing_counts = Counter()
    for row in metadata:
        missing_counts[str(row.get("missing_safe_feature_count", 0))] += 1
    return {
        "total_rows": len(items),
        "shadow_scores_present_count": sum(isinstance(item.get("shadow_scores"), dict) for item in items),
        "learned_score_non_null_count": learned_non_null,
        "learned_score_null_rate": float((len(items) - learned_non_null) / len(items)) if items else None,
        "missing_feature_count_distribution": dict(sorted(missing_counts.items())),
        "artifact_loaded_count": sum(bool(row.get("artifact_loaded")) for row in metadata),
        "artifact_validation_status_counts": dict(
            sorted(Counter(str(row.get("artifact_validation_status") or "missing") for row in metadata).items())
        ),
        "inference_error_count": sum(bool(row.get("inference_error")) for row in metadata),
    }


def _distribution_drift(fields: dict, baseline_report: dict | None, performance: dict | None = None) -> dict:
    learned = (fields.get("learned_boundary_shape_only_score") or {})
    current = learned.get("distribution") or {}
    if not baseline_report:
        return {
            "baseline_available": False,
            "learned_boundary_shape_only_score": {
                "median_delta": None,
                "p95_delta": None,
                "p95_shift_std": None,
                "top100_jaccard_delta": None,
                "top100_jaccard_relative_change": None,
                "score_distribution_psi": None,
                "null_rate_delta": None,
                "coverage_delta": None,
                "inference_error_rate_delta": None,
                "latency_p95_delta_ms": None,
            },
        }
    base_field = ((baseline_report.get("fields") or {}).get("learned_boundary_shape_only_score") or {})
    base_dist = base_field.get("distribution") or {}
    base_top100 = (((base_field.get("topk_overlap") or {}).get("k100") or {}).get("jaccard"))
    current_top100 = (((learned.get("topk_overlap") or {}).get("k100") or {}).get("jaccard"))
    p95_delta = _optional_delta(current.get("p95"), base_dist.get("p95"))
    median_delta = _optional_delta(current.get("median"), base_dist.get("median"))
    base_std = base_dist.get("std")
    p95_shift_std = float(abs(p95_delta) / base_std) if p95_delta is not None and _num(base_std) > 1e-12 else None
    top_delta = _optional_delta(current_top100, base_top100)
    top_relative = float(abs(top_delta) / abs(float(base_top100))) if top_delta is not None and base_top100 not in {None, 0} else None
    current_coverage = learned.get("coverage") or {}
    base_coverage = base_field.get("coverage") or {}
    current_perf = performance or {}
    base_perf = baseline_report.get("performance") or {}
    return {
        "baseline_available": True,
        "baseline_input": baseline_report.get("input_review_queue"),
        "baseline_window_id": baseline_report.get("window_id"),
        "learned_boundary_shape_only_score": {
            "median_delta": median_delta,
            "p95_delta": p95_delta,
            "p95_shift_std": p95_shift_std,
            "top100_jaccard_delta": top_delta,
            "top100_jaccard_relative_change": top_relative,
            "score_distribution_psi": _distribution_psi(base_dist, current),
            "null_rate_delta": _optional_delta(current_coverage.get("null_rate"), base_coverage.get("null_rate")),
            "coverage_delta": _optional_delta(current_coverage.get("non_null_count"), base_coverage.get("non_null_count")),
            "inference_error_rate_delta": _optional_delta(
                None,
                _inference_error_rate_from_report_like(baseline_report, baseline_report=True),
            ),
            "latency_p95_delta_ms": _optional_delta(
                (((current_perf.get("per_item_learned_inference_latency_ms") or {}).get("p95"))),
                (((base_perf.get("per_item_learned_inference_latency_ms") or {}).get("p95"))),
            ),
        },
    }


def _performance_summary(items: list[dict], baseline_report: dict | None = None, performance_context: dict | None = None) -> dict:
    latencies = []
    scoring_latencies = []
    artifact_latencies = []
    for item in items:
        metadata = item.get("shadow_score_metadata") if isinstance(item.get("shadow_score_metadata"), dict) else {}
        value = metadata.get("learned_inference_latency_ms")
        if value is not None:
            latencies.append(_num(value))
        scoring = metadata.get("shadow_scoring_latency_ms")
        if scoring is not None:
            scoring_latencies.append(_num(scoring))
        artifact = metadata.get("artifact_load_latency_ms")
        if artifact is not None:
            artifact_latencies.append(_num(artifact))
    context = performance_context or {}
    replay_latency = context.get("latency") if isinstance(context.get("latency"), dict) else context
    current = {
        "schema_version": "shadow_latency_v1",
        "queue_generation_latency_ms": None,
        "queue_generation_latency_unavailable_reason": "not measured by shadow replay" if not context.get("queue_generation_latency_ms") else None,
        "shadow_scoring_total_time_ms": replay_latency.get("shadow_scoring_total_time_ms"),
        "monitoring_latency_ms": None,
        "review_packet_export_latency_ms": None,
        "artifact_load_latency_ms": _latency_distribution(artifact_latencies),
        "per_item_learned_inference_latency_ms": _latency_distribution(latencies),
        "per_item_shadow_scoring_latency_ms": _latency_distribution(scoring_latencies),
        "latency_fields_available": bool(latencies or scoring_latencies or artifact_latencies or replay_latency.get("shadow_scoring_total_time_ms") is not None),
        "timing_available": bool(latencies or scoring_latencies or artifact_latencies or replay_latency.get("shadow_scoring_total_time_ms") is not None),
        "timing_source": "time.perf_counter",
        "artifact_loaded_once": replay_latency.get("artifact_loaded_once"),
        "batch_size": replay_latency.get("batch_size") or len(items),
        "rows_processed": len(items),
    }
    if replay_latency.get("artifact_load_latency_ms"):
        current["artifact_load_latency_ms"] = replay_latency["artifact_load_latency_ms"]
    baseline_perf = (baseline_report or {}).get("performance") or {}
    current["queue_generation_latency_regression"] = None
    if current["queue_generation_latency_ms"] is not None and baseline_perf.get("queue_generation_latency_ms"):
        current["queue_generation_latency_regression"] = (
            current["queue_generation_latency_ms"] - baseline_perf["queue_generation_latency_ms"]
        ) / max(1e-9, baseline_perf["queue_generation_latency_ms"])
    return current


def _latency_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _disagreement_volume(current: list[float], shadow: list[float | None]) -> dict:
    valid = [idx for idx, value in enumerate(shadow) if value is not None]
    if not valid:
        return {
            "high_learned_low_current_count": 0,
            "high_current_low_learned_count": 0,
            "low_current_high_learned_count": 0,
            "thresholds": {"high": 0.7, "low": 0.3},
        }
    high = 0.7
    low = 0.3
    high_learned_low_current = [idx for idx in valid if float(shadow[idx]) >= high and current[idx] <= low]
    high_current_low_learned = [idx for idx in valid if current[idx] >= high and float(shadow[idx]) <= low]
    return {
        "high_learned_low_current_count": len(high_learned_low_current),
        "high_current_low_learned_count": len(high_current_low_learned),
        "low_current_high_learned_count": len(high_learned_low_current),
        "thresholds": {"high": high, "low": low},
    }


def _alerts(
    fields: dict,
    metadata: dict,
    total_rows: int,
    distribution_drift: dict | None = None,
    performance: dict | None = None,
    labels_read: bool = False,
) -> dict:
    learned = fields.get("learned_boundary_shape_only_score") or {}
    coverage = learned.get("coverage") or {}
    validation_counts = metadata.get("artifact_validation_status_counts") or {}
    inference_errors = int(metadata.get("inference_error_count") or 0)
    missing_counts = _expand_missing_counts(metadata)
    checks = {
        "learned_score_null_rate": _alert_value(
            coverage.get("null_rate"),
            DEFAULT_ALERT_THRESHOLDS["learned_score_null_rate"],
            ">",
        ),
        "active_artifact_non_valid": {
            "triggered": any(status != "valid" and count for status, count in validation_counts.items()),
            "value": validation_counts,
            "threshold": "all active queue rows valid",
        },
        "inference_error_rate": _alert_value(
            float(inference_errors / total_rows) if total_rows else 0.0,
            DEFAULT_ALERT_THRESHOLDS["inference_error_rate"],
            ">",
        ),
        "missing_safe_feature_count_p95": _alert_value(
            _missing_count_p95(missing_counts),
            DEFAULT_ALERT_THRESHOLDS["missing_safe_feature_count_p95"],
            ">",
        ),
        "shadow_only": {
            "triggered": metadata.get("shadow_only") is not True,
            "value": metadata.get("shadow_only"),
            "threshold": True,
        },
        "affects_default_ranking": {
            "triggered": metadata.get("affects_default_ranking") is not False,
            "value": metadata.get("affects_default_ranking"),
            "threshold": False,
        },
        "queue_generation_latency_regression": _alert_value(
            (performance or {}).get("queue_generation_latency_regression"),
            DEFAULT_ALERT_THRESHOLDS["queue_generation_latency_regression"],
            ">",
        ),
        "shadow_scoring_latency_p95": {
            **_configured_latency_alert(
                (((performance or {}).get("per_item_shadow_scoring_latency_ms") or {}).get("p95")),
                "SEGMENTATION_SHADOW_SCORING_LATENCY_P95_THRESHOLD_MS",
                "shadow_scoring_latency_p95_ms",
            ),
        },
        "artifact_load_latency": {
            **_configured_latency_alert(
                (((performance or {}).get("artifact_load_latency_ms") or {}).get("p95")),
                "SEGMENTATION_ARTIFACT_LOAD_LATENCY_THRESHOLD_MS",
                "artifact_load_latency_ms",
            ),
        },
        "score_distribution_p95_shift_std": _alert_value(
            (((distribution_drift or {}).get("learned_boundary_shape_only_score") or {}).get("p95_shift_std")),
            DEFAULT_ALERT_THRESHOLDS["score_distribution_p95_shift_std"],
            ">",
        ),
        "top100_jaccard_relative_change": _alert_value(
            (((distribution_drift or {}).get("learned_boundary_shape_only_score") or {}).get("top100_jaccard_relative_change")),
            DEFAULT_ALERT_THRESHOLDS["top100_jaccard_relative_change"],
            ">",
        ),
        "production_labels_read": {
            "triggered": bool(labels_read),
            "value": bool(labels_read),
            "threshold": False,
        },
    }
    return {"triggered": any(row.get("triggered") for row in checks.values()), "checks": checks}


def _expand_missing_counts(metadata: dict) -> list[int]:
    dist = metadata.get("missing_feature_count_distribution") or {}
    values = []
    for key, count in dist.items():
        values.extend([int(key)] * int(count))
    return values


def _missing_count_p95(values: list[int]) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), 95)) if values else None


def _alert_value(value: Any, threshold: float, op: str) -> dict:
    number = None if value is None else float(value)
    triggered = False if number is None else number > threshold if op == ">" else number < threshold
    return {"triggered": triggered, "value": number, "threshold": threshold, "operator": op}


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _configured_latency_alert(value: Any, env_name: str, threshold_key: str) -> dict:
    configured = os.getenv(env_name)
    threshold = DEFAULT_ALERT_THRESHOLDS.get(threshold_key)
    if configured is not None:
        try:
            threshold = float(configured)
        except ValueError:
            threshold = None
    if threshold is None:
        return {"triggered": False, "value": None if value is None else float(value), "threshold": "not_configured"}
    return _alert_value(value, float(threshold), ">")


def _distribution_psi(base: dict, current: dict) -> float | None:
    keys = ["p05", "p25", "median", "p75", "p95"]
    if not all(base.get(key) is not None and current.get(key) is not None for key in keys):
        return None
    eps = 1e-6
    total = 0.0
    for key in keys:
        expected = max(eps, float(base[key]))
        actual = max(eps, float(current[key]))
        total += (actual - expected) * np.log(actual / expected)
    return float(total / len(keys))


def _inference_error_rate_from_report_like(report: dict | None, baseline_report: bool = False) -> float | None:
    if not report:
        return None
    safety = report.get("safety_summary") or {}
    rows = int(report.get("n_samples") or 0)
    if not rows:
        return None
    return float(int(safety.get("inference_error_count") or 0) / rows)


def _coverage(items: list[dict], values: list[float | None]) -> dict:
    non_null = sum(value is not None for value in values)
    return {
        "total_rows": len(items),
        "shadow_scores_present_count": sum(isinstance(item.get("shadow_scores"), dict) for item in items),
        "non_null_count": non_null,
        "null_count": len(values) - non_null,
        "null_rate": float((len(values) - non_null) / len(values)) if values else None,
    }


def _shadow_value(item: dict, field: str) -> float | None:
    shadow = item.get("shadow_scores") if isinstance(item.get("shadow_scores"), dict) else {}
    value = shadow.get(field)
    if value is None:
        return None
    return _clip(_num(value))


def _topk_overlap(current: list[float], shadow: list[float | None], k: int) -> dict:
    valid = [idx for idx, value in enumerate(shadow) if value is not None]
    if not valid:
        return {"k": int(k), "intersection_size": 0, "jaccard": None}
    limit = min(int(k), len(current), len(valid))
    current_top = set(sorted(range(len(current)), key=lambda idx: (-current[idx], idx))[:limit])
    shadow_top = set(sorted(valid, key=lambda idx: (-float(shadow[idx]), idx))[:limit])
    union = current_top | shadow_top
    return {
        "k": int(k),
        "current_top_size": len(current_top),
        "shadow_top_size": len(shadow_top),
        "intersection_size": len(current_top & shadow_top),
        "jaccard": float(len(current_top & shadow_top) / len(union)) if union else None,
    }


def _disagreement_examples(items: list[dict], current: list[float], shadow: list[float | None], field: str) -> list[dict]:
    valid = [idx for idx, value in enumerate(shadow) if value is not None]
    if not valid:
        return []
    current_rank = _rank_map(current)
    shadow_filled = [float(value) if value is not None else -1.0 for value in shadow]
    shadow_rank = _rank_map(shadow_filled)
    examples = []
    high_current_low_shadow = sorted(valid, key=lambda idx: (shadow_rank[idx] - current_rank[idx], idx), reverse=True)[:5]
    low_current_high_shadow = sorted(valid, key=lambda idx: (current_rank[idx] - shadow_rank[idx], idx), reverse=True)[:5]
    high_learned_low_current = low_current_high_shadow
    for kind, idxs in [
        ("high_current_low_shadow", high_current_low_shadow),
        ("low_current_high_shadow", low_current_high_shadow),
        ("high_learned_low_current", high_learned_low_current),
    ]:
        for idx in idxs:
            item = items[idx]
            examples.append(
                {
                    "field": field,
                    "kind": kind,
                    "task_id": item.get("task_id"),
                    "sample_id": item.get("sample_id"),
                    "image_id": item.get("image_id"),
                    "annotation_id": item.get("annotation_id"),
                    "current_score": current[idx],
                    "shadow_score": shadow[idx],
                    "current_rank": current_rank[idx] + 1,
                    "shadow_rank": shadow_rank[idx] + 1,
                    "rank_delta": (current_rank[idx] + 1) - (shadow_rank[idx] + 1),
                    "prediction_time_safe_features": safe_prediction_feature_vector(item),
                    "shadow_metadata_status": (
                        item.get("shadow_score_metadata") or {}
                    ).get("artifact_validation_status")
                    if isinstance(item.get("shadow_score_metadata"), dict)
                    else None,
                }
            )
    return examples


def _rank_map(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _rank_delta_summary(current: list[float], shadow: list[float | None]) -> dict:
    valid = [idx for idx, value in enumerate(shadow) if value is not None]
    if not valid:
        return {"count": 0, "mean_abs_delta": None, "median_abs_delta": None, "p95_abs_delta": None, "max_abs_delta": None}
    current_rank = _rank_map(current)
    shadow_rank = _rank_map([float(value) if value is not None else -1.0 for value in shadow])
    deltas = np.asarray([abs(current_rank[idx] - shadow_rank[idx]) for idx in valid], dtype=float)
    return {
        "count": len(valid),
        "mean_abs_delta": float(np.mean(deltas)),
        "median_abs_delta": float(np.percentile(deltas, 50)),
        "p95_abs_delta": float(np.percentile(deltas, 95)),
        "max_abs_delta": float(np.max(deltas)),
    }


def _bucket_major_rate(scores: list[float], labels: list[int], buckets: int = 5) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    out = []
    for bucket in range(buckets):
        start = int(len(order) * bucket / buckets)
        end = int(len(order) * (bucket + 1) / buckets)
        idxs = order[start:end]
        out.append(
            {
                "bucket": bucket + 1,
                "count": len(idxs),
                "score_min": min((scores[idx] for idx in idxs), default=None),
                "score_max": max((scores[idx] for idx in idxs), default=None),
                "major_rate": float(sum(labels[idx] for idx in idxs) / len(idxs)) if idxs else None,
            }
        )
    return out


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _pearson(left: list[float], right: list[float | None]) -> float | None:
    pairs = [(a, float(b)) for a, b in zip(left, right) if b is not None]
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: list[float], right: list[float | None]) -> float | None:
    pairs = [(a, float(b)) for a, b in zip(left, right) if b is not None]
    if len(pairs) < 2:
        return None
    return _pearson(_rank_values([pair[0] for pair in pairs]), _rank_values([pair[1] for pair in pairs]))


def _rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            hits += 1
            total += hits / rank
    return float(total / positives)


def _precision_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    if not labels:
        return None
    limit = min(int(k), len(labels))
    if limit <= 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    return float(sum(labels[idx] for idx in order[:limit]) / limit)


def _lift_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    precision = _precision_at_k(labels, scores, k)
    base = float(sum(labels) / len(labels)) if labels else None
    return float(precision / base) if precision is not None and base else None


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return float(wins / (len(positives) * len(negatives)))


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare shadow scores against current review priority.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-id")
    parser.add_argument("--baseline", help="Previous shadow_monitoring.json used for drift checks.")
    parser.add_argument("--production-mode", action="store_true", help="Do not read labels or evaluation-only fields.")
    parser.add_argument("--no-labels", action="store_true", help="Disable offline label metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_shadow_scores(
        args.review_queue,
        args.output_dir,
        window_id=args.window_id,
        baseline=args.baseline,
        production_mode=args.production_mode,
        no_labels=args.no_labels,
    )
    print(f"[OK] shadow monitoring samples={result['n_samples']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
