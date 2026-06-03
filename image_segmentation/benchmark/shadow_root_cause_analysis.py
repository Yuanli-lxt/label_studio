from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.prediction_feature_scoring import (
    PREDICTION_FEATURE_NAMES,
    prediction_features,
    safe_prediction_feature_vector,
)


DRIFT_FEATURES = [
    *PREDICTION_FEATURE_NAMES,
    "boundary_shape_calibrated_score",
    "boundary_shape_rank_score",
]
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
    "boundary_metadata",
    "evaluation_only",
    "delta",
}


def analyze_shadow_drift(
    multi_window_root: str,
    baseline_window: str,
    windows: list[str],
    output_dir: str,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_window(root, baseline_window)
    targets = {window: _load_window(root, window) for window in windows}
    feature_diffs = {
        window: _feature_distribution_diff(baseline["items"], payload["items"])
        for window, payload in targets.items()
    }
    report = {
        "multi_window_root": str(root),
        "baseline_window": baseline_window,
        "windows": windows,
        "baseline": _window_comparison_summary(baseline["items"]),
        "targets": {window: _window_comparison_summary(payload["items"]) for window, payload in targets.items()},
        "queue_mix_summary": {
            "baseline": _queue_mix_summary(baseline["items"]),
            "targets": {window: _queue_mix_summary(payload["items"]) for window, payload in targets.items()},
        },
        "feature_distribution_diffs": feature_diffs,
        "likely_explanations": {
            window: _drift_explanation(feature_diffs[window], baseline["items"], payload["items"])
            for window, payload in targets.items()
        },
        "safe_fields_only": True,
    }
    _write_json(output / "drift_root_cause.json", report)
    _write_json(output / "window_feature_distribution_diffs.json", feature_diffs)
    _write_json(output / "queue_mix_summary.json", report["queue_mix_summary"])
    outliers = []
    for window, payload in targets.items():
        outliers.extend(_safe_outlier_rows(payload["items"], window, limit=50))
    _write_jsonl(output / "drift_outlier_samples.jsonl", outliers)
    (output / "drift_root_cause.md").write_text(_render_drift_markdown(report), encoding="utf-8")
    return report


def analyze_schema_aware_shadow_drift(
    multi_window_root: str,
    classification_path: str,
    output_dir: str,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classification = _read_json(Path(classification_path))
    windows = _classification_windows(classification)
    compatibility = classification.get("pairwise_compatibility_matrix") or {}
    comparable = []
    non_comparable = []
    for key, pair in sorted(compatibility.items()):
        left_id, right_id = _split_pair_key(key)
        if left_id not in windows or right_id not in windows:
            continue
        if pair.get("compatible_for_drift_comparison") is True:
            row = _schema_aware_drift_pair(root, left_id, right_id, pair, windows)
            comparable.append(row)
        else:
            non_comparable.append(_schema_non_comparable_row(key, pair))
    report = {
        "multi_window_root": str(root),
        "classification": str(classification_path),
        "schema_aware": True,
        "safe_fields_only": True,
        "comparable_pairs": comparable,
        "non_comparable_pairs": non_comparable,
        "alert_severity_summary": _severity_summary(comparable + non_comparable),
        "stable_conclusion_available": bool(comparable),
        "conclusion": _schema_aware_conclusion(comparable, non_comparable, "model drift"),
    }
    _write_json(output / "schema_aware_drift.json", report)
    _write_jsonl(output / "comparable_window_pairs.jsonl", comparable)
    _write_jsonl(output / "non_comparable_window_pairs.jsonl", non_comparable)
    (output / "schema_aware_drift.md").write_text(_render_schema_aware_drift_markdown(report), encoding="utf-8")
    return report


def analyze_shadow_missing_features(
    multi_window_root: str,
    windows: list[str],
    output_dir: str,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_window = {}
    samples = []
    for window in windows:
        items = _load_window(root, window)["items"]
        summary = _missing_summary(items)
        per_window[window] = summary
        samples.extend(_safe_missing_rows(items, window, limit=50))
    report = {
        "multi_window_root": str(root),
        "windows": windows,
        "per_window": per_window,
        "safe_fields_only": True,
        "overall_assessment": _missing_overall_assessment(per_window),
    }
    _write_json(output / "missing_feature_analysis.json", report)
    _write_jsonl(output / "missing_feature_samples.jsonl", samples)
    (output / "missing_feature_analysis.md").write_text(_render_missing_markdown(report), encoding="utf-8")
    return report


def analyze_schema_aware_shadow_missing_features(
    multi_window_root: str,
    classification_path: str,
    output_dir: str,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classification = _read_json(Path(classification_path))
    windows = _classification_windows(classification)
    per_window = []
    for window_id, window in sorted(windows.items()):
        missing = window.get("missing_feature_names") or []
        schema = window.get("queue_schema_version")
        completeness = window.get("feature_completeness_bucket")
        if schema == "full_prediction_features" and missing:
            severity = "stability_alert"
            actionable = True
            interpretation = "full prediction-feature window has missing safe features"
        elif schema == "old_schema_fallback" and missing:
            severity = "compatibility_warning"
            actionable = False
            interpretation = "old schema fallback is expected to miss raw prediction-time features; score quality may be lower"
        elif completeness == "mixed":
            severity = "compatibility_warning"
            actionable = False
            interpretation = "mixed feature completeness makes missing-feature alerts schema dependent"
        else:
            severity = "info"
            actionable = False
            interpretation = "no schema-aware missing-feature stability alert"
        per_window.append(
            {
                "window_id": window_id,
                "queue_schema_version": schema,
                "feature_completeness_bucket": completeness,
                "queue_type": window.get("queue_type"),
                "rows": window.get("rows"),
                "missing_feature_names": missing,
                "safe_feature_presence_rate": window.get("safe_feature_presence_rate"),
                "neutral_fallback_rate": window.get("neutral_fallback_rate"),
                "learned_score_coverage": window.get("learned_score_coverage"),
                "alert_severity": severity,
                "actionable": actionable,
                "interpretation": interpretation,
            }
        )
    buckets = _missing_schema_buckets(per_window)
    report = {
        "multi_window_root": str(root),
        "classification": str(classification_path),
        "schema_aware": True,
        "safe_fields_only": True,
        "per_window": {row["window_id"]: row for row in per_window},
        "schema_buckets": buckets,
        "alert_severity_summary": _severity_summary(per_window),
        "overall_assessment": {
            "old_schema_fallback_downgraded_to_compatibility_warning": any(
                row.get("queue_schema_version") == "old_schema_fallback" and row.get("alert_severity") == "compatibility_warning"
                for row in per_window
            ),
            "full_feature_missing_stability_alert": any(row.get("alert_severity") == "stability_alert" for row in per_window),
        },
    }
    _write_json(output / "schema_aware_missing_features.json", report)
    _write_jsonl(output / "missing_feature_schema_buckets.jsonl", _flatten_bucket_rows(buckets))
    (output / "schema_aware_missing_features.md").write_text(_render_schema_aware_missing_markdown(report), encoding="utf-8")
    return report


def analyze_shadow_topk_jaccard(
    multi_window_root: str,
    baseline_window: str,
    windows: list[str],
    output_dir: str,
    k: int = 100,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_items = _load_window(root, baseline_window)["items"]
    baseline = _topk_sets(baseline_items, k)
    per_window = {}
    changes = []
    for window in windows:
        items = _load_window(root, window)["items"]
        current = _topk_sets(items, k)
        shared_ids = sorted(set(baseline["all_ids"]) & set(current["all_ids"]))
        shared = _shared_sample_jaccard(baseline_items, items, shared_ids, k)
        learned_only = current["learned_ids"] - current["current_ids"]
        current_only = current["current_ids"] - current["learned_ids"]
        row = {
            "window": window,
            "k": int(k),
            "baseline_top_current_size": len(baseline["current_ids"]),
            "baseline_top_learned_size": len(baseline["learned_ids"]),
            "target_top_current_size": len(current["current_ids"]),
            "target_top_learned_size": len(current["learned_ids"]),
            "overlap_count": len(current["current_ids"] & current["learned_ids"]),
            "jaccard": _jaccard(current["current_ids"], current["learned_ids"]),
            "baseline_jaccard": _jaccard(baseline["current_ids"], baseline["learned_ids"]),
            "relative_change_vs_baseline": _relative_change(
                _jaccard(current["current_ids"], current["learned_ids"]),
                _jaccard(baseline["current_ids"], baseline["learned_ids"]),
            ),
            "same_sample_overlap_size": len(shared_ids),
            "normalized_jaccard_on_shared_samples": shared,
            "likely_cause": _topk_likely_cause(baseline_items, items, shared_ids),
        }
        per_window[window] = row
        for sample_id in sorted(list(learned_only))[:50]:
            changes.append(_membership_change_row(items, sample_id, window, "learned_only_top"))
        for sample_id in sorted(list(current_only))[:50]:
            changes.append(_membership_change_row(items, sample_id, window, "current_only_top"))
    report = {
        "multi_window_root": str(root),
        "baseline_window": baseline_window,
        "windows": windows,
        "per_window": per_window,
        "safe_fields_only": True,
    }
    _write_json(output / "topk_jaccard_analysis.json", report)
    _write_jsonl(output / "topk_membership_changes.jsonl", [row for row in changes if row])
    (output / "topk_jaccard_analysis.md").write_text(_render_topk_markdown(report), encoding="utf-8")
    return report


def analyze_schema_aware_shadow_topk_jaccard(
    multi_window_root: str,
    classification_path: str,
    output_dir: str,
    k: int = 100,
) -> dict:
    root = Path(multi_window_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classification = _read_json(Path(classification_path))
    windows = _classification_windows(classification)
    compatibility = classification.get("pairwise_compatibility_matrix") or {}
    comparable = []
    non_comparable = []
    for key, pair in sorted(compatibility.items()):
        left_id, right_id = _split_pair_key(key)
        if left_id not in windows or right_id not in windows:
            continue
        if pair.get("compatible_for_drift_comparison") is True and (pair.get("same_sample_overlap") or 0) > 0:
            comparable.append(_schema_aware_topk_pair(root, left_id, right_id, pair, windows, k))
        else:
            row = _schema_non_comparable_row(key, pair)
            row["not_directly_comparable"] = True
            row["likely_cause"] = "sample mix / queue type shift" if (pair.get("same_sample_overlap") or 0) == 0 else row.get("reason")
            non_comparable.append(row)
    report = {
        "multi_window_root": str(root),
        "classification": str(classification_path),
        "schema_aware": True,
        "safe_fields_only": True,
        "k": int(k),
        "comparable_pairs": comparable,
        "non_comparable_pairs": non_comparable,
        "alert_severity_summary": _severity_summary(comparable + non_comparable),
        "stable_conclusion_available": bool(comparable),
        "conclusion": _schema_aware_conclusion(comparable, non_comparable, "top-k"),
    }
    _write_json(output / "schema_aware_topk_jaccard.json", report)
    _write_jsonl(output / "topk_comparable_pairs.jsonl", comparable)
    _write_jsonl(output / "topk_non_comparable_pairs.jsonl", non_comparable)
    (output / "schema_aware_topk_jaccard.md").write_text(_render_schema_aware_topk_markdown(report), encoding="utf-8")
    return report


def build_human_review_assignment(task_packet_dir: str, output_dir: str) -> dict:
    source = Path(task_packet_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.glob("review_tasks_*.jsonl")):
        for row in _read_jsonl(path):
            rows.append(_assignment_row(row))
    jsonl_path = output / "review_assignment.jsonl"
    csv_path = output / "review_assignment_template.csv"
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    guidelines = output / "review_guidelines.md"
    guidelines.write_text(_review_guidelines(), encoding="utf-8")
    summary = {
        "source_task_packet_dir": str(source),
        "output_dir": str(output),
        "rows": len(rows),
        "outputs": {
            "review_assignment": str(jsonl_path),
            "review_assignment_template_csv": str(csv_path),
            "review_guidelines": str(guidelines),
        },
        "safe_fields_only": True,
    }
    _write_json(output / "review_assignment_summary.json", summary)
    return summary


def write_drift_feedback_decision_report(
    multi_window_root: str,
    drift_dir: str,
    missing_dir: str,
    topk_dir: str,
    assignment_dir: str,
    feedback_dir: str,
    output_path: str,
) -> dict:
    root = Path(multi_window_root)
    broader = _read_json(root / "broader_shadow_multi_window_summary.json")
    drift = _read_json(Path(drift_dir) / "drift_root_cause.json")
    missing = _read_json(Path(missing_dir) / "missing_feature_analysis.json")
    topk = _read_json(Path(topk_dir) / "topk_jaccard_analysis.json")
    feedback = _read_json(Path(feedback_dir) / "human_feedback_summary.json")
    hard = ((broader.get("alerts") or {}).get("hard_safety_alert") is True)
    missing_explained = (missing.get("overall_assessment") or {}).get("likely_backward_compatible_or_schema_mix") is True
    drift_explained = all(
        (row.get("likely_actionable") is False)
        for row in (drift.get("likely_explanations") or {}).values()
    )
    topk_understood = all(
        (row.get("likely_cause") or {}).get("sample_mix_changed") is True
        or (row.get("likely_cause") or {}).get("replay_like_or_duplicate_window") is True
        for row in (topk.get("per_window") or {}).values()
    )
    total_reviewed = int(feedback.get("total_reviewed") or 0)
    if hard:
        action = "pause_or_rollback"
        reason = "hard safety alert present"
    elif not (missing_explained and drift_explained and topk_understood):
        action = "continue_hold"
        reason = "technical alerts still need explanation or mitigation"
    elif total_reviewed == 0:
        action = "resume_same_shadow_traffic"
        reason = "technical alerts are explainable, but feedback is unavailable; keep promotion on hold"
    else:
        action = "resume_same_shadow_traffic"
        reason = "technical alerts are explainable and initial feedback is available"
    payload = {
        "decision": action,
        "reason": reason,
        "hard_safety_alert": hard,
        "missing_features_explained": missing_explained,
        "drift_explained": drift_explained,
        "topk_alerts_understood": topk_understood,
        "total_reviewed": total_reviewed,
        "ready_for_default_sorting": False,
        "ready_for_weight_promotion": False,
    }
    lines = [
        "# Production Shadow Drift / Feedback Decision Report",
        "",
        "## Scope",
        "Offline drift, missing-feature, top-k agreement, and human-feedback analysis. Default Layer 5 weights and sorting remain unchanged.",
        "",
        "## Current Rollout Status",
        f"- broader_root: {root}",
        f"- aggregate_alert: {((broader.get('alerts') or {}).get('aggregate_alert_status'))}",
        f"- hard_safety_alert: {hard}",
        "",
        "## Drift Outlier Summary",
        f"- {(broader.get('distribution_stability') or {}).get('p95_drift_outliers')}",
        "",
        "## Missing Feature Summary",
        f"- {missing.get('overall_assessment')}",
        "",
        "## Top-k Jaccard Alert Analysis",
        f"- {topk.get('per_window')}",
        "",
        "## Queue Mix Explanation",
        f"- {(drift.get('queue_mix_summary') or {})}",
        "",
        "## Human Review Assignment Path",
        f"- {assignment_dir}",
        "",
        "## Human Feedback Summary",
        f"- total_reviewed: {total_reviewed}",
        f"- recommendation: {feedback.get('recommendation')}",
        "",
        "## Safety / Leakage",
        f"- safety_stability: {broader.get('safety_stability')}",
        f"- production labels read remains false: {not ((broader.get('safety_stability') or {}).get('production_labels_read_any'))}",
        "",
        "## Decision",
        f"- action: {payload['decision']}",
        f"- reason: {payload['reason']}",
        "",
        "## Promotion Status",
        "- not ready for default sorting",
        "- not ready for weight promotion",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _classification_windows(classification: dict) -> dict[str, dict]:
    return {
        str(row.get("window_id")): row
        for row in classification.get("windows", [])
        if row.get("window_id") is not None
    }


def _split_pair_key(key: str) -> tuple[str, str]:
    left, _, right = key.partition("__")
    return left, right


def _schema_aware_drift_pair(root: Path, left_id: str, right_id: str, compatibility: dict, windows: dict[str, dict]) -> dict:
    left_items = _load_window(root, left_id)["items"]
    right_items = _load_window(root, right_id)["items"]
    left_learned = [_shadow_value(row, "learned_boundary_shape_only_score") or 0.0 for row in left_items]
    right_learned = [_shadow_value(row, "learned_boundary_shape_only_score") or 0.0 for row in right_items]
    left_current = [_num(row.get("priority_score")) for row in left_items]
    right_current = [_num(row.get("priority_score")) for row in right_items]
    learned_left_dist = _distribution(left_learned)
    learned_right_dist = _distribution(right_learned)
    current_left_dist = _distribution(left_current)
    current_right_dist = _distribution(right_current)
    learned_diff = _distribution_diff(learned_left_dist, learned_right_dist)
    current_diff = _distribution_diff(current_left_dist, current_right_dist)
    score_distribution_psi = _distribution_psi(learned_left_dist, learned_right_dist)
    triggered = bool(
        (learned_diff.get("p95_delta") is not None and abs(learned_diff["p95_delta"]) > 0.25)
        or (score_distribution_psi is not None and score_distribution_psi > 0.20)
    )
    return {
        "pair": f"{left_id}__{right_id}",
        "left_window": left_id,
        "right_window": right_id,
        "not_comparable": False,
        "compatible_for_drift_comparison": True,
        "same_sample_overlap": compatibility.get("same_sample_overlap"),
        "sample_set_changed": compatibility.get("sample_set_changed"),
        "left_schema": _window_schema_summary(windows.get(left_id) or {}),
        "right_schema": _window_schema_summary(windows.get(right_id) or {}),
        "learned_score_distribution_diff": learned_diff,
        "current_priority_distribution_diff": current_diff,
        "p95_drift": learned_diff.get("p95_delta"),
        "median_drift": learned_diff.get("median_delta"),
        "score_distribution_psi": score_distribution_psi,
        "alert_status": "triggered" if triggered else "clear",
        "alert_severity": "stability_alert" if triggered else "info",
        "actionable": triggered,
    }


def _schema_aware_topk_pair(root: Path, left_id: str, right_id: str, compatibility: dict, windows: dict[str, dict], k: int) -> dict:
    left_items = _load_window(root, left_id)["items"]
    right_items = _load_window(root, right_id)["items"]
    left = _topk_sets(left_items, k)
    right = _topk_sets(right_items, k)
    shared_ids = sorted(set(left["all_ids"]) & set(right["all_ids"]))
    shared = _shared_sample_jaccard(left_items, right_items, shared_ids, k)
    raw_learned = _jaccard(left["learned_ids"], right["learned_ids"])
    raw_current = _jaccard(left["current_ids"], right["current_ids"])
    actionable = bool(raw_learned is not None and raw_learned < 0.50 and shared.get("available"))
    return {
        "pair": f"{left_id}__{right_id}",
        "left_window": left_id,
        "right_window": right_id,
        "not_directly_comparable": False,
        "compatible_for_drift_comparison": True,
        "k": int(k),
        "raw_learned_topk_jaccard": raw_learned,
        "raw_current_topk_jaccard": raw_current,
        "shared_sample_normalized_jaccard": shared,
        "same_sample_overlap": compatibility.get("same_sample_overlap"),
        "sample_set_changed": compatibility.get("sample_set_changed"),
        "left_schema": _window_schema_summary(windows.get(left_id) or {}),
        "right_schema": _window_schema_summary(windows.get(right_id) or {}),
        "alert_status": "triggered" if actionable else "clear",
        "alert_severity": "stability_alert" if actionable else "info",
        "actionable": actionable,
    }


def _schema_non_comparable_row(key: str, compatibility: dict) -> dict:
    left_id, right_id = _split_pair_key(key)
    return {
        "pair": key,
        "left_window": left_id,
        "right_window": right_id,
        "not_comparable": True,
        "compatible_for_drift_comparison": False,
        "reason": compatibility.get("reason"),
        "skipped_reason": compatibility.get("reason"),
        "same_sample_overlap": compatibility.get("same_sample_overlap"),
        "same_sample_overlap_rate_min_window": compatibility.get("same_sample_overlap_rate_min_window"),
        "sample_set_changed": compatibility.get("sample_set_changed"),
        "direct_topk_comparison_limited": compatibility.get("direct_topk_comparison_limited"),
        "alert_status": "skipped",
        "alert_severity": "compatibility_warning",
        "actionable": False,
    }


def _window_schema_summary(window: dict) -> dict:
    return {
        "queue_schema_version": window.get("queue_schema_version"),
        "feature_completeness_bucket": window.get("feature_completeness_bucket"),
        "queue_type": window.get("queue_type"),
        "rows": window.get("rows"),
        "safe_feature_presence_rate": window.get("safe_feature_presence_rate"),
    }


def _distribution_diff(left: dict, right: dict) -> dict:
    return {
        "left": left,
        "right": right,
        "median_delta": _optional_delta(right.get("median"), left.get("median")),
        "p95_delta": _optional_delta(right.get("p95"), left.get("p95")),
        "mean_delta": _optional_delta(right.get("mean"), left.get("mean")),
    }


def _distribution_psi(left: dict, right: dict) -> float | None:
    keys = ["p05", "p25", "median", "p75", "p95"]
    if not all(left.get(key) is not None and right.get(key) is not None for key in keys):
        return None
    eps = 1e-6
    total = 0.0
    for key in keys:
        expected = max(eps, float(left[key]))
        actual = max(eps, float(right[key]))
        total += (actual - expected) * math.log(actual / expected)
    return float(total / len(keys))


def _severity_summary(rows: list[dict]) -> dict:
    return dict(Counter(str(row.get("alert_severity") or "info") for row in rows))


def _schema_aware_conclusion(comparable: list[dict], non_comparable: list[dict], subject: str) -> str:
    if not comparable and non_comparable:
        return f"No stable conclusion about {subject} can be drawn from mixed-schema or non-comparable windows."
    if any(row.get("alert_severity") == "stability_alert" for row in comparable):
        return f"Comparable windows contain {subject} stability alerts; hold expansion."
    if comparable:
        return f"Comparable windows do not show actionable {subject} stability alerts."
    return f"No stable conclusion about {subject} can be drawn."


def _missing_schema_buckets(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for key in ("queue_schema_version", "feature_completeness_bucket", "queue_type"):
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(str(row.get(key) or "unknown"), []).append(row)
        out[key] = {
            name: {
                "windows": len(bucket_rows),
                "rows": sum(int(row.get("rows") or 0) for row in bucket_rows),
                "missing_feature_names": sorted({name for row in bucket_rows for name in (row.get("missing_feature_names") or [])}),
                "neutral_fallback_rate_mean": _mean([row.get("neutral_fallback_rate") for row in bucket_rows]),
                "learned_score_coverage_mean": _mean([row.get("learned_score_coverage") for row in bucket_rows]),
                "alert_severity_counts": _severity_summary(bucket_rows),
            }
            for name, bucket_rows in sorted(buckets.items())
        }
    return out


def _flatten_bucket_rows(buckets: dict) -> list[dict]:
    rows = []
    for dimension, values in buckets.items():
        for bucket, payload in values.items():
            row = {"dimension": dimension, "bucket": bucket}
            row.update(payload)
            rows.append(row)
    return rows


def _mean(values: list[Any]) -> float | None:
    numbers = [_num_or_none(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _load_window(root: Path, window: str) -> dict:
    queue = root / window / "live_review_queue.shadow_scored.jsonl"
    return {"queue": str(queue), "items": _read_jsonl(queue)}


def _window_comparison_summary(items: list[dict]) -> dict:
    current = [_num(row.get("priority_score")) for row in items]
    learned = [_shadow_value(row, "learned_boundary_shape_only_score") for row in items]
    calibrated = [_safe_feature(row, "boundary_shape_calibrated_score") for row in items]
    rank_score = [_shadow_value(row, "boundary_shape_rank_score") for row in items]
    rank_delta = _rank_delta(current, learned)
    return {
        "rows": len(items),
        "learned_score_distribution": _distribution([value for value in learned if value is not None]),
        "current_priority_distribution": _distribution(current),
        "boundary_shape_calibrated_score_distribution": _distribution(calibrated),
        "boundary_shape_rank_score_distribution": _distribution([value for value in rank_score if value is not None]),
        "rank_delta_distribution": _distribution(rank_delta),
        "topk": {
            "k20": _topk_overlap(current, learned, 20),
            "k50": _topk_overlap(current, learned, 50),
            "k100": _topk_overlap(current, learned, 100),
        },
    }


def _queue_mix_summary(items: list[dict]) -> dict:
    return {
        "dataset": _counter(items, "dataset"),
        "benchmark_name": _counter(items, "benchmark_name"),
        "source": _counter(items, "source"),
        "prediction_source": _counter(items, "prediction_source"),
        "model_name": _counter(items, "model_name"),
        "project_id": _counter(items, "project_id"),
        "queue_type": _counter(items, "queue_type"),
        "image_size_bucket": dict(Counter(_image_size_bucket(row) for row in items)),
        "predicted_mask_area_bucket": dict(Counter(_area_bucket(_safe_feature(row, "pred_area_ratio")) for row in items)),
        "missing_feature_count_bucket": dict(Counter(str(_missing_count(row)) for row in items)),
        "artifact_validation_status": dict(Counter(_metadata(row).get("artifact_validation_status") or "missing" for row in items)),
        "shadow_metadata_status": dict(Counter(_metadata(row).get("artifact_validation_status") or "missing" for row in items)),
    }


def _feature_distribution_diff(baseline: list[dict], target: list[dict]) -> dict:
    out = {}
    for feature in DRIFT_FEATURES:
        base_values, base_missing = _feature_values_and_missing(baseline, feature)
        target_values, target_missing = _feature_values_and_missing(target, feature)
        base_dist = _distribution(base_values)
        target_dist = _distribution(target_values)
        median_shift = _optional_delta(target_dist.get("median"), base_dist.get("median"))
        p95_shift = _optional_delta(target_dist.get("p95"), base_dist.get("p95"))
        rel = None
        if p95_shift is not None and base_dist.get("p95") not in {None, 0}:
            rel = float(p95_shift / abs(float(base_dist["p95"])))
        missing_rate_delta = _optional_delta(
            target_missing / max(1, len(target)),
            base_missing / max(1, len(baseline)),
        )
        out[feature] = {
            "baseline_median": base_dist.get("median"),
            "baseline_p95": base_dist.get("p95"),
            "target_median": target_dist.get("median"),
            "target_p95": target_dist.get("p95"),
            "median_shift": median_shift,
            "p95_shift": p95_shift,
            "relative_p95_shift": rel,
            "baseline_missing_rate": base_missing / max(1, len(baseline)),
            "target_missing_rate": target_missing / max(1, len(target)),
            "missing_rate_difference": missing_rate_delta,
            "likely_contributes_to_learned_score_drift": bool(
                (rel is not None and abs(rel) > 0.25) or (missing_rate_delta is not None and abs(missing_rate_delta) > 0.05)
            ),
        }
    return out


def _drift_explanation(feature_diffs: dict, baseline: list[dict], target: list[dict]) -> dict:
    contributors = [
        name for name, row in feature_diffs.items()
        if row.get("likely_contributes_to_learned_score_drift")
    ]
    baseline_ids = {_sample_key(row) for row in baseline}
    target_ids = {_sample_key(row) for row in target}
    shared = len(baseline_ids & target_ids)
    sample_mix_changed = shared < min(len(baseline_ids), len(target_ids)) * 0.5 if baseline_ids and target_ids else True
    return {
        "likely_actionable": bool(contributors) and not sample_mix_changed,
        "sample_mix_changed": sample_mix_changed,
        "same_sample_overlap_size": shared,
        "contributing_features": contributors[:10],
        "interpretation": "sample-mix/queue-type shift likely explains cross-window drift" if sample_mix_changed else "feature distribution shifts may be actionable",
    }


def _missing_summary(items: list[dict]) -> dict:
    name_counts = Counter()
    combo_counts = Counter()
    missing_rows = []
    for row in items:
        missing = _missing_names(row)
        if missing:
            missing_rows.append(row)
        name_counts.update(missing)
        combo_counts.update(["|".join(sorted(missing)) if missing else "none"])
    learned_null = sum(_shadow_value(row, "learned_boundary_shape_only_score") is None for row in items)
    high_learned_missing = [
        row for row in missing_rows
        if (_shadow_value(row, "learned_boundary_shape_only_score") or 0.0) >= 0.7
    ]
    return {
        "rows": len(items),
        "rows_with_missing_features": len(missing_rows),
        "learned_score_null_count": learned_null,
        "neutral_fallback_used": bool(missing_rows and learned_null == 0),
        "missing_feature_names_frequency": dict(name_counts.most_common()),
        "missing_feature_combinations": dict(combo_counts.most_common(20)),
        "missing_rate_by_queue_source_model": _missing_rate_by(items, ["source", "prediction_source", "model_name", "dataset"]),
        "missing_rate_by_image_size_bucket": _missing_rate_by_bucket(items, _image_size_bucket),
        "missing_rate_by_prediction_source": _missing_rate_by(items, ["prediction_source"]),
        "high_learned_missing_count": len(high_learned_missing),
        "correlates_with_high_learned_score": bool(high_learned_missing),
        "likely_old_queue_schema_or_replay": bool(missing_rows and learned_null == 0),
    }


def _missing_overall_assessment(per_window: dict) -> dict:
    any_missing = any(row.get("rows_with_missing_features", 0) for row in per_window.values())
    any_null = any(row.get("learned_score_null_count", 0) for row in per_window.values())
    high = any(row.get("correlates_with_high_learned_score") for row in per_window.values())
    return {
        "missing_features_present": any_missing,
        "learned_score_became_null": any_null,
        "likely_backward_compatible_or_schema_mix": bool(any_missing and not any_null),
        "needs_extractor_fix_before_expansion": bool(any_null or high),
        "recommendation": "hold rollout and inspect extractor/schema mapping" if any_null or high else "compatible missing fields used neutral fallback; inspect before expansion",
    }


def _safe_outlier_rows(items: list[dict], window: str, limit: int) -> list[dict]:
    ranked = sorted(
        items,
        key=lambda row: abs(_rank_delta_for_row(items, row)),
        reverse=True,
    )
    return [_safe_analysis_row(row, window) for row in ranked[:limit]]


def _safe_missing_rows(items: list[dict], window: str, limit: int) -> list[dict]:
    rows = [row for row in items if _missing_names(row)]
    return [_safe_analysis_row(row, window) for row in rows[:limit]]


def _safe_analysis_row(row: dict, window: str) -> dict:
    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "prediction_id": row.get("prediction_id"),
        "window_id": window,
        "current_priority_score": row.get("priority_score"),
        "learned_shadow_score": _shadow_value(row, "learned_boundary_shape_only_score"),
        "rank_current": row.get("rank"),
        "rank_learned": None,
        "rank_delta": None,
        "prediction_time_safe_features": safe_prediction_feature_vector(row),
        "missing_safe_features": _missing_names(row),
        "artifact_status": _metadata(row).get("artifact_validation_status"),
    }


def _assignment_row(row: dict) -> dict:
    features = row.get("prediction_time_safe_features") if isinstance(row.get("prediction_time_safe_features"), dict) else {}
    selected = {key: features.get(key) for key in DRIFT_FEATURES if key in features}
    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "group": row.get("group"),
        "current_priority_score": row.get("current_priority_score"),
        "learned_shadow_score": row.get("learned_shadow_score"),
        "rank_current": row.get("rank_current"),
        "rank_learned": row.get("rank_learned"),
        "rank_delta": row.get("rank_delta"),
        "selected_safe_prediction_features": selected,
        "window_ids": row.get("window_ids") or [],
        "human_review_outcome": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": None,
    }


def read_feedback_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    return _read_jsonl(path)


def _topk_sets(items: list[dict], k: int) -> dict:
    current_pairs = sorted(enumerate(items), key=lambda pair: (-_num(pair[1].get("priority_score")), pair[0]))[:k]
    learned_pairs = sorted(enumerate(items), key=lambda pair: (-(_shadow_value(pair[1], "learned_boundary_shape_only_score") or -1.0), pair[0]))[:k]
    return {
        "current_ids": {_sample_key(row) for _, row in current_pairs},
        "learned_ids": {_sample_key(row) for _, row in learned_pairs},
        "all_ids": {_sample_key(row) for row in items},
    }


def _shared_sample_jaccard(baseline: list[dict], target: list[dict], shared_ids: list[str], k: int) -> dict:
    if not shared_ids:
        return {"available": False, "shared_sample_count": 0, "current_jaccard": None, "learned_jaccard": None}
    shared = set(shared_ids)
    base = [row for row in baseline if _sample_key(row) in shared]
    tgt = [row for row in target if _sample_key(row) in shared]
    b = _topk_sets(base, min(k, len(shared)))
    t = _topk_sets(tgt, min(k, len(shared)))
    return {
        "available": True,
        "shared_sample_count": len(shared),
        "current_jaccard": _jaccard(b["current_ids"], t["current_ids"]),
        "learned_jaccard": _jaccard(b["learned_ids"], t["learned_ids"]),
    }


def _topk_likely_cause(baseline: list[dict], target: list[dict], shared_ids: list[str]) -> dict:
    sample_mix_changed = len(shared_ids) < min(len(baseline), len(target)) * 0.5 if baseline and target else True
    replay_like = len(shared_ids) == min(len(baseline), len(target)) and len(baseline) == len(target)
    return {
        "sample_mix_changed": sample_mix_changed,
        "replay_like_or_duplicate_window": replay_like,
        "current_priority_distribution_changed": _distribution_shift(
            [_num(row.get("priority_score")) for row in baseline],
            [_num(row.get("priority_score")) for row in target],
        ),
        "learned_score_distribution_changed": _distribution_shift(
            [_shadow_value(row, "learned_boundary_shape_only_score") or 0.0 for row in baseline],
            [_shadow_value(row, "learned_boundary_shape_only_score") or 0.0 for row in target],
        ),
        "note": "Direct cross-window top-k Jaccard is weak evidence when sample sets differ substantially.",
    }


def _membership_change_row(items: list[dict], sample_id: str, window: str, kind: str) -> dict | None:
    for row in items:
        if _sample_key(row) == sample_id:
            out = _safe_analysis_row(row, window)
            out["kind"] = kind
            return out
    return None


def _distribution(values: list[Any]) -> dict:
    numbers = [_num_or_none(value) for value in values]
    arr = np.asarray([value for value in numbers if value is not None], dtype=float)
    if arr.size == 0:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(arr.size),
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


def _topk_overlap(current: list[float], learned: list[float | None], k: int) -> dict:
    valid = [idx for idx, value in enumerate(learned) if value is not None]
    limit = min(k, len(current), len(valid))
    if limit <= 0:
        return {"k": k, "intersection_size": 0, "jaccard": None}
    current_top = set(sorted(range(len(current)), key=lambda idx: (-current[idx], idx))[:limit])
    learned_top = set(sorted(valid, key=lambda idx: (-float(learned[idx]), idx))[:limit])
    return {"k": k, "intersection_size": len(current_top & learned_top), "jaccard": _jaccard(current_top, learned_top)}


def _rank_delta(current: list[float], learned: list[float | None]) -> list[float]:
    if not current:
        return []
    cr = _rank_map(current)
    lr = _rank_map([value if value is not None else -1.0 for value in learned])
    return [abs(cr[idx] - lr[idx]) for idx in range(len(current)) if learned[idx] is not None]


def _rank_delta_for_row(items: list[dict], row: dict) -> float:
    current = [_num(item.get("priority_score")) for item in items]
    learned = [_shadow_value(item, "learned_boundary_shape_only_score") for item in items]
    cr = _rank_map(current)
    lr = _rank_map([value if value is not None else -1.0 for value in learned])
    idx = items.index(row)
    return float(cr[idx] - lr[idx])


def _rank_map(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda idx: (-_num(values[idx]), idx))
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _feature_values_and_missing(items: list[dict], feature: str) -> tuple[list[float], int]:
    values = []
    missing = 0
    for row in items:
        value = _safe_feature(row, feature)
        if _num_or_none(value) is None:
            missing += 1
        else:
            values.append(float(value))
    return values, missing


def _safe_feature(row: dict, feature: str) -> Any:
    if feature == "boundary_shape_calibrated_score":
        return (safe_prediction_feature_vector(row) or {}).get("boundary_shape_calibrated_score")
    if feature == "boundary_shape_rank_score":
        return _shadow_value(row, "boundary_shape_rank_score")
    return prediction_features(row).get(feature)


def _missing_names(row: dict) -> list[str]:
    metadata = _metadata(row)
    names = metadata.get("missing_safe_features")
    if isinstance(names, list):
        return [str(name) for name in names]
    features = prediction_features(row)
    missing = []
    for name in PREDICTION_FEATURE_NAMES:
        if _num_or_none(features.get(name)) is None:
            missing.append(name)
    return missing


def _missing_count(row: dict) -> int:
    metadata = _metadata(row)
    if metadata.get("missing_safe_feature_count") is not None:
        return int(metadata.get("missing_safe_feature_count") or 0)
    return len(_missing_names(row))


def _metadata(row: dict) -> dict:
    return row.get("shadow_score_metadata") if isinstance(row.get("shadow_score_metadata"), dict) else {}


def _shadow_value(row: dict, field: str) -> float | None:
    shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
    return _num_or_none(shadow.get(field))


def _sample_key(row: dict) -> str:
    for key in ("sample_id", "image_id", "annotation_id", "task_id", "prediction_id"):
        if row.get(key) is not None:
            return f"{key}:{row.get(key)}"
    return json.dumps({key: row.get(key) for key in ("rank", "priority_score")}, sort_keys=True)


def _counter(items: list[dict], key: str) -> dict:
    values = [str(row.get(key) or "missing") for row in items]
    return dict(Counter(values).most_common(20))


def _image_size_bucket(row: dict) -> str:
    width = _num_or_none(row.get("image_width") or row.get("width"))
    height = _num_or_none(row.get("image_height") or row.get("height"))
    if width is None or height is None:
        return "missing"
    pixels = width * height
    if pixels < 256 * 256:
        return "small"
    if pixels < 1024 * 1024:
        return "medium"
    return "large"


def _area_bucket(value: Any) -> str:
    number = _num_or_none(value)
    if number is None:
        return "missing"
    if number < 0.01:
        return "tiny"
    if number < 0.05:
        return "small"
    if number < 0.25:
        return "medium"
    return "large"


def _missing_rate_by(items: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        buckets: dict[str, list[dict]] = {}
        for row in items:
            buckets.setdefault(str(row.get(key) or "missing"), []).append(row)
        out[key] = {
            name: _missing_rate(rows)
            for name, rows in sorted(buckets.items())
        }
    return out


def _missing_rate_by_bucket(items: list[dict], bucket_fn) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in items:
        buckets.setdefault(bucket_fn(row), []).append(row)
    return {name: _missing_rate(rows) for name, rows in sorted(buckets.items())}


def _missing_rate(rows: list[dict]) -> dict:
    missing = sum(_missing_count(row) > 0 for row in rows)
    return {"rows": len(rows), "rows_with_missing": missing, "missing_rate": float(missing / len(rows)) if rows else None}


def _distribution_shift(left: list[float], right: list[float]) -> dict:
    l = _distribution(left)
    r = _distribution(right)
    return {"median_delta": _optional_delta(r.get("median"), l.get("median")), "p95_delta": _optional_delta(r.get("p95"), l.get("p95"))}


def _relative_change(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline in {None, 0}:
        return None
    return float(abs(current - baseline) / abs(baseline))


def _jaccard(left: set, right: set) -> float | None:
    union = left | right
    return float(len(left & right) / len(union)) if union else None


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _num(value: Any) -> float:
    number = _num_or_none(value)
    return 0.0 if number is None else number


def _num_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_strip_forbidden(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["task_id", "sample_id", "image_id", "annotation_id", "group", "current_priority_score", "learned_shadow_score", "rank_current", "rank_learned", "rank_delta", "window_ids", "human_review_outcome", "reviewer", "reviewed_at", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat["window_ids"] = ",".join(row.get("window_ids") or [])
            writer.writerow(flat)


def _strip_forbidden(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in FORBIDDEN_OUTPUT_KEYS}


def _render_drift_markdown(report: dict) -> str:
    lines = ["# Shadow Drift Root-Cause Analysis", "", f"- baseline_window: {report.get('baseline_window')}", f"- windows: {report.get('windows')}", ""]
    for window, explanation in (report.get("likely_explanations") or {}).items():
        lines.extend([f"## {window}", f"- explanation: {explanation}", ""])
    return "\n".join(lines)


def _render_missing_markdown(report: dict) -> str:
    lines = ["# Shadow Missing Feature Analysis", "", f"- windows: {report.get('windows')}", f"- assessment: {report.get('overall_assessment')}", ""]
    for window, row in (report.get("per_window") or {}).items():
        lines.extend([f"## {window}", f"- rows_with_missing_features: {row.get('rows_with_missing_features')}", f"- missing_feature_names_frequency: {row.get('missing_feature_names_frequency')}", ""])
    return "\n".join(lines)


def _render_topk_markdown(report: dict) -> str:
    lines = ["# Top-k Jaccard Alert Analysis", "", f"- baseline_window: {report.get('baseline_window')}", ""]
    for window, row in (report.get("per_window") or {}).items():
        lines.extend([f"## {window}", f"- jaccard: {row.get('jaccard')}", f"- relative_change_vs_baseline: {row.get('relative_change_vs_baseline')}", f"- likely_cause: {row.get('likely_cause')}", ""])
    return "\n".join(lines)


def _render_schema_aware_drift_markdown(report: dict) -> str:
    lines = [
        "# Schema-Aware Shadow Drift Analysis",
        "",
        f"- classification: {report.get('classification')}",
        f"- stable_conclusion_available: {report.get('stable_conclusion_available')}",
        f"- conclusion: {report.get('conclusion')}",
        f"- alert_severity_summary: {report.get('alert_severity_summary')}",
        "",
        "## Comparable Pairs",
        "",
    ]
    for row in report.get("comparable_pairs") or []:
        lines.append(
            f"- {row.get('pair')}: severity={row.get('alert_severity')} "
            f"p95_drift={row.get('p95_drift')} psi={row.get('score_distribution_psi')}"
        )
    lines.extend(["", "## Non-Comparable Pairs", ""])
    for row in report.get("non_comparable_pairs") or []:
        lines.append(f"- {row.get('pair')}: severity={row.get('alert_severity')} reason={row.get('reason')}")
    return "\n".join(lines) + "\n"


def _render_schema_aware_topk_markdown(report: dict) -> str:
    lines = [
        "# Schema-Aware Top-k Jaccard Analysis",
        "",
        f"- classification: {report.get('classification')}",
        f"- k: {report.get('k')}",
        f"- stable_conclusion_available: {report.get('stable_conclusion_available')}",
        f"- conclusion: {report.get('conclusion')}",
        f"- alert_severity_summary: {report.get('alert_severity_summary')}",
        "",
        "## Comparable Pairs",
        "",
    ]
    for row in report.get("comparable_pairs") or []:
        lines.append(
            f"- {row.get('pair')}: severity={row.get('alert_severity')} "
            f"raw_learned_jaccard={row.get('raw_learned_topk_jaccard')} "
            f"shared={row.get('same_sample_overlap')}"
        )
    lines.extend(["", "## Non-Comparable Pairs", ""])
    for row in report.get("non_comparable_pairs") or []:
        lines.append(f"- {row.get('pair')}: severity={row.get('alert_severity')} likely_cause={row.get('likely_cause')}")
    return "\n".join(lines) + "\n"


def _render_schema_aware_missing_markdown(report: dict) -> str:
    lines = [
        "# Schema-Aware Missing Feature Analysis",
        "",
        f"- classification: {report.get('classification')}",
        f"- alert_severity_summary: {report.get('alert_severity_summary')}",
        f"- overall_assessment: {report.get('overall_assessment')}",
        "",
        "## Windows",
        "",
    ]
    for window, row in (report.get("per_window") or {}).items():
        lines.append(
            f"- {window}: schema={row.get('queue_schema_version')} "
            f"completeness={row.get('feature_completeness_bucket')} "
            f"severity={row.get('alert_severity')} missing={row.get('missing_feature_names')}"
        )
    lines.extend(["", "## Schema Buckets", f"- {report.get('schema_buckets')}"])
    return "\n".join(lines) + "\n"


def _review_guidelines() -> str:
    return "\n".join(
        [
            "# Shadow Review Guidelines",
            "",
            "Use only the provided safe production-visible fields.",
            "",
            "## Outcomes",
            "- major_correction_needed",
            "- minor_correction_needed",
            "- ok",
            "- unclear",
            "",
            "Judge whether the current prediction clearly needs human correction, whether learned-high/current-low samples are missed risks, and whether learned score over-prioritizes safe samples. If unclear, mark unclear rather than guessing.",
        ]
    ) + "\n"
