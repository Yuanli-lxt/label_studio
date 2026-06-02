from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np

from image_segmentation.benchmark.metrics import safe_mean, safe_median


def build_evaluation_report(queue_items: list[dict], correction_records: list[dict]) -> dict:
    ok_records = [row for row in correction_records if isinstance(row.get("delta"), dict)]
    deltas = [row["delta"] for row in ok_records]
    ious = [_num(delta.get("model_human_iou")) for delta in deltas if delta.get("model_human_iou") is not None]
    dice = [_num(delta.get("model_human_dice")) for delta in deltas if delta.get("model_human_dice") is not None]
    precision = [
        _num(delta.get("model_human_precision")) for delta in deltas if delta.get("model_human_precision") is not None
    ]
    recall = [_num(delta.get("model_human_recall")) for delta in deltas if delta.get("model_human_recall") is not None]
    boundary_rows = [delta.get("boundary") for delta in deltas if isinstance(delta.get("boundary"), dict)]
    severities = Counter(str(delta.get("correction_severity") or "unknown") for delta in deltas)
    major_count = sum(1 for delta in deltas if delta.get("major_correction") is True)
    base_rate = _safe_div(major_count, len(ok_records))
    degenerate = len(ok_records) > 0 and major_count in (0, len(ok_records))
    warnings = []
    if degenerate:
        warnings.append("Degenerate major_correction labels; queue ranking metrics are not informative.")
    effectiveness = review_queue_effectiveness(queue_items, degenerate=degenerate)
    if (
        not degenerate
        and effectiveness.get("average_precision_for_major_correction") is not None
        and base_rate is not None
        and effectiveness["average_precision_for_major_correction"] < base_rate
    ):
        warnings.append("Queue AP is below the positive base rate; ranking may be worse than random for this run.")
    if (
        not degenerate
        and effectiveness.get("precision_at_20_percent") is not None
        and base_rate is not None
        and effectiveness["precision_at_20_percent"] < base_rate
    ):
        warnings.append("precision@20% is below random expectation.")
    overall = {
        "n_samples": len(ok_records),
        "n_unique_images": len({str(row.get("image_id")) for row in ok_records if row.get("image_id") is not None}),
        "n_categories": len(
            {str(row.get("category_name") or row.get("category_id")) for row in ok_records if row.get("category_name") or row.get("category_id")}
        ),
        "mean_model_gt_iou": safe_mean(ious),
        "median_model_gt_iou": safe_median(ious),
        "mean_dice": safe_mean(dice),
        "mean_precision": safe_mean(precision),
        "mean_recall": safe_mean(recall),
        "major_correction_count": major_count,
        "major_correction_rate": base_rate,
        "correction_severity_distribution": dict(sorted(severities.items())),
        "degenerate_major_correction_labels": degenerate,
        "metric_warnings": warnings,
    }
    return {
        "overall_quality": overall,
        "boundary_quality": _boundary_quality(boundary_rows),
        "base_rate": {
            "major_correction_base_rate": base_rate,
            "random_expected_precision_at_10_percent": base_rate,
            "random_expected_precision_at_20_percent": base_rate,
            "random_expected_recall_at_10_percent": 0.10 if base_rate not in (None, 0.0) else None,
            "random_expected_recall_at_20_percent": 0.20 if base_rate not in (None, 0.0) else None,
        },
        "degenerate_major_correction_labels": degenerate,
        "metric_warnings": warnings,
        "by_dataset": _group_records(ok_records, lambda row: str(row.get("dataset") or "unknown")),
        "by_difficulty_tag": _group_by_tag(ok_records, queue_items),
        "review_queue_effectiveness": effectiveness,
        "baseline_comparison": baseline_comparison(queue_items, degenerate=degenerate),
        "top_10_percent_samples": top_fraction_details(queue_items, 0.10),
        "top_20_percent_samples": top_fraction_details(queue_items, 0.20),
        "false_negatives": false_negative_details(queue_items, 0.20, limit=20),
        "major_correction_diagnostics": major_correction_diagnostics(ok_records, queue_items),
        "boundary_stress_diagnostics": boundary_stress_diagnostics(ok_records, queue_items),
        "category_frequency_breakdown": category_frequency_breakdown(ok_records),
        "score_component_summary": score_component_summary(queue_items),
        "uncertainty_summary": uncertainty_summary(queue_items),
    }


def review_queue_effectiveness(queue_items: list[dict], degenerate: bool = False) -> dict:
    return {
        "precision_at_10_percent": precision_at_fraction(queue_items, 0.10),
        "precision_at_20_percent": precision_at_fraction(queue_items, 0.20),
        "recall_at_10_percent": recall_at_fraction(queue_items, 0.10),
        "recall_at_20_percent": recall_at_fraction(queue_items, 0.20),
        "lift_at_10_percent_over_random": None if degenerate else lift_at_fraction(queue_items, 0.10),
        "lift_at_20_percent_over_random": None if degenerate else lift_at_fraction(queue_items, 0.20),
        "average_precision_for_major_correction": None if degenerate else average_precision(queue_items),
        "top_k_major_correction_capture_curve": capture_curve(queue_items),
        "ranking_metrics_informative": not degenerate,
    }


def baseline_comparison(queue_items: list[dict], degenerate: bool = False) -> dict:
    baselines = {
        "mask_quality_only": sorted(
            queue_items,
            key=lambda item: -_num((item.get("score_components") or {}).get("rule_review_score")),
        ),
        "uncertainty_only": sorted(
            queue_items,
            key=lambda item: -_num((item.get("score_components") or {}).get("uncertainty_score")),
        ),
        "correction_risk_only": sorted(
            queue_items,
            key=lambda item: -_num((item.get("score_components") or {}).get("correction_risk_score")),
        ),
        "full_layer5_priority_score": sorted(queue_items, key=lambda item: -_num(item.get("priority_score"))),
    }
    out = {"random": _random_baseline(queue_items, degenerate=degenerate)}
    for name, rows in baselines.items():
        out[name] = {
            "precision_at_10_percent": precision_at_fraction(rows, 0.10),
            "precision_at_20_percent": precision_at_fraction(rows, 0.20),
            "recall_at_10_percent": recall_at_fraction(rows, 0.10),
            "recall_at_20_percent": recall_at_fraction(rows, 0.20),
            "average_precision_for_major_correction": None if degenerate else average_precision(rows),
            "ranking_metrics_informative": not degenerate,
        }
    return out


def _random_baseline(items: list[dict], degenerate: bool = False) -> dict:
    base_rate = _safe_div(sum(1 for item in items if _is_major(item)), len(items))
    return {
        "precision_at_10_percent": base_rate,
        "precision_at_20_percent": base_rate,
        "recall_at_10_percent": 0.10 if items and base_rate not in (None, 0.0) else None,
        "recall_at_20_percent": 0.20 if items and base_rate not in (None, 0.0) else None,
        "average_precision_for_major_correction": None if degenerate else base_rate,
        "ranking_metrics_informative": not degenerate,
    }


def precision_at_fraction(items: list[dict], fraction: float) -> float | None:
    top = _top_fraction(items, fraction)
    if not top:
        return None
    return float(sum(1 for item in top if _is_major(item)) / len(top))


def recall_at_fraction(items: list[dict], fraction: float) -> float | None:
    total_major = sum(1 for item in items if _is_major(item))
    if total_major == 0:
        return None
    return float(sum(1 for item in _top_fraction(items, fraction) if _is_major(item)) / total_major)


def lift_at_fraction(items: list[dict], fraction: float) -> float | None:
    precision = precision_at_fraction(items, fraction)
    base_rate = _safe_div(sum(1 for item in items if _is_major(item)), len(items))
    if precision is None or base_rate in (None, 0.0):
        return None
    return float(precision / base_rate)


def average_precision(items: list[dict]) -> float | None:
    total_major = sum(1 for item in items if _is_major(item))
    if total_major == 0:
        return None
    hits = 0
    precision_sum = 0.0
    for idx, item in enumerate(items, start=1):
        if _is_major(item):
            hits += 1
            precision_sum += hits / idx
    return float(precision_sum / total_major)


def capture_curve(items: list[dict]) -> list[dict]:
    return [
        {
            "fraction": fraction,
            "k": len(_top_fraction(items, fraction)),
            "major_corrections_captured": sum(1 for item in _top_fraction(items, fraction) if _is_major(item)),
            "recall": recall_at_fraction(items, fraction),
        }
        for fraction in [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    ]


def top_fraction_details(items: list[dict], fraction: float) -> list[dict]:
    return [_sample_detail(item) for item in _top_fraction(items, fraction)]


def false_negative_details(items: list[dict], top_fraction: float = 0.20, limit: int = 20) -> list[dict]:
    top_count = len(_top_fraction(items, top_fraction))
    out = []
    for item in items[top_count:]:
        if _is_major(item):
            detail = _sample_detail(item)
            delta = _delta(item)
            detail["correction_reason"] = delta.get("correction_reason")
            out.append(detail)
            if len(out) >= limit:
                break
    return out


def major_correction_diagnostics(records: list[dict], queue_items: list[dict]) -> dict:
    major_count = sum(1 for row in records if (row.get("delta") or {}).get("major_correction") is True)
    return {
        "major_correction_count": major_count,
        "category_level_major_correction_top20": _group_records(
            records,
            lambda row: str(row.get("category_name") or row.get("label") or "unknown"),
            min_n=5,
        )[:20],
        "difficulty_tag_major_correction_rates": sorted(
            _group_by_tag(records, queue_items), key=lambda row: (-_num(row.get("major_correction_rate")), row["tag"])
        ),
        "top_false_negatives_by_correction_area_ratio": sorted(
            false_negative_details(queue_items, 0.20, limit=len(queue_items)),
            key=lambda row: -_num(row.get("correction_area_ratio")),
        )[:20],
        "top_false_negatives_by_low_model_human_iou": sorted(
            false_negative_details(queue_items, 0.20, limit=len(queue_items)),
            key=lambda row: _num(row.get("model_human_iou")),
        )[:20],
    }


def _boundary_quality(boundaries: list[dict]) -> dict:
    return {
        "mean_boundary_iou": safe_mean([_num(row.get("boundary_iou")) for row in boundaries if row.get("boundary_iou") is not None]),
        "mean_boundary_f1": safe_mean([_num(row.get("boundary_f1")) for row in boundaries if row.get("boundary_f1") is not None]),
        "mean_boundary_precision": safe_mean([_num(row.get("boundary_precision")) for row in boundaries if row.get("boundary_precision") is not None]),
        "mean_boundary_recall": safe_mean([_num(row.get("boundary_recall")) for row in boundaries if row.get("boundary_recall") is not None]),
        "mean_boundary_error_area_ratio": safe_mean([_num(row.get("boundary_error_area_ratio")) for row in boundaries if row.get("boundary_error_area_ratio") is not None]),
        "available_count": len(boundaries),
    }


def boundary_stress_diagnostics(records: list[dict], queue_items: list[dict]) -> dict:
    tag_rows = {row["tag"]: row for row in _group_by_tag(records, queue_items)}
    focused = {}
    for tag in ["high_boundary_complexity", "thin_structure", "elongated_object", "small_object", "touches_border"]:
        row = dict(tag_rows.get(tag) or {"tag": tag, "n_samples": 0, "major_correction_rate": None})
        rows = [record for record in records if tag in (record.get("difficulty_tags") or [])]
        row["mean_boundary_f1"] = safe_mean([
            _num(((record.get("delta") or {}).get("boundary") or {}).get("boundary_f1"))
            for record in rows
            if (((record.get("delta") or {}).get("boundary") or {}).get("boundary_f1")) is not None
        ])
        focused[tag] = row
    false_negatives = false_negative_details(queue_items, 0.20, limit=len(queue_items))
    return {
        "difficulty_tags": focused,
        "mean_boundary_f1_by_difficulty_tag": {
            tag: row.get("mean_boundary_f1") for tag, row in focused.items()
        },
        "top_false_negatives_by_low_boundary_f1": sorted(
            false_negatives,
            key=lambda row: _num(row.get("boundary_f1")),
        )[:20],
        "top_false_negatives_by_high_correction_area_ratio": sorted(
            false_negatives,
            key=lambda row: -_num(row.get("correction_area_ratio")),
        )[:20],
    }


def category_frequency_breakdown(records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        value = row.get("category_frequency")
        if value is None and isinstance(row.get("metadata"), dict):
            value = row["metadata"].get("category_frequency")
        groups[str(value if value is not None else "unknown")].append(row)
    out = []
    for frequency, rows in sorted(groups.items()):
        deltas = [row["delta"] for row in rows]
        ious = [_num(delta.get("model_human_iou")) for delta in deltas if delta.get("model_human_iou") is not None]
        out.append(
            {
                "frequency": frequency,
                "n_samples": len(rows),
                "mean_iou": safe_mean(ious),
                "major_correction_rate": _safe_div(
                    sum(1 for delta in deltas if delta.get("major_correction") is True), len(rows)
                ),
            }
        )
    return out


def score_component_summary(items: list[dict]) -> dict:
    component_names = [
        "correction_risk_score",
        "uncertainty_score",
        "rule_review_score",
        "geometry_complexity_score",
        "boundary_shape_score",
        "diversity_score",
    ]
    out = {}
    for name in component_names:
        values = [
            _num((item.get("score_components") or {}).get(name))
            for item in items
            if isinstance((item.get("score_components") or {}).get(name), (int, float))
        ]
        out[name] = {
            "mean": safe_mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "non_null_count": len(values),
        }
    return out


def uncertainty_summary(items: list[dict]) -> dict:
    rows = []
    for item in items:
        metadata = item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {}
        uncertainty = metadata.get("uncertainty") if isinstance(metadata.get("uncertainty"), dict) else {}
        if uncertainty:
            rows.append((item, uncertainty))
    unstable = [(item, u) for item, u in rows if u.get("stable") is False or str(u.get("stability_bucket")).lower() == "low"]
    stable = [(item, u) for item, u in rows if u.get("stable") is True]
    return {
        "enabled_count": sum(1 for _, u in rows if u.get("enabled") is True),
        "mean_uncertainty_score": safe_mean([
            _num((item.get("score_components") or {}).get("uncertainty_score")) for item, _ in rows
        ]),
        "mean_pairwise_iou": safe_mean([
            _num(u.get("mean_pairwise_iou")) for _, u in rows if u.get("mean_pairwise_iou") is not None
        ]),
        "unstable_count": len(unstable),
        "unstable_major_correction_rate": _major_rate([item for item, _ in unstable]),
        "stable_major_correction_rate": _major_rate([item for item, _ in stable]),
    }


def render_markdown_report(report: dict) -> str:
    overall = report.get("overall_quality", {})
    queue = report.get("review_queue_effectiveness", {})
    backend = report.get("backend") or {}
    runtime = report.get("runtime") or {}
    warnings = report.get("metric_warnings") or []
    base = report.get("base_rate") or {}
    boundary = report.get("boundary_quality") or {}
    lines = [
        "# Benchmark v0.1 Evaluation Report",
        "",
        "## Backend",
        f"- requested: {backend.get('backend_requested', 'unknown')}",
        f"- resolved: {backend.get('backend_resolved', 'unknown')}",
        "",
        "## Runtime",
        *_runtime_lines(runtime),
        "",
        "## Overall Quality",
        f"- samples: {overall.get('n_samples')}",
        f"- unique images: {overall.get('n_unique_images')}",
        f"- categories: {overall.get('n_categories')}",
        f"- mean model/GT IoU: {_fmt(overall.get('mean_model_gt_iou'))}",
        f"- median model/GT IoU: {_fmt(overall.get('median_model_gt_iou'))}",
        f"- mean Dice: {_fmt(overall.get('mean_dice'))}",
        f"- mean precision: {_fmt(overall.get('mean_precision'))}",
        f"- mean recall: {_fmt(overall.get('mean_recall'))}",
        f"- major correction count: {overall.get('major_correction_count')}",
        f"- major correction rate: {_fmt(overall.get('major_correction_rate'))}",
        f"- correction severity distribution: {overall.get('correction_severity_distribution')}",
        "",
        "## Boundary Quality",
        f"- mean boundary IoU: {_fmt(boundary.get('mean_boundary_iou'))}",
        f"- mean boundary F1: {_fmt(boundary.get('mean_boundary_f1'))}",
        f"- mean boundary precision: {_fmt(boundary.get('mean_boundary_precision'))}",
        f"- mean boundary recall: {_fmt(boundary.get('mean_boundary_recall'))}",
        f"- mean boundary error area ratio: {_fmt(boundary.get('mean_boundary_error_area_ratio'))}",
        "",
        "## Base Rate",
        f"- major correction base rate: {_fmt(base.get('major_correction_base_rate'))}",
        f"- random expected precision@10%: {_fmt(base.get('random_expected_precision_at_10_percent'))}",
        f"- random expected precision@20%: {_fmt(base.get('random_expected_precision_at_20_percent'))}",
        f"- random expected recall@10%: {_fmt(base.get('random_expected_recall_at_10_percent'))}",
        f"- random expected recall@20%: {_fmt(base.get('random_expected_recall_at_20_percent'))}",
        "",
        "## Review Queue Effectiveness",
        f"- precision@10%: {_fmt(queue.get('precision_at_10_percent'))}",
        f"- precision@20%: {_fmt(queue.get('precision_at_20_percent'))}",
        f"- recall@10%: {_fmt(queue.get('recall_at_10_percent'))}",
        f"- recall@20%: {_fmt(queue.get('recall_at_20_percent'))}",
        f"- lift@10% over random: {_fmt(queue.get('lift_at_10_percent_over_random'))}",
        f"- lift@20% over random: {_fmt(queue.get('lift_at_20_percent_over_random'))}",
        f"- AP for major correction: {_fmt(queue.get('average_precision_for_major_correction'))}",
        "",
        "## Uncertainty Summary",
        *_uncertainty_lines(report.get("uncertainty_summary") or {}),
        "",
        "## Score Component Summary",
        *_summary_lines(report.get("score_component_summary") or {}),
        "",
        "## Top 10% Samples",
        *_sample_lines(report.get("top_10_percent_samples") or []),
        "",
        "## Top 20% Samples",
        *_sample_lines(report.get("top_20_percent_samples") or []),
        "",
        "## False Negatives",
        *_sample_lines(report.get("false_negatives") or []),
        "",
        "## Major Correction Diagnostics",
        *_diagnostic_lines(report.get("major_correction_diagnostics") or {}),
        "",
        "## Boundary Stress Diagnostics",
        *_boundary_stress_lines(report.get("boundary_stress_diagnostics") or {}),
        "",
        "## Category Frequency Breakdown",
        *_frequency_lines(report.get("category_frequency_breakdown") or []),
        "",
        "## Warnings",
        *(f"- {warning}" for warning in warnings),
        *([] if warnings else ["- none"]),
        "",
        "## Leakage Guard",
        "Delta metrics are used only for evaluation and `evaluation_only.delta`, not for priority scoring.",
        "",
    ]
    return "\n".join(lines)


def _sample_detail(item: dict) -> dict:
    delta = _delta(item)
    source = item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {}
    return {
        "rank": item.get("rank"),
        "sample_id": item.get("task_id") or item.get("sample_id"),
        "dataset": item.get("dataset") or item.get("benchmark_dataset") or "COCO",
        "category_name": item.get("label") or item.get("category_name"),
        "priority_score": item.get("priority_score"),
        "priority_bucket": item.get("priority_bucket"),
        "review_reasons": item.get("review_reasons") or [],
        "score_components": item.get("score_components") or {},
        "major_correction": delta.get("major_correction"),
        "model_human_iou": delta.get("model_human_iou"),
        "correction_area_ratio": delta.get("correction_area_ratio"),
        "boundary_f1": (delta.get("boundary") or {}).get("boundary_f1") if isinstance(delta.get("boundary"), dict) else None,
        "boundary_iou": (delta.get("boundary") or {}).get("boundary_iou") if isinstance(delta.get("boundary"), dict) else None,
        "correction_reason": delta.get("correction_reason"),
        "source_metadata": source,
    }


def _summary_lines(summary: dict) -> list[str]:
    if not summary:
        return ["- none"]
    lines = []
    for name, row in summary.items():
        lines.append(
            f"- {name}: mean={_fmt(row.get('mean'))}, min={_fmt(row.get('min'))}, "
            f"max={_fmt(row.get('max'))}, n={row.get('non_null_count')}"
        )
    return lines


def _sample_lines(samples: list[dict]) -> list[str]:
    if not samples:
        return ["- none"]
    return [
        f"- rank {sample.get('rank')}: {sample.get('sample_id')} "
        f"score={_fmt(sample.get('priority_score'))} major={sample.get('major_correction')} "
        f"iou={_fmt(sample.get('model_human_iou'))} reasons={sample.get('review_reasons')}"
        for sample in samples
    ]


def _uncertainty_lines(summary: dict) -> list[str]:
    if not summary:
        return ["- none"]
    return [
        f"- enabled_count: {summary.get('enabled_count')}",
        f"- mean_uncertainty_score: {_fmt(summary.get('mean_uncertainty_score'))}",
        f"- mean_pairwise_iou: {_fmt(summary.get('mean_pairwise_iou'))}",
        f"- unstable_count: {summary.get('unstable_count')}",
        f"- unstable_major_correction_rate: {_fmt(summary.get('unstable_major_correction_rate'))}",
        f"- stable_major_correction_rate: {_fmt(summary.get('stable_major_correction_rate'))}",
    ]


def _runtime_lines(runtime: dict) -> list[str]:
    if not runtime:
        return ["- unavailable"]
    return [
        f"- requested device: {runtime.get('requested_device')}",
        f"- resolved device: {runtime.get('resolved_device')}",
        f"- cuda available: {runtime.get('cuda_available')}",
        f"- cuda device: {runtime.get('cuda_device_name')}",
        f"- samples completed: {runtime.get('n_samples_completed')} / {runtime.get('n_samples_requested')}",
        f"- elapsed seconds: {_fmt(runtime.get('elapsed_seconds'))}",
        f"- seconds/sample mean: {_fmt(runtime.get('seconds_per_sample_mean'))}",
        f"- peak cuda allocated MB: {_fmt(runtime.get('peak_cuda_memory_allocated_mb'))}",
        f"- peak cuda reserved MB: {_fmt(runtime.get('peak_cuda_memory_reserved_mb'))}",
    ]


def _diagnostic_lines(diagnostics: dict) -> list[str]:
    if not diagnostics:
        return ["- none"]
    lines = [f"- major correction count: {diagnostics.get('major_correction_count')}"]
    categories = diagnostics.get("category_level_major_correction_top20") or []
    lines.append("- categories with highest major correction rate:")
    lines.extend(_group_lines(categories, "dataset"))
    tags = diagnostics.get("difficulty_tag_major_correction_rates") or []
    lines.append("- difficulty tags with highest major correction rate:")
    lines.extend(_group_lines(tags, "tag"))
    lines.append("- top false negatives by correction_area_ratio:")
    lines.extend(_sample_lines(diagnostics.get("top_false_negatives_by_correction_area_ratio") or []))
    lines.append("- top false negatives by low model_human_iou:")
    lines.extend(_sample_lines(diagnostics.get("top_false_negatives_by_low_model_human_iou") or []))
    return lines


def _boundary_stress_lines(diagnostics: dict) -> list[str]:
    if not diagnostics:
        return ["- none"]
    rows = diagnostics.get("difficulty_tags") or {}
    lines = []
    for tag in ["high_boundary_complexity", "thin_structure", "elongated_object", "small_object", "touches_border"]:
        row = rows.get(tag) or {}
        lines.append(
            f"- {tag}: n={row.get('n_samples', 0)}, major_rate={_fmt(row.get('major_correction_rate'))}, "
            f"mean_boundary_f1={_fmt(row.get('mean_boundary_f1'))}"
        )
    lines.append("- top false negatives by low boundary F1:")
    lines.extend(_sample_lines(diagnostics.get("top_false_negatives_by_low_boundary_f1") or []))
    lines.append("- top false negatives by high correction_area_ratio:")
    lines.extend(_sample_lines(diagnostics.get("top_false_negatives_by_high_correction_area_ratio") or []))
    return lines


def _frequency_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        f"- {row.get('frequency')}: n={row.get('n_samples')}, mean_iou={_fmt(row.get('mean_iou'))}, "
        f"major_rate={_fmt(row.get('major_correction_rate'))}"
        for row in rows
    ]


def _group_lines(rows: list[dict], name_key: str) -> list[str]:
    if not rows:
        return ["  - none"]
    return [
        f"  - {row.get(name_key)}: n={row.get('n_samples')}, major_rate={_fmt(row.get('major_correction_rate'))}"
        for row in rows[:20]
    ]


def _group_records(records: list[dict], key_fn: Callable[[dict], str], min_n: int = 1) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[key_fn(row)].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        if len(rows) < min_n:
            continue
        deltas = [row["delta"] for row in rows]
        ious = [_num(delta.get("model_human_iou")) for delta in deltas if delta.get("model_human_iou") is not None]
        ratios = [
            _num(delta.get("correction_area_ratio"))
            for delta in deltas
            if delta.get("correction_area_ratio") is not None
        ]
        out.append(
            {
                "dataset": key,
                "n_samples": len(rows),
                "mean_iou": safe_mean(ious),
                "median_iou": safe_median(ious),
                "major_correction_rate": _safe_div(
                    sum(1 for delta in deltas if delta.get("major_correction") is True), len(rows)
                ),
                "mean_correction_area_ratio": safe_mean(ratios),
            }
        )
    return sorted(out, key=lambda row: (-_num(row.get("major_correction_rate")), row["dataset"]))


def _group_by_tag(records: list[dict], queue_items: list[dict]) -> list[dict]:
    priority_by_sample = {item.get("task_id"): _num(item.get("priority_score")) for item in queue_items}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        for tag in row.get("difficulty_tags") or ["untagged"]:
            groups[str(tag)].append(row)
    out = []
    for tag, rows in sorted(groups.items()):
        deltas = [row["delta"] for row in rows]
        ious = [_num(delta.get("model_human_iou")) for delta in deltas if delta.get("model_human_iou") is not None]
        boundary_f1 = [
            _num((delta.get("boundary") or {}).get("boundary_f1"))
            for delta in deltas
            if isinstance(delta.get("boundary"), dict) and (delta.get("boundary") or {}).get("boundary_f1") is not None
        ]
        out.append(
            {
                "tag": tag,
                "n_samples": len(rows),
                "mean_iou": safe_mean(ious),
                "mean_boundary_f1": safe_mean(boundary_f1),
                "major_correction_rate": _safe_div(
                    sum(1 for delta in deltas if delta.get("major_correction") is True), len(rows)
                ),
                "mean_priority_score": safe_mean([priority_by_sample.get(row.get("task_id"), 0.0) for row in rows]),
            }
        )
    return out


def _top_fraction(items: list[dict], fraction: float) -> list[dict]:
    if not items:
        return []
    k = max(1, int(np.ceil(len(items) * fraction)))
    return items[:k]


def _is_major(item: dict) -> bool:
    delta = _delta(item)
    return delta.get("major_correction") is True


def _delta(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    return ((item.get("evaluation_only") or {}).get("delta") or {}) if isinstance(item, dict) else {}


def _major_rate(items: list[dict]) -> float | None:
    return _safe_div(sum(1 for item in items if _is_major(item)), len(items))


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
