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
    severities = Counter(str(delta.get("correction_severity") or "unknown") for delta in deltas)
    major_count = sum(1 for delta in deltas if delta.get("major_correction") is True)
    overall = {
        "n_samples": len(ok_records),
        "mean_model_gt_iou": safe_mean(ious),
        "median_model_gt_iou": safe_median(ious),
        "mean_dice": safe_mean(dice),
        "mean_precision": safe_mean(precision),
        "mean_recall": safe_mean(recall),
        "major_correction_rate": _safe_div(major_count, len(ok_records)),
        "correction_severity_distribution": dict(sorted(severities.items())),
    }
    return {
        "overall_quality": overall,
        "by_dataset": _group_records(ok_records, lambda row: str(row.get("dataset") or "unknown")),
        "by_difficulty_tag": _group_by_tag(ok_records, queue_items),
        "review_queue_effectiveness": review_queue_effectiveness(queue_items),
        "baseline_comparison": baseline_comparison(queue_items),
    }


def review_queue_effectiveness(queue_items: list[dict]) -> dict:
    return {
        "precision_at_10_percent": precision_at_fraction(queue_items, 0.10),
        "precision_at_20_percent": precision_at_fraction(queue_items, 0.20),
        "recall_at_10_percent": recall_at_fraction(queue_items, 0.10),
        "recall_at_20_percent": recall_at_fraction(queue_items, 0.20),
        "lift_at_10_percent_over_random": lift_at_fraction(queue_items, 0.10),
        "lift_at_20_percent_over_random": lift_at_fraction(queue_items, 0.20),
        "average_precision_for_major_correction": average_precision(queue_items),
        "top_k_major_correction_capture_curve": capture_curve(queue_items),
    }


def baseline_comparison(queue_items: list[dict]) -> dict:
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
    out = {"random": _random_baseline(queue_items)}
    for name, rows in baselines.items():
        out[name] = {
            "precision_at_10_percent": precision_at_fraction(rows, 0.10),
            "precision_at_20_percent": precision_at_fraction(rows, 0.20),
            "recall_at_10_percent": recall_at_fraction(rows, 0.10),
            "recall_at_20_percent": recall_at_fraction(rows, 0.20),
            "average_precision_for_major_correction": average_precision(rows),
        }
    return out


def _random_baseline(items: list[dict]) -> dict:
    base_rate = _safe_div(sum(1 for item in items if _is_major(item)), len(items))
    return {
        "precision_at_10_percent": base_rate,
        "precision_at_20_percent": base_rate,
        "recall_at_10_percent": 0.10 if items and base_rate not in (None, 0.0) else None,
        "recall_at_20_percent": 0.20 if items and base_rate not in (None, 0.0) else None,
        "average_precision_for_major_correction": base_rate,
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


def render_markdown_report(report: dict) -> str:
    overall = report.get("overall_quality", {})
    queue = report.get("review_queue_effectiveness", {})
    lines = [
        "# Benchmark v0.1 Evaluation Report",
        "",
        "## Overall Quality",
        f"- samples: {overall.get('n_samples')}",
        f"- mean model/GT IoU: {_fmt(overall.get('mean_model_gt_iou'))}",
        f"- median model/GT IoU: {_fmt(overall.get('median_model_gt_iou'))}",
        f"- mean Dice: {_fmt(overall.get('mean_dice'))}",
        f"- mean precision: {_fmt(overall.get('mean_precision'))}",
        f"- mean recall: {_fmt(overall.get('mean_recall'))}",
        f"- major correction rate: {_fmt(overall.get('major_correction_rate'))}",
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
        "## Leakage Guard",
        "Delta metrics are used only for evaluation and `evaluation_only.delta`, not for priority scoring.",
        "",
    ]
    return "\n".join(lines)


def _group_records(records: list[dict], key_fn: Callable[[dict], str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[key_fn(row)].append(row)
    out = []
    for key, rows in sorted(groups.items()):
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
    return out


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
        out.append(
            {
                "tag": tag,
                "n_samples": len(rows),
                "mean_iou": safe_mean(ious),
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
    delta = ((item.get("evaluation_only") or {}).get("delta") or {}) if isinstance(item, dict) else {}
    return delta.get("major_correction") is True


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
