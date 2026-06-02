from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.compare_review_strategies import BOOTSTRAP_METRICS
from image_segmentation.benchmark.evaluation import (
    average_precision,
    lift_at_fraction,
    precision_at_fraction,
    recall_at_fraction,
)
from image_segmentation.benchmark.prediction_feature_scoring import (
    calibrated_boundary_shape_score,
    rank_boundary_shape_scores,
)


COMPONENTS = [
    "correction_risk_score",
    "uncertainty_score",
    "rule_review_score",
    "geometry_complexity_score",
    "boundary_shape_score",
    "diversity_score",
]

DEFAULT_PRESETS = {
    "current_full_priority": {
        "description": "Use existing priority_score from review_queue; do not recompute.",
        "use_existing_priority_score": True,
    },
    "risk_only": {
        "correction_risk_score": 1.0,
        "uncertainty_score": 0.0,
        "rule_review_score": 0.0,
        "geometry_complexity_score": 0.0,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.0,
    },
    "risk_heavy": {
        "correction_risk_score": 0.65,
        "uncertainty_score": 0.10,
        "rule_review_score": 0.10,
        "geometry_complexity_score": 0.10,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.05,
    },
    "risk_dominant": {
        "correction_risk_score": 0.80,
        "uncertainty_score": 0.05,
        "rule_review_score": 0.05,
        "geometry_complexity_score": 0.05,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.05,
    },
    "uncertainty_heavy": {
        "correction_risk_score": 0.35,
        "uncertainty_score": 0.35,
        "rule_review_score": 0.10,
        "geometry_complexity_score": 0.10,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.10,
    },
    "quality_heavy": {
        "correction_risk_score": 0.35,
        "uncertainty_score": 0.10,
        "rule_review_score": 0.30,
        "geometry_complexity_score": 0.15,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.10,
    },
    "no_diversity": {
        "correction_risk_score": 0.55,
        "uncertainty_score": 0.15,
        "rule_review_score": 0.15,
        "geometry_complexity_score": 0.15,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.0,
    },
    "balanced_no_risk": {
        "correction_risk_score": 0.0,
        "uncertainty_score": 0.25,
        "rule_review_score": 0.30,
        "geometry_complexity_score": 0.25,
        "boundary_shape_score": 0.0,
        "diversity_score": 0.20,
    },
    "boundary_shape_experimental": {
        "correction_risk_score": 0.25,
        "uncertainty_score": 0.10,
        "rule_review_score": 0.20,
        "geometry_complexity_score": 0.15,
        "boundary_shape_score": 0.20,
        "diversity_score": 0.10,
        "description": "Experimental prediction-time boundary/shape fusion; does not use GT boundary metadata or delta metrics.",
    },
    "boundary_shape_rank_score": {
        "correction_risk_score": 0.25,
        "uncertainty_score": 0.10,
        "rule_review_score": 0.20,
        "geometry_complexity_score": 0.15,
        "boundary_shape_score": 0.20,
        "diversity_score": 0.10,
        "boundary_shape_component": "boundary_shape_rank_score",
        "description": "Experimental rank-normalized prediction-time boundary/shape score.",
    },
    "boundary_shape_calibrated_score": {
        "correction_risk_score": 0.25,
        "uncertainty_score": 0.10,
        "rule_review_score": 0.20,
        "geometry_complexity_score": 0.15,
        "boundary_shape_score": 0.20,
        "diversity_score": 0.10,
        "boundary_shape_component": "boundary_shape_calibrated_score",
        "description": "Experimental robust prediction-time boundary/shape score with log clipping and neutral missing fill.",
    },
}


def ablate_review_weights(
    review_queue: str,
    output_dir: str,
    presets_file: str | None = None,
    bootstrap_iters: int = 0,
    random_seed: int = 42,
) -> dict:
    items = _read_jsonl(review_queue)
    output = Path(output_dir)
    rankings_dir = output / "preset_rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    presets = load_weight_presets(presets_file)
    labels = [_is_major(item) for item in items]
    positive_count = sum(labels)
    base_rate = _safe_div(positive_count, len(items))
    warnings = []
    if len(set(labels)) < 2:
        warnings.append("degenerate_labels")
    if positive_count < 30:
        warnings.append("positive_count_below_30")
    diagnostics = component_diagnostics(items)
    results: dict[str, Any] = {}
    ranked_by_preset: dict[str, list[dict]] = {}
    for preset_name, preset in presets.items():
        ranked, preset_warnings = rank_for_preset(items, preset_name, preset)
        ranked_by_preset[preset_name] = ranked
        _write_jsonl(rankings_dir / f"{preset_name}.review_queue.jsonl", ranked)
        metrics = metrics_for_ranked(ranked, base_rate=base_rate)
        if bootstrap_iters:
            metrics["bootstrap_ci"] = bootstrap_ci(ranked, preset_name, int(bootstrap_iters), int(random_seed))
        metrics["warnings"] = preset_warnings
        metrics["normalized_weights"] = normalize_preset_weights(preset)
        metrics["description"] = preset.get("description")
        metrics["available"] = True
        results[preset_name] = metrics
    pairwise = pairwise_comparisons(results)
    overlaps = topk_overlaps(ranked_by_preset)
    recommendation_text = recommendation(results)
    report = {
        "input_review_queue": review_queue,
        "n_samples": len(items),
        "positive_count": positive_count,
        "major_correction_base_rate": base_rate,
        "bootstrap_iters": int(bootstrap_iters),
        "random_seed": int(random_seed),
        "presets": results,
        "leaderboard_by_ap": _leaderboard(results, "average_precision_for_major_correction"),
        "leaderboard_by_lift_at_20": _leaderboard(results, "lift_at_20_percent_over_random"),
        "pairwise_comparison": pairwise,
        "topk_overlap": overlaps,
        "component_diagnostics": diagnostics,
        "recommendation": recommendation_text,
        "warnings": warnings,
    }
    _write_json(output / "weight_ablation.json", report)
    (output / "weight_ablation.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def load_weight_presets(path: str | None = None) -> dict:
    if not path:
        return dict(DEFAULT_PRESETS)
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("weight presets file must contain a mapping")
    return data


def normalize_preset_weights(preset: dict) -> dict:
    if preset.get("use_existing_priority_score") is True:
        return {}
    raw = {name: max(0.0, _num(preset.get(name))) for name in COMPONENTS}
    total = sum(raw.values())
    if total <= 0:
        return {name: 0.0 for name in COMPONENTS}
    return {name: value / total for name, value in raw.items()}


def rank_for_preset(items: list[dict], preset_name: str, preset: dict) -> tuple[list[dict], list[str]]:
    weights = normalize_preset_weights(preset)
    warnings = []
    ranked = []
    rank_scores = rank_boundary_shape_scores(items)
    for index, item in enumerate(items):
        row = dict(item)
        components, component_warnings = _component_values(item, rank_scores[index] if index < len(rank_scores) else None)
        warnings.extend(component_warnings)
        if preset.get("use_existing_priority_score") is True:
            score = _clip(_num(item.get("priority_score")))
        else:
            boundary_component = preset.get("boundary_shape_component")
            if boundary_component:
                components["boundary_shape_score"] = components.get(str(boundary_component), 0.0)
            score = sum(components[name] * weights.get(name, 0.0) for name in COMPONENTS)
            score = _clip(score)
        row["ablation_preset"] = preset_name
        row["ablation_priority_score"] = score
        row["ablation_score_components"] = components
        row["ablation_weights"] = weights
        row["_input_index"] = index
        ranked.append(row)
    ranked.sort(key=lambda row: (-_num(row.get("ablation_priority_score")), row["_input_index"], str(row.get("task_id"))))
    for rank, row in enumerate(ranked, start=1):
        row["ablation_rank"] = rank
        row.pop("_input_index", None)
    return ranked, sorted(set(warnings))


def _component_values(item: dict, rank_score: float | None = None) -> tuple[dict, list[str]]:
    raw = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
    values = {}
    warnings = []
    for name in COMPONENTS:
        value = raw.get(name)
        if not isinstance(value, (int, float)):
            values[name] = 0.0
            warnings.append(f"missing_{name}")
            continue
        number = float(value)
        if number < 0.0 or number > 1.0:
            warnings.append(f"clipped_{name}")
        values[name] = _clip(number)
    values["boundary_shape_rank_score"] = _clip(_num(rank_score))
    calibrated = raw.get("boundary_shape_calibrated_score")
    if not isinstance(calibrated, (int, float)):
        calibrated = calibrated_boundary_shape_score(item)
    values["boundary_shape_calibrated_score"] = _clip(_num(calibrated))
    return values, warnings


def metrics_for_ranked(items: list[dict], base_rate: float | None = None) -> dict:
    top20 = _top_fraction(items, 0.20)
    positives = sum(1 for item in items if _is_major(item))
    labels = [1 if _is_major(item) else 0 for item in items]
    scores = [_num(item.get("ablation_priority_score", item.get("priority_score"))) for item in items]
    binary = _binary_metrics(labels, scores)
    return {
        "n_samples": len(items),
        "positive_count": positives,
        "major_correction_base_rate": base_rate if base_rate is not None else _safe_div(positives, len(items)),
        "precision_at_10_percent": precision_at_fraction(items, 0.10),
        "precision_at_20_percent": precision_at_fraction(items, 0.20),
        "recall_at_10_percent": recall_at_fraction(items, 0.10),
        "recall_at_20_percent": recall_at_fraction(items, 0.20),
        "lift_at_10_percent_over_random": lift_at_fraction(items, 0.10),
        "lift_at_20_percent_over_random": lift_at_fraction(items, 0.20),
        "average_precision_for_major_correction": average_precision(items),
        "roc_auc_for_major_correction": binary.get("roc_auc"),
        "brier_score_for_major_correction": binary.get("brier_score"),
        "top_20_percent_major_correction_count": sum(1 for item in top20 if _is_major(item)),
        "top_20_percent_size": len(top20),
        "meets_initial_standard": bool(
            _num(lift_at_fraction(items, 0.20)) > 1.5
            and base_rate is not None
            and _num(average_precision(items)) > base_rate
            and _num(precision_at_fraction(items, 0.20)) > base_rate
        ),
    }


def bootstrap_ci(items: list[dict], preset_name: str, bootstrap_iters: int, random_seed: int) -> dict:
    labels = [_is_major(item) for item in items]
    if len(set(labels)) < 2:
        return {name: {"value": None, "ci95_low": None, "ci95_high": None, "warning": "degenerate_labels"} for name in BOOTSTRAP_METRICS}
    rng = random.Random(random_seed + sum(ord(ch) for ch in preset_name))
    values = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(max(0, int(bootstrap_iters))):
        sample = sorted(
            [items[rng.randrange(len(items))] for _ in items],
            key=lambda row: -_num(row.get("ablation_priority_score")),
        )
        metrics = {
            "precision_at_20_percent": precision_at_fraction(sample, 0.20),
            "recall_at_20_percent": recall_at_fraction(sample, 0.20),
            "lift_at_20_percent_over_random": lift_at_fraction(sample, 0.20),
            "average_precision_for_major_correction": average_precision(sample),
        }
        for name, value in metrics.items():
            if value is not None:
                values[name].append(float(value))
    current = {
        "precision_at_20_percent": precision_at_fraction(items, 0.20),
        "recall_at_20_percent": recall_at_fraction(items, 0.20),
        "lift_at_20_percent_over_random": lift_at_fraction(items, 0.20),
        "average_precision_for_major_correction": average_precision(items),
    }
    out = {}
    for name in BOOTSTRAP_METRICS:
        samples = sorted(values[name])
        out[name] = {
            "value": current[name],
            "ci95_low": _percentile(samples, 2.5) if samples else None,
            "ci95_high": _percentile(samples, 97.5) if samples else None,
        }
        if not samples:
            out[name]["warning"] = "bootstrap_metric_degenerate"
    return out


def pairwise_comparisons(results: dict) -> dict:
    pairs = [
        ("risk_heavy", "current_full_priority"),
        ("risk_dominant", "current_full_priority"),
        ("risk_only", "current_full_priority"),
        ("risk_heavy", "risk_only"),
        ("no_diversity", "current_full_priority"),
        ("balanced_no_risk", "current_full_priority"),
        ("boundary_shape_experimental", "current_full_priority"),
        ("boundary_shape_experimental", "risk_heavy"),
        ("boundary_shape_rank_score", "current_full_priority"),
        ("boundary_shape_calibrated_score", "current_full_priority"),
    ]
    out = {}
    for left, right in pairs:
        if left not in results or right not in results:
            continue
        out[f"{left}_vs_{right}"] = {
            "ap_delta": _num(results[left].get("average_precision_for_major_correction")) - _num(results[right].get("average_precision_for_major_correction")),
            "lift_at_20_delta": _num(results[left].get("lift_at_20_percent_over_random")) - _num(results[right].get("lift_at_20_percent_over_random")),
            "precision_at_20_delta": _num(results[left].get("precision_at_20_percent")) - _num(results[right].get("precision_at_20_percent")),
            "left_beats_right_by_ap": _num(results[left].get("average_precision_for_major_correction")) > _num(results[right].get("average_precision_for_major_correction")),
            "left_beats_right_by_lift_at_20": _num(results[left].get("lift_at_20_percent_over_random")) > _num(results[right].get("lift_at_20_percent_over_random")),
        }
    return out


def topk_overlaps(ranked_by_preset: dict[str, list[dict]]) -> dict:
    pairs = [
        ("current_full_priority", "risk_heavy"),
        ("current_full_priority", "risk_only"),
        ("risk_heavy", "risk_only"),
        ("current_full_priority", "boundary_shape_experimental"),
        ("risk_heavy", "boundary_shape_experimental"),
        ("current_full_priority", "boundary_shape_rank_score"),
        ("current_full_priority", "boundary_shape_calibrated_score"),
    ]
    out = {}
    for left, right in pairs:
        if left not in ranked_by_preset or right not in ranked_by_preset:
            continue
        left_ids = _top_ids(ranked_by_preset[left], 0.20)
        right_ids = _top_ids(ranked_by_preset[right], 0.20)
        union = left_ids | right_ids
        out[f"{left}_vs_{right}"] = {
            "left_top20_size": len(left_ids),
            "right_top20_size": len(right_ids),
            "intersection_size": len(left_ids & right_ids),
            "jaccard": float(len(left_ids & right_ids) / len(union)) if union else None,
        }
    return out


def component_diagnostics(items: list[dict]) -> dict:
    out = {}
    labels = [1.0 if _is_major(item) else 0.0 for item in items]
    for name in COMPONENTS:
        values = []
        missing = 0
        clipped = 0
        for item in items:
            components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
            value = components.get(name)
            if not isinstance(value, (int, float)):
                missing += 1
                values.append(0.0)
                continue
            if value < 0.0 or value > 1.0:
                clipped += 1
            values.append(_clip(float(value)))
        out[name] = {
            "mean": _mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "missing_count": missing,
            "clipped_count": clipped,
            "correlation_with_major_correction": _correlation(values, labels),
        }
    return out


def recommendation(results: dict) -> str:
    current = results.get("current_full_priority") or {}
    risk_heavy = results.get("risk_heavy") or {}
    risk_only = results.get("risk_only") or {}
    risk_heavy_beats = (
        _num(risk_heavy.get("average_precision_for_major_correction")) > _num(current.get("average_precision_for_major_correction"))
        and _num(risk_heavy.get("lift_at_20_percent_over_random")) > _num(current.get("lift_at_20_percent_over_random"))
    )
    lift_ci = ((risk_heavy.get("bootstrap_ci") or {}).get("lift_at_20_percent_over_random") or {})
    if risk_heavy_beats and (
        (lift_ci.get("ci95_low") is not None and float(lift_ci["ci95_low"]) > _num(current.get("lift_at_20_percent_over_random")))
        or (lift_ci.get("ci95_low") is not None and float(lift_ci["ci95_low"]) > 1.0)
    ):
        return "risk_heavy is promising; validate on a second dataset or COCO300 uncertainty run before changing default."
    best = _leaderboard(results, "average_precision_for_major_correction")[0]["preset"] if results else None
    if best == "risk_only":
        risk_gap = _num(risk_only.get("average_precision_for_major_correction")) - _num(risk_heavy.get("average_precision_for_major_correction"))
        if risk_gap <= 0.03:
            return "risk dominates; consider risk-heavy fusion to preserve safety/explainability."
        return "risk-only is strongest; inspect whether other components are noisy or mis-scaled."
    if not any(
        _num(row.get("average_precision_for_major_correction")) > _num(current.get("average_precision_for_major_correction"))
        for row in results.values()
    ):
        return "keep current full priority."
    base = _num(current.get("major_correction_base_rate"))
    best_ap = max(_num(row.get("average_precision_for_major_correction")) for row in results.values()) if results else 0.0
    if best_ap <= base * 1.10:
        return "improve features or add harder datasets."
    return "validate the strongest preset on another split or dataset before changing defaults."


def render_markdown(report: dict) -> str:
    lines = [
        "# Layer 5 Weight Ablation",
        "",
        f"- input review queue: {report.get('input_review_queue')}",
        f"- n_samples: {report.get('n_samples')}",
        f"- positive_count: {report.get('positive_count')}",
        f"- base rate: {_fmt(report.get('major_correction_base_rate'))}",
        f"- bootstrap_iters: {report.get('bootstrap_iters')}",
        f"- random_seed: {report.get('random_seed')}",
        "",
        "## Preset Leaderboard",
        "### By AP",
        *_leaderboard_lines(report.get("leaderboard_by_ap") or []),
        "",
        "### By Lift@20",
        *_leaderboard_lines(report.get("leaderboard_by_lift_at_20") or []),
        "",
        "## Metrics",
        "| preset | precision@20 | recall@20 | lift@20 | AP | ROC-AUC | Brier | lift@20 CI | top20 major | meets initial standard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, row in (report.get("presets") or {}).items():
        label = name
        if name == "current_full_priority":
            label = f"**{name}**"
        if (report.get("leaderboard_by_ap") or [{}])[0].get("preset") == name:
            label = f"**{label} (best)**"
        ci = ((row.get("bootstrap_ci") or {}).get("lift_at_20_percent_over_random") or {})
        ci_text = f"{_fmt(ci.get('ci95_low'))}-{_fmt(ci.get('ci95_high'))}" if ci else "n/a"
        lines.append(
            f"| {label} | {_fmt(row.get('precision_at_20_percent'))} | {_fmt(row.get('recall_at_20_percent'))} | "
            f"{_fmt(row.get('lift_at_20_percent_over_random'))} | {_fmt(row.get('average_precision_for_major_correction'))} | "
            f"{_fmt(row.get('roc_auc_for_major_correction'))} | {_fmt(row.get('brier_score_for_major_correction'))} | "
            f"{ci_text} | {row.get('top_20_percent_major_correction_count')} | {row.get('meets_initial_standard')} |"
        )
    lines.extend([
        "",
        "## Pairwise Comparison",
        *_pairwise_lines(report.get("pairwise_comparison") or {}),
        "",
        "## Top-K Overlap",
        *_overlap_lines(report.get("topk_overlap") or {}),
        "",
        "## Component Diagnostics",
        *_component_lines(report.get("component_diagnostics") or {}),
        "",
        "## Recommendation",
        f"- {report.get('recommendation')}",
    ])
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", *(f"- {warning}" for warning in warnings)])
    return "\n".join(lines)


def _leaderboard(results: dict, metric: str) -> list[dict]:
    return [
        {"preset": name, metric: row.get(metric)}
        for name, row in sorted(results.items(), key=lambda item: -_num(item[1].get(metric)))
    ]


def _leaderboard_lines(rows: list[dict]) -> list[str]:
    return [f"- {idx}. {row.get('preset')}: {_fmt(next((v for k, v in row.items() if k != 'preset'), None))}" for idx, row in enumerate(rows, start=1)]


def _pairwise_lines(rows: dict) -> list[str]:
    return [
        f"- {name}: AP delta={_fmt(row.get('ap_delta'))}, lift@20 delta={_fmt(row.get('lift_at_20_delta'))}, "
        f"precision@20 delta={_fmt(row.get('precision_at_20_delta'))}"
        for name, row in rows.items()
    ] or ["- none"]


def _overlap_lines(rows: dict) -> list[str]:
    return [
        f"- {name}: intersection={row.get('intersection_size')}, jaccard={_fmt(row.get('jaccard'))}"
        for name, row in rows.items()
    ] or ["- none"]


def _component_lines(rows: dict) -> list[str]:
    return [
        f"- {name}: mean={_fmt(row.get('mean'))}, min={_fmt(row.get('min'))}, max={_fmt(row.get('max'))}, "
        f"missing={row.get('missing_count')}, clipped={row.get('clipped_count')}, corr={_fmt(row.get('correlation_with_major_correction'))}"
        for name, row in rows.items()
    ] or ["- none"]


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_major(item: dict) -> bool:
    delta = ((item.get("evaluation_only") or {}).get("delta") or {}) if isinstance(item, dict) else {}
    return delta.get("major_correction") is True


def _top_fraction(items: list[dict], fraction: float) -> list[dict]:
    import math

    return items[: max(1, int(math.ceil(len(items) * fraction)))] if items else []


def _top_ids(items: list[dict], fraction: float) -> set[str]:
    return {str(item.get("task_id") or item.get("sample_id") or item.get("prediction_id")) for item in _top_fraction(items, fraction)}


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _correlation(values: list[float], labels: list[float]) -> float | None:
    if not values or len(values) != len(labels) or len(set(labels)) < 2 or len(set(values)) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(values, labels)
        return float(result.correlation) if result.correlation == result.correlation else None
    except Exception:
        pass
    mean_x = sum(values) / len(values)
    mean_y = sum(labels) / len(labels)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, labels))
    denom_x = sum((x - mean_x) ** 2 for x in values) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in labels) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return float(numerator / (denom_x * denom_y))


def _binary_metrics(labels: list[int], scores: list[float]) -> dict:
    if not labels:
        return {"roc_auc": None, "brier_score": None}
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return {"roc_auc": None, "brier_score": float(brier)}
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return {"roc_auc": float(wins / total), "brier_score": float(brier)}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(values, percentile))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline Layer 5 review-weight ablation.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--presets-file")
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = ablate_review_weights(
        args.review_queue,
        args.output_dir,
        presets_file=args.presets_file,
        bootstrap_iters=args.bootstrap_iters,
        random_seed=args.random_seed,
    )
    print(
        f"[OK] weight ablation presets={len(result['presets'])} samples={result['n_samples']} "
        f"output_dir={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
