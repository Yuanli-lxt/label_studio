from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.prediction_feature_scoring import (
    PREDICTION_FEATURE_NAMES,
    calibrated_boundary_shape_score,
    prediction_features,
    rank_boundary_shape_scores,
    target_label,
)


def diagnose_prediction_features(review_queue: str, output_dir: str, bins: int = 5) -> dict:
    items = _read_jsonl(review_queue)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = [target_label(item) for item in items]
    label_available = all(label is not None for label in labels) and bool(labels)
    y = [1 if label else 0 for label in labels] if label_available else None
    feature_names = list(PREDICTION_FEATURE_NAMES) + [
        "boundary_shape_score",
        "boundary_shape_rank_score",
        "boundary_shape_calibrated_score",
    ]
    rank_scores = rank_boundary_shape_scores(items)
    diagnostics = {}
    bin_rows = []
    for name in feature_names:
        values, missing = _values_for_feature(items, name, rank_scores)
        row = _feature_summary(name, values, missing, y, bins=bins)
        diagnostics[name] = row
        for bin_row in row.get("bins") or []:
            bin_rows.append({"feature": name, **bin_row})
    report = {
        "input_review_queue": review_queue,
        "n_samples": len(items),
        "label": "major_correction",
        "label_available": label_available,
        "positive_count": int(sum(y)) if y is not None else None,
        "features": diagnostics,
        "warnings": [] if label_available else ["missing_major_correction_label; skipped label-dependent diagnostics"],
    }
    _write_json(output / "prediction_feature_diagnostics.json", report)
    _write_jsonl(output / "prediction_feature_bins.jsonl", bin_rows)
    (output / "prediction_feature_diagnostics.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _values_for_feature(items: list[dict], name: str, rank_scores: list[float]) -> tuple[list[float | None], int]:
    values: list[float | None] = []
    missing = 0
    for index, item in enumerate(items):
        if name == "boundary_shape_score":
            value = ((item.get("score_components") or {}).get("boundary_shape_score")) if isinstance(item.get("score_components"), dict) else None
        elif name == "boundary_shape_rank_score":
            value = rank_scores[index] if index < len(rank_scores) else None
        elif name == "boundary_shape_calibrated_score":
            value = ((item.get("score_components") or {}).get("boundary_shape_calibrated_score")) if isinstance(item.get("score_components"), dict) else None
            if value is None:
                value = calibrated_boundary_shape_score(item)
        else:
            value = prediction_features(item).get(name)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
        else:
            missing += 1
            values.append(None)
    return values, missing


def _feature_summary(name: str, values: list[float | None], missing: int, labels: list[int] | None, bins: int) -> dict:
    numeric = [float(value) for value in values if value is not None]
    filled = [float(value) if value is not None else 0.0 for value in values]
    unique_count = len(set(numeric))
    summary = {
        "missing_count": missing,
        "missing_rate": float(missing / len(values)) if values else None,
        "unique_count": unique_count,
        "constant": bool(unique_count <= 1),
        "near_constant": bool(unique_count <= 2 or (np.std(numeric) < 1e-9 if numeric else True)),
        "min": _percentile(numeric, 0),
        "p01": _percentile(numeric, 1),
        "p05": _percentile(numeric, 5),
        "p25": _percentile(numeric, 25),
        "median": _percentile(numeric, 50),
        "p75": _percentile(numeric, 75),
        "p95": _percentile(numeric, 95),
        "p99": _percentile(numeric, 99),
        "max": _percentile(numeric, 100),
        "mean": float(np.mean(numeric)) if numeric else None,
        "std": float(np.std(numeric)) if numeric else None,
    }
    if labels is None or len(set(labels)) < 2 or unique_count < 2:
        summary.update(
            {
                "average_precision": None,
                "roc_auc": None,
                "pr_auc": None,
                "spearman": None,
                "top_quantile_major_correction_rate": None,
                "bottom_quantile_major_correction_rate": None,
                "direction_suggestion": "weak_or_no_signal",
                "bins": [],
            }
        )
        return summary
    corr = _spearman(filled, labels)
    ap_high = _average_precision(labels, filled)
    ap_low = _average_precision(labels, [-value for value in filled])
    roc_high = _roc_auc(labels, filled)
    roc_low = _roc_auc(labels, [-value for value in filled])
    feature_bins = _bins(values, labels, bins)
    top_rate, bottom_rate = _top_bottom_rates(filled, labels)
    best_ap = max(ap_high or 0.0, ap_low or 0.0)
    direction = _direction(corr, ap_high, ap_low, feature_bins)
    summary.update(
        {
            "average_precision": best_ap,
            "roc_auc": max(roc_high or 0.0, roc_low or 0.0),
            "pr_auc": best_ap,
            "spearman": corr,
            "top_quantile_major_correction_rate": top_rate,
            "bottom_quantile_major_correction_rate": bottom_rate,
            "direction_suggestion": direction,
            "bins": feature_bins,
        }
    )
    return summary


def render_markdown(report: dict) -> str:
    lines = [
        "# Prediction-Time Boundary/Shape Feature Diagnostics",
        "",
        f"- input review queue: {report.get('input_review_queue')}",
        f"- n_samples: {report.get('n_samples')}",
        f"- positive_count: {report.get('positive_count')}",
        f"- label_available: {report.get('label_available')}",
        "",
        "| feature | missing | mean | p05 | median | p95 | AP | ROC-AUC | Spearman | direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in (report.get("features") or {}).items():
        lines.append(
            f"| {name} | {row.get('missing_count')} | {_fmt(row.get('mean'))} | {_fmt(row.get('p05'))} | "
            f"{_fmt(row.get('median'))} | {_fmt(row.get('p95'))} | {_fmt(row.get('average_precision'))} | "
            f"{_fmt(row.get('roc_auc'))} | {_fmt(row.get('spearman'))} | {row.get('direction_suggestion')} |"
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", *(f"- {warning}" for warning in warnings)])
    return "\n".join(lines)


def _bins(values: list[float | None], labels: list[int], count: int) -> list[dict]:
    pairs = [(float(value), label) for value, label in zip(values, labels) if value is not None]
    if not pairs:
        return []
    pairs.sort(key=lambda pair: pair[0])
    out = []
    for idx, chunk in enumerate(np.array_split(np.asarray(pairs, dtype=float), max(1, int(count))), start=1):
        if len(chunk) == 0:
            continue
        ys = chunk[:, 1]
        xs = chunk[:, 0]
        out.append(
            {
                "bin": idx,
                "min": float(np.min(xs)),
                "max": float(np.max(xs)),
                "n": int(len(chunk)),
                "major_correction_count": int(np.sum(ys)),
                "major_correction_rate": float(np.mean(ys)),
            }
        )
    return out


def _direction(corr: float | None, ap_high: float | None, ap_low: float | None, bins: list[dict]) -> str:
    if corr is None or max(ap_high or 0.0, ap_low or 0.0) <= 0:
        return "weak_or_no_signal"
    rates = [row.get("major_correction_rate") for row in bins if row.get("major_correction_rate") is not None]
    monotonic_up = all(rates[idx] <= rates[idx + 1] + 1e-12 for idx in range(len(rates) - 1)) if len(rates) > 1 else False
    monotonic_down = all(rates[idx] + 1e-12 >= rates[idx + 1] for idx in range(len(rates) - 1)) if len(rates) > 1 else False
    if abs(corr) < 0.05 and abs((ap_high or 0.0) - (ap_low or 0.0)) < 0.02:
        return "weak_or_no_signal"
    if monotonic_up or ((ap_high or 0.0) >= (ap_low or 0.0) and corr > 0):
        return "higher_is_risk"
    if monotonic_down or ((ap_low or 0.0) > (ap_high or 0.0) and corr < 0):
        return "lower_is_risk"
    return "non_monotonic"


def _top_bottom_rates(values: list[float], labels: list[int], fraction: float = 0.20) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    k = max(1, int(np.ceil(len(values) * fraction)))
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    bottom = [labels[idx] for idx in order[:k]]
    top = [labels[idx] for idx in order[-k:]]
    return float(np.mean(top)), float(np.mean(bottom))


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


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return float(wins / total)


def _spearman(values: list[float], labels: list[int]) -> float | None:
    if len(set(values)) < 2 or len(set(labels)) < 2:
        return None
    rx = _ranks(values)
    ry = _ranks([float(v) for v in labels])
    return _corr(rx, ry)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _corr(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose prediction-time boundary/shape feature signal.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = diagnose_prediction_features(args.review_queue, args.output_dir, bins=args.bins)
    print(f"[OK] prediction feature diagnostics features={len(result['features'])} samples={result['n_samples']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
