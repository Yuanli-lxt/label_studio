from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.learn_boundary_shape_fusion import (
    bootstrap_ci_report,
    render_bootstrap_markdown,
)
from image_segmentation.benchmark.prediction_feature_scoring import (
    assert_no_leaky_feature_names,
    rank_boundary_shape_scores,
    target_label,
)
from image_segmentation.benchmark.shadow_scoring import (
    boundary_shape_gate,
    gated_boundary_shape_score,
    gated_current_boundary_score,
)


GATED_EXPERIMENTS = [
    "current_full_priority",
    "boundary_shape_calibrated_score",
    "boundary_shape_rank_score",
    "gated_boundary_shape_score",
    "gated_current_boundary_score",
]


def evaluate_gated_fusion(
    review_queue: str,
    output_dir: str,
    n_splits: int = 5,
    random_seed: int = 42,
    bootstrap_iters: int = 0,
) -> dict:
    assert_no_leaky_feature_names(
        [
            "current_priority_score",
            "boundary_shape_calibrated_score",
            "boundary_shape_rank_score",
            "correction_risk_score",
            "uncertainty_score",
            "geometry_complexity_score",
        ]
    )
    items = _read_jsonl(review_queue)
    labels = [target_label(item) for item in items]
    if any(label is None for label in labels):
        raise ValueError("major_correction label is required for gated fusion evaluation")
    y = [1 if label else 0 for label in labels]
    rank_scores = rank_boundary_shape_scores(items)
    scores = {name: [] for name in GATED_EXPERIMENTS}
    gate_values = []
    prediction_rows = []
    for idx, item in enumerate(items):
        rank_score = rank_scores[idx] if idx < len(rank_scores) else 0.5
        calibrated = _calibrated(item)
        gate = boundary_shape_gate(item, calibrated)
        gate_values.append(gate)
        row_scores = {
            "current_full_priority": _clip(_num(item.get("priority_score"))),
            "boundary_shape_calibrated_score": calibrated,
            "boundary_shape_rank_score": _clip(_num(rank_score)),
            "gated_boundary_shape_score": gated_boundary_shape_score(item, rank_score),
            "gated_current_boundary_score": gated_current_boundary_score(item, calibrated=calibrated),
        }
        for name, value in row_scores.items():
            scores[name].append(value)
        prediction_rows.append(
            {
                "task_id": item.get("task_id"),
                "sample_id": item.get("sample_id"),
                "label": y[idx],
                "boundary_shape_gate": gate,
                **row_scores,
            }
        )
    experiments = {name: {"oof": True, **_metrics(y, values)} for name, values in scores.items()}
    ci_report = None
    if bootstrap_iters:
        ci_report = bootstrap_ci_report(y, scores, int(bootstrap_iters), int(random_seed))
        for name, row in experiments.items():
            row["bootstrap_ci"] = (ci_report.get("experiments") or {}).get(name)
    gate_summary = gate_distribution(items, y, gate_values)
    report = {
        "input_review_queue": review_queue,
        "n_samples": len(items),
        "positive_count": int(sum(y)),
        "n_splits": int(n_splits),
        "random_seed": int(random_seed),
        "bootstrap_iters": int(bootstrap_iters),
        "shadow_only": True,
        "leakage_guard": "gate reads prediction-time boundary/shape scores and existing safe score components only",
        "experiments": experiments,
        "gate_distribution": gate_summary,
        "bootstrap_ci": ci_report,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "gated_fusion_results.json", report)
    _write_json(output / "gated_boundary_shape_fusion.json", report)
    _write_jsonl(output / "gated_fusion_predictions.jsonl", prediction_rows)
    if ci_report is not None:
        _write_json(output / "gated_fusion_bootstrap_ci.json", ci_report)
        (output / "gated_fusion_bootstrap_ci.md").write_text(render_bootstrap_markdown(ci_report), encoding="utf-8")
    (output / "gated_fusion_results.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def gate_distribution(items: list[dict], labels: list[int], gates: list[float]) -> dict:
    datasets = sorted({str(item.get("dataset") or "unknown") for item in items})
    out = {
        "overall": _gate_stats(labels, gates),
        "by_dataset": {},
    }
    for dataset in datasets:
        idxs = [idx for idx, item in enumerate(items) if str(item.get("dataset") or "unknown") == dataset]
        out["by_dataset"][dataset] = _gate_stats([labels[idx] for idx in idxs], [gates[idx] for idx in idxs])
    return out


def _gate_stats(labels: list[int], gates: list[float]) -> dict:
    values = list(gates)
    high = [idx for idx, value in enumerate(values) if value >= 0.5]
    return {
        "min": _percentile(values, 0),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "max": _percentile(values, 100),
        "high_gate_sample_count": len(high),
        "high_gate_major_rate": float(sum(labels[idx] for idx in high) / len(high)) if high else None,
    }


def _calibrated(item: dict) -> float:
    components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
    value = components.get("boundary_shape_calibrated_score")
    if isinstance(value, (int, float)):
        return _clip(float(value))
    shadow = item.get("shadow_scores") if isinstance(item.get("shadow_scores"), dict) else {}
    value = shadow.get("boundary_shape_calibrated_score")
    return _clip(_num(value))


def _metrics(labels: list[int], scores: list[float]) -> dict:
    positives = sum(labels)
    k20 = max(1, int(np.ceil(len(labels) * 0.20))) if labels else 0
    ranked = sorted(zip(labels, scores), key=lambda pair: -pair[1])
    precision20 = float(sum(label for label, _ in ranked[:k20]) / k20) if k20 else None
    base_rate = float(positives / len(labels)) if labels else None
    return {
        "positives": int(positives),
        "average_precision": _average_precision(labels, scores),
        "roc_auc": _roc_auc(labels, scores),
        "brier": float(np.mean([(score - label) ** 2 for label, score in zip(labels, scores)])) if labels else None,
        "lift_at_20": float(precision20 / base_rate) if precision20 is not None and base_rate else None,
        "precision_at_20": precision20,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Gated Boundary/Shape Fusion",
        "",
        f"- input review queue: {report.get('input_review_queue')}",
        f"- n_samples: {report.get('n_samples')}",
        f"- positive_count: {report.get('positive_count')}",
        "- shadow_only: True",
        "",
        "| experiment | OOF | AP | ROC-AUC | Brier | lift@20 | precision@20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in (report.get("experiments") or {}).items():
        lines.append(
            f"| {name} | {row.get('oof')} | {_fmt(row.get('average_precision'))} | {_fmt(row.get('roc_auc'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('lift_at_20'))} | {_fmt(row.get('precision_at_20'))} |"
        )
    overall = (report.get("gate_distribution") or {}).get("overall") or {}
    lines.extend(
        [
            "",
            "## Gate Distribution",
            f"- min/p25/median/p75/max: {_fmt(overall.get('min'))} / {_fmt(overall.get('p25'))} / {_fmt(overall.get('median'))} / {_fmt(overall.get('p75'))} / {_fmt(overall.get('max'))}",
            f"- high_gate_sample_count: {overall.get('high_gate_sample_count')}",
            f"- high_gate_major_rate: {_fmt(overall.get('high_gate_major_rate'))}",
        ]
    )
    return "\n".join(lines)


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
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return float(wins / (len(positives) * len(negatives)))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, percentile))


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    parser = argparse.ArgumentParser(description="Evaluate shadow-only gated boundary/shape fusion.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_gated_fusion(
        args.review_queue,
        args.output_dir,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        bootstrap_iters=args.bootstrap,
    )
    print(f"[OK] gated fusion samples={result['n_samples']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
