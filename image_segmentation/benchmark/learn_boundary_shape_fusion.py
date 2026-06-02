from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.ablate_review_weights import DEFAULT_PRESETS, rank_for_preset
from image_segmentation.benchmark.prediction_feature_scoring import (
    PREDICTION_FEATURE_NAMES,
    assert_no_leaky_feature_names,
    rank_boundary_shape_scores,
    safe_prediction_feature_vector,
    target_label,
)


BOUNDARY_FEATURES = list(PREDICTION_FEATURE_NAMES) + [
    "boundary_shape_score",
    "boundary_shape_calibrated_score",
]
CURRENT_PLUS_FEATURES = [
    "current_priority_score",
    "correction_risk_score",
    "uncertainty_score",
    "rule_review_score",
    "geometry_complexity_score",
    "boundary_shape_score",
    "boundary_shape_calibrated_score",
]


def learn_boundary_shape_fusion(
    review_queue: str,
    output_dir: str,
    n_splits: int = 5,
    random_seed: int = 42,
) -> dict:
    items = _read_jsonl(review_queue)
    labels = [target_label(item) for item in items]
    if any(label is None for label in labels):
        raise ValueError("major_correction label is required for OOF learned fusion")
    y = np.asarray([1 if label else 0 for label in labels], dtype=int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    usable_splits = min(int(n_splits), positives, negatives) if positives and negatives else 0
    if usable_splits < 2:
        raise ValueError("not enough positive/negative samples for OOF learned fusion")
    folds = _stratified_fold_indices(y.tolist(), usable_splits, random_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments = {
        "learned_boundary_shape_only": BOUNDARY_FEATURES,
        "learned_current_plus_boundary_shape": CURRENT_PLUS_FEATURES,
    }
    predictions_by_experiment: dict[str, list[float]] = {name: [0.0] * len(items) for name in experiments}
    fold_rows = []
    model_type = "sklearn_logistic_regression" if _sklearn_available() else "numpy_logistic_regression"
    for fold_idx, eval_idx in enumerate(folds):
        train_idx = [idx for idx in range(len(items)) if idx not in set(eval_idx)]
        for name, feature_names in experiments.items():
            assert_no_leaky_feature_names(feature_names)
            x_train = _feature_matrix([items[idx] for idx in train_idx], feature_names)
            x_eval = _feature_matrix([items[idx] for idx in eval_idx], feature_names)
            scaler = _fit_scaler(x_train)
            x_train_scaled = _apply_scaler(x_train, scaler)
            x_eval_scaled = _apply_scaler(x_eval, scaler)
            probs = _fit_predict(x_train_scaled, y[train_idx], x_eval_scaled, random_seed + fold_idx)
            for local, idx in enumerate(eval_idx):
                predictions_by_experiment[name][idx] = float(probs[local])
            metrics = _metrics(y[eval_idx].tolist(), probs.tolist())
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "experiment": name,
                    "n_train": len(train_idx),
                    "n_eval": len(eval_idx),
                    "positives": int(y[eval_idx].sum()),
                    "feature_names": feature_names,
                    **metrics,
                }
            )
    baseline_scores = _baseline_scores(items)
    experiments_report = {}
    for name, scores in {**baseline_scores, **predictions_by_experiment}.items():
        experiments_report[name] = {
            "oof": name.startswith("learned_"),
            **_metrics(y.tolist(), scores),
        }
    ranked_rows = []
    for idx, item in enumerate(items):
        row = {"task_id": item.get("task_id"), "sample_id": item.get("sample_id"), "label": int(y[idx])}
        for name, scores in predictions_by_experiment.items():
            row[name] = scores[idx]
        ranked_rows.append(row)
    report = {
        "input_review_queue": review_queue,
        "n_samples": len(items),
        "positive_count": positives,
        "n_splits": usable_splits,
        "random_seed": int(random_seed),
        "model_type": model_type,
        "feature_sets": experiments,
        "fold_metrics": fold_rows,
        "experiments": experiments_report,
        "warnings": ["sklearn_unavailable_used_numpy_fallback"] if model_type.startswith("numpy") else [],
    }
    _write_json(output / "learned_boundary_shape_fusion.json", report)
    _write_jsonl(output / "learned_boundary_shape_oof_predictions.jsonl", ranked_rows)
    (output / "learned_boundary_shape_fusion.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _baseline_scores(items: list[dict]) -> dict[str, list[float]]:
    out = {"current_full_priority": [_clip(_num(item.get("priority_score"))) for item in items]}
    for name in [
        "risk_only",
        "risk_heavy",
        "boundary_shape_experimental",
        "boundary_shape_rank_score",
        "boundary_shape_calibrated_score",
    ]:
        ranked, _ = rank_for_preset(items, name, DEFAULT_PRESETS[name])
        scores_by_id = {_row_id(row, idx): _clip(_num(row.get("ablation_priority_score"))) for idx, row in enumerate(ranked)}
        out[name] = [scores_by_id.get(_row_id(item, idx), 0.0) for idx, item in enumerate(items)]
    return out


def _feature_matrix(items: list[dict], feature_names: list[str]) -> np.ndarray:
    rank_scores = rank_boundary_shape_scores(items)
    rows = []
    for idx, item in enumerate(items):
        components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
        vector = safe_prediction_feature_vector(item)
        vector.update(
            {
                "boundary_shape_rank_score": rank_scores[idx] if idx < len(rank_scores) else 0.0,
                "current_priority_score": _clip(_num(item.get("priority_score"))),
                "correction_risk_score": _clip(_num(components.get("correction_risk_score"))),
                "uncertainty_score": _clip(_num(components.get("uncertainty_score"))),
                "rule_review_score": _clip(_num(components.get("rule_review_score"))),
                "geometry_complexity_score": _clip(_num(components.get("geometry_complexity_score"))),
            }
        )
        rows.append([float(vector.get(name, 0.0)) for name in feature_names])
    return np.asarray(rows, dtype=float)


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        return model.predict_proba(x_eval)[:, 1]
    except Exception:
        return _numpy_logistic_predict(x_train, y_train, x_eval)


def _numpy_logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones(len(x_train)), x_train]
    x_eval_bias = np.c_[np.ones(len(x_eval)), x_eval]
    weights = np.zeros(x.shape[1], dtype=float)
    lr = 0.1
    reg = 0.05
    for _ in range(300):
        pred = 1.0 / (1.0 + np.exp(-np.clip(x @ weights, -30, 30)))
        grad = (x.T @ (pred - y_train)) / max(1, len(y_train))
        grad[1:] += reg * weights[1:]
        weights -= lr * grad
    return 1.0 / (1.0 + np.exp(-np.clip(x_eval_bias @ weights, -30, 30)))


def _fit_scaler(x: np.ndarray) -> dict:
    median = np.median(x, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)
    scale = p75 - p25
    scale[scale < 1e-9] = 1.0
    return {"median": median, "scale": scale}


def _apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return np.clip((x - scaler["median"]) / scaler["scale"], -5.0, 5.0)


def _metrics(labels: list[int], scores: list[float]) -> dict:
    ranked = sorted(zip(labels, scores), key=lambda pair: -pair[1])
    positives = sum(labels)
    k20 = max(1, int(np.ceil(len(labels) * 0.20))) if labels else 0
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
        "# OOF Learned Boundary/Shape Fusion",
        "",
        f"- input review queue: {report.get('input_review_queue')}",
        f"- n_samples: {report.get('n_samples')}",
        f"- positive_count: {report.get('positive_count')}",
        f"- n_splits: {report.get('n_splits')}",
        f"- model_type: {report.get('model_type')}",
        "",
        "| experiment | OOF | AP | ROC-AUC | Brier | lift@20 | precision@20 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in (report.get("experiments") or {}).items():
        lines.append(
            f"| {name} | {row.get('oof')} | {_fmt(row.get('average_precision'))} | {_fmt(row.get('roc_auc'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('lift_at_20'))} | {_fmt(row.get('precision_at_20'))} |"
        )
    return "\n".join(lines)


def _stratified_fold_indices(labels: list[int], n_splits: int, random_seed: int) -> list[list[int]]:
    pos = [idx for idx, value in enumerate(labels) if value]
    neg = [idx for idx, value in enumerate(labels) if not value]
    rng = random.Random(random_seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(n_splits)]
    for index, value in enumerate(pos):
        folds[index % n_splits].append(value)
    for index, value in enumerate(neg):
        folds[index % n_splits].append(value)
    return [sorted(fold) for fold in folds]


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


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _row_id(row: dict, fallback: int) -> str:
    return str(row.get("task_id") or row.get("sample_id") or row.get("prediction_id") or fallback)


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
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-safe OOF learned boundary/shape fusion.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = learn_boundary_shape_fusion(
        args.review_queue,
        args.output_dir,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )
    print(f"[OK] learned boundary fusion samples={result['n_samples']} folds={result['n_splits']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
