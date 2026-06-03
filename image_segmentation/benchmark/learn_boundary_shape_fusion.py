from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

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
ARTIFACT_VERSION = "learned_boundary_shape_shadow_v1"


def learn_boundary_shape_fusion(
    review_queue: str,
    output_dir: str,
    n_splits: int = 5,
    random_seed: int = 42,
    bootstrap_iters: int = 0,
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
    ci_report = None
    if bootstrap_iters:
        ci_report = bootstrap_ci_report(y.tolist(), {**baseline_scores, **predictions_by_experiment}, int(bootstrap_iters), int(random_seed))
        for name, row in experiments_report.items():
            row["bootstrap_ci"] = (ci_report.get("experiments") or {}).get(name)
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
        "bootstrap_iters": int(bootstrap_iters),
        "model_type": model_type,
        "feature_sets": experiments,
        "fold_metrics": fold_rows,
        "experiments": experiments_report,
        "bootstrap_ci": ci_report,
        "warnings": ["sklearn_unavailable_used_numpy_fallback"] if model_type.startswith("numpy") else [],
    }
    _write_json(output / "learned_boundary_shape_fusion.json", report)
    _write_json(output / "learned_fusion_results.json", report)
    _write_jsonl(output / "learned_boundary_shape_oof_predictions.jsonl", ranked_rows)
    _write_jsonl(output / "learned_fusion_oof_predictions.jsonl", ranked_rows)
    if ci_report is not None:
        _write_json(output / "learned_fusion_bootstrap_ci.json", ci_report)
        (output / "learned_fusion_bootstrap_ci.md").write_text(render_bootstrap_markdown(ci_report), encoding="utf-8")
    (output / "learned_boundary_shape_fusion.md").write_text(render_markdown(report), encoding="utf-8")
    (output / "learned_fusion_results.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def export_learned_fusion_artifacts(
    review_queue: str,
    artifact_dir: str,
    training_dataset_names: list[str] | None = None,
    random_seed: int = 42,
    oof_metrics_reference_path: str | None = None,
    benchmark_report_reference_path: str | None = None,
    training_command: str | None = None,
    validate_review_queue: str | None = None,
) -> dict:
    items = _read_jsonl(review_queue)
    labels = [target_label(item) for item in items]
    if any(label is None for label in labels):
        raise ValueError("major_correction label is required for full-data learned fusion artifact training")
    y = np.asarray([1 if label else 0 for label in labels], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("artifact training requires at least one positive and one negative sample")
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments = {
        "learned_boundary_shape_only": BOUNDARY_FEATURES,
        "learned_current_plus_boundary_shape": CURRENT_PLUS_FEATURES,
    }
    models = {}
    for name, feature_names in experiments.items():
        assert_no_leaky_feature_names(feature_names)
        x = _feature_matrix(items, feature_names)
        scaler = _fit_scaler(x)
        model = _fit_model(_apply_scaler(x, scaler), y, int(random_seed))
        models[name] = {
            "experiment": name,
            "feature_names": feature_names,
            "scaler": _json_scaler(scaler),
            "model": model,
            "model_type": _model_type(model),
        }
        filename = f"{name}_model.pkl"
        with (output / filename).open("wb") as f:
            pickle.dump(models[name], f)
    feature_schema = {
        "artifact_version": ARTIFACT_VERSION,
        "score_version": "learned_boundary_shape_shadow_v1",
        "experiments": {name: {"feature_names": features} for name, features in experiments.items()},
        "strict_schema_match": True,
        "missing_feature_policy": "neutral_zero_fill",
    }
    metadata = {
        "experimental_shadow_only": True,
        "shadow_only": True,
        "artifact_version": ARTIFACT_VERSION,
        "affects_default_ranking": False,
        "warning": "This artifact is for shadow scoring only and must not change default Layer 5 ordering.",
        "model_type": sorted({row["model_type"] for row in models.values()}),
        "training_dataset_names": training_dataset_names or [],
        "training_sample_count": len(items),
        "positive_count": int(y.sum()),
        "feature_names": experiments,
        "feature_transforms": {
            "safe_prediction_features": "numeric zero fill; pred_touches_border boolean to 0/1",
            "scaler": "robust median/IQR with clipping to [-5, 5]",
            "missing_feature_policy": "neutral_zero_fill",
        },
        "excluded_leaky_fields": sorted(
            [
                "GT mask",
                "GT mask path",
                "IoU",
                "Dice",
                "boundary IoU",
                "boundary F1",
                "boundary delta",
                "correction delta",
                "major correction label",
                "severity label",
                "dataset GT-only metadata",
                "boundary_metadata",
            ]
        ),
        "scaler_type": "robust_median_iqr",
        "sklearn_version": _sklearn_version(),
        "torch_version": _torch_version(),
        "random_seed": int(random_seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_command": training_command or " ".join(sys.argv),
        "validation_report_path": str(output / "validation_report.json"),
        "oof_metrics_reference_path": oof_metrics_reference_path,
        "benchmark_report_reference_path": benchmark_report_reference_path,
    }
    _write_json(output / "feature_schema.json", feature_schema)
    _write_json(output / "model_metadata.json", metadata)
    validation = validate_learned_fusion_artifact(str(output), validate_review_queue or review_queue, output_dir=str(output))
    return {
        "artifact_dir": str(output),
        "feature_schema": feature_schema,
        "model_metadata": metadata,
        "validation_report": validation,
        "model_files": [f"{name}_model.pkl" for name in experiments],
    }


def predict_with_artifacts(item: dict, artifact_dir: str) -> dict[str, float | None]:
    root = Path(artifact_dir)
    schema = _read_json(root / "feature_schema.json")
    experiments = schema.get("experiments") if isinstance(schema.get("experiments"), dict) else {}
    out: dict[str, float | None] = {}
    for name in ["learned_boundary_shape_only", "learned_current_plus_boundary_shape"]:
        model_path = root / f"{name}_model.pkl"
        if not model_path.exists():
            out[name] = None
            continue
        with model_path.open("rb") as f:
            bundle = pickle.load(f)
        schema_features = ((experiments.get(name) or {}).get("feature_names") or [])
        feature_names = bundle.get("feature_names") or []
        if list(schema_features) != list(feature_names):
            raise ValueError(f"feature schema mismatch for {name}")
        assert_no_leaky_feature_names(list(feature_names))
        x = _feature_matrix([item], list(feature_names))
        scaler = _scaler_from_json(bundle.get("scaler") or {})
        probs = _predict_model(bundle.get("model"), _apply_scaler(x, scaler))
        out[name] = _clip(_num(probs[0])) if len(probs) else None
    return out


def validate_learned_fusion_artifact(artifact_dir: str, review_queue: str, output_dir: str | None = None, max_rows: int = 5) -> dict:
    root = Path(artifact_dir)
    queue_items = _read_jsonl(review_queue)
    schema = _read_json(root / "feature_schema.json")
    metadata = _read_json(root / "model_metadata.json")
    checks = {
        "feature_schema_present": (root / "feature_schema.json").exists(),
        "model_metadata_present": (root / "model_metadata.json").exists(),
        "shadow_only": bool(metadata.get("experimental_shadow_only") is True and metadata.get("affects_default_ranking") is False),
        "missing_feature_policy_explicit": bool(schema.get("missing_feature_policy")),
        "models_load": True,
        "feature_schema_match": True,
        "features_prediction_time_safe": True,
        "inference_succeeds": True,
        "score_range_sane": True,
    }
    errors = []
    experiments = schema.get("experiments") if isinstance(schema.get("experiments"), dict) else {}
    for name in ["learned_boundary_shape_only", "learned_current_plus_boundary_shape"]:
        try:
            model_path = root / f"{name}_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(str(model_path))
            with model_path.open("rb") as f:
                bundle = pickle.load(f)
            schema_features = ((experiments.get(name) or {}).get("feature_names") or [])
            bundle_features = bundle.get("feature_names") or []
            if list(schema_features) != list(bundle_features):
                checks["feature_schema_match"] = False
                errors.append(f"{name}: feature schema mismatch")
            assert_no_leaky_feature_names(list(bundle_features))
        except Exception as exc:
            checks["models_load"] = False
            checks["features_prediction_time_safe"] = False
            errors.append(f"{name}: {exc}")
    scores = []
    for item in queue_items[: max(0, int(max_rows))]:
        try:
            row = predict_with_artifacts(item, artifact_dir)
            scores.extend(value for value in row.values() if value is not None)
        except Exception as exc:
            checks["inference_succeeds"] = False
            errors.append(f"inference: {exc}")
            break
    if any(value < 0.0 or value > 1.0 for value in scores):
        checks["score_range_sane"] = False
        errors.append("inference score outside [0, 1]")
    passed = all(checks.values())
    report = {
        "artifact_dir": artifact_dir,
        "review_queue": review_queue,
        "artifact_version": metadata.get("artifact_version"),
        "validation_sample_count": min(len(queue_items), max(0, int(max_rows))),
        "checks": checks,
        "errors": errors,
        "passed": passed,
    }
    target = Path(output_dir) if output_dir else root
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "validation_report.json", report)
    (target / "validation_report.md").write_text(render_artifact_validation_markdown(report), encoding="utf-8")
    if not passed:
        raise ValueError(f"learned fusion artifact validation failed: {errors[:3]}")
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
    model = _fit_model(x_train, y_train, seed)
    return _predict_model(model, x_eval)


def _fit_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        return model
    except Exception:
        return {"type": "numpy_logistic_regression", "weights": _numpy_logistic_weights(x_train, y_train).tolist()}


def _predict_model(model: Any, x_eval: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_eval)[:, 1]
    if isinstance(model, dict) and model.get("type") == "numpy_logistic_regression":
        weights = np.asarray(model.get("weights") or [], dtype=float)
        x_eval_bias = np.c_[np.ones(len(x_eval)), x_eval]
        return 1.0 / (1.0 + np.exp(-np.clip(x_eval_bias @ weights, -30, 30)))
    raise ValueError("unsupported learned fusion model artifact")


def _numpy_logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    weights = _numpy_logistic_weights(x_train, y_train)
    x_eval_bias = np.c_[np.ones(len(x_eval)), x_eval]
    return 1.0 / (1.0 + np.exp(-np.clip(x_eval_bias @ weights, -30, 30)))


def _numpy_logistic_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x = np.c_[np.ones(len(x_train)), x_train]
    weights = np.zeros(x.shape[1], dtype=float)
    lr = 0.1
    reg = 0.05
    for _ in range(300):
        pred = 1.0 / (1.0 + np.exp(-np.clip(x @ weights, -30, 30)))
        grad = (x.T @ (pred - y_train)) / max(1, len(y_train))
        grad[1:] += reg * weights[1:]
        weights -= lr * grad
    return weights


def _fit_scaler(x: np.ndarray) -> dict:
    median = np.median(x, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)
    scale = p75 - p25
    scale[scale < 1e-9] = 1.0
    return {"median": median, "scale": scale}


def _apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return np.clip((x - scaler["median"]) / scaler["scale"], -5.0, 5.0)


def _json_scaler(scaler: dict) -> dict:
    return {
        "median": np.asarray(scaler["median"], dtype=float).tolist(),
        "scale": np.asarray(scaler["scale"], dtype=float).tolist(),
    }


def _scaler_from_json(scaler: dict) -> dict:
    return {
        "median": np.asarray(scaler.get("median") or [], dtype=float),
        "scale": np.asarray(scaler.get("scale") or [], dtype=float),
    }


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


def bootstrap_ci_report(labels: list[int], scores_by_experiment: dict[str, list[float]], bootstrap_iters: int, random_seed: int) -> dict:
    metrics = ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]
    if len(set(labels)) < 2:
        empty = {metric: {"value": None, "ci95_low": None, "ci95_high": None, "warning": "degenerate_labels"} for metric in metrics}
        return {
            "bootstrap_iters": int(bootstrap_iters),
            "random_seed": int(random_seed),
            "metrics": metrics,
            "experiments": {name: empty for name in scores_by_experiment},
            "pairwise_delta_vs_current": {},
        }
    experiments = {
        name: _bootstrap_metric_ci(labels, scores, name, bootstrap_iters, random_seed)
        for name, scores in scores_by_experiment.items()
    }
    pairwise = {}
    current = scores_by_experiment.get("current_full_priority")
    if current is not None:
        for name, scores in scores_by_experiment.items():
            if name == "current_full_priority":
                continue
            pairwise[f"{name}_vs_current_full_priority"] = _bootstrap_delta_ci(
                labels,
                scores,
                current,
                name,
                bootstrap_iters,
                random_seed,
                metrics,
            )
    return {
        "bootstrap_iters": int(bootstrap_iters),
        "random_seed": int(random_seed),
        "metrics": metrics,
        "experiments": experiments,
        "pairwise_delta_vs_current": pairwise,
    }


def _bootstrap_metric_ci(labels: list[int], scores: list[float], name: str, bootstrap_iters: int, random_seed: int) -> dict:
    metric_names = ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]
    rng = random.Random(random_seed + sum(ord(ch) for ch in name))
    values = {metric: [] for metric in metric_names}
    for _ in range(max(0, int(bootstrap_iters))):
        idxs = [rng.randrange(len(labels)) for _ in labels]
        row = _metrics([labels[idx] for idx in idxs], [scores[idx] for idx in idxs])
        for metric in metric_names:
            if row.get(metric) is not None:
                values[metric].append(float(row[metric]))
    current = _metrics(labels, scores)
    return {
        metric: {
            "value": current.get(metric),
            "ci95_low": _percentile(sorted(values[metric]), 2.5) if values[metric] else None,
            "ci95_high": _percentile(sorted(values[metric]), 97.5) if values[metric] else None,
        }
        for metric in metric_names
    }


def _bootstrap_delta_ci(
    labels: list[int],
    left_scores: list[float],
    right_scores: list[float],
    name: str,
    bootstrap_iters: int,
    random_seed: int,
    metric_names: list[str],
) -> dict:
    rng = random.Random(random_seed + sum(ord(ch) for ch in name + "_delta"))
    values = {metric: [] for metric in metric_names}
    for _ in range(max(0, int(bootstrap_iters))):
        idxs = [rng.randrange(len(labels)) for _ in labels]
        sampled_labels = [labels[idx] for idx in idxs]
        left = _metrics(sampled_labels, [left_scores[idx] for idx in idxs])
        right = _metrics(sampled_labels, [right_scores[idx] for idx in idxs])
        for metric in metric_names:
            if left.get(metric) is not None and right.get(metric) is not None:
                values[metric].append(float(left[metric]) - float(right[metric]))
    current_left = _metrics(labels, left_scores)
    current_right = _metrics(labels, right_scores)
    out = {}
    for metric in metric_names:
        samples = sorted(values[metric])
        value = (
            float(current_left[metric]) - float(current_right[metric])
            if current_left.get(metric) is not None and current_right.get(metric) is not None
            else None
        )
        out[metric] = {
            "value": value,
            "ci95_low": _percentile(samples, 2.5) if samples else None,
            "ci95_high": _percentile(samples, 97.5) if samples else None,
        }
        if not samples:
            out[metric]["warning"] = "bootstrap_metric_degenerate"
    return out


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


def render_bootstrap_markdown(report: dict) -> str:
    lines = [
        "# Learned Fusion Bootstrap Confidence Intervals",
        "",
        f"- bootstrap_iters: {report.get('bootstrap_iters')}",
        f"- random_seed: {report.get('random_seed')}",
        "",
        "## Experiments",
        "| experiment | metric | value | ci95 low | ci95 high |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for experiment, metrics in (report.get("experiments") or {}).items():
        for metric, row in (metrics or {}).items():
            lines.append(
                f"| {experiment} | {metric} | {_fmt(row.get('value'))} | {_fmt(row.get('ci95_low'))} | {_fmt(row.get('ci95_high'))} |"
            )
    lines.extend(
        [
            "",
            "## Pairwise Delta Vs Current",
            "| comparison | metric | delta | ci95 low | ci95 high |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for comparison, metrics in (report.get("pairwise_delta_vs_current") or {}).items():
        for metric, row in (metrics or {}).items():
            lines.append(
                f"| {comparison} | {metric} | {_fmt(row.get('value'))} | {_fmt(row.get('ci95_low'))} | {_fmt(row.get('ci95_high'))} |"
            )
    return "\n".join(lines)


def render_artifact_validation_markdown(report: dict) -> str:
    lines = [
        "# Learned Fusion Artifact Validation",
        "",
        f"- artifact_dir: {report.get('artifact_dir')}",
        f"- artifact_version: {report.get('artifact_version')}",
        f"- review_queue: {report.get('review_queue')}",
        f"- passed: {report.get('passed')}",
        "",
        "| check | passed |",
        "| --- | --- |",
    ]
    for name, passed in (report.get("checks") or {}).items():
        lines.append(f"| {name} | {passed} |")
    errors = report.get("errors") or []
    if errors:
        lines.extend(["", "## Errors", *(f"- {error}" for error in errors)])
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


def _sklearn_version() -> str | None:
    try:
        import sklearn

        return str(sklearn.__version__)
    except Exception:
        return None


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return None


def _model_type(model: Any) -> str:
    if hasattr(model, "predict_proba"):
        return "sklearn_logistic_regression"
    if isinstance(model, dict):
        return str(model.get("type") or "unknown")
    return type(model).__name__


def _row_id(row: dict, fallback: int) -> str:
    return str(row.get("task_id") or row.get("sample_id") or row.get("prediction_id") or fallback)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, percentile))


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
    parser.add_argument("--review-queue")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--export-artifact")
    parser.add_argument("--training-dataset-name", action="append", default=[])
    parser.add_argument("--oof-metrics-reference-path")
    parser.add_argument("--benchmark-report-reference-path")
    parser.add_argument("--validate-artifact")
    parser.add_argument("--validation-output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_artifact:
        if not args.review_queue:
            raise SystemExit("--review-queue is required with --validate-artifact")
        report = validate_learned_fusion_artifact(args.validate_artifact, args.review_queue, output_dir=args.validation_output_dir)
        print(f"[OK] artifact validation passed={report['passed']} artifact_dir={args.validate_artifact}")
        return 0
    if not args.review_queue or not args.output_dir:
        raise SystemExit("--review-queue and --output-dir are required unless --validate-artifact is used")
    result = learn_boundary_shape_fusion(
        args.review_queue,
        args.output_dir,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        bootstrap_iters=args.bootstrap,
    )
    artifact_dir = args.export_artifact or args.artifact_dir
    if artifact_dir:
        export_learned_fusion_artifacts(
            args.review_queue,
            artifact_dir,
            training_dataset_names=args.training_dataset_name,
            random_seed=args.random_seed,
            oof_metrics_reference_path=args.oof_metrics_reference_path or str(Path(args.output_dir) / "learned_fusion_results.json"),
            benchmark_report_reference_path=args.benchmark_report_reference_path,
        )
    print(f"[OK] learned boundary fusion samples={result['n_samples']} folds={result['n_splits']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
