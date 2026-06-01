from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.build_review_queue_from_delta import build_review_queue_from_delta
from image_segmentation.benchmark.compare_review_strategies import compare_review_strategies
from image_segmentation.benchmark.predict_risk_scores import predict_risk_scores
from image_segmentation.benchmark.train_risk_model import train_from_delta_dataset


def crossfit_risk_evaluation(
    delta_dataset: str,
    output_dir: str,
    n_splits: int = 5,
    random_seed: int = 42,
    bootstrap_iters: int = 0,
) -> dict:
    rows = _read_jsonl(delta_dataset)
    output = Path(output_dir)
    folds_dir = output / "folds"
    models_dir = output / "models"
    folds_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    labels = [_is_major(row) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    requested_splits = int(n_splits)
    usable_splits = min(requested_splits, positives, negatives) if positives and negatives else 0
    if usable_splits < requested_splits:
        warnings.append(f"reduced_n_splits_from_{requested_splits}_to_{usable_splits}_because_of_class_counts")
    if usable_splits < 2:
        raise ValueError("not enough positive/negative samples for crossfit; use a larger dataset or smaller n_splits")

    fold_indices = _stratified_fold_indices(labels, usable_splits, int(random_seed))
    fold_metrics = []
    all_predictions = []
    assignments = {}
    for fold_idx, eval_idx in enumerate(fold_indices):
        eval_set = set(eval_idx)
        train_idx = [idx for idx in range(len(rows)) if idx not in eval_set]
        train_rows = [rows[idx] for idx in train_idx]
        eval_rows = [rows[idx] for idx in eval_idx]
        model_dir = models_dir / f"fold_{fold_idx}"
        train_path = folds_dir / f"fold_{fold_idx}_train_delta_dataset.jsonl"
        eval_path = folds_dir / f"fold_{fold_idx}_eval_delta_dataset.jsonl"
        pred_path = folds_dir / f"fold_{fold_idx}_delta_with_risk.jsonl"
        _write_jsonl(train_path, train_rows)
        _write_jsonl(eval_path, eval_rows)
        train_from_delta_dataset(str(train_path), str(model_dir), test_size=0.2, random_seed=random_seed)
        predict_risk_scores(str(eval_path), str(model_dir), str(pred_path))
        predictions = _read_jsonl(str(pred_path))
        all_predictions.extend(predictions)
        for idx in eval_idx:
            assignments[_record_id(rows[idx], idx)] = {
                "fold": fold_idx,
                "train_ids": [_record_id(rows[j], j) for j in train_idx],
                "eval_ids": [_record_id(rows[j], j) for j in eval_idx],
            }
        fold_metrics.append(_fold_metric_row(fold_idx, train_rows, eval_rows, predictions))

    oof_path = output / "oof_delta_with_risk.jsonl"
    _write_jsonl(oof_path, all_predictions)
    queue_path = output / "oof_review_queue.jsonl"
    build_review_queue_from_delta(str(oof_path), str(queue_path))
    comparison = compare_review_strategies(
        str(queue_path),
        str(output),
        random_seed=random_seed,
        evaluation_type="out_of_fold",
        train_size=None,
        eval_size=len(all_predictions),
        eval_positive_rate=_positive_rate(all_predictions),
        bootstrap_iters=bootstrap_iters,
    )
    (output / "strategy_comparison_oof.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "strategy_comparison_oof.md").write_text(
        (output / "strategy_comparison.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    fold_positive_counts = [sum(1 for idx in fold if labels[idx]) for fold in fold_indices]
    if positives < 30:
        warnings.append("positive_count_below_30; metrics may be high variance")
    summary = _oof_summary(all_predictions, comparison, warnings, usable_splits, fold_positive_counts)
    metadata = {
        "n_samples": len(rows),
        "requested_n_splits": requested_splits,
        "n_splits": usable_splits,
        "random_seed": int(random_seed),
        "warnings": warnings,
        "fold_assignment_path": str(output / "fold_assignments.json"),
        "oof_delta_with_risk_path": str(oof_path),
        "oof_review_queue_path": str(queue_path),
    }
    _write_json(output / "crossfit_metadata.json", metadata)
    _write_json(output / "fold_metrics.json", fold_metrics)
    _write_json(output / "oof_summary.json", summary)
    _write_json(output / "fold_assignments.json", assignments)
    return {"metadata": metadata, "fold_metrics": fold_metrics, "oof_summary": summary, "strategy_comparison": comparison}


def _stratified_fold_indices(labels: list[bool], n_splits: int, random_seed: int) -> list[list[int]]:
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        indices = list(range(len(labels)))
        y = [1 if value else 0 for value in labels]
        return [sorted([indices[idx] for idx in eval_idx]) for _, eval_idx in splitter.split(indices, y)]
    except Exception:
        pass
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
    for fold in folds:
        fold.sort()
    return folds


def _fold_metric_row(fold_idx: int, train_rows: list[dict], eval_rows: list[dict], predictions: list[dict]) -> dict:
    warnings = []
    y_true = [1 if _is_major(row) else 0 for row in predictions]
    y_score = [float(row.get("correction_risk_score") or 0.0) for row in predictions]
    if len(set(y_true)) < 2:
        warnings.append("single_class_eval_fold")
    metrics = _binary_metrics(y_true, y_score)
    return {
        "fold": fold_idx,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_positive_rate": _positive_rate(train_rows),
        "eval_positive_rate": _positive_rate(eval_rows),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "warnings": warnings,
    }


def _oof_summary(
    predictions: list[dict],
    comparison: dict,
    warnings: list[str],
    n_splits: int,
    fold_positive_counts: list[int],
) -> dict:
    y_true = [1 if _is_major(row) else 0 for row in predictions]
    y_score = [float(row.get("correction_risk_score") or 0.0) for row in predictions]
    metrics = _binary_metrics(y_true, y_score)
    summary = comparison.get("summary") or {}
    positive_count = sum(y_true)
    recommendation = _recommendation(comparison, positive_count)
    return {
        "n_samples": len(predictions),
        "positive_count": positive_count,
        "positive_rate": _positive_rate(predictions),
        "n_splits": n_splits,
        "fold_positive_counts": fold_positive_counts,
        "oof_pr_auc": metrics.get("pr_auc"),
        "oof_roc_auc": metrics.get("roc_auc"),
        "oof_brier_score": metrics.get("brier_score"),
        "strategy_best_by_ap": summary.get("best_strategy_by_ap"),
        "full_priority_meets_initial_standard": summary.get("meets_initial_effectiveness_standard"),
        "correction_risk_only_meets_initial_standard": summary.get(
            "correction_risk_only_meets_initial_effectiveness_standard"
        ),
        "bootstrap_ci": _full_priority_bootstrap_ci(comparison),
        "recommendation": recommendation,
        "warnings": warnings,
    }


def _recommendation(comparison: dict, positive_count: int) -> str:
    strategies = comparison.get("strategies") or {}
    summary = comparison.get("summary") or {}
    full = strategies.get("full_priority") if isinstance(strategies.get("full_priority"), dict) else {}
    risk = strategies.get("correction_risk_only") if isinstance(strategies.get("correction_risk_only"), dict) else {}
    uncertainty = strategies.get("uncertainty_only") if isinstance(strategies.get("uncertainty_only"), dict) else {}
    random_row = strategies.get("random") if isinstance(strategies.get("random"), dict) else {}
    notes = []
    full_ap = _num(full.get("average_precision_for_major_correction"))
    risk_ap = _num(risk.get("average_precision_for_major_correction"))
    uncertainty_ap = _num(uncertainty.get("average_precision_for_major_correction"))
    random_ap = _num(random_row.get("average_precision_for_major_correction"))
    if risk.get("available") and risk_ap > full_ap * 1.10:
        notes.append("consider increasing correction_risk weight, but validate on another split before changing defaults")
    elif uncertainty.get("available") and uncertainty_ap > full_ap * 1.10:
        notes.append("consider increasing uncertainty weight or improving risk model")
    elif summary.get("best_strategy_by_ap") == "full_priority" and summary.get("meets_initial_effectiveness_standard"):
        notes.append("current Layer 5 fusion is promising; validate on harder datasets next")
    elif max(full_ap, risk_ap, uncertainty_ap, random_ap) <= random_ap * 1.10 + 1e-12:
        notes.append("increase sample size, improve features, or add harder datasets before tuning weights")
    else:
        notes.append("do not tune Layer 5 defaults yet; validate the observed ranking on another split")
    if positive_count < 30:
        notes.append("positive count is still low; metrics may be high variance")
    lift_ci = ((full.get("bootstrap_ci") or {}).get("lift_at_20_percent_over_random") or {})
    if lift_ci.get("ci95_low") is not None and float(lift_ci["ci95_low"]) <= 1.0:
        notes.append("lift confidence interval includes no improvement over random")
    return "; ".join(_dedupe(notes))


def _full_priority_bootstrap_ci(comparison: dict) -> dict:
    full = ((comparison.get("strategies") or {}).get("full_priority") or {})
    return full.get("bootstrap_ci") or {}


def _binary_metrics(y_true: list[int], y_score: list[float]) -> dict:
    if not y_true:
        return {"roc_auc": None, "pr_auc": None, "brier_score": None}
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except Exception as exc:
        return {"roc_auc": None, "pr_auc": None, "brier_score": None, "warning": f"missing_sklearn: {exc}"}
    two_classes = len(set(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else None,
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def _record_id(record: dict, fallback: int) -> str:
    for key in ("sample_id", "task_id", "prediction_id", "annotation_id", "image_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"row_{fallback}"


def _is_major(record: dict) -> bool:
    delta = record.get("delta") if isinstance(record.get("delta"), dict) else {}
    return delta.get("major_correction") is True or str(delta.get("correction_severity") or "").lower() == "major"


def _positive_rate(records: list[dict]) -> float | None:
    return float(sum(1 for row in records if _is_major(row)) / len(records)) if records else None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run out-of-fold correction-risk evaluation.")
    parser.add_argument("--delta-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = crossfit_risk_evaluation(
        args.delta_dataset,
        args.output_dir,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        bootstrap_iters=args.bootstrap_iters,
    )
    print(f"[OK] OOF crossfit folds={result['metadata']['n_splits']} samples={result['oof_summary']['n_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
