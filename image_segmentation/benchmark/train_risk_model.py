from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINER_DIR = ROOT / "services" / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_correction_risk import (  # noqa: E402
    FEATURE_NAMES,
    build_correction_risk_dataset,
    train_correction_risk_model,
)


LEAKY_FEATURE_TOKENS = {
    "delta",
    "model_human_iou",
    "model_human_dice",
    "precision",
    "recall",
    "added_area_px",
    "removed_area_px",
    "correction_area_ratio",
    "major_correction",
    "correction_severity",
    "correction_reason",
    "human_bbox",
    "human_area_px",
    "intersection_area_px",
    "union_area_px",
}


def train_from_delta_dataset(
    delta_dataset: str,
    output_dir: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    _assert_safe_feature_names(FEATURE_NAMES)
    records = _read_jsonl(delta_dataset)
    features, labels, summary = build_correction_risk_dataset(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not labels or len(set(labels)) < 2:
        evaluation = {
            "status": "skipped",
            "warning": "single_class_training_data",
            "n_records": len(labels),
            "positive_rate": _safe_div(sum(labels), len(labels)),
        }
        _write_json(output / "evaluation.json", evaluation)
        metadata = train_correction_risk_model(records, str(output))
        return {"metadata": metadata, "evaluation": evaluation}

    metadata = train_correction_risk_model(records, str(output))
    evaluation = _evaluate_holdout(features, labels, test_size=test_size, random_seed=random_seed)
    evaluation["feature_names_safe"] = True
    evaluation["feature_names"] = list(FEATURE_NAMES)
    evaluation["dataset_summary"] = summary
    _write_json(output / "evaluation.json", evaluation)
    return {"metadata": metadata, "evaluation": evaluation}


def _evaluate_holdout(features: list[dict], labels: list[int], test_size: float, random_seed: int) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        return {"status": "skipped", "warning": f"missing_sklearn: {exc}"}

    x = np.asarray([[float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES] for row in features], dtype=float)
    y = np.asarray(labels, dtype=int)
    stratify = y if len(set(labels)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=max(0.05, min(0.95, float(test_size))),
        random_state=int(random_seed),
        stratify=stratify,
    )
    classifier = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=int(random_seed))
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = classifier.predict(x_test)
    two_classes = len(set(y_test.tolist())) == 2
    return {
        "status": "evaluated",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": _safe_div(int(y.sum()), int(len(y))),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_test, probabilities)) if two_classes else None,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }


def _assert_safe_feature_names(feature_names: list[str]) -> None:
    joined = "\n".join(feature_names).lower()
    leaked = sorted(token for token in LEAKY_FEATURE_TOKENS if token.lower() in joined)
    if leaked:
        raise ValueError(f"target leakage feature names detected: {leaked}")


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train benchmark-facing Layer 4 correction-risk model.")
    parser.add_argument("--delta-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_from_delta_dataset(args.delta_dataset, args.output_dir, args.test_size, args.random_seed)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] correction-risk training status: {result['evaluation'].get('status')}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

