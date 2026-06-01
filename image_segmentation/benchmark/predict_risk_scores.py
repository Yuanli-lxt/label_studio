from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.train_risk_model import _assert_safe_feature_names


ROOT = Path(__file__).resolve().parents[2]
TRAINER_DIR = ROOT / "services" / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_correction_risk import (  # noqa: E402
    FEATURE_NAMES,
    extract_correction_risk_features,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def predict_risk_scores(delta_dataset: str, risk_model_dir: str, output: str) -> dict:
    if joblib is None:
        raise RuntimeError("joblib is required for risk prediction")
    model_dir = Path(risk_model_dir)
    feature_names = _read_json(model_dir / "feature_names.json") or list(FEATURE_NAMES)
    _assert_safe_feature_names(feature_names)
    classifier = joblib.load(model_dir / "classifier.joblib")
    metadata = _read_json(model_dir / "metadata.json") or {}
    model_version = (metadata.get("correction_risk") or {}).get("model_version") or _metadata_hash(metadata)

    rows = _read_jsonl(delta_dataset)
    scored = []
    for row in rows:
        features = extract_correction_risk_features(row)
        x = [[float(features.get(name, 0.0) or 0.0) for name in feature_names]]
        if hasattr(classifier, "predict_proba"):
            risk_score = float(classifier.predict_proba(x)[0][1])
        else:
            risk_score = float(classifier.predict(x)[0])
        out = dict(row)
        out["correction_risk"] = {
            "risk_score": risk_score,
            "risk_bucket": _risk_bucket(risk_score),
            "predicted_major_correction": bool(risk_score >= 0.5),
            "model_version": model_version,
            "risk_model_dir": str(model_dir),
        }
        out["correction_risk_score"] = risk_score
        out["correction_risk_bucket"] = _risk_bucket(risk_score)
        out["risk_model_dir"] = str(model_dir)
        out["risk_model_version"] = model_version
        scored.append(out)

    _write_jsonl(Path(output), scored)
    return {"n_records": len(scored), "risk_model_dir": str(model_dir), "feature_names": feature_names}


def _risk_bucket(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _metadata_hash(metadata: Any) -> str:
    import hashlib

    raw = json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict held-out correction-risk scores for delta records.")
    parser.add_argument("--delta-dataset", required=True)
    parser.add_argument("--risk-model-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = predict_risk_scores(args.delta_dataset, args.risk_model_dir, args.output)
    print(f"[OK] wrote {result['n_records']} risk-scored records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
