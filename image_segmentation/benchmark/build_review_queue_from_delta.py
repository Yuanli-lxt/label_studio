from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER_DIR = ROOT / "services" / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_review_queue import build_segmentation_review_queue  # noqa: E402


def build_review_queue_from_delta(delta_dataset: str, output: str) -> dict:
    records = _read_jsonl(delta_dataset)
    normalized = [_normalize_risk(row) for row in records]
    return build_segmentation_review_queue(normalized, output)


def _normalize_risk(row: dict) -> dict:
    out = dict(row)
    risk = out.get("correction_risk")
    if not isinstance(risk, dict) and isinstance(out.get("correction_risk_score"), (int, float)):
        score = float(out["correction_risk_score"])
        out["correction_risk"] = {
            "risk_score": score,
            "risk_bucket": out.get("correction_risk_bucket") or _risk_bucket(score),
            "predicted_major_correction": bool(score >= 0.5),
            "model_version": out.get("risk_model_version"),
            "risk_model_dir": out.get("risk_model_dir"),
        }
    return out


def _risk_bucket(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Layer 5 review queue from risk-scored delta records.")
    parser.add_argument("--delta-dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_review_queue_from_delta(args.delta_dataset, args.output)
    print(f"[OK] wrote review queue with {len(result.get('items') or [])} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
