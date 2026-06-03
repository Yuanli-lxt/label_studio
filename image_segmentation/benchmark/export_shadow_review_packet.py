from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.prediction_feature_scoring import safe_prediction_feature_vector
from image_segmentation.benchmark.prediction_feature_scoring import target_label


def export_shadow_review_packet(
    review_queue: str,
    output_dir: str,
    top_k: int = 50,
    shadow_field: str = "learned_boundary_shape_only_score",
    production_mode: bool = False,
    include_evaluation_labels: bool = False,
) -> dict:
    items = _read_jsonl(review_queue)
    current_scores = [_num(item.get("priority_score")) for item in items]
    shadow_scores = [_shadow_value(item, shadow_field) for item in items]
    valid = [idx for idx, value in enumerate(shadow_scores) if value is not None]
    current_rank = _rank_map(current_scores)
    shadow_rank = _rank_map([float(value) if value is not None else -1.0 for value in shadow_scores])
    limit = max(0, int(top_k))

    top_current = sorted(range(len(items)), key=lambda idx: (-current_scores[idx], idx))[:limit]
    top_learned = sorted(valid, key=lambda idx: (-float(shadow_scores[idx]), idx))[:limit]
    high_learned_low_current = sorted(
        valid,
        key=lambda idx: (current_rank[idx] - shadow_rank[idx], idx),
        reverse=True,
    )[:limit]
    high_current_low_learned = sorted(
        valid,
        key=lambda idx: (shadow_rank[idx] - current_rank[idx], idx),
        reverse=True,
    )[:limit]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    packet_paths = {
        "high_learned_low_current": output / "high_learned_low_current.jsonl",
        "high_current_low_learned": output / "high_current_low_learned.jsonl",
        "top_learned": output / "top_learned.jsonl",
        "top_current": output / "top_current.jsonl",
    }
    groups = {
        "high_learned_low_current": high_learned_low_current,
        "high_current_low_learned": high_current_low_learned,
        "top_learned": top_learned,
        "top_current": top_current,
    }
    for name, idxs in groups.items():
        _write_jsonl(
            packet_paths[name],
            [
                _safe_packet_row(
                    items[idx],
                    idx,
                    current_scores,
                    shadow_scores,
                    current_rank,
                    shadow_rank,
                    include_evaluation_labels=include_evaluation_labels and not production_mode,
                )
                for idx in idxs
            ],
        )
    summary = {
        "input_review_queue": review_queue,
        "output_dir": str(output),
        "shadow_field": shadow_field,
        "top_k": limit,
        "total_rows": len(items),
        "learned_non_null_count": len(valid),
        "safe_fields_only": True,
        "production_mode": bool(production_mode),
        "evaluation_only_metrics_included": bool(include_evaluation_labels and not production_mode),
        "outputs": {name: str(path) for name, path in packet_paths.items()},
    }
    _write_json(output / "review_packet_summary.json", summary)
    (output / "review_packet_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _safe_packet_row(
    item: dict,
    idx: int,
    current_scores: list[float],
    shadow_scores: list[float | None],
    current_rank: list[int],
    shadow_rank: list[int],
    include_evaluation_labels: bool = False,
) -> dict:
    metadata = item.get("shadow_score_metadata") if isinstance(item.get("shadow_score_metadata"), dict) else {}
    row = {
        "task_id": item.get("task_id"),
        "sample_id": item.get("sample_id"),
        "image_id": item.get("image_id"),
        "annotation_id": item.get("annotation_id"),
        "prediction_id": item.get("prediction_id"),
        "dataset": item.get("dataset"),
        "category_name": item.get("category_name"),
        "current_priority_score": current_scores[idx],
        "learned_shadow_score": shadow_scores[idx],
        "rank_current": current_rank[idx] + 1,
        "rank_learned": shadow_rank[idx] + 1,
        "rank_delta": (current_rank[idx] + 1) - (shadow_rank[idx] + 1),
        "prediction_time_safe_features": safe_prediction_feature_vector(item),
        "shadow_metadata_status": metadata.get("artifact_validation_status"),
        "artifact_version": metadata.get("artifact_version"),
        "shadow_only": metadata.get("shadow_only"),
        "affects_default_ranking": metadata.get("affects_default_ranking"),
    }
    if include_evaluation_labels:
        row["evaluation_only"] = {"label": target_label(item)}
    return row


def _shadow_value(item: dict, field: str) -> float | None:
    shadow = item.get("shadow_scores") if isinstance(item.get("shadow_scores"), dict) else {}
    value = shadow.get(field)
    if value is None:
        return None
    return _num(value)


def _rank_map(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Shadow Review Packet",
            "",
            f"- input_review_queue: {summary.get('input_review_queue')}",
            f"- shadow_field: {summary.get('shadow_field')}",
            f"- total_rows: {summary.get('total_rows')}",
            f"- learned_non_null_count: {summary.get('learned_non_null_count')}",
            f"- top_k: {summary.get('top_k')}",
            f"- production_mode: {summary.get('production_mode')}",
            f"- safe_fields_only: {summary.get('safe_fields_only')}",
            f"- evaluation_only_metrics_included: {summary.get('evaluation_only_metrics_included')}",
            "",
            "## Outputs",
            *[f"- {name}: {path}" for name, path in sorted((summary.get("outputs") or {}).items())],
        ]
    )


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export safe shadow-vs-current disagreement packets.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--shadow-field", default="learned_boundary_shape_only_score")
    parser.add_argument("--production-mode", action="store_true")
    parser.add_argument("--include-evaluation-labels", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_shadow_review_packet(
        args.review_queue,
        args.output_dir,
        top_k=args.top_k,
        shadow_field=args.shadow_field,
        production_mode=args.production_mode,
        include_evaluation_labels=args.include_evaluation_labels,
    )
    print(f"[OK] shadow review packet rows={result['total_rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
