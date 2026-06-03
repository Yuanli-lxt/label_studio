from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.shadow_scoring import attach_shadow_scores, shadow_scoring_runtime_config


def apply_shadow_scores(
    review_queue: str,
    output: str,
    learned_artifact_dir: str | None = None,
    enable_learned_shadow_scores: bool = False,
    enable_gated_shadow_scores: bool = False,
) -> dict:
    started = time.perf_counter()
    items = _read_jsonl(review_queue)
    before = [_order_key(item, idx) for idx, item in enumerate(items)]
    before_defaults = [_default_fields(item) for item in items]
    scoring_started = time.perf_counter()
    scored = attach_shadow_scores(
        items,
        learned_artifact_dir=learned_artifact_dir,
        enable_shadow_scoring=True,
        enable_learned_shadow_scores=enable_learned_shadow_scores,
        enable_gated_shadow_scores=enable_gated_shadow_scores,
    )
    shadow_scoring_total_time_ms = (time.perf_counter() - scoring_started) * 1000.0
    after = [_order_key(item, idx) for idx, item in enumerate(scored)]
    after_defaults = [_default_fields(item) for item in scored]
    if before != after:
        raise ValueError("shadow score replay changed queue ordering or default priority fields")
    if before_defaults != after_defaults:
        raise ValueError("shadow score replay changed default queue fields")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, scored)
    summary = {
        "input_review_queue": review_queue,
        "output_review_queue": str(path),
        "n_samples": len(scored),
        "total_rows": len(scored),
        "shadow_rows_written": sum(isinstance(item.get("shadow_scores"), dict) for item in scored),
        "learned_non_null_count": _learned_non_null_count(scored),
        "learned_null_rate": _learned_null_rate(scored),
        "artifact_status": _artifact_status(scored),
        "inference_errors": _inference_error_count(scored),
        "default_field_equality_check": before_defaults == after_defaults,
        "ordering_equality_check": before == after,
        "preserved_order_and_default_priority": True,
        "learned_artifact_dir": learned_artifact_dir,
        "enable_learned_shadow_scores": bool(enable_learned_shadow_scores),
        "enable_gated_shadow_scores": bool(enable_gated_shadow_scores),
        "shadow_runtime_config": shadow_scoring_runtime_config(
            learned_artifact_dir=learned_artifact_dir,
            enable_shadow_scoring=True,
            enable_learned_shadow_scores=enable_learned_shadow_scores,
            enable_gated_shadow_scores=enable_gated_shadow_scores,
        ),
        "coverage": _coverage(scored),
        "latency": _latency_summary(
            scored,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
            shadow_scoring_total_time_ms=shadow_scoring_total_time_ms,
        ),
    }
    (path.parent / "shadow_score_replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _order_key(item: dict, idx: int) -> tuple[Any, Any, Any, Any, Any]:
    return (
        idx,
        item.get("rank"),
        item.get("task_id"),
        item.get("sample_id"),
        item.get("priority_score"),
    )


def _default_fields(item: dict) -> dict:
    return {
        "rank": item.get("rank"),
        "priority_score": item.get("priority_score"),
        "priority_bucket": item.get("priority_bucket"),
        "review_weight_preset": item.get("review_weight_preset"),
        "review_weight_weights": item.get("review_weight_weights"),
        "score_components": item.get("score_components"),
        "review_reasons": item.get("review_reasons"),
    }


def _learned_non_null_count(items: list[dict]) -> int:
    return sum(
        ((item.get("shadow_scores") or {}).get("learned_boundary_shape_only_score") is not None)
        for item in items
        if isinstance(item.get("shadow_scores"), dict)
    )


def _learned_null_rate(items: list[dict]) -> float | None:
    if not items:
        return None
    return float((len(items) - _learned_non_null_count(items)) / len(items))


def _artifact_status(items: list[dict]) -> dict:
    out: dict[str, int] = {}
    for item in items:
        metadata = item.get("shadow_score_metadata") if isinstance(item.get("shadow_score_metadata"), dict) else {}
        status = str(metadata.get("artifact_validation_status") or "missing")
        out[status] = out.get(status, 0) + 1
    return dict(sorted(out.items()))


def _inference_error_count(items: list[dict]) -> int:
    return sum(
        bool((item.get("shadow_score_metadata") or {}).get("inference_error"))
        for item in items
        if isinstance(item.get("shadow_score_metadata"), dict)
    )


def _coverage(items: list[dict]) -> dict:
    fields = [
        "boundary_shape_calibrated_score",
        "boundary_shape_rank_score",
        "learned_boundary_shape_only_score",
        "learned_current_plus_boundary_shape_score",
        "gated_boundary_shape_score",
        "gated_current_boundary_score",
    ]
    out = {}
    for field in fields:
        non_null = 0
        for item in items:
            scores = item.get("shadow_scores") if isinstance(item.get("shadow_scores"), dict) else {}
            if scores.get(field) is not None:
                non_null += 1
        out[field] = {
            "non_null_count": non_null,
            "null_count": len(items) - non_null,
            "non_null_rate": float(non_null / len(items)) if items else None,
        }
    return out


def _latency_summary(items: list[dict], total_time_ms: float, shadow_scoring_total_time_ms: float) -> dict:
    learned = _metadata_latencies(items, "learned_inference_latency_ms")
    scoring = _metadata_latencies(items, "shadow_scoring_latency_ms")
    artifact = _metadata_latencies(items, "artifact_load_latency_ms")
    return {
        "timing_source": "time.perf_counter",
        "timing_available": True,
        "rows_processed": len(items),
        "batch_size": len(items),
        "artifact_loaded_once": False,
        "artifact_load_latency_ms": _latency_distribution(artifact),
        "shadow_scoring_total_time_ms": float(shadow_scoring_total_time_ms),
        "total_replay_time_ms": float(total_time_ms),
        "per_item_learned_inference_latency_ms": _latency_distribution(learned),
        "per_item_shadow_scoring_latency_ms": _latency_distribution(scoring),
    }


def _metadata_latencies(items: list[dict], key: str) -> list[float]:
    values = []
    for item in items:
        metadata = item.get("shadow_score_metadata") if isinstance(item.get("shadow_score_metadata"), dict) else {}
        value = metadata.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _latency_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "mean": float(sum(ordered) / len(ordered)),
        "max": float(max(ordered)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * (percentile / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach shadow-only review scores to a review queue without changing default ordering.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learned-artifact-dir")
    parser.add_argument("--artifact-dir", dest="artifact_dir")
    parser.add_argument("--enable-learned-shadow-scores", action="store_true")
    parser.add_argument("--enable-learned-boundary-shape-shadow", action="store_true")
    parser.add_argument("--enable-gated-shadow-scores", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = apply_shadow_scores(
        args.review_queue,
        args.output,
        learned_artifact_dir=args.learned_artifact_dir or args.artifact_dir,
        enable_learned_shadow_scores=args.enable_learned_shadow_scores or args.enable_learned_boundary_shape_shadow,
        enable_gated_shadow_scores=args.enable_gated_shadow_scores,
    )
    print(f"[OK] shadow scores applied samples={result['n_samples']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
