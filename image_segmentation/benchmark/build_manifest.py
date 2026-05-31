from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_segmentation.benchmark.config import enabled_coco_config, load_config
from image_segmentation.benchmark.datasets.coco import iter_coco_manifest_samples
from image_segmentation.benchmark.sampling import sample_manifest_rows
from image_segmentation.benchmark.schema import validate_manifest_sample


def build_manifest(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    benchmark_id = str(config.get("benchmark_id") or "benchmark_v0_1")
    seed = int(config.get("random_seed") or 42)
    max_total = config.get("max_samples_total")
    max_total = int(max_total) if max_total is not None else None

    rows: list[dict] = []
    coco = enabled_coco_config(config)
    if coco:
        rows.extend(
            iter_coco_manifest_samples(
                coco["images_dir"],
                coco["annotations_file"],
                benchmark_id=benchmark_id,
                max_samples=int(coco["max_samples"]) if coco.get("max_samples") is not None else None,
            )
        )
    if not rows:
        raise RuntimeError("no enabled benchmark datasets produced samples")
    rows = sample_manifest_rows(rows, max_total, seed=seed)

    errors = []
    for row in rows:
        row_errors = validate_manifest_sample(row)
        if row_errors:
            errors.append({"sample_id": row.get("sample_id"), "errors": row_errors})
    if errors:
        raise RuntimeError(f"manifest validation failed for {len(errors)} samples: {errors[:3]}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"output_path": str(output), "samples": len(rows), "benchmark_id": benchmark_id}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Benchmark v0.1 JSONL manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_manifest(args.config, args.output)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] wrote {summary['samples']} samples to {summary['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

