from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.config import enabled_coco_config, enabled_dataset_configs, load_config
from image_segmentation.benchmark.datasets.coco import iter_coco_manifest_samples
from image_segmentation.benchmark.datasets.lvis import iter_lvis_manifest_samples, lvis_annotation_stats
from image_segmentation.benchmark.datasets.mask_folder import iter_mask_folder_manifest_samples
from image_segmentation.benchmark.datasets.open_images import iter_open_images_manifest_samples
from image_segmentation.benchmark.sampling import sample_manifest_rows
from image_segmentation.benchmark.schema import validate_manifest_sample

ROOT = Path(__file__).resolve().parents[2]


def build_manifest(config_path: str, output_path: str, summary_output: str | None = None) -> dict:
    config = load_config(config_path)
    benchmark_id = str(config.get("benchmark_id") or "benchmark_v0_1")
    seed = int(config.get("random_seed") or 42)
    max_total = config.get("max_samples_total")
    max_total = int(max_total) if max_total is not None else None

    rows: list[dict] = []
    coco = enabled_coco_config(config)
    enabled_datasets = enabled_dataset_configs(config)
    sampling = config.get("sampling") if isinstance(config.get("sampling"), dict) else {}
    if coco:
        coco_max_samples = int(coco["max_samples"]) if coco.get("max_samples") is not None else None
        raw_coco_limit = None if sampling.get("image_diversity") is True else coco_max_samples
        rows.extend(
            iter_coco_manifest_samples(
                coco["images_dir"],
                coco["annotations_file"],
                benchmark_id=benchmark_id,
                max_samples=raw_coco_limit,
            )
        )
    for dataset_name, dataset_config in enabled_datasets.items():
        if dataset_name == "coco":
            continue
        dataset_max_samples = int(dataset_config["max_samples"]) if dataset_config.get("max_samples") is not None else None
        raw_limit = None if sampling.get("image_diversity") is True else dataset_max_samples
        if dataset_name == "lvis":
            rows.extend(
                iter_lvis_manifest_samples(
                    dataset_config["images_dir"],
                    dataset_config["annotations_file"],
                    benchmark_id=benchmark_id,
                    max_samples=raw_limit,
                )
            )
        elif dataset_name in {"dis5k", "cod10k", "camo"}:
            rows.extend(
                iter_mask_folder_manifest_samples(
                    dataset_config["images_dir"],
                    dataset_config["masks_dir"],
                    benchmark_id=benchmark_id,
                    dataset_name=dataset_name,
                    max_samples=raw_limit,
                    image_glob=str(dataset_config.get("image_glob") or "*.*"),
                    mask_glob=str(dataset_config.get("mask_glob") or "*.png"),
                    mask_match_strategy=str(dataset_config.get("mask_match_strategy") or "same_stem"),
                    include_image_patterns=dataset_config.get("include_image_patterns"),
                    include_mask_patterns=dataset_config.get("include_mask_patterns"),
                    exclude_image_patterns=dataset_config.get("exclude_image_patterns"),
                    exclude_mask_patterns=dataset_config.get("exclude_mask_patterns"),
                    category_name=str(dataset_config.get("category_name") or "foreground_object"),
                    category_id=str(dataset_config.get("category_id") or dataset_config.get("category_name") or "foreground_object"),
                    default_difficulty_tags=list(dataset_config.get("default_difficulty_tags") or []),
                )
            )
        elif dataset_name == "open_images_v7":
            rows.extend(
                iter_open_images_manifest_samples(
                    dataset_config["dataset_dir"],
                    benchmark_id=benchmark_id,
                    max_samples=raw_limit,
                )
            )
    if not rows:
        raise RuntimeError("no enabled benchmark datasets produced samples")
    if max_total is None and coco and coco.get("max_samples") is not None:
        max_total = int(coco["max_samples"])
    rows = sample_manifest_rows(
        rows,
        max_total,
        seed=seed,
        category_balance=sampling.get("category_balance") is not False,
        image_diversity=sampling.get("image_diversity") is True,
        max_instances_per_image=sampling.get("max_instances_per_image"),
    )
    warnings = []
    if max_total is not None and len(rows) < max_total:
        warnings.append(
            f"sampled_{len(rows)}_of_requested_{max_total}; image diversity constraints may have limited available rows"
        )

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
    summary = manifest_summary(rows, seed=seed, warnings=warnings)
    if "lvis" in enabled_datasets:
        try:
            lvis = enabled_datasets["lvis"]
            stats = lvis_annotation_stats(lvis["images_dir"], lvis["annotations_file"])
            summary["frequency_distribution"] = frequency_distribution(rows)
            summary["missing_image_count"] = stats["n_missing_images"]
            summary["missing_image_examples"] = stats["missing_image_examples"]
            summary["skipped_annotation_count"] = stats["skipped_annotation_count"]
            summary["skipped_annotation_reasons"] = stats["skipped_annotation_reasons"]
        except Exception as exc:
            summary.setdefault("warnings", []).append(f"lvis_summary_stats_unavailable: {exc}")
    if any(str(row.get("dataset")).upper() == "DIS5K" for row in rows):
        summary["boundary_complexity_distribution"] = _distribution(
            [_num((row.get("boundary_metadata") or {}).get("perimeter_area_ratio")) for row in rows]
        )
        summary["thin_structure_count"] = sum(1 for row in rows if "thin_structure" in (row.get("difficulty_tags") or []))
        summary["touches_border_count"] = sum(1 for row in rows if "touches_border" in (row.get("difficulty_tags") or []))
        summary["skipped_samples"] = []
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(output),
        "samples": len(rows),
        "benchmark_id": benchmark_id,
        "summary": summary,
        "warnings": warnings,
    }


def manifest_summary(rows: list[dict], seed: int = 42, warnings: list[str] | None = None) -> dict:
    category_counts = Counter(str(row.get("category_name") or row.get("category_id") or "unknown") for row in rows)
    tag_counts: Counter[str] = Counter()
    image_counts = Counter(str(row.get("image_id")) for row in rows)
    area_ratios = [_num(row.get("gt_area_ratio")) for row in rows if row.get("gt_area_ratio") is not None]
    for row in rows:
        tag_counts.update(str(tag) for tag in (row.get("difficulty_tags") or []))
    return {
        "n_samples": len(rows),
        "n_unique_images": len(image_counts),
        "n_categories": len(category_counts),
        "category_distribution": dict(sorted(category_counts.items())),
        "category_distribution_top20": dict(category_counts.most_common(20)),
        "frequency_distribution": frequency_distribution(rows),
        "difficulty_tag_distribution": dict(sorted(tag_counts.items())),
        "max_instances_per_image_observed": max(image_counts.values()) if image_counts else 0,
        "area_ratio_distribution": _distribution(area_ratios),
        "boundary_complexity_distribution": _distribution(
            [_num((row.get("boundary_metadata") or {}).get("perimeter_area_ratio")) for row in rows if isinstance(row.get("boundary_metadata"), dict)]
        ),
        "thin_structure_count": sum(1 for row in rows if "thin_structure" in (row.get("difficulty_tags") or [])),
        "touches_border_count": sum(1 for row in rows if "touches_border" in (row.get("difficulty_tags") or [])),
        "random_seed": int(seed),
        "warnings": warnings or [],
    }


def frequency_distribution(rows: list[dict]) -> dict:
    values = Counter()
    for row in rows:
        value = row.get("category_frequency")
        if value is None and isinstance(row.get("metadata"), dict):
            value = row["metadata"].get("category_frequency")
        if value is not None:
            values[str(value)] += 1
    return dict(sorted(values.items()))


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Benchmark v0.1 JSONL manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--summary-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.output
        summary_output = args.summary_output
        if not output:
            config = load_config(args.config)
            benchmark_id = str(config.get("benchmark_id") or Path(args.config).stem)
            output = str(ROOT / "demo_data" / "model_state" / "image_segmentation" / "benchmark" / f"{benchmark_id}_manifest.jsonl")
            summary_output = summary_output or str(
                ROOT / "demo_data" / "model_state" / "image_segmentation" / "benchmark" / f"{benchmark_id}_manifest_summary.json"
            )
        summary = build_manifest(args.config, output, summary_output=summary_output)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] wrote {summary['samples']} samples to {summary['output_path']}")
    if args.summary_output:
        print(f"[OK] wrote manifest summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
