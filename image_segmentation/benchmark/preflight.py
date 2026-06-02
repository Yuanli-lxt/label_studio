from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_segmentation.benchmark.config import enabled_dataset_configs, load_config
from image_segmentation.benchmark.datasets.coco import iter_coco_manifest_samples, load_coco_json
from image_segmentation.benchmark.datasets.lvis import (
    category_frequency_distribution,
    iter_lvis_manifest_samples,
    lvis_annotation_stats,
    missing_lvis_image_examples,
    missing_lvis_images,
)
from image_segmentation.benchmark.datasets.mask_folder import iter_mask_folder_manifest_samples, mask_folder_stats
from image_segmentation.benchmark.datasets.open_images import iter_open_images_manifest_samples


def run_preflight(config_path: str, json_output: str | None = None) -> tuple[bool, list[str]]:
    lines: list[str] = []
    reports: list[dict] = []
    path = Path(config_path)
    if not path.exists():
        return False, [f"[FAIL] config not found: {path}"]
    lines.append(f"[OK] config exists: {path}")
    try:
        config = load_config(path)
    except Exception as exc:
        return False, lines + [f"[FAIL] config is not readable: {exc}"]

    enabled = enabled_dataset_configs(config)
    if not enabled:
        return False, lines + ["[FAIL] no enabled benchmark dataset in config"]
    ok = True
    for dataset_name, dataset_config in enabled.items():
        report, dataset_lines = _preflight_dataset(dataset_name, dataset_config, str(config.get("benchmark_id") or "preflight"))
        reports.append(report)
        lines.extend(dataset_lines)
        ok = ok and report["status"] in {"ok", "warning"}
    command = (
        "python -m image_segmentation.benchmark.build_manifest "
        f"--config {path} --output demo_data/model_state/image_segmentation/benchmark/{config.get('benchmark_id', 'benchmark')}_manifest.jsonl"
    )
    lines.append(f"[NEXT] {command}")
    if json_output:
        output = Path(json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"config": str(path), "status": "ok" if ok else "failed", "datasets": reports}, indent=2), encoding="utf-8")
        lines.append(f"[OK] wrote preflight JSON: {output}")
    return ok, lines


def _preflight_dataset(dataset_name: str, dataset_config: dict, benchmark_id: str) -> tuple[dict, list[str]]:
    loader_type = _loader_type(dataset_name)
    report = {
        "dataset_name": dataset_name,
        "dataset": _display_dataset(dataset_name),
        "loader_type": loader_type,
        "images_dir": dataset_config.get("images_dir"),
        "annotations_file": dataset_config.get("annotations_file"),
        "masks_dir": dataset_config.get("masks_dir"),
        "n_images_found": 0,
        "n_annotations_or_masks_found": 0,
        "n_images_scanned": 0,
        "n_masks_scanned": 0,
        "n_images_excluded": 0,
        "n_masks_excluded": 0,
        "excluded_image_examples": [],
        "excluded_mask_examples": [],
        "n_images": 0,
        "n_masks": 0,
        "n_pairs": 0,
        "missing_mask_count": 0,
        "missing_image_count": 0,
        "n_valid_samples_estimated": 0,
        "n_missing_images": 0,
        "n_empty_masks": 0,
        "n_shape_mismatch": 0,
        "category_count": 0,
        "difficulty_tag_preview": [],
        "status": "ok",
        "warnings": [],
        "errors": [],
        "sample_check": {},
    }
    lines = [f"[DATASET] {dataset_name} loader_type={loader_type}"]
    try:
        if dataset_name == "coco":
            _preflight_coco(dataset_config, benchmark_id, report, lines)
        elif dataset_name == "lvis":
            _preflight_lvis(dataset_config, benchmark_id, report, lines)
        elif dataset_name in {"dis5k", "cod10k", "camo"}:
            _preflight_mask_folder(dataset_name, dataset_config, benchmark_id, report, lines)
        elif dataset_name == "open_images_v7":
            _preflight_open_images(dataset_config, benchmark_id, report, lines)
        else:
            report["status"] = "failed"
            lines.append(f"[FAIL] unsupported dataset in config: {dataset_name}")
    except Exception as exc:
        report["status"] = "failed"
        lines.append(f"[FAIL] {dataset_name} preflight failed: {exc}")
    return report, lines


def _preflight_coco(dataset_config: dict, benchmark_id: str, report: dict, lines: list[str]) -> None:
    images_dir = Path(dataset_config.get("images_dir", ""))
    annotations_file = Path(dataset_config.get("annotations_file", ""))
    _check_path(images_dir, "COCO images_dir", lines, report)
    _check_path(annotations_file, "COCO annotations_file", lines, report)
    if report["status"] == "failed":
        lines.append("[NEXT] Run: python -m image_segmentation.benchmark.download_data --dataset coco_val2017 --output-dir data/external")
        return
    data = load_coco_json(annotations_file)
    report["n_images_found"] = len(data.get("images", []))
    report["n_annotations_or_masks_found"] = len(data.get("annotations", []))
    report["category_count"] = len(data.get("categories", []))
    first = next(iter(iter_coco_manifest_samples(images_dir, annotations_file, benchmark_id, max_samples=1)))
    report["n_valid_samples_estimated"] = 1
    report["difficulty_tag_preview"] = first.get("difficulty_tags", [])
    lines.append(f"[OK] parsed one valid image+annotation+mask: {first['sample_id']}")


def _preflight_lvis(dataset_config: dict, benchmark_id: str, report: dict, lines: list[str]) -> None:
    images_dir = Path(dataset_config.get("images_dir", ""))
    annotations_file = Path(dataset_config.get("annotations_file", ""))
    _check_path(images_dir, "LVIS images_dir", lines, report)
    _check_path(annotations_file, "LVIS annotations_file", lines, report)
    if report["status"] == "failed":
        return
    data = load_coco_json(annotations_file)
    stats = lvis_annotation_stats(images_dir, annotations_file)
    report["n_images"] = stats["n_images"]
    report["n_annotations"] = stats["n_annotations"]
    report["n_categories"] = stats["n_categories"]
    report["n_images_found"] = stats["n_images"]
    report["n_annotations_or_masks_found"] = stats["n_annotations"]
    report["category_count"] = stats["n_categories"]
    report["n_missing_images"] = stats["n_missing_images"]
    report["missing_image_examples"] = stats["missing_image_examples"]
    report["skipped_annotation_count"] = stats["skipped_annotation_count"]
    report["skipped_annotation_reasons"] = stats["skipped_annotation_reasons"]
    freq = category_frequency_distribution(annotations_file)
    lines.append(f"[INFO] LVIS category frequency distribution: {freq}")
    if report["n_missing_images"]:
        report["status"] = "warning"
        report["warnings"].append(f"missing_images:{report['n_missing_images']}")
        lines.append(f"[WARN] LVIS missing images referenced by annotations: {report['n_missing_images']}")
        lines.append(f"[WARN] missing image examples: {report['missing_image_examples']}")
    first = next(iter(iter_lvis_manifest_samples(images_dir, annotations_file, benchmark_id, max_samples=1)))
    report["n_valid_samples_estimated"] = 1
    report["difficulty_tag_preview"] = first.get("difficulty_tags", [])
    report["sample_check"] = {
        "sample_id": first.get("sample_id"),
        "image_path": first.get("image_path"),
        "width": first.get("width"),
        "height": first.get("height"),
        "bbox": first.get("gt_bbox_xyxy"),
        "category_frequency": first.get("category_frequency"),
        "mask_decoded": True,
    }
    lines.append(f"[OK] parsed one valid LVIS sample: {first['sample_id']}")


def _preflight_mask_folder(dataset_name: str, dataset_config: dict, benchmark_id: str, report: dict, lines: list[str]) -> None:
    images_dir = Path(dataset_config.get("images_dir", ""))
    masks_dir = Path(dataset_config.get("masks_dir", ""))
    _check_path(images_dir, f"{dataset_name} images_dir", lines, report)
    _check_path(masks_dir, f"{dataset_name} masks_dir", lines, report)
    if report["status"] == "failed":
        lines.append(f"[NEXT] Run manual placement, then rerun this preflight for {dataset_name}.")
        return
    image_glob = str(dataset_config.get("image_glob") or "*.*")
    mask_glob = str(dataset_config.get("mask_glob") or "*.png")
    mask_match_strategy = str(dataset_config.get("mask_match_strategy") or "same_stem")
    include_image_patterns = dataset_config.get("include_image_patterns")
    include_mask_patterns = dataset_config.get("include_mask_patterns")
    exclude_image_patterns = dataset_config.get("exclude_image_patterns")
    exclude_mask_patterns = dataset_config.get("exclude_mask_patterns")
    stats = mask_folder_stats(
        images_dir,
        masks_dir,
        image_glob=image_glob,
        mask_glob=mask_glob,
        mask_match_strategy=mask_match_strategy,
        include_image_patterns=include_image_patterns,
        include_mask_patterns=include_mask_patterns,
        exclude_image_patterns=exclude_image_patterns,
        exclude_mask_patterns=exclude_mask_patterns,
    )
    report["n_images_scanned"] = stats["n_images_scanned"]
    report["n_masks_scanned"] = stats["n_masks_scanned"]
    report["n_images_found"] = stats["n_images_found"]
    report["n_annotations_or_masks_found"] = stats["n_masks_found"]
    report["n_images"] = stats["n_images_found"]
    report["n_masks"] = stats["n_masks_found"]
    report["n_images_excluded"] = stats["n_images_excluded"]
    report["n_masks_excluded"] = stats["n_masks_excluded"]
    report["excluded_image_examples"] = stats["excluded_image_examples"]
    report["excluded_mask_examples"] = stats["excluded_mask_examples"]
    report["n_pairs"] = stats["n_pairs"]
    report["missing_mask_count"] = stats.get("missing_mask_count", 0)
    report["missing_image_count"] = stats.get("missing_image_count", 0)
    report["n_valid_samples_estimated"] = stats["n_valid"]
    report["n_empty_masks"] = stats["n_empty_masks"]
    report["n_shape_mismatch"] = stats["n_shape_mismatch"]
    if report["n_images_excluded"] or report["n_masks_excluded"]:
        lines.append(
            f"[INFO] file filters excluded images={report['n_images_excluded']} "
            f"masks={report['n_masks_excluded']}"
        )
        if report["excluded_image_examples"]:
            lines.append(f"[INFO] excluded image examples: {report['excluded_image_examples']}")
        if report["excluded_mask_examples"]:
            lines.append(f"[INFO] excluded mask examples: {report['excluded_mask_examples']}")
    if report["n_empty_masks"] or report["n_shape_mismatch"]:
        report["status"] = "failed" if dataset_name == "dis5k" else "warning"
        if report["n_empty_masks"]:
            report["errors" if dataset_name == "dis5k" else "warnings"].append(f"empty_masks:{report['n_empty_masks']}")
        if report["n_shape_mismatch"]:
            report["errors" if dataset_name == "dis5k" else "warnings"].append(f"shape_mismatch:{report['n_shape_mismatch']}")
        lines.append(
            f"[WARN] mask folder issues: empty_masks={report['n_empty_masks']} "
            f"shape_mismatch={report['n_shape_mismatch']}"
        )
    if report["n_pairs"] <= 0 or report["n_valid_samples_estimated"] <= 0:
        report["status"] = "failed"
        report["errors"].append("no_valid_image_mask_pairs")
        lines.append(f"[FAIL] no valid image/mask pairs found for {dataset_name}")
        return
    if report["status"] != "failed":
        first = next(
            iter(
                iter_mask_folder_manifest_samples(
                    images_dir,
                    masks_dir,
                    benchmark_id,
                    dataset_name=dataset_name,
                    max_samples=1,
                    image_glob=image_glob,
                    mask_glob=mask_glob,
                    mask_match_strategy=mask_match_strategy,
                    include_image_patterns=include_image_patterns,
                    include_mask_patterns=include_mask_patterns,
                    exclude_image_patterns=exclude_image_patterns,
                    exclude_mask_patterns=exclude_mask_patterns,
                    category_name=str(dataset_config.get("category_name") or "foreground_object"),
                    category_id=str(dataset_config.get("category_id") or dataset_config.get("category_name") or "foreground_object"),
                    default_difficulty_tags=list(dataset_config.get("default_difficulty_tags") or []),
                )
            )
        )
        report["difficulty_tag_preview"] = first.get("difficulty_tags", [])
        report["sample_check"] = {
            "sample_id": first.get("sample_id"),
            "image_path": first.get("image_path"),
            "mask_path": first.get("mask_path") or (first.get("metadata") or {}).get("mask_path"),
            "width": first.get("width"),
            "height": first.get("height"),
            "bbox": first.get("gt_bbox_xyxy"),
            "gt_area": first.get("gt_area"),
            "binary_mask": True,
            "boundary_metadata": first.get("boundary_metadata"),
        }
        lines.append(f"[OK] found {report['n_valid_samples_estimated']} image/mask pairs; parsed {first['sample_id']}")


def _preflight_open_images(dataset_config: dict, benchmark_id: str, report: dict, lines: list[str]) -> None:
    dataset_dir = Path(dataset_config.get("dataset_dir", ""))
    annotations = dataset_dir / "annotations.jsonl"
    report["images_dir"] = str(dataset_dir / "images")
    report["annotations_file"] = str(annotations)
    _check_path(dataset_dir / "images", "Open Images images_dir", lines, report)
    _check_path(dataset_dir / "segmentations", "Open Images segmentations_dir", lines, report)
    _check_path(annotations, "Open Images annotations.jsonl", lines, report)
    if report["status"] == "failed":
        return
    report["n_images_found"] = len([p for p in (dataset_dir / "images").glob("*") if p.is_file()])
    report["n_annotations_or_masks_found"] = sum(1 for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip())
    first = next(iter(iter_open_images_manifest_samples(dataset_dir, benchmark_id, max_samples=1)))
    report["n_valid_samples_estimated"] = 1
    report["difficulty_tag_preview"] = first.get("difficulty_tags", [])
    lines.append(f"[OK] parsed one Open Images sample: {first['sample_id']}")


def _check_path(path: Path, label: str, lines: list[str], report: dict) -> None:
    if path.exists():
        lines.append(f"[OK] {label} exists: {path}")
    else:
        report["status"] = "failed"
        report.setdefault("errors", []).append(f"{label} missing: {path}")
        lines.append(f"[FAIL] {label} missing: {path}")


def _loader_type(dataset_name: str) -> str:
    return {
        "coco": "coco",
        "lvis": "lvis",
        "dis5k": "mask_folder",
        "cod10k": "mask_folder",
        "camo": "mask_folder",
        "open_images_v7": "open_images",
    }.get(dataset_name, "unknown")


def _display_dataset(dataset_name: str) -> str:
    return {
        "coco": "COCO",
        "lvis": "LVIS",
        "dis5k": "DIS5K",
        "cod10k": "COD10K",
        "camo": "CAMO",
        "open_images_v7": "Open Images V7",
    }.get(dataset_name, dataset_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local benchmark data before running.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ok, lines = run_preflight(args.config, json_output=args.json_output)
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
