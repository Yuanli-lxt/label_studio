from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from image_segmentation.benchmark.datasets.coco import load_coco_json, segmentation_to_mask
from image_segmentation.benchmark.metrics import mask_bbox, touches_border
from image_segmentation.benchmark.schema import coco_xywh_to_xyxy


FREQUENCY_TAGS = {"r": "rare_category", "c": "common_category", "f": "frequent_category"}


def iter_lvis_manifest_samples(
    images_dir: str | Path,
    annotations_file: str | Path,
    benchmark_id: str,
    max_samples: int | None = None,
) -> Iterable[dict]:
    data = load_coco_json(annotations_file)
    images_by_id = {row["id"]: row for row in data.get("images", []) if isinstance(row, dict) and "id" in row}
    categories = {
        row["id"]: row
        for row in data.get("categories", [])
        if isinstance(row, dict) and "id" in row
    }
    annotations = [row for row in data.get("annotations", []) if _valid_annotation(row)]
    instances_per_image = Counter(row.get("image_id") for row in annotations)
    count = 0
    for ann in annotations:
        image = images_by_id.get(ann.get("image_id"))
        if not image:
            continue
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        image_file_name = _image_file_name(image)
        image_path = Path(images_dir) / image_file_name
        if not image_path.exists():
            continue
        try:
            bbox_xyxy = coco_xywh_to_xyxy(ann["bbox"], width, height)
            mask = segmentation_to_mask(ann.get("segmentation"), width, height, ann.get("iscrowd", 0))
        except Exception:
            continue
        area = int(mask.sum())
        if area <= 0:
            continue
        area_ratio = float(area / (width * height)) if width * height else 0.0
        category = categories.get(ann.get("category_id"), {})
        tags = _difficulty_tags(
            area_ratio=area_ratio,
            bbox_xyxy=bbox_xyxy,
            gt_mask=mask,
            width=width,
            height=height,
            instances_in_image=instances_per_image.get(ann.get("image_id"), 0),
            frequency=category.get("frequency"),
        )
        yield {
            "benchmark_id": benchmark_id,
            "dataset": "LVIS",
            "sample_id": f"LVIS_{image.get('id')}_{ann.get('id')}",
            "image_id": str(image.get("id")),
            "annotation_id": str(ann.get("id")),
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "category_id": str(ann.get("category_id")),
            "category_name": str(category.get("name", ann.get("category_id"))),
            "category_frequency": category.get("frequency"),
            "gt_bbox_xyxy": bbox_xyxy,
            "gt_area": area,
            "gt_area_ratio": area_ratio,
            "gt_touches_border": touches_border(mask, bbox_xyxy, width, height),
            "difficulty_tags": tags,
            "split": benchmark_id,
            "gt_segmentation": ann.get("segmentation"),
            "gt_iscrowd": int(ann.get("iscrowd") or 0),
            "metadata": {
                "lvis_file_name": image.get("file_name"),
                "coco_url": image.get("coco_url"),
                "category_frequency": category.get("frequency"),
            },
        }
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def category_frequency_distribution(annotations_file: str | Path) -> dict[str, int]:
    data = load_coco_json(annotations_file)
    counter: Counter[str] = Counter()
    for category in data.get("categories", []):
        if isinstance(category, dict) and category.get("frequency"):
            counter[str(category["frequency"])] += 1
    return dict(counter)


def missing_lvis_images(images_dir: str | Path, annotations_file: str | Path) -> int:
    return len(missing_lvis_image_examples(images_dir, annotations_file, limit=None))


def missing_lvis_image_examples(images_dir: str | Path, annotations_file: str | Path, limit: int | None = 10) -> list[str]:
    data = load_coco_json(annotations_file)
    root = Path(images_dir)
    missing = [
        _image_file_name(row)
        for row in data.get("images", [])
        if isinstance(row, dict) and not (root / _image_file_name(row)).exists()
    ]
    return missing if limit is None else missing[:limit]


def lvis_annotation_stats(images_dir: str | Path, annotations_file: str | Path) -> dict:
    data = load_coco_json(annotations_file)
    images_by_id = {row.get("id"): row for row in data.get("images", []) if isinstance(row, dict)}
    categories_by_id = {row.get("id"): row for row in data.get("categories", []) if isinstance(row, dict)}
    missing_image_ids = {
        row.get("id")
        for row in data.get("images", [])
        if isinstance(row, dict) and not (Path(images_dir) / _image_file_name(row)).exists()
    }
    reasons: Counter[str] = Counter()
    usable = 0
    for ann in data.get("annotations", []):
        if not isinstance(ann, dict):
            reasons["invalid_annotation_type"] += 1
            continue
        image = images_by_id.get(ann.get("image_id"))
        if image is None:
            reasons["missing_image_metadata"] += 1
        elif ann.get("image_id") in missing_image_ids:
            reasons["missing_image_file"] += 1
        elif ann.get("category_id") not in categories_by_id:
            reasons["missing_category_metadata"] += 1
        elif not _valid_annotation(ann):
            reasons["invalid_or_empty_annotation"] += 1
        else:
            image = images_by_id.get(ann.get("image_id"))
            try:
                coco_xywh_to_xyxy(ann["bbox"], int(image.get("width") or 0), int(image.get("height") or 0))
            except Exception:
                reasons["invalid_clipped_bbox"] += 1
                continue
            usable += 1
    return {
        "n_images": len(data.get("images", [])),
        "n_annotations": len(data.get("annotations", [])),
        "n_categories": len(data.get("categories", [])),
        "n_usable_annotations_estimated": usable,
        "n_missing_images": len(missing_image_ids),
        "missing_image_examples": missing_lvis_image_examples(images_dir, annotations_file, limit=10),
        "skipped_annotation_count": int(sum(reasons.values())),
        "skipped_annotation_reasons": dict(sorted(reasons.items())),
        "frequency_distribution": category_frequency_distribution(annotations_file),
    }


def _difficulty_tags(
    area_ratio: float,
    bbox_xyxy: list[float],
    gt_mask: Any,
    width: int,
    height: int,
    instances_in_image: int,
    frequency: Any,
) -> list[str]:
    tags: list[str] = []
    if area_ratio < 0.01:
        tags.append("small_object")
    elif area_ratio < 0.10:
        tags.append("medium_object")
    else:
        tags.append("large_object")
    if touches_border(gt_mask, bbox_xyxy, width, height):
        tags.append("touches_border")
    box_width = max(0.0, bbox_xyxy[2] - bbox_xyxy[0])
    box_height = max(0.0, bbox_xyxy[3] - bbox_xyxy[1])
    aspect = box_width / box_height if box_height else 0.0
    if aspect > 4.0 or (aspect > 0 and aspect < 0.25):
        tags.append("elongated_object")
    if instances_in_image >= 5:
        tags.append("crowded_or_multi_instance")
    frequency_tag = FREQUENCY_TAGS.get(str(frequency or ""))
    if frequency_tag:
        tags.append(frequency_tag)
    return tags


def _valid_annotation(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and row.get("segmentation") is not None
        and isinstance(row.get("bbox"), list)
        and len(row.get("bbox")) == 4
        and float(row.get("area") or 0) > 0
    )


def _image_file_name(image: dict) -> str:
    value = image.get("file_name")
    if value:
        return str(value)
    coco_url = str(image.get("coco_url") or "")
    if coco_url:
        return coco_url.rstrip("/").split("/")[-1]
    image_id = image.get("id")
    if image_id is not None:
        try:
            return f"{int(image_id):012d}.jpg"
        except (TypeError, ValueError):
            return f"{image_id}.jpg"
    return ""
