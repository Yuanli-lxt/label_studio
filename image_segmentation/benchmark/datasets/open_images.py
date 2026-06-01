from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from image_segmentation.benchmark.datasets.mask_folder import load_mask_folder_sample


def iter_open_images_manifest_samples(
    dataset_dir: str | Path,
    benchmark_id: str,
    max_samples: int | None = None,
) -> Iterable[dict]:
    root = Path(dataset_dir)
    annotations = root / "annotations.jsonl"
    count = 0
    with annotations.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = root / str(row.get("image_path") or row.get("image") or "")
            mask_path = root / str(row.get("mask_path") or row.get("mask") or "")
            label = str(row.get("label") or row.get("category_name") or "object")
            sample = load_mask_folder_sample(
                image_path=image_path,
                mask_path=mask_path,
                benchmark_id=benchmark_id,
                dataset_name="open_images_v7",
                category_name=label,
                category_id=str(row.get("label_id") or label),
                default_difficulty_tags=_metadata_tags(row),
            )
            if sample is None:
                continue
            sample["metadata"].update({key: row[key] for key in ("is_occluded", "is_truncated", "is_group_of") if key in row})
            yield sample
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def _metadata_tags(row: dict) -> list[str]:
    tags: list[str] = []
    for key, tag in (
        ("is_occluded", "is_occluded"),
        ("is_truncated", "is_truncated"),
        ("is_group_of", "is_group_of"),
    ):
        if row.get(key) is True or row.get(key) == 1:
            tags.append(tag)
    return tags
