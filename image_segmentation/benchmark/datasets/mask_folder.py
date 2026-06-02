from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from image_segmentation.benchmark.metrics import mask_bbox, mask_shape_metadata, touches_border


def iter_mask_folder_manifest_samples(
    images_dir: str | Path,
    masks_dir: str | Path,
    benchmark_id: str,
    dataset_name: str,
    max_samples: int | None = None,
    image_glob: str = "*.*",
    mask_glob: str = "*.png",
    mask_match_strategy: str = "same_stem",
    category_name: str = "foreground_object",
    category_id: str | None = None,
    default_difficulty_tags: list[str] | None = None,
) -> Iterable[dict]:
    if mask_match_strategy not in {"same_stem", "same_stem_strip_suffix"}:
        raise ValueError("mask_match_strategy must be 'same_stem' or 'same_stem_strip_suffix'")
    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    masks_by_stem = _paths_by_match_stem(mask_root, mask_glob, mask_match_strategy)
    count = 0
    for image_path in _files_matching(image_root, image_glob):
        if image_path.is_dir():
            continue
        mask_path = masks_by_stem.get(_match_stem(image_path, mask_match_strategy))
        if mask_path is None:
            continue
        sample = load_mask_folder_sample(
            image_path=image_path,
            mask_path=mask_path,
            benchmark_id=benchmark_id,
            dataset_name=dataset_name,
            category_name=category_name,
            category_id=category_id or category_name,
            default_difficulty_tags=default_difficulty_tags or [],
        )
        if sample is None:
            continue
        yield sample
        count += 1
        if max_samples is not None and count >= max_samples:
            break


def load_mask_folder_sample(
    image_path: str | Path,
    mask_path: str | Path,
    benchmark_id: str,
    dataset_name: str,
    category_name: str = "foreground_object",
    category_id: str = "foreground_object",
    default_difficulty_tags: list[str] | None = None,
) -> dict | None:
    from PIL import Image

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    with Image.open(image_path) as image:
        width, height = int(image.width), int(image.height)
    with Image.open(mask_path) as mask_image:
        mask_arr = np.asarray(mask_image.convert("L"))
    if mask_arr.shape != (height, width):
        raise ValueError(f"mask shape mismatch for {image_path.name}: image={(height, width)} mask={mask_arr.shape}")
    mask = (mask_arr > 127).astype(np.uint8)
    area = int(mask.sum())
    if area <= 0:
        warnings.warn(f"empty mask skipped: {mask_path}", RuntimeWarning)
        return None
    bbox = mask_bbox(mask)
    area_ratio = float(area / (width * height)) if width * height else 0.0
    boundary_metadata = mask_shape_metadata(mask, bbox)
    tags = _mask_difficulty_tags(mask, bbox, width, height, area_ratio, boundary_metadata)
    for tag in default_difficulty_tags or []:
        if tag not in tags:
            tags.append(tag)
    return {
        "benchmark_id": benchmark_id,
        "dataset": dataset_name.upper(),
        "sample_id": f"{dataset_name}_{image_path.stem}",
        "image_id": image_path.stem,
        "annotation_id": image_path.stem,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "width": width,
        "height": height,
        "category_id": str(category_id),
        "category_name": str(category_name),
        "gt_bbox_xyxy": bbox,
        "gt_area": area,
        "gt_area_ratio": area_ratio,
        "gt_touches_border": touches_border(mask, bbox, width, height),
        "difficulty_tags": tags,
        "boundary_metadata": boundary_metadata,
        "split": benchmark_id,
        "metadata": {"mask_path": str(mask_path), "mask_threshold": 127},
    }


def count_pairs(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_glob: str = "*.*",
    mask_glob: str = "*.png",
    mask_match_strategy: str = "same_stem",
) -> int:
    image_stems = {_match_stem(path, mask_match_strategy) for path in _files_matching(Path(images_dir), image_glob)}
    mask_stems = set(_paths_by_match_stem(Path(masks_dir), mask_glob, mask_match_strategy))
    return len(image_stems & mask_stems)


def mask_folder_stats(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_glob: str = "*.*",
    mask_glob: str = "*.png",
    mask_match_strategy: str = "same_stem",
) -> dict:
    from PIL import Image

    image_root = Path(images_dir)
    mask_root = Path(masks_dir)
    image_paths = _files_matching(image_root, image_glob)
    masks_by_stem = _paths_by_match_stem(mask_root, mask_glob, mask_match_strategy)
    stats = {
        "n_images_found": len(image_paths),
        "n_masks_found": len(masks_by_stem),
        "n_pairs": 0,
        "missing_mask_count": 0,
        "missing_image_count": 0,
        "n_valid": 0,
        "n_empty_masks": 0,
        "n_shape_mismatch": 0,
    }
    image_stems = {_match_stem(path, mask_match_strategy) for path in image_paths}
    mask_stems = set(masks_by_stem)
    stats["missing_mask_count"] = len(image_stems - mask_stems)
    stats["missing_image_count"] = len(mask_stems - image_stems)
    for image_path in image_paths:
        if not image_path.is_file():
            continue
        mask_path = masks_by_stem.get(_match_stem(image_path, mask_match_strategy))
        if mask_path is None:
            continue
        stats["n_pairs"] += 1
        with Image.open(image_path) as image:
            image_shape = (int(image.height), int(image.width))
        with Image.open(mask_path) as mask_image:
            mask_arr = np.asarray(mask_image.convert("L"))
        if mask_arr.shape != image_shape:
            stats["n_shape_mismatch"] += 1
            continue
        if int((mask_arr > 127).sum()) <= 0:
            stats["n_empty_masks"] += 1
            continue
        stats["n_valid"] += 1
    return stats


def _files_matching(root: Path, glob_spec: str) -> list[Path]:
    patterns = [part.strip() for part in glob_spec.replace(";", ",").split(",") if part.strip()]
    if not patterns:
        patterns = ["*.*"]
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                paths[str(path)] = path
    return sorted(paths.values())


def _paths_by_match_stem(root: Path, glob_spec: str, strategy: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in _files_matching(root, glob_spec):
        out.setdefault(_match_stem(path, strategy), path)
    return out


def _match_stem(path: Path, strategy: str) -> str:
    stem = path.stem
    if strategy == "same_stem_strip_suffix":
        lowered = stem.lower()
        for suffix in ("_mask", "-mask", "_gt", "-gt", "_label", "-label", "_seg", "-seg"):
            if lowered.endswith(suffix):
                return stem[: -len(suffix)]
    return stem


def _mask_difficulty_tags(
    mask: np.ndarray,
    bbox: list[float] | None,
    width: int,
    height: int,
    area_ratio: float,
    boundary_metadata: dict | None = None,
) -> list[str]:
    tags: list[str] = []
    if area_ratio < 0.01:
        tags.append("small_object")
    elif area_ratio < 0.10:
        tags.append("medium_object")
    else:
        tags.append("large_object")
    if touches_border(mask, bbox, width, height):
        tags.append("touches_border")
    if bbox:
        box_width = max(0.0, bbox[2] - bbox[0])
        box_height = max(0.0, bbox[3] - bbox[1])
        aspect = box_width / box_height if box_height else 0.0
        if aspect > 4.0 or (aspect > 0 and aspect < 0.25):
            tags.append("elongated_object")
    boundary_metadata = boundary_metadata or {}
    if float(boundary_metadata.get("thin_structure_score") or 0.0) >= 0.60:
        tags.append("thin_structure")
    if float(boundary_metadata.get("perimeter_area_ratio") or 0.0) >= 0.50:
        tags.append("high_boundary_complexity")
    return tags
