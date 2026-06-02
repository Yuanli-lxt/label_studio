import warnings
from pathlib import Path

import pytest
from PIL import Image

from image_segmentation.benchmark.datasets.mask_folder import iter_mask_folder_manifest_samples


def _write_pair(root: Path, mask_value=255, mask_size=(10, 10), image_size=(10, 10)):
    images = root / "images"
    masks = root / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", image_size, "white").save(images / "a.jpg")
    mask = Image.new("L", mask_size, 0)
    if mask_value:
        for x in range(2, 5):
            for y in range(3, 7):
                mask.putpixel((x, y), mask_value)
    mask.save(masks / "a.png")
    return images, masks


def test_mask_folder_loader_pairs_by_stem(tmp_path):
    images, masks = _write_pair(tmp_path)
    rows = list(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k"))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "a"
    assert rows[0]["mask_path"].endswith("a.png")


def test_mask_folder_loader_binary_threshold(tmp_path):
    images, masks = _write_pair(tmp_path, mask_value=128)
    row = next(iter(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k")))
    assert row["gt_area"] == 12


def test_mask_folder_loader_bbox_from_mask(tmp_path):
    images, masks = _write_pair(tmp_path)
    row = next(iter(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k")))
    assert row["gt_bbox_xyxy"] == [2.0, 3.0, 5.0, 7.0]


def test_dis5k_loader_pairs_image_and_mask_by_stem(tmp_path):
    images, masks = _write_pair(tmp_path)
    Image.new("RGB", (10, 10), "white").save(images / "unmatched.jpg")
    rows = list(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k"))
    assert [row["image_id"] for row in rows] == ["a"]


def test_dis5k_manifest_has_boundary_metadata(tmp_path):
    images, masks = _write_pair(tmp_path)
    row = next(iter(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k")))
    assert row["category_name"] == "foreground_object"
    assert "boundary_metadata" in row
    assert row["boundary_metadata"]["perimeter_px"] > 0


def test_mask_folder_loader_empty_mask_warning(tmp_path):
    images, masks = _write_pair(tmp_path, mask_value=0)
    with warnings.catch_warnings(record=True) as caught:
        rows = list(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k"))
    assert rows == []
    assert any("empty mask" in str(item.message) for item in caught)


def test_mask_folder_loader_shape_mismatch_error(tmp_path):
    images, masks = _write_pair(tmp_path, mask_size=(8, 8))
    with pytest.raises(ValueError, match="shape mismatch"):
        list(iter_mask_folder_manifest_samples(images, masks, "bench", "dis5k"))


def test_dis5k_config_exists():
    assert Path("configs/benchmark_v0_1.dis5k300.yaml").exists()


def test_cod10k_config_exists():
    assert Path("configs/benchmark_v0_1.cod10k300.yaml").exists()


def test_camo_config_exists():
    assert Path("configs/benchmark_v0_1.camo300.yaml").exists()


def test_cod10k_default_difficulty_tags(tmp_path):
    images, masks = _write_pair(tmp_path)
    row = next(
        iter(
            iter_mask_folder_manifest_samples(
                images, masks, "bench", "cod10k", category_name="camouflaged_object", default_difficulty_tags=["low_contrast", "camouflaged_object"]
            )
        )
    )
    assert "low_contrast" in row["difficulty_tags"]
    assert row["category_name"] == "camouflaged_object"


def test_camo_default_difficulty_tags(tmp_path):
    images, masks = _write_pair(tmp_path)
    row = next(
        iter(
            iter_mask_folder_manifest_samples(
                images, masks, "bench", "camo", category_name="camouflaged_object", default_difficulty_tags=["low_contrast", "camouflaged_object"]
            )
        )
    )
    assert "camouflaged_object" in row["difficulty_tags"]
