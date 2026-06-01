import subprocess
import sys

from image_segmentation.benchmark.dataset_registry import DATASET_REGISTRY


def test_dataset_registry_contains_expected_datasets():
    assert {
        "coco_val2017",
        "lvis_val",
        "dis5k",
        "cod10k",
        "camo",
        "open_images_v7_segmentations",
    }.issubset(DATASET_REGISTRY)


def test_dataset_registry_download_modes():
    assert DATASET_REGISTRY["coco_val2017"].download_mode == "scripted"
    assert DATASET_REGISTRY["lvis_val"].supports_scripted_download is True
    assert DATASET_REGISTRY["dis5k"].download_mode == "manual_or_external_archive"
    assert DATASET_REGISTRY["open_images_v7_segmentations"].download_mode == "optional_fiftyone_subset"


def test_list_datasets_cli_outputs_entries():
    result = subprocess.run(
        [sys.executable, "-m", "image_segmentation.benchmark.list_datasets"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "lvis_val" in result.stdout
    assert "open_images_v7_segmentations" in result.stdout
