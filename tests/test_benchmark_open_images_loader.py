import json
from pathlib import Path

from PIL import Image

from image_segmentation.benchmark.datasets.open_images import iter_open_images_manifest_samples
from image_segmentation.benchmark.download_data import main


def test_open_images_download_requires_fiftyone(tmp_path, capsys):
    assert main(["--dataset", "open_images_v7_segmentations", "--output-dir", str(tmp_path), "--use-fiftyone", "true"]) == 1
    assert "FiftyOne" in capsys.readouterr().out or "export wiring" in capsys.readouterr().out


def test_open_images_subset_config_exists():
    assert Path("configs/benchmark_v0_1.open_images500.yaml").exists()


def test_open_images_loader_tiny_fixture_if_implemented(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "segmentations").mkdir()
    Image.new("RGB", (6, 6), "white").save(tmp_path / "images" / "one.jpg")
    mask = Image.new("L", (6, 6), 0)
    mask.putpixel((2, 2), 255)
    mask.save(tmp_path / "segmentations" / "one.png")
    (tmp_path / "annotations.jsonl").write_text(
        json.dumps({"image_path": "images/one.jpg", "mask_path": "segmentations/one.png", "label": "person"}) + "\n",
        encoding="utf-8",
    )
    row = next(iter(iter_open_images_manifest_samples(tmp_path, "bench")))
    assert row["category_name"] == "person"


def test_open_images_metadata_tags_if_present(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "segmentations").mkdir()
    Image.new("RGB", (6, 6), "white").save(tmp_path / "images" / "one.jpg")
    mask = Image.new("L", (6, 6), 0)
    mask.putpixel((2, 2), 255)
    mask.save(tmp_path / "segmentations" / "one.png")
    (tmp_path / "annotations.jsonl").write_text(
        json.dumps(
            {
                "image_path": "images/one.jpg",
                "mask_path": "segmentations/one.png",
                "label": "person",
                "is_occluded": True,
                "is_truncated": True,
                "is_group_of": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = next(iter(iter_open_images_manifest_samples(tmp_path, "bench")))
    assert {"is_occluded", "is_truncated", "is_group_of"}.issubset(row["difficulty_tags"])
