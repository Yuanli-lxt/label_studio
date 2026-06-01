import json
from pathlib import Path

from PIL import Image

from image_segmentation.benchmark.datasets.lvis import iter_lvis_manifest_samples, missing_lvis_images
from image_segmentation.benchmark.build_manifest import build_manifest
from image_segmentation.benchmark.schema import validate_manifest_sample


def _tiny_lvis(root: Path, include_missing=False):
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (10, 10), "white").save(images / "one.jpg")
    rows = [{"id": 1, "file_name": "one.jpg", "width": 10, "height": 10}]
    if include_missing:
        rows.append({"id": 2, "file_name": "missing.jpg", "width": 10, "height": 10})
    anns = [
        {
            "id": 7,
            "image_id": 1,
            "category_id": 3,
            "bbox": [1, 2, 3, 4],
            "area": 12,
            "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
            "iscrowd": 0,
        }
    ]
    if include_missing:
        anns.append({**anns[0], "id": 8, "image_id": 2})
    ann_file = root / "lvis.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": rows,
                "annotations": anns,
                "categories": [{"id": 3, "name": "thing", "frequency": "r"}],
            }
        ),
        encoding="utf-8",
    )
    return images, ann_file


def test_lvis_loader_parses_tiny_fixture(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    row = next(iter(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1)))
    assert row["dataset"] == "LVIS"
    assert row["category_name"] == "thing"


def test_lvis_bbox_xywh_to_xyxy(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    row = next(iter(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1)))
    assert row["gt_bbox_xyxy"] == [1.0, 2.0, 4.0, 6.0]


def test_lvis_category_frequency_tags(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    row = next(iter(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1)))
    assert "rare_category" in row["difficulty_tags"]


def test_lvis_missing_images_warning(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path, include_missing=True)
    assert missing_lvis_images(images, ann_file) == 1


def test_lvis_manifest_schema(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    row = next(iter(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1)))
    assert validate_manifest_sample(row) == []


def test_lvis_manifest_outputs_expected_schema(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: benchmark_v0_1_lvis500",
                "max_samples_total: 1",
                "datasets:",
                "  lvis:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    annotations_file: {ann_file}",
                "    max_samples: 1",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    build_manifest(str(config), str(output))
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["benchmark_id"] == "benchmark_v0_1_lvis500"
    assert row["dataset"] == "LVIS"
    assert row["split"] == "benchmark_v0_1_lvis500"


def test_lvis_manifest_has_category_frequency(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    row = next(iter(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1)))
    assert row["category_frequency"] == "r"


def test_lvis_manifest_summary_frequency_distribution(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: benchmark_v0_1_lvis500",
                "max_samples_total: 1",
                "datasets:",
                "  lvis:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    annotations_file: {ann_file}",
                "    max_samples: 1",
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    build_manifest(str(config), str(tmp_path / "manifest.jsonl"), summary_output=str(summary))
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["frequency_distribution"] == {"r": 1}
    assert data["missing_image_count"] == 0


def test_lvis_manifest_respects_max_instances_per_image(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: benchmark_v0_1_lvis500",
                "max_samples_total: 1",
                "datasets:",
                "  lvis:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    annotations_file: {ann_file}",
                "    max_samples: 1",
                "sampling:",
                "  image_diversity: true",
                "  max_instances_per_image: 1",
            ]
        ),
        encoding="utf-8",
    )
    result = build_manifest(str(config), str(tmp_path / "manifest.jsonl"))
    assert result["samples"] == 1


def test_lvis_manifest_does_not_use_delta_fields_for_sampling(tmp_path):
    images, ann_file = _tiny_lvis(tmp_path)
    rows = list(iter_lvis_manifest_samples(images, ann_file, "bench", max_samples=1))
    assert "delta" not in rows[0]
    assert "model_human_iou" not in rows[0]


def test_lvis_config_exists():
    assert Path("configs/benchmark_v0_1.lvis500.yaml").exists()
