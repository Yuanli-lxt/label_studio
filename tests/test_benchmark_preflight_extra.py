import json
from pathlib import Path

from PIL import Image

from image_segmentation.benchmark.preflight import run_preflight


def test_preflight_json_output(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (5, 5), "white").save(images / "a.jpg")
    mask = Image.new("L", (5, 5), 0)
    mask.putpixel((2, 2), 255)
    mask.save(masks / "a.png")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench",
                "datasets:",
                "  dis5k:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    masks_dir: {masks}",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "preflight.json"
    ok, lines = run_preflight(str(config), json_output=str(out))
    assert ok
    assert out.exists()
    assert json.loads(out.read_text())["datasets"][0]["loader_type"] == "mask_folder"
    assert json.loads(out.read_text())["datasets"][0]["n_pairs"] == 1
    assert any("build_manifest" in line for line in lines)


def test_dis5k_preflight_json_output_fields(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (5, 5), "white").save(images / "a.jpg")
    mask = Image.new("L", (5, 5), 0)
    mask.putpixel((2, 2), 255)
    mask.save(masks / "a.png")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["benchmark_id: bench", "datasets:", "  dis5k:", "    enabled: true", f"    images_dir: {images}", f"    masks_dir: {masks}"]),
        encoding="utf-8",
    )
    out = tmp_path / "preflight.json"
    ok, _ = run_preflight(str(config), json_output=str(out))
    report = json.loads(out.read_text())["datasets"][0]
    assert ok
    assert report["dataset"] == "DIS5K"
    assert report["status"] == "ok"
    assert report["n_images"] == 1
    assert report["n_masks"] == 1
    assert report["n_pairs"] == 1
    assert report["sample_check"]["gt_area"] == 1


def test_dis5k_preflight_detects_empty_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (5, 5), "white").save(images / "a.jpg")
    Image.new("L", (5, 5), 0).save(masks / "a.png")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["benchmark_id: bench", "datasets:", "  dis5k:", "    enabled: true", f"    images_dir: {images}", f"    masks_dir: {masks}"]),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("empty_masks=1" in line for line in lines)


def test_dis5k_preflight_detects_size_mismatch(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (5, 5), "white").save(images / "a.jpg")
    Image.new("L", (4, 4), 255).save(masks / "a.png")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["benchmark_id: bench", "datasets:", "  dis5k:", "    enabled: true", f"    images_dir: {images}", f"    masks_dir: {masks}"]),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("shape_mismatch=1" in line for line in lines)


def test_mask_folder_preflight_reports_include_exclude_filter_stats(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem, mask_value in [("COD10K-CAM-good", 255), ("COD10K-NonCAM-empty", 0)]:
        Image.new("RGB", (5, 5), "white").save(images / f"{stem}.jpg")
        mask = Image.new("L", (5, 5), 0)
        if mask_value:
            mask.putpixel((2, 2), mask_value)
        mask.save(masks / f"{stem}.png")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench",
                "datasets:",
                "  cod10k:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    masks_dir: {masks}",
                "    include_image_patterns:",
                "      - COD10K-CAM-*",
                "    include_mask_patterns:",
                "      - COD10K-CAM-*",
                "    exclude_image_patterns:",
                "      - COD10K-NonCAM-*",
                "    exclude_mask_patterns:",
                "      - COD10K-NonCAM-*",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "preflight.json"
    ok, lines = run_preflight(str(config), json_output=str(out))
    report = json.loads(out.read_text(encoding="utf-8"))["datasets"][0]
    assert ok
    assert report["n_images_scanned"] == 2
    assert report["n_masks_scanned"] == 2
    assert report["n_images"] == 1
    assert report["n_masks"] == 1
    assert report["n_images_excluded"] == 1
    assert report["n_masks_excluded"] == 1
    assert report["n_empty_masks"] == 0
    assert any("file filters excluded images=1 masks=1" in line for line in lines)


def test_preflight_reports_loader_type(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench",
                "datasets:",
                "  dis5k:",
                "    enabled: true",
                f"    images_dir: {tmp_path / 'missing_images'}",
                f"    masks_dir: {tmp_path / 'missing_masks'}",
            ]
        ),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("loader_type=mask_folder" in line for line in lines)


def test_preflight_suggests_build_manifest_command(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench",
                "datasets:",
                "  cod10k:",
                "    enabled: true",
                f"    images_dir: {tmp_path / 'images'}",
                f"    masks_dir: {tmp_path / 'masks'}",
            ]
        ),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("build_manifest" in line for line in lines)


def test_preflight_reports_manual_dataset_missing_dirs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench",
                "datasets:",
                "  cod10k:",
                "    enabled: true",
                f"    images_dir: {tmp_path / 'images'}",
                f"    masks_dir: {tmp_path / 'masks'}",
            ]
        ),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("missing" in line for line in lines)


def _tiny_lvis_preflight(root: Path, include_missing=False):
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (10, 10), "white").save(images / "one.jpg")
    ann = {
        "images": [{"id": 1, "file_name": "one.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "thing", "frequency": "f"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                "iscrowd": 0,
            }
        ],
    }
    if include_missing:
        ann["images"].append({"id": 2, "file_name": "missing.jpg", "width": 10, "height": 10})
        ann["annotations"].append({**ann["annotations"][0], "id": 2, "image_id": 2})
    ann_file = root / "lvis.json"
    ann_file.write_text(json.dumps(ann), encoding="utf-8")
    config = root / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark_id: bench_lvis",
                "datasets:",
                "  lvis:",
                "    enabled: true",
                f"    images_dir: {images}",
                f"    annotations_file: {ann_file}",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_lvis_preflight_success_with_tiny_fixture(tmp_path):
    ok, lines = run_preflight(str(_tiny_lvis_preflight(tmp_path)))
    assert ok
    assert any("parsed one valid LVIS sample" in line for line in lines)


def test_lvis_preflight_missing_annotation_fails_cleanly(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(["benchmark_id: bench", "datasets:", "  lvis:", "    enabled: true", f"    images_dir: {images}", f"    annotations_file: {tmp_path / 'missing.json'}"]),
        encoding="utf-8",
    )
    ok, lines = run_preflight(str(config))
    assert not ok
    assert any("annotations_file missing" in line for line in lines)


def test_lvis_preflight_missing_images_reports_examples(tmp_path):
    ok, lines = run_preflight(str(_tiny_lvis_preflight(tmp_path, include_missing=True)))
    assert ok
    assert any("missing.jpg" in line for line in lines)


def test_lvis_preflight_json_output_fields(tmp_path):
    out = tmp_path / "preflight.json"
    ok, _ = run_preflight(str(_tiny_lvis_preflight(tmp_path)), json_output=str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    report = data["datasets"][0]
    assert ok
    assert report["dataset"] == "LVIS"
    assert report["n_images"] == 1
    assert report["n_annotations"] == 1
    assert report["n_categories"] == 1
    assert report["sample_check"]["mask_decoded"] is True
