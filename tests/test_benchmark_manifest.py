import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_segmentation.benchmark.build_manifest import build_manifest
from image_segmentation.benchmark.schema import REQUIRED_MANIFEST_FIELDS, validate_manifest_sample


class BenchmarkManifestTests(unittest.TestCase):
    def tiny_coco_project(self, root: Path, n_images=4, anns_per_image=3):
        images = root / "val2017"
        annotations = root / "annotations"
        images.mkdir()
        annotations.mkdir()
        coco_images = []
        coco_annotations = []
        ann_id = 1
        for image_id in range(1, n_images + 1):
            Image.new("RGB", (20, 20), "white").save(images / f"{image_id}.jpg")
            coco_images.append({"id": image_id, "file_name": f"{image_id}.jpg", "width": 20, "height": 20})
            for j in range(anns_per_image):
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": (j % 2) + 1,
                        "bbox": [1 + j, 2 + j, 5, 5],
                        "area": 25,
                        "segmentation": [[1 + j, 2 + j, 6 + j, 2 + j, 6 + j, 7 + j, 1 + j, 7 + j]],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        ann_file = annotations / "instances_val2017.json"
        ann_file.write_text(
            json.dumps(
                {
                    "images": coco_images,
                    "categories": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}],
                    "annotations": coco_annotations,
                }
            ),
            encoding="utf-8",
        )
        return images, ann_file

    def test_manifest_schema_from_tiny_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "val2017"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            Image.new("RGB", (10, 10), "white").save(images / "tiny.jpg")
            coco = {
                "images": [{"id": 1, "file_name": "tiny.jpg", "width": 10, "height": 10}],
                "categories": [{"id": 3, "name": "thing"}],
                "annotations": [
                    {
                        "id": 7,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [1, 2, 3, 4],
                        "area": 12,
                        "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
                        "iscrowd": 0,
                    }
                ],
            }
            ann_file = annotations / "instances_val2017.json"
            ann_file.write_text(json.dumps(coco), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1",
                        "random_seed: 42",
                        "max_samples_total: 10",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 10",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            build_manifest(str(config), str(output))
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            for field in REQUIRED_MANIFEST_FIELDS:
                self.assertIn(field, row)
            self.assertEqual([], validate_manifest_sample(row))
            self.assertEqual([1.0, 2.0, 4.0, 6.0], row["gt_bbox_xyxy"])
            self.assertGreater(row["gt_area_ratio"], 0)
            self.assertTrue(Path(row["image_path"]).exists())
            self.assertEqual("COCO", row["dataset"])
            self.assertTrue(row["sample_id"])

    def test_coco1000_config_exists(self):
        self.assertTrue(Path("configs/benchmark_v0_1.coco1000.yaml").exists())

    def test_coco100_config_exists(self):
        self.assertTrue(Path("configs/benchmark_v0_1.coco100.yaml").exists())

    def test_coco300_config_exists(self):
        self.assertTrue(Path("configs/benchmark_v0_1.coco300.yaml").exists())

    def test_manifest_respects_max_samples_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco1000",
                        "random_seed: 42",
                        "max_samples_total: 5",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 12",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            result = build_manifest(str(config), str(output))
            self.assertEqual(5, result["samples"])

    def test_manifest_image_diversity_max_instances_per_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root, n_images=4, anns_per_image=4)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco1000",
                        "random_seed: 42",
                        "max_samples_total: 8",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 16",
                        "sampling:",
                        "  strategy: stratified",
                        "  category_balance: true",
                        "  image_diversity: true",
                        "  max_instances_per_image: 2",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            build_manifest(str(config), str(output))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            counts = {}
            for row in rows:
                counts[row["image_id"]] = counts.get(row["image_id"], 0) + 1
            self.assertLessEqual(max(counts.values()), 2)

    def test_manifest_benchmark_id_coco1000(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root, n_images=1, anns_per_image=1)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco1000",
                        "random_seed: 42",
                        "max_samples_total: 1",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 1",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            build_manifest(str(config), str(output))
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("benchmark_v0_1_coco1000", row["benchmark_id"])
            self.assertEqual("benchmark_v0_1_coco1000", row["split"])

    def test_coco100_manifest_benchmark_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root, n_images=1, anns_per_image=1)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco100",
                        "random_seed: 42",
                        "max_samples_total: 1",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 1",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            build_manifest(str(config), str(output))
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("benchmark_v0_1_coco100", row["benchmark_id"])

    def test_coco300_manifest_benchmark_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root, n_images=1, anns_per_image=1)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco300",
                        "random_seed: 42",
                        "max_samples_total: 1",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 1",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            build_manifest(str(config), str(output))
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("benchmark_v0_1_coco300", row["benchmark_id"])

    def test_manifest_summary_outputs_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, ann_file = self.tiny_coco_project(root)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "benchmark_id: benchmark_v0_1_coco1000",
                        "random_seed: 42",
                        "max_samples_total: 4",
                        "datasets:",
                        "  coco:",
                        "    enabled: true",
                        f"    images_dir: {images}",
                        f"    annotations_file: {ann_file}",
                        "    max_samples: 12",
                    ]
                ),
                encoding="utf-8",
            )
            summary = root / "summary.json"
            build_manifest(str(config), str(root / "manifest.jsonl"), summary_output=str(summary))
            payload = json.loads(summary.read_text(encoding="utf-8"))
            for field in [
                "n_samples",
                "n_unique_images",
                "n_categories",
                "category_distribution_top20",
                "difficulty_tag_distribution",
                "max_instances_per_image_observed",
                "area_ratio_distribution",
                "random_seed",
            ]:
                self.assertIn(field, payload)
            self.assertTrue(payload["category_distribution_top20"])
            self.assertTrue(payload["difficulty_tag_distribution"])


if __name__ == "__main__":
    unittest.main()
