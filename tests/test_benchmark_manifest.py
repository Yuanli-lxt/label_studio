import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_segmentation.benchmark.build_manifest import build_manifest
from image_segmentation.benchmark.schema import REQUIRED_MANIFEST_FIELDS, validate_manifest_sample


class BenchmarkManifestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

