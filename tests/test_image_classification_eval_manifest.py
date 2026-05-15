import json
import unittest
from pathlib import Path


class ImageEvalManifestTests(unittest.TestCase):
    def test_manifest_schema_and_balance(self):
        path = Path("demo_data/tasks/image_classification_eval_manifest.json")
        self.assertTrue(path.exists(), f"missing eval manifest: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        self.assertEqual("image-cls-eval-v1", data.get("eval_set_id"))

        images = data.get("images")
        self.assertIsInstance(images, list)
        self.assertEqual(8, len(images))

        seen = set()
        label_counts = {"Product": 0, "Other": 0}
        for item in images:
            self.assertIsInstance(item, dict)
            filename = item.get("filename")
            label = item.get("label")
            self.assertIsInstance(filename, str)
            self.assertTrue(filename.endswith(".png"))
            self.assertNotIn(filename, seen)
            seen.add(filename)
            self.assertIn(label, {"Product", "Other"})
            label_counts[label] += 1

        self.assertEqual({"Product": 4, "Other": 4}, label_counts)


if __name__ == "__main__":
    unittest.main()
