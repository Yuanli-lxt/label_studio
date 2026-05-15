import json
import unittest
from pathlib import Path


class RegressionProbeFileTests(unittest.TestCase):
    def test_probe_file_schema_and_categories(self):
        path = Path("demo_data/tasks/text_classification_regression_probes.json")
        self.assertTrue(path.exists(), f"missing probe file: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 5)
        self.assertLessEqual(len(data), 10)

        ids = set()
        categories = set()
        for item in data:
            self.assertIsInstance(item, dict)
            for key in ("id", "category", "text", "expected_label"):
                self.assertIn(key, item)
                self.assertIsInstance(item[key], str)
                self.assertTrue(item[key].strip())

            self.assertNotIn(item["id"], ids)
            ids.add(item["id"])
            categories.add(item["category"])

            self.assertIn(item["expected_label"], {"Positive", "Negative"})

            if "min_score" in item:
                self.assertIsInstance(item["min_score"], (int, float))
                self.assertGreaterEqual(float(item["min_score"]), 0.0)
                self.assertLessEqual(float(item["min_score"]), 1.0)

            if "expected_uncertain" in item:
                self.assertIsInstance(item["expected_uncertain"], bool)

        self.assertIn("obvious_positive", categories)
        self.assertIn("obvious_negative", categories)
        self.assertIn("nuanced_positive", categories)
        self.assertIn("subtle_negative", categories)


if __name__ == "__main__":
    unittest.main()
