import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_image_review_tasks_from_metadata.sh"


class ExportImageReviewTasksTests(unittest.TestCase):
    def test_export_review_tasks_from_metadata(self):
        self.assertTrue(SCRIPT.exists(), f"missing script: {SCRIPT}")

        metadata = {
            "model_version": "image-cls-v9999",
            "metrics": {
                "eval_predictions": [
                    {
                        "filename": "demo_a.png",
                        "image": "/data/local-files/?d=images/demo_a.png",
                        "true_label": "Product",
                        "predicted_label": "Product",
                        "confidence": 0.95,
                        "correct": True,
                    },
                    {
                        "filename": "demo_b.png",
                        "image": "/data/local-files/?d=images/demo_b.png",
                        "true_label": "Other",
                        "predicted_label": "Other",
                        "confidence": 0.62,
                        "correct": True,
                    },
                    {
                        "filename": "demo_c.png",
                        "image": "/data/local-files/?d=images/demo_c.png",
                        "true_label": "Product",
                        "predicted_label": "Other",
                        "confidence": 0.93,
                        "correct": False,
                    },
                ],
                "eval_misclassified": [
                    {
                        "filename": "demo_c.png",
                        "image": "/data/local-files/?d=images/demo_c.png",
                        "true_label": "Product",
                        "predicted_label": "Other",
                        "confidence": 0.93,
                        "correct": False,
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "review_tasks.json"
            metadata_path = tmp_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            env = os.environ.copy()
            env["IMAGE_REVIEW_METADATA_PATH"] = str(metadata_path)
            subprocess.run(
                [str(SCRIPT), "0.7", str(out_path)],
                check=True,
                cwd=str(ROOT),
                env=env,
            )
            self.assertTrue(out_path.exists())
            tasks = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(2, len(tasks))
        filenames = {task.get("meta", {}).get("filename") for task in tasks}
        self.assertEqual({"demo_b.png", "demo_c.png"}, filenames)
        for task in tasks:
            self.assertIn("image", task.get("data", {}))
            self.assertIn("review_reason", task.get("meta", {}))


if __name__ == "__main__":
    unittest.main()
