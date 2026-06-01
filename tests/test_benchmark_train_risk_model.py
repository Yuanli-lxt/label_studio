import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.train_risk_model import LEAKY_FEATURE_TOKENS, train_from_delta_dataset
from segmentation_correction_risk import FEATURE_NAMES


def record(idx, major):
    return {
        "task_id": f"s{idx}",
        "prediction_id": f"p{idx}",
        "annotation_id": f"a{idx}",
        "record_status": "ok",
        "image_width": 100,
        "image_height": 100,
        "prompt_bbox": [10, 10, 40, 40],
        "model_mask_bbox": [10, 10, 40, 40],
        "mask_quality": {
            "mask_area_ratio": 0.09 + idx * 0.001,
            "bbox_iou_prompt_mask": 0.2 if major else 0.9,
            "mask_touches_border": major,
            "valid_mask": True,
            "rle_length": 100 + idx,
            "image_width": 100,
            "image_height": 100,
        },
        "review": {
            "needs_review": major,
            "review_priority": "high" if major else "low",
            "review_priority_score": 80 if major else 0,
            "review_reason": ["low_prompt_mask_bbox_iou"] if major else [],
        },
        "uncertainty": {
            "enabled": True,
            "num_prompt_variants": 7,
            "num_valid_masks": 7,
            "mean_pairwise_iou": 0.4 if major else 0.95,
            "min_pairwise_iou": 0.2 if major else 0.9,
            "max_pairwise_iou": 0.8 if major else 1.0,
            "disagreement_area_ratio": 0.2 if major else 0.01,
            "stable": not major,
            "stability_bucket": "low" if major else "high",
            "reason": ["unstable"] if major else [],
        },
        "delta": {"major_correction": major, "correction_severity": "major" if major else "none"},
    }


class BenchmarkTrainRiskModelTests(unittest.TestCase):
    def test_train_risk_model_no_target_leakage(self):
        joined = "\n".join(FEATURE_NAMES).lower()
        for token in LEAKY_FEATURE_TOKENS:
            self.assertNotIn(token, joined)

    def test_train_risk_model_outputs_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            rows = [record(i, i % 2 == 0) for i in range(12)]
            delta.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            train_from_delta_dataset(str(delta), str(root / "risk"), test_size=0.25, random_seed=42)
            for name in ["classifier.joblib", "metadata.json", "feature_names.json", "training_dataset.jsonl", "evaluation.json"]:
                self.assertTrue((root / "risk" / name).exists(), name)

    def test_train_risk_model_single_class_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            delta.write_text("\n".join(json.dumps(record(i, True)) for i in range(6)) + "\n", encoding="utf-8")
            result = train_from_delta_dataset(str(delta), str(root / "risk"))
            self.assertEqual("single_class_training_data", result["evaluation"]["warning"])
            self.assertTrue((root / "risk" / "evaluation.json").exists())

    def test_train_risk_model_feature_names_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            rows = [record(i, i % 2 == 0) for i in range(12)]
            delta.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = train_from_delta_dataset(str(delta), str(root / "risk"))
            self.assertTrue(result["evaluation"]["feature_names_safe"])


if __name__ == "__main__":
    unittest.main()

