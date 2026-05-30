import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "services" / "trainer" / "segmentation_correction_risk.py"


def load_module():
    spec = importlib.util.spec_from_file_location("segmentation_correction_risk", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def correction_record(major=False, idx=1, include_uncertainty=True):
    severity = "major" if major else "minor"
    record = {
        "record_status": "ok",
        "task_id": idx,
        "prediction_id": idx + 100,
        "annotation_id": idx + 200,
        "image_width": 100,
        "image_height": 80,
        "prompt_bbox": [10, 10, 50, 50],
        "model_mask_bbox": [12, 12, 48, 48],
        "mask_quality": {
            "mask_area_ratio": 0.35 if major else 0.08,
            "bbox_iou_prompt_mask": 0.25 if major else 0.9,
            "mask_touches_border": major,
            "valid_mask": True,
            "rle_length": 180 if major else 40,
            "image_width": 100,
            "image_height": 80,
        },
        "review": {
            "needs_review": major,
            "review_priority": "high" if major else "low",
            "review_priority_score": 90 if major else 0,
            "review_reason": ["low_prompt_mask_bbox_iou"] if major else [],
        },
        "delta": {
            "major_correction": major,
            "correction_severity": severity,
            "model_human_iou": 0.3 if major else 0.92,
            "correction_area_ratio": 0.2 if major else 0.01,
        },
    }
    if include_uncertainty:
        record["uncertainty"] = {
            "enabled": True,
            "num_prompt_variants": 7,
            "num_valid_masks": 7,
            "mean_pairwise_iou": 0.45 if major else 0.95,
            "min_pairwise_iou": 0.20 if major else 0.88,
            "max_pairwise_iou": 0.80 if major else 1.0,
            "disagreement_area_ratio": 0.25 if major else 0.01,
            "stable": not major,
            "stability_bucket": "low" if major else "high",
            "reason": ["unstable_masks"] if major else [],
        }
    return record


class SegmentationCorrectionRiskTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_feature_extraction_uses_precorrection_metadata_without_delta_leakage(self):
        record = correction_record(major=True)

        features = self.module.extract_correction_risk_features(record)

        self.assertIn("mask_quality.mask_area_ratio", features)
        self.assertIn("review.review_priority_level", features)
        self.assertIn("uncertainty.disagreement_area_ratio", features)
        self.assertIn("geometry.prompt_bbox_area_ratio", features)
        forbidden = {
            "delta.model_human_iou",
            "delta.model_human_dice",
            "delta.correction_area_ratio",
            "delta.added_area_ratio",
            "delta.removed_area_ratio",
            "delta.major_correction",
            "delta.correction_severity",
        }
        self.assertTrue(forbidden.isdisjoint(features.keys()))

    def test_missing_metadata_is_imputed_deterministically(self):
        features = self.module.extract_correction_risk_features({"image_width": 100, "image_height": 50})

        self.assertEqual(set(self.module.FEATURE_NAMES), set(features.keys()))
        self.assertEqual(0.0, features["mask_quality.mask_area_ratio"])
        self.assertEqual(2.0, features["geometry.image_aspect_ratio"])
        self.assertEqual(-1.0, features["uncertainty.stability_bucket_level"])

    def test_dataset_building_skips_records_without_labels(self):
        records = [
            correction_record(major=True, idx=1),
            correction_record(major=False, idx=2),
            {"record_status": "skipped", "delta": {"major_correction": True}},
            {"record_status": "ok", "delta": {}},
        ]

        rows, labels, summary = self.module.build_correction_risk_dataset(records)

        self.assertEqual(2, len(rows))
        self.assertEqual([1, 0], labels)
        self.assertEqual(2, summary["usable_records"])
        self.assertEqual(1, summary["positive_records"])
        self.assertEqual(1, summary["negative_records"])
        self.assertEqual(2, summary["skipped_records"])

    def test_quality_guard_too_few_records_skips_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.module.train_correction_risk_model(
                [correction_record(major=True), correction_record(major=False)],
                tmp,
            )

            risk = metadata["correction_risk"]
            self.assertEqual("skipped", risk["status"])
            self.assertEqual("not_enough_records", risk["skip_reason"])
            self.assertFalse((Path(tmp) / "classifier.joblib").exists())
            self.assertTrue((Path(tmp) / "metadata.json").exists())

    def test_quality_guard_one_class_skips_training(self):
        records = [correction_record(major=True, idx=i) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.module.train_correction_risk_model(records, tmp)

            risk = metadata["correction_risk"]
            self.assertEqual("skipped", risk["status"])
            self.assertEqual("not_enough_negative_records", risk["skip_reason"])

    def test_successful_training_writes_artifacts_and_metadata(self):
        records = [correction_record(major=idx % 2 == 0, idx=idx) for idx in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self.module.train_correction_risk_model(records, tmp)

            risk = metadata["correction_risk"]
            self.assertEqual("trained", risk["status"])
            self.assertEqual("LogisticRegression", risk["model_type"])
            self.assertTrue((Path(tmp) / "classifier.joblib").exists())
            self.assertTrue((Path(tmp) / "metadata.json").exists())
            self.assertTrue((Path(tmp) / "feature_names.json").exists())
            self.assertTrue((Path(tmp) / "training_dataset.jsonl").exists())
            self.assertIn("train_accuracy", risk["metrics"])
            self.assertGreater(len(risk["feature_weights"]), 0)
            self.assertEqual(
                self.module.FEATURE_NAMES,
                json.loads((Path(tmp) / "feature_names.json").read_text(encoding="utf-8")),
            )

    def test_missing_sklearn_writes_error_metadata_without_fallback(self):
        records = [correction_record(major=idx % 2 == 0, idx=idx) for idx in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                self.module,
                "_sklearn_training_dependencies",
                side_effect=ImportError("sklearn"),
            ):
                metadata = self.module.train_correction_risk_model(records, tmp)

            risk = metadata["correction_risk"]
            self.assertEqual("error", risk["status"])
            self.assertEqual("missing_scikit_learn", risk["skip_reason"])
            self.assertEqual("missing_scikit_learn", risk["error"])
            self.assertIn("scikit-learn is required", risk["message"])
            self.assertEqual("LogisticRegression", risk["model_type"])
            self.assertFalse((Path(tmp) / "classifier.joblib").exists())
            self.assertTrue((Path(tmp) / "metadata.json").exists())

    def test_prediction_helper_returns_risk_fields(self):
        records = [correction_record(major=idx % 2 == 0, idx=idx) for idx in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            self.module.train_correction_risk_model(records, tmp)

            prediction = self.module.predict_correction_risk(correction_record(major=True, idx=999), tmp)

            risk = prediction["correction_risk"]
            self.assertIsInstance(risk["risk_score"], float)
            self.assertIn(risk["risk_bucket"], ["low", "medium", "high"])
            self.assertIsInstance(risk["predicted_major_correction"], bool)
            self.assertIsInstance(risk["top_risk_features"], list)


if __name__ == "__main__":
    unittest.main()
