import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.run_benchmark import TRAINER_DIR

import sys
import inspect

if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_correction_risk import FEATURE_NAMES  # noqa: E402
from segmentation_review_queue import score_review_candidate  # noqa: E402
from image_segmentation.benchmark import sampling  # noqa: E402
from image_segmentation.benchmark.learn_boundary_shape_fusion import BOUNDARY_FEATURES, CURRENT_PLUS_FEATURES  # noqa: E402
from image_segmentation.benchmark.prediction_feature_scoring import assert_no_leaky_feature_names  # noqa: E402
from image_segmentation.benchmark.shadow_scoring import shadow_scores_for_item  # noqa: E402


LEAKY_TOKENS = {
    "delta",
    "model_human_iou",
    "correction_area_ratio",
    "major_correction",
    "correction_severity",
}


def candidate(delta_major: bool) -> dict:
    return {
        "task_id": "same",
        "image_width": 100,
        "image_height": 100,
        "prompt_bbox": [10, 10, 40, 40],
        "model_mask_bbox": [10, 10, 40, 40],
        "mask_quality": {
            "mask_area_ratio": 0.09,
            "bbox_iou_prompt_mask": 1.0,
            "mask_touches_border": False,
            "valid_mask": True,
            "rle_length": 10,
        },
        "review": {"needs_review": False, "review_priority": "low", "review_priority_score": 0, "review_reason": []},
        "uncertainty": {"enabled": False},
        "delta": {
            "major_correction": delta_major,
            "correction_severity": "major" if delta_major else "none",
            "model_human_iou": 0.1 if delta_major else 1.0,
            "correction_area_ratio": 0.5 if delta_major else 0.0,
        },
    }


class BenchmarkLeakageTests(unittest.TestCase):
    def test_no_target_leakage_in_risk_features(self):
        joined = "\n".join(FEATURE_NAMES)
        for token in LEAKY_TOKENS:
            self.assertNotIn(token, joined)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_names.json"
            path.write_text(json.dumps(FEATURE_NAMES), encoding="utf-8")
            saved = "\n".join(json.loads(path.read_text(encoding="utf-8")))
            for token in LEAKY_TOKENS:
                self.assertNotIn(token, saved)

    def test_review_queue_does_not_use_delta(self):
        first = score_review_candidate(candidate(False))
        second = score_review_candidate(candidate(True))
        self.assertEqual(first["priority_score"], second["priority_score"])
        self.assertEqual(first["score_components"], second["score_components"])
        self.assertFalse(first["evaluation_only"]["delta"]["major_correction"])
        self.assertTrue(second["evaluation_only"]["delta"]["major_correction"])

    def test_review_queue_ignores_gt_boundary_metadata_and_boundary_delta(self):
        base = candidate(False)
        changed = candidate(False)
        changed["boundary_metadata"] = {
            "perimeter_px": 9999.0,
            "perimeter_area_ratio": 999.0,
            "thin_structure_score": 1.0,
        }
        changed["delta"].update(
            {
                "model_human_iou": 0.0,
                "dice": 0.0,
                "boundary": {
                    "boundary_iou": 0.0,
                    "boundary_f1": 0.0,
                    "boundary_precision": 0.0,
                    "boundary_recall": 0.0,
                    "boundary_error_area_ratio": 1.0,
                },
            }
        )
        first = score_review_candidate(base)
        second = score_review_candidate(changed)
        self.assertEqual(first["priority_score"], second["priority_score"])
        self.assertEqual(first["score_components"], second["score_components"])
        self.assertNotIn("boundary_metadata", first["source_metadata"])
        self.assertNotIn("boundary_metadata", second["source_metadata"])

    def test_shadow_scores_ignore_gt_only_fields(self):
        first = score_review_candidate(candidate(False))
        changed = score_review_candidate(candidate(True))
        changed["boundary_metadata"] = {"perimeter_area_ratio": 999.0}
        changed["evaluation_only"]["delta"]["boundary"] = {"boundary_f1": 0.0}
        self.assertEqual(shadow_scores_for_item(first), shadow_scores_for_item(changed))

    def test_learned_boundary_feature_sets_reject_gt_derived_fields(self):
        assert_no_leaky_feature_names(BOUNDARY_FEATURES)
        assert_no_leaky_feature_names(CURRENT_PLUS_FEATURES)
        with self.assertRaises(ValueError):
            assert_no_leaky_feature_names(["pred_area_ratio", "model_human_iou"])

    def test_manifest_does_not_use_delta_fields_for_sampling(self):
        source = inspect.getsource(sampling.sample_manifest_rows)
        for token in LEAKY_TOKENS:
            self.assertNotIn(token, source)

    def test_runtime_metadata_not_used_in_risk_features(self):
        joined = "\n".join(FEATURE_NAMES)
        for token in ("runtime", "cuda", "device", "elapsed_seconds", "seconds_per_sample"):
            self.assertNotIn(token, joined)


if __name__ == "__main__":
    unittest.main()
