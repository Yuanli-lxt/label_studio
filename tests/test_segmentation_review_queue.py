import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "services" / "trainer" / "segmentation_review_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("segmentation_review_queue", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def candidate(idx=1, risk=0.0, uncertainty=True, delta_major=False, area=0.08):
    risk_like = bool(risk is not None and risk >= 0.7)
    row = {
        "task_id": idx,
        "image": f"/data/local-files/?d=images/demo_{idx}.png",
        "prediction_id": idx + 100,
        "model_version": "mobilesam-seg-v0001",
        "label": "Object",
        "image_width": 100,
        "image_height": 80,
        "prompt_bbox": [10, 10, 50, 50],
        "model_mask_bbox": [12, 12, 48, 48],
        "mask_quality": {
            "mask_area_ratio": area,
            "bbox_iou_prompt_mask": 0.3 if risk_like else 0.9,
            "mask_touches_border": risk_like,
            "valid_mask": True,
            "rle_length": 120,
            "prediction_time_boundary_shape": {
                "pred_area_ratio": area,
                "pred_bbox_area_ratio": area * 1.2,
                "pred_extent": 0.4 if risk_like else 0.9,
                "pred_aspect_ratio": 5.0 if risk_like else 1.0,
                "pred_touches_border": risk_like,
                "pred_boundary_complexity": 8.0 if risk_like else 1.2,
                "pred_boundary_density": 12.0 if risk_like else 2.0,
                "pred_component_count": 3 if risk_like else 1,
                "pred_largest_component_ratio": 0.6 if risk_like else 1.0,
                "pred_hole_count": 1 if risk_like else 0,
                "pred_thinness_proxy": 0.7 if risk_like else 0.1,
            },
        },
        "prediction_features": {
            "pred_area_ratio": area,
            "pred_bbox_area_ratio": area * 1.2,
            "pred_extent": 0.4 if risk_like else 0.9,
            "pred_aspect_ratio": 5.0 if risk_like else 1.0,
            "pred_touches_border": risk_like,
            "pred_boundary_complexity": 8.0 if risk_like else 1.2,
            "pred_boundary_density": 12.0 if risk_like else 2.0,
            "pred_component_count": 3 if risk_like else 1,
            "pred_largest_component_ratio": 0.6 if risk_like else 1.0,
            "pred_hole_count": 1 if risk_like else 0,
            "pred_thinness_proxy": 0.7 if risk_like else 0.1,
        },
        "review": {
            "needs_review": risk_like,
            "review_priority": "high" if risk_like else "low",
            "review_priority_score": 90 if risk_like else 0,
            "review_reason": ["low_prompt_mask_bbox_iou"] if risk_like else [],
        },
        "delta": {
            "major_correction": delta_major,
            "correction_severity": "major" if delta_major else "none",
            "model_human_iou": 0.1 if delta_major else 1.0,
        },
    }
    if uncertainty:
        row["uncertainty"] = {
            "enabled": True,
            "num_prompt_variants": 7,
            "num_valid_masks": 7,
            "mean_pairwise_iou": 0.4 if risk_like else 0.95,
            "min_pairwise_iou": 0.2 if risk_like else 0.9,
            "max_pairwise_iou": 0.8 if risk_like else 1.0,
            "disagreement_area_ratio": 0.25 if risk_like else 0.01,
            "stable": not risk_like,
            "stability_bucket": "low" if risk_like else "high",
            "reason": ["unstable_masks"] if risk_like else [],
        }
    if risk is not None:
        row["correction_risk"] = {
            "risk_score": risk,
            "risk_bucket": "high" if risk >= 0.7 else "low",
            "predicted_major_correction": risk >= 0.5,
        }
    return row


class SegmentationReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_score_candidate_with_full_metadata(self):
        item = self.module.score_review_candidate(candidate(risk=0.82))

        self.assertIsInstance(item["priority_score"], float)
        self.assertIn(item["priority_bucket"], ["high", "medium", "low"])
        self.assertIn("correction_risk_score", item["score_components"])
        self.assertIn("uncertainty_score", item["score_components"])
        self.assertIn("boundary_shape_score", item["score_components"])
        self.assertIsInstance(item["review_reasons"], list)
        self.assertIn("high_correction_risk", item["review_reasons"])
        self.assertIn("source_metadata", item)

    def test_missing_uncertainty_is_deterministic(self):
        item = self.module.score_review_candidate(candidate(risk=0.1, uncertainty=False))

        self.assertEqual(0.0, item["score_components"]["uncertainty_score"])
        self.assertIn("missing_uncertainty_metadata", item["review_reasons"])

    def test_missing_correction_risk_is_deterministic(self):
        item = self.module.score_review_candidate(candidate(risk=None))

        self.assertEqual(0.0, item["score_components"]["correction_risk_score"])
        self.assertIn("missing_correction_risk_model", item["review_reasons"])

    def test_high_risk_candidate_scores_above_low_risk_candidate(self):
        high = self.module.score_review_candidate(candidate(risk=0.9))
        low = self.module.score_review_candidate(candidate(risk=0.05))

        self.assertGreater(high["priority_score"], low["priority_score"])

    def test_delta_does_not_affect_score(self):
        base = candidate(risk=0.2, delta_major=False)
        changed = candidate(risk=0.2, delta_major=True)

        first = self.module.score_review_candidate(base)
        second = self.module.score_review_candidate(changed)

        self.assertEqual(first["priority_score"], second["priority_score"])
        self.assertFalse(first["evaluation_only"]["delta"]["major_correction"])
        self.assertTrue(second["evaluation_only"]["delta"]["major_correction"])

    def test_boundary_shape_score_uses_prediction_features_only(self):
        base = candidate(risk=0.2, delta_major=False)
        changed = candidate(risk=0.2, delta_major=True)
        changed["boundary_metadata"] = {"perimeter_area_ratio": 100.0, "thin_structure_score": 1.0}
        changed["delta"]["boundary"] = {"boundary_f1": 0.0, "boundary_error_area_ratio": 1.0}
        weights = {
            "correction_risk_score": 0.0,
            "uncertainty_score": 0.0,
            "rule_review_score": 0.0,
            "geometry_complexity_score": 0.0,
            "boundary_shape_score": 1.0,
            "diversity_score": 0.0,
        }
        first = self.module.score_review_candidate(base, weights=weights)
        second = self.module.score_review_candidate(changed, weights=weights)
        self.assertEqual(first["priority_score"], second["priority_score"])
        self.assertNotIn("boundary_metadata", first["source_metadata"])

    def test_missing_prediction_features_backward_compatible(self):
        row = candidate(risk=0.2)
        row.pop("prediction_features", None)
        row["mask_quality"].pop("prediction_time_boundary_shape", None)
        item = self.module.score_review_candidate(row)
        self.assertEqual(0.0, item["score_components"]["boundary_shape_score"])

    def test_queue_ordering_and_ranks_are_deterministic(self):
        records = [candidate(idx=1, risk=0.1), candidate(idx=2, risk=0.9), candidate(idx=3, risk=0.5)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "review_queue.jsonl"
            result = self.module.build_segmentation_review_queue(records, str(output))

            items = result["items"]
            scores = [item["priority_score"] for item in items]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual([1, 2, 3], [item["rank"] for item in items])
            self.assertTrue(output.exists())
            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([1, 2, 3], [item["rank"] for item in written])

    def test_metadata_summary_has_counts_and_weights(self):
        records = [candidate(idx=1, risk=0.1), candidate(idx=2, risk=0.9)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "review_queue.jsonl"
            summary = self.module.build_segmentation_review_queue(records, str(output))["review_queue"]

            self.assertEqual("generated", summary["status"])
            self.assertEqual(2, summary["records_total"])
            self.assertEqual(2, summary["records_scored"])
            self.assertIn("high", summary["priority_bucket_counts"])
            self.assertIn("correction_risk_score", summary["score_weights"])
            self.assertIn("output_path", summary)
            self.assertTrue(summary["reason_counts"])

    def test_review_queue_accepts_weight_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            presets = root / "presets.json"
            presets.write_text(json.dumps({"risk_heavy": {"correction_risk_score": 1.0}}), encoding="utf-8")
            result = self.module.build_segmentation_review_queue(
                [candidate(idx=1, risk=0.1), candidate(idx=2, risk=0.9)],
                str(root / "queue.jsonl"),
                weight_preset="risk_heavy",
                weight_presets_file=str(presets),
            )
            self.assertEqual("risk_heavy", result["items"][0]["review_weight_preset"])

    def test_review_queue_default_weights_unchanged(self):
        item = self.module.score_review_candidate(candidate(risk=0.5))
        self.assertEqual(self.module.DEFAULT_SCORE_WEIGHTS.keys(), item["review_weight_weights"].keys())
        self.assertEqual("current", item["review_weight_preset"])

    def test_review_queue_records_weight_preset(self):
        item = self.module.score_review_candidate(candidate(risk=0.5), weight_preset="current")
        self.assertIn("review_weight_preset", item)
        self.assertIn("review_weight_weights", item)

    def test_review_queue_custom_weights_no_delta_leakage(self):
        weights = {"correction_risk_score": 1.0, "uncertainty_score": 0.0}
        first = self.module.score_review_candidate(candidate(risk=0.5, delta_major=False), weights=weights)
        second = self.module.score_review_candidate(candidate(risk=0.5, delta_major=True), weights=weights)
        self.assertEqual(first["priority_score"], second["priority_score"])

    def test_empty_input_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "review_queue.jsonl"
            result = self.module.build_segmentation_review_queue([], str(output))

            summary = result["review_queue"]
            self.assertEqual("skipped", summary["status"])
            self.assertEqual("no_review_candidates", summary["skip_reason"])
            self.assertEqual([], result["items"])
            self.assertTrue(output.exists())

    def test_diversity_heuristic_is_deterministic(self):
        records = [
            candidate(idx=1, risk=0.7, area=0.005),
            candidate(idx=2, risk=0.7, area=0.40),
            candidate(idx=3, risk=0.7, area=0.08),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            first = self.module.build_segmentation_review_queue(records, str(Path(tmp) / "a.jsonl"))["items"]
            second = self.module.build_segmentation_review_queue(records, str(Path(tmp) / "b.jsonl"))["items"]

            self.assertEqual(
                [(item["task_id"], item["priority_score"], item["review_reasons"]) for item in first],
                [(item["task_id"], item["priority_score"], item["review_reasons"]) for item in second],
            )
            self.assertTrue(any("diversity_boost" in item["review_reasons"] for item in first))


if __name__ == "__main__":
    unittest.main()
