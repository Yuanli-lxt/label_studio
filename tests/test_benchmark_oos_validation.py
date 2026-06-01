import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.build_review_queue_from_delta import build_review_queue_from_delta
from image_segmentation.benchmark.crossfit_risk_evaluation import crossfit_risk_evaluation
from image_segmentation.benchmark.make_eval_splits import make_eval_splits
from image_segmentation.benchmark.predict_risk_scores import predict_risk_scores
from image_segmentation.benchmark.train_risk_model import train_from_delta_dataset


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
            "mask_area_ratio": 0.02 if major else 0.15,
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


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BenchmarkOOSValidationTests(unittest.TestCase):
    def rows(self):
        return [record(i, i in {1, 3, 5, 7, 9, 11}) for i in range(18)]

    def test_make_eval_splits_outputs_files_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            metadata = make_eval_splits(str(delta), str(root / "splits"), train_ratio=0.67, random_seed=1)
            for name in [
                "split_metadata.json",
                "train_ids.json",
                "eval_ids.json",
                "train_delta_dataset.jsonl",
                "eval_delta_dataset.jsonl",
            ]:
                self.assertTrue((root / "splits" / name).exists(), name)
            train_ids = json.loads((root / "splits" / "train_ids.json").read_text(encoding="utf-8"))
            eval_ids = json.loads((root / "splits" / "eval_ids.json").read_text(encoding="utf-8"))
            self.assertFalse(set(train_ids) & set(eval_ids))
            self.assertEqual(metadata["n_total"], len(train_ids) + len(eval_ids))
            self.assertGreater(metadata["train_positive_rate"], 0)
            self.assertGreater(metadata["eval_positive_rate"], 0)

    def test_make_eval_splits_warns_when_too_few_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, [record(1, True)] + [record(i, False) for i in range(2, 8)])
            metadata = make_eval_splits(str(delta), str(root / "splits"), random_seed=1)
            self.assertIn("too_few_positive_samples_for_stable_stratified_split", metadata["warnings"])

    def test_predict_risk_scores_outputs_scores_and_preserves_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            rows = self.rows()
            write_jsonl(delta, rows)
            train_from_delta_dataset(str(delta), str(root / "risk"))
            out = root / "delta_with_risk.jsonl"
            predict_risk_scores(str(delta), str(root / "risk"), str(out))
            scored = read_jsonl(out)
            self.assertEqual(len(rows), len(scored))
            self.assertIn("correction_risk_score", scored[0])
            self.assertEqual(rows[0]["delta"], scored[0]["delta"])

    def test_predict_risk_scores_rejects_leaky_feature_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            train_from_delta_dataset(str(delta), str(root / "risk"))
            (root / "risk" / "feature_names.json").write_text(json.dumps(["delta.major_correction"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                predict_risk_scores(str(delta), str(root / "risk"), str(root / "out.jsonl"))

    def test_build_review_queue_from_delta_uses_risk_and_keeps_delta_evaluation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(record(1, True), correction_risk_score=0.95), dict(record(2, False), correction_risk_score=0.05)]
            delta = root / "delta_with_risk.jsonl"
            write_jsonl(delta, rows)
            result = build_review_queue_from_delta(str(delta), str(root / "queue.jsonl"))
            item = result["items"][0]
            self.assertGreater(item["score_components"]["correction_risk_score"], 0.9)
            self.assertNotIn("delta", item["score_components"])
            self.assertTrue(item["evaluation_only"]["delta"]["major_correction"])

    def test_crossfit_outputs_expected_files_and_scores_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            result = crossfit_risk_evaluation(str(delta), str(root / "oof"), n_splits=3, random_seed=2)
            for name in [
                "crossfit_metadata.json",
                "fold_metrics.json",
                "oof_summary.json",
                "oof_delta_with_risk.jsonl",
                "oof_review_queue.jsonl",
                "strategy_comparison_oof.json",
            ]:
                self.assertTrue((root / "oof" / name).exists(), name)
            scored = read_jsonl(root / "oof" / "oof_delta_with_risk.jsonl")
            self.assertEqual(len(self.rows()), len(scored))
            self.assertEqual(len(scored), len({row["task_id"] for row in scored}))
            assignments = json.loads((root / "oof" / "fold_assignments.json").read_text(encoding="utf-8"))
            for sample_id, assignment in assignments.items():
                self.assertNotIn(sample_id, assignment["train_ids"])
            self.assertEqual(3, result["metadata"]["n_splits"])
            self.assertIn("positive_count", result["oof_summary"])
            self.assertIn("fold_positive_counts", result["oof_summary"])

    def test_crossfit_reduces_splits_when_too_few_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            rows = [record(i, i in {1, 3, 5}) for i in range(12)]
            write_jsonl(delta, rows)
            result = crossfit_risk_evaluation(str(delta), str(root / "oof"), n_splits=5, random_seed=2)
            self.assertEqual(3, result["metadata"]["n_splits"])

    def test_crossfit_bootstrap_ci_present_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            result = crossfit_risk_evaluation(str(delta), str(root / "oof"), n_splits=3, random_seed=2, bootstrap_iters=5)
            self.assertIn("bootstrap_ci", result["oof_summary"])
            self.assertIn("precision_at_20_percent", result["oof_summary"]["bootstrap_ci"])

    def test_crossfit_summary_initial_standard_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            result = crossfit_risk_evaluation(str(delta), str(root / "oof"), n_splits=3, random_seed=2)
            summary = result["oof_summary"]
            self.assertIn("full_priority_meets_initial_standard", summary)
            self.assertIn("correction_risk_only_meets_initial_standard", summary)
            self.assertIn("recommendation", summary)

    def test_oof_recommendation_low_positive_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.jsonl"
            write_jsonl(delta, self.rows())
            result = crossfit_risk_evaluation(str(delta), str(root / "oof"), n_splits=3, random_seed=2)
            self.assertIn("positive count is still low", result["oof_summary"]["recommendation"])


if __name__ == "__main__":
    unittest.main()
