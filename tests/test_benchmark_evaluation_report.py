import unittest

from image_segmentation.benchmark.evaluation import build_evaluation_report, render_markdown_report


def queue_item(idx, major, score, components=None):
    return {
        "rank": idx,
        "task_id": f"s{idx}",
        "dataset": "COCO",
        "category_name": "thing",
        "priority_score": score,
        "priority_bucket": "high" if score > 0.6 else "low",
        "review_reasons": ["reason"],
        "score_components": components
        or {
            "correction_risk_score": score,
            "uncertainty_score": 1 - score,
            "rule_review_score": score / 2,
            "geometry_complexity_score": 0.1,
            "diversity_score": 0.2,
        },
        "source_metadata": {
            "uncertainty": {
                "enabled": True,
                "stable": not major,
                "stability_bucket": "low" if major else "high",
                "mean_pairwise_iou": 0.4 if major else 0.9,
            }
        },
        "evaluation_only": {
            "delta": {
                "major_correction": major,
                "model_human_iou": 0.2 if major else 0.9,
                "correction_area_ratio": 0.3 if major else 0.01,
                "correction_reason": ["low_iou"] if major else [],
            }
        },
    }


class BenchmarkEvaluationReportTests(unittest.TestCase):
    def test_evaluation_report_has_base_rate(self):
        report = build_evaluation_report(
            [queue_item(1, False, 0.9), queue_item(2, True, 0.1)],
            [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in [queue_item(1, False, 0.9), queue_item(2, True, 0.1)]],
        )
        self.assertEqual(0.5, report["base_rate"]["major_correction_base_rate"])
        self.assertEqual(0.5, report["base_rate"]["random_expected_precision_at_20_percent"])

    def test_evaluation_report_has_top_k_details(self):
        report = build_evaluation_report(
            [queue_item(1, False, 0.9), queue_item(2, True, 0.1)],
            [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in [queue_item(1, False, 0.9), queue_item(2, True, 0.1)]],
        )
        self.assertEqual("s1", report["top_10_percent_samples"][0]["sample_id"])
        self.assertIn("score_components", report["top_20_percent_samples"][0])

    def test_evaluation_report_has_false_negatives(self):
        items = [queue_item(1, False, 0.9), queue_item(2, False, 0.8), queue_item(3, True, 0.1)]
        report = build_evaluation_report(items, [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in items])
        self.assertEqual("s3", report["false_negatives"][0]["sample_id"])
        self.assertEqual(["low_iou"], report["false_negatives"][0]["correction_reason"])

    def test_score_component_summary(self):
        report = build_evaluation_report(
            [queue_item(1, False, 0.9), queue_item(2, True, 0.1)],
            [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in [queue_item(1, False, 0.9), queue_item(2, True, 0.1)]],
        )
        self.assertEqual(2, report["score_component_summary"]["correction_risk_score"]["non_null_count"])
        self.assertAlmostEqual(0.5, report["score_component_summary"]["correction_risk_score"]["mean"])

    def test_ap_below_base_rate_warning(self):
        items = [
            queue_item(1, False, 0.9),
            queue_item(2, False, 0.8),
            queue_item(3, True, 0.2),
            queue_item(4, True, 0.1),
        ]
        report = build_evaluation_report(items, [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in items])
        self.assertTrue(any("Queue AP is below" in warning for warning in report["metric_warnings"]))

    def test_precision_below_random_warning(self):
        items = [
            queue_item(1, False, 0.9),
            queue_item(2, False, 0.8),
            queue_item(3, False, 0.7),
            queue_item(4, False, 0.6),
            queue_item(5, True, 0.1),
        ]
        report = build_evaluation_report(items, [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in items])
        self.assertTrue(any("precision@20%" in warning for warning in report["metric_warnings"]))

    def test_uncertainty_summary_in_report(self):
        items = [queue_item(1, False, 0.9), queue_item(2, True, 0.1)]
        report = build_evaluation_report(items, [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in items])
        self.assertEqual(2, report["uncertainty_summary"]["enabled_count"])
        self.assertEqual(1, report["uncertainty_summary"]["unstable_count"])

    def test_evaluation_report_has_major_correction_count(self):
        items = [queue_item(1, False, 0.9), queue_item(2, True, 0.1)]
        report = build_evaluation_report(items, [{"dataset": "COCO", "delta": item["evaluation_only"]["delta"]} for item in items])
        self.assertEqual(1, report["overall_quality"]["major_correction_count"])
        self.assertEqual(1, report["major_correction_diagnostics"]["major_correction_count"])
        self.assertIn("major correction count", render_markdown_report(report))

    def test_evaluation_report_has_category_level_diagnostics(self):
        items = [queue_item(i, i in {1, 2, 3}, 1 / i) for i in range(1, 7)]
        records = [
            {"dataset": "COCO", "category_name": "thing", "delta": item["evaluation_only"]["delta"]}
            for item in items
        ]
        report = build_evaluation_report(items, records)
        self.assertEqual("thing", report["major_correction_diagnostics"]["category_level_major_correction_top20"][0]["dataset"])

    def test_evaluation_report_has_difficulty_tag_diagnostics(self):
        items = [queue_item(1, True, 0.1), queue_item(2, False, 0.9)]
        records = [
            {"dataset": "COCO", "difficulty_tags": ["small_object"], "delta": items[0]["evaluation_only"]["delta"]},
            {"dataset": "COCO", "difficulty_tags": ["large_object"], "delta": items[1]["evaluation_only"]["delta"]},
        ]
        report = build_evaluation_report(items, records)
        tags = {row["tag"] for row in report["major_correction_diagnostics"]["difficulty_tag_major_correction_rates"]}
        self.assertIn("small_object", tags)

    def test_evaluation_report_diagnostics_do_not_affect_priority_score(self):
        first = queue_item(1, False, 0.5)
        second = queue_item(1, True, 0.5)
        self.assertEqual(first["priority_score"], second["priority_score"])
        self.assertEqual(first["score_components"], second["score_components"])


if __name__ == "__main__":
    unittest.main()
