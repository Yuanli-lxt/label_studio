import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.compare_review_strategies import compare_review_strategies, _strategy_recommendation


def item(idx, major, score):
    return {
        "rank": idx,
        "task_id": f"s{idx}",
        "priority_score": score,
        "score_components": {
            "rule_review_score": score,
            "uncertainty_score": 1 - score,
            "correction_risk_score": score / 2,
        },
        "evaluation_only": {
            "delta": {
                "major_correction": major,
                "model_human_iou": 0.2 if major else 0.9,
                "correction_area_ratio": 0.3 if major else 0.01,
            }
        },
    }


def custom_item(idx, major, priority, rule, uncertainty, risk):
    row = item(idx, major, priority)
    row["score_components"] = {
        "rule_review_score": rule,
        "uncertainty_score": uncertainty,
        "correction_risk_score": risk,
    }
    return row


class BenchmarkCompareStrategiesTests(unittest.TestCase):
    def write_queue(self, root: Path, rows):
        path = root / "review_queue.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_compare_review_strategies_outputs_all_available_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = self.write_queue(root, [item(1, True, 0.9), item(2, False, 0.1)])
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertEqual(
                {"random", "mask_quality_only", "uncertainty_only", "correction_risk_only", "full_priority"},
                set(report["strategies"]),
            )

    def test_compare_review_strategies_missing_fields_mark_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = self.write_queue(root, [{"task_id": "s1", "evaluation_only": {"delta": {"major_correction": True}}}])
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertFalse(report["strategies"]["uncertainty_only"]["available"])

    def test_compare_review_strategies_no_delta_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(1, False, 0.9), item(2, True, 0.1)]
            queue = self.write_queue(root, rows)
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertEqual(0.0, report["strategies"]["full_priority"]["precision_at_20_percent"])

    def test_compare_review_strategies_random_seed_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i % 2 == 0, i / 10) for i in range(1, 8)]
            queue = self.write_queue(root, rows)
            first = compare_review_strategies(str(queue), str(root / "a"), random_seed=7)
            second = compare_review_strategies(str(queue), str(root / "b"), random_seed=7)
            self.assertEqual(first["strategies"]["random"], second["strategies"]["random"])

    def test_full_priority_vs_baselines_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = self.write_queue(root, [item(1, True, 0.9), item(2, False, 0.1)])
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertIn("full_priority_exceeds_random", report["summary"])
            self.assertTrue((root / "out" / "strategy_comparison.md").exists())

    def test_bootstrap_ci_reproducible_and_has_low_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i % 3 == 0, i / 20) for i in range(1, 16)]
            queue = self.write_queue(root, rows)
            first = compare_review_strategies(str(queue), str(root / "a"), random_seed=7, bootstrap_iters=25)
            second = compare_review_strategies(str(queue), str(root / "b"), random_seed=7, bootstrap_iters=25)
            ci = first["strategies"]["full_priority"]["bootstrap_ci"]["precision_at_20_percent"]
            self.assertIn("ci95_low", ci)
            self.assertIn("ci95_high", ci)
            self.assertEqual(
                first["strategies"]["full_priority"]["bootstrap_ci"],
                second["strategies"]["full_priority"]["bootstrap_ci"],
            )

    def test_bootstrap_ci_warns_on_degenerate_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = self.write_queue(root, [item(i, False, i / 10) for i in range(1, 8)])
            report = compare_review_strategies(str(queue), str(root / "out"), bootstrap_iters=10)
            ci = report["strategies"]["full_priority"]["bootstrap_ci"]["precision_at_20_percent"]
            self.assertEqual("degenerate_labels", ci["warning"])

    def test_oof_recommendation_risk_beats_full_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                custom_item(1, False, 0.9, 0.9, 0.1, 0.1),
                custom_item(2, False, 0.8, 0.8, 0.1, 0.2),
                custom_item(3, True, 0.1, 0.1, 0.1, 0.9),
                custom_item(4, True, 0.2, 0.2, 0.1, 0.8),
            ]
            queue = self.write_queue(root, rows)
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertIn("consider increasing correction_risk weight", report["recommendation"])

    def test_oof_recommendation_all_near_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [custom_item(i, i % 2 == 0, 0.5, 0.5, 0.5, 0.5) for i in range(1, 12)]
            queue = self.write_queue(root, rows)
            report = compare_review_strategies(str(queue), str(root / "out"))
            self.assertIn("increase sample size", report["recommendation"])

    def test_oof_recommendation_ci_includes_no_improvement(self):
        report = {
            "eval_split_size": 100,
            "major_correction_base_rate": 0.2,
            "summary": {"best_strategy_by_ap": "full_priority", "meets_initial_effectiveness_standard": True},
            "strategies": {
                "random": {"available": True, "average_precision_for_major_correction": 0.2},
                "full_priority": {
                    "available": True,
                    "average_precision_for_major_correction": 0.5,
                    "bootstrap_ci": {"lift_at_20_percent_over_random": {"ci95_low": 1.0}},
                },
                "correction_risk_only": {"available": True, "average_precision_for_major_correction": 0.3},
                "uncertainty_only": {"available": True, "average_precision_for_major_correction": 0.3},
            },
        }
        self.assertIn("lift confidence interval includes no improvement", _strategy_recommendation(report))


if __name__ == "__main__":
    unittest.main()
