import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.ablate_review_weights import (
    DEFAULT_PRESETS,
    ablate_review_weights,
    load_weight_presets,
    normalize_preset_weights,
)


def item(idx, major, priority=0.0, components=None):
    return {
        "rank": idx,
        "task_id": f"s{idx}",
        "priority_score": priority,
        "score_components": {
            "correction_risk_score": priority,
            "uncertainty_score": 1.0 - priority,
            "rule_review_score": priority / 2,
            "geometry_complexity_score": 0.2,
            "diversity_score": 0.1,
        }
        if components is None
        else components,
        "evaluation_only": {"delta": {"major_correction": major}},
    }


def write_queue(root: Path, rows):
    path = root / "queue.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BenchmarkAblateReviewWeightsTests(unittest.TestCase):
    def rows(self):
        return [item(i, i in {2, 4, 6}, priority=i / 10) for i in range(1, 11)]

    def test_default_weight_presets_exist(self):
        for name in [
            "current_full_priority",
            "risk_heavy",
            "risk_dominant",
            "risk_only",
            "uncertainty_heavy",
            "quality_heavy",
            "no_diversity",
            "balanced_no_risk",
        ]:
            self.assertIn(name, DEFAULT_PRESETS)

    def test_weight_presets_are_normalized(self):
        weights = normalize_preset_weights(DEFAULT_PRESETS["risk_heavy"])
        self.assertAlmostEqual(1.0, sum(weights.values()))

    def test_custom_presets_file_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.json"
            path.write_text(json.dumps({"mine": {"correction_risk_score": 2.0}}), encoding="utf-8")
            self.assertIn("mine", load_weight_presets(str(path)))

    def test_current_full_priority_uses_existing_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(1, False, priority=0.1), item(2, True, priority=0.9)])
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "current_full_priority.review_queue.jsonl")
            self.assertEqual("s2", ranked[0]["task_id"])
            self.assertEqual(ranked[0]["priority_score"], ranked[0]["ablation_priority_score"])

    def test_ablate_review_weights_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"), bootstrap_iters=5)
            self.assertTrue((root / "out" / "weight_ablation.json").exists())
            self.assertTrue((root / "out" / "weight_ablation.md").exists())

    def test_ablate_review_weights_preserves_original_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "risk_heavy.review_queue.jsonl")
            self.assertIn("priority_score", ranked[0])
            self.assertIn("ablation_priority_score", ranked[0])
            self.assertNotEqual("priority_score", "ablation_priority_score")

    def test_ablate_review_weights_does_not_use_delta_for_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = item(1, False, priority=0.5)
            changed = dict(base, task_id="s2", evaluation_only={"delta": {"major_correction": True}})
            queue = write_queue(root, [base, changed])
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "risk_heavy.review_queue.jsonl")
            self.assertEqual(ranked[0]["ablation_priority_score"], ranked[1]["ablation_priority_score"])

    def test_ablate_review_weights_handles_missing_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(1, False, components={}), item(2, True, priority=0.9)])
            result = ablate_review_weights(str(queue), str(root / "out"))
            self.assertGreater(result["component_diagnostics"]["correction_risk_score"]["missing_count"], 0)

    def test_ablate_review_weights_clips_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(1, True, components={"correction_risk_score": 2.0})])
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "risk_only.review_queue.jsonl")
            self.assertEqual(1.0, ranked[0]["ablation_score_components"]["correction_risk_score"])

    def test_ablate_review_weights_outputs_ranked_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            for name in DEFAULT_PRESETS:
                self.assertTrue((root / "out" / "preset_rankings" / f"{name}.review_queue.jsonl").exists())

    def test_ablation_metrics_include_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = ablate_review_weights(str(queue), str(root / "out"))
            row = result["presets"]["risk_heavy"]
            for field in [
                "precision_at_20_percent",
                "recall_at_20_percent",
                "lift_at_20_percent_over_random",
                "average_precision_for_major_correction",
                "top_20_percent_size",
                "major_correction_base_rate",
                "positive_count",
                "n_samples",
            ]:
                self.assertIn(field, row)

    def test_ablation_bootstrap_ci_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = ablate_review_weights(str(queue), str(root / "out"), bootstrap_iters=5)
            self.assertIn("bootstrap_ci", result["presets"]["risk_heavy"])

    def test_ablation_bootstrap_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            first = ablate_review_weights(str(queue), str(root / "a"), bootstrap_iters=5, random_seed=7)
            second = ablate_review_weights(str(queue), str(root / "b"), bootstrap_iters=5, random_seed=7)
            self.assertEqual(
                first["presets"]["risk_heavy"]["bootstrap_ci"],
                second["presets"]["risk_heavy"]["bootstrap_ci"],
            )

    def test_ablation_warns_on_degenerate_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(i, False, priority=i / 10) for i in range(1, 5)])
            result = ablate_review_weights(str(queue), str(root / "out"), bootstrap_iters=5)
            self.assertIn("degenerate_labels", result["warnings"])

    def test_weight_ablation_report_contains_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            text = (root / "out" / "weight_ablation.md").read_text(encoding="utf-8")
            self.assertIn("Preset Leaderboard", text)

    def test_weight_ablation_report_contains_pairwise_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("Pairwise Comparison", (root / "out" / "weight_ablation.md").read_text(encoding="utf-8"))

    def test_weight_ablation_report_contains_topk_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("Top-K Overlap", (root / "out" / "weight_ablation.md").read_text(encoding="utf-8"))

    def test_weight_ablation_report_contains_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("Recommendation", (root / "out" / "weight_ablation.md").read_text(encoding="utf-8"))

    def test_weight_ablation_component_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("correction_risk_score", result["component_diagnostics"])


if __name__ == "__main__":
    unittest.main()
