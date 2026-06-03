import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.classify_shadow_windows import main as classify_main
from image_segmentation.benchmark.prepare_controlled_human_review_assignment import prepare_controlled_human_review_assignment
from image_segmentation.benchmark.select_comparable_shadow_windows import select_comparable_shadow_windows
from image_segmentation.benchmark.shadow_root_cause_analysis import (
    analyze_schema_aware_shadow_drift,
    analyze_schema_aware_shadow_missing_features,
    analyze_schema_aware_shadow_topk_jaccard,
)
from image_segmentation.benchmark.shadow_window_schema import (
    classify_pairwise_compatibility,
    classify_shadow_window,
    classify_shadow_windows,
)
from image_segmentation.benchmark.write_shadow_schema_aware_decision_report import write_shadow_schema_aware_decision_report
from image_segmentation.benchmark.write_controlled_shadow_validation_report import write_controlled_shadow_validation_report


FEATURES = {
    "pred_area_ratio": 0.08,
    "pred_bbox_area_ratio": 0.10,
    "pred_extent": 0.8,
    "pred_aspect_ratio": 1.2,
    "pred_touches_border": False,
    "pred_boundary_complexity": 1.4,
    "pred_boundary_density": 2.0,
    "pred_component_count": 1,
    "pred_largest_component_ratio": 1.0,
    "pred_hole_count": 0,
    "pred_thinness_proxy": 0.1,
}


def row(idx, *, full=True, queue_type="benchmark_replay", sample_offset=0, learned=0.4, metadata_missing=None):
    missing = sorted(FEATURES) if not full else list(metadata_missing or [])
    item = {
        "task_id": f"task-{idx + sample_offset}",
        "sample_id": f"sample-{idx + sample_offset}",
        "priority_score": idx / 10,
        "queue_type": queue_type,
        "benchmark_name": "unit_benchmark" if queue_type == "benchmark_replay" else None,
        "shadow_scores": {
            "learned_boundary_shape_only_score": learned,
            "shadow_only": True,
        },
        "shadow_score_metadata": {
            "shadow_only": True,
            "affects_default_ranking": False,
            "artifact_validation_status": "valid",
            "artifact_version": "unit",
            "missing_safe_feature_count": len(missing),
            "missing_safe_features": missing,
        },
        "evaluation_only": {"delta": {"major_correction": True}, "boundary_iou": 0.1},
        "boundary_metadata": {"thin_structure_score": 999},
        "gt_mask_path": "/private/gt.png",
        "labels": ["major"],
    }
    if full:
        item["prediction_features"] = dict(FEATURES)
        item["mask_quality"] = {"prediction_time_boundary_shape": dict(FEATURES)}
    else:
        item["mask_quality"] = {}
    return item


def write_window(root: Path, window_id: str, rows):
    window = root / window_id
    window.mkdir(parents=True)
    path = window / "live_review_queue.shadow_scored.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    return path


class ShadowWindowSchemaTests(unittest.TestCase):
    def test_classifies_full_prediction_feature_window(self):
        rows = [row(idx, full=True, queue_type="live") for idx in range(1, 6)]
        result = classify_shadow_window("window_full", rows)

        self.assertEqual("full_prediction_features", result["queue_schema_version"])
        self.assertEqual("full", result["feature_completeness_bucket"])
        self.assertEqual("live", result["queue_type"])
        self.assertEqual(1.0, result["learned_score_coverage"])
        self.assertEqual(0.0, result["neutral_fallback_rate"])
        self.assertTrue(result["shadow_only_all_true"])
        self.assertTrue(result["affects_default_ranking_all_false"])

    def test_classifies_old_schema_neutral_fallback_window(self):
        rows = [row(idx, full=False, learned=0.5) for idx in range(1, 6)]
        result = classify_shadow_window("window_old", rows)

        self.assertEqual("old_schema_fallback", result["queue_schema_version"])
        self.assertEqual("missing_boundary_shape", result["feature_completeness_bucket"])
        self.assertEqual("benchmark_replay", result["queue_type"])
        self.assertEqual(1.0, result["learned_score_coverage"])
        self.assertEqual(1.0, result["neutral_fallback_rate"])
        self.assertIn("pred_thinness_proxy", result["missing_feature_names"])

    def test_classifies_mixed_feature_completeness(self):
        rows = [row(1, full=True), row(2, full=True), row(3, full=False), row(4, full=False)]
        result = classify_shadow_window("window_mixed", rows)

        self.assertEqual("mixed", result["queue_schema_version"])
        self.assertEqual("mixed", result["feature_completeness_bucket"])

    def test_pairwise_compatibility_keeps_zero_overlap_as_limited_not_blocking(self):
        left = classify_shadow_window("window_a", [row(idx, full=True, queue_type="live") for idx in range(1, 4)])
        right = classify_shadow_window("window_b", [row(idx, full=True, queue_type="live", sample_offset=100) for idx in range(1, 4)])
        result = classify_pairwise_compatibility(left, right)

        self.assertTrue(result["compatible_for_drift_comparison"])
        self.assertTrue(result["sample_set_changed"])
        self.assertEqual(0, result["same_sample_overlap"])
        self.assertTrue(result["direct_topk_comparison_limited"])

    def test_schema_mismatch_is_incompatible(self):
        full = classify_shadow_window("window_full", [row(idx, full=True) for idx in range(1, 4)])
        old = classify_shadow_window("window_old", [row(idx, full=False) for idx in range(1, 4)])
        result = classify_pairwise_compatibility(full, old)

        self.assertFalse(result["compatible_for_drift_comparison"])
        self.assertIn("schema differs", result["reason"])

    def test_cli_outputs_classification_without_forbidden_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, queue_type="live") for idx in range(1, 4)])
            write_window(root, "window_004", [row(idx, full=False, sample_offset=100) for idx in range(1, 4)])
            out = root / "schema_aware_analysis"

            result = classify_shadow_windows(str(root), str(out))

            self.assertEqual(2, len(result["windows"]))
            self.assertTrue((out / "shadow_window_classification.json").exists())
            self.assertTrue((out / "shadow_window_classification.md").exists())
            self.assertTrue((out / "window_schema_rows.jsonl").exists())
            payload = json.loads((out / "shadow_window_classification.json").read_text(encoding="utf-8"))
            self.assertIn("window_001__window_004", payload["pairwise_compatibility_matrix"])
            written_text = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir())
            for forbidden in ["evaluation_only", "boundary_metadata", "gt_mask_path", "labels", "boundary_iou", "major_correction"]:
                self.assertNotIn(forbidden, written_text)

    def test_cli_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, queue_type="live") for idx in range(1, 3)])
            code = classify_main(["--multi-window-root", str(root), "--output-dir", str(root / "out")])
            self.assertEqual(0, code)
            self.assertTrue((root / "out" / "shadow_window_classification.json").exists())

    def test_schema_aware_drift_skips_non_compatible_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, queue_type="live", learned=0.1) for idx in range(1, 4)])
            write_window(root, "window_004", [row(idx, full=False, sample_offset=100, learned=0.9) for idx in range(1, 4)])
            classification = classify_shadow_windows(str(root), str(root / "schema"))

            result = analyze_schema_aware_shadow_drift(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "drift"),
            )

            self.assertEqual([], result["comparable_pairs"])
            self.assertEqual(1, len(result["non_comparable_pairs"]))
            self.assertEqual("compatibility_warning", result["non_comparable_pairs"][0]["alert_severity"])
            self.assertIn("No stable conclusion", result["conclusion"])
            self.assertIn("window_001__window_004", classification["pairwise_compatibility_matrix"])

    def test_schema_aware_drift_stability_alert_for_comparable_full_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, queue_type="live", learned=0.1) for idx in range(1, 5)])
            write_window(root, "window_002", [row(idx, full=True, queue_type="live", learned=0.95) for idx in range(1, 5)])
            classify_shadow_windows(str(root), str(root / "schema"))

            result = analyze_schema_aware_shadow_drift(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "drift"),
            )

            self.assertEqual(1, len(result["comparable_pairs"]))
            self.assertEqual("stability_alert", result["comparable_pairs"][0]["alert_severity"])
            self.assertIn("p95_drift", result["comparable_pairs"][0])

    def test_schema_aware_topk_zero_overlap_is_compatibility_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, queue_type="live", learned=idx / 10) for idx in range(1, 5)])
            write_window(root, "window_002", [row(idx, full=True, queue_type="live", sample_offset=100, learned=idx / 10) for idx in range(1, 5)])
            classify_shadow_windows(str(root), str(root / "schema"))

            result = analyze_schema_aware_shadow_topk_jaccard(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "topk"),
                k=2,
            )

            self.assertEqual([], result["comparable_pairs"])
            self.assertEqual(1, len(result["non_comparable_pairs"]))
            self.assertTrue(result["non_comparable_pairs"][0]["not_directly_comparable"])
            self.assertEqual("compatibility_warning", result["non_comparable_pairs"][0]["alert_severity"])

    def test_schema_aware_missing_feature_reclassification_and_decision_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=False, learned=0.5) for idx in range(1, 3)])
            write_window(
                root,
                "window_002",
                [row(idx, full=True, queue_type="live", metadata_missing=["pred_thinness_proxy"]) for idx in range(1, 3)],
            )
            classify_shadow_windows(str(root), str(root / "schema"))
            missing = analyze_schema_aware_shadow_missing_features(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "missing"),
            )
            drift = analyze_schema_aware_shadow_drift(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "drift"),
            )
            topk = analyze_schema_aware_shadow_topk_jaccard(
                str(root),
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "topk"),
                k=2,
            )

            self.assertEqual("compatibility_warning", missing["per_window"]["window_001"]["alert_severity"])
            self.assertEqual("stability_alert", missing["per_window"]["window_002"]["alert_severity"])
            decision = write_shadow_schema_aware_decision_report(
                str(root / "schema" / "shadow_window_classification.json"),
                str(root / "drift" / "schema_aware_drift.json"),
                str(root / "topk" / "schema_aware_topk_jaccard.json"),
                str(root / "missing" / "schema_aware_missing_features.json"),
                str(root / "decision.md"),
            )
            self.assertEqual("hold_expansion", decision["decision"]["recommended_action"])
            self.assertTrue(decision["decision"]["default_layer_5_weights_unchanged"])
            self.assertTrue((root / "decision.md").exists())

    def test_comparable_window_selector_filters_full_feature_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True) for idx in range(1, 3)])
            write_window(root, "window_004", [row(idx, full=False) for idx in range(1, 3)])
            classify_shadow_windows(str(root), str(root / "schema"))

            selection = select_comparable_shadow_windows(
                str(root / "schema" / "shadow_window_classification.json"),
                str(root),
                str(root / "controlled"),
            )

            self.assertEqual(["window_001"], selection["selected_windows"])
            self.assertEqual(1, selection["rejected_window_count"])
            self.assertIn("old_schema_fallback", selection["rejected_windows"][0]["reason"])
            self.assertTrue((root / "controlled" / "selected_shadow_window_classification.json").exists())

    def test_controlled_human_review_assignment_safe_fields_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, learned=idx / 10) for idx in range(1, 6)])
            selected = root / "selected_windows.txt"
            selected.write_text("window_001\n", encoding="utf-8")

            summary = prepare_controlled_human_review_assignment(
                str(root),
                str(selected),
                str(root / "assignment"),
                high_learned_low_current=2,
                high_current_low_learned=2,
                top_learned=2,
                control_current_top=2,
            )

            self.assertTrue(summary["safe_fields_only"])
            payload = "\n".join(
                (root / "assignment" / name).read_text(encoding="utf-8")
                for name in ["review_assignment.jsonl", "review_assignment_template.csv", "controlled_human_review_assignment_summary.json"]
            )
            self.assertIn("prediction_time_safe_features", payload)
            for forbidden in ["evaluation_only", "boundary_metadata", "gt_mask_path", "labels", "boundary_iou", "major_correction"]:
                self.assertNotIn(forbidden, payload)

    def test_controlled_validation_report_keeps_promotion_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_window(root, "window_001", [row(idx, full=True, learned=0.3) for idx in range(1, 4)])
            write_window(root, "window_002", [row(idx, full=True, learned=0.3) for idx in range(1, 4)])
            classify_shadow_windows(str(root), str(root / "schema"))
            selection = select_comparable_shadow_windows(
                str(root / "schema" / "shadow_window_classification.json"),
                str(root),
                str(root / "controlled"),
            )
            drift = analyze_schema_aware_shadow_drift(
                str(root),
                str(root / "controlled" / "selected_shadow_window_classification.json"),
                str(root / "controlled_drift"),
            )
            topk = analyze_schema_aware_shadow_topk_jaccard(
                str(root),
                str(root / "controlled" / "selected_shadow_window_classification.json"),
                str(root / "controlled_topk"),
                k=2,
            )
            missing = analyze_schema_aware_shadow_missing_features(
                str(root),
                str(root / "controlled" / "selected_shadow_window_classification.json"),
                str(root / "controlled_missing"),
            )
            prepare_controlled_human_review_assignment(
                str(root),
                str(root / "controlled" / "selected_windows.txt"),
                str(root / "assignment"),
                top_learned=2,
                control_current_top=2,
            )
            report = write_controlled_shadow_validation_report(
                str(root / "controlled" / "comparable_window_selection.json"),
                str(root / "controlled_drift" / "schema_aware_drift.json"),
                str(root / "controlled_topk" / "schema_aware_topk_jaccard.json"),
                str(root / "controlled_missing" / "schema_aware_missing_features.json"),
                str(root / "assignment"),
                str(root / "controlled_report.md"),
            )

            self.assertEqual(["window_001", "window_002"], selection["selected_windows"])
            self.assertFalse(report["decision"]["ready_for_default_sorting"])
            self.assertFalse(report["decision"]["ready_for_weight_promotion"])
            self.assertTrue(report["decision"]["default_sorting_unchanged"])
            self.assertIn("controlled_stability_conclusion", report)


if __name__ == "__main__":
    unittest.main()
