import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_segmentation.benchmark.ablate_review_weights import (
    DEFAULT_PRESETS,
    ablate_review_weights,
    load_weight_presets,
    normalize_preset_weights,
)
from image_segmentation.benchmark.diagnose_prediction_features import diagnose_prediction_features
from image_segmentation.benchmark.learn_boundary_shape_fusion import learn_boundary_shape_fusion
from image_segmentation.benchmark.learn_boundary_shape_fusion import (
    export_learned_fusion_artifacts,
    predict_with_artifacts,
    validate_learned_fusion_artifact,
)
from image_segmentation.benchmark.evaluate_gated_fusion import evaluate_gated_fusion
from image_segmentation.benchmark.compare_shadow_scores import compare_shadow_scores
from image_segmentation.benchmark.export_shadow_review_packet import export_shadow_review_packet
from image_segmentation.benchmark.apply_shadow_scores import apply_shadow_scores
from image_segmentation.benchmark.live_shadow_rollout import build_human_review_task_packet
from image_segmentation.benchmark.live_shadow_rollout import build_multi_window_summary, run_live_shadow_rollout
from image_segmentation.benchmark.live_shadow_rollout import run_multi_window_shadow_observation
from image_segmentation.benchmark.shadow_scoring import validate_artifact_for_shadow
from image_segmentation.benchmark.summarize_shadow_human_feedback import summarize_shadow_human_feedback
from image_segmentation.benchmark.shadow_root_cause_analysis import analyze_shadow_drift
from image_segmentation.benchmark.shadow_root_cause_analysis import analyze_shadow_missing_features
from image_segmentation.benchmark.shadow_root_cause_analysis import analyze_shadow_topk_jaccard
from image_segmentation.benchmark.shadow_root_cause_analysis import build_human_review_assignment


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
            "boundary_shape_score": 0.3,
            "diversity_score": 0.1,
        }
        if components is None
        else components,
        "prediction_features": {
            "pred_area_ratio": 0.005 if major else 0.08,
            "pred_bbox_area_ratio": 0.02 if major else 0.09,
            "pred_extent": 0.35 if major else 0.9,
            "pred_aspect_ratio": 6.0 if major else 1.2,
            "pred_touches_border": bool(major),
            "pred_boundary_complexity": 8.0 if major else 1.1,
            "pred_boundary_density": 12.0 if major else 2.0,
            "pred_component_count": 3 if major else 1,
            "pred_largest_component_ratio": 0.55 if major else 1.0,
            "pred_hole_count": 1 if major else 0,
            "pred_thinness_proxy": 0.75 if major else 0.1,
        },
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
            "boundary_shape_experimental",
            "boundary_shape_rank_score",
            "boundary_shape_calibrated_score",
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
            self.assertTrue((root / "out" / "ablation_results.json").exists())
            self.assertTrue((root / "out" / "bootstrap_ci.json").exists())

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
                "roc_auc_for_major_correction",
                "brier_score_for_major_correction",
            ]:
                self.assertIn(field, row)

    def test_ablate_review_weights_includes_boundary_shape_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("boundary_shape_rank_score", result["presets"])
            self.assertIn("boundary_shape_calibrated_score", result["presets"])
            ranked = read_jsonl(root / "out" / "preset_rankings" / "boundary_shape_calibrated_score.review_queue.jsonl")
            self.assertIn("boundary_shape_calibrated_score", ranked[0]["ablation_score_components"])

    def test_boundary_shape_experimental_preset_schema(self):
        weights = normalize_preset_weights(DEFAULT_PRESETS["boundary_shape_experimental"])
        self.assertIn("boundary_shape_score", weights)
        self.assertGreater(weights["boundary_shape_score"], 0.0)
        self.assertAlmostEqual(1.0, sum(weights.values()))

    def test_ablate_review_weights_missing_boundary_shape_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4}, priority=i / 10) for i in range(1, 6)]
            for row in rows:
                row["score_components"].pop("boundary_shape_score", None)
            queue = write_queue(root, rows)
            result = ablate_review_weights(str(queue), str(root / "out"))
            self.assertIn("boundary_shape_experimental", result["presets"])
            self.assertGreater(result["component_diagnostics"]["boundary_shape_score"]["missing_count"], 0)

    def test_boundary_shape_ablation_does_not_use_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            components = {
                "correction_risk_score": 0.0,
                "uncertainty_score": 0.0,
                "rule_review_score": 0.0,
                "geometry_complexity_score": 0.0,
                "boundary_shape_score": 0.5,
                "diversity_score": 0.0,
            }
            base = item(1, False, components=components)
            changed = item(2, True, components=dict(components))
            changed["source_metadata"] = {"boundary_metadata": {"perimeter_area_ratio": 999.0}}
            changed["evaluation_only"] = {
                "delta": {
                    "major_correction": True,
                    "model_human_iou": 0.0,
                    "boundary": {"boundary_f1": 0.0, "boundary_error_area_ratio": 1.0},
                }
            }
            queue = write_queue(root, [base, changed])
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "boundary_shape_experimental.review_queue.jsonl")
            self.assertEqual(ranked[0]["ablation_priority_score"], ranked[1]["ablation_priority_score"])

    def test_calibrated_boundary_ablation_does_not_use_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = item(1, False, priority=0.5)
            changed = json.loads(json.dumps(base))
            changed["task_id"] = "s2"
            changed["evaluation_only"] = {
                "delta": {
                    "major_correction": True,
                    "model_human_iou": 0.0,
                    "model_human_dice": 0.0,
                    "boundary": {"boundary_iou": 0.0, "boundary_f1": 0.0},
                }
            }
            changed["boundary_metadata"] = {"thin_structure_score": 999.0}
            queue = write_queue(root, [base, changed])
            ablate_review_weights(str(queue), str(root / "out"))
            ranked = read_jsonl(root / "out" / "preset_rankings" / "boundary_shape_calibrated_score.review_queue.jsonl")
            self.assertEqual(ranked[0]["ablation_priority_score"], ranked[1]["ablation_priority_score"])

    def test_prediction_feature_diagnostics_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = diagnose_prediction_features(str(queue), str(root / "diag"))
            self.assertTrue((root / "diag" / "prediction_feature_diagnostics.json").exists())
            self.assertTrue((root / "diag" / "prediction_feature_diagnostics.md").exists())
            self.assertIn("pred_boundary_complexity", result["features"])

    def test_learn_boundary_shape_fusion_outputs_oof_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            result = learn_boundary_shape_fusion(str(queue), str(root / "learned"), n_splits=3, bootstrap_iters=5)
            self.assertTrue((root / "learned" / "learned_boundary_shape_fusion.json").exists())
            self.assertTrue((root / "learned" / "learned_fusion_results.json").exists())
            self.assertTrue((root / "learned" / "learned_fusion_bootstrap_ci.json").exists())
            self.assertIn("learned_boundary_shape_only", result["experiments"])
            self.assertTrue(result["experiments"]["learned_boundary_shape_only"]["oof"])
            feature_text = "\n".join(result["feature_sets"]["learned_current_plus_boundary_shape"])
            self.assertNotIn("major_correction", feature_text)
            self.assertNotIn("model_human_iou", feature_text)

    def test_learned_artifact_save_load_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            result = export_learned_fusion_artifacts(str(queue), str(root / "artifacts"), training_dataset_names=["unit"])
            self.assertTrue((root / "artifacts" / "feature_schema.json").exists())
            self.assertTrue((root / "artifacts" / "model_metadata.json").exists())
            self.assertTrue((root / "artifacts" / "validation_report.json").exists())
            self.assertIn("learned_boundary_shape_only_model.pkl", result["model_files"])
            self.assertEqual("learned_boundary_shape_shadow_v1", result["model_metadata"]["artifact_version"])
            self.assertTrue(result["model_metadata"]["shadow_only"])
            self.assertFalse(result["model_metadata"]["affects_default_ranking"])
            self.assertIn("validation_report_path", result["model_metadata"])
            scores = predict_with_artifacts(rows[0], str(root / "artifacts"))
            self.assertIsInstance(scores["learned_boundary_shape_only"], float)
            self.assertIsInstance(scores["learned_current_plus_boundary_shape"], float)
            validation = validate_learned_fusion_artifact(str(root / "artifacts"), str(queue))
            self.assertTrue(validation["passed"])

    def test_learned_artifact_schema_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            export_learned_fusion_artifacts(str(queue), str(root / "artifacts"))
            schema_path = root / "artifacts" / "feature_schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["experiments"]["learned_boundary_shape_only"]["feature_names"] = ["pred_area_ratio"]
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaises(ValueError):
                predict_with_artifacts(rows[0], str(root / "artifacts"))

    def test_apply_shadow_scores_preserves_order_and_adds_learned_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            export_learned_fusion_artifacts(str(queue), str(root / "artifacts"))
            output = root / "shadow_queue.jsonl"
            result = apply_shadow_scores(
                str(queue),
                str(output),
                learned_artifact_dir=str(root / "artifacts"),
                enable_learned_shadow_scores=True,
            )
            scored = read_jsonl(output)
            self.assertTrue(result["preserved_order_and_default_priority"])
            self.assertTrue(result["default_field_equality_check"])
            self.assertTrue(result["ordering_equality_check"])
            self.assertEqual(len(rows), result["shadow_rows_written"])
            self.assertEqual(len(rows), result["learned_non_null_count"])
            self.assertEqual([row["task_id"] for row in rows], [row["task_id"] for row in scored])
            self.assertEqual([row["priority_score"] for row in rows], [row["priority_score"] for row in scored])
            self.assertIsInstance(scored[0]["shadow_scores"]["learned_boundary_shape_only_score"], float)
            self.assertFalse(scored[0]["shadow_score_metadata"]["affects_default_ranking"])

    def test_artifact_registry_current_resolution_and_version_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            registry = root / "registry"
            v1 = registry / "learned_boundary_shape_only_v1"
            v2 = registry / "learned_boundary_shape_only_v2"
            export_learned_fusion_artifacts(str(queue), str(v1))
            export_learned_fusion_artifacts(str(queue), str(v2))
            (registry / "CURRENT").write_text("learned_boundary_shape_only_v1\n", encoding="utf-8")
            env = {
                "SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY": str(registry),
                "SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW": "true",
                "SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW": "true",
            }
            with patch.dict(os.environ, env, clear=True):
                first = apply_shadow_scores(str(queue), str(root / "shadow_v1.jsonl"), enable_learned_shadow_scores=True)
            self.assertEqual(str(v1), first["shadow_runtime_config"]["artifact_path"])
            with patch.dict(os.environ, {**env, "SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION": "learned_boundary_shape_only_v2"}, clear=True):
                second = apply_shadow_scores(str(queue), str(root / "shadow_v2.jsonl"), enable_learned_shadow_scores=True)
            self.assertEqual(str(v2), second["shadow_runtime_config"]["artifact_path"])

    def test_apply_shadow_scores_missing_artifact_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(i, i in {2, 4}, priority=i / 10) for i in range(1, 6)])
            output = root / "shadow_queue.jsonl"
            apply_shadow_scores(
                str(queue),
                str(output),
                learned_artifact_dir=str(root / "missing_artifact"),
                enable_learned_shadow_scores=True,
            )
            scored = read_jsonl(output)
            self.assertIsNone(scored[0]["shadow_scores"]["learned_boundary_shape_only_score"])
            self.assertEqual("missing", scored[0]["shadow_score_metadata"]["artifact_validation_status"])

    def test_apply_shadow_scores_schema_mismatch_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            export_learned_fusion_artifacts(str(queue), str(root / "artifacts"))
            schema_path = root / "artifacts" / "feature_schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["experiments"]["learned_boundary_shape_only"]["feature_names"] = ["pred_area_ratio"]
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            output = root / "shadow_queue.jsonl"
            apply_shadow_scores(
                str(queue),
                str(output),
                learned_artifact_dir=str(root / "artifacts"),
                enable_learned_shadow_scores=True,
            )
            scored = read_jsonl(output)
            self.assertIsNone(scored[0]["shadow_scores"]["learned_boundary_shape_only_score"])
            self.assertEqual("schema_mismatch", scored[0]["shadow_score_metadata"]["artifact_validation_status"])
            with self.assertRaises(ValueError):
                validate_artifact_for_shadow(str(root / "artifacts"))

    def test_apply_shadow_scores_inference_error_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, [item(i, i in {2, 4}, priority=i / 10) for i in range(1, 6)])
            artifact = root / "bad_artifact"
            artifact.mkdir()
            features = ["pred_area_ratio"]
            (artifact / "feature_schema.json").write_text(
                json.dumps(
                    {
                        "artifact_version": "unit_bad",
                        "experiments": {
                            "learned_boundary_shape_only": {"feature_names": features},
                            "learned_current_plus_boundary_shape": {"feature_names": features},
                        },
                        "missing_feature_policy": "neutral_zero_fill",
                    }
                ),
                encoding="utf-8",
            )
            (artifact / "model_metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_version": "unit_bad",
                        "experimental_shadow_only": True,
                        "affects_default_ranking": False,
                    }
                ),
                encoding="utf-8",
            )
            for name in ["learned_boundary_shape_only", "learned_current_plus_boundary_shape"]:
                with (artifact / f"{name}_model.pkl").open("wb") as f:
                    pickle.dump(
                        {
                            "feature_names": features,
                            "scaler": {"median": [0.0], "scale": [1.0]},
                            "model": {"type": "unsupported"},
                        },
                        f,
                    )
            output = root / "shadow_queue.jsonl"
            apply_shadow_scores(
                str(queue),
                str(output),
                learned_artifact_dir=str(artifact),
                enable_learned_shadow_scores=True,
            )
            scored = read_jsonl(output)
            self.assertIsNone(scored[0]["shadow_scores"]["learned_boundary_shape_only_score"])
            self.assertEqual("inference_failed", scored[0]["shadow_score_metadata"]["artifact_validation_status"])
            self.assertIn("unsupported", scored[0]["shadow_score_metadata"]["inference_error"])

    def test_gated_fusion_outputs_shadow_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            result = evaluate_gated_fusion(str(queue), str(root / "gated"), bootstrap_iters=5)
            self.assertTrue((root / "gated" / "gated_fusion_results.json").exists())
            self.assertIn("gated_boundary_shape_score", result["experiments"])
            self.assertIn("overall", result["gate_distribution"])

    def test_shadow_monitoring_outputs_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            for row in rows:
                row["shadow_scores"] = {
                    "boundary_shape_calibrated_score": row["score_components"]["boundary_shape_score"],
                    "boundary_shape_rank_score": row["score_components"]["boundary_shape_score"],
                    "learned_boundary_shape_only_score": row["score_components"]["boundary_shape_score"],
                    "learned_current_plus_boundary_shape_score": row["priority_score"],
                    "gated_boundary_shape_score": row["priority_score"],
                    "gated_current_boundary_score": row["priority_score"],
                    "shadow_only": True,
                }
                row["shadow_score_metadata"] = {
                    "score_version": "boundary_shape_shadow_v1",
                    "shadow_only": True,
                    "affects_default_ranking": False,
                    "artifact_version": "unit",
                }
            queue = write_queue(root, rows)
            result = compare_shadow_scores(str(queue), str(root / "monitor"))
            self.assertTrue((root / "monitor" / "shadow_monitoring.json").exists())
            self.assertTrue((root / "monitor" / "shadow_disagreement_examples.jsonl").exists())
            self.assertIn("learned_boundary_shape_only_score", result["fields"])
            self.assertIn("k20", result["fields"]["learned_boundary_shape_only_score"]["topk_overlap"])
            self.assertIn("inference_error_count", result["coverage"])
            self.assertTrue(result["fields"]["learned_boundary_shape_only_score"]["offline_label_metrics"]["evaluation_only"])

    def test_shadow_monitoring_production_mode_does_not_read_labels_and_has_alert_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            for row in rows:
                row["shadow_scores"] = {
                    "learned_boundary_shape_only_score": row["score_components"]["boundary_shape_score"],
                    "boundary_shape_calibrated_score": row["score_components"]["boundary_shape_score"],
                }
                row["shadow_score_metadata"] = {
                    "score_version": "boundary_shape_shadow_v1",
                    "shadow_only": True,
                    "affects_default_ranking": False,
                    "artifact_loaded": True,
                    "artifact_validation_status": "valid",
                    "artifact_version": "unit",
                    "missing_safe_feature_count": 0,
                }
            queue = write_queue(root, rows)
            result = compare_shadow_scores(
                str(queue),
                str(root / "monitor"),
                window_id="unit-window",
                production_mode=True,
                no_labels=True,
            )
            self.assertTrue(result["production_mode"])
            self.assertFalse(result["labels_read"])
            self.assertFalse(result["label_metrics_available"])
            self.assertNotIn("offline_label_metrics", result["fields"]["learned_boundary_shape_only_score"])
            self.assertIn("alert_thresholds", result)
            self.assertIn("alerts", result)
            self.assertIn("performance", result)
            self.assertIn("latency_fields_available", result["performance"])
            self.assertIn("schema_version", result["performance"])
            self.assertIn("per_item_shadow_scoring_latency_ms", result["performance"])
            self.assertIn("artifact_load_latency_ms", result["performance"])
            self.assertIsNone(result["performance"]["queue_generation_latency_ms"])
            self.assertIn("production_labels_read", result["alerts"]["checks"])
            self.assertFalse(result["alerts"]["checks"]["production_labels_read"]["triggered"])
            self.assertIn("shadow_scoring_latency_p95", result["alerts"]["checks"])
            self.assertIn("artifact_load_latency", result["alerts"]["checks"])
            self.assertIn("score_distribution_p95_shift_std", result["alerts"]["checks"])
            self.assertIn("top100_jaccard_relative_change", result["alerts"]["checks"])

    def test_shadow_monitoring_baseline_drift_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            for row in rows:
                row["shadow_scores"] = {"learned_boundary_shape_only_score": row["priority_score"]}
                row["shadow_score_metadata"] = {
                    "score_version": "boundary_shape_shadow_v1",
                    "shadow_only": True,
                    "affects_default_ranking": False,
                    "artifact_loaded": True,
                    "artifact_validation_status": "valid",
                    "missing_safe_feature_count": 0,
                }
            queue = write_queue(root, rows)
            baseline = compare_shadow_scores(str(queue), str(root / "baseline"), window_id="baseline", production_mode=True)
            current = compare_shadow_scores(
                str(queue),
                str(root / "current"),
                window_id="current",
                baseline=str(root / "baseline" / "shadow_monitoring.json"),
                production_mode=True,
            )
            self.assertFalse(baseline["distribution_drift"]["baseline_available"])
            self.assertTrue(current["distribution_drift"]["baseline_available"])
            self.assertIn("top100_jaccard_relative_change", current["distribution_drift"]["learned_boundary_shape_only_score"])
            self.assertIn("median_delta", current["distribution_drift"]["learned_boundary_shape_only_score"])
            self.assertIn("score_distribution_psi", current["distribution_drift"]["learned_boundary_shape_only_score"])

    def test_shadow_review_packet_safe_fields_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            for row in rows:
                row["shadow_scores"] = {"learned_boundary_shape_only_score": 1.0 - row["priority_score"]}
                row["shadow_score_metadata"] = {
                    "artifact_validation_status": "valid",
                    "artifact_version": "unit",
                    "shadow_only": True,
                    "affects_default_ranking": False,
                }
            queue = write_queue(root, rows)
            result = export_shadow_review_packet(str(queue), str(root / "packet"), top_k=3)
            self.assertTrue((root / "packet" / "high_learned_low_current.jsonl").exists())
            packet_rows = read_jsonl(root / "packet" / "high_learned_low_current.jsonl")
            self.assertTrue(result["safe_fields_only"])
            self.assertTrue(packet_rows)
            self.assertNotIn("evaluation_only", packet_rows[0])
            self.assertNotIn("delta", packet_rows[0])
            self.assertIn("prediction_time_safe_features", packet_rows[0])

    def test_shadow_review_packet_production_mode_safe_and_offline_labels_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            for row in rows:
                row["shadow_scores"] = {"learned_boundary_shape_only_score": 1.0 - row["priority_score"]}
                row["shadow_score_metadata"] = {
                    "artifact_validation_status": "valid",
                    "artifact_version": "unit",
                    "shadow_only": True,
                    "affects_default_ranking": False,
                }
            queue = write_queue(root, rows)
            prod = export_shadow_review_packet(
                str(queue),
                str(root / "packet_prod"),
                top_k=3,
                production_mode=True,
                include_evaluation_labels=True,
            )
            prod_rows = read_jsonl(root / "packet_prod" / "high_learned_low_current.jsonl")
            self.assertTrue(prod["production_mode"])
            self.assertFalse(prod["evaluation_only_metrics_included"])
            self.assertNotIn("evaluation_only", prod_rows[0])
            offline = export_shadow_review_packet(
                str(queue),
                str(root / "packet_offline"),
                top_k=3,
                include_evaluation_labels=True,
            )
            offline_rows = read_jsonl(root / "packet_offline" / "high_learned_low_current.jsonl")
            self.assertTrue(offline["evaluation_only_metrics_included"])
            self.assertIn("evaluation_only", offline_rows[0])

    def test_live_shadow_rollout_first_window_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            queue = write_queue(root, rows)
            export_learned_fusion_artifacts(str(queue), str(root / "artifacts"))
            report_path = root / "production_shadow_live_window_001_report.md"
            result = run_live_shadow_rollout(
                str(queue),
                str(root / "live"),
                artifact_dir=str(root / "artifacts"),
                report_path=str(report_path),
            )
            self.assertTrue((root / "live" / "live_review_queue.shadow_scored.jsonl").exists())
            self.assertTrue((root / "live" / "shadow_monitoring" / "shadow_monitoring.json").exists())
            self.assertTrue((root / "live" / "shadow_review_packet" / "review_packet_summary.md").exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(result["queue_batch_summary"]["default_field_equality_check"])
            self.assertTrue(result["queue_batch_summary"]["ordering_equality_check"])
            self.assertFalse(result["leakage_guard_summary"]["production_labels_read"])
            self.assertFalse(result["recommendation"]["ready_for_default_promotion"])
            self.assertTrue(result["recommendation"]["default_layer_5_weights_unchanged"])
            self.assertTrue(result["recommendation"]["default_sorting_unchanged"])
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Rollout Config", text)
            self.assertIn("Recommendation", text)

    def test_multi_window_shadow_observation_outputs_summary_and_safe_combined_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_a = [item(i, i in {2, 4, 6, 8}, priority=i / 12) for i in range(1, 13)]
            rows_b = [item(i, i in {1, 3, 5, 7}, priority=(13 - i) / 12) for i in range(1, 13)]
            queue_a = root / "queue_a.jsonl"
            queue_b = root / "queue_b.jsonl"
            queue_a.write_text("\n".join(json.dumps(row) for row in rows_a) + "\n", encoding="utf-8")
            queue_b.write_text("\n".join(json.dumps(row) for row in rows_b) + "\n", encoding="utf-8")
            export_learned_fusion_artifacts(str(queue_a), str(root / "artifacts"))
            result = run_multi_window_shadow_observation(
                [str(queue_a), str(queue_b)],
                str(root / "multi"),
                artifact_dir=str(root / "artifacts"),
                decision_report_path=str(root / "decision.md"),
            )
            self.assertEqual(2, result["windows_processed"])
            self.assertTrue((root / "multi" / "window_001" / "live_review_queue.shadow_scored.jsonl").exists())
            self.assertTrue((root / "multi" / "multi_window_shadow_summary.json").exists())
            self.assertTrue((root / "multi" / "broader_shadow_multi_window_summary.json").exists())
            self.assertTrue((root / "decision.md").exists())
            self.assertIn("coverage_stability", result)
            self.assertIn("agreement_stability", result)
            self.assertIn("median_drift_vs_baseline_per_window", result["distribution_stability"])
            self.assertIn("p95_drift_outliers", result["distribution_stability"])
            self.assertTrue((root / "multi" / "expanded_shadow_multi_window_summary.json").exists())
            self.assertIn("human_disagreement_review_summary", result)
            self.assertIn("human_review_task_packet", result)
            self.assertTrue((root / "multi" / "human_review_task_packet" / "review_tasks_high_learned_low_current.jsonl").exists())
            combined = read_jsonl(root / "multi" / "human_disagreement_review_summary" / "combined_high_learned_low_current.jsonl")
            self.assertTrue(combined)
            self.assertIn("window_ids", combined[0])
            self.assertIn("human_review_outcome", combined[0])
            self.assertNotIn("evaluation_only", combined[0])
            self.assertNotIn("delta", combined[0])
            self.assertTrue(all(row["default_field_equality"] for row in result["windows"]))
            self.assertTrue(all(row["ordering_equality"] for row in result["windows"]))
            self.assertTrue(all(row["production_labels_read"] is False for row in result["windows"]))

    def test_multi_window_hard_safety_alert_blocks_expansion(self):
        good = {
            "window_id": "w1",
            "queue_batch_summary": {"total_rows": 10, "default_field_equality_check": True, "ordering_equality_check": True},
            "shadow_coverage": {
                "total_rows": 10,
                "learned_score_non_null_count": 10,
                "learned_score_null_rate": 0.0,
                "artifact_validation_status_counts": {"valid": 10},
                "inference_error_count": 0,
            },
            "monitoring": {
                "labels_read": False,
                "alerts": {"triggered": False, "checks": {"missing_safe_feature_count_p95": {"value": 0}}},
                "safety_summary": {"shadow_only": True, "affects_default_ranking": False, "leakage_guard_status": "metadata_present"},
                "fields": {"learned_boundary_shape_only_score": {"topk_overlap": {}, "correlation": {}, "rank_delta_summary": {}}},
                "distribution_drift": {"learned_boundary_shape_only_score": {}},
            },
        }
        bad = json.loads(json.dumps(good))
        bad["window_id"] = "w2"
        bad["queue_batch_summary"]["ordering_equality_check"] = False
        result = build_multi_window_summary([good, bad], output_root="out")
        self.assertTrue(result["alerts"]["hard_safety_alert"])
        self.assertEqual("rollback", result["decision"]["recommended_action"])
        self.assertFalse(result["decision"]["expand_shadow_traffic"])

    def test_multi_window_drift_outlier_schema(self):
        good = {
            "window_id": "w1",
            "input_review_queue": "queue.jsonl",
            "queue_batch_summary": {"total_rows": 10, "default_field_equality_check": True, "ordering_equality_check": True},
            "shadow_coverage": {
                "total_rows": 10,
                "learned_score_non_null_count": 10,
                "learned_score_null_rate": 0.0,
                "artifact_validation_status_counts": {"valid": 10},
                "inference_error_count": 0,
            },
            "monitoring": {
                "labels_read": False,
                "alerts": {"triggered": False, "checks": {"missing_safe_feature_count_p95": {"value": 0}}},
                "safety_summary": {"shadow_only": True, "affects_default_ranking": False, "leakage_guard_status": "metadata_present"},
                "fields": {"learned_boundary_shape_only_score": {"topk_overlap": {}, "correlation": {}, "rank_delta_summary": {}}},
                "distribution_drift": {"learned_boundary_shape_only_score": {"p95_shift_std": 1.2, "score_distribution_psi": 0.05}},
            },
        }
        result = build_multi_window_summary([good], output_root="out")
        self.assertEqual(1, len(result["distribution_stability"]["p95_drift_outliers"]))
        self.assertEqual("w1", result["distribution_stability"]["p95_drift_outliers"][0]["window_id"])

    def test_human_review_task_packet_safe_fields_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "combined_high_learned_low_current.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "current_priority_score": 0.1,
                        "learned_shadow_score": 0.9,
                        "rank_current": 10,
                        "rank_learned": 1,
                        "rank_delta": 9,
                        "prediction_time_safe_features": {"pred_area_ratio": 0.1},
                        "artifact_version": "unit",
                        "shadow_metadata_status": "valid",
                        "window_ids": ["w1"],
                        "evaluation_only": {"label": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = {
                "output_dir": str(root),
                "outputs": {
                    "high_learned_low_current": str(source),
                    "high_current_low_learned": str(source),
                    "top_learned": str(source),
                    "top_current": str(source),
                },
            }
            result = build_human_review_task_packet(summary, str(root / "tasks"), high_learned_low_current_limit=1, other_limit=1)
            task_rows = read_jsonl(root / "tasks" / "review_tasks_high_learned_low_current.jsonl")
            self.assertTrue(result["safe_fields_only"])
            self.assertEqual("high_learned_low_current", task_rows[0]["group"])
            self.assertIn("reviewer", task_rows[0])
            self.assertIn("reviewed_at", task_rows[0])
            self.assertNotIn("evaluation_only", task_rows[0])

    def test_shadow_human_feedback_summary_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "feedback.jsonl"
            rows = [
                {"sample_id": "a", "group": "high_learned_low_current", "learned_shadow_score": 0.9, "current_priority_score": 0.1, "human_review_outcome": "major_correction_needed"},
                {"sample_id": "b", "group": "top_learned", "learned_shadow_score": 0.8, "current_priority_score": 0.2, "human_review_outcome": "minor_correction_needed"},
                {"sample_id": "c", "group": "control_current_top", "learned_shadow_score": 0.2, "current_priority_score": 0.9, "human_review_outcome": "ok"},
            ]
            packet.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = summarize_shadow_human_feedback(str(packet), str(root / "summary"))
            self.assertTrue(result["offline_analysis_only"])
            self.assertEqual(3, result["rows"])
            self.assertIn("correction_rate_by_learned_score_bucket", result)
            self.assertIn("total_reviewed", result)
            self.assertIn("reviewed_by_group", result)
            self.assertIn("learned_lift_vs_control", result)
            self.assertIn("inter_reviewer_agreement", result)
            self.assertTrue((root / "summary" / "human_feedback_summary.json").exists())
            self.assertTrue((root / "summary" / "human_feedback_examples.jsonl").exists())

    def test_shadow_human_feedback_empty_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = summarize_shadow_human_feedback(str(root / "missing.jsonl"), str(root / "summary"))
            self.assertEqual(0, result["total_reviewed"])
            self.assertFalse(result["feedback_available"])
            self.assertEqual({}, result["outcome_counts"])

    def test_shadow_human_feedback_examples_require_reviewed_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "feedback.jsonl"
            rows = [
                {
                    "sample_id": "a",
                    "group": "high_learned_low_current",
                    "learned_shadow_score": 0.9,
                    "current_priority_score": 0.1,
                    "human_review_outcome": None,
                },
                {
                    "sample_id": "b",
                    "group": "top_learned",
                    "learned_shadow_score": 0.8,
                    "current_priority_score": 0.2,
                    "human_review_outcome": "",
                },
            ]
            packet.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = summarize_shadow_human_feedback(str(packet), str(root / "summary"))
            examples = read_jsonl(root / "summary" / "human_feedback_examples.jsonl")
            self.assertEqual(0, result["total_reviewed"])
            self.assertEqual([], result["learned_found_missed_risks"])
            self.assertEqual([], result["learned_over_prioritized"])
            self.assertEqual([], examples)

    def test_shadow_root_cause_analysis_outputs_schema_and_safe_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "window_001"
            target = root / "window_004"
            base.mkdir()
            target.mkdir()
            rows_base = [item(i, i % 3 == 0, priority=i / 20) for i in range(1, 13)]
            rows_target = [item(i + 100, i % 2 == 0, priority=(13 - i) / 20) for i in range(1, 13)]
            for idx, row in enumerate(rows_base):
                row["shadow_scores"] = {"learned_boundary_shape_only_score": row["priority_score"], "boundary_shape_rank_score": row["priority_score"]}
                row["shadow_score_metadata"] = {"artifact_validation_status": "valid", "missing_safe_feature_count": 0, "missing_safe_features": []}
            for idx, row in enumerate(rows_target):
                row["prediction_features"].pop("pred_thinness_proxy", None)
                row["shadow_scores"] = {"learned_boundary_shape_only_score": 1.0 - row["priority_score"], "boundary_shape_rank_score": 1.0 - row["priority_score"]}
                row["shadow_score_metadata"] = {"artifact_validation_status": "valid", "missing_safe_feature_count": 1, "missing_safe_features": ["pred_thinness_proxy"]}
                row["evaluation_only"] = {"label": True}
            (base / "live_review_queue.shadow_scored.jsonl").write_text("\n".join(json.dumps(row) for row in rows_base) + "\n", encoding="utf-8")
            (target / "live_review_queue.shadow_scored.jsonl").write_text("\n".join(json.dumps(row) for row in rows_target) + "\n", encoding="utf-8")
            drift = analyze_shadow_drift(str(root), "window_001", ["window_004"], str(root / "drift"))
            missing = analyze_shadow_missing_features(str(root), ["window_004"], str(root / "missing"))
            topk = analyze_shadow_topk_jaccard(str(root), "window_001", ["window_004"], str(root / "topk"), k=5)
            self.assertIn("feature_distribution_diffs", drift)
            self.assertTrue((root / "drift" / "drift_outlier_samples.jsonl").exists())
            outlier_rows = read_jsonl(root / "drift" / "drift_outlier_samples.jsonl")
            self.assertNotIn("evaluation_only", outlier_rows[0])
            self.assertIn("pred_thinness_proxy", missing["per_window"]["window_004"]["missing_feature_names_frequency"])
            self.assertIn("normalized_jaccard_on_shared_samples", topk["per_window"]["window_004"])

    def test_human_review_assignment_and_feedback_csv_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "tasks"
            packet.mkdir()
            row = {
                "task_id": "task-1",
                "group": "high_learned_low_current",
                "current_priority_score": 0.1,
                "learned_shadow_score": 0.9,
                "rank_current": 10,
                "rank_learned": 1,
                "rank_delta": 9,
                "prediction_time_safe_features": {"pred_area_ratio": 0.01},
                "window_ids": ["window_004"],
                "evaluation_only": {"label": True},
            }
            (packet / "review_tasks_high_learned_low_current.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            assignment = build_human_review_assignment(str(packet), str(root / "assignment"))
            rows = read_jsonl(root / "assignment" / "review_assignment.jsonl")
            self.assertEqual(1, assignment["rows"])
            self.assertNotIn("evaluation_only", rows[0])
            csv_path = root / "feedback.csv"
            csv_path.write_text(
                "sample_id,group,human_review_outcome,reviewer,reviewed_at,notes,current_priority_score,learned_shadow_score\n"
                "s1,high_learned_low_current,major_correction_needed,r1,2026-06-03T00:00:00Z,,0.1,0.9\n"
                "s2,control_current_top,ok,r1,2026-06-03T00:00:00Z,,0.9,0.2\n",
                encoding="utf-8",
            )
            summary = summarize_shadow_human_feedback(str(csv_path), str(root / "feedback_summary"))
            self.assertEqual(2, summary["total_reviewed"])
            self.assertIn("learned_missed_risk_discovery_rate", summary)
            self.assertIn("over_prioritization_rate", summary)

    def test_ablation_bootstrap_ci_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = write_queue(root, self.rows())
            result = ablate_review_weights(str(queue), str(root / "out"), bootstrap_iters=5)
            self.assertIn("bootstrap_ci", result["presets"]["risk_heavy"])
            self.assertIn("pairwise_delta_vs_current", result["bootstrap_ci"])
            self.assertIn("roc_auc_for_major_correction", result["bootstrap_ci"]["metrics"])

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
