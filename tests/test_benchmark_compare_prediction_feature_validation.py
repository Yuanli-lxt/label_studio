import json
from pathlib import Path

from image_segmentation.benchmark.compare_prediction_feature_validation import compare_prediction_feature_validation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_prediction_feature_validation_accepts_run_dirs(tmp_path):
    root = tmp_path / "run"
    _write_json(
        root / "evaluation_report.json",
        {"overall_quality": {"n_samples": 2, "mean_model_gt_iou": 0.5, "median_model_gt_iou": 0.5, "mean_dice": 0.6, "major_correction_rate": 0.5}},
    )
    _write_json(
        root / "prediction_feature_diagnostics" / "prediction_feature_diagnostics.json",
        {
            "positive_count": 1,
            "features": {
                name: {"missing_count": 0, "constant": False, "average_precision": 0.5, "roc_auc": 0.5, "direction_suggestion": "higher_is_risk"}
                for name in [
                    "pred_hole_count",
                    "pred_area_ratio",
                    "pred_extent",
                    "pred_component_count",
                    "pred_boundary_complexity",
                    "pred_boundary_density",
                    "pred_thinness_proxy",
                    "pred_touches_border",
                ]
            },
        },
    )
    _write_json(
        root / "ablation" / "ablation_results.json",
        {
            "presets": {
                "current_full_priority": {"average_precision_for_major_correction": 0.5},
                "boundary_shape_calibrated_score": {"average_precision_for_major_correction": 0.6},
            }
        },
    )
    _write_json(
        root / "ablation" / "bootstrap_ci.json",
        {
            "presets": {
                "boundary_shape_calibrated_score": {
                    "average_precision_for_major_correction": {"value": 0.6, "ci95_low": 0.55, "ci95_high": 0.65}
                }
            },
            "pairwise_delta_vs_current": {
                "boundary_shape_calibrated_score_vs_current_full_priority": {
                    "average_precision_for_major_correction": {"value": 0.1, "ci95_low": 0.01, "ci95_high": 0.2}
                }
            },
        },
    )
    _write_json(
        root / "learned_fusion" / "learned_fusion_results.json",
        {"experiments": {"learned_boundary_shape_only": {"oof": True, "average_precision": 0.7}}},
    )
    _write_json(
        root / "learned_fusion" / "learned_fusion_bootstrap_ci.json",
        {
            "experiments": {
                "learned_boundary_shape_only": {
                    "average_precision": {"value": 0.7, "ci95_low": 0.6, "ci95_high": 0.8}
                }
            },
            "pairwise_delta_vs_current": {
                "learned_boundary_shape_only_vs_current_full_priority": {
                    "average_precision": {"value": 0.2, "ci95_low": 0.05, "ci95_high": 0.3}
                }
            },
        },
    )
    result = compare_prediction_feature_validation([f"toy={root}"], str(tmp_path / "comparison.md"))
    assert result["runs"][0]["name"] == "toy"
    assert (tmp_path / "comparison.md").exists()
    text = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "0.6000 [0.5500, 0.6500]" in text
    assert "0.2000 [0.0500, 0.3000]" in text
