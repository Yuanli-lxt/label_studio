import json
from pathlib import Path

from image_segmentation.benchmark.compare_benchmark_runs import compare_benchmark_runs, compare_benchmark_runs_multi


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _report(rate=0.2):
    return {
        "overall_quality": {
            "n_samples": 10,
            "n_unique_images": 8,
            "n_categories": 3,
            "mean_model_gt_iou": 0.7,
            "median_model_gt_iou": 0.75,
            "mean_dice": 0.8,
            "mean_precision": 0.82,
            "mean_recall": 0.78,
            "major_correction_count": int(rate * 10),
            "major_correction_rate": rate,
        },
        "major_correction_diagnostics": {
            "category_level_major_correction_top20": [{"dataset": "thing", "major_correction_rate": rate}],
            "difficulty_tag_major_correction_rates": [{"tag": "small_object", "major_correction_rate": rate}],
        },
        "boundary_quality": {"mean_boundary_f1": 0.55},
        "category_frequency_breakdown": [{"frequency": "r", "major_correction_rate": rate}],
    }


def _ablation(best="risk_only"):
    presets = {
        "current_full_priority": {"average_precision_for_major_correction": 0.2, "lift_at_20_percent_over_random": 1.1},
        "risk_only": {"average_precision_for_major_correction": 0.4, "lift_at_20_percent_over_random": 2.0},
        "risk_heavy": {"average_precision_for_major_correction": 0.3, "lift_at_20_percent_over_random": 1.6},
    }
    return {"presets": presets, "leaderboard_by_ap": [{"preset": best, "average_precision_for_major_correction": presets[best]["average_precision_for_major_correction"]}]}


def test_compare_benchmark_runs_outputs_report(tmp_path):
    left_report = tmp_path / "left_report.json"
    right_report = tmp_path / "right_report.json"
    left_oof = tmp_path / "left_oof.json"
    right_oof = tmp_path / "right_oof.json"
    left_ablation = tmp_path / "left_ablation.json"
    right_ablation = tmp_path / "right_ablation.json"
    for path, payload in [
        (left_report, _report(0.2)),
        (right_report, _report(0.4)),
        (left_oof, {"positive_count": 2, "oof_pr_auc": 0.5}),
        (right_oof, {"positive_count": 4, "oof_pr_auc": 0.6}),
        (left_ablation, _ablation()),
        (right_ablation, _ablation()),
    ]:
        _write(path, payload)
    out = tmp_path / "comparison.md"
    compare_benchmark_runs("COCO1000", str(left_report), str(left_oof), str(left_ablation), "LVIS500", str(right_report), str(right_oof), str(right_ablation), str(out))
    assert out.exists()
    assert "Quality Comparison" in out.read_text(encoding="utf-8")


def test_compare_benchmark_runs_handles_missing_ablation(tmp_path):
    left_report = tmp_path / "left_report.json"
    right_report = tmp_path / "right_report.json"
    _write(left_report, _report(0.2))
    _write(right_report, _report(0.4))
    out = tmp_path / "comparison.md"
    result = compare_benchmark_runs("A", str(left_report), None, None, "B", str(right_report), None, None, str(out))
    assert result["left"]["strategy"]["available"] is False
    assert out.exists()


def test_compare_benchmark_runs_recommendation_present(tmp_path):
    left_report = tmp_path / "left_report.json"
    right_report = tmp_path / "right_report.json"
    _write(left_report, _report(0.2))
    _write(right_report, _report(0.5))
    out = tmp_path / "comparison.md"
    result = compare_benchmark_runs("A", str(left_report), None, None, "B", str(right_report), None, None, str(out))
    assert result["recommendation"]
    assert "Recommendation" in out.read_text(encoding="utf-8")


def test_compare_benchmark_runs_multi_run_outputs_report(tmp_path):
    specs = []
    for name in ["COCO1000", "LVIS500", "DIS5K300"]:
        report = tmp_path / f"{name}_report.json"
        oof = tmp_path / f"{name}_oof.json"
        ablation = tmp_path / f"{name}_ablation.json"
        _write(report, _report(0.3))
        _write(oof, {"positive_count": 3, "oof_pr_auc": 0.6, "fold_positive_counts": [1, 1, 1]})
        _write(ablation, _ablation())
        specs.append(f"{name}:{report}:{oof}:{ablation}")
    out = tmp_path / "multi.md"
    result = compare_benchmark_runs_multi(specs, str(out))
    assert len(result["runs"]) == 3
    text = out.read_text(encoding="utf-8")
    assert "DIS5K300" in text
    assert "boundary F1" in text
    assert "risk_heavy beats current across all runs" in text
