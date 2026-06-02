from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_benchmark_runs(
    left_name: str,
    left_report: str,
    left_oof: str | None,
    left_ablation: str | None,
    right_name: str,
    right_report: str,
    right_oof: str | None,
    right_ablation: str | None,
    output: str,
) -> dict:
    left = _run_bundle(left_name, left_report, left_oof, left_ablation)
    right = _run_bundle(right_name, right_report, right_oof, right_ablation)
    recommendation = _recommendation(left, right)
    markdown = render_markdown(left, right, recommendation)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {"left": left, "right": right, "recommendation": recommendation, "output": str(output_path)}


def compare_benchmark_runs_multi(runs: list[str], output: str) -> dict:
    bundles = [_parse_run_spec(spec) for spec in runs]
    recommendation = _multi_recommendation(bundles)
    markdown = render_multi_markdown(bundles, recommendation)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {"runs": bundles, "recommendation": recommendation, "output": str(output_path)}


def render_multi_markdown(runs: list[dict], recommendation: str) -> str:
    names = [run["name"] for run in runs]
    lines = [
        "# Benchmark Run Comparison",
        "",
        "## Quality Comparison",
        "| metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join(["---:"] * len(runs)) + " |",
    ]
    for key, label in [
        ("samples", "samples"),
        ("mean_iou", "mean IoU"),
        ("median_iou", "median IoU"),
        ("dice", "Dice"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("boundary_f1", "boundary F1"),
        ("major_correction_count", "major correction count"),
        ("major_correction_rate", "major correction base rate"),
    ]:
        lines.append("| " + label + " | " + " | ".join(_fmt(run["quality"].get(key)) for run in runs) + " |")
    lines.extend(["", "## OOF Risk Comparison", "| metric | " + " | ".join(names) + " |", "| --- | " + " | ".join(["---:"] * len(runs)) + " |"])
    for key, label in [
        ("positive_count", "positive count"),
        ("oof_pr_auc", "PR-AUC"),
        ("oof_roc_auc", "ROC-AUC"),
        ("oof_brier_score", "Brier score"),
        ("fold_positive_counts", "fold positive counts"),
    ]:
        lines.append("| " + label + " | " + " | ".join(_fmt(run["oof"].get(key)) for run in runs) + " |")
    lines.extend(["", "## Strategy Comparison"])
    for run in runs:
        lines.append(f"- {run['name']} best by AP: {run['strategy'].get('best_by_ap')}")
    lines.append(f"- risk_heavy beats current across all runs: {_bool_text(all(_beats_current(run, 'risk_heavy') for run in runs))}")
    lines.append(f"- risk_only remains strongest across all runs: {_bool_text(all(_is_best(run, 'risk_only') for run in runs))}")
    lines.extend(["", "| preset metric | " + " | ".join(names) + " |", "| --- | " + " | ".join(["---:"] * len(runs)) + " |"])
    for preset in [
        "current_full_priority",
        "risk_heavy",
        "risk_only",
        "no_diversity",
        "boundary_shape_experimental",
        "boundary_shape_rank_score",
        "boundary_shape_calibrated_score",
    ]:
        for metric in ["average_precision_for_major_correction", "lift_at_20_percent_over_random"]:
            lines.append("| " + f"{preset} {metric}" + " | " + " | ".join(_fmt(_preset_metric(run, preset, metric)) for run in runs) + " |")
    lines.extend(["", "## Failure-Mode Comparison"])
    for run in runs:
        lines.append(f"- {run['name']} top failure categories: {_top_failures(run)}")
        lines.append(f"- {run['name']} difficulty/boundary diagnostics: {run['failure_modes'].get('difficulty_tags')}")
    lines.extend(["", "## Recommendation", f"- {recommendation}", "", "## Leakage Guard", "- Boundary metrics and GT-derived manifest metadata are evaluation-only and are not priority scoring inputs.", ""])
    return "\n".join(lines)


def render_markdown(left: dict, right: dict, recommendation: str) -> str:
    return "\n".join(
        [
            f"# {left['name']} vs {right['name']} Benchmark Comparison",
            "",
            "## Quality Comparison",
            "| metric | " + left["name"] + " | " + right["name"] + " |",
            "| --- | ---: | ---: |",
            *_metric_rows(left["quality"], right["quality"], [
                ("samples", "samples"),
                ("n_unique_images", "unique images"),
                ("n_categories", "categories"),
                ("mean_iou", "mean IoU"),
                ("median_iou", "median IoU"),
                ("dice", "Dice"),
                ("precision", "precision"),
                ("recall", "recall"),
                ("major_correction_count", "major correction count"),
                ("major_correction_rate", "major correction base rate"),
            ]),
            "",
            "## OOF Risk Comparison",
            "| metric | " + left["name"] + " | " + right["name"] + " |",
            "| --- | ---: | ---: |",
            *_metric_rows(left["oof"], right["oof"], [
                ("positive_count", "positive count"),
                ("positive_rate", "positive rate"),
                ("oof_pr_auc", "PR-AUC"),
                ("oof_roc_auc", "ROC-AUC"),
                ("oof_brier_score", "Brier score"),
                ("fold_positive_counts", "fold positive counts"),
            ]),
            "",
            "## Strategy Comparison",
            f"- {left['name']} best by AP: {left['strategy'].get('best_by_ap')}",
            f"- {right['name']} best by AP: {right['strategy'].get('best_by_ap')}",
            f"- risk_heavy beats current in both datasets: {_bool_text(_beats_current(left, 'risk_heavy') and _beats_current(right, 'risk_heavy'))}",
            f"- correction_risk_only remains strongest in both datasets: {_bool_text(_is_best(left, 'risk_only') and _is_best(right, 'risk_only'))}",
            "",
            "| preset metric | " + left["name"] + " | " + right["name"] + " |",
            "| --- | ---: | ---: |",
            *_strategy_rows(left, right),
            "",
            "## Failure-Mode Comparison",
            f"- {left['name']} top failure categories: {_top_failures(left)}",
            f"- {right['name']} top failure categories: {_top_failures(right)}",
            f"- {right['name']} frequency breakdown: {right['failure_modes'].get('frequency_breakdown')}",
            f"- {left['name']} difficulty tag breakdown: {left['failure_modes'].get('difficulty_tags')}",
            f"- {right['name']} difficulty tag breakdown: {right['failure_modes'].get('difficulty_tags')}",
            "",
            "## Recommendation",
            f"- {recommendation}",
            "",
            "## Leakage Guard",
            "- Ground truth, IoU, correction deltas, and evaluation-only labels are used only for evaluation reports and comparison, not for priority scoring.",
            "",
        ]
    )


def _run_bundle(name: str, report_path: str, oof_path: str | None, ablation_path: str | None) -> dict:
    report = _read_json(report_path)
    oof = _read_json(oof_path) if oof_path and Path(oof_path).exists() else {}
    ablation = _read_json(ablation_path) if ablation_path and Path(ablation_path).exists() else {}
    return {
        "name": name,
        "quality": _quality(report),
        "oof": oof,
        "strategy": _strategy(ablation),
        "failure_modes": _failure_modes(report),
        "paths": {"report": report_path, "oof": oof_path, "ablation": ablation_path},
    }


def _quality(report: dict) -> dict:
    overall = report.get("overall_quality") or {}
    boundary = report.get("boundary_quality") or {}
    return {
        "samples": overall.get("n_samples"),
        "n_unique_images": overall.get("n_unique_images"),
        "n_categories": overall.get("n_categories"),
        "mean_iou": overall.get("mean_model_gt_iou"),
        "median_iou": overall.get("median_model_gt_iou"),
        "dice": overall.get("mean_dice"),
        "precision": overall.get("mean_precision"),
        "recall": overall.get("mean_recall"),
        "boundary_f1": boundary.get("mean_boundary_f1"),
        "major_correction_count": overall.get("major_correction_count"),
        "major_correction_rate": overall.get("major_correction_rate"),
    }


def _strategy(ablation: dict) -> dict:
    presets = ablation.get("presets") if isinstance(ablation.get("presets"), dict) else {}
    leaderboard = ablation.get("leaderboard_by_ap") or []
    return {
        "available": bool(presets),
        "best_by_ap": leaderboard[0].get("preset") if leaderboard else None,
        "presets": presets,
    }


def _failure_modes(report: dict) -> dict:
    diagnostics = report.get("major_correction_diagnostics") or {}
    boundary = report.get("boundary_stress_diagnostics") or {}
    return {
        "top_failure_categories": diagnostics.get("category_level_major_correction_top20") or [],
        "difficulty_tags": boundary.get("difficulty_tags") or diagnostics.get("difficulty_tag_major_correction_rates") or [],
        "frequency_breakdown": report.get("category_frequency_breakdown") or diagnostics.get("category_frequency_breakdown") or [],
    }


def _parse_run_spec(spec: str) -> dict:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError("--run must be NAME:evaluation_report.json:oof_summary.json:weight_ablation.json")
    return _run_bundle(parts[0], parts[1], parts[2], parts[3])


def _metric_rows(left: dict, right: dict, metrics: list[tuple[str, str]]) -> list[str]:
    return [f"| {label} | {_fmt(left.get(key))} | {_fmt(right.get(key))} |" for key, label in metrics]


def _strategy_rows(left: dict, right: dict) -> list[str]:
    rows = []
    for preset in [
        "current_full_priority",
        "risk_only",
        "risk_heavy",
        "risk_dominant",
        "no_diversity",
        "balanced_no_risk",
        "boundary_shape_experimental",
        "boundary_shape_rank_score",
        "boundary_shape_calibrated_score",
    ]:
        for metric in ["average_precision_for_major_correction", "lift_at_20_percent_over_random"]:
            rows.append(f"| {preset} {metric} | {_fmt(_preset_metric(left, preset, metric))} | {_fmt(_preset_metric(right, preset, metric))} |")
    return rows


def _preset_metric(bundle: dict, preset: str, metric: str) -> Any:
    presets = ((bundle.get("strategy") or {}).get("presets") or {})
    row = presets.get(preset) or {}
    return row.get(metric)


def _beats_current(bundle: dict, preset: str) -> bool:
    return _num(_preset_metric(bundle, preset, "average_precision_for_major_correction")) > _num(
        _preset_metric(bundle, "current_full_priority", "average_precision_for_major_correction")
    )


def _is_best(bundle: dict, preset: str) -> bool:
    return ((bundle.get("strategy") or {}).get("best_by_ap")) == preset


def _top_failures(bundle: dict) -> list[str]:
    return [
        str(row.get("dataset") or row.get("category_name") or "unknown")
        for row in (bundle.get("failure_modes") or {}).get("top_failure_categories", [])[:5]
    ]


def _recommendation(left: dict, right: dict) -> str:
    right_harder = _num(right["quality"].get("major_correction_rate")) > _num(left["quality"].get("major_correction_rate")) * 1.25
    risk_heavy_both = _beats_current(left, "risk_heavy") and _beats_current(right, "risk_heavy")
    if risk_heavy_both:
        return "Keep the default unchanged for now, but mark risk_heavy as a strong experimental candidate across COCO/LVIS."
    if _is_best(right, "risk_only") and not _is_best(left, "risk_only"):
        return "LVIS favors a different preset; do not change defaults yet and validate DIS5K or COD10K next."
    if right_harder:
        return "LVIS appears harder and should be used as a primary risk-training stress test before any Layer 5 default change."
    return "Keep the current default unchanged; continue with DIS5K or COD10K validation before Layer 5 tuning."


def _multi_recommendation(runs: list[dict]) -> str:
    risk_heavy_all = all(_beats_current(run, "risk_heavy") for run in runs)
    dis5k = next((run for run in runs if "DIS5K" in run["name"].upper()), None)
    if risk_heavy_all:
        base = "Keep the default unchanged, but promote risk_heavy as a strong experimental preset across the compared runs."
    else:
        base = "Keep the default unchanged; preset behavior is not yet consistent across all compared runs."
    if dis5k and _num(dis5k["quality"].get("boundary_f1")) and _num(dis5k["quality"].get("boundary_f1")) < 0.60:
        return base + " DIS5K boundary quality is weak enough to prioritize prediction-time boundary features before COD10K/CAMO."
    if dis5k:
        return base + " If boundary failures remain captured by risk, COD10K/CAMO low-contrast uncertainty stress is the next useful validation."
    return base


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two benchmark runs.")
    parser.add_argument("--run", action="append", default=[], help="NAME:report:oof_summary:weight_ablation; may be repeated")
    parser.add_argument("--left-name")
    parser.add_argument("--left-report")
    parser.add_argument("--left-oof")
    parser.add_argument("--left-ablation")
    parser.add_argument("--right-name")
    parser.add_argument("--right-report")
    parser.add_argument("--right-oof")
    parser.add_argument("--right-ablation")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.run:
            result = compare_benchmark_runs_multi(args.run, args.output)
        else:
            if not args.left_name or not args.left_report or not args.right_name or not args.right_report:
                raise ValueError("provide either repeated --run specs or the legacy left/right arguments")
            result = compare_benchmark_runs(
            left_name=args.left_name,
            left_report=args.left_report,
            left_oof=args.left_oof,
            left_ablation=args.left_ablation,
            right_name=args.right_name,
            right_report=args.right_report,
            right_oof=args.right_oof,
            right_ablation=args.right_ablation,
            output=args.output,
            )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] wrote comparison report: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
