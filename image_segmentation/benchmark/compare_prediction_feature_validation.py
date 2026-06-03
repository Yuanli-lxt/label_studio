from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FEATURES_OF_INTEREST = [
    "pred_hole_count",
    "pred_area_ratio",
    "pred_extent",
    "pred_component_count",
    "pred_boundary_complexity",
    "pred_boundary_density",
    "pred_thinness_proxy",
    "pred_touches_border",
]

ABLATION_PRESETS = [
    "current_full_priority",
    "risk_only",
    "risk_heavy",
    "risk_dominant",
    "no_diversity",
    "balanced_no_risk",
    "boundary_shape_experimental",
    "boundary_shape_rank_score",
    "boundary_shape_calibrated_score",
]

LEARNED_EXPERIMENTS = [
    "current_full_priority",
    "risk_only",
    "risk_heavy",
    "boundary_shape_calibrated_score",
    "learned_boundary_shape_only",
    "learned_current_plus_boundary_shape",
]

GATED_EXPERIMENTS = [
    "current_full_priority",
    "boundary_shape_calibrated_score",
    "boundary_shape_rank_score",
    "gated_boundary_shape_score",
    "gated_current_boundary_score",
]


def compare_prediction_feature_validation(runs: list[str], output: str) -> dict:
    bundles = [_parse_run(spec) for spec in runs]
    recommendation = _recommendation(bundles)
    markdown = render_markdown(bundles, recommendation)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return {"runs": bundles, "recommendation": recommendation, "output": str(path)}


def render_markdown(runs: list[dict], recommendation: str) -> str:
    lines = [
        "# Cross-Dataset Prediction-Feature Boundary/Shape Validation",
        "",
        "## Dataset Summary",
        "| dataset | samples | positives | positive rate | mean IoU | median IoU | Dice | major rate | raw features complete | shadow scores complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        q = run["quality"]
        lines.append(
            f"| {run['name']} | {_fmt(q.get('samples'), 0)} | {_fmt(run['diagnostics'].get('positive_count'), 0)} | "
            f"{_fmt(q.get('major_correction_rate'))} | {_fmt(q.get('mean_iou'))} | {_fmt(q.get('median_iou'))} | "
            f"{_fmt(q.get('dice'))} | {_fmt(q.get('major_correction_rate'))} | {_bool(run['raw_features_complete'])} | {_bool(run['shadow_scores_complete'])} |"
        )
    lines.extend(["", "## Strongest Features"])
    for run in runs:
        lines.append(f"### {run['name']}")
        lines.extend(_feature_lines(run))
    lines.extend(
        [
            "",
            "## GPU Runtime Summary",
            "| dataset | requested | resolved | CPU fallback | CUDA device | sec/sample | peak allocated MB | peak reserved MB |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        runtime = run.get("runtime") or {}
        lines.append(
            f"| {run['name']} | {runtime.get('requested_device') or 'n/a'} | {runtime.get('resolved_device') or 'n/a'} | "
            f"{runtime.get('cpu_fallback')} | {runtime.get('cuda_device_name') or 'n/a'} | {_fmt(runtime.get('seconds_per_sample') or runtime.get('seconds_per_sample_mean'))} | "
            f"{_fmt(runtime.get('peak_cuda_allocated_mb') or runtime.get('peak_cuda_memory_allocated_mb'))} | "
            f"{_fmt(runtime.get('peak_cuda_reserved_mb') or runtime.get('peak_cuda_memory_reserved_mb'))} |"
        )
    lines.extend(
        [
            "",
            "## Feature Direction Consistency",
            "| feature | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for feature in FEATURES_OF_INTEREST:
        lines.append("| " + feature + " | " + " | ".join(_feature_direction(run, feature) for run in runs) + " |")
    lines.extend(
        [
            "",
        "## Calibrated Ablation",
            "| preset | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for preset in ABLATION_PRESETS:
        for metric in ["average_precision_for_major_correction", "roc_auc_for_major_correction", "brier_score_for_major_correction", "lift_at_20_percent_over_random", "precision_at_20_percent"]:
            lines.append("| " + preset + " | " + metric + " | " + " | ".join(_fmt_ci(_ablation_metric(run, preset, metric), _ablation_ci(run, preset, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## Ablation Delta Vs Current",
            "| comparison | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for preset in [name for name in ABLATION_PRESETS if name != "current_full_priority"]:
        comparison = f"{preset}_vs_current_full_priority"
        for metric in ["average_precision_for_major_correction", "lift_at_20_percent_over_random", "precision_at_20_percent"]:
            lines.append("| " + comparison + " | " + metric + " | " + " | ".join(_fmt_delta_ci(_ablation_delta_ci(run, comparison, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## OOF Learned Fusion",
            "| experiment | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for experiment in LEARNED_EXPERIMENTS:
        for metric in ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]:
            lines.append("| " + experiment + " | " + metric + " | " + " | ".join(_fmt_ci(_learned_metric(run, experiment, metric), _learned_ci(run, experiment, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## Learned Delta Vs Current",
            "| comparison | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for experiment in [name for name in LEARNED_EXPERIMENTS if name != "current_full_priority"]:
        comparison = f"{experiment}_vs_current_full_priority"
        for metric in ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]:
            lines.append("| " + comparison + " | " + metric + " | " + " | ".join(_fmt_delta_ci(_learned_delta_ci(run, comparison, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## Gated Fusion OOF",
            "| experiment | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for experiment in GATED_EXPERIMENTS:
        for metric in ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]:
            lines.append("| " + experiment + " | " + metric + " | " + " | ".join(_fmt_ci(_gated_metric(run, experiment, metric), _gated_ci(run, experiment, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## Gated Delta Vs Current",
            "| comparison | metric | " + " | ".join(run["name"] for run in runs) + " |",
            "| --- | --- | " + " | ".join(["---"] * len(runs)) + " |",
        ]
    )
    for experiment in [name for name in GATED_EXPERIMENTS if name != "current_full_priority"]:
        comparison = f"{experiment}_vs_current_full_priority"
        for metric in ["average_precision", "roc_auc", "brier", "lift_at_20", "precision_at_20"]:
            lines.append("| " + comparison + " | " + metric + " | " + " | ".join(_fmt_delta_ci(_gated_delta_ci(run, comparison, metric)) for run in runs) + " |")
    lines.extend(
        [
            "",
            "## Gate Behavior By Dataset",
            "| dataset | min | p25 | median | p75 | max | high-gate n | high-gate major rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        overall = (((run.get("gated") or {}).get("gate_distribution") or {}).get("overall") or {})
        lines.append(
            f"| {run['name']} | {_fmt(overall.get('min'))} | {_fmt(overall.get('p25'))} | {_fmt(overall.get('median'))} | "
            f"{_fmt(overall.get('p75'))} | {_fmt(overall.get('max'))} | {_fmt(overall.get('high_gate_sample_count'), 0)} | {_fmt(overall.get('high_gate_major_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Shadow Monitoring",
            "| dataset | learned boundary non-null | learned boundary top20 overlap | learned boundary AP | artifact validation |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for run in runs:
        monitoring = run.get("shadow_monitoring") or {}
        field = ((monitoring.get("fields") or {}).get("learned_boundary_shape_only_score") or {})
        lines.append(
            f"| {run['name']} | {_fmt(field.get('non_null_count'), 0)} | "
            f"{_fmt((((field.get('topk_overlap') or {}).get('k20') or {}).get('jaccard')))} | "
            f"{_fmt(((field.get('offline_label_metrics') or {}).get('average_precision')))} | "
            f"{_artifact_validation_status(run)} |"
        )
    lines.extend(["", "## Winners"])
    for run in runs:
        lines.append(
            f"- {run['name']}: best AP preset={_best_preset(run, 'average_precision_for_major_correction')}; "
            f"best lift@20 preset={_best_preset(run, 'lift_at_20_percent_over_random')}; "
            f"best OOF learned AP={_best_learned(run)}; best gated AP={_best_gated(run)}"
        )
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "- Prediction-feature diagnostics use labels only offline.",
            "- Production scoring and learned feature sets are restricted to prediction-time fields and existing safe score components.",
            "- GT masks, GT paths, IoU/Dice, boundary deltas, correction deltas, severity labels, major-correction labels, and boundary metadata are not learned-fusion inputs.",
            "",
            "## Recommendation",
            f"- {recommendation}",
            "",
            "## Promotion Criteria Checklist",
            "- OOF: required for learned/gated candidates.",
            "- No leakage: GT masks, GT paths, IoU/Dice, boundary deltas, correction deltas, severity labels, major-correction labels, and boundary metadata remain excluded from scoring features.",
            "- Shadow-only: these scores do not change default review queue ordering.",
            "- Promotion target: production-adjacent shadow or feature-flag only; do not change default Layer 5 weights from this report.",
        ]
    )
    return "\n".join(lines)


def _parse_run(spec: str) -> dict:
    if "=" in spec and ":" not in spec.split("=", 1)[0]:
        return _parse_run_dir(spec)
    parts = spec.split(":", 4)
    if len(parts) != 5:
        raise ValueError("--run must be NAME:evaluation_report.json:prediction_feature_diagnostics.json:weight_ablation.json:learned_boundary_shape_fusion.json")
    name, report_path, diagnostics_path, ablation_path, learned_path = parts
    report = _read_json(report_path)
    diagnostics = _read_json(diagnostics_path)
    ablation = _read_json(ablation_path)
    learned = _read_json(learned_path)
    ablation_ci_path = _sibling_if_exists(ablation_path, "bootstrap_ci.json")
    learned_ci_path = _sibling_if_exists(learned_path, "learned_fusion_bootstrap_ci.json")
    return {
        "name": name,
        "quality": _quality(report),
        "diagnostics": diagnostics,
        "ablation": ablation,
        "learned": learned,
        "runtime": _read_json(Path(report_path).parent / "runtime_metadata.json") if (Path(report_path).parent / "runtime_metadata.json").exists() else {},
        "ablation_ci": _read_json(ablation_ci_path) if ablation_ci_path else None,
        "learned_ci": _read_json(learned_ci_path) if learned_ci_path else None,
        "gated": {},
        "gated_ci": None,
        "raw_features_complete": _raw_features_complete(diagnostics),
        "shadow_scores_complete": _shadow_scores_complete(report_path),
        "paths": {
            "report": report_path,
            "diagnostics": diagnostics_path,
            "ablation": ablation_path,
            "learned": learned_path,
            "ablation_ci": ablation_ci_path,
            "learned_ci": learned_ci_path,
        },
    }


def _parse_run_dir(spec: str) -> dict:
    name, root_text = spec.split("=", 1)
    root = Path(root_text)
    if not name or not root.exists():
        raise ValueError(f"--runs entry must be NAME=existing_output_dir, got {spec!r}")
    report_path = _first_existing(root, ["evaluation_report.json", "benchmark_report.json", "*evaluation_report*.json"])
    diagnostics_path = _first_existing(root, ["prediction_feature_diagnostics/prediction_feature_diagnostics.json", "prediction_feature_diagnostics.json"])
    ablation_path = _first_existing(root, ["ablation/ablation_results.json", "ablation/weight_ablation.json", "ablation_results.json", "weight_ablation.json"])
    learned_path = _first_existing(
        root,
        [
            "learned_fusion/learned_fusion_results.json",
            "learned_fusion/learned_boundary_shape_fusion.json",
            "learned_fusion_results.json",
            "learned_boundary_shape_fusion.json",
        ],
    )
    parsed = _parse_run(":".join([name, str(report_path), str(diagnostics_path), str(ablation_path), str(learned_path)]))
    gated_path = _optional_first_existing(
        root,
        [
            "gated_fusion/gated_fusion_results.json",
            "gated_fusion/gated_boundary_shape_fusion.json",
            "gated_fusion_results.json",
            "gated_boundary_shape_fusion.json",
        ],
    )
    if gated_path:
        parsed["gated"] = _read_json(gated_path)
        gated_ci_path = _sibling_if_exists(gated_path, "gated_fusion_bootstrap_ci.json")
        parsed["gated_ci"] = _read_json(gated_ci_path) if gated_ci_path else None
        parsed["paths"]["gated"] = str(gated_path)
        parsed["paths"]["gated_ci"] = gated_ci_path
    monitoring_path = _optional_first_existing(
        root,
        [
            "shadow_monitoring/shadow_monitoring.json",
            "shadow_monitoring.json",
        ],
    )
    if monitoring_path:
        parsed["shadow_monitoring"] = _read_json(monitoring_path)
        parsed["paths"]["shadow_monitoring"] = str(monitoring_path)
    validation_path = _optional_first_existing(
        root,
        [
            "learned_fusion_artifacts/validation_report.json",
            "validation_report.json",
        ],
    )
    if validation_path:
        parsed["artifact_validation"] = _read_json(validation_path)
        parsed["paths"]["artifact_validation"] = str(validation_path)
    return parsed


def _first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
        matches = sorted(root.glob(candidate))
        if matches:
            return matches[0]
    raise ValueError(f"could not find any of {candidates} under {root}")


def _optional_first_existing(root: Path, candidates: list[str]) -> Path | None:
    try:
        return _first_existing(root, candidates)
    except ValueError:
        return None


def _sibling_if_exists(path: str | Path, filename: str) -> str | None:
    candidate = Path(path).parent / filename
    return str(candidate) if candidate.exists() else None


def _quality(report: dict) -> dict:
    overall = report.get("overall_quality") or {}
    return {
        "samples": overall.get("n_samples"),
        "mean_iou": overall.get("mean_model_gt_iou"),
        "median_iou": overall.get("median_model_gt_iou"),
        "dice": overall.get("mean_dice"),
        "precision": overall.get("mean_precision"),
        "recall": overall.get("mean_recall"),
        "major_correction_count": overall.get("major_correction_count"),
        "major_correction_rate": overall.get("major_correction_rate"),
        "severity": overall.get("correction_severity_distribution"),
    }


def _raw_features_complete(diagnostics: dict) -> bool:
    features = diagnostics.get("features") or {}
    for name in FEATURES_OF_INTEREST:
        row = features.get(name) or {}
        if row.get("missing_count") not in (0, 0.0):
            return False
        if row.get("constant") is True:
            return False
    return True


def _feature_lines(run: dict) -> list[str]:
    features = run["diagnostics"].get("features") or {}
    ranked = sorted(
        [
            (
                row.get("average_precision") or 0.0,
                row.get("roc_auc") or 0.0,
                name,
                row.get("direction_suggestion"),
            )
            for name, row in features.items()
            if name in FEATURES_OF_INTEREST
        ],
        reverse=True,
    )
    return [
        f"- {name}: AP={_fmt(ap)}, ROC-AUC={_fmt(roc)}, direction={direction}"
        for ap, roc, name, direction in ranked[:5]
    ] or ["- none"]


def _feature_direction(run: dict, feature: str) -> str:
    row = ((run.get("diagnostics") or {}).get("features") or {}).get(feature) or {}
    return f"{row.get('direction_suggestion') or 'n/a'} / AP {_fmt(row.get('average_precision'))}"


def _ablation_metric(run: dict, preset: str, metric: str) -> Any:
    return (((run.get("ablation") or {}).get("presets") or {}).get(preset) or {}).get(metric)


def _ablation_ci(run: dict, preset: str, metric: str) -> dict | None:
    return ((((run.get("ablation_ci") or {}).get("presets") or {}).get(preset) or {}).get(metric))


def _ablation_delta_ci(run: dict, comparison: str, metric: str) -> dict | None:
    return ((((run.get("ablation_ci") or {}).get("pairwise_delta_vs_current") or {}).get(comparison) or {}).get(metric))


def _learned_metric(run: dict, experiment: str, metric: str) -> Any:
    return (((run.get("learned") or {}).get("experiments") or {}).get(experiment) or {}).get(metric)


def _learned_ci(run: dict, experiment: str, metric: str) -> dict | None:
    return ((((run.get("learned_ci") or {}).get("experiments") or {}).get(experiment) or {}).get(metric))


def _learned_delta_ci(run: dict, comparison: str, metric: str) -> dict | None:
    return ((((run.get("learned_ci") or {}).get("pairwise_delta_vs_current") or {}).get(comparison) or {}).get(metric))


def _gated_metric(run: dict, experiment: str, metric: str) -> Any:
    return (((run.get("gated") or {}).get("experiments") or {}).get(experiment) or {}).get(metric)


def _gated_ci(run: dict, experiment: str, metric: str) -> dict | None:
    return ((((run.get("gated_ci") or {}).get("experiments") or {}).get(experiment) or {}).get(metric))


def _gated_delta_ci(run: dict, comparison: str, metric: str) -> dict | None:
    return ((((run.get("gated_ci") or {}).get("pairwise_delta_vs_current") or {}).get(comparison) or {}).get(metric))


def _best_preset(run: dict, metric: str) -> str:
    presets = (run.get("ablation") or {}).get("presets") or {}
    if not presets:
        return "n/a"
    return max(presets, key=lambda name: _num((presets.get(name) or {}).get(metric)))


def _best_learned(run: dict) -> str:
    rows = {
        name: row
        for name, row in ((run.get("learned") or {}).get("experiments") or {}).items()
        if (row or {}).get("oof") is True
    }
    if not rows:
        return "n/a"
    return max(rows, key=lambda name: _num((rows.get(name) or {}).get("average_precision")))


def _best_gated(run: dict) -> str:
    rows = (run.get("gated") or {}).get("experiments") or {}
    if not rows:
        return "n/a"
    return max(rows, key=lambda name: _num((rows.get(name) or {}).get("average_precision")))


def _artifact_validation_status(run: dict) -> str:
    report = run.get("artifact_validation") or {}
    if not report:
        return "n/a"
    return "passed" if report.get("passed") is True else "failed"


def _shadow_scores_complete(report_path: str | Path) -> bool:
    queue = Path(report_path).parent / "review_queue.jsonl"
    if not queue.exists():
        return False
    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return False
    required = [
        "boundary_shape_calibrated_score",
        "boundary_shape_rank_score",
        "gated_boundary_shape_score",
        "gated_current_boundary_score",
    ]
    for row in rows:
        shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
        if any(shadow.get(name) is None for name in required):
            return False
    return True


def _recommendation(runs: list[dict]) -> str:
    complete = all(run.get("raw_features_complete") for run in runs)
    calibrated_non_degrade = all(
        _num(_ablation_metric(run, "boundary_shape_calibrated_score", "average_precision_for_major_correction"))
        + 0.01
        >= _num(_ablation_metric(run, "current_full_priority", "average_precision_for_major_correction"))
        for run in runs
    )
    learned_boundary_wins = all(_best_learned(run) == "learned_boundary_shape_only" for run in runs)
    if complete and learned_boundary_wins and calibrated_non_degrade:
        return "Boundary/shape learned fusion is a strong experimental candidate, but keep Layer 5 defaults unchanged until a full rerun with bootstrap confidence intervals and another stress dataset."
    if complete and calibrated_non_degrade:
        return "Keep default Layer 5 weights unchanged; retain boundary_shape_calibrated_score for experiments and continue OOF validation."
    return "Keep default Layer 5 weights unchanged; boundary/shape behavior is not yet stable enough across datasets."


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if digits == 0:
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _fmt_ci(value: Any, ci: dict | None) -> str:
    if not ci:
        return _fmt(value)
    ci_value = ci.get("value", value)
    return f"{_fmt(ci_value)} [{_fmt(ci.get('ci95_low'))}, {_fmt(ci.get('ci95_high'))}]"


def _fmt_delta_ci(ci: dict | None) -> str:
    if not ci:
        return "n/a"
    return f"{_fmt(ci.get('value'))} [{_fmt(ci.get('ci95_low'))}, {_fmt(ci.get('ci95_high'))}]"


def _bool(value: Any) -> str:
    return "yes" if value else "no"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare cross-dataset prediction-time boundary/shape validation.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="NAME:evaluation_report.json:prediction_feature_diagnostics.json:weight_ablation.json:learned_boundary_shape_fusion.json",
    )
    parser.add_argument(
        "--runs",
        action="append",
        default=[],
        help="NAME=benchmark_output_dir; may be repeated",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = list(args.run or []) + list(args.runs or [])
    if not runs:
        raise SystemExit("--run or --runs is required")
    result = compare_prediction_feature_validation(runs, args.output)
    print(f"[OK] wrote prediction-feature comparison for {len(result['runs'])} runs to {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
