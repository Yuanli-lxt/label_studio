from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.apply_shadow_scores import apply_shadow_scores
from image_segmentation.benchmark.compare_shadow_scores import compare_shadow_scores
from image_segmentation.benchmark.export_shadow_review_packet import export_shadow_review_packet
from image_segmentation.benchmark.shadow_scoring import shadow_scoring_runtime_config


DEFAULT_REPORT_PATH = "demo_data/model_state/image_segmentation/benchmark/production_shadow_live_window_001_report.md"
DEFAULT_DECISION_REPORT_PATH = "demo_data/model_state/image_segmentation/benchmark/production_shadow_expanded_rollout_decision_report.md"
P95_DRIFT_OUTLIER_THRESHOLD = 1.0
SAFE_PACKET_FILES = {
    "high_learned_low_current": "high_learned_low_current.jsonl",
    "high_current_low_learned": "high_current_low_learned.jsonl",
    "top_learned": "top_learned.jsonl",
    "top_current": "top_current.jsonl",
}
HUMAN_REVIEW_TASK_FILES = {
    "high_learned_low_current": "review_tasks_high_learned_low_current.jsonl",
    "high_current_low_learned": "review_tasks_high_current_low_learned.jsonl",
    "top_learned": "review_tasks_top_learned.jsonl",
    "control_current_top": "review_tasks_control_current_top.jsonl",
}


def run_live_shadow_rollout(
    review_queue: str,
    output_dir: str,
    artifact_dir: str | None = None,
    window_id: str = "live_shadow_window_001",
    baseline: str | None = None,
    report_path: str = DEFAULT_REPORT_PATH,
    top_k: int = 50,
) -> dict:
    """Run the first-window shadow-only live rollout workflow."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    live_queue = output / "live_review_queue.shadow_scored.jsonl"

    replay_summary = apply_shadow_scores(
        review_queue,
        str(live_queue),
        learned_artifact_dir=artifact_dir,
        enable_learned_shadow_scores=True,
        enable_gated_shadow_scores=False,
    )
    monitoring = compare_shadow_scores(
        str(live_queue),
        str(output / "shadow_monitoring"),
        window_id=window_id,
        baseline=baseline,
        production_mode=True,
        no_labels=True,
        performance_context=replay_summary,
    )
    packet_started = time.perf_counter()
    packet = export_shadow_review_packet(
        str(live_queue),
        str(output / "shadow_review_packet"),
        top_k=top_k,
        production_mode=True,
    )
    monitoring.setdefault("performance", {})["review_packet_export_latency_ms"] = (time.perf_counter() - packet_started) * 1000.0
    _write_json(output / "shadow_monitoring" / "shadow_monitoring.json", monitoring)

    report = _rollout_report(
        review_queue=review_queue,
        live_queue=str(live_queue),
        output_dir=str(output),
        artifact_dir=artifact_dir,
        window_id=window_id,
        baseline=baseline,
        replay_summary=replay_summary,
        monitoring=monitoring,
        packet=packet,
    )
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_live_rollout_markdown(report), encoding="utf-8")
    _write_json(output / "live_shadow_rollout_summary.json", report)
    return report


def run_multi_window_shadow_observation(
    input_queues: list[str] | None,
    output_root: str,
    artifact_dir: str | None = None,
    input_dir: str | None = None,
    window_glob: str = "review_queue_*.jsonl",
    baseline: str | None = None,
    max_windows: int | None = None,
    top_k: int = 50,
    decision_report_path: str = DEFAULT_DECISION_REPORT_PATH,
) -> dict:
    """Run ordered shadow-only windows and summarize cross-window stability."""
    queues = _resolve_input_queues(input_queues, input_dir, window_glob)
    if max_windows is not None:
        queues = queues[: max(0, int(max_windows))]
    if not queues:
        raise ValueError("no input queues found for multi-window shadow observation")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    window_reports = []
    current_baseline = baseline
    for index, queue in enumerate(queues, start=1):
        window_name = f"window_{index:03d}"
        window_id = f"live_shadow_window_{index:03d}"
        window_dir = root / window_name
        report_path = window_dir / f"production_shadow_live_window_{index:03d}_report.md"
        report = run_live_shadow_rollout(
            queue,
            str(window_dir),
            artifact_dir=artifact_dir,
            window_id=window_id,
            baseline=current_baseline,
            report_path=str(report_path),
            top_k=top_k,
        )
        report["window_name"] = window_name
        report["window_report_path"] = str(report_path)
        window_reports.append(report)
        current_baseline = str(window_dir / "shadow_monitoring" / "shadow_monitoring.json")

    human_summary = build_human_disagreement_review_summary(
        window_reports,
        str(root / "human_disagreement_review_summary"),
    )
    human_task_summary = build_human_review_task_packet(
        human_summary,
        str(root / "human_review_task_packet"),
    )
    summary = build_multi_window_summary(
        window_reports,
        output_root=str(root),
        human_summary=human_summary,
        human_task_summary=human_task_summary,
    )
    decision_path = Path(decision_report_path)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(render_decision_report_markdown(summary), encoding="utf-8")
    summary["decision_report_path"] = str(decision_path)
    _write_json(root / "multi_window_shadow_summary.json", summary)
    _write_json(root / "expanded_shadow_multi_window_summary.json", summary)
    _write_json(root / "broader_shadow_multi_window_summary.json", summary)
    (root / "multi_window_shadow_summary.md").write_text(
        render_multi_window_summary_markdown(summary),
        encoding="utf-8",
    )
    (root / "expanded_shadow_multi_window_summary.md").write_text(
        render_multi_window_summary_markdown(summary),
        encoding="utf-8",
    )
    (root / "broader_shadow_multi_window_summary.md").write_text(
        render_multi_window_summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def build_multi_window_summary(
    window_reports: list[dict],
    output_root: str,
    human_summary: dict | None = None,
    human_task_summary: dict | None = None,
) -> dict:
    windows = [_window_summary(row) for row in window_reports]
    alerts = _aggregate_alerts(windows)
    decision = _decision(windows, alerts)
    return {
        "output_root": output_root,
        "windows_processed": len(windows),
        "artifact": _artifact_summary(window_reports),
        "windows": windows,
        "coverage_stability": {
            "rows_per_window": [row["rows_processed"] for row in windows],
            "learned_coverage_per_window": [row["learned_score_non_null_count"] for row in windows],
            "null_rate_per_window": [row["learned_score_null_rate"] for row in windows],
            "missing_feature_count_p95_per_window": [row["missing_safe_feature_count_p95"] for row in windows],
            "artifact_status_per_window": [row["artifact_validation_status_counts"] for row in windows],
            "inference_error_rate_per_window": [row["inference_error_rate"] for row in windows],
        },
        "distribution_stability": {
            "learned_score_distribution_per_window": [row["score_distribution"] for row in windows],
            "p95_drift_vs_baseline_per_window": [row["p95_drift_vs_baseline"] for row in windows],
            "median_drift_vs_baseline_per_window": [row["median_drift_vs_baseline"] for row in windows],
            "score_distribution_psi_per_window": [row["score_distribution_psi"] for row in windows],
            "simple_distribution_drift_vs_first_window": _distribution_drift_series(window_reports),
            "null_rate_drift_vs_baseline_per_window": [row["null_rate_drift_vs_baseline"] for row in windows],
            "coverage_drift_vs_baseline_per_window": [row["coverage_drift_vs_baseline"] for row in windows],
            "latency_p95_drift_vs_baseline_ms_per_window": [row["latency_p95_drift_vs_baseline_ms"] for row in windows],
            "score_distribution_alert_per_window": [
                row["alert_checks"].get("score_distribution_p95_shift_std") for row in windows
            ],
            "p95_drift_outliers": _drift_outliers(windows),
            "p95_drift_outlier_threshold": P95_DRIFT_OUTLIER_THRESHOLD,
        },
        "agreement_stability": {
            "topk_jaccard_per_window": [row["topk_jaccard"] for row in windows],
            "top100_jaccard_change_vs_baseline_per_window": [
                row["top100_jaccard_relative_change"] for row in windows
            ],
            "spearman_per_window": [row["spearman"] for row in windows],
            "pearson_per_window": [row["pearson"] for row in windows],
            "rank_delta_per_window": [row["rank_delta_summary"] for row in windows],
            "high_disagreement_count_per_window": [row["high_disagreement_count"] for row in windows],
            "high_learned_low_current_count_per_window": [row["high_learned_low_current_count"] for row in windows],
            "high_current_low_learned_count_per_window": [row["high_current_low_learned_count"] for row in windows],
        },
        "performance_stability": [row["performance"] for row in windows],
        "safety_stability": {
            "shadow_only_true_count": sum(row["shadow_only"] is True for row in windows),
            "affects_default_ranking_false_count": sum(row["affects_default_ranking"] is False for row in windows),
            "any_default_field_equality_failure": any(row["default_field_equality"] is not True for row in windows),
            "any_ordering_equality_failure": any(row["ordering_equality"] is not True for row in windows),
            "production_labels_read_any": any(row["production_labels_read"] is True for row in windows),
            "leakage_guard_status_per_window": [row["leakage_guard_status"] for row in windows],
        },
        "alerts": alerts,
        "human_disagreement_review_summary": human_summary,
        "human_review_task_packet": human_task_summary,
        "rollback_readiness": {
            "disable_learned_shadow": "SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=false",
            "disable_all_shadow": "SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=false",
            "fail_open": True,
        },
        "decision": decision,
        "promotion_status": {
            "ready_for_default_sorting": False,
            "ready_for_weight_promotion": False,
            "note": "Default promotion requires future label-backed or human-feedback-backed evidence and is outside this shadow-only observation stage.",
        },
    }


def build_human_disagreement_review_summary(window_reports: list[dict], output_dir: str) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[dict]] = {name: [] for name in SAFE_PACKET_FILES}
    for report in window_reports:
        window_id = report.get("window_id")
        packet_dir = Path(str(report.get("review_packet_path")))
        for group, filename in SAFE_PACKET_FILES.items():
            for row in _read_jsonl(packet_dir / filename):
                safe = _safe_review_row(row)
                safe["window_ids"] = [window_id]
                safe["human_review_status"] = None
                safe["human_review_outcome"] = None
                safe["human_review_notes"] = None
                combined[group].append(safe)
    outputs = {}
    counts = {}
    for group, rows in combined.items():
        merged = _sort_review_rows(group, _dedupe_review_rows(rows))
        path = output / f"combined_{SAFE_PACKET_FILES[group]}"
        _write_jsonl(path, merged)
        outputs[group] = str(path)
        counts[group] = len(merged)
    summary = {
        "output_dir": str(output),
        "safe_fields_only": True,
        "dedupe_key": "sample_id|image_id|annotation_id|task_id",
        "outputs": outputs,
        "counts": counts,
        "sampling_recommendation": {
            "top_high_learned_low_current": outputs["high_learned_low_current"],
            "top_high_current_low_learned": outputs["high_current_low_learned"],
            "score_distribution_boundary_samples": outputs["top_learned"],
            "artifact_metadata_abnormal_samples": "inspect rows with shadow_metadata_status != valid in combined packets",
        },
    }
    _write_json(output / "disagreement_review_summary.json", summary)
    (output / "disagreement_review_summary.md").write_text(
        render_human_disagreement_markdown(summary),
        encoding="utf-8",
    )
    return summary


def build_human_review_task_packet(
    human_summary: dict | None,
    output_dir: str,
    high_learned_low_current_limit: int = 100,
    other_limit: int = 50,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = (human_summary or {}).get("outputs") or {}
    sources = {
        "high_learned_low_current": inputs.get("high_learned_low_current"),
        "high_current_low_learned": inputs.get("high_current_low_learned"),
        "top_learned": inputs.get("top_learned"),
        "control_current_top": inputs.get("top_current"),
    }
    limits = {
        "high_learned_low_current": int(high_learned_low_current_limit),
        "high_current_low_learned": int(other_limit),
        "top_learned": int(other_limit),
        "control_current_top": int(other_limit),
    }
    outputs = {}
    counts = {}
    for group, source in sources.items():
        rows = _read_jsonl(Path(source)) if source else []
        tasks = [_human_review_task_row(row, group) for row in rows[: limits[group]]]
        path = output / HUMAN_REVIEW_TASK_FILES[group]
        _write_jsonl(path, tasks)
        outputs[group] = str(path)
        counts[group] = len(tasks)
    summary = {
        "output_dir": str(output),
        "safe_fields_only": True,
        "source_human_disagreement_summary": (human_summary or {}).get("output_dir"),
        "outputs": outputs,
        "counts": counts,
        "review_outcome_values": [
            "major_correction_needed",
            "minor_correction_needed",
            "ok",
            "unclear",
        ],
    }
    _write_json(output / "human_review_task_summary.json", summary)
    (output / "human_review_task_summary.md").write_text(
        render_human_review_task_markdown(summary),
        encoding="utf-8",
    )
    return summary


def render_human_review_task_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Human Review Task Packet",
            "",
            f"- output_dir: {summary.get('output_dir')}",
            f"- safe_fields_only: {summary.get('safe_fields_only')}",
            f"- counts: {summary.get('counts')}",
            "",
            "## Outputs",
            *[f"- {group}: {path}" for group, path in sorted((summary.get("outputs") or {}).items())],
            "",
            "## Review Outcomes",
            *[f"- {value}" for value in summary.get("review_outcome_values") or []],
        ]
    ) + "\n"


def render_live_rollout_markdown(report: dict) -> str:
    runtime = report.get("rollout_config") or {}
    queue = report.get("queue_batch_summary") or {}
    coverage = report.get("shadow_coverage") or {}
    monitoring = report.get("monitoring") or {}
    safety = report.get("leakage_guard_summary") or {}
    recommendation = report.get("recommendation") or {}
    topk = report.get("topk_jaccard") or {}
    rank_delta = report.get("rank_delta_summary") or {}
    lines = [
        "# Production Shadow Live Window 001",
        "",
        "## Rollout Config",
        f"- enable_boundary_shape_shadow: {runtime.get('enable_boundary_shape_shadow')}",
        f"- enable_learned_boundary_shape_shadow: {runtime.get('enable_learned_boundary_shape_shadow')}",
        f"- fail_open: {runtime.get('fail_open')}",
        f"- shadow_only: {runtime.get('shadow_only')}",
        f"- affects_default_ranking: {runtime.get('affects_default_ranking')}",
        "",
        "## Artifact",
        f"- artifact_path: {runtime.get('artifact_path')}",
        f"- artifact_registry: {runtime.get('artifact_registry')}",
        f"- artifact_configured_version: {runtime.get('artifact_configured_version')}",
        f"- artifact_version: {runtime.get('artifact_version')}",
        f"- artifact_load_status: {runtime.get('artifact_load_status')}",
        f"- artifact_validation_status: {runtime.get('artifact_validation_status')}",
        "",
        "## Queue Batch Summary",
        f"- input_review_queue: {report.get('input_review_queue')}",
        f"- live_queue_output_path: {report.get('live_queue_output_path')}",
        f"- total_rows: {queue.get('total_rows')}",
        f"- default_field_equality: {queue.get('default_field_equality_check')}",
        f"- ordering_equality: {queue.get('ordering_equality_check')}",
        "",
        "## Shadow Coverage",
        f"- shadow_score_present_count: {coverage.get('shadow_scores_present_count')}",
        f"- learned_score_non_null_count: {coverage.get('learned_score_non_null_count')}",
        f"- learned_score_null_rate: {_fmt(coverage.get('learned_score_null_rate'))}",
        "",
        "## Null And Error Rate",
        f"- inference_error_count: {coverage.get('inference_error_count')}",
        f"- artifact_validation_status_counts: {coverage.get('artifact_validation_status_counts')}",
        f"- missing_feature_count_distribution: {coverage.get('missing_feature_count_distribution')}",
        "",
        "## Score Distribution",
        f"- learned_boundary_shape_only_score: {report.get('score_distribution')}",
        "",
        "## Drift Vs Baseline",
        f"- baseline_available: {((monitoring.get('distribution_drift') or {}).get('baseline_available'))}",
        f"- learned_boundary_shape_only_score: {((monitoring.get('distribution_drift') or {}).get('learned_boundary_shape_only_score'))}",
        "",
        "## Top-K Overlap",
        f"- k20: {topk.get('k20')}",
        f"- k50: {topk.get('k50')}",
        f"- k100: {topk.get('k100')}",
        "",
        "## Rank Delta Summary",
        f"- learned_boundary_shape_only_score: {rank_delta}",
        "",
        "## Alert Status",
        f"- triggered: {((monitoring.get('alerts') or {}).get('triggered'))}",
        f"- checks: {((monitoring.get('alerts') or {}).get('checks'))}",
        "",
        "## Review Packet",
        f"- path: {report.get('review_packet_path')}",
        "",
        "## Latency Summary",
        f"- performance: {monitoring.get('performance')}",
        "",
        "## Leakage Guard Summary",
        f"- production_labels_read: {safety.get('production_labels_read')}",
        f"- label_metrics_available: {safety.get('label_metrics_available')}",
        f"- evaluation_only_metrics_in_review_packet: {safety.get('evaluation_only_metrics_in_review_packet')}",
        f"- leakage_guard_status: {safety.get('leakage_guard_status')}",
        "",
        "## Rollback Readiness",
        f"- feature_flag_can_disable_learned_shadow: {report.get('rollback_readiness', {}).get('feature_flag_can_disable_learned_shadow')}",
        f"- feature_flag_can_disable_all_shadow: {report.get('rollback_readiness', {}).get('feature_flag_can_disable_all_shadow')}",
        "",
        "## Recommendation",
        f"- continue_shadow_rollout: {recommendation.get('continue_shadow_rollout')}",
        f"- pause_or_rollback: {recommendation.get('pause_or_rollback')}",
        f"- default_layer_5_weights_unchanged: {recommendation.get('default_layer_5_weights_unchanged')}",
        f"- default_sorting_unchanged: {recommendation.get('default_sorting_unchanged')}",
        f"- ready_for_default_promotion: {recommendation.get('ready_for_default_promotion')}",
        f"- note: {recommendation.get('note')}",
    ]
    return "\n".join(lines) + "\n"


def render_multi_window_summary_markdown(summary: dict) -> str:
    lines = [
        "# Multi-Window Shadow Summary",
        "",
        f"- output_root: {summary.get('output_root')}",
        f"- windows_processed: {summary.get('windows_processed')}",
        f"- aggregate_alert_status: {((summary.get('alerts') or {}).get('aggregate_alert_status'))}",
        f"- recommended_action: {((summary.get('decision') or {}).get('recommended_action'))}",
        "",
        "## Windows",
        "| window | rows | coverage | null rate | artifact status | inference error rate | labels read | alerts | top100 jaccard | default fields | ordering |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in summary.get("windows") or []:
        topk = row.get("topk_jaccard") or {}
        lines.append(
            f"| {row.get('window_id')} | {row.get('rows_processed')} | {row.get('learned_score_non_null_count')} | "
            f"{_fmt(row.get('learned_score_null_rate'))} | {row.get('artifact_validation_status_counts')} | "
            f"{_fmt(row.get('inference_error_rate'))} | {row.get('production_labels_read')} | "
            f"{row.get('alert_status')} | {_fmt(topk.get('k100'))} | {row.get('default_field_equality')} | "
            f"{row.get('ordering_equality')} |"
        )
    lines.extend(
        [
            "",
            "## Alert Summary",
            f"- hard_safety_alert: {((summary.get('alerts') or {}).get('hard_safety_alert'))}",
            f"- reasons: {((summary.get('alerts') or {}).get('alert_reasons'))}",
            "",
            "## Human Disagreement Review",
            f"- path: {((summary.get('human_disagreement_review_summary') or {}).get('output_dir'))}",
            f"- counts: {((summary.get('human_disagreement_review_summary') or {}).get('counts'))}",
            "",
            "## Human Review Task Packet",
            f"- path: {((summary.get('human_review_task_packet') or {}).get('output_dir'))}",
            f"- counts: {((summary.get('human_review_task_packet') or {}).get('counts'))}",
            "",
            "## Promotion Status",
            f"- ready_for_default_sorting: {((summary.get('promotion_status') or {}).get('ready_for_default_sorting'))}",
            f"- ready_for_weight_promotion: {((summary.get('promotion_status') or {}).get('ready_for_weight_promotion'))}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_decision_report_markdown(summary: dict) -> str:
    decision = summary.get("decision") or {}
    artifact = summary.get("artifact") or {}
    alerts = summary.get("alerts") or {}
    return "\n".join(
        [
            "# Production Shadow Multi-Window Decision Report",
            "",
            "## Scope",
            "Shadow-only observation for learned_boundary_shape_only_score. Default Layer 5 weights and review queue sorting remain unchanged.",
            "",
            "## Feature Flag Config",
            "- SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=true",
            "- SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=true",
            "- SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN=true",
            "",
            "## Artifact Version / Path",
            f"- artifact_path: {artifact.get('artifact_path')}",
            f"- artifact_version: {artifact.get('artifact_version')}",
            "",
            "## Windows Processed",
            f"- windows_processed: {summary.get('windows_processed')}",
            "",
            "## Coverage Stability",
            f"- {summary.get('coverage_stability')}",
            "",
            "## Null / Error Stability",
            f"- inference_error_rate_per_window: {((summary.get('coverage_stability') or {}).get('inference_error_rate_per_window'))}",
            "",
            "## Distribution Drift",
            f"- {summary.get('distribution_stability')}",
            "",
            "## Agreement / Disagreement Stability",
            f"- {summary.get('agreement_stability')}",
            "",
            "## Latency Summary",
            f"- {summary.get('performance_stability')}",
            "",
            "## Alert Summary",
            f"- aggregate_alert_status: {alerts.get('aggregate_alert_status')}",
            f"- hard_safety_alert: {alerts.get('hard_safety_alert')}",
            f"- alert_reasons: {alerts.get('alert_reasons')}",
            "",
            "## Human Review Task Packet Path",
            f"- {((summary.get('human_review_task_packet') or {}).get('output_dir'))}",
            "",
            "## Human Disagreement Review Packet Path",
            f"- {((summary.get('human_disagreement_review_summary') or {}).get('output_dir'))}",
            "",
            "## Optional Human Feedback Summary",
            "- not available; offline feedback import CLI is available via summarize_shadow_human_feedback.",
            "",
            "## Leakage Guard Summary",
            f"- {summary.get('safety_stability')}",
            "",
            "## Rollback Readiness",
            f"- {summary.get('rollback_readiness')}",
            "",
            "## Decision",
            f"- recommended_action: {decision.get('recommended_action')}",
            f"- continue_same_traffic: {decision.get('continue_same_traffic')}",
            f"- expand_shadow_traffic: {decision.get('expand_shadow_traffic')}",
            f"- hold_for_more_observation: {decision.get('hold_for_more_observation')}",
            f"- hold_for_human_feedback: {decision.get('hold_for_human_feedback')}",
            f"- pause: {decision.get('pause')}",
            f"- rollback: {decision.get('rollback')}",
            f"- reason: {decision.get('reason')}",
            "",
            "## Promotion Status",
            "- not ready for default sorting",
            "- not ready for weight promotion",
        ]
    ) + "\n"


def render_human_disagreement_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Human Disagreement Review Summary",
            "",
            f"- output_dir: {summary.get('output_dir')}",
            f"- safe_fields_only: {summary.get('safe_fields_only')}",
            f"- dedupe_key: {summary.get('dedupe_key')}",
            f"- counts: {summary.get('counts')}",
            "",
            "## Sampling Recommendation",
            *[
                f"- {key}: {value}"
                for key, value in sorted((summary.get("sampling_recommendation") or {}).items())
            ],
        ]
    ) + "\n"


def _rollout_report(
    review_queue: str,
    live_queue: str,
    output_dir: str,
    artifact_dir: str | None,
    window_id: str,
    baseline: str | None,
    replay_summary: dict,
    monitoring: dict,
    packet: dict,
) -> dict:
    coverage = monitoring.get("coverage") or {}
    field = (monitoring.get("fields") or {}).get("learned_boundary_shape_only_score") or {}
    alerts = monitoring.get("alerts") or {}
    default_ok = replay_summary.get("default_field_equality_check") is True
    ordering_ok = replay_summary.get("ordering_equality_check") is True
    labels_read = monitoring.get("labels_read") is True
    alerts_triggered = alerts.get("triggered") is True
    continue_shadow = default_ok and ordering_ok and not labels_read and not alerts_triggered
    return {
        "window_id": window_id,
        "input_review_queue": review_queue,
        "live_queue_output_path": live_queue,
        "monitoring_output_path": str(Path(output_dir) / "shadow_monitoring"),
        "review_packet_path": str(Path(output_dir) / "shadow_review_packet"),
        "baseline": baseline,
        "rollout_config": shadow_scoring_runtime_config(
            learned_artifact_dir=artifact_dir,
            enable_shadow_scoring=True,
            enable_learned_shadow_scores=True,
            enable_gated_shadow_scores=False,
        ),
        "queue_batch_summary": {
            "total_rows": replay_summary.get("total_rows"),
            "shadow_rows_written": replay_summary.get("shadow_rows_written"),
            "default_field_equality_check": default_ok,
            "ordering_equality_check": ordering_ok,
            "default_layer_5_weights_unchanged": True,
            "default_sorting_unchanged": ordering_ok,
        },
        "shadow_coverage": coverage,
        "artifact_validation_status": coverage.get("artifact_validation_status_counts"),
        "score_distribution": field.get("distribution"),
        "topk_jaccard": _jaccard_only(field.get("topk_overlap") or {}),
        "rank_delta_summary": field.get("rank_delta_summary"),
        "monitoring": monitoring,
        "review_packet": packet,
        "latency_summary": monitoring.get("performance"),
        "leakage_guard_summary": {
            "production_labels_read": monitoring.get("labels_read"),
            "label_metrics_available": monitoring.get("label_metrics_available"),
            "evaluation_only_metrics_in_review_packet": packet.get("evaluation_only_metrics_included"),
            "leakage_guard_status": (monitoring.get("safety_summary") or {}).get("leakage_guard_status"),
            "production_mode": monitoring.get("production_mode"),
        },
        "rollback_readiness": {
            "feature_flag_can_disable_learned_shadow": "SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=false",
            "feature_flag_can_disable_all_shadow": "SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=false",
            "fail_open": True,
        },
        "recommendation": {
            "continue_shadow_rollout": continue_shadow,
            "pause_or_rollback": not continue_shadow,
            "default_layer_5_weights_unchanged": True,
            "default_sorting_unchanged": ordering_ok,
            "ready_for_default_promotion": False,
            "note": "Continue small-window shadow-only observation if no alerts are triggered; do not promote to default sorting or Layer 5 weights.",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _window_summary(report: dict) -> dict:
    monitoring = report.get("monitoring") or {}
    coverage = report.get("shadow_coverage") or {}
    field = (monitoring.get("fields") or {}).get("learned_boundary_shape_only_score") or {}
    safety = monitoring.get("safety_summary") or {}
    checks = (monitoring.get("alerts") or {}).get("checks") or {}
    drift = ((monitoring.get("distribution_drift") or {}).get("learned_boundary_shape_only_score") or {})
    rows = int(coverage.get("total_rows") or 0)
    inference_errors = int(coverage.get("inference_error_count") or 0)
    disagreement = monitoring.get("disagreement_volume") or {}
    return {
        "window_id": report.get("window_id"),
        "window_name": report.get("window_name"),
        "input_queue_path": report.get("input_review_queue"),
        "output_queue_path": report.get("live_queue_output_path"),
        "monitoring_output_path": report.get("monitoring_output_path"),
        "review_packet_path": report.get("review_packet_path"),
        "window_report_path": report.get("window_report_path"),
        "artifact_path": (report.get("rollout_config") or {}).get("artifact_path"),
        "artifact_version": (report.get("rollout_config") or {}).get("artifact_version"),
        "rows_processed": rows,
        "learned_score_non_null_count": coverage.get("learned_score_non_null_count"),
        "learned_score_null_rate": coverage.get("learned_score_null_rate"),
        "artifact_validation_status_counts": coverage.get("artifact_validation_status_counts"),
        "inference_error_count": inference_errors,
        "inference_error_rate": float(inference_errors / rows) if rows else 0.0,
        "missing_safe_feature_count_p95": (checks.get("missing_safe_feature_count_p95") or {}).get("value"),
        "default_field_equality": (report.get("queue_batch_summary") or {}).get("default_field_equality_check"),
        "ordering_equality": (report.get("queue_batch_summary") or {}).get("ordering_equality_check"),
        "production_labels_read": monitoring.get("labels_read"),
        "alert_status": (monitoring.get("alerts") or {}).get("triggered"),
        "alert_checks": checks,
        "shadow_only": safety.get("shadow_only"),
        "affects_default_ranking": safety.get("affects_default_ranking"),
        "leakage_guard_status": safety.get("leakage_guard_status"),
        "topk_jaccard": report.get("topk_jaccard"),
        "score_distribution": report.get("score_distribution"),
        "p95_drift_vs_baseline": drift.get("p95_shift_std"),
        "median_drift_vs_baseline": drift.get("median_delta"),
        "score_distribution_psi": drift.get("score_distribution_psi"),
        "null_rate_drift_vs_baseline": drift.get("null_rate_delta"),
        "coverage_drift_vs_baseline": drift.get("coverage_delta"),
        "latency_p95_drift_vs_baseline_ms": drift.get("latency_p95_delta_ms"),
        "top100_jaccard_relative_change": drift.get("top100_jaccard_relative_change"),
        "spearman": (field.get("correlation") or {}).get("spearman"),
        "pearson": (field.get("correlation") or {}).get("pearson"),
        "rank_delta_summary": field.get("rank_delta_summary"),
        "high_learned_low_current_count": disagreement.get("high_learned_low_current_count") or 0,
        "high_current_low_learned_count": disagreement.get("high_current_low_learned_count") or 0,
        "high_disagreement_count": (disagreement.get("high_learned_low_current_count") or 0)
        + (disagreement.get("high_current_low_learned_count") or 0),
        "performance": monitoring.get("performance"),
    }


def _aggregate_alerts(windows: list[dict]) -> dict:
    reasons = []
    hard = False
    for row in windows:
        prefix = row.get("window_id") or row.get("window_name")
        if row.get("learned_score_null_rate") is not None and row["learned_score_null_rate"] > 0.05:
            reasons.append(f"{prefix}: learned_score_null_rate > 0.05")
        if any(status != "valid" and count for status, count in (row.get("artifact_validation_status_counts") or {}).items()):
            reasons.append(f"{prefix}: artifact_validation_status != valid")
        if row.get("inference_error_rate", 0.0) > 0.01:
            reasons.append(f"{prefix}: inference_error_rate > 0.01")
        if row.get("missing_safe_feature_count_p95") is not None and row["missing_safe_feature_count_p95"] > 0:
            reasons.append(f"{prefix}: missing_safe_feature_count_p95 > 0")
        for check_name in (
            "queue_generation_latency_regression",
            "shadow_scoring_latency_p95",
            "artifact_load_latency",
            "score_distribution_p95_shift_std",
            "top100_jaccard_relative_change",
        ):
            if (row.get("alert_checks", {}).get(check_name) or {}).get("triggered"):
                reasons.append(f"{prefix}: {check_name} alert")
        safety_failures = []
        if row.get("default_field_equality") is not True:
            safety_failures.append("default_field_equality")
        if row.get("ordering_equality") is not True:
            safety_failures.append("ordering_equality")
        if row.get("production_labels_read") is not False:
            safety_failures.append("production_labels_read")
        if row.get("affects_default_ranking") is not False:
            safety_failures.append("affects_default_ranking")
        if row.get("shadow_only") is not True:
            safety_failures.append("shadow_only")
        if row.get("leakage_guard_status") not in {"metadata_present", None}:
            safety_failures.append("leakage_guard_failure")
        for name in safety_failures:
            hard = True
            reasons.append(f"{prefix}: hard safety alert {name}")
        if row.get("alert_status"):
            reasons.append(f"{prefix}: per-window monitor alert triggered")
    return {
        "per_window_alert_status": {row.get("window_id"): row.get("alert_status") for row in windows},
        "aggregate_alert_status": bool(reasons),
        "hard_safety_alert": hard,
        "alert_reasons": sorted(set(reasons)),
    }


def _decision(windows: list[dict], alerts: dict) -> dict:
    aggregate = alerts.get("aggregate_alert_status") is True
    hard = alerts.get("hard_safety_alert") is True
    stable = (
        len(windows) >= 10
        and not aggregate
        and all((row.get("learned_score_null_rate") or 0.0) < 0.05 for row in windows)
        and all((row.get("inference_error_rate") or 0.0) < 0.01 for row in windows)
        and all(row.get("default_field_equality") is True and row.get("ordering_equality") is True for row in windows)
        and all(row.get("production_labels_read") is False for row in windows)
        and all(
            not any(status != "valid" and count for status, count in (row.get("artifact_validation_status_counts") or {}).items())
            for row in windows
        )
    )
    if hard:
        action = "rollback"
        reason = "hard safety alert triggered; do not expand shadow rollout"
    elif aggregate:
        action = "pause"
        reason = "non-safety alert triggered; investigate before continuing"
    elif stable:
        action = "expand"
        reason = "at least ten broader shadow-only windows were stable with no aggregate alerts"
    else:
        action = "hold"
        reason = "fewer than ten stable broader windows or insufficient observation for expansion"
    return {
        "recommended_action": action,
        "continue_same_traffic": action in {"hold", "expand"},
        "expand_shadow_traffic": action == "expand",
        "hold_for_more_observation": action == "hold",
        "hold_for_human_feedback": action in {"hold", "expand"},
        "pause": action == "pause",
        "rollback": action == "rollback",
        "default_layer_5_weights_unchanged": True,
        "default_sorting_unchanged": all(row.get("ordering_equality") is True for row in windows),
        "reason": reason,
    }


def _artifact_summary(window_reports: list[dict]) -> dict:
    first = (window_reports[0].get("rollout_config") or {}) if window_reports else {}
    return {
        "artifact_path": first.get("artifact_path"),
        "artifact_version": first.get("artifact_version"),
        "artifact_validation_status": first.get("artifact_validation_status"),
    }


def _resolve_input_queues(input_queues: list[str] | None, input_dir: str | None, window_glob: str) -> list[str]:
    queues = list(input_queues or [])
    if input_dir:
        queues.extend(str(path) for path in sorted(Path(input_dir).glob(window_glob)))
    return queues


def _safe_review_row(row: dict) -> dict:
    allowed = {
        "task_id",
        "sample_id",
        "image_id",
        "annotation_id",
        "prediction_id",
        "dataset",
        "category_name",
        "current_priority_score",
        "learned_shadow_score",
        "rank_current",
        "rank_learned",
        "rank_delta",
        "prediction_time_safe_features",
        "shadow_metadata_status",
        "artifact_version",
        "shadow_only",
        "affects_default_ranking",
    }
    return {key: row.get(key) for key in sorted(allowed) if key in row}


def _human_review_task_row(row: dict, group: str) -> dict:
    safe = _safe_review_row(row)
    return {
        **safe,
        "group": group,
        "window_ids": list(row.get("window_ids") or []),
        "human_review_status": None,
        "human_review_outcome": None,
        "human_review_notes": None,
        "reviewer": None,
        "reviewed_at": None,
    }


def _drift_outliers(windows: list[dict]) -> list[dict]:
    outliers = []
    for row in windows:
        value = row.get("p95_drift_vs_baseline")
        if value is None or float(value) <= P95_DRIFT_OUTLIER_THRESHOLD:
            continue
        outliers.append(
            {
                "window_id": row.get("window_id"),
                "window_name": row.get("window_name"),
                "input_queue_path": row.get("input_queue_path"),
                "p95_drift_vs_baseline": float(value),
                "score_distribution_psi": row.get("score_distribution_psi"),
                "alert_status": row.get("alert_status"),
                "interpretation": "p95 drift exceeded observation threshold; inspect sample mix and learned score distribution before default promotion.",
            }
        )
    return outliers


def _dedupe_review_rows(rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in rows:
        key = _dedupe_key(row)
        if key not in merged:
            merged[key] = dict(row)
            continue
        existing = merged[key]
        windows = sorted(set((existing.get("window_ids") or []) + (row.get("window_ids") or [])))
        if abs(float(row.get("rank_delta") or 0.0)) > abs(float(existing.get("rank_delta") or 0.0)):
            merged[key] = dict(row)
        merged[key]["window_ids"] = windows
    return list(merged.values())


def _sort_review_rows(group: str, rows: list[dict]) -> list[dict]:
    if group == "high_learned_low_current":
        return sorted(rows, key=lambda row: (-abs(float(row.get("rank_delta") or 0.0)), -float(row.get("learned_shadow_score") or 0.0), float(row.get("current_priority_score") or 0.0)))
    if group == "high_current_low_learned":
        return sorted(rows, key=lambda row: (-abs(float(row.get("rank_delta") or 0.0)), -float(row.get("current_priority_score") or 0.0), float(row.get("learned_shadow_score") or 0.0)))
    if group == "top_learned":
        return sorted(rows, key=lambda row: (-float(row.get("learned_shadow_score") or 0.0), row.get("rank_learned") or 10**9))
    return sorted(rows, key=lambda row: (-float(row.get("current_priority_score") or 0.0), row.get("rank_current") or 10**9))


def _dedupe_key(row: dict) -> str:
    for key in ("sample_id", "image_id", "annotation_id", "task_id"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    return json.dumps(row, sort_keys=True)


def _jaccard_only(topk: dict) -> dict:
    return {key: (value or {}).get("jaccard") for key, value in topk.items()}


def _distribution_drift_series(window_reports: list[dict]) -> list[float | None]:
    if not window_reports:
        return []
    base = window_reports[0].get("score_distribution") or {}
    return [_simple_distribution_drift(base, report.get("score_distribution") or {}) for report in window_reports]


def _simple_distribution_drift(base: dict, current: dict) -> float | None:
    keys = ["p05", "p25", "median", "p75", "p95"]
    if not all(base.get(key) is not None and current.get(key) is not None for key in keys):
        return None
    scale = max(float(base.get("std") or 0.0), 1e-9)
    return float(sum(abs(float(current[key]) - float(base[key])) for key in keys) / (len(keys) * scale))


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run learned boundary/shape shadow-only live rollout windows.")
    parser.add_argument("--review-queue", help="Single-window input queue. Kept for backward compatibility.")
    parser.add_argument("--output-dir", help="Single-window output dir. Kept for backward compatibility.")
    parser.add_argument("--input-queues", nargs="*", help="Multi-window input queues in order.")
    parser.add_argument("--input-dir", help="Directory containing window queues.")
    parser.add_argument("--window-glob", default="review_queue_*.jsonl")
    parser.add_argument("--output-root", help="Multi-window output root.")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--enable-learned-boundary-shape-shadow", action="store_true")
    parser.add_argument("--production-mode", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--window-id", default="live_shadow_window_001")
    parser.add_argument("--baseline")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--decision-report-path", default=DEFAULT_DECISION_REPORT_PATH)
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_queues or args.input_dir or args.output_root:
        if not args.output_root:
            raise SystemExit("--output-root is required for multi-window mode")
        result = run_multi_window_shadow_observation(
            args.input_queues,
            args.output_root,
            artifact_dir=args.artifact_dir,
            input_dir=args.input_dir,
            window_glob=args.window_glob,
            baseline=args.baseline,
            max_windows=args.max_windows,
            top_k=args.top_k,
            decision_report_path=args.decision_report_path,
        )
        print(
            "[OK] multi-window shadow observation "
            f"windows={result['windows_processed']} "
            f"alerts={result['alerts']['aggregate_alert_status']} "
            f"decision={result['decision']['recommended_action']} "
            f"output_root={args.output_root}"
        )
        return 0
    if not args.review_queue or not args.output_dir:
        raise SystemExit("single-window mode requires --review-queue and --output-dir")
    result = run_live_shadow_rollout(
        args.review_queue,
        args.output_dir,
        artifact_dir=args.artifact_dir,
        window_id=args.window_id,
        baseline=args.baseline,
        report_path=args.report_path,
        top_k=args.top_k,
    )
    print(
        "[OK] live shadow rollout "
        f"rows={result['queue_batch_summary']['total_rows']} "
        f"alerts={result['monitoring']['alerts']['triggered']} "
        f"report={args.report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
