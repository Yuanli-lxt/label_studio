from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_shadow_schema_aware_decision_report(
    classification_path: str,
    drift_path: str,
    topk_path: str,
    missing_features_path: str,
    output_path: str,
) -> dict:
    classification = _read_json(Path(classification_path))
    drift = _read_json(Path(drift_path))
    topk = _read_json(Path(topk_path))
    missing = _read_json(Path(missing_features_path))
    severities = Counter()
    for report in (drift, topk, missing):
        severities.update(report.get("alert_severity_summary") or {})
    hard = severities.get("hard_safety_alert", 0) > 0
    stability = severities.get("stability_alert", 0) > 0
    compatibility = severities.get("compatibility_warning", 0) > 0
    if hard:
        action = "pause_or_rollback"
    elif stability:
        action = "hold_expansion"
    elif compatibility:
        action = "resume_same_shadow_traffic_cautiously"
    else:
        action = "continue_same_scope_observation"
    decision = {
        "recommended_action": action,
        "resume_same_shadow_traffic": not hard,
        "resume_small_expansion": False,
        "hold_broader_expansion": True,
        "pause_or_rollback": hard,
        "ready_for_default_sorting": False,
        "ready_for_weight_promotion": False,
        "default_layer_5_weights_unchanged": True,
        "default_sorting_unchanged": True,
    }
    payload = {
        "classification": classification_path,
        "drift": drift_path,
        "topk": topk_path,
        "missing_features": missing_features_path,
        "window_classification_summary": _classification_summary(classification),
        "pairwise_compatibility_summary": _compatibility_summary(classification),
        "alert_severity_summary": dict(sorted(severities.items())),
        "decision": decision,
        "human_feedback_status": "not_integrated",
        "leakage_guard_summary": {
            "schema_aware_reports_safe_fields_only": all(report.get("safe_fields_only") is True for report in (drift, topk, missing)),
            "production_labels_read": False,
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(payload, drift, topk, missing), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write schema-aware shadow rollout decision report.")
    parser.add_argument("--classification", required=True)
    parser.add_argument("--drift", required=True)
    parser.add_argument("--topk", required=True)
    parser.add_argument("--missing-features", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_shadow_schema_aware_decision_report(
        args.classification,
        args.drift,
        args.topk,
        args.missing_features,
        args.output,
    )
    print(f"[OK] schema-aware decision action={result['decision']['recommended_action']} output={args.output}")
    return 0


def _classification_summary(classification: dict) -> dict:
    windows = classification.get("windows") or []
    return {
        "windows": len(windows),
        "queue_schema_versions": dict(Counter(str(row.get("queue_schema_version") or "unknown") for row in windows)),
        "feature_completeness_buckets": dict(Counter(str(row.get("feature_completeness_bucket") or "unknown") for row in windows)),
        "queue_types": dict(Counter(str(row.get("queue_type") or "unknown") for row in windows)),
    }


def _compatibility_summary(classification: dict) -> dict:
    pairs = classification.get("pairwise_compatibility_matrix") or {}
    comparable = sum(row.get("compatible_for_drift_comparison") is True for row in pairs.values())
    zero_overlap = sum((row.get("same_sample_overlap") or 0) == 0 for row in pairs.values())
    return {
        "pairs": len(pairs),
        "comparable_pairs": comparable,
        "non_comparable_pairs": len(pairs) - comparable,
        "zero_same_sample_overlap_pairs": zero_overlap,
    }


def _render_report(payload: dict, drift: dict, topk: dict, missing: dict) -> str:
    decision = payload["decision"]
    lines = [
        "# Production Shadow Schema-Aware Decision Report",
        "",
        "## Scope",
        "Schema-aware drift, top-k Jaccard, and missing-feature analysis. Default Layer 5 weights and sorting remain unchanged.",
        "",
        "## Window Classification Summary",
        f"- {payload['window_classification_summary']}",
        "",
        "## Pairwise Compatibility Matrix Summary",
        f"- {payload['pairwise_compatibility_summary']}",
        "",
        "## Schema-Aware Drift Results",
        f"- conclusion: {drift.get('conclusion')}",
        f"- alert_severity_summary: {drift.get('alert_severity_summary')}",
        "",
        "## Schema-Aware Top-k Jaccard Results",
        f"- conclusion: {topk.get('conclusion')}",
        f"- alert_severity_summary: {topk.get('alert_severity_summary')}",
        "",
        "## Schema-Aware Missing Feature Results",
        f"- overall_assessment: {missing.get('overall_assessment')}",
        f"- alert_severity_summary: {missing.get('alert_severity_summary')}",
        "",
        "## Alert Severity Summary",
        f"- {payload['alert_severity_summary']}",
        "",
        "## Human Feedback Status",
        f"- {payload['human_feedback_status']}",
        "",
        "## Safety / Leakage",
        f"- {payload['leakage_guard_summary']}",
        "",
        "## Decision",
        f"- recommended_action: {decision['recommended_action']}",
        f"- resume_same_shadow_traffic: {decision['resume_same_shadow_traffic']}",
        f"- resume_small_expansion: {decision['resume_small_expansion']}",
        f"- hold_broader_expansion: {decision['hold_broader_expansion']}",
        f"- pause_or_rollback: {decision['pause_or_rollback']}",
        "",
        "## Promotion Status",
        "- not ready for default sorting",
        "- not ready for weight promotion",
    ]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
