from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def write_controlled_shadow_validation_report(
    selection_path: str,
    drift_path: str,
    topk_path: str,
    missing_features_path: str,
    human_review_assignment_dir: str,
    output_path: str,
    feedback_summary_path: str | None = None,
) -> dict:
    selection = _read_json(Path(selection_path))
    drift = _read_json(Path(drift_path))
    topk = _read_json(Path(topk_path))
    missing = _read_json(Path(missing_features_path))
    assignment = _read_json(Path(human_review_assignment_dir) / "controlled_human_review_assignment_summary.json")
    feedback = _read_json(Path(feedback_summary_path)) if feedback_summary_path else {}
    severities = Counter()
    for report in (drift, topk, missing):
        severities.update(report.get("alert_severity_summary") or {})
    hard = severities.get("hard_safety_alert", 0) > 0
    stability = severities.get("stability_alert", 0) > 0
    feedback_available = bool(feedback.get("feedback_available"))
    decision = {
        "resume_same_shadow_traffic": not hard,
        "resume_small_expansion": bool(not hard and not stability and feedback_available),
        "hold_broader_expansion": True,
        "pause_or_rollback": hard,
        "ready_for_default_sorting": False,
        "ready_for_weight_promotion": False,
        "default_layer_5_weights_unchanged": True,
        "default_sorting_unchanged": True,
    }
    if hard:
        controlled_conclusion = "hard safety alert present; pause or rollback"
    elif stability:
        controlled_conclusion = "controlled comparable windows still contain stability alerts; hold expansion"
    elif not feedback_available:
        controlled_conclusion = "controlled comparable windows are prepared; feedback is still needed before expansion"
    else:
        controlled_conclusion = "controlled comparable windows are stable with feedback available; consider only small same-scope expansion"
    payload = {
        "selection": selection_path,
        "drift": drift_path,
        "topk": topk_path,
        "missing_features": missing_features_path,
        "human_review_assignment_dir": human_review_assignment_dir,
        "feedback_summary": feedback_summary_path,
        "selected_windows": selection.get("selected_windows") or [],
        "rejected_window_count": selection.get("rejected_window_count"),
        "alert_severity_summary": dict(sorted(severities.items())),
        "controlled_stability_conclusion": controlled_conclusion,
        "feedback_available": feedback_available,
        "decision": decision,
        "leakage_guard_summary": {
            "safe_fields_only": all(
                report.get("safe_fields_only") is True
                for report in (selection, drift, topk, missing, assignment)
            ),
            "human_feedback_offline_only": assignment.get("offline_analysis_only") is True,
            "production_labels_read": False,
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(payload, selection, drift, topk, missing, assignment, feedback), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write controlled comparable-window shadow validation report.")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--drift", required=True)
    parser.add_argument("--topk", required=True)
    parser.add_argument("--missing-features", required=True)
    parser.add_argument("--human-review-assignment-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feedback-summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_controlled_shadow_validation_report(
        args.selection,
        args.drift,
        args.topk,
        args.missing_features,
        args.human_review_assignment_dir,
        args.output,
        feedback_summary_path=args.feedback_summary,
    )
    print(f"[OK] controlled validation report conclusion={result['controlled_stability_conclusion']} output={args.output}")
    return 0


def _render_report(payload: dict, selection: dict, drift: dict, topk: dict, missing: dict, assignment: dict, feedback: dict) -> str:
    decision = payload["decision"]
    lines = [
        "# Controlled Comparable Shadow Validation Report",
        "",
        "## Scope",
        "Controlled comparable-window validation and offline human feedback preparation. Default Layer 5 weights and sorting remain unchanged.",
        "",
        "## Selected Comparable Windows",
        f"- {payload.get('selected_windows')}",
        "",
        "## Rejected / Skipped Windows",
        f"- count: {selection.get('rejected_window_count')}",
        f"- rejected_windows: {selection.get('rejected_windows')}",
        "",
        "## Controlled Replay Setup",
        "- existing full-feature comparable windows were used; no production replay path was modified",
        "",
        "## Feature Completeness",
        f"- missing overall assessment: {missing.get('overall_assessment')}",
        "",
        "## Same-Sample Overlap",
        f"- drift comparable_pairs: {len(drift.get('comparable_pairs') or [])}",
        f"- topk comparable_pairs: {len(topk.get('comparable_pairs') or [])}",
        "",
        "## Controlled Drift Results",
        f"- conclusion: {drift.get('conclusion')}",
        f"- alert_severity_summary: {drift.get('alert_severity_summary')}",
        "",
        "## Controlled Top-k Results",
        f"- conclusion: {topk.get('conclusion')}",
        f"- alert_severity_summary: {topk.get('alert_severity_summary')}",
        "",
        "## Missing Feature Results",
        f"- alert_severity_summary: {missing.get('alert_severity_summary')}",
        "",
        "## Latency Summary",
        "- not separately measured in this controlled selector pass",
        "",
        "## Human Review Assignment Path",
        f"- {assignment.get('outputs')}",
        "",
        "## Feedback Summary",
        f"- feedback_available: {payload.get('feedback_available')}",
        f"- recommendation: {feedback.get('recommendation')}",
        "",
        "## Alert Severity Summary",
        f"- {payload.get('alert_severity_summary')}",
        "",
        "## Safety / Leakage",
        f"- {payload.get('leakage_guard_summary')}",
        "",
        "## Decision",
        f"- resume_same_shadow_traffic: {decision['resume_same_shadow_traffic']}",
        f"- resume_small_expansion: {decision['resume_small_expansion']}",
        f"- hold_broader_expansion: {decision['hold_broader_expansion']}",
        f"- pause_or_rollback: {decision['pause_or_rollback']}",
        "",
        "## Promotion Status",
        "- not ready for default sorting",
        "- not ready for weight promotion",
        "",
        "## Controlled Stability Conclusion",
        f"- {payload.get('controlled_stability_conclusion')}",
    ]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
