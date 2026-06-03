from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_comparable_shadow_windows(
    classification_path: str,
    multi_window_root: str,
    output_dir: str,
    require_schema: str = "full_prediction_features",
    require_feature_completeness: str = "full",
    require_queue_type: str = "benchmark_replay",
    min_learned_coverage: float = 0.95,
    max_null_rate: float = 0.05,
) -> dict:
    classification = _read_json(Path(classification_path))
    broader = _read_json(Path(multi_window_root) / "broader_shadow_multi_window_summary.json")
    safety_by_window = _safety_by_window(broader)
    selected = []
    rejected = []
    for window in classification.get("windows") or []:
        decision = _window_selection_decision(
            window,
            safety_by_window.get(window.get("window_id")) or {},
            require_schema=require_schema,
            require_feature_completeness=require_feature_completeness,
            require_queue_type=require_queue_type,
            min_learned_coverage=min_learned_coverage,
            max_null_rate=max_null_rate,
        )
        if decision["selected"]:
            selected.append(window)
        else:
            rejected.append(decision)
    selected_ids = {row.get("window_id") for row in selected}
    pairwise = {
        key: value
        for key, value in (classification.get("pairwise_compatibility_matrix") or {}).items()
        if _pair_in_selected(key, selected_ids)
    }
    overlap = {
        key: value
        for key, value in (classification.get("pairwise_same_sample_overlap_matrix") or {}).items()
        if _pair_in_selected(key, selected_ids)
    }
    filtered_classification = {
        "multi_window_root": classification.get("multi_window_root"),
        "windows": selected,
        "pairwise_same_sample_overlap_matrix": overlap,
        "pairwise_compatibility_matrix": pairwise,
        "safe_fields_only": True,
        "production_safe_classification": True,
        "controlled_selection": True,
    }
    report = {
        "classification": classification_path,
        "multi_window_root": multi_window_root,
        "selection_criteria": {
            "require_schema": require_schema,
            "require_feature_completeness": require_feature_completeness,
            "require_queue_type": require_queue_type,
            "min_learned_coverage": min_learned_coverage,
            "max_null_rate": max_null_rate,
        },
        "selected_windows": [row.get("window_id") for row in selected],
        "selected_window_count": len(selected),
        "rejected_window_count": len(rejected),
        "rejected_windows": rejected,
        "filtered_classification_path": str(Path(output_dir) / "selected_shadow_window_classification.json"),
        "safe_fields_only": True,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "comparable_window_selection.json", report)
    _write_json(output / "selected_shadow_window_classification.json", filtered_classification)
    _write_jsonl(output / "rejected_windows.jsonl", rejected)
    (output / "selected_windows.txt").write_text("\n".join(report["selected_windows"]) + ("\n" if selected else ""), encoding="utf-8")
    (output / "comparable_window_selection.md").write_text(_render_selection_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select controlled comparable shadow windows from schema classification output.")
    parser.add_argument("--classification", required=True)
    parser.add_argument("--multi-window-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-schema", default="full_prediction_features")
    parser.add_argument("--require-feature-completeness", default="full")
    parser.add_argument("--require-queue-type", default="benchmark_replay")
    parser.add_argument("--min-learned-coverage", type=float, default=0.95)
    parser.add_argument("--max-null-rate", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = select_comparable_shadow_windows(
        args.classification,
        args.multi_window_root,
        args.output_dir,
        require_schema=args.require_schema,
        require_feature_completeness=args.require_feature_completeness,
        require_queue_type=args.require_queue_type,
        min_learned_coverage=args.min_learned_coverage,
        max_null_rate=args.max_null_rate,
    )
    print(f"[OK] selected comparable windows={result['selected_window_count']} output_dir={args.output_dir}")
    return 0


def _window_selection_decision(
    window: dict,
    safety: dict,
    *,
    require_schema: str,
    require_feature_completeness: str,
    require_queue_type: str,
    min_learned_coverage: float,
    max_null_rate: float,
) -> dict:
    reasons = []
    severity = "info"
    if window.get("queue_schema_version") != require_schema:
        reasons.append(f"schema {window.get('queue_schema_version')} != {require_schema}")
        severity = "compatibility_warning"
    if window.get("feature_completeness_bucket") != require_feature_completeness:
        reasons.append(f"feature completeness {window.get('feature_completeness_bucket')} != {require_feature_completeness}")
        severity = "compatibility_warning"
    if require_queue_type and window.get("queue_type") != require_queue_type:
        reasons.append(f"queue type {window.get('queue_type')} != {require_queue_type}")
        severity = "compatibility_warning"
    if _num(window.get("learned_score_coverage")) < min_learned_coverage:
        reasons.append("learned score coverage below threshold")
        severity = "stability_alert"
    if _num(window.get("learned_score_null_rate")) > max_null_rate:
        reasons.append("learned score null rate above threshold")
        severity = "stability_alert"
    artifact_counts = window.get("artifact_status_counts") or {}
    if any(status != "valid" and count for status, count in artifact_counts.items()):
        reasons.append(f"artifact status not all valid: {artifact_counts}")
        severity = "stability_alert"
    if window.get("shadow_only_all_true") is not True:
        reasons.append("shadow_only is not true for all rows")
        severity = "hard_safety_alert"
    if window.get("affects_default_ranking_all_false") is not True:
        reasons.append("affects_default_ranking is not false for all rows")
        severity = "hard_safety_alert"
    if safety.get("default_field_equality") is False:
        reasons.append("default field equality failed")
        severity = "hard_safety_alert"
    if safety.get("ordering_equality") is False:
        reasons.append("ordering equality failed")
        severity = "hard_safety_alert"
    if safety.get("production_labels_read") is True:
        reasons.append("production labels were read")
        severity = "hard_safety_alert"
    return {
        "window_id": window.get("window_id"),
        "selected": not reasons,
        "reasons": reasons,
        "reason": "; ".join(reasons) if reasons else "selected",
        "severity": severity,
        "queue_schema_version": window.get("queue_schema_version"),
        "feature_completeness_bucket": window.get("feature_completeness_bucket"),
        "queue_type": window.get("queue_type"),
    }


def _safety_by_window(summary: dict) -> dict[str, dict]:
    return {
        str(row.get("window_id") or row.get("window_name")): row
        for row in summary.get("windows", [])
        if row.get("window_id") or row.get("window_name")
    }


def _pair_in_selected(key: str, selected_ids: set) -> bool:
    left, _, right = key.partition("__")
    return left in selected_ids and right in selected_ids


def _render_selection_markdown(report: dict) -> str:
    lines = [
        "# Controlled Comparable Window Selection",
        "",
        f"- selected_window_count: {report.get('selected_window_count')}",
        f"- rejected_window_count: {report.get('rejected_window_count')}",
        f"- selected_windows: {report.get('selected_windows')}",
        f"- filtered_classification_path: {report.get('filtered_classification_path')}",
        "",
        "## Rejected Windows",
        "",
    ]
    for row in report.get("rejected_windows") or []:
        lines.append(f"- {row.get('window_id')}: severity={row.get('severity')} reason={row.get('reason')}")
    return "\n".join(lines) + "\n"


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
