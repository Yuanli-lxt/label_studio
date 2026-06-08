from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

GROUPS = [
    "high_learned_low_current",
    "high_current_low_learned",
    "top_learned",
    "control_current_top",
]

OUTCOMES = ["no_fix", "minor_fix", "major_fix", "redo", "skip"]

OUTCOME_FROM_NAMES = {"review_outcome", "correction_effort"}

OUTCOME_ALIASES = {
    "no fix": "no_fix",
    "no_fix": "no_fix",
    "ok": "no_fix",
    "minor fix": "minor_fix",
    "minor_fix": "minor_fix",
    "minor_correction_needed": "minor_fix",
    "major fix": "major_fix",
    "major_fix": "major_fix",
    "major_correction_needed": "major_fix",
    "redo": "redo",
    "skip": "skip",
    "unclear": "skip",
}


def summarize_balanced_pilot_reviews(
    assignment_path: str,
    output_dir: str,
    label_studio_export_path: str | None = None,
    pilot_id: str = "segmentation_balanced_pilot_2026_06_04",
) -> dict[str, Any]:
    assignment = _read_jsonl(Path(assignment_path))
    export_rows = _read_json_or_empty(Path(label_studio_export_path)) if label_studio_export_path else []
    export_outcomes = _label_studio_outcomes_by_key(export_rows)

    reviewed_rows = []
    for row in assignment:
        key = _row_key(row)
        export = export_outcomes.get(key, {})
        outcome = export.get("outcome") or _normalize_outcome(row.get("human_review_outcome"))
        if not outcome:
            continue
        reviewed_rows.append(
            {
                "task_id": row.get("task_id"),
                "sample_id": row.get("sample_id"),
                "group": row.get("group") or export.get("group"),
                "review_outcome": outcome,
                "reviewer": export.get("reviewer") or row.get("reviewer"),
                "reviewed_at": export.get("reviewed_at") or row.get("reviewed_at"),
            }
        )

    summary = {
        "pilot_id": pilot_id,
        "assignment_path": str(assignment_path),
        "label_studio_export_path": str(label_studio_export_path) if label_studio_export_path else None,
        "reviewed_total": len(reviewed_rows),
        "groups": _group_summary(reviewed_rows),
        "review_outcome_is_primary_metric": True,
        "pixel_mask_edits_are_auxiliary": True,
        "shadow_only": True,
        "affects_default_ranking": False,
        "learned_boundary_shape_promoted": False,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "balanced_pilot_review_summary.json", summary)
    (output / "balanced_pilot_review_summary.md").write_text(_summary_md(summary), encoding="utf-8")
    _write_jsonl(output / "balanced_pilot_review_rows.jsonl", reviewed_rows)
    return summary


def export_label_studio_project_tasks(project_id: int, output_path: str) -> list[dict[str, Any]]:
    from scripts.lib.label_studio_client import LabelStudioClient, LabelStudioSettings

    client = LabelStudioClient(LabelStudioSettings.from_env(require_token=True))
    rows = client.list_tasks(int(project_id))
    _write_json(Path(output_path), rows)
    return rows


def _label_studio_outcomes_by_key(tasks: Any) -> dict[str, dict[str, Any]]:
    if isinstance(tasks, dict) and isinstance(tasks.get("tasks"), list):
        tasks = tasks["tasks"]
    if not isinstance(tasks, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
        outcome = _task_review_outcome(task)
        if not outcome:
            continue
        payload = {
            "outcome": outcome,
            "group": meta.get("review_group") or meta.get("group"),
            **_annotation_audit(task),
        }
        for key in _task_keys(task):
            out[key] = payload
    return out


def _task_review_outcome(task: dict[str, Any]) -> str | None:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        annotations = task.get("completions")
    if not isinstance(annotations, list):
        return None
    for annotation in reversed([item for item in annotations if isinstance(item, dict)]):
        result = annotation.get("result")
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict) or item.get("from_name") not in OUTCOME_FROM_NAMES:
                continue
            value = item.get("value") if isinstance(item.get("value"), dict) else {}
            choices = value.get("choices")
            if isinstance(choices, list) and choices:
                outcome = _normalize_outcome(choices[0])
                if outcome:
                    return outcome
    return None


def _annotation_audit(task: dict[str, Any]) -> dict[str, Any]:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return {}
    annotations = [item for item in annotations if isinstance(item, dict)]
    if not annotations:
        return {}
    annotation = annotations[-1]
    reviewer = annotation.get("completed_by")
    if isinstance(reviewer, dict):
        reviewer = reviewer.get("email") or reviewer.get("username") or reviewer.get("id")
    return {
        "reviewer": reviewer,
        "reviewed_at": annotation.get("updated_at") or annotation.get("created_at"),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row.get("group") == group]
        counts = {name: sum(row.get("review_outcome") == name for row in group_rows) for name in OUTCOMES}
        total = len(group_rows)
        counts["total"] = total
        counts["major_or_redo_rate"] = (counts["major_fix"] + counts["redo"]) / total if total else 0.0
        summary[group] = counts
    return summary


def _normalize_outcome(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return OUTCOME_ALIASES.get(key)


def _task_keys(task: dict[str, Any]) -> list[str]:
    keys = []
    for value in (task.get("id"), task.get("task_id")):
        if value is not None:
            text = str(value)
            keys.append(text)
            if ":task_id:" in text:
                keys.append(text.rsplit(":task_id:", 1)[1])
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    for field in ("sample_id", "task_id", "prediction_id"):
        if meta.get(field) is not None:
            text = str(meta[field])
            keys.append(text)
            if text.endswith("_mask"):
                keys.append(text[: -len("_mask")])
    return keys


def _row_key(row: dict[str, Any]) -> str:
    for field in ("sample_id", "task_id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return str(row.get("image") or "")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Balanced Pilot Review Summary",
        "",
        f"Pilot: `{summary['pilot_id']}`",
        "",
        "| Group | Total | No fix | Minor fix | Major fix | Redo | Skip | Major/Redo rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = summary["groups"][group]
        lines.append(
            f"| {group} | {row['total']} | {row['no_fix']} | {row['minor_fix']} | "
            f"{row['major_fix']} | {row['redo']} | {row['skip']} | {row['major_or_redo_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Primary metric: required `review_outcome` quality judgment. Pixel-level mask edits are auxiliary.",
            "",
            "Safety: shadow-only analysis; default Layer 5 weights and sorting are unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and summarize balanced pilot Label Studio outcomes.")
    parser.add_argument("--assignment-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-id", default="segmentation_balanced_pilot_2026_06_04")
    parser.add_argument("--label-studio-export-path")
    parser.add_argument("--project-id", type=int, help="export Label Studio project tasks before summarizing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_path = args.label_studio_export_path
    if args.project_id is not None:
        export_path = export_path or str(Path(args.output_dir) / "label_studio_project_tasks_export.json")
        export_label_studio_project_tasks(args.project_id, export_path)
    summary = summarize_balanced_pilot_reviews(
        assignment_path=args.assignment_path,
        output_dir=args.output_dir,
        label_studio_export_path=export_path,
        pilot_id=args.pilot_id,
    )
    print(f"[OK] reviewed_total={summary['reviewed_total']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
