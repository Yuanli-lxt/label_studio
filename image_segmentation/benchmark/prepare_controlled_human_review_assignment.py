from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.prediction_feature_scoring import safe_prediction_feature_vector


GROUP_LIMITS = {
    "high_learned_low_current": 100,
    "high_current_low_learned": 50,
    "top_learned": 50,
    "control_current_top": 50,
}


def prepare_controlled_human_review_assignment(
    multi_window_root: str,
    selected_windows_file: str,
    output_dir: str,
    high_learned_low_current: int = 100,
    high_current_low_learned: int = 50,
    top_learned: int = 50,
    control_current_top: int = 50,
) -> dict:
    limits = {
        "high_learned_low_current": high_learned_low_current,
        "high_current_low_learned": high_current_low_learned,
        "top_learned": top_learned,
        "control_current_top": control_current_top,
    }
    root = Path(multi_window_root)
    selected_windows = [
        line.strip()
        for line in Path(selected_windows_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for window in selected_windows:
        for row in _read_jsonl(root / window / "live_review_queue.shadow_scored.jsonl"):
            rows.append(_safe_candidate(row, window))
    groups = {
        "high_learned_low_current": _top_rank_delta(rows, learned_high=True, limit=limits["high_learned_low_current"]),
        "high_current_low_learned": _top_rank_delta(rows, learned_high=False, limit=limits["high_current_low_learned"]),
        "top_learned": _top_by(rows, "learned_shadow_score", limits["top_learned"]),
        "control_current_top": _top_by(rows, "current_priority_score", limits["control_current_top"]),
    }
    assignment = []
    seen = set()
    for group, group_rows in groups.items():
        for row in group_rows:
            key = _dedupe_key(row, group)
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["group"] = group
            out["human_review_outcome"] = None
            out["reviewer"] = None
            out["reviewed_at"] = None
            out["notes"] = None
            assignment.append(out)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "review_assignment.jsonl", assignment)
    _write_csv(output / "review_assignment_template.csv", assignment)
    (output / "review_guidelines.md").write_text(_guidelines(), encoding="utf-8")
    summary = {
        "multi_window_root": multi_window_root,
        "selected_windows_file": selected_windows_file,
        "selected_windows": selected_windows,
        "rows": len(assignment),
        "group_limits": limits,
        "group_counts": {group: sum(row.get("group") == group for row in assignment) for group in groups},
        "outputs": {
            "review_assignment": str(output / "review_assignment.jsonl"),
            "review_assignment_template_csv": str(output / "review_assignment_template.csv"),
            "review_guidelines": str(output / "review_guidelines.md"),
        },
        "safe_fields_only": True,
        "offline_analysis_only": True,
    }
    _write_json(output / "controlled_human_review_assignment_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare safe human review assignment from controlled comparable shadow windows.")
    parser.add_argument("--multi-window-root", required=True)
    parser.add_argument("--selected-windows-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-learned-low-current", type=int, default=100)
    parser.add_argument("--high-current-low-learned", type=int, default=50)
    parser.add_argument("--top-learned", type=int, default=50)
    parser.add_argument("--control-current-top", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_controlled_human_review_assignment(
        args.multi_window_root,
        args.selected_windows_file,
        args.output_dir,
        high_learned_low_current=args.high_learned_low_current,
        high_current_low_learned=args.high_current_low_learned,
        top_learned=args.top_learned,
        control_current_top=args.control_current_top,
    )
    print(f"[OK] controlled human review assignment rows={result['rows']} output_dir={args.output_dir}")
    return 0


def _safe_candidate(row: dict, window: str) -> dict:
    learned = _shadow_value(row, "learned_boundary_shape_only_score")
    current = _num(row.get("priority_score"))
    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "current_priority_score": current,
        "learned_shadow_score": learned,
        "rank_current": row.get("rank"),
        "rank_learned": None,
        "rank_delta": None,
        "prediction_time_safe_features": safe_prediction_feature_vector(row),
        "artifact_version": (_metadata(row) or {}).get("artifact_version"),
        "shadow_metadata_status": (_metadata(row) or {}).get("artifact_validation_status"),
        "window_ids": [window],
    }


def _top_rank_delta(rows: list[dict], learned_high: bool, limit: int) -> list[dict]:
    current_rank = _rank_map(rows, "current_priority_score")
    learned_rank = _rank_map(rows, "learned_shadow_score")
    scored = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["rank_current"] = current_rank[idx] + 1
        out["rank_learned"] = learned_rank[idx] + 1
        out["rank_delta"] = abs(out["rank_current"] - out["rank_learned"])
        if learned_high and out["rank_learned"] < out["rank_current"]:
            scored.append(out)
        if not learned_high and out["rank_current"] < out["rank_learned"]:
            scored.append(out)
    return sorted(scored, key=lambda row: (-_num(row.get("rank_delta")), row.get("rank_learned") or 10**9))[:limit]


def _top_by(rows: list[dict], field: str, limit: int) -> list[dict]:
    ranked = []
    learned_rank = _rank_map(rows, "learned_shadow_score")
    current_rank = _rank_map(rows, "current_priority_score")
    for idx, row in enumerate(rows):
        out = dict(row)
        out["rank_current"] = current_rank[idx] + 1
        out["rank_learned"] = learned_rank[idx] + 1
        out["rank_delta"] = abs(out["rank_current"] - out["rank_learned"])
        ranked.append(out)
    return sorted(ranked, key=lambda row: (-_num(row.get(field)), row.get("task_id") or ""))[:limit]


def _rank_map(rows: list[dict], field: str) -> list[int]:
    order = sorted(range(len(rows)), key=lambda idx: (-_num(rows[idx].get(field)), idx))
    ranks = [0] * len(rows)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _dedupe_key(row: dict, group: str) -> str:
    for key in ("sample_id", "image_id", "annotation_id", "task_id"):
        if row.get(key) is not None:
            return f"{group}:{key}:{row.get(key)}"
    return f"{group}:{json.dumps(row, sort_keys=True)}"


def _metadata(row: dict) -> dict:
    return row.get("shadow_score_metadata") if isinstance(row.get("shadow_score_metadata"), dict) else {}


def _shadow_value(row: dict, field: str) -> float | None:
    shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
    try:
        return float(shadow.get(field))
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _guidelines() -> str:
    return "\n".join(
        [
            "# Controlled Shadow Review Guidelines",
            "",
            "Use only the provided production-safe fields.",
            "",
            "Valid outcomes: major_correction_needed, minor_correction_needed, ok, unclear.",
            "This feedback is offline analysis only and must not feed production scoring, default sorting, or Layer 5 weights.",
        ]
    ) + "\n"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "task_id",
        "sample_id",
        "image_id",
        "annotation_id",
        "group",
        "current_priority_score",
        "learned_shadow_score",
        "rank_current",
        "rank_learned",
        "rank_delta",
        "artifact_version",
        "shadow_metadata_status",
        "window_ids",
        "human_review_outcome",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            flat["window_ids"] = ",".join(row.get("window_ids") or [])
            writer.writerow(flat)


if __name__ == "__main__":
    raise SystemExit(main())
