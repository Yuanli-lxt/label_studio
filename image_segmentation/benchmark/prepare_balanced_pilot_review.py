from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GROUPS = [
    "high_learned_low_current",
    "high_current_low_learned",
    "top_learned",
    "control_current_top",
]

FORBIDDEN_REVIEW_FIELDS = {
    "evaluation_only",
    "boundary_metadata",
    "gt_mask_path",
    "gt_mask",
    "gt_path",
    "iou",
    "dice",
    "correction_delta",
    "severity",
    "major_correction_label",
    "labels",
}


def prepare_balanced_pilot_review(
    review_queue: str,
    output_dir: str,
    per_group: int = 15,
    pilot_id: str = "segmentation_balanced_pilot_2026_06_04",
) -> dict:
    rows = [_safe_candidate(row) for row in _read_jsonl(Path(review_queue))]
    rows = [row for row in rows if _is_importable(row)]
    groups = _candidate_groups(rows)
    assignment = _balanced_assignment(groups, int(per_group))
    _assert_group_counts(assignment, int(per_group))

    tasks = [_label_studio_task(row, pilot_id) for row in assignment]
    _assert_safe_payload(assignment)
    _assert_safe_payload(tasks)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "review_assignment": str(output / "review_assignment.jsonl"),
        "review_assignment_template_csv": str(output / "review_assignment_template.csv"),
        "label_studio_tasks": str(output / "label_studio_tasks.json"),
        "review_guidelines": str(output / "review_guidelines.md"),
        "summary": str(output / "balanced_pilot_summary.json"),
    }

    summary = {
        "input_review_queue": review_queue,
        "output_dir": str(output),
        "pilot_id": pilot_id,
        "per_group": int(per_group),
        "rows": len(assignment),
        "group_counts": {group: sum(row.get("group") == group for row in assignment) for group in GROUPS},
        "safe_fields_only": True,
        "offline_analysis_only": True,
        "shadow_only": True,
        "affects_default_ranking": False,
        "outputs": outputs,
    }

    _write_jsonl(Path(outputs["review_assignment"]), assignment)
    _write_csv(Path(outputs["review_assignment_template_csv"]), assignment)
    _write_json(Path(outputs["label_studio_tasks"]), tasks)
    Path(outputs["review_guidelines"]).write_text(_guidelines(), encoding="utf-8")
    _write_json(Path(outputs["summary"]), summary)
    return summary


def _safe_candidate(row: dict) -> dict:
    metadata = row.get("shadow_score_metadata") if isinstance(row.get("shadow_score_metadata"), dict) else {}
    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "prediction_id": row.get("prediction_id"),
        "dataset": row.get("dataset"),
        "category_name": row.get("category_name"),
        "image": row.get("image"),
        "gt_reference": _first_text(
            row,
            "gt_reference",
            "gt_reference_image",
            "gt_preview",
            "gt_mask_preview",
            "gt_mask_reference",
        ),
        "mobilesam_preview": _first_text(
            row,
            "mobilesam_preview",
            "mask_preview",
            "model_mask_preview",
            "prediction_preview",
            "preview_image",
        ),
        "bbox": _prompt_bbox(row),
        "current_priority_score": _num(row.get("priority_score")),
        "learned_shadow_score": _shadow_value(row, "learned_boundary_shape_only_score"),
        "rank_current": row.get("rank"),
        "rank_learned": None,
        "rank_delta": None,
        "prediction_time_safe_features": _safe_prediction_features(row),
        "artifact_version": metadata.get("artifact_version"),
        "shadow_metadata_status": metadata.get("artifact_validation_status"),
        "shadow_only": True,
        "affects_default_ranking": False,
        "human_review_outcome": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": None,
    }


def _prompt_bbox(row: dict) -> list[float] | None:
    for key in ("prompt_bbox", "prompt_box", "bbox"):
        value = row.get(key)
        if isinstance(value, list) and len(value) == 4:
            return value

    mask_quality = row.get("mask_quality") if isinstance(row.get("mask_quality"), dict) else {}
    value = mask_quality.get("prompt_bbox") or mask_quality.get("prompt_box")
    if isinstance(value, list) and len(value) == 4:
        return value

    source = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
    quality = source.get("mask_quality") if isinstance(source.get("mask_quality"), dict) else {}
    value = quality.get("prompt_bbox") or quality.get("prompt_box")
    return value if isinstance(value, list) and len(value) == 4 else None


def _safe_prediction_features(row: dict) -> dict:
    features = row.get("prediction_features") if isinstance(row.get("prediction_features"), dict) else {}
    if not features:
        quality = row.get("mask_quality") if isinstance(row.get("mask_quality"), dict) else {}
        features = quality.get("prediction_time_boundary_shape")
        features = features if isinstance(features, dict) else {}
    return {str(key): value for key, value in features.items() if str(key).startswith("pred_")}


def _candidate_groups(rows: list[dict]) -> dict[str, list[dict]]:
    current_rank = _rank_map(rows, "current_priority_score")
    learned_rank = _rank_map(rows, "learned_shadow_score")
    enriched = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["rank_current"] = current_rank[idx] + 1
        out["rank_learned"] = learned_rank[idx] + 1
        out["rank_delta"] = out["rank_current"] - out["rank_learned"]
        enriched.append(out)

    return {
        "high_learned_low_current": sorted(
            [row for row in enriched if _num(row.get("rank_delta")) > 0],
            key=lambda row: (-_num(row.get("rank_delta")), row.get("rank_learned") or 10**9),
        ),
        "high_current_low_learned": sorted(
            [row for row in enriched if _num(row.get("rank_delta")) < 0],
            key=lambda row: (_num(row.get("rank_delta")), row.get("rank_current") or 10**9),
        ),
        "top_learned": sorted(
            enriched,
            key=lambda row: (-_num(row.get("learned_shadow_score")), row.get("rank_learned") or 10**9),
        ),
        "control_current_top": sorted(
            enriched,
            key=lambda row: (-_num(row.get("current_priority_score")), row.get("rank_current") or 10**9),
        ),
    }


def _balanced_assignment(groups: dict[str, list[dict]], per_group: int) -> list[dict]:
    assignment = []
    seen = set()
    for group in GROUPS:
        for row in groups.get(group, []):
            key = _dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["group"] = group
            assignment.append(out)
            if sum(item.get("group") == group for item in assignment) >= per_group:
                break
    return assignment


def _assert_group_counts(rows: list[dict], per_group: int) -> None:
    counts = {group: sum(row.get("group") == group for row in rows) for group in GROUPS}
    short = {group: count for group, count in counts.items() if count < per_group}
    if short:
        raise ValueError(f"fewer than {per_group} importable rows for groups: {short}")


def _label_studio_task(row: dict, pilot_id: str) -> dict:
    image = str(row.get("image")).strip()
    data = {
        "image": image,
        "gt_reference": str(row.get("gt_reference") or image).strip(),
        "mobilesam_preview": str(row.get("mobilesam_preview") or image).strip(),
    }
    if row.get("bbox") is not None:
        data["bbox"] = row["bbox"]

    meta = {
        "review_group": row.get("group"),
        "shadow_review_pilot_id": pilot_id,
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "prediction_id": row.get("prediction_id"),
        "dataset": row.get("dataset"),
        "category_name": row.get("category_name"),
        "current_priority_score": row.get("current_priority_score"),
        "learned_shadow_score": row.get("learned_shadow_score"),
        "rank_current": row.get("rank_current"),
        "rank_learned": row.get("rank_learned"),
        "rank_delta": row.get("rank_delta"),
        "artifact_version": row.get("artifact_version"),
        "shadow_metadata_status": row.get("shadow_metadata_status"),
        "shadow_only": True,
        "affects_default_ranking": False,
    }
    return {"id": f"{pilot_id}:{row.get('group')}:{_dedupe_key(row)}", "data": data, "meta": meta}


def _assert_safe_payload(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False)
    for forbidden in FORBIDDEN_REVIEW_FIELDS:
        if forbidden in text:
            raise ValueError(f"forbidden field leaked into balanced pilot payload: {forbidden}")


def _is_importable(row: dict) -> bool:
    return isinstance(row.get("image"), str) and bool(row["image"].strip()) and row.get("learned_shadow_score") is not None


def _first_text(row: dict, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _rank_map(rows: list[dict], field: str) -> list[int]:
    order = sorted(range(len(rows)), key=lambda idx: (-_num(rows[idx].get(field)), idx))
    ranks = [0] * len(rows)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _dedupe_key(row: dict) -> str:
    if row.get("sample_id") is not None:
        return f"sample_id:{row.get('sample_id')}"
    if row.get("image_id") is not None and row.get("annotation_id") is not None:
        return f"image_annotation:{row.get('image_id')}:{row.get('annotation_id')}"
    if row.get("task_id") is not None:
        return f"task_id:{row.get('task_id')}"
    return f"image:{row.get('image')}"


def _shadow_value(row: dict, field: str) -> float | None:
    shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
    value = shadow.get(field)
    if value is None:
        return None
    return _num(value)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _read_jsonl(path: Path) -> list[dict]:
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
        "human_review_outcome",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _guidelines() -> str:
    return "\n".join(
        [
            "# Segmentation Balanced Pilot Review Guidelines",
            "",
            "Valid outcomes: ok, minor_correction_needed, major_correction_needed, unclear.",
            "This feedback is offline analysis only.",
            "Do not use these outcomes to change default Layer 5 weights or sorting.",
        ]
    ) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a 4-group balanced segmentation human review pilot.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=15)
    parser.add_argument("--pilot-id", default="segmentation_balanced_pilot_2026_06_04")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_balanced_pilot_review(
        args.review_queue,
        args.output_dir,
        per_group=args.per_group,
        pilot_id=args.pilot_id,
    )
    print(f"[OK] balanced pilot rows={result['rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
