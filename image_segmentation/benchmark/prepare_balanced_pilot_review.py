from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, unquote


GROUPS = [
    "high_learned_low_current",
    "high_current_low_learned",
    "top_learned",
    "control_current_top",
]

GT_PREVIEW_COLOR = (34, 210, 160, 135)
MOBILESAM_PREVIEW_COLOR = (255, 214, 30, 135)

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
    validation_round: str = "round1_gt_assisted_calibration",
    repeat_review_overlap: int = 0,
    source_window_id: str | None = None,
    previous_assignment: str | None = None,
    local_files_root: str | None = None,
    local_files_subdir: str | None = None,
    manifest_paths: list[str] | None = None,
    prediction_paths: list[str] | None = None,
    preview_style: str = "overlay",
    gt_preview_style: str | None = None,
    prediction_preview_style: str | None = None,
    prediction_display_model_version: str | None = None,
) -> dict:
    rows = [_safe_candidate(row) for row in _read_jsonl(Path(review_queue))]
    rows = [row for row in rows if _is_importable(row)]
    groups = _candidate_groups(rows)
    assignment = _balanced_assignment(groups, int(per_group))
    _assert_group_counts(assignment, int(per_group))
    _mark_repeat_review_overlap(assignment, int(repeat_review_overlap))
    if local_files_subdir:
        _localize_assignment_refs(
            assignment,
            local_files_root=local_files_root or "demo_data/local-files",
            local_files_subdir=local_files_subdir,
        )
        _attach_preview_assets(
            assignment,
            local_files_root=local_files_root or "demo_data/local-files",
            local_files_subdir=local_files_subdir,
            manifest_paths=manifest_paths or [],
            prediction_paths=prediction_paths or [],
            gt_preview_style=gt_preview_style or preview_style,
            prediction_preview_style=prediction_preview_style or preview_style,
            prediction_display_model_version=prediction_display_model_version,
        )

    tasks = [_label_studio_task(row, pilot_id) for row in assignment]
    for row in assignment:
        row.pop("_label_studio_preannotation", None)
    _assert_safe_payload(assignment)
    _assert_safe_payload(tasks)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "review_assignment": str(output / "review_assignment.jsonl"),
        "review_assignment_template_csv": str(output / "review_assignment_template.csv"),
        "label_studio_tasks": str(output / "label_studio_tasks.json"),
        "review_guidelines": str(output / "review_guidelines.md"),
        "summary_template": str(output / "balanced_pilot_review_summary_template.json"),
        "closure_report_template": str(output / "balanced_pilot_closure_report_template.md"),
        "artifact_validation": str(output / "artifact_validation.json"),
        "summary": str(output / "balanced_pilot_summary.json"),
    }

    summary = {
        "input_review_queue": review_queue,
        "output_dir": str(output),
        "pilot_id": pilot_id,
        "validation_round": validation_round,
        "source_window_id": source_window_id,
        "previous_assignment_path": previous_assignment,
        "local_files_root": local_files_root,
        "local_files_subdir": local_files_subdir,
        "manifest_paths": manifest_paths or [],
        "prediction_paths": prediction_paths or [],
        "preview_style": preview_style,
        "gt_preview_style": gt_preview_style or preview_style,
        "prediction_preview_style": prediction_preview_style or preview_style,
        "prediction_display_model_version": prediction_display_model_version,
        "per_group": int(per_group),
        "repeat_review_overlap_per_group": int(repeat_review_overlap),
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
    Path(outputs["review_guidelines"]).write_text(
        _guidelines(
            pilot_id=pilot_id,
            validation_round=validation_round,
            repeat_review_overlap=int(repeat_review_overlap),
        ),
        encoding="utf-8",
    )
    _write_json(Path(outputs["summary_template"]), _summary_template(summary))
    Path(outputs["closure_report_template"]).write_text(_closure_report_template(summary), encoding="utf-8")
    validation = validate_balanced_pilot_artifacts(
        outputs["review_assignment"],
        outputs["label_studio_tasks"],
        int(per_group),
        local_files_root=local_files_root,
    )
    _write_json(Path(outputs["artifact_validation"]), validation)
    _write_json(Path(outputs["summary"]), summary)
    return summary


def validate_balanced_pilot_artifacts(
    assignment_path: str,
    tasks_path: str,
    per_group: int,
    local_files_root: str | None = None,
) -> dict[str, Any]:
    assignment = _read_jsonl(Path(assignment_path))
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        tasks = []
    errors: list[str] = []
    group_counts = {group: sum(row.get("group") == group for row in assignment) for group in GROUPS}
    for group, count in group_counts.items():
        if count != per_group:
            errors.append(f"group {group} has {count} rows, expected {per_group}")

    keys = [_dedupe_key(row) for row in assignment]
    duplicate_samples = len(keys) - len(set(keys))
    if duplicate_samples:
        errors.append(f"duplicate assignment samples: {duplicate_samples}")

    task_ids = [str(task.get("id")) for task in tasks if isinstance(task, dict) and task.get("id") is not None]
    duplicate_task_ids = len(task_ids) - len(set(task_ids))
    if duplicate_task_ids:
        errors.append(f"duplicate Label Studio task ids: {duplicate_task_ids}")

    forbidden_hits = _forbidden_hits({"assignment": assignment, "tasks": tasks})
    if forbidden_hits:
        errors.append(f"forbidden fields leaked: {', '.join(forbidden_hits)}")

    missing_file_refs = 0
    checked_file_refs = 0
    preview_fallback_counts = {"gt_reference": 0, "mobilesam_preview": 0}
    for idx, task in enumerate(tasks, start=1):
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
        if not _valid_label_studio_segmentation_task(data, meta):
            errors.append(f"task[{idx}] does not match segmentation review contract")
            continue
        for field in ("image", "gt_reference", "mobilesam_preview", "gt_mask_preview", "mask_preview"):
            checked_file_refs += 1
            if not _file_ref_exists(data.get(field), local_files_root=local_files_root):
                missing_file_refs += 1
        if data.get("gt_reference") == data.get("image"):
            preview_fallback_counts["gt_reference"] += 1
        if data.get("mobilesam_preview") == data.get("image"):
            preview_fallback_counts["mobilesam_preview"] += 1
    if missing_file_refs:
        errors.append(f"missing file refs: {missing_file_refs}")

    shadow_only_all_true = all(
        isinstance(task, dict)
        and isinstance(task.get("meta"), dict)
        and task["meta"].get("shadow_only") is True
        for task in tasks
    )
    affects_default_ranking_all_false = all(
        isinstance(task, dict)
        and isinstance(task.get("meta"), dict)
        and task["meta"].get("affects_default_ranking") is False
        for task in tasks
    )
    if not shadow_only_all_true:
        errors.append("not all tasks have meta.shadow_only=true")
    if not affects_default_ranking_all_false:
        errors.append("not all tasks have meta.affects_default_ranking=false")

    label_studio_segmentation_contract = not any(
        "segmentation review contract" in error for error in errors
    )
    return {
        "valid": not errors,
        "errors": errors,
        "assignment_count": len(assignment),
        "task_count": len(tasks),
        "group_counts": group_counts,
        "duplicate_samples": duplicate_samples,
        "duplicate_task_ids": duplicate_task_ids,
        "checked_file_refs": checked_file_refs,
        "missing_file_refs": missing_file_refs,
        "preview_fallback_counts": preview_fallback_counts,
        "forbidden_field_hits": forbidden_hits,
        "shadow_only_all_true": shadow_only_all_true,
        "affects_default_ranking_all_false": affects_default_ranking_all_false,
        "label_studio_segmentation_contract": label_studio_segmentation_contract,
        "default_ranking_fields_not_written": ["priority_score", "rank", "priority_bucket"],
    }


def _localize_assignment_refs(
    rows: list[dict],
    local_files_root: str,
    local_files_subdir: str,
) -> None:
    root = Path(local_files_root)
    target_dir = root / local_files_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = _file_ref_to_path(str(row.get("image") or ""), local_files_root=local_files_root)
        if source is None or not source.exists():
            continue
        filename = f"{_safe_filename(row.get('group'))}_{_safe_filename(_dedupe_key(row))}{source.suffix or '.jpg'}"
        target = target_dir / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        local_ref = f"/data/local-files/?d={local_files_subdir}/{filename}"
        row["image"] = local_ref
        if not row.get("gt_reference"):
            row["gt_reference"] = local_ref
        if not row.get("mobilesam_preview"):
            row["mobilesam_preview"] = local_ref


def _attach_preview_assets(
    rows: list[dict],
    local_files_root: str,
    local_files_subdir: str,
    manifest_paths: list[str],
    prediction_paths: list[str],
    gt_preview_style: str = "overlay",
    prediction_preview_style: str = "overlay",
    prediction_display_model_version: str | None = None,
) -> None:
    if not manifest_paths and not prediction_paths:
        return
    manifests = _rows_by_sample_key(manifest_paths)
    predictions = _rows_by_sample_key(prediction_paths)
    preview_root = Path(local_files_root) / local_files_subdir
    gt_dir = preview_root / "gt_previews"
    pred_dir = preview_root / "mobilesam_previews"
    for row in rows:
        key = _sample_lookup_key(row)
        image_path = _file_ref_to_path(str(row.get("image") or ""), local_files_root=local_files_root)
        if image_path is None or not image_path.exists():
            continue
        base = f"{_safe_filename(row.get('group'))}_{_safe_filename(_dedupe_key(row))}"
        manifest = manifests.get(key)
        if manifest:
            try:
                gt_dir.mkdir(parents=True, exist_ok=True)
                target = gt_dir / f"{base}_gt.png"
                _render_gt_preview(image_path, manifest, target, preview_style=gt_preview_style)
                row["gt_reference"] = f"/data/local-files/?d={local_files_subdir}/gt_previews/{target.name}"
            except Exception:
                pass
        prediction = predictions.get(key)
        if prediction:
            try:
                pred_dir.mkdir(parents=True, exist_ok=True)
                target = pred_dir / f"{base}_mobilesam.png"
                _render_prediction_preview(image_path, prediction, target, preview_style=prediction_preview_style)
                row["mobilesam_preview"] = f"/data/local-files/?d={local_files_subdir}/mobilesam_previews/{target.name}"
                preannotation = _label_studio_prediction_payload(prediction)
                if preannotation:
                    if prediction_display_model_version:
                        preannotation["model_version"] = prediction_display_model_version
                    row["_label_studio_preannotation"] = preannotation
            except Exception:
                pass


def _rows_by_sample_key(paths: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        if not path:
            continue
        candidate = Path(path)
        if not candidate.exists():
            continue
        for row in _read_jsonl(candidate):
            key = _sample_lookup_key(row)
            if key and key not in rows:
                rows[key] = row
    return rows


def _sample_lookup_key(row: dict) -> str:
    for field in ("sample_id", "task_id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    image_id = row.get("image_id")
    annotation_id = row.get("annotation_id")
    dataset = row.get("dataset")
    if dataset and image_id and annotation_id:
        return f"{dataset}_{image_id}_{annotation_id}"
    return ""


def _render_gt_preview(image_path: Path, manifest: dict, target: Path, preview_style: str = "overlay") -> None:
    from PIL import Image
    from image_segmentation.benchmark.datasets.coco import segmentation_to_mask

    width = int(manifest.get("width") or 0)
    height = int(manifest.get("height") or 0)
    if width <= 0 or height <= 0:
        with Image.open(image_path) as image:
            width, height = image.size
    mask = segmentation_to_mask(manifest.get("gt_segmentation"), width, height, int(manifest.get("gt_iscrowd") or 0))
    _render_mask_preview(image_path, mask, target, color=GT_PREVIEW_COLOR, preview_style=preview_style)


def _render_prediction_preview(
    image_path: Path,
    prediction: dict,
    target: Path,
    preview_style: str = "overlay",
) -> None:
    result = _first_prediction_result(prediction)
    value = result.get("value") if isinstance(result.get("value"), dict) else {}
    rle = value.get("rle")
    width = int(result.get("original_width") or 0)
    height = int(result.get("original_height") or 0)
    if not isinstance(rle, list) or width <= 0 or height <= 0:
        raise ValueError("prediction row lacks Label Studio RLE dimensions")
    mask = _decode_label_studio_rle(rle, width, height)
    _render_mask_preview(image_path, mask, target, color=MOBILESAM_PREVIEW_COLOR, preview_style=preview_style)


def _first_prediction_result(prediction: dict) -> dict:
    result = prediction.get("result")
    if isinstance(result, dict):
        return result
    pred = prediction.get("prediction") if isinstance(prediction.get("prediction"), dict) else {}
    results = pred.get("result") if isinstance(pred.get("result"), list) else []
    return results[0] if results and isinstance(results[0], dict) else {}


def _label_studio_prediction_payload(prediction: dict) -> dict | None:
    result = _first_prediction_result(prediction)
    value = result.get("value") if isinstance(result.get("value"), dict) else {}
    if not isinstance(value.get("rle"), list):
        return None
    preannotation_result = dict(result)
    preannotation_result["from_name"] = "mask_label"
    preannotation_result["to_name"] = "image"
    preannotation_result["type"] = "brushlabels"
    preannotation_result.setdefault("value", value)
    preannotation_result["value"] = dict(preannotation_result["value"])
    preannotation_result["value"].setdefault("brushlabels", ["Object"])
    pred = prediction.get("prediction") if isinstance(prediction.get("prediction"), dict) else {}
    score = pred.get("score", prediction.get("score"))
    payload = {
        "model_version": prediction.get("model_version") or pred.get("model_version") or "mobile_sam",
        "result": [preannotation_result],
    }
    if score is not None:
        payload["score"] = score
    return payload


def _decode_label_studio_rle(rle: list[int], width: int, height: int):
    import numpy as np

    try:
        from label_studio_converter.brush import decode_rle

        decoded = decode_rle(rle)
        arr = np.asarray(decoded, dtype=np.uint8)
        if arr.size == width * height * 4:
            rgba = arr.reshape((height, width, 4))
            return (rgba[:, :, 3] > 0).astype(np.uint8)
    except Exception:
        pass
    values: list[int] = []
    current = 0
    for count in rle:
        values.extend([current] * int(count))
        current = 1 - current
    total = width * height
    if len(values) < total:
        values.extend([0] * (total - len(values)))
    return np.asarray(values[:total], dtype=np.uint8).reshape((height, width))


def _render_mask_overlay(image_path: Path, mask: Any, target: Path, color: tuple[int, int, int, int]) -> None:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        base = image.convert("RGBA")
    mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_arr.shape != (base.height, base.width):
        mask_image = Image.fromarray(mask_arr * 255, mode="L").resize(base.size, resample=Image.NEAREST)
        mask_arr = (np.asarray(mask_image) > 0).astype(np.uint8)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = Image.fromarray((mask_arr * int(color[3])).astype(np.uint8), mode="L")
    overlay.paste(Image.new("RGBA", base.size, color), (0, 0), alpha)
    boundary = _mask_boundary(mask_arr)
    boundary_alpha = Image.fromarray((boundary * 255).astype(np.uint8), mode="L")
    overlay.paste(Image.new("RGBA", base.size, (color[0], color[1], color[2], 255)), (0, 0), boundary_alpha)
    Image.alpha_composite(base, overlay).convert("RGB").save(target)


def _mask_boundary(mask_arr: Any) -> Any:
    import numpy as np

    mask_bool = np.asarray(mask_arr).astype(bool)
    if not mask_bool.any():
        return np.zeros(mask_bool.shape, dtype=np.uint8)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return (mask_bool & ~eroded).astype(np.uint8)


def _render_mask_preview(
    image_path: Path,
    mask: Any,
    target: Path,
    color: tuple[int, int, int, int],
    preview_style: str,
) -> None:
    if preview_style == "mask":
        _render_mask_pixels(image_path, mask, target, color=color[:3])
        return
    if preview_style != "overlay":
        raise ValueError(f"unsupported preview_style={preview_style!r}")
    _render_mask_overlay(image_path, mask, target, color=color)


def _render_mask_pixels(image_path: Path, mask: Any, target: Path, color: tuple[int, int, int]) -> None:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        size = image.size
    mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_arr.shape != (size[1], size[0]):
        mask_image = Image.fromarray(mask_arr * 255, mode="L").resize(size, resample=Image.NEAREST)
        mask_arr = (np.asarray(mask_image) > 0).astype(np.uint8)
    out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    out[mask_arr > 0] = color
    Image.fromarray(out, mode="RGB").save(target)


def _safe_filename(value: Any) -> str:
    text = str(value or "item")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_") or "item"


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
        "repeat_review_overlap": False,
        "review_outcome_secondary": None,
        "reviewer_secondary": None,
        "reviewed_at_secondary": None,
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


def _mark_repeat_review_overlap(rows: list[dict], repeat_review_overlap: int) -> None:
    if repeat_review_overlap <= 0:
        return
    for group in GROUPS:
        count = 0
        for row in rows:
            if row.get("group") != group:
                continue
            if count < repeat_review_overlap:
                row["repeat_review_overlap"] = True
                count += 1


def _assert_group_counts(rows: list[dict], per_group: int) -> None:
    counts = {group: sum(row.get("group") == group for row in rows) for group in GROUPS}
    short = {group: count for group, count in counts.items() if count < per_group}
    if short:
        raise ValueError(f"fewer than {per_group} importable rows for groups: {short}")


def _label_studio_task(row: dict, pilot_id: str) -> dict:
    image = str(row.get("image")).strip()
    gt_reference = str(row.get("gt_reference") or image).strip()
    mobilesam_preview = str(row.get("mobilesam_preview") or image).strip()
    data = {
        "image": image,
        "gt_reference": gt_reference,
        "mobilesam_preview": mobilesam_preview,
        "gt_mask_preview": gt_reference,
        "mask_preview": mobilesam_preview,
        "dataset": row.get("dataset"),
        "category_name": row.get("category_name"),
        "review_group": row.get("group"),
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
    task = {"id": f"{pilot_id}:{row.get('group')}:{_dedupe_key(row)}", "data": data, "meta": meta}
    preannotation = row.get("_label_studio_preannotation")
    if isinstance(preannotation, dict):
        task["predictions"] = [preannotation]
    return task


def _assert_safe_payload(value: Any) -> None:
    for forbidden in _forbidden_hits(value):
        raise ValueError(f"forbidden field leaked into balanced pilot payload: {forbidden}")


def _forbidden_hits(value: Any) -> list[str]:
    hits: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key_text = str(key)
                if key_text in FORBIDDEN_REVIEW_FIELDS:
                    hits.add(key_text)
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return sorted(hits)


def _valid_label_studio_segmentation_task(data: dict[str, Any], meta: dict[str, Any]) -> bool:
    return (
        isinstance(data.get("image"), str)
        and bool(data["image"].strip())
        and isinstance(data.get("gt_reference"), str)
        and bool(data["gt_reference"].strip())
        and isinstance(data.get("mobilesam_preview"), str)
        and bool(data["mobilesam_preview"].strip())
        and isinstance(data.get("gt_mask_preview"), str)
        and bool(data["gt_mask_preview"].strip())
        and isinstance(data.get("mask_preview"), str)
        and bool(data["mask_preview"].strip())
        and meta.get("review_group") in GROUPS
        and meta.get("shadow_only") is True
        and meta.get("affects_default_ranking") is False
    )


def _file_ref_exists(value: Any, local_files_root: str | None = None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = _file_ref_to_path(value, local_files_root=local_files_root)
    if path is None:
        return True
    return path.exists()


def _file_ref_to_path(value: str, local_files_root: str | None = None) -> Path | None:
    value = value.strip()
    root = Path(local_files_root) if local_files_root else Path("demo_data/local-files")
    if value.startswith("/data/local-files/?"):
        parsed = urlparse(value)
        data = parse_qs(parsed.query).get("d", [""])[0]
        return root / unquote(data).lstrip("/")
    if value.startswith("/data/local-files/?d="):
        return root / unquote(value.split("?d=", 1)[1]).lstrip("/")
    if value.startswith("/label-studio/files/"):
        return root / value[len("/label-studio/files/") :].lstrip("/")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return None
    return Path(value)


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
        "repeat_review_overlap",
        "human_review_outcome",
        "reviewer",
        "reviewed_at",
        "review_outcome_secondary",
        "reviewer_secondary",
        "reviewed_at_secondary",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _guidelines(
    pilot_id: str = "segmentation_balanced_pilot_2026_06_04",
    validation_round: str = "round1_gt_assisted_calibration",
    repeat_review_overlap: int = 0,
) -> str:
    return "\n".join(
        [
            "# Segmentation Balanced Pilot Review Guidelines",
            "",
            f"Pilot: `{pilot_id}`",
            f"Validation round: `{validation_round}`",
            "",
            "Review each task using the GT reference, MobileSAM preview, and editable mask panes.",
            "",
            "Required outcome values:",
            "",
            "- No fix: MobileSAM mask is acceptable for the target object.",
            "- Minor fix: mask needs small boundary/detail edits.",
            "- Major fix: mask needs substantial correction but target is still recoverable.",
            "- Redo: mask is wrong enough that a fresh mask is the right action.",
            "- Skip: image, class, reference, or tooling is ambiguous/broken.",
            "",
            "Record `reviewer` and `reviewed_at` for every reviewed row.",
            f"Rows with `repeat_review_overlap=true` should receive a secondary independent review; planned overlap per group: {repeat_review_overlap}.",
            "",
            "Fields allowed for offline analysis: group, review outcome, reviewer, reviewed_at, repeat-review overlap, notes, safe ids, dataset/category, current/learned scores, ranks, rank delta, and prediction-time `pred_*` features.",
            "Fields forbidden as production scoring inputs: review outcomes, reviewer identity, reviewed_at, repeat-review agreement, notes, GT references, IoU/Dice, correction severity, labels, correction deltas, and any evaluation-only or GT-derived field.",
            "",
            "This packet is offline analysis only. Do not use these outcomes to change default Layer 5 weights, default rank, `priority_score`, `priority_bucket`, or sorting.",
        ]
    ) + "\n"


def _summary_template(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pilot_id": summary["pilot_id"],
        "validation_round": summary["validation_round"],
        "source_window_id": summary["source_window_id"],
        "planned_rows": summary["rows"],
        "per_group": summary["per_group"],
        "group_counts": summary["group_counts"],
        "review_outcomes": ["no_fix", "minor_fix", "major_fix", "redo", "skip"],
        "repeat_review_overlap_per_group": summary["repeat_review_overlap_per_group"],
        "metrics_to_fill_after_review": {
            group: {
                "total_reviewed": 0,
                "no_fix": 0,
                "minor_fix": 0,
                "major_fix": 0,
                "redo": 0,
                "skip": 0,
                "major_or_redo_rate": None,
            }
            for group in GROUPS
        },
        "offline_analysis_only": True,
        "shadow_only": True,
        "affects_default_ranking": False,
        "learned_boundary_shape_promoted": False,
        "production_scoring_inputs_allowed": [
            "prediction-time pred_* features only when captured before GT/review",
            "existing default Layer 5 fields that already feed production scoring",
        ],
        "production_scoring_inputs_forbidden": [
            "human_review_outcome",
            "reviewer",
            "reviewed_at",
            "repeat_review_overlap",
            "review_outcome_secondary",
            "GT references or masks",
            "IoU/Dice",
            "correction severity",
            "correction deltas",
            "evaluation_only fields",
        ],
    }


def _closure_report_template(summary: dict[str, Any]) -> str:
    lines = [
        "# Segmentation Balanced Pilot Round 2 Independent Validation Closure Report",
        "",
        f"Pilot: `{summary['pilot_id']}`",
        f"Validation round: `{summary['validation_round']}`",
        f"Source window/data mix: `{summary['source_window_id'] or 'fill after run'}`",
        "",
        "## Scope",
        "",
        "This is an independent GT-assisted calibration round for learned boundary/shape shadow scoring. It is not a production no-GT review result.",
        "",
        "## Planned Counts",
        "",
        "| Group | Planned | Reviewed | No fix | Minor fix | Major fix | Redo | Skip | Major/Redo rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        lines.append(f"| `{group}` | {summary['group_counts'][group]} | 0 | 0 | 0 | 0 | 0 | 0 | TBD |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Default Layer 5 weights unchanged.",
            "- Default rank, `priority_score`, `priority_bucket`, and sorting unchanged.",
            "- Learned boundary/shape remains shadow-only.",
            "- Human review outcomes are offline analysis only.",
            "- Cannot claim production efficiency lift from this GT-assisted calibration.",
            "",
            "## Decision",
            "",
            "Fill after review: replicate lift / mixed / not replicated / insufficient reviewed rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a 4-group balanced segmentation human review pilot.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=15)
    parser.add_argument("--pilot-id", default="segmentation_balanced_pilot_2026_06_04")
    parser.add_argument("--validation-round", default="round1_gt_assisted_calibration")
    parser.add_argument("--repeat-review-overlap", type=int, default=0)
    parser.add_argument("--source-window-id")
    parser.add_argument("--previous-assignment")
    parser.add_argument("--local-files-root")
    parser.add_argument("--local-files-subdir")
    parser.add_argument("--manifest-paths", nargs="*", default=[])
    parser.add_argument("--prediction-paths", nargs="*", default=[])
    parser.add_argument("--preview-style", choices=["overlay", "mask"], default="overlay")
    parser.add_argument("--gt-preview-style", choices=["overlay", "mask"], default=None)
    parser.add_argument("--prediction-preview-style", choices=["overlay", "mask"], default=None)
    parser.add_argument("--prediction-display-model-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_balanced_pilot_review(
        args.review_queue,
        args.output_dir,
        per_group=args.per_group,
        pilot_id=args.pilot_id,
        validation_round=args.validation_round,
        repeat_review_overlap=args.repeat_review_overlap,
        source_window_id=args.source_window_id,
        previous_assignment=args.previous_assignment,
        local_files_root=args.local_files_root,
            local_files_subdir=args.local_files_subdir,
            manifest_paths=args.manifest_paths,
            prediction_paths=args.prediction_paths,
            preview_style=args.preview_style,
            gt_preview_style=args.gt_preview_style,
            prediction_preview_style=args.prediction_preview_style,
            prediction_display_model_version=args.prediction_display_model_version,
        )
    print(f"[OK] balanced pilot rows={result['rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
