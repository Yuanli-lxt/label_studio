#!/usr/bin/env python3
"""Create deterministic non-eval image review tasks for HITL smoke testing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_PATH = ROOT / "demo_data" / "tasks" / "image_classification_labeled_export.json"
DEFAULT_EVAL_MANIFEST_PATH = ROOT / "demo_data" / "tasks" / "image_classification_eval_manifest.json"
DEFAULT_OUT_PATH = ROOT / "demo_data" / "tasks" / "image_classification_non_eval_review_tasks.json"
DEFAULT_TASK_COUNT = 4


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _filename_from_image_ref(image_ref: str) -> str | None:
    if not isinstance(image_ref, str) or not image_ref.strip():
        return None
    raw = image_ref.strip()
    if "?d=" in raw:
        raw = raw.split("?d=", 1)[1]
    if "/images/" in raw:
        raw = "images/" + raw.split("/images/", 1)[1]
    return os.path.basename(raw)


def load_eval_filenames(path: Path) -> set[str]:
    manifest = _read_json(path)
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError(f"eval manifest has no images[]: {path}")
    filenames = set()
    for row in images:
        if not isinstance(row, dict):
            continue
        filename = row.get("filename")
        if isinstance(filename, str) and filename.strip():
            filenames.add(filename.strip())
    if not filenames:
        raise ValueError(f"eval manifest yielded no filenames: {path}")
    return filenames


def _extract_label(task: dict) -> str | None:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return None
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result") or []:
            if not isinstance(result, dict):
                continue
            choices = (result.get("value") or {}).get("choices") if isinstance(result.get("value"), dict) else None
            if isinstance(choices, list) and choices:
                label = choices[0]
                if label in {"Product", "Other"}:
                    return label
    return None


def build_non_eval_review_tasks(
    export_path: Path = DEFAULT_EXPORT_PATH,
    eval_manifest_path: Path = DEFAULT_EVAL_MANIFEST_PATH,
    task_count: int = DEFAULT_TASK_COUNT,
) -> tuple[list[dict], dict]:
    export = _read_json(export_path)
    if not isinstance(export, list):
        raise ValueError(f"labeled export must be a JSON list: {export_path}")
    eval_filenames = load_eval_filenames(eval_manifest_path)

    candidates = []
    eval_filtered = 0
    invalid = 0
    for task in export:
        if not isinstance(task, dict):
            invalid += 1
            continue
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
        image = data.get("image")
        filename = _filename_from_image_ref(image)
        label = _extract_label(task)
        if not filename or not isinstance(image, str) or label not in {"Product", "Other"}:
            invalid += 1
            continue
        if filename in eval_filenames or meta.get("dataset_split") == "eval":
            eval_filtered += 1
            continue
        candidates.append(
            {
                "filename": filename,
                "image": image,
                "label": label,
                "caption": data.get("caption") or f"Non-eval review sample: {filename}",
                "source_task_id": task.get("id"),
            }
        )

    if len(candidates) < task_count:
        raise ValueError(
            f"not enough non-eval image candidates: requested {task_count}, found {len(candidates)}, "
            f"eval_filtered={eval_filtered}, invalid={invalid}"
        )

    # Deterministic mix of labels for a concise manual-review pool.
    products = [row for row in candidates if row["label"] == "Product"]
    others = [row for row in candidates if row["label"] == "Other"]
    selected = []
    while len(selected) < task_count and (products or others):
        if products and len(selected) < task_count:
            selected.append(products.pop(0))
        if others and len(selected) < task_count:
            selected.append(others.pop(0))

    review_tasks = []
    for idx, row in enumerate(selected, start=1):
        review_tasks.append(
            {
                "id": f"image-non-eval-review-{idx:03d}",
                "data": {
                    "image": row["image"],
                    "caption": (
                        f"Non-eval review {row['filename']} | source_label={row['label']} | "
                        "purpose=verify candidate enters retrain"
                    ),
                },
                "meta": {
                    "review_reason": "non_eval_training_candidate_review",
                    "filename": row["filename"],
                    "source_label": row["label"],
                    "predicted_label": row["label"],
                    "source_task_id": row["source_task_id"],
                    "source": "image_classification_labeled_export_train_pool",
                    "eval_filtered": False,
                },
            }
        )

    summary = {
        "total_export_tasks": len(export),
        "total_candidates": len(candidates),
        "eval_filtered": eval_filtered,
        "invalid_skipped": invalid,
        "written_tasks": len(review_tasks),
    }
    return review_tasks, summary


def write_non_eval_review_tasks(out_path: Path, tasks: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-path", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--eval-manifest-path", type=Path, default=DEFAULT_EVAL_MANIFEST_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks, summary = build_non_eval_review_tasks(args.export_path, args.eval_manifest_path, args.task_count)
    write_non_eval_review_tasks(args.out_path, tasks)
    print("Created non-eval image review tasks")
    print(f"total_candidates={summary['total_candidates']}")
    print(f"eval_filtered={summary['eval_filtered']}")
    print(f"invalid_skipped={summary['invalid_skipped']}")
    print(f"written_tasks={summary['written_tasks']}")
    print(f"output={args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
