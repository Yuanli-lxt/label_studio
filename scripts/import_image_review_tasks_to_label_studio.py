#!/usr/bin/env python3
"""Import image-classification review tasks into a Label Studio project, with dedupe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.label_studio_client import (  # noqa: E402
    LabelStudioAPIError,
    LabelStudioClient,
    LabelStudioConfigError,
    LabelStudioSettings,
    project_data_url,
    validate_review_task,
)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def load_review_tasks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"review tasks file not found: {path}. Run scripts/export_image_review_tasks_from_metadata.sh first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"review tasks file is not valid JSON: {path}: {exc}") from exc
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return [item for item in payload["tasks"] if isinstance(item, dict)]
    raise ValueError(f"review tasks file must be a JSON list or object with tasks[]: {path}")


def validate_tasks_for_summary(tasks: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    valid = []
    invalid_reasons = []
    for idx, task in enumerate(tasks, start=1):
        normalized, reason = validate_review_task(task)
        if reason:
            if len(invalid_reasons) < 10:
                invalid_reasons.append(f"task[{idx}]: {reason}")
            continue
        assert normalized is not None
        valid.append(normalized)
    return valid, invalid_reasons


def resolve_project_id(
    client: LabelStudioClient,
    project_id: Optional[int],
    project_title: Optional[str],
) -> int:
    if project_id is not None:
        return int(project_id)
    title = (project_title or client.settings.image_review_project_title).strip()
    project = client.find_project_by_title(title)
    if not project:
        raise RuntimeError(
            f"Label Studio project not found by title {title!r}. Run scripts/bootstrap_label_studio_image_review.py first."
        )
    return int(project["id"])


def import_review_tasks(
    client: LabelStudioClient,
    tasks_path: Path,
    project_id: Optional[int] = None,
    project_title: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    tasks = load_review_tasks(tasks_path)
    resolved_project_id = resolve_project_id(client, project_id, project_title)
    valid_tasks, validation_reasons = validate_tasks_for_summary(tasks)
    result = client.import_tasks_dedup(resolved_project_id, valid_tasks, dry_run=dry_run)
    result["source_file"] = str(tasks_path)
    result["total_read"] = len(tasks)
    result["pre_validation_invalid"] = len(tasks) - len(valid_tasks)
    result["invalid_reasons"] = validation_reasons + result.get("invalid_reasons", [])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-path", type=Path, default=None, help="review task JSON path")
    parser.add_argument("--project-id", type=int, default=None, help="Label Studio project id")
    parser.add_argument("--project-title", default=None, help="Label Studio project title")
    parser.add_argument("--dry-run", action="store_true", help="show import plan without importing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = LabelStudioSettings.from_env(require_token=True)
        tasks_path = args.tasks_path or settings.image_review_tasks_path
        client = LabelStudioClient(settings)
        info(f"Label Studio URL: {settings.url}")
        info(f"tasks path: {tasks_path}")
        info(f"project selector: id={args.project_id} title={args.project_title or settings.image_review_project_title}")
        info(f"dry_run: {args.dry_run}")
        client.health_check()
        result = import_review_tasks(
            client,
            Path(tasks_path),
            project_id=args.project_id,
            project_title=args.project_title,
            dry_run=args.dry_run,
        )
    except LabelStudioConfigError as exc:
        error(str(exc))
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        error(str(exc))
        return 1
    except LabelStudioAPIError as exc:
        error(str(exc))
        return 1

    if result.get("invalid_reasons"):
        warn("invalid task summary: " + "; ".join(result["invalid_reasons"][:10]))

    ok("Image review task import summary")
    print(f"project_id={result['project_id']}")
    print(f"source_file={result['source_file']}")
    print(f"total_tasks_read={result['total_read']}")
    print(f"planned_import={result['planned_import']}")
    print(f"imported_count={result['imported']}")
    print(f"skipped_duplicate={result['skipped_duplicate']}")
    print(f"skipped_invalid={result['pre_validation_invalid'] + result['skipped_invalid']}")
    print(f"dry_run={result['dry_run']}")
    print(f"project_url={project_data_url(settings.url, result['project_id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
