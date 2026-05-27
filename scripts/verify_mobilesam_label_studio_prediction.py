#!/usr/bin/env python3
"""Verify that Label Studio has a persisted MobileSAM segmentation prediction.
怎么确认 Label Studio 真连上 GPU MobileSAM”的问题
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_LABEL_STUDIO_URL = "http://127.0.0.1:18080"
DEFAULT_PROJECT_ID = 3
DEFAULT_BACKEND_URL = "http://ml-backend-gpu:9090"
DEFAULT_MODEL_VERSION = "mobilesam-seg-v0001"
DEFAULT_COMPOSE_DIR = "infra"


class VerificationError(RuntimeError):
    """Raised when verification fails."""


def run_command(args: List[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise VerificationError(f"command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise VerificationError(f"command failed ({' '.join(args)}): {message}") from exc
    return completed.stdout


def compose_cmd(compose_dir: Path, sql: str) -> List[str]:
    return [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "labelstudio",
        "-d",
        "labelstudio",
        "-P",
        "pager=off",
        "-t",
        "-A",
        "-c",
        sql,
    ]


def check_label_studio_health(base_url: str, timeout: float) -> str:
    for path in ("/health", "/api/version"):
        url = base_url.rstrip("/") + path
        try:
            with urlopen(url, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
                return f"{url} HTTP {response.status}"
        except URLError:
            continue
    raise VerificationError(f"Label Studio is not reachable at {base_url}")


def query_one_json(compose_dir: Path, sql: str) -> Dict[str, Any]:
    wrapped_sql = f"select coalesce(json_agg(row_to_json(q)), '[]'::json) from ({sql}) q;"
    output = run_command(compose_cmd(compose_dir, wrapped_sql), compose_dir).strip()
    try:
        rows = json.loads(output or "[]")
    except json.JSONDecodeError as exc:
        raise VerificationError(f"database query did not return JSON: {output}") from exc
    if not rows:
        return {}
    if not isinstance(rows[0], dict):
        raise VerificationError(f"database query returned unexpected JSON: {rows}")
    return rows[0]


def backend_row(compose_dir: Path, project_id: int) -> Dict[str, Any]:
    return query_one_json(
        compose_dir,
        f"""
        select id, project_id, url, title, is_interactive, created_at, updated_at
        from ml_mlbackend
        where project_id = {int(project_id)}
        order by updated_at desc, id desc
        limit 1
        """,
    )


def latest_prediction_row(compose_dir: Path, project_id: int, model_version: str) -> Dict[str, Any]:
    escaped_version = model_version.replace("'", "''")
    return query_one_json(
        compose_dir,
        f"""
        select
          p.id as prediction_id,
          p.task_id,
          p.model_version,
          p.score,
          p.created_at,
          p.result::text like '%brushlabels%' as has_brushlabels,
          p.result::text like '%rle%' as has_rle,
          p.result::text like '%choices%' as has_choices,
          p.result::text like '%mobilesam%' as has_mobilesam,
          p.result::text like '%backend%' as has_backend_meta,
          p.result::text like '%prompt_box%' as has_prompt_box
        from prediction p
        join task t on t.id = p.task_id
        where t.project_id = {int(project_id)}
          and p.model_version = '{escaped_version}'
        order by p.created_at desc, p.id desc
        limit 1
        """,
    )


def verify_prediction(row: Dict[str, Any], expected_model_version: str) -> None:
    if not row:
        raise VerificationError(f"no prediction found for model_version={expected_model_version}")
    failures = []
    if row.get("model_version") != expected_model_version:
        failures.append(f"model_version={row.get('model_version')!r}")
    for key in ("has_brushlabels", "has_rle", "has_prompt_box"):
        if row.get(key) is not True:
            failures.append(f"{key}={row.get(key)!r}")
    if row.get("has_choices") is not False:
        failures.append(f"has_choices={row.get('has_choices')!r}")
    if not (row.get("has_mobilesam") or row.get("has_backend_meta") or row.get("has_prompt_box")):
        failures.append("missing mobilesam/backend/prompt_box metadata")
    if failures:
        raise VerificationError("latest prediction failed validation: " + ", ".join(failures))


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    compose_dir = args.compose_dir.resolve()
    if not compose_dir.exists():
        raise VerificationError(f"compose directory not found: {compose_dir}")

    health = check_label_studio_health(args.label_studio_url, args.timeout)
    backend = backend_row(compose_dir, args.project_id)
    if not backend:
        raise VerificationError(f"no ML backend configured for project_id={args.project_id}")
    if backend.get("url") != args.expected_backend_url:
        raise VerificationError(
            f"project {args.project_id} backend URL is {backend.get('url')!r}, "
            f"expected {args.expected_backend_url!r}"
        )

    prediction = latest_prediction_row(compose_dir, args.project_id, args.expected_model_version)
    verify_prediction(prediction, args.expected_model_version)
    return {
        "label_studio_health": health,
        "project_id": args.project_id,
        "backend_url": backend.get("url"),
        "latest_prediction_id": prediction.get("prediction_id"),
        "task_id": prediction.get("task_id"),
        "model_version": prediction.get("model_version"),
        "has_brushlabels": prediction.get("has_brushlabels"),
        "has_rle": prediction.get("has_rle"),
        "has_choices": prediction.get("has_choices"),
        "has_mobilesam": prediction.get("has_mobilesam"),
        "has_backend_meta": prediction.get("has_backend_meta"),
        "has_prompt_box": prediction.get("has_prompt_box"),
    }


def print_report(report: Dict[str, Any]) -> None:
    for key, value in report.items():
        print(f"{key}: {value}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-studio-url", default=os.getenv("LABEL_STUDIO_URL", DEFAULT_LABEL_STUDIO_URL))
    parser.add_argument("--project-id", type=int, default=int(os.getenv("LABEL_STUDIO_PROJECT_ID", DEFAULT_PROJECT_ID)))
    parser.add_argument(
        "--expected-backend-url",
        default=os.getenv("EXPECTED_ML_BACKEND_URL", DEFAULT_BACKEND_URL),
    )
    parser.add_argument(
        "--expected-model-version",
        default=os.getenv("EXPECTED_MODEL_VERSION", DEFAULT_MODEL_VERSION),
    )
    parser.add_argument("--compose-dir", type=Path, default=Path(os.getenv("COMPOSE_DIR", DEFAULT_COMPOSE_DIR)))
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        print_report(build_report(args))
    except VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
