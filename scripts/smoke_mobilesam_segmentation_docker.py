#!/usr/bin/env python3
"""Smoke-test the Docker MobileSAM segmentation path through Label Studio."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug_label_studio_rle_overlay import (  # noqa: E402
    RLEOverlayError,
    build_report as build_overlay_report,
)
from scripts.verify_mobilesam_label_studio_prediction import (  # noqa: E402
    DEFAULT_BACKEND_URL,
    DEFAULT_COMPOSE_DIR,
    DEFAULT_LABEL_STUDIO_URL,
    DEFAULT_MODEL_VERSION,
    DEFAULT_PROJECT_ID,
    VerificationError,
    backend_row,
    query_one_json,
    run_command,
)


DEFAULT_HOST_ML_BACKEND_URLS = ("http://127.0.0.1:9092", "http://127.0.0.1:19092")
CHECKPOINT_PATH = "/app/models/mobilesam/mobile_sam.pt"
MODEL_STATE_PATH = "/app/demo_data/model_state/current_image_segmentation_model.json"
REQUIRED_RUNNING_SERVICES = ("postgres", "redis", "label-studio", "ml-backend-gpu")


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


class SmokeError(RuntimeError):
    """Raised when a smoke-test check cannot pass."""


def normalize_host_backend_urls(explicit_url: Optional[str]) -> List[str]:
    urls: List[str] = []
    if explicit_url:
        urls.append(explicit_url)
    for url in DEFAULT_HOST_ML_BACKEND_URLS:
        if url not in urls:
            urls.append(url)
    return urls


def area_ratio_sanity(area_ratio: float, min_ratio: float = 0.0005, max_ratio: float = 0.95) -> Tuple[bool, str]:
    if area_ratio <= 0:
        return False, "mask is empty"
    if area_ratio < min_ratio:
        return False, f"area ratio {area_ratio:.6f} is too close to 0"
    if area_ratio > max_ratio:
        return False, f"area ratio {area_ratio:.6f} is too close to 1"
    return True, f"area ratio {area_ratio:.6f} is within sanity bounds"


def bbox_intersects(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> bool:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return False
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    return max(ax1, bx1) <= min(ax2, bx2) and max(ay1, by1) <= min(ay2, by2)


def aggregate_pass(results: Iterable[StepResult]) -> bool:
    return all(result.ok for result in results)


def _walk_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _iter_prediction_results(result: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                yield item
    elif isinstance(result, dict):
        nested = result.get("result")
        if isinstance(nested, list):
            yield from _iter_prediction_results(nested)


def summarize_prediction_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    result = row.get("result")
    items = list(_iter_prediction_results(result))
    text_values = [str(value).lower() for value in _walk_values(result)]
    has_mobilesam = any("mobilesam" in value or "mobile-sam" in value for value in text_values)
    has_backend_meta = any(
        isinstance(item.get("meta"), dict) and bool(item["meta"].get("backend")) for item in items
    )
    has_prompt_box = any(
        isinstance(value, dict) and "prompt_box" in value and bool(value.get("prompt_box"))
        for value in _walk_values(result)
    )
    has_mask_quality = any(
        isinstance(value, dict) and _valid_mask_quality_schema(value.get("mask_quality"))
        for value in _walk_values(result)
    )
    has_review = any(
        isinstance(value, dict) and _valid_review_schema(value.get("review"))
        for value in _walk_values(result)
    )
    has_uncertainty = any(
        isinstance(value, dict) and _valid_uncertainty_schema(value.get("uncertainty"))
        for value in _walk_values(result)
    )
    return {
        "has_brushlabels": any(item.get("type") == "brushlabels" for item in items),
        "has_rle": any(
            isinstance(item.get("value"), dict)
            and item["value"].get("format") == "rle"
            and isinstance(item["value"].get("rle"), list)
            and len(item["value"].get("rle", [])) > 0
            for item in items
        ),
        "has_choices": any(item.get("type") == "choices" for item in items),
        "has_mobilesam": has_mobilesam,
        "has_backend_meta": has_backend_meta,
        "has_prompt_box": has_prompt_box,
        "has_mask_quality": has_mask_quality,
        "has_review": has_review,
        "has_uncertainty": has_uncertainty,
    }


def _valid_mask_quality_schema(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = ("valid_mask", "mask_area_px", "mask_area_ratio", "image_width", "image_height")
    if any(key not in value for key in required):
        return False
    if not isinstance(value.get("valid_mask"), bool):
        return False
    if not isinstance(value.get("mask_area_px"), int):
        return False
    if not isinstance(value.get("mask_area_ratio"), (int, float)):
        return False
    if not isinstance(value.get("image_width"), int) or not isinstance(value.get("image_height"), int):
        return False
    if value["image_width"] > 0 and value["image_height"] > 0:
        return 0 <= float(value["mask_area_ratio"]) <= 1
    return True


def _valid_review_schema(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("needs_review"), bool):
        return False
    if value.get("review_priority") not in {"low", "medium", "high"}:
        return False
    if not isinstance(value.get("review_priority_score"), (int, float)):
        return False
    return isinstance(value.get("review_reason"), list)


def _valid_uncertainty_schema(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("method") != "prompt_stability":
        return False
    if value.get("enabled") is not True:
        return False
    if not isinstance(value.get("stable"), bool):
        return False
    if value.get("stability_bucket") not in {"high", "medium", "low", "unknown"}:
        return False
    return isinstance(value.get("reason"), list)


def validate_prediction_summary(
    row: Dict[str, Any],
    expected_model_version: str,
    require_uncertainty: bool = False,
) -> List[str]:
    failures: List[str] = []
    if not row:
        return [f"no prediction found for model_version={expected_model_version}"]
    if row.get("model_version") != expected_model_version:
        failures.append(f"model_version={row.get('model_version')!r}")
    flags = summarize_prediction_flags(row)
    for key in (
        "has_brushlabels",
        "has_rle",
        "has_mobilesam",
        "has_backend_meta",
        "has_prompt_box",
        "has_mask_quality",
        "has_review",
    ):
        if flags.get(key) is not True:
            failures.append(f"{key}={flags.get(key)!r}")
    if flags.get("has_choices") is not False:
        failures.append(f"has_choices={flags.get('has_choices')!r}")
    if require_uncertainty and flags.get("has_uncertainty") is not True:
        failures.append("has_uncertainty=False")
    return failures


def docker_compose_args(compose_dir: Path, *args: str) -> List[str]:
    return ["docker", "compose", *args]


def run_capture(args: Sequence[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> str:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as exc:
        raise SmokeError(f"command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip()
        raise SmokeError(f"command failed ({' '.join(args)}): {message}") from exc
    return completed.stdout


def check_backend_health(urls: Sequence[str], timeout: float) -> Tuple[str, str]:
    errors = []
    for base_url in urls:
        url = base_url.rstrip("/") + "/health"
        try:
            with urlopen(url, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
                if response.status == 200:
                    return base_url, body
                errors.append(f"{url}: HTTP {response.status}")
        except URLError as exc:
            errors.append(f"{url}: {exc}")
        except TimeoutError as exc:
            errors.append(f"{url}: {exc}")
    raise SmokeError("ml-backend-gpu health failed; tried " + "; ".join(errors))


def running_services(compose_dir: Path) -> List[str]:
    output = run_capture(docker_compose_args(compose_dir, "ps", "--services", "--filter", "status=running"), compose_dir)
    return [line.strip() for line in output.splitlines() if line.strip()]


def check_container_file(compose_dir: Path, service: str, path: str) -> None:
    run_capture(docker_compose_args(compose_dir, "exec", "-T", service, "test", "-f", path), compose_dir)


def read_container_file(compose_dir: Path, service: str, path: str) -> str:
    return run_capture(docker_compose_args(compose_dir, "exec", "-T", service, "cat", path), compose_dir)


def find_model_version(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        direct = payload.get("model_version")
        if isinstance(direct, str):
            return direct
        for value in payload.values():
            found = find_model_version(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_model_version(value)
            if found:
                return found
    return None


def update_backend_url(compose_dir: Path, backend_id: int, expected_url: str) -> Dict[str, Any]:
    escaped_url = expected_url.replace("'", "''")
    return query_one_json(
        compose_dir,
        f"""
        update ml_mlbackend
        set url = '{escaped_url}', updated_at = now()
        where id = {int(backend_id)}
        returning id, project_id, url, updated_at
        """,
    )


def task_count(compose_dir: Path, project_id: int) -> int:
    row = query_one_json(
        compose_dir,
        f"select count(*)::int as count from task where project_id = {int(project_id)}",
    )
    return int(row.get("count") or 0)


def latest_prediction_row(
    compose_dir: Path,
    project_id: int,
    expected_model_version: str,
    task_id: Optional[int] = None,
    prediction_id: Optional[int] = None,
) -> Dict[str, Any]:
    filters = [f"t.project_id = {int(project_id)}"]
    if prediction_id is not None:
        filters.append(f"p.id = {int(prediction_id)}")
    else:
        escaped_version = expected_model_version.replace("'", "''")
        filters.append(f"p.model_version = '{escaped_version}'")
    if task_id is not None:
        filters.append(f"p.task_id = {int(task_id)}")
    where = " and ".join(filters)
    return query_one_json(
        compose_dir,
        f"""
        select
          p.id as prediction_id,
          p.task_id,
          p.model_version,
          p.score,
          p.created_at,
          p.result,
          t.data as task_data,
          t.meta as task_meta
        from prediction p
        join task t on t.id = p.task_id
        where {where}
        order by p.created_at desc, p.id desc
        limit 1
        """,
    )


def local_image_path(task_data: Dict[str, Any], repo_root: Path) -> Optional[Path]:
    image = task_data.get("image") if isinstance(task_data, dict) else None
    if not isinstance(image, str) or not image:
        return None
    parsed = urlparse(image)
    query = parse_qs(parsed.query)
    if "d" in query and query["d"]:
        return repo_root / "demo_data" / "local-files" / unquote(query["d"][0]).lstrip("/")
    if image.startswith("/data/local-files/?d="):
        return repo_root / "demo_data" / "local-files" / unquote(image.split("?d=", 1)[1]).lstrip("/")
    for prefix in ("/label-studio/files/", "/app/demo_data/local-files/", "/demo-local-files/"):
        if image.startswith(prefix):
            return repo_root / "demo_data" / "local-files" / image[len(prefix) :].lstrip("/")
    path = Path(image)
    if path.is_absolute() and path.exists():
        return path
    candidate = repo_root / image
    return candidate if candidate.exists() else None


def _bbox_from_mapping(raw: Dict[str, Any], width: Optional[int], height: Optional[int]) -> Optional[List[float]]:
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(raw):
        values = [raw["x_min"], raw["y_min"], raw["x_max"], raw["y_max"]]
    elif {"x", "y", "width", "height"}.issubset(raw):
        x = float(raw["x"])
        y = float(raw["y"])
        values = [x, y, x + float(raw["width"]), y + float(raw["height"])]
    else:
        return None
    values = [float(value) for value in values]
    unit = str(raw.get("unit") or raw.get("units") or raw.get("coordinate_system") or "").lower()
    if unit in {"percent", "percentage", "pct", "%"} and width and height:
        return [values[0] * width / 100.0, values[1] * height / 100.0, values[2] * width / 100.0, values[3] * height / 100.0]
    return values


def normalize_prompt_bbox(raw: Any, width: Optional[int] = None, height: Optional[int] = None) -> Optional[List[float]]:
    if isinstance(raw, dict):
        return _bbox_from_mapping(raw, width, height)
    if isinstance(raw, list) and len(raw) == 4:
        try:
            return [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
    return None


def task_prompt_bbox(task_data: Dict[str, Any], task_meta: Dict[str, Any], width: Optional[int], height: Optional[int]) -> Optional[List[float]]:
    for container in (task_data, task_meta):
        if not isinstance(container, dict):
            continue
        for key in ("bbox", "box"):
            bbox = normalize_prompt_bbox(container.get(key), width, height)
            if bbox:
                return bbox
    return None


def write_prediction_result(row: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mobilesam_prediction_{row['prediction_id']}_result.json"
    path.write_text(json.dumps(row.get("result", []), indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_optional_import(args: argparse.Namespace, repo_root: Path) -> str:
    env = os.environ.copy()
    env.setdefault("LABEL_STUDIO_URL", args.label_studio_url)
    output = run_capture(
        [
            sys.executable,
            str(repo_root / "scripts" / "import_image_segmentation_review_tasks_to_label_studio.py"),
            "--project-id",
            str(args.project_id),
        ],
        repo_root,
        env=env,
    )
    return output.strip()


def print_step(result: StepResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    suffix = f" - {result.detail}" if result.detail else ""
    print(f"[{status}] {result.name}{suffix}")


def add_step(results: List[StepResult], name: str, ok: bool, detail: str = "") -> None:
    result = StepResult(name, ok, detail)
    results.append(result)
    print_step(result)


def run_smoke(args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    results: List[StepResult] = []
    report: Dict[str, Any] = {
        "project_id": args.project_id,
        "expected_backend_url": args.expected_backend_url,
        "expected_model_version": args.expected_model_version,
        "host_backend_urls_tried": normalize_host_backend_urls(args.host_ml_backend_url),
    }
    compose_dir = args.compose_dir.resolve()
    out_dir = args.out_dir.resolve()

    try:
        run_capture(["docker", "--version"], ROOT)
        run_capture(["docker", "compose", "version"], ROOT)
        add_step(results, "Docker and docker compose available", True)
    except SmokeError as exc:
        add_step(results, "Docker and docker compose available", False, str(exc))

    try:
        run_capture(docker_compose_args(compose_dir, "config"), compose_dir)
        add_step(results, "docker compose config", True)
    except SmokeError as exc:
        add_step(results, "docker compose config", False, str(exc))

    try:
        running = set(running_services(compose_dir))
        missing = [service for service in REQUIRED_RUNNING_SERVICES if service not in running]
        if missing:
            raise SmokeError(f"missing running services: {', '.join(missing)}")
        add_step(results, "required compose services running", True, ", ".join(REQUIRED_RUNNING_SERVICES))
    except SmokeError as exc:
        add_step(results, "required compose services running", False, str(exc))

    try:
        selected_url, health_body = check_backend_health(report["host_backend_urls_tried"], args.timeout)
        report["host_backend_url"] = selected_url
        report["ml_backend_health"] = health_body
        add_step(results, "ml-backend-gpu host health", True, selected_url)
    except SmokeError as exc:
        add_step(results, "ml-backend-gpu host health", False, str(exc))

    try:
        check_container_file(compose_dir, "ml-backend-gpu", CHECKPOINT_PATH)
        add_step(results, "MobileSAM checkpoint exists in container", True, CHECKPOINT_PATH)
    except SmokeError as exc:
        add_step(results, "MobileSAM checkpoint exists in container", False, str(exc))

    try:
        check_container_file(compose_dir, "ml-backend-gpu", MODEL_STATE_PATH)
        state = json.loads(read_container_file(compose_dir, "ml-backend-gpu", MODEL_STATE_PATH))
        state_version = find_model_version(state)
        report["model_state_version"] = state_version
        if state_version != args.expected_model_version:
            raise SmokeError(f"model_version={state_version!r}, expected {args.expected_model_version!r}")
        add_step(results, "GPU model state model_version", True, state_version or "")
    except (SmokeError, json.JSONDecodeError) as exc:
        add_step(results, "GPU model state model_version", False, str(exc))

    try:
        backend = backend_row(compose_dir, args.project_id)
        report["project_backend_url"] = backend.get("url")
        report["project_backend_id"] = backend.get("id")
        if not backend:
            raise SmokeError(f"no ML backend configured for project_id={args.project_id}")
        if backend.get("url") != args.expected_backend_url:
            detail = f"actual {backend.get('url')!r}, expected {args.expected_backend_url!r}"
            if args.fix_backend_url:
                fixed = update_backend_url(compose_dir, int(backend["id"]), args.expected_backend_url)
                report["project_backend_url"] = fixed.get("url")
                detail = f"fixed backend id {fixed.get('id')} to {fixed.get('url')}"
            else:
                detail += (
                    "; rerun with --fix-backend-url to update ml_mlbackend.url explicitly"
                )
                raise SmokeError(detail)
        add_step(results, "Label Studio project backend URL", True, report["project_backend_url"] or "")
    except (SmokeError, VerificationError) as exc:
        add_step(results, "Label Studio project backend URL", False, str(exc))

    try:
        count = task_count(compose_dir, args.project_id)
        report["task_count"] = count
        if count <= 0:
            raise SmokeError(
                "no tasks found; run scripts/bootstrap_label_studio_image_segmentation_review.py "
                "and scripts/import_image_segmentation_review_tasks_to_label_studio.py"
            )
        add_step(results, "Label Studio demo tasks exist", True, f"{count} task(s)")
    except (SmokeError, VerificationError) as exc:
        add_step(results, "Label Studio demo tasks exist", False, str(exc))

    if args.import_demo_tasks:
        try:
            output = run_optional_import(args, ROOT)
            add_step(results, "optional demo task import", True, output.splitlines()[-1] if output else "done")
        except SmokeError as exc:
            add_step(results, "optional demo task import", False, str(exc))

    if args.trigger_prediction:
        add_step(
            results,
            "optional prediction trigger",
            True,
            "automatic Label Studio prediction trigger is intentionally not implemented; verifying latest persisted prediction",
        )

    prediction: Dict[str, Any] = {}
    try:
        prediction = latest_prediction_row(
            compose_dir,
            args.project_id,
            args.expected_model_version,
            task_id=args.task_id,
            prediction_id=args.prediction_id,
        )
        failures = validate_prediction_summary(
            prediction,
            args.expected_model_version,
            require_uncertainty=args.enable_prompt_stability,
        )
        if failures:
            raise SmokeError("; ".join(failures))
        flags = summarize_prediction_flags(prediction)
        report.update(
            {
                "latest_prediction_id": prediction.get("prediction_id"),
                "task_id": prediction.get("task_id"),
                "model_version": prediction.get("model_version"),
                **flags,
            }
        )
        add_step(
            results,
            "latest MobileSAM prediction fields",
            True,
            f"prediction_id={prediction.get('prediction_id')} task_id={prediction.get('task_id')}",
        )
    except (SmokeError, VerificationError) as exc:
        add_step(results, "latest MobileSAM prediction fields", False, str(exc))

    result_path: Optional[Path] = None
    if prediction:
        try:
            result_path = write_prediction_result(prediction, out_dir)
            report["prediction_result_json"] = str(result_path)
            add_step(results, "export latest prediction result", True, str(result_path))
        except OSError as exc:
            add_step(results, "export latest prediction result", False, str(exc))

    if prediction and not args.skip_overlay and result_path:
        try:
            image_path = local_image_path(prediction.get("task_data") or {}, ROOT)
            if not image_path:
                raise SmokeError("could not resolve task data.image to a local demo_data/local-files path")
            out_prefix = out_dir / f"mobilesam_prediction_{prediction['prediction_id']}"
            overlay = build_overlay_report(result_path, image_path, out_prefix)
            report["mask_nonzero"] = overlay["nonzero"]
            report["mask_bbox"] = overlay["bbox"]
            report["area_ratio"] = overlay["area_ratio"]
            report["mask_png"] = overlay["mask_png"]
            report["overlay_png"] = overlay["overlay_png"]
            ok, detail = area_ratio_sanity(float(overlay["area_ratio"]))
            if not ok:
                raise SmokeError(detail)
            prompt_bbox = task_prompt_bbox(
                prediction.get("task_data") or {},
                prediction.get("task_meta") or {},
                int(overlay["original_width"]),
                int(overlay["original_height"]),
            )
            report["prompt_bbox"] = prompt_bbox
            if prompt_bbox and not bbox_intersects(overlay["bbox"], prompt_bbox):
                raise SmokeError(f"mask bbox {overlay['bbox']} does not intersect prompt bbox {prompt_bbox}")
            add_step(
                results,
                "RLE overlay and mask sanity",
                True,
                f"bbox={overlay['bbox']} area_ratio={overlay['area_ratio']:.6f}",
            )
        except (SmokeError, RLEOverlayError, OSError) as exc:
            add_step(results, "RLE overlay and mask sanity", False, str(exc))
    elif args.skip_overlay:
        add_step(results, "RLE overlay and mask sanity", True, "skipped by --skip-overlay")

    report["steps"] = [result.__dict__ for result in results]
    return aggregate_pass(results), report


def print_report(report: Dict[str, Any], passed: bool) -> None:
    print("")
    print("Smoke report:")
    for key in (
        "project_id",
        "host_backend_url",
        "project_backend_url",
        "model_state_version",
        "latest_prediction_id",
        "task_id",
        "model_version",
        "has_brushlabels",
        "has_rle",
        "has_choices",
        "has_mobilesam",
        "has_backend_meta",
        "has_prompt_box",
        "has_mask_quality",
        "has_review",
        "has_uncertainty",
        "prediction_result_json",
        "mask_nonzero",
        "mask_bbox",
        "prompt_bbox",
        "area_ratio",
        "mask_png",
        "overlay_png",
    ):
        if key in report:
            print(f"{key}: {report[key]}")
    print(f"result: {'PASS' if passed else 'FAIL'}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-dir", type=Path, default=Path(os.getenv("COMPOSE_DIR", DEFAULT_COMPOSE_DIR)))
    parser.add_argument("--project-id", type=int, default=int(os.getenv("LABEL_STUDIO_PROJECT_ID", DEFAULT_PROJECT_ID)))
    parser.add_argument("--expected-backend-url", default=os.getenv("EXPECTED_ML_BACKEND_URL", DEFAULT_BACKEND_URL))
    parser.add_argument("--expected-model-version", default=os.getenv("EXPECTED_MODEL_VERSION", DEFAULT_MODEL_VERSION))
    parser.add_argument("--host-ml-backend-url", default=os.getenv("HOST_ML_BACKEND_URL"))
    parser.add_argument("--label-studio-url", default=os.getenv("LABEL_STUDIO_URL", DEFAULT_LABEL_STUDIO_URL))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--prediction-id", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--fix-backend-url", action="store_true")
    parser.add_argument("--trigger-prediction", action="store_true")
    parser.add_argument("--import-demo-tasks", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument(
        "--enable-prompt-stability",
        action="store_true",
        help="require prompt-stability uncertainty metadata on the latest persisted prediction",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    passed, report = run_smoke(args)
    print_report(report, passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
