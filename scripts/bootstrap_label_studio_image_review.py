#!/usr/bin/env python3
"""Bootstrap the Label Studio image-classification human review project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.label_studio_client import (  # noqa: E402
    LabelStudioAPIError,
    LabelStudioClient,
    LabelStudioConfigError,
    LabelStudioSettings,
    project_data_url,
)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def read_label_config(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"label config not found: {path}. Set IMAGE_LABEL_CONFIG_PATH or restore label_configs/image_classification.xml"
        )
    return path.read_text(encoding="utf-8")


def bootstrap_image_review(
    client: LabelStudioClient,
    settings: LabelStudioSettings,
    skip_ml_backend: bool = False,
    skip_webhook: bool = False,
) -> dict:
    label_config = read_label_config(settings.image_label_config_path)

    info(f"Label Studio URL: {settings.url}")
    info(f"project title: {settings.image_review_project_title}")
    info(f"label config path: {settings.image_label_config_path}")
    info(f"ML backend URL: {settings.ml_backend_url}")
    info(f"trainer webhook URL: {settings.trainer_webhook_url}")

    info("checking Label Studio API access")
    client.health_check()
    ok("Label Studio API is reachable")

    project, project_status = client.ensure_project(
        settings.image_review_project_title,
        settings.image_review_project_description,
        label_config,
    )
    project_id = int(project["id"])
    ok(f"project {project_status}: id={project_id}")

    ml_status = "skipped"
    if skip_ml_backend or settings.skip_ml_backend_setup:
        warn("ML backend setup skipped")
    else:
        try:
            _backend, ml_status = client.ensure_ml_backend(project_id, settings.ml_backend_url)
            ok(f"ML backend {ml_status}: {settings.ml_backend_url}")
        except Exception as exc:  # Label Studio OSS endpoint can vary by version.
            ml_status = f"warning: {exc}"
            warn(
                "ML backend setup did not complete. Project is still ready; connect manually in "
                f"Project Settings -> Model with URL {settings.ml_backend_url}. Details: {exc}"
            )

    webhook_status = "skipped"
    if skip_webhook or settings.skip_webhook_setup:
        warn("webhook setup skipped")
    else:
        try:
            _webhook, webhook_status = client.ensure_webhook(project_id, settings.trainer_webhook_url)
            ok(f"webhook {webhook_status}: {settings.trainer_webhook_url}")
        except Exception as exc:
            webhook_status = f"warning: {exc}"
            warn(
                "webhook setup did not complete. Configure manually in project webhooks with URL "
                f"{settings.trainer_webhook_url}. Details: {exc}"
            )

    return {
        "project_id": project_id,
        "project_title": settings.image_review_project_title,
        "project_url": project_data_url(settings.url, project_id),
        "ml_backend_status": ml_status,
        "webhook_status": webhook_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ml-backend", action="store_true", help="skip Label Studio ML backend setup")
    parser.add_argument("--skip-webhook", action="store_true", help="skip Label Studio webhook setup")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = LabelStudioSettings.from_env(require_token=True)
        client = LabelStudioClient(settings)
        result = bootstrap_image_review(
            client,
            settings,
            skip_ml_backend=args.skip_ml_backend,
            skip_webhook=args.skip_webhook,
        )
    except LabelStudioConfigError as exc:
        error(str(exc))
        return 2
    except FileNotFoundError as exc:
        error(str(exc))
        return 2
    except LabelStudioAPIError as exc:
        error(str(exc))
        error(
            "Check that Label Studio is running: docker compose --env-file .env -f infra/docker-compose.yml ps"
        )
        return 1
    except Exception as exc:
        error(str(exc))
        return 1

    print("\n[OK] Image review bootstrap summary")
    print(f"project_id={result['project_id']}")
    print(f"project_title={result['project_title']}")
    print(f"project_url={result['project_url']}")
    print(f"ml_backend_status={result['ml_backend_status']}")
    print(f"webhook_status={result['webhook_status']}")
    print("next_step=scripts/import_image_review_tasks_to_label_studio.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
