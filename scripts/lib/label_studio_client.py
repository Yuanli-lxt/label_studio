#!/usr/bin/env python3
"""Small Label Studio API helper for local HITL review scripts."""

from __future__ import annotations

import os
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_LABEL_STUDIO_URL = "http://localhost:18080"
DEFAULT_PROJECT_TITLE = "Image Classification Human Review"
DEFAULT_PROJECT_DESCRIPTION = (
    "Human review project for image classification model mistakes and low-confidence cases."
)
DEFAULT_ML_BACKEND_URL = "http://ml-backend:9090"
DEFAULT_TRAINER_WEBHOOK_URL = "http://host.docker.internal:9091/webhook/label-studio"
DEFAULT_IMAGE_REVIEW_TASKS_PATH = "demo_data/tasks/image_classification_review_tasks.json"
DEFAULT_IMAGE_TRAINING_CANDIDATES_PATH = "demo_data/tasks/image_classification_training_candidates.jsonl"
DEFAULT_IMAGE_EVAL_MANIFEST_PATH = "demo_data/tasks/image_classification_eval_manifest.json"
DEFAULT_IMAGE_LABEL_CONFIG_PATH = "label_configs/image_classification.xml"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_WEBHOOK_ACTIONS = ("ANNOTATION_CREATED", "ANNOTATION_UPDATED")


class LabelStudioConfigError(ValueError):
    """Raised when local Label Studio configuration is incomplete."""


class LabelStudioAPIError(RuntimeError):
    """Raised when Label Studio returns a non-success response."""

    def __init__(self, method: str, url: str, status_code: int, body: str):
        body = body.strip()
        if len(body) > 2000:
            body = body[:2000] + "..."
        super().__init__(
            f"Label Studio API {method} {url} failed with HTTP {status_code}; response body: {body}"
        )
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class LabelStudioSettings:
    url: str
    api_token: str
    timeout_seconds: float
    image_review_project_title: str
    image_review_project_description: str
    ml_backend_url: str
    trainer_webhook_url: str
    image_review_tasks_path: Path
    image_training_candidates_path: Path
    image_eval_manifest_path: Path
    image_label_config_path: Path
    skip_ml_backend_setup: bool
    skip_webhook_setup: bool

    @classmethod
    def from_env(cls, require_token: bool = True) -> "LabelStudioSettings":
        token = os.getenv("LABEL_STUDIO_API_TOKEN", "").strip()
        if require_token and not token:
            raise LabelStudioConfigError(
                "LABEL_STUDIO_API_TOKEN is required. Create/copy a token in Label Studio, then run: "
                "export LABEL_STUDIO_API_TOKEN='<your-token>'"
            )

        timeout_raw = os.getenv("LABEL_STUDIO_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise LabelStudioConfigError(
                f"LABEL_STUDIO_TIMEOUT_SECONDS must be numeric, got {timeout_raw!r}"
            ) from exc
        if timeout <= 0:
            raise LabelStudioConfigError("LABEL_STUDIO_TIMEOUT_SECONDS must be > 0")

        eval_manifest = os.getenv(
            "IMAGE_EVAL_MANIFEST_PATH",
            os.getenv("IMAGE_CLS_EVAL_MANIFEST_PATH", DEFAULT_IMAGE_EVAL_MANIFEST_PATH),
        )

        return cls(
            url=_clean_base_url(os.getenv("LABEL_STUDIO_URL", DEFAULT_LABEL_STUDIO_URL)),
            api_token=token,
            timeout_seconds=timeout,
            image_review_project_title=os.getenv(
                "LABEL_STUDIO_IMAGE_REVIEW_PROJECT_TITLE", DEFAULT_PROJECT_TITLE
            ).strip()
            or DEFAULT_PROJECT_TITLE,
            image_review_project_description=os.getenv(
                "LABEL_STUDIO_IMAGE_REVIEW_PROJECT_DESCRIPTION", DEFAULT_PROJECT_DESCRIPTION
            ).strip()
            or DEFAULT_PROJECT_DESCRIPTION,
            ml_backend_url=os.getenv("LABEL_STUDIO_ML_BACKEND_URL", DEFAULT_ML_BACKEND_URL).strip()
            or DEFAULT_ML_BACKEND_URL,
            trainer_webhook_url=os.getenv(
                "LABEL_STUDIO_TRAINER_WEBHOOK_URL", DEFAULT_TRAINER_WEBHOOK_URL
            ).strip()
            or DEFAULT_TRAINER_WEBHOOK_URL,
            image_review_tasks_path=Path(
                os.getenv("IMAGE_REVIEW_TASKS_PATH", DEFAULT_IMAGE_REVIEW_TASKS_PATH)
            ),
            image_training_candidates_path=Path(
                os.getenv("IMAGE_TRAINING_CANDIDATES_PATH", DEFAULT_IMAGE_TRAINING_CANDIDATES_PATH)
            ),
            image_eval_manifest_path=Path(eval_manifest),
            image_label_config_path=Path(
                os.getenv("IMAGE_LABEL_CONFIG_PATH", DEFAULT_IMAGE_LABEL_CONFIG_PATH)
            ),
            skip_ml_backend_setup=_env_flag("LABEL_STUDIO_SKIP_ML_BACKEND_SETUP"),
            skip_webhook_setup=_env_flag("LABEL_STUDIO_SKIP_WEBHOOK_SETUP"),
        )


def _clean_base_url(value: str) -> str:
    value = (value or DEFAULT_LABEL_STUDIO_URL).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LabelStudioConfigError(f"LABEL_STUDIO_URL must be an http(s) URL, got {value!r}")
    return value


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
        return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None


def _jwt_token_type(token: str) -> Optional[str]:
    data = _jwt_payload(token)
    token_type = data.get("token_type") if isinstance(data, dict) else None
    return token_type if token_type in {"access", "refresh"} else None


def project_data_url(base_url: str, project_id: int) -> str:
    return f"{base_url.rstrip('/')}/projects/{int(project_id)}/data"


def normalize_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def stable_task_key(task: Dict[str, Any]) -> Optional[str]:
    if not isinstance(task, dict):
        return None
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    for key in ("image", "image_url", "url"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("image", "image_url", "url"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    task_id = task.get("id") or task.get("task_id")
    if task_id is not None:
        return f"id:{task_id}"
    return None


def normalize_task_for_import(task: Dict[str, Any]) -> Dict[str, Any]:
    """Return a Label Studio import task preserving current demo task shape."""
    data = task.get("data") if isinstance(task.get("data"), dict) else None
    if data is None:
        data = {}
        for key in ("image", "caption"):
            if task.get(key) is not None:
                data[key] = task[key]
    normalized: Dict[str, Any] = {"data": data}
    if task.get("id") is not None:
        normalized["id"] = task.get("id")
    if isinstance(task.get("meta"), dict):
        normalized["meta"] = task["meta"]
    return normalized


def validate_review_task(task: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(task, dict):
        return None, "task is not an object"
    normalized = normalize_task_for_import(task)
    data = normalized.get("data") if isinstance(normalized.get("data"), dict) else {}
    image = data.get("image")
    if not isinstance(image, str) or not image.strip():
        return None, "missing data.image"
    normalized["data"]["image"] = image.strip()
    return normalized, None


def normalize_list_response(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "tasks", "webhooks", "ml_backends", "models"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


class LabelStudioClient:
    def __init__(
        self,
        settings: LabelStudioSettings,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = settings
        self.base_url = settings.url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.session.headers.update({"Authorization": self._auth_header_value(settings.api_token)})

    def _auth_header_value(self, token: str) -> str:
        token = token.strip()
        token_type = _jwt_token_type(token)
        if token_type == "access":
            return f"Bearer {token}"
        if token_type == "refresh":
            return f"Bearer {self._refresh_jwt_access_token(token)}"
        return f"Token {token}"

    def _refresh_jwt_access_token(self, refresh_token: str) -> str:
        url = f"{self.base_url}/api/token/refresh/"
        try:
            response = self.session.post(
                url,
                json={"refresh": refresh_token},
                timeout=self.settings.timeout_seconds,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Label Studio API POST {url} failed while refreshing JWT token: {exc}") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise LabelStudioAPIError("POST", url, response.status_code, response.text)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Label Studio API POST {url} returned invalid JSON while refreshing JWT token") from exc
        access_token = payload.get("access") if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError(f"Label Studio API POST {url} did not return an access token")
        return access_token.strip()

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.base_url}{path}"
        headers = dict(kwargs.pop("headers", {}) or {})
        if "json" in kwargs:
            headers.setdefault("Content-Type", "application/json")
        try:
            response = self.session.request(
                method.upper(),
                url,
                timeout=self.settings.timeout_seconds,
                headers=headers,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Label Studio API {method.upper()} {url} failed: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise LabelStudioAPIError(method.upper(), url, response.status_code, response.text)
        if response.status_code == 204 or not response.text.strip():
            return None
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            return response.text
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"Label Studio API {method.upper()} {url} returned invalid JSON") from exc

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Any:
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def version(self) -> Any:
        return self.get("/api/version")

    def health_check(self) -> Any:
        try:
            return self.version()
        except LabelStudioAPIError:
            return self.get("/health")

    def list_projects(self) -> List[Dict[str, Any]]:
        payload = self.get("/api/projects/", params={"page_size": 1000})
        return normalize_list_response(payload)

    def find_project_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        for project in self.list_projects():
            if str(project.get("title", "")).strip() == title:
                return project
        return None

    def create_project(self, title: str, description: str, label_config: str) -> Dict[str, Any]:
        payload = {"title": title, "description": description, "label_config": label_config}
        created = self.post("/api/projects/", json=payload)
        if not isinstance(created, dict):
            raise RuntimeError("Label Studio create project returned non-object response")
        return created

    def update_project_label_config(self, project_id: int, label_config: str) -> Dict[str, Any]:
        updated = self.patch(f"/api/projects/{int(project_id)}/", json={"label_config": label_config})
        return updated if isinstance(updated, dict) else {"id": project_id}

    def ensure_project(self, title: str, description: str, label_config: str) -> Tuple[Dict[str, Any], str]:
        project = self.find_project_by_title(title)
        if project is None:
            project = self.create_project(title, description, label_config)
            return project, "created"
        project_id = int(project["id"])
        self.update_project_label_config(project_id, label_config)
        refreshed = self.get(f"/api/projects/{project_id}/")
        return (refreshed if isinstance(refreshed, dict) else project), "reused"

    def list_tasks(self, project_id: int) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get(
                "/api/tasks/",
                params={"project": int(project_id), "page": page, "page_size": 1000},
            )
            batch = normalize_list_response(payload)
            tasks.extend(batch)
            if not isinstance(payload, dict) or not payload.get("next") or not batch:
                break
            page += 1
        return tasks

    def import_tasks(self, project_id: int, tasks: Sequence[Dict[str, Any]]) -> Any:
        return self.post(f"/api/projects/{int(project_id)}/import", json=list(tasks))

    def import_tasks_dedup(
        self,
        project_id: int,
        tasks: Sequence[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        existing_keys = {key for key in (stable_task_key(task) for task in self.list_tasks(project_id)) if key}
        seen_in_batch = set()
        to_import = []
        skipped_duplicate = 0
        skipped_invalid = 0
        invalid_reasons: List[str] = []

        for task in tasks:
            normalized, error = validate_review_task(task)
            if error:
                skipped_invalid += 1
                if len(invalid_reasons) < 10:
                    invalid_reasons.append(error)
                continue
            assert normalized is not None
            key = stable_task_key(normalized)
            if not key:
                skipped_invalid += 1
                if len(invalid_reasons) < 10:
                    invalid_reasons.append("missing stable task key")
                continue
            if key in existing_keys or key in seen_in_batch:
                skipped_duplicate += 1
                continue
            seen_in_batch.add(key)
            to_import.append(normalized)

        response = None
        if to_import and not dry_run:
            response = self.import_tasks(project_id, to_import)

        return {
            "project_id": int(project_id),
            "total": len(tasks),
            "imported": 0 if dry_run else len(to_import),
            "planned_import": len(to_import),
            "skipped_duplicate": skipped_duplicate,
            "skipped_invalid": skipped_invalid,
            "invalid_reasons": invalid_reasons,
            "dry_run": dry_run,
            "response": response,
        }

    def list_webhooks(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"project": int(project_id)} if project_id is not None else None
        payload = self.get("/api/webhooks/", params=params)
        return normalize_list_response(payload)

    def create_webhook(
        self,
        project_id: int,
        url: str,
        actions: Sequence[str] = DEFAULT_WEBHOOK_ACTIONS,
    ) -> Dict[str, Any]:
        payload = webhook_payload(project_id, url, actions)
        created = self.post("/api/webhooks/", json=payload)
        if not isinstance(created, dict):
            raise RuntimeError("Label Studio create webhook returned non-object response")
        return created

    def update_webhook(
        self,
        webhook_id: int,
        project_id: int,
        url: str,
        actions: Sequence[str] = DEFAULT_WEBHOOK_ACTIONS,
    ) -> Dict[str, Any]:
        updated = self.patch(f"/api/webhooks/{int(webhook_id)}/", json=webhook_payload(project_id, url, actions))
        return updated if isinstance(updated, dict) else {"id": webhook_id}

    def ensure_webhook(
        self,
        project_id: int,
        url: str,
        actions: Sequence[str] = DEFAULT_WEBHOOK_ACTIONS,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        for webhook in self.list_webhooks(project_id):
            if normalize_url(webhook.get("url")) == normalize_url(url):
                webhook_id = webhook.get("id")
                if webhook_id is not None:
                    return self.update_webhook(int(webhook_id), project_id, url, actions), "updated"
                return webhook, "reused"
        return self.create_webhook(project_id, url, actions), "created"

    def list_ml_backends(self, project_id: int) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for path in ("/api/ml/", "/api/ml-backends/"):
            try:
                payload = self.get(path, params={"project": int(project_id)})
                return normalize_list_response(payload)
            except LabelStudioAPIError as exc:
                last_error = exc
                if exc.status_code not in {404, 405}:
                    raise
        if last_error:
            raise last_error
        return []

    def connect_ml_backend(self, project_id: int, url: str) -> Dict[str, Any]:
        payload = {"project": int(project_id), "url": url}
        last_error: Optional[Exception] = None
        for path in ("/api/ml/", "/api/ml-backends/"):
            try:
                created = self.post(path, json=payload)
                if not isinstance(created, dict):
                    raise RuntimeError("Label Studio ML backend connect returned non-object response")
                return created
            except LabelStudioAPIError as exc:
                last_error = exc
                if exc.status_code not in {404, 405}:
                    raise
        if last_error:
            raise last_error
        raise RuntimeError("unable to connect ML backend")

    def ensure_ml_backend(self, project_id: int, url: str) -> Tuple[Optional[Dict[str, Any]], str]:
        for backend in self.list_ml_backends(project_id):
            if normalize_url(backend.get("url")) == normalize_url(url):
                return backend, "reused"
        return self.connect_ml_backend(project_id, url), "created"


def webhook_payload(
    project_id: int,
    url: str,
    actions: Sequence[str] = DEFAULT_WEBHOOK_ACTIONS,
) -> Dict[str, Any]:
    action_list = list(actions)
    return {
        "project": int(project_id),
        "url": url,
        "actions": action_list,
        "send_payload": True,
        "send_for_all_actions": False,
        "headers": {},
        "is_active": True,
    }


def build_client_from_env(require_token: bool = True) -> LabelStudioClient:
    return LabelStudioClient(LabelStudioSettings.from_env(require_token=require_token))


__all__ = [
    "DEFAULT_IMAGE_EVAL_MANIFEST_PATH",
    "DEFAULT_IMAGE_LABEL_CONFIG_PATH",
    "DEFAULT_IMAGE_REVIEW_TASKS_PATH",
    "DEFAULT_IMAGE_TRAINING_CANDIDATES_PATH",
    "DEFAULT_LABEL_STUDIO_URL",
    "DEFAULT_ML_BACKEND_URL",
    "DEFAULT_PROJECT_DESCRIPTION",
    "DEFAULT_PROJECT_TITLE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TRAINER_WEBHOOK_URL",
    "DEFAULT_WEBHOOK_ACTIONS",
    "LabelStudioAPIError",
    "LabelStudioClient",
    "LabelStudioConfigError",
    "LabelStudioSettings",
    "build_client_from_env",
    "normalize_task_for_import",
    "project_data_url",
    "stable_task_key",
    "validate_review_task",
    "webhook_payload",
]
