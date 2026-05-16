import os
import base64
import json
import unittest
from contextlib import contextmanager

from scripts.lib.label_studio_client import (
    DEFAULT_LABEL_STUDIO_URL,
    DEFAULT_TIMEOUT_SECONDS,
    LabelStudioClient,
    LabelStudioConfigError,
    LabelStudioSettings,
    webhook_payload,
)


@contextmanager
def patched_env(values):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = "{}" if payload is None else __import__("json").dumps(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, timeout=None, headers=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "kwargs": kwargs})
        if method == "GET" and url.endswith("/api/projects/"):
            return FakeResponse({"results": [{"id": 7, "title": "Image Classification Human Review"}]})
        if method == "PATCH" and url.endswith("/api/projects/7/"):
            return FakeResponse({"id": 7, "title": "Image Classification Human Review"})
        if method == "GET" and url.endswith("/api/projects/7/"):
            return FakeResponse({"id": 7, "title": "Image Classification Human Review", "label_config": "<View/>"})
        if method == "GET" and url.endswith("/api/webhooks/"):
            return FakeResponse({"results": [{"id": 11, "project": 7, "url": "http://trainer:9091/webhook/label-studio"}]})
        if method == "PATCH" and url.endswith("/api/webhooks/11/"):
            return FakeResponse({"id": 11, **kwargs["json"]})
        if method == "GET" and url.endswith("/api/ml/"):
            return FakeResponse({"results": [{"id": 12, "project": 7, "url": "http://ml-backend:9090"}]})
        raise AssertionError(f"unexpected request: {method} {url}")


def fake_jwt(token_type):
    def enc(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc({'token_type': token_type})}.signature"


class FakeJWTSession:
    def __init__(self):
        self.headers = {}
        self.refresh_calls = 0

    def post(self, url, json=None, timeout=None, headers=None):
        self.refresh_calls += 1
        return FakeResponse({"access": fake_jwt("access")})


class LabelStudioClientConfigTests(unittest.TestCase):
    def test_from_env_defaults_and_missing_token(self):
        with patched_env({"LABEL_STUDIO_API_TOKEN": None, "LABEL_STUDIO_URL": None, "LABEL_STUDIO_TIMEOUT_SECONDS": None}):
            with self.assertRaises(LabelStudioConfigError):
                LabelStudioSettings.from_env(require_token=True)

        with patched_env({"LABEL_STUDIO_API_TOKEN": "token-123", "LABEL_STUDIO_URL": None, "LABEL_STUDIO_TIMEOUT_SECONDS": None}):
            settings = LabelStudioSettings.from_env(require_token=True)
        self.assertEqual(DEFAULT_LABEL_STUDIO_URL, settings.url)
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, settings.timeout_seconds)
        self.assertFalse(settings.skip_ml_backend_setup)
        self.assertFalse(settings.skip_webhook_setup)

    def test_skip_flags_parse(self):
        with patched_env(
            {
                "LABEL_STUDIO_API_TOKEN": "token-123",
                "LABEL_STUDIO_SKIP_ML_BACKEND_SETUP": "1",
                "LABEL_STUDIO_SKIP_WEBHOOK_SETUP": "true",
            }
        ):
            settings = LabelStudioSettings.from_env(require_token=True)
        self.assertTrue(settings.skip_ml_backend_setup)
        self.assertTrue(settings.skip_webhook_setup)

    def test_project_webhook_and_ml_backend_idempotency(self):
        settings = LabelStudioSettings.from_env(require_token=False)
        client = LabelStudioClient(settings, session=FakeSession())

        project, project_status = client.ensure_project(
            "Image Classification Human Review",
            "desc",
            "<View/>",
        )
        webhook, webhook_status = client.ensure_webhook(7, "http://trainer:9091/webhook/label-studio")
        backend, backend_status = client.ensure_ml_backend(7, "http://ml-backend:9090")

        self.assertEqual(7, project["id"])
        self.assertEqual("reused", project_status)
        self.assertEqual("updated", webhook_status)
        self.assertEqual("reused", backend_status)
        self.assertEqual(12, backend["id"])

        post_calls = [call for call in client.session.calls if call["method"] == "POST"]
        self.assertEqual([], post_calls)
        patch_calls = [call for call in client.session.calls if call["method"] == "PATCH"]
        self.assertEqual(2, len(patch_calls))

    def test_webhook_payload_contains_annotation_events(self):
        payload = webhook_payload(3, "http://trainer:9091/webhook/label-studio")
        self.assertEqual(3, payload["project"])
        self.assertEqual("http://trainer:9091/webhook/label-studio", payload["url"])
        self.assertIn("ANNOTATION_CREATED", payload["actions"])
        self.assertIn("ANNOTATION_UPDATED", payload["actions"])
        self.assertTrue(payload["send_payload"])

    def test_jwt_refresh_token_is_exchanged_for_bearer_access_token(self):
        settings = LabelStudioSettings.from_env(require_token=False).__class__(
            **{
                **LabelStudioSettings.from_env(require_token=False).__dict__,
                "api_token": fake_jwt("refresh"),
            }
        )
        session = FakeJWTSession()
        client = LabelStudioClient(settings, session=session)

        self.assertEqual(1, session.refresh_calls)
        self.assertTrue(client.session.headers["Authorization"].startswith("Bearer "))


if __name__ == "__main__":
    unittest.main()
