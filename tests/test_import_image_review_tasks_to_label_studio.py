import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_image_review_tasks_to_label_studio import import_review_tasks, load_review_tasks
from scripts.lib.label_studio_client import LabelStudioClient, LabelStudioSettings


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(payload if payload is not None else {})

    def json(self):
        return self.payload


class ImportFakeSession:
    def __init__(self):
        self.headers = {}
        self.import_payloads = []

    def request(self, method, url, timeout=None, headers=None, **kwargs):
        if method == "GET" and url.endswith("/api/projects/"):
            return FakeResponse({"results": [{"id": 33, "title": "Image Classification Human Review"}]})
        if method == "GET" and url.endswith("/api/tasks/"):
            return FakeResponse(
                {
                    "results": [
                        {"id": 1, "data": {"image": "/data/local-files/?d=images/existing.png"}}
                    ],
                    "next": None,
                }
            )
        if method == "POST" and url.endswith("/api/projects/33/import"):
            self.import_payloads.append(kwargs["json"])
            return FakeResponse({"task_count": len(kwargs["json"])})
        raise AssertionError(f"unexpected request: {method} {url}")


class ImportImageReviewTasksTests(unittest.TestCase):
    def _client(self, session):
        settings = LabelStudioSettings.from_env(require_token=False)
        return LabelStudioClient(settings, session=session)

    def test_loads_current_demo_json_format(self):
        tasks = load_review_tasks(Path("demo_data/tasks/image_classification_review_tasks.json"))
        self.assertIsInstance(tasks, list)
        self.assertGreaterEqual(len(tasks), 1)
        self.assertIn("image", tasks[0].get("data", {}))

    def test_import_dedupes_existing_and_batch_duplicates_and_skips_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": "a", "data": {"image": "/data/local-files/?d=images/existing.png"}},
                        {"id": "b", "data": {"image": "/data/local-files/?d=images/new.png"}},
                        {"id": "c", "data": {"image": "/data/local-files/?d=images/new.png"}},
                        {"id": "d", "data": {"caption": "missing image"}},
                    ]
                ),
                encoding="utf-8",
            )
            session = ImportFakeSession()
            result = import_review_tasks(self._client(session), path, project_title="Image Classification Human Review")

        self.assertEqual(4, result["total_read"])
        self.assertEqual(1, result["imported"])
        self.assertEqual(1, result["planned_import"])
        self.assertEqual(2, result["skipped_duplicate"])
        self.assertEqual(1, result["pre_validation_invalid"])
        self.assertEqual(1, len(session.import_payloads))
        self.assertEqual("/data/local-files/?d=images/new.png", session.import_payloads[0][0]["data"]["image"])

    def test_dry_run_does_not_call_import_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            path.write_text(
                json.dumps([{"id": "b", "data": {"image": "/data/local-files/?d=images/new.png"}}]),
                encoding="utf-8",
            )
            session = ImportFakeSession()
            result = import_review_tasks(self._client(session), path, project_id=33, dry_run=True)

        self.assertEqual(1, result["planned_import"])
        self.assertEqual(0, result["imported"])
        self.assertEqual([], session.import_payloads)


if __name__ == "__main__":
    unittest.main()
