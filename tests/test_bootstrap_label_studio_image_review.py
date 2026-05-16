import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_label_studio_image_review import bootstrap_image_review
from scripts.lib.label_studio_client import LabelStudioSettings


class FakeBootstrapClient:
    def __init__(self):
        self.health_called = False
        self.project_args = None
        self.ml_called = False
        self.webhook_called = False

    def health_check(self):
        self.health_called = True
        return {"version": "fake"}

    def ensure_project(self, title, description, label_config):
        self.project_args = {"title": title, "description": description, "label_config": label_config}
        return {"id": 42, "title": title}, "created"

    def ensure_ml_backend(self, project_id, url):
        self.ml_called = True
        return {"id": 5, "project": project_id, "url": url}, "created"

    def ensure_webhook(self, project_id, url):
        self.webhook_called = True
        return {"id": 6, "project": project_id, "url": url}, "created"


class BootstrapImageReviewTests(unittest.TestCase):
    def test_bootstrap_applies_label_config_and_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "image.xml"
            cfg.write_text("<View><Image name='image' value='$image'/></View>", encoding="utf-8")
            settings = LabelStudioSettings.from_env(require_token=False).__class__(
                **{**LabelStudioSettings.from_env(require_token=False).__dict__, "image_label_config_path": cfg}
            )
            client = FakeBootstrapClient()
            result = bootstrap_image_review(client, settings)

        self.assertTrue(client.health_called)
        self.assertEqual(42, result["project_id"])
        self.assertEqual("created", result["ml_backend_status"])
        self.assertEqual("created", result["webhook_status"])
        self.assertIn("<Image", client.project_args["label_config"])
        self.assertTrue(client.ml_called)
        self.assertTrue(client.webhook_called)

    def test_skip_flags_do_not_call_optional_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "image.xml"
            cfg.write_text("<View/>", encoding="utf-8")
            base = LabelStudioSettings.from_env(require_token=False)
            settings = base.__class__(**{**base.__dict__, "image_label_config_path": cfg})
            client = FakeBootstrapClient()
            result = bootstrap_image_review(client, settings, skip_ml_backend=True, skip_webhook=True)

        self.assertEqual("skipped", result["ml_backend_status"])
        self.assertEqual("skipped", result["webhook_status"])
        self.assertFalse(client.ml_called)
        self.assertFalse(client.webhook_called)


if __name__ == "__main__":
    unittest.main()
