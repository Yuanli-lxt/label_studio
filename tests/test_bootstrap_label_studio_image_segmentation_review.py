import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_label_studio_image_segmentation_review import bootstrap_image_segmentation_review
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


class BootstrapImageSegmentationReviewTests(unittest.TestCase):
    def test_bootstrap_uses_segmentation_label_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "seg.xml"
            cfg.write_text(
                "<View><Image name='image' value='$image'/><BrushLabels name='mask_label' toName='image'/></View>",
                encoding="utf-8",
            )
            base = LabelStudioSettings.from_env(require_token=False)
            settings = base.__class__(
                **{
                    **base.__dict__,
                    "image_label_config_path": cfg,
                    "image_review_project_title": "Image Segmentation Human Review",
                    "image_review_project_description": "Human review project for image segmentation masks.",
                }
            )
            client = FakeBootstrapClient()

            result = bootstrap_image_segmentation_review(client, settings)

        self.assertTrue(client.health_called)
        self.assertEqual(42, result["project_id"])
        self.assertEqual("Image Segmentation Human Review", result["project_title"])
        self.assertIn("<BrushLabels", client.project_args["label_config"])
        self.assertTrue(client.ml_called)
        self.assertTrue(client.webhook_called)


if __name__ == "__main__":
    unittest.main()
