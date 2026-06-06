import tempfile
import unittest
import xml.etree.ElementTree as ET
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
                (
                    "<View><Image name='gt_reference' value='$gt_reference'/>"
                    "<Image name='mobilesam_preview' value='$mobilesam_preview'/>"
                    "<Image name='image' value='$image'/>"
                    "<BrushLabels name='mask_label' toName='image'/>"
                    "<Choices name='review_outcome' toName='image' required='true'>"
                    "<Choice value='No fix'/></Choices></View>"
                ),
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

    def test_default_segmentation_config_is_review_form(self):
        config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
        root = ET.fromstring(config)
        images = {node.attrib.get("name"): node.attrib.get("value") for node in root.iter("Image")}
        choices = next(node for node in root.iter("Choices") if node.attrib.get("name") == "review_outcome")

        self.assertEqual("$gt_reference", images["gt_reference"])
        self.assertEqual("$mobilesam_preview", images["mobilesam_preview"])
        self.assertEqual("$image", images["image"])
        self.assertEqual("true", choices.attrib.get("required"))
        self.assertEqual("single", choices.attrib.get("choice"))
        self.assertEqual(["No fix", "Minor fix", "Major fix", "Redo", "Skip"], [
            node.attrib.get("value") for node in choices.iter("Choice")
        ])


if __name__ == "__main__":
    unittest.main()
