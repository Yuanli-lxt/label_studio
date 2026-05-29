import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_mobilesam_segmentation_docker.py"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_mobilesam_segmentation_docker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeMobileSAMSegmentationDockerTests(unittest.TestCase):
    def test_host_backend_url_defaults_try_9092_then_19092(self):
        module = load_module()

        self.assertEqual(
            ["http://127.0.0.1:9092", "http://127.0.0.1:19092"],
            module.normalize_host_backend_urls(None),
        )

    def test_host_backend_url_explicit_is_first_without_duplicate(self):
        module = load_module()

        self.assertEqual(
            ["http://127.0.0.1:19092", "http://127.0.0.1:9092"],
            module.normalize_host_backend_urls("http://127.0.0.1:19092"),
        )

    def test_bbox_intersects_inclusive_edges(self):
        module = load_module()

        self.assertTrue(module.bbox_intersects([10, 10, 20, 20], [20, 15, 30, 25]))
        self.assertFalse(module.bbox_intersects([10, 10, 20, 20], [21, 15, 30, 25]))
        self.assertFalse(module.bbox_intersects(None, [21, 15, 30, 25]))

    def test_area_ratio_sanity_rejects_empty_tiny_and_full_masks(self):
        module = load_module()

        self.assertFalse(module.area_ratio_sanity(0.0)[0])
        self.assertFalse(module.area_ratio_sanity(0.0001)[0])
        self.assertFalse(module.area_ratio_sanity(0.99)[0])
        self.assertTrue(module.area_ratio_sanity(0.25)[0])

    def test_prediction_flags_require_brush_rle_mobilesam_metadata_and_no_choices(self):
        module = load_module()
        row = {
            "model_version": "mobilesam-seg-v0001",
            "result": [
                {
                    "type": "brushlabels",
                    "value": {"format": "rle", "rle": [1, 2, 3], "brushlabels": ["Object"]},
                    "meta": {
                        "backend": "mobilesam",
                        "prompt_box": [1, 2, 3, 4],
                        "mask_quality": {
                            "valid_mask": True,
                            "mask_area_px": 10,
                            "mask_area_ratio": 0.1,
                            "image_width": 10,
                            "image_height": 10,
                        },
                        "review": {
                            "needs_review": False,
                            "review_priority": "low",
                            "review_priority_score": 0,
                            "review_reason": [],
                        },
                        "uncertainty": {
                            "method": "prompt_stability",
                            "enabled": True,
                            "stable": True,
                            "stability_bucket": "high",
                            "reason": [],
                        },
                    },
                }
            ],
        }

        self.assertEqual([], module.validate_prediction_summary(row, "mobilesam-seg-v0001"))
        flags = module.summarize_prediction_flags(row)
        self.assertTrue(flags["has_brushlabels"])
        self.assertTrue(flags["has_rle"])
        self.assertFalse(flags["has_choices"])
        self.assertTrue(flags["has_mobilesam"])
        self.assertTrue(flags["has_backend_meta"])
        self.assertTrue(flags["has_prompt_box"])
        self.assertTrue(flags["has_mask_quality"])
        self.assertTrue(flags["has_review"])
        self.assertTrue(flags["has_uncertainty"])

    def test_prediction_summary_requires_uncertainty_only_when_requested(self):
        module = load_module()
        row = {
            "model_version": "mobilesam-seg-v0001",
            "result": [
                {
                    "type": "brushlabels",
                    "value": {"format": "rle", "rle": [1, 2, 3], "brushlabels": ["Object"]},
                    "meta": {
                        "backend": "mobilesam",
                        "prompt_box": [1, 2, 3, 4],
                        "mask_quality": {
                            "valid_mask": True,
                            "mask_area_px": 10,
                            "mask_area_ratio": 0.1,
                            "image_width": 10,
                            "image_height": 10,
                        },
                        "review": {
                            "needs_review": False,
                            "review_priority": "low",
                            "review_priority_score": 0,
                            "review_reason": [],
                        },
                    },
                }
            ],
        }

        self.assertEqual([], module.validate_prediction_summary(row, "mobilesam-seg-v0001"))
        self.assertIn(
            "has_uncertainty=False",
            module.validate_prediction_summary(row, "mobilesam-seg-v0001", require_uncertainty=True),
        )

    def test_prediction_flags_report_choices_and_wrong_version(self):
        module = load_module()
        row = {
            "model_version": "other",
            "result": [{"type": "choices", "value": {"choices": ["Object"]}}],
        }

        failures = module.validate_prediction_summary(row, "mobilesam-seg-v0001")

        self.assertIn("model_version='other'", failures)
        self.assertIn("has_choices=True", failures)
        self.assertIn("has_brushlabels=False", failures)

    def test_aggregate_pass_requires_all_steps_ok(self):
        module = load_module()

        self.assertTrue(module.aggregate_pass([module.StepResult("a", True), module.StepResult("b", True)]))
        self.assertFalse(module.aggregate_pass([module.StepResult("a", True), module.StepResult("b", False)]))

    def test_parse_args_defaults(self):
        module = load_module()

        args = module.parse_args([])

        self.assertEqual(3, args.project_id)
        self.assertEqual("http://ml-backend-gpu:9090", args.expected_backend_url)
        self.assertEqual("mobilesam-seg-v0001", args.expected_model_version)
        self.assertEqual(Path("infra"), args.compose_dir)
        self.assertIsNone(args.host_ml_backend_url)
        self.assertIs(args.enable_prompt_stability, False)


if __name__ == "__main__":
    unittest.main()
