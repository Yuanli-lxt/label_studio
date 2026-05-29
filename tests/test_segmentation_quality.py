import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QUALITY_MODULE = ROOT / "services" / "ml-backend" / "segmentation_quality.py"


def load_module():
    spec = importlib.util.spec_from_file_location("segmentation_quality", QUALITY_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SegmentationQualityTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def _evaluate(self, mask, **kwargs):
        return self.module.evaluate_segmentation_quality(mask, image_width=100, image_height=100, **kwargs)

    def test_valid_normal_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 1

        output = self._evaluate(mask, prompt_bbox=[28, 18, 72, 62], mask_bbox=[30, 20, 69, 59], rle_length=16)

        self.assertIn("mask_quality", output)
        self.assertIn("review", output)
        self.assertIs(output["mask_quality"]["valid_mask"], True)
        self.assertGreater(output["mask_quality"]["mask_area_px"], 0)
        self.assertGreater(output["mask_quality"]["mask_area_ratio"], 0)
        self.assertLess(output["mask_quality"]["mask_area_ratio"], 1)
        self.assertIs(output["review"]["needs_review"], False)
        self.assertEqual("low", output["review"]["review_priority"])
        self.assertEqual([], output["review"]["review_reason"])

    def test_empty_mask_needs_high_review(self):
        output = self._evaluate(np.zeros((100, 100), dtype=np.uint8), mask_bbox=None)

        self.assertIs(output["mask_quality"]["valid_mask"], False)
        self.assertEqual(0, output["mask_quality"]["mask_area_px"])
        self.assertIs(output["review"]["needs_review"], True)
        self.assertEqual("high", output["review"]["review_priority"])
        self.assertIn("empty_or_invalid_mask", output["review"]["review_reason"])

    def test_very_small_mask_needs_review(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1

        output = self._evaluate(mask, mask_bbox=[50, 50, 51, 51])

        self.assertIs(output["review"]["needs_review"], True)
        self.assertIn("mask_area_too_small", output["review"]["review_reason"])

    def test_very_large_mask_needs_review(self):
        mask = np.ones((100, 100), dtype=np.uint8)

        output = self._evaluate(mask, mask_bbox=[0, 0, 99, 99])

        self.assertIs(output["review"]["needs_review"], True)
        self.assertIn("mask_area_too_large", output["review"]["review_reason"])

    def test_mask_touching_border_needs_review(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 20:40] = 1

        output = self._evaluate(mask, mask_bbox=[20, 0, 39, 9])

        self.assertIs(output["review"]["needs_review"], True)
        self.assertIn("mask_touches_image_border", output["review"]["review_reason"])

    def test_low_prompt_mask_bbox_iou_needs_review(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        output = self._evaluate(mask, prompt_bbox=[70, 70, 90, 90], mask_bbox=[10, 10, 29, 29])

        self.assertIsNotNone(output["mask_quality"]["bbox_iou_prompt_mask"])
        self.assertIs(output["review"]["needs_review"], True)
        self.assertIn("low_prompt_mask_bbox_iou", output["review"]["review_reason"])

    def test_missing_prompt_bbox_does_not_crash(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 1

        output = self._evaluate(mask, prompt_bbox=None, mask_bbox=[30, 20, 69, 59])

        self.assertIn("mask_quality", output)
        self.assertIn("review", output)
        self.assertIsNone(output["mask_quality"]["bbox_iou_prompt_mask"])

    def test_missing_mask_bbox_does_not_crash(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 1

        output = self._evaluate(mask, prompt_bbox=[28, 18, 72, 62], mask_bbox=None)

        self.assertIn("mask_quality", output)
        self.assertIn("review", output)
        self.assertIsNone(output["mask_quality"]["bbox_iou_prompt_mask"])

    def test_backend_fallback_or_error_needs_high_review(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 1

        output = self._evaluate(
            mask,
            prompt_bbox=[28, 18, 72, 62],
            mask_bbox=[30, 20, 69, 59],
            backend_metadata={"fallback": "placeholder", "backend_error": "missing checkpoint"},
        )

        self.assertIs(output["review"]["needs_review"], True)
        self.assertEqual("high", output["review"]["review_priority"])
        self.assertIn("backend_fallback_or_error", output["review"]["review_reason"])


if __name__ == "__main__":
    unittest.main()
