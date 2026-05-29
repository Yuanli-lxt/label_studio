import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNCERTAINTY_MODULE = ROOT / "services" / "ml-backend" / "segmentation_uncertainty.py"


def load_module():
    spec = importlib.util.spec_from_file_location("segmentation_uncertainty", UNCERTAINTY_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SegmentationUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_bbox_variant_generation_is_deterministic_and_clamped(self):
        variants = self.module.generate_bbox_prompt_variants([1, 1, 8, 8], 10, 10, jitter_ratio=0.25)

        self.assertEqual(7, len(variants))
        self.assertEqual("original", variants[0]["name"])
        self.assertEqual([1.0, 1.0, 8.0, 8.0], variants[0]["bbox"])
        for variant in variants:
            x_min, y_min, x_max, y_max = variant["bbox"]
            self.assertGreaterEqual(x_min, 0)
            self.assertGreaterEqual(y_min, 0)
            self.assertLessEqual(x_max, 9)
            self.assertLessEqual(y_max, 9)
            self.assertGreater(x_max, x_min)
            self.assertGreater(y_max, y_min)

    def test_invalid_bbox_returns_empty_variants(self):
        self.assertEqual([], self.module.generate_bbox_prompt_variants(None, 10, 10))
        self.assertEqual([], self.module.generate_bbox_prompt_variants([5, 5, 5, 6], 10, 10))
        self.assertEqual([], self.module.generate_bbox_prompt_variants([1, 2, 3], 10, 10))

    def test_mask_iou_cases(self):
        a = np.zeros((5, 5), dtype=np.uint8)
        a[1:3, 1:3] = 1
        identical = a.copy()
        disjoint = np.zeros((5, 5), dtype=np.uint8)
        disjoint[3:5, 3:5] = 1
        partial = np.zeros((5, 5), dtype=np.uint8)
        partial[2:4, 2:4] = 1

        self.assertEqual(1.0, self.module.compute_mask_iou(a, identical))
        self.assertEqual(0.0, self.module.compute_mask_iou(a, disjoint))
        self.assertAlmostEqual(1 / 7, self.module.compute_mask_iou(a, partial))
        self.assertEqual(1.0, self.module.compute_mask_iou(np.zeros((2, 2)), np.zeros((2, 2))))
        self.assertEqual(0.0, self.module.compute_mask_iou(a, np.zeros((5, 5))))

    def test_mask_iou_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.module.compute_mask_iou(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_stable_masks_bucket_high_or_medium(self):
        masks = []
        for offset in (0, 0, 1):
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[5 : 15 + offset, 5:15] = 1
            masks.append(mask)

        output = self.module.evaluate_prompt_stability(masks, image_width=20, image_height=20)["uncertainty"]

        self.assertIs(output["stable"], True)
        self.assertIn(output["stability_bucket"], ["high", "medium"])

    def test_unstable_masks_bucket_low_with_reason(self):
        masks = []
        for start in (1, 8, 14):
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[start : start + 4, start : start + 4] = 1
            masks.append(mask)

        output = self.module.evaluate_prompt_stability(masks, image_width=20, image_height=20)["uncertainty"]

        self.assertIs(output["stable"], False)
        self.assertEqual("low", output["stability_bucket"])
        self.assertTrue(
            {
                "low_mean_pairwise_iou",
                "low_min_pairwise_iou",
                "high_disagreement_area_ratio",
            }.intersection(output["reason"])
        )

    def test_insufficient_masks_are_unknown(self):
        for masks in ([], [np.ones((5, 5), dtype=np.uint8)]):
            output = self.module.evaluate_prompt_stability(masks, image_width=5, image_height=5)["uncertainty"]

            self.assertIs(output["stable"], False)
            self.assertEqual("unknown", output["stability_bucket"])
            self.assertIn("insufficient_valid_masks", output["reason"])


if __name__ == "__main__":
    unittest.main()
