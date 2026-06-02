import unittest
from unittest.mock import patch

import numpy as np

from image_segmentation.benchmark.metrics import binary_mask_metrics, boundary_metrics, prediction_time_boundary_shape_features


class BenchmarkMaskMetricTests(unittest.TestCase):
    def test_mask_metrics(self):
        model = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        gt = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.uint8)
        metrics = binary_mask_metrics(model, gt)
        self.assertAlmostEqual(1 / 3, metrics["iou"])
        self.assertAlmostEqual(0.5, metrics["dice"])
        self.assertAlmostEqual(0.5, metrics["precision"])
        self.assertAlmostEqual(0.5, metrics["recall"])

    def test_boundary_metrics_perfect_match(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        metrics = boundary_metrics(mask, mask, tolerance_px=1)
        self.assertAlmostEqual(1.0, metrics["boundary_f1"])
        self.assertAlmostEqual(1.0, metrics["boundary_precision"])
        self.assertAlmostEqual(1.0, metrics["boundary_recall"])

    def test_boundary_metrics_shifted_mask(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        model[2:6, 2:6] = 1
        gt[3:7, 3:7] = 1
        metrics = boundary_metrics(model, gt, tolerance_px=0)
        self.assertLess(metrics["boundary_f1"], 1.0)
        self.assertGreater(metrics["boundary_error_area_ratio"], 0.0)

    def test_boundary_metrics_empty_mask_safe(self):
        metrics = boundary_metrics(np.zeros((4, 4)), np.zeros((4, 4)))
        self.assertIsNone(metrics["boundary_f1"])
        self.assertEqual("empty_boundaries", metrics["warning"])

    def test_boundary_metrics_tolerance_effect(self):
        model = np.zeros((12, 12), dtype=np.uint8)
        gt = np.zeros((12, 12), dtype=np.uint8)
        model[2:6, 2:6] = 1
        gt[4:8, 2:6] = 1
        strict = boundary_metrics(model, gt, tolerance_px=0)
        tolerant = boundary_metrics(model, gt, tolerance_px=2)
        self.assertGreater(tolerant["boundary_f1"], strict["boundary_f1"])

    def test_prediction_time_features_empty_mask_safe(self):
        features = prediction_time_boundary_shape_features(np.zeros((4, 4), dtype=np.uint8), width=4, height=4)
        self.assertEqual(0.0, features["pred_area_ratio"])
        self.assertEqual(0, features["pred_component_count"])
        self.assertEqual(0, features["pred_hole_count"])

    def test_prediction_time_features_full_mask(self):
        features = prediction_time_boundary_shape_features(np.ones((5, 5), dtype=np.uint8), width=5, height=5)
        self.assertEqual(1.0, features["pred_area_ratio"])
        self.assertEqual(1.0, features["pred_largest_component_ratio"])
        self.assertTrue(features["pred_touches_border"])

    def test_prediction_time_features_border_and_components(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        mask[5:7, 5:7] = 1
        features = prediction_time_boundary_shape_features(mask, width=8, height=8)
        self.assertTrue(features["pred_touches_border"])
        self.assertEqual(2, features["pred_component_count"])
        self.assertAlmostEqual(0.5, features["pred_largest_component_ratio"])

    def test_prediction_time_features_hole_and_thinness(self):
        mask = np.ones((7, 7), dtype=np.uint8)
        mask[3, 3] = 0
        features = prediction_time_boundary_shape_features(mask, width=7, height=7)
        self.assertEqual(1, features["pred_hole_count"])
        self.assertGreater(features["pred_thinness_proxy"], 0.0)

    def test_prediction_time_features_hole_count_without_scipy(self):
        original_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name == "scipy":
                raise ImportError("blocked for fallback test")
            return original_import(name, *args, **kwargs)

        mask = np.ones((7, 7), dtype=np.uint8)
        mask[2, 2] = 0
        mask[4, 4] = 0
        with patch("builtins.__import__", side_effect=blocked_import):
            features = prediction_time_boundary_shape_features(mask, width=7, height=7)
        self.assertEqual(2, features["pred_hole_count"])


if __name__ == "__main__":
    unittest.main()
