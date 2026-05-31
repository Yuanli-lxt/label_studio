import unittest

import numpy as np

from image_segmentation.benchmark.metrics import binary_mask_metrics


class BenchmarkMaskMetricTests(unittest.TestCase):
    def test_mask_metrics(self):
        model = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        gt = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.uint8)
        metrics = binary_mask_metrics(model, gt)
        self.assertAlmostEqual(1 / 3, metrics["iou"])
        self.assertAlmostEqual(0.5, metrics["dice"])
        self.assertAlmostEqual(0.5, metrics["precision"])
        self.assertAlmostEqual(0.5, metrics["recall"])


if __name__ == "__main__":
    unittest.main()

