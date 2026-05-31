import unittest

from image_segmentation.benchmark.evaluation import (
    average_precision,
    lift_at_fraction,
    precision_at_fraction,
    recall_at_fraction,
)


def item(major: bool) -> dict:
    return {"evaluation_only": {"delta": {"major_correction": major}}}


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_evaluation_metrics(self):
        queue = [item(True), item(False), item(True), item(False), item(False)]
        self.assertEqual(1.0, precision_at_fraction(queue, 0.2))
        self.assertEqual(0.5, recall_at_fraction(queue, 0.2))
        self.assertAlmostEqual(2.5, lift_at_fraction(queue, 0.2))
        self.assertAlmostEqual((1 / 1 + 2 / 3) / 2, average_precision(queue))


if __name__ == "__main__":
    unittest.main()

