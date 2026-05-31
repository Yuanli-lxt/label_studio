import unittest

from image_segmentation.benchmark.schema import clip_xyxy, coco_xywh_to_xyxy


class BenchmarkBBoxTests(unittest.TestCase):
    def test_coco_xywh_to_xyxy(self):
        self.assertEqual([10.0, 20.0, 40.0, 60.0], coco_xywh_to_xyxy([10, 20, 30, 40], 100, 100))

    def test_bbox_clip(self):
        self.assertEqual([0.0, 0.0, 99.0, 49.0], clip_xyxy([-5, -2, 120, 60], 100, 50))


if __name__ == "__main__":
    unittest.main()

