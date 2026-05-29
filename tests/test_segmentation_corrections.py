import importlib.util
import unittest
from pathlib import Path

import numpy as np
from label_studio_converter.brush import mask2rle


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "services" / "trainer" / "segmentation_corrections.py"


def load_module():
    spec = importlib.util.spec_from_file_location("segmentation_corrections", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def brush_result(mask, result_id="mask", meta=None, from_name="mask_label", to_name="image"):
    height, width = mask.shape
    result = {
        "id": result_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "brushlabels",
        "original_width": width,
        "original_height": height,
        "value": {
            "format": "rle",
            "rle": mask2rle(mask.astype(np.uint8)),
            "brushlabels": ["Object"],
        },
    }
    if meta is not None:
        result["meta"] = meta
    return result


def task_with_masks(model_mask, human_mask, prediction_meta=None):
    return {
        "id": 101,
        "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
        "predictions": [
            {
                "id": 201,
                "model_version": "mobilesam-seg-v0001",
                "result": [brush_result(model_mask, result_id="pred", meta=prediction_meta)],
            }
        ],
        "annotations": [{"id": 301, "result": [brush_result(human_mask, result_id="ann")]}],
    }


class SegmentationCorrectionMetricTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_identical_masks_have_no_correction(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        delta = self.module.compute_mask_delta_metrics(mask, mask.copy(), 10, 10)

        self.assertEqual(1.0, delta["model_human_iou"])
        self.assertEqual(1.0, delta["model_human_dice"])
        self.assertEqual(0.0, delta["correction_area_ratio"])
        self.assertFalse(delta["major_correction"])
        self.assertEqual("none", delta["correction_severity"])

    def test_disjoint_masks_are_major_correction(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)
        model[1:3, 1:3] = 1
        human[7:9, 7:9] = 1

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertEqual(0.0, delta["model_human_iou"])
        self.assertTrue(delta["major_correction"])
        self.assertEqual("major", delta["correction_severity"])

    def test_human_adds_area(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)
        model[2:5, 2:5] = 1
        human[2:8, 2:8] = 1

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertGreater(delta["added_area_px"], 0)
        self.assertEqual(0, delta["removed_area_px"])
        self.assertIn("large_added_area", delta["correction_reason"])

    def test_human_removes_area(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)
        model[2:8, 2:8] = 1
        human[2:5, 2:5] = 1

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertGreater(delta["removed_area_px"], 0)
        self.assertEqual(0, delta["added_area_px"])
        self.assertIn("large_removed_area", delta["correction_reason"])

    def test_one_empty_mask_is_major(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)
        human[2:5, 2:5] = 1

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertTrue(delta["major_correction"])
        self.assertEqual("major", delta["correction_severity"])
        self.assertIn("one_mask_empty", delta["correction_reason"])

    def test_both_empty_masks_have_no_correction(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertEqual(1.0, delta["model_human_iou"])
        self.assertEqual(1.0, delta["model_human_dice"])
        self.assertEqual("none", delta["correction_severity"])

    def test_centroid_shift_is_numeric(self):
        model = np.zeros((10, 10), dtype=np.float32)
        human = np.zeros((10, 10), dtype=bool)
        model[1:3, 1:3] = 1.0
        human[6:8, 6:8] = True

        delta = self.module.compute_mask_delta_metrics(model, human, 10, 10)

        self.assertIsInstance(delta["centroid_shift_px"], float)
        self.assertIsInstance(delta["centroid_shift_ratio"], float)
        self.assertIn("large_centroid_shift", delta["correction_reason"])

    def test_bbox_iou_computes_or_returns_none(self):
        model = np.zeros((10, 10), dtype=np.uint8)
        human = np.zeros((10, 10), dtype=np.uint8)
        model[1:5, 1:5] = 1
        human[3:7, 3:7] = 1

        with_boxes = self.module.compute_mask_delta_metrics(
            model,
            human,
            10,
            10,
            model_bbox=[1, 1, 4, 4],
            human_bbox=[3, 3, 6, 6],
        )
        self.assertIsInstance(with_boxes["model_human_bbox_iou"], float)
        self.assertIsNone(
            self.module.compute_mask_delta_metrics(
                np.zeros((10, 10), dtype=np.uint8),
                np.zeros((10, 10), dtype=np.uint8),
                10,
                10,
                model_bbox=[1, 1, 1, 4],
                human_bbox=None,
            )["model_human_bbox_iou"]
        )


class SegmentationCorrectionRecordTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_label_studio_style_record_carries_metadata(self):
        model = np.zeros((8, 8), dtype=np.uint8)
        human = np.zeros((8, 8), dtype=np.uint8)
        model[2:5, 2:5] = 1
        human[2:6, 2:6] = 1
        prediction_meta = {
            "prompt_box": [2, 2, 5, 5],
            "mask_bbox": [2, 2, 4, 4],
            "mask_quality": {"valid_mask": True, "mask_area_px": 9},
            "review": {"needs_review": False, "review_reason": []},
            "uncertainty": {"method": "prompt_stability", "stable": True},
        }

        record = self.module.build_segmentation_correction_record_from_task(
            task_with_masks(model, human, prediction_meta)
        )

        self.assertEqual("ok", record["record_status"])
        self.assertIsNone(record["skip_reason"])
        self.assertIn("model_human_iou", record["delta"])
        self.assertIn("correction_area_ratio", record["delta"])
        self.assertIn("major_correction", record["delta"])
        self.assertEqual(prediction_meta["mask_quality"], record["mask_quality"])
        self.assertEqual(prediction_meta["review"], record["review"])
        self.assertEqual(prediction_meta["uncertainty"], record["uncertainty"])
        self.assertEqual(201, record["prediction_id"])
        self.assertEqual(301, record["annotation_id"])

    def test_missing_prediction_is_skipped(self):
        human = np.zeros((8, 8), dtype=np.uint8)
        human[2:5, 2:5] = 1
        task = {
            "id": 101,
            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
            "annotations": [{"id": 301, "result": [brush_result(human)]}],
        }

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("missing_model_prediction", record["skip_reason"])

    def test_missing_annotation_is_skipped(self):
        record = self.module.build_segmentation_correction_record_from_task({"id": 101, "predictions": []})

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("missing_human_annotation", record["skip_reason"])

    def test_missing_rle_is_skipped(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        task = task_with_masks(mask, mask)
        del task["annotations"][0]["result"][0]["value"]["rle"]

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("missing_human_rle", record["skip_reason"])

    def test_missing_model_rle_is_skipped(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        task = task_with_masks(mask, mask)
        del task["predictions"][0]["result"][0]["value"]["rle"]

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("missing_model_rle", record["skip_reason"])

    def test_invalid_model_rle_is_skipped(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        task = task_with_masks(mask, mask)
        task["predictions"][0]["result"][0]["value"]["rle"] = [1, 2, 3]

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("decode_model_rle_failed", record["skip_reason"])

    def test_invalid_human_rle_is_skipped(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        task = task_with_masks(mask, mask)
        task["annotations"][0]["result"][0]["value"]["rle"] = [1, 2, 3]

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("decode_human_rle_failed", record["skip_reason"])

    def test_missing_image_dimensions_is_skipped(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        task = task_with_masks(mask, mask)
        del task["predictions"][0]["result"][0]["original_width"]
        del task["predictions"][0]["result"][0]["original_height"]
        del task["annotations"][0]["result"][0]["original_width"]
        del task["annotations"][0]["result"][0]["original_height"]

        record = self.module.build_segmentation_correction_record_from_task(task)

        self.assertEqual("skipped", record["record_status"])
        self.assertEqual("missing_image_dimensions", record["skip_reason"])


if __name__ == "__main__":
    unittest.main()
