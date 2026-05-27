import importlib.util
import unittest
from pathlib import Path

import numpy as np
from label_studio_converter.brush import mask2rle

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "debug_label_studio_rle_overlay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("debug_label_studio_rle_overlay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class LabelStudioRLEOverlayDebugTests(unittest.TestCase):
    def test_extract_brush_rle_result_from_predict_response(self):
        module = load_module()
        result = {
            "type": "brushlabels",
            "original_width": 4,
            "original_height": 3,
            "value": {"format": "rle", "rle": [1, 2, 3], "brushlabels": ["Object"]},
        }
        payload = {"results": [{"model_version": "m", "result": [result]}]}

        self.assertIs(result, module.extract_brush_rle_result(payload))

    def test_mask_bbox_returns_inclusive_bounds(self):
        module = load_module()
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1

        self.assertEqual((2, 1, 4, 3), module.mask_bbox(mask))

    def test_decode_brush_mask_uses_label_studio_converter_rle(self):
        module = load_module()
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        result = {
            "type": "brushlabels",
            "original_width": 4,
            "original_height": 3,
            "value": {"format": "rle", "rle": mask2rle(mask)},
        }

        decoded, width, height = module.decode_brush_mask(result)

        self.assertEqual(4, width)
        self.assertEqual(3, height)
        self.assertEqual((2, 1, 3, 2), module.mask_bbox(decoded))
        self.assertEqual(4, int(decoded.sum()))

    def test_extract_brush_rle_result_fails_clearly_when_missing(self):
        module = load_module()

        with self.assertRaisesRegex(module.RLEOverlayError, "could not find"):
            module.extract_brush_rle_result({"results": [{"result": []}]})


if __name__ == "__main__":
    unittest.main()
