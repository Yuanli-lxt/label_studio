import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_BACKEND_APP = ROOT / "services" / "ml-backend" / "app.py"


def load_backend(env_overrides):
    backup = {}
    module_backup = {}
    if "joblib" not in sys.modules:
        module_backup["joblib"] = None
        joblib = types.ModuleType("joblib")
        joblib.load = lambda *args, **kwargs: None
        sys.modules["joblib"] = joblib
    if "numpy" not in sys.modules:
        module_backup["numpy"] = None
        numpy = types.ModuleType("numpy")
        numpy.float32 = float
        numpy.asarray = lambda values, dtype=None: values
        sys.modules["numpy"] = numpy
    for key, value in env_overrides.items():
        backup[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        module_name = f"test_image_seg_prediction_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(ML_BACKEND_APP))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for key, previous in backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        for name, previous in module_backup.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class ImageSegmentationPredictionTests(unittest.TestCase):
    def _backend(self, tmp_dir):
        return load_backend(
            {
                "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_segmentation"),
                "IMAGE_LOCAL_FILES_ROOT": str(ROOT / "demo_data" / "local-files"),
            }
        )

    def test_parse_brushlabels_config_infers_image_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            parsed = backend._parse_label_config(config)
            self.assertEqual("mask_label", parsed["brushlabels"]["name"])
            self.assertEqual("image", parsed["brushlabels"]["to_name"])
            self.assertEqual(["Object"], parsed["brushlabels"]["labels"])
            self.assertEqual(
                "image_segmentation",
                backend._infer_mode(parsed, [{"data": {"image": "/data/local-files/?d=images/demo_blue.png"}}]),
            )

    def test_predict_returns_brushlabels_rle_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            response = backend._predict(
                {
                    "label_config": config,
                    "tasks": [
                        {
                            "id": "seg-1",
                            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                        }
                    ],
                }
            )
            self.assertEqual("image_segmentation", response["mode"])
            prediction = response["results"][0]
            self.assertEqual("placeholder-image-segmentation", prediction["prediction_source"])
            result = prediction["result"][0]
            self.assertEqual("brushlabels", result["type"])
            self.assertEqual("mask_label", result["from_name"])
            self.assertEqual("image", result["to_name"])
            self.assertGreater(result["original_width"], 0)
            self.assertGreater(result["original_height"], 0)
            self.assertEqual("rle", result["value"]["format"])
            self.assertEqual(["Object"], result["value"]["brushlabels"])
            self.assertIsInstance(result["value"]["rle"], list)
            self.assertGreater(len(result["value"]["rle"]), 0)

    def test_segmentation_placeholder_score_ignores_state_boost(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "current_model.json").write_text(
                json.dumps({"behavior": {"score_boost": 0.2}}),
                encoding="utf-8",
            )
            backend = self._backend(tmp_dir)
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            response = backend._predict(
                {
                    "label_config": config,
                    "tasks": [
                        {
                            "id": "seg-boost",
                            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                        }
                    ],
                }
            )
            prediction = response["results"][0]
            self.assertEqual(0.65, prediction["score"])
            self.assertEqual(0.65, prediction["confidence"]["confidence"])
            self.assertEqual("medium", prediction["confidence"]["confidence_bucket"])


if __name__ == "__main__":
    unittest.main()
