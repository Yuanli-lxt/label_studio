import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

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

    def test_segmentation_fallback_rle_encoder_is_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")

            original_import = __import__

            def block_label_studio_converter(name, *args, **kwargs):
                if name.startswith("label_studio_converter"):
                    raise ImportError("blocked label_studio_converter for test")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=block_label_studio_converter):
                response = backend._predict(
                    {
                        "label_config": config,
                        "tasks": [
                            {
                                "id": "seg-fallback",
                                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                            }
                        ],
                    }
                )

            prediction = response["results"][0]
            self.assertEqual("fallback-minimal", prediction["confidence"]["rle_encoder"])
            self.assertEqual("rle", prediction["result"][0]["value"]["format"])

    def test_prediction_uses_trained_segmentation_metadata_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            artifacts = tmp_dir / "image_segmentation"
            artifacts.mkdir(parents=True)
            (artifacts / "metadata.json").write_text(
                json.dumps({"model_version": "image-seg-v0042", "task_type": "image_segmentation"}),
                encoding="utf-8",
            )
            backend = self._backend(tmp_dir)
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")

            response = backend._predict(
                {
                    "label_config": config,
                    "tasks": [
                        {
                            "id": "seg-trained",
                            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                        }
                    ],
                }
            )

            self.assertEqual("image-seg-v0042", response["model_version"])
            self.assertEqual("image-seg-v0042", response["results"][0]["model_version"])

    def test_mobilesam_backend_returns_real_backend_prediction_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            checkpoint = tmp_dir / "mobile_sam.pt"
            checkpoint.write_text("fake checkpoint", encoding="utf-8")
            fake_segment_anything = types.ModuleType("segment_anything")

            class FakeModel:
                def to(self, device=None):
                    self.device = device
                    return self

            class FakePredictor:
                def __init__(self, model):
                    self.model = model

                def set_image(self, image):
                    self.image = image

                def predict(self, box=None, multimask_output=False):
                    mask = [[0 for _ in range(10)] for _ in range(8)]
                    for y in range(2, 6):
                        for x in range(3, 8):
                            mask[y][x] = 1
                    return [mask], [0.91], None

            fake_segment_anything.sam_model_registry = {
                "vit_t": lambda checkpoint=None: FakeModel(),
            }
            fake_segment_anything.SamPredictor = FakePredictor

            with patch.dict(sys.modules, {"segment_anything": fake_segment_anything}):
                backend = load_backend(
                    {
                        "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                        "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                        "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                        "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_segmentation"),
                        "IMAGE_LOCAL_FILES_ROOT": str(ROOT / "demo_data" / "local-files"),
                        "IMAGE_SEG_BACKEND": "mobilesam",
                        "IMAGE_SEG_MODEL_TYPE": "vit_t",
                        "IMAGE_SEG_CHECKPOINT": str(checkpoint),
                        "IMAGE_SEG_DEVICE": "cpu",
                    }
                )
                backend._load_image_rgb_array = lambda image_path: [[0, 0, 0]]
                config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
                response = backend._predict(
                    {
                        "label_config": config,
                        "tasks": [
                            {
                                "id": "seg-mobile",
                                "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                            }
                        ],
                    }
                )

            prediction = response["results"][0]
            self.assertEqual("mobilesam-image-segmentation", prediction["prediction_source"])
            self.assertEqual("mobilesam-image-segmentation", prediction["confidence"]["prediction_source"])
            self.assertEqual("mobilesam", prediction["confidence"]["backend"])
            self.assertGreaterEqual(prediction["score"], 0.9)
            self.assertIsInstance(prediction["result"][0]["value"]["rle"], list)
            self.assertGreater(len(prediction["result"][0]["value"]["rle"]), 0)

    def test_sam2_backend_missing_dependency_falls_back_with_backend_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            backend = load_backend(
                {
                    "MODEL_STATE_PATH": str(tmp_dir / "current_model.json"),
                    "TEXT_MODEL_ARTIFACTS_DIR": str(tmp_dir / "text_artifacts"),
                    "IMAGE_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_artifacts"),
                    "IMAGE_SEG_MODEL_ARTIFACTS_DIR": str(tmp_dir / "image_segmentation"),
                    "IMAGE_LOCAL_FILES_ROOT": str(ROOT / "demo_data" / "local-files"),
                    "IMAGE_SEG_BACKEND": "sam2",
                    "IMAGE_SEG_MODEL_ID": "facebook/sam2-hiera-large",
                    "IMAGE_SEG_DEVICE": "cpu",
                }
            )
            config = Path("label_configs/image_segmentation.xml").read_text(encoding="utf-8")
            response = backend._predict(
                {
                    "label_config": config,
                    "tasks": [
                        {
                            "id": "seg-sam2-missing",
                            "data": {"image": "/data/local-files/?d=images/demo_blue.png"},
                        }
                    ],
                }
            )

            prediction = response["results"][0]
            self.assertEqual("placeholder-image-segmentation", prediction["prediction_source"])
            self.assertEqual("sam2", prediction["confidence"]["requested_backend"])
            self.assertIn("backend_error", prediction["confidence"])
            self.assertNotEqual("", prediction["confidence"]["backend_error"])


if __name__ == "__main__":
    unittest.main()
