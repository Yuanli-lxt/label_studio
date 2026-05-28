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

    def test_segmentation_box_prompt_prefers_task_data_bbox_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {"data": {"bbox": [51, 55, 207, 165]}},
                320,
                240,
            )

            self.assertEqual("data.bbox", prompt["source"])
            self.assertEqual("pixel_xyxy", prompt["coordinate_system"])
            self.assertEqual([51.0, 55.0, 207.0, 165.0], prompt["box"])

    def test_segmentation_box_prompt_supports_explicit_percent_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {
                        "box": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 25,
                            "unit": "percent",
                        }
                    }
                },
                320,
                240,
            )

            self.assertEqual("data.box", prompt["source"])
            self.assertEqual("percent_xyxy", prompt["coordinate_system"])
            self.assertEqual([32.0, 48.0, 128.0, 108.0], prompt["box"])

    def test_segmentation_box_prompt_supports_normalized_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {"data": {"bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.45, "normalized": True}}},
                320,
                240,
            )

            self.assertEqual("data.bbox", prompt["source"])
            self.assertEqual("normalized_xyxy", prompt["coordinate_system"])
            self.assertEqual([32.0, 48.0, 128.0, 108.0], prompt["box"])

    def test_segmentation_box_prompt_keeps_data_bbox_ahead_of_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {
                        "bbox": [10, 20, 60, 90],
                        "candidates": [{"bbox": [100, 110, 220, 230], "score": 0.99}],
                    }
                },
                320,
                240,
            )

            self.assertEqual("data.bbox", prompt["source"])
            self.assertEqual([10.0, 20.0, 60.0, 90.0], prompt["box"])

    def test_segmentation_box_prompt_supports_meta_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {"candidates": [{"bbox": [100, 110, 220, 230], "score": 0.99}]},
                    "meta": {"bbox": [20, 30, 70, 100]},
                },
                320,
                240,
            )

            self.assertEqual("meta.bbox", prompt["source"])
            self.assertEqual([20.0, 30.0, 70.0, 100.0], prompt["box"])

    def test_segmentation_box_prompt_uses_highest_scored_data_candidate_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {
                        "candidates": [
                            {"bbox": [10, 10, 50, 50], "score": 0.2},
                            {"bbox": [80, 70, 140, 150], "confidence": 0.9},
                        ]
                    }
                },
                320,
                240,
            )

            self.assertEqual("webhook_candidate.bbox", prompt["source"])
            self.assertEqual([80.0, 70.0, 140.0, 150.0], prompt["box"])

    def test_segmentation_box_prompt_uses_highest_scored_meta_candidate_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "meta": {
                        "candidates": [
                            {"box": [10, 10, 50, 50], "confidence": 0.4},
                            {"box": [90, 70, 150, 160], "score": 0.95},
                        ]
                    }
                },
                320,
                240,
            )

            self.assertEqual("webhook_candidate.box", prompt["source"])
            self.assertEqual([90.0, 70.0, 150.0, 160.0], prompt["box"])

    def test_segmentation_box_prompt_uses_first_valid_unscored_candidate_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {
                        "candidates": [
                            {"bbox": [20, 30, 80, 100]},
                            {"bbox": [90, 100, 180, 200]},
                        ]
                    }
                },
                320,
                240,
            )

            self.assertEqual("webhook_candidate.bbox", prompt["source"])
            self.assertEqual([20.0, 30.0, 80.0, 100.0], prompt["box"])

    def test_segmentation_box_prompt_skips_invalid_candidate_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {
                        "candidates": [
                            {"bbox": [80, 80, 10, 10], "score": 0.99},
                            {"bbox": [40, 50, 120, 140], "score": 0.3},
                        ]
                    }
                },
                320,
                240,
            )

            self.assertEqual("webhook_candidate.bbox", prompt["source"])
            self.assertEqual([40.0, 50.0, 120.0, 140.0], prompt["box"])

    def test_segmentation_box_prompt_converts_prediction_rectanglelabels(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "predictions": [
                        {
                            "result": [
                                {
                                    "type": "rectanglelabels",
                                    "value": {
                                        "x": 10,
                                        "y": 20,
                                        "width": 30,
                                        "height": 25,
                                        "rectanglelabels": ["Object"],
                                    },
                                }
                            ]
                        }
                    ]
                },
                320,
                240,
            )

            self.assertEqual("prediction.rectanglelabels", prompt["source"])
            self.assertEqual("percent_xyxy", prompt["coordinate_system"])
            self.assertEqual([32.0, 48.0, 128.0, 108.0], prompt["box"])

    def test_segmentation_box_prompt_ignores_classification_choices_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "predictions": [
                        {
                            "result": [
                                {
                                    "type": "choices",
                                    "value": {"choices": ["Product"]},
                                }
                            ]
                        }
                    ]
                },
                320,
                240,
            )

            self.assertEqual("center_fallback", prompt["source"])

    def test_segmentation_box_prompt_falls_back_to_center_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {"data": {"image": "/data/local-files/?d=images/demo_blue.png"}},
                320,
                240,
            )

            self.assertEqual("center_fallback", prompt["source"])
            self.assertEqual("pixel_xyxy", prompt["coordinate_system"])
            for actual, expected in zip(prompt["box"], [57.6, 43.2, 262.4, 196.8]):
                self.assertAlmostEqual(expected, actual, places=3)

    def test_segmentation_box_prompt_falls_back_when_all_candidates_lack_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(Path(tmp))
            prompt = backend._segmentation_box_prompt_for_task(
                {
                    "data": {"candidates": [{"label": "Product"}, {"score": 0.9}]},
                    "meta": {"candidates": [{"value": {"choices": ["Other"]}}]},
                },
                320,
                240,
            )

            self.assertEqual("center_fallback", prompt["source"])
            for actual, expected in zip(prompt["box"], [57.6, 43.2, 262.4, 196.8]):
                self.assertAlmostEqual(expected, actual, places=3)

    def test_mobilesam_result_meta_records_candidate_prompt_source_and_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            checkpoint = tmp_dir / "mobile_sam.pt"
            checkpoint.write_text("fake checkpoint", encoding="utf-8")
            fake_mobile_sam = types.ModuleType("mobile_sam")

            class FakeModel:
                def to(self, device=None):
                    return self

            class FakePredictor:
                def __init__(self, model):
                    self.model = model

                def set_image(self, image):
                    pass

                def predict(self, box=None, multimask_output=False):
                    mask = [[1 for _ in range(4)] for _ in range(4)]
                    return [mask], [0.88], None

            fake_mobile_sam.sam_model_registry = {"vit_t": lambda checkpoint=None: FakeModel()}
            fake_mobile_sam.SamPredictor = FakePredictor

            with patch.dict(sys.modules, {"mobile_sam": fake_mobile_sam, "segment_anything": None}):
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
                                "id": "seg-candidate",
                                "data": {
                                    "image": "/data/local-files/?d=images/demo_blue.png",
                                    "candidates": [{"bbox": [25, 35, 120, 160], "score": 0.77}],
                                },
                            }
                        ],
                    }
                )

            meta = response["results"][0]["result"][0]["meta"]
            self.assertEqual("mobilesam", meta["backend"])
            self.assertEqual("mobilesam-image-segmentation", meta["prediction_source"])
            self.assertEqual("webhook_candidate.bbox", meta["prompt"])
            self.assertEqual([25.0, 35.0, 120.0, 160.0], meta["prompt_box"])

    def test_mobilesam_backend_returns_real_backend_prediction_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            checkpoint = tmp_dir / "mobile_sam.pt"
            checkpoint.write_text("fake checkpoint", encoding="utf-8")
            fake_mobile_sam = types.ModuleType("mobile_sam")

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
                    FakePredictor.last_box = [float(item) for item in box]
                    mask = [[0 for _ in range(10)] for _ in range(8)]
                    for y in range(2, 6):
                        for x in range(3, 8):
                            mask[y][x] = 1
                    return [mask], [0.91], None

            fake_mobile_sam.sam_model_registry = {
                "vit_t": lambda checkpoint=None: FakeModel(),
            }
            fake_mobile_sam.SamPredictor = FakePredictor

            with patch.dict(sys.modules, {"mobile_sam": fake_mobile_sam, "segment_anything": None}):
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
                                "data": {
                                    "image": "/data/local-files/?d=images/demo_blue.png",
                                    "bbox": [179, 47, 284, 118],
                                },
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
            self.assertEqual([179.0, 47.0, 284.0, 118.0], FakePredictor.last_box)
            self.assertEqual("data.bbox", prediction["confidence"]["prompt"])
            self.assertEqual("mobilesam", prediction["result"][0]["meta"]["backend"])
            self.assertEqual("data.bbox", prediction["result"][0]["meta"]["prompt"])

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
