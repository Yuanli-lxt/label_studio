import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_segmentation.benchmark.preflight_backend import preflight_backend


class BenchmarkBackendPreflightTests(unittest.TestCase):
    def test_mobile_sam_checkpoint_missing_clear_error(self):
        result = preflight_backend(
            "mobile_sam",
            checkpoint="/tmp/definitely_missing_mobile_sam.pt",
            construct_predictor=False,
        )
        self.assertEqual("failed", result["status"])
        self.assertIn("IMAGE_SEG_CHECKPOINT does not exist", result["error"])

    def test_preflight_backend_reports_missing_checkpoint(self):
        result = preflight_backend(
            "mobile_sam",
            checkpoint="/tmp/not_here_mobile_sam.pt",
            construct_predictor=False,
        )
        self.assertEqual("mobile_sam", result["backend"])
        self.assertIn("/tmp/not_here_mobile_sam.pt", result["checkpoint_path"])
        self.assertIsNotNone(result["error"])

    def test_mobile_sam_backend_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mobile_sam.pt"
            checkpoint.write_bytes(b"fake")
            with patch("image_segmentation.benchmark.preflight_backend._dependency_error", return_value=None):
                result = preflight_backend("mobile_sam", checkpoint=str(checkpoint), construct_predictor=False)
        self.assertEqual("ok", result["status"])
        self.assertEqual("mobile_sam", result["backend"])
        self.assertIn("checkpoint_path", result)
        self.assertIn("device", result)
        self.assertIn("model_class", result)

    def test_preflight_backend_cuda_unavailable_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mobile_sam.pt"
            checkpoint.write_bytes(b"fake")
            with patch("image_segmentation.benchmark.preflight_backend._dependency_error", return_value=None), patch(
                "image_segmentation.benchmark.preflight_backend._device_info",
                return_value={
                    "requested_device": "cuda",
                    "resolved_device": None,
                    "cuda_available": False,
                    "cuda_device_name": None,
                    "cuda_memory": None,
                    "error": "IMAGE_SEG_DEVICE=cuda was requested, but torch.cuda.is_available() is false",
                },
            ):
                result = preflight_backend("mobile_sam", checkpoint=str(checkpoint), device="cuda")
        self.assertEqual("failed", result["status"])
        self.assertIn("torch.cuda.is_available() is false", result["error"])

    def test_preflight_backend_reports_resolved_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mobile_sam.pt"
            checkpoint.write_bytes(b"fake")
            with patch("image_segmentation.benchmark.preflight_backend._dependency_error", return_value=None), patch(
                "image_segmentation.benchmark.preflight_backend._device_info",
                return_value={
                    "requested_device": "cuda",
                    "resolved_device": "cuda",
                    "cuda_available": True,
                    "cuda_device_name": "Fake GPU",
                    "cuda_memory": {"total_memory": 1, "allocated": 0, "reserved": 0},
                    "error": None,
                },
            ):
                result = preflight_backend(
                    "mobile_sam",
                    checkpoint=str(checkpoint),
                    device="cuda",
                    construct_predictor=False,
                )
        self.assertEqual("ok", result["status"])
        self.assertEqual("cuda", result["requested_device"])
        self.assertEqual("cuda", result["resolved_device"])
        self.assertTrue(result["cuda_available"])

    def test_mobile_sam_no_silent_cpu_fallback_when_cuda_requested(self):
        class Param:
            device = "cpu"

        class Model:
            def parameters(self):
                return iter([Param()])

        class Predictor:
            model = Model()

        class App:
            def _load_image_segmentation_predictor(self, backend):
                return {"predictor": Predictor()}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mobile_sam.pt"
            checkpoint.write_bytes(b"fake")
            with patch("image_segmentation.benchmark.preflight_backend._dependency_error", return_value=None), patch(
                "image_segmentation.benchmark.preflight_backend._device_info",
                return_value={
                    "requested_device": "cuda",
                    "resolved_device": "cuda",
                    "cuda_available": True,
                    "cuda_device_name": "Fake GPU",
                    "cuda_memory": {"total_memory": 1, "allocated": 0, "reserved": 0},
                    "error": None,
                },
            ), patch("image_segmentation.benchmark.preflight_backend._load_backend_app", return_value=App()):
                result = preflight_backend("mobile_sam", checkpoint=str(checkpoint), device="cuda")
        self.assertEqual("failed", result["status"])
        self.assertIn("not on cuda", result["error"])


if __name__ == "__main__":
    unittest.main()
