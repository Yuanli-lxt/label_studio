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


if __name__ == "__main__":
    unittest.main()

