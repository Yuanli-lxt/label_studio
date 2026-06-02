import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_segmentation.benchmark.evaluation import build_evaluation_report
from image_segmentation.benchmark.run_benchmark import _minimal_rle, run_benchmark
from segmentation_uncertainty import generate_bbox_prompt_variants


def write_manifest(root: Path, bbox=None) -> Path:
    image_path = root / "tiny.jpg"
    Image.new("RGB", (20, 20), "white").save(image_path)
    bbox = bbox or [5.0, 5.0, 13.0, 13.0]
    row = {
        "benchmark_id": "benchmark_v0_1",
        "dataset": "COCO",
        "sample_id": "COCO_1_2",
        "image_id": "1",
        "annotation_id": "2",
        "image_path": str(image_path),
        "width": 20,
        "height": 20,
        "category_id": "1",
        "category_name": "thing",
        "gt_bbox_xyxy": bbox,
        "gt_area": 64,
        "gt_area_ratio": 64 / 400,
        "gt_touches_border": False,
        "difficulty_tags": ["medium_object"],
        "split": "benchmark_v0_1",
        "gt_segmentation": [[5, 5, 13, 5, 13, 13, 5, 13]],
        "gt_iscrowd": 0,
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def write_dis5k_manifest(root: Path) -> Path:
    image_path = root / "tiny.jpg"
    mask_path = root / "tiny.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:13, 5:13] = 255
    Image.fromarray(mask).save(mask_path)
    row = {
        "benchmark_id": "benchmark_v0_1_dis5k300",
        "dataset": "DIS5K",
        "sample_id": "DIS5K_tiny",
        "image_id": "tiny",
        "annotation_id": "tiny",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "width": 20,
        "height": 20,
        "category_id": "foreground_object",
        "category_name": "foreground_object",
        "gt_bbox_xyxy": [5.0, 5.0, 13.0, 13.0],
        "gt_area": 64,
        "gt_area_ratio": 64 / 400,
        "gt_touches_border": False,
        "difficulty_tags": ["medium_object", "thin_structure"],
        "boundary_metadata": {"perimeter_px": 28.0},
        "split": "benchmark_v0_1_dis5k300",
    }
    manifest = root / "manifest_dis5k.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


class FakeMobileSamApp:
    def __init__(self, fallback=False, shape=(20, 20)):
        self.fallback = fallback
        self.shape = shape

    def _predict(self, payload):
        task = payload["tasks"][0]
        height, width = self.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[5:min(13, height), 5:min(13, width)] = 1
        source = "placeholder-image-segmentation" if self.fallback else "mobilesam-image-segmentation"
        meta = {
            "prediction_source": source,
            "rle_encoder": "fallback-minimal",
            "backend": "mobilesam",
            "prompt": "benchmark.gt_bbox",
            "prompt_box": task["data"]["bbox"],
            "prompt_coordinate_system": "pixel_xyxy",
            "mask_bbox": [5.0, 5.0, float(min(13, width)), float(min(13, height))],
        }
        if self.fallback:
            meta["fallback"] = "placeholder"
        return {
            "results": [
                {
                    "model_version": "mobilesam-test-v1",
                    "score": 0.9,
                    "prediction_source": source,
                    "confidence": {"fallback": "placeholder"} if self.fallback else {"backend": "mobilesam"},
                    "result": [
                        {
                            "id": f"{task['id']}_mask",
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": width,
                            "original_height": height,
                            "value": {"format": "rle", "rle": _minimal_rle(mask), "brushlabels": ["thing"]},
                            "meta": meta,
                        }
                    ],
                }
            ]
        }


class BenchmarkRunSafetyTests(unittest.TestCase):
    def test_unknown_backend_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_manifest(Path(tmp))
            with self.assertRaisesRegex(ValueError, "unknown benchmark backend"):
                run_benchmark(str(manifest), str(Path(tmp) / "out"), backend="not_a_backend")

    def test_backend_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("bbox_rect", row["backend_requested"])
            self.assertEqual("bbox_rect", row["backend_resolved"])
            self.assertEqual("bbox-rect-benchmark-baseline", row["prediction_source"])
            self.assertEqual("benchmark-bbox-rect-v1", row["model_version"])

    def test_prompt_box_within_image_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            prompt = row["result"]["meta"]["prompt_box"]
            self.assertGreaterEqual(prompt[0], 0)
            self.assertGreaterEqual(prompt[1], 0)
            self.assertLessEqual(prompt[2], row["result"]["original_width"])
            self.assertLessEqual(prompt[3], row["result"]["original_height"])

    def test_prediction_and_gt_mask_same_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            with patch(
                "image_segmentation.benchmark.run_benchmark.segmentation_to_mask",
                return_value=np.zeros((10, 10), dtype=np.uint8),
            ):
                with self.assertRaisesRegex(ValueError, "prediction and GT mask shape mismatch"):
                    run_benchmark(str(manifest), str(root / "out"), backend="bbox_rect")

    def test_mask_quality_receives_prompt_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            row = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            mask_quality = row["mask_quality"]
            self.assertIsNotNone(mask_quality["prompt_bbox"])
            self.assertIsNotNone(mask_quality["bbox_iou_prompt_mask"])
            self.assertGreaterEqual(mask_quality["bbox_iou_prompt_mask"], 0.0)

    def test_degenerate_ranking_metrics_warning(self):
        queue = [
            {"evaluation_only": {"delta": {"major_correction": True}}},
            {"evaluation_only": {"delta": {"major_correction": True}}},
        ]
        records = [
            {"dataset": "COCO", "difficulty_tags": ["small"], "delta": {"major_correction": True}},
            {"dataset": "COCO", "difficulty_tags": ["small"], "delta": {"major_correction": True}},
        ]
        report = build_evaluation_report(queue, records)
        self.assertTrue(report["degenerate_major_correction_labels"])
        self.assertIn("Degenerate major_correction labels", report["metric_warnings"][0])
        self.assertIsNone(report["review_queue_effectiveness"]["average_precision_for_major_correction"])
        self.assertIsNone(report["review_queue_effectiveness"]["lift_at_10_percent_over_random"])

    def test_mobile_sam_backend_registered_or_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_manifest(Path(tmp))
            with patch("image_segmentation.benchmark.run_benchmark._load_backend_app", return_value=FakeMobileSamApp()):
                result = run_benchmark(str(manifest), str(Path(tmp) / "out"), backend="mobile_sam")
            self.assertEqual(1, result["n_samples"])

    def test_mobile_sam_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_manifest(Path(tmp))
            with patch(
                "image_segmentation.benchmark.run_benchmark._load_backend_app",
                return_value=FakeMobileSamApp(fallback=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "fell back|failed"):
                    run_benchmark(str(manifest), str(Path(tmp) / "out"), backend="mobile_sam")

    def test_mobile_sam_prediction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            with patch("image_segmentation.benchmark.run_benchmark._load_backend_app", return_value=FakeMobileSamApp()):
                run_benchmark(str(manifest), str(root / "out"), backend="mobile_sam")
            row = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("mobile_sam", row["backend_requested"])
            self.assertEqual("mobile_sam", row["backend_resolved"])
            self.assertEqual("mobilesam-image-segmentation", row["prediction_source"])
            self.assertIn("model_config", row["result"]["meta"])

    def test_mobile_sam_mask_shape_matches_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            with patch("image_segmentation.benchmark.run_benchmark._load_backend_app", return_value=FakeMobileSamApp()):
                run_benchmark(str(manifest), str(root / "out"), backend="mobile_sam")
            row = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(20, row["result"]["original_width"])
            self.assertEqual(20, row["result"]["original_height"])

    def test_mobile_sam_prompt_bbox_passed_to_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            with patch("image_segmentation.benchmark.run_benchmark._load_backend_app", return_value=FakeMobileSamApp()):
                run_benchmark(str(manifest), str(root / "out"), backend="mobile_sam")
            row = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNotNone(row["mask_quality"]["prompt_bbox"])
            self.assertIsNotNone(row["mask_quality"]["bbox_iou_prompt_mask"])

    def test_dis5k_benchmark_outputs_boundary_metrics_with_mock_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_dis5k_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            for name in [
                "predictions.jsonl",
                "correction_delta_dataset.jsonl",
                "review_queue.jsonl",
                "evaluation_report.json",
                "evaluation_report.md",
                "runtime_metadata.json",
            ]:
                self.assertTrue((out / name).exists())
            delta = json.loads((out / "correction_delta_dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])["delta"]
            self.assertIn("boundary", delta)
            report = json.loads((out / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertIn("boundary_quality", report)
            self.assertIn("Boundary Quality", (out / "evaluation_report.md").read_text(encoding="utf-8"))

    def test_run_benchmark_writes_prediction_time_shape_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            pred = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            delta = json.loads((out / "correction_delta_dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])
            queue = json.loads((out / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()[0])
            for row in [pred, delta]:
                features = row["prediction_features"]
                self.assertIn("pred_boundary_complexity", features)
                self.assertIn("pred_component_count", features)
                self.assertIn("pred_thinness_proxy", features)
            self.assertIn("boundary_shape_score", queue["score_components"])
            self.assertIn("prediction_features", queue["source_metadata"])

    def test_uncertainty_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            run_benchmark(str(manifest), str(root / "out"), backend="bbox_rect", enable_prompt_stability=True)
            row = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("uncertainty", row)
            self.assertTrue(row["uncertainty"]["enabled"])

    def test_uncertainty_score_used_by_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            run_benchmark(str(manifest), str(root / "out"), backend="bbox_rect", enable_prompt_stability=True)
            row = json.loads((root / "out" / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("uncertainty_score", row["score_components"])

    def test_run_benchmark_max_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            first = manifest.read_text(encoding="utf-8")
            manifest.write_text(first + first.replace("COCO_1_2", "COCO_1_3").replace('"annotation_id": "2"', '"annotation_id": "3"'), encoding="utf-8")
            result = run_benchmark(str(manifest), str(root / "out"), backend="bbox_rect", max_samples=1)
            self.assertEqual(1, result["n_samples"])

    def test_run_benchmark_resume_skips_existing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            with patch("image_segmentation.benchmark.run_benchmark._predict_sample") as predict:
                run_benchmark(str(manifest), str(out), backend="bbox_rect", resume=True)
            predict.assert_not_called()

    def test_run_benchmark_resume_deduplicates_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(
                text + text.replace('"annotation_id": "2"', '"annotation_id": "2"'),
                encoding="utf-8",
            )
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect", resume=True)
            rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, len({row["sample_id"] for row in rows}))

    def test_run_benchmark_resume_regenerates_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            (out / "evaluation_report.json").unlink()
            run_benchmark(str(manifest), str(out), backend="bbox_rect", resume=True)
            self.assertTrue((out / "evaluation_report.json").exists())
            self.assertTrue((out / "evaluation_report.md").exists())

    def test_run_benchmark_coco1000_outputs_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "benchmark_v0_1_coco1000_mobile_sam"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            for name in [
                "predictions.jsonl",
                "correction_delta_dataset.jsonl",
                "review_queue.jsonl",
                "evaluation_report.json",
                "evaluation_report.md",
            ]:
                self.assertTrue((out / name).exists(), name)

    def test_runtime_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            run_benchmark(str(manifest), str(out), backend="bbox_rect")
            self.assertTrue((out / "runtime_metadata.json").exists())
            payload = json.loads((out / "runtime_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(1, payload["n_samples_completed"])
            self.assertIn("elapsed_seconds", payload)

    def test_runtime_metadata_contains_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            with patch(
                "image_segmentation.benchmark.run_benchmark._runtime_device_info",
                return_value={
                    "requested_device": "cuda",
                    "resolved_device": "cuda",
                    "cuda_available": True,
                    "cuda_device_name": "Fake GPU",
                },
            ):
                run_benchmark(str(manifest), str(out), backend="bbox_rect")
            payload = json.loads((out / "runtime_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("cuda", payload["requested_device"])
            self.assertEqual("cuda", payload["resolved_device"])
            self.assertEqual("Fake GPU", payload["cuda_device_name"])

    def test_mobile_sam_cuda_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            out = root / "out"
            with patch("image_segmentation.benchmark.run_benchmark._load_backend_app", return_value=FakeMobileSamApp()), patch(
                "image_segmentation.benchmark.run_benchmark._runtime_device_info",
                return_value={
                    "requested_device": "cuda",
                    "resolved_device": "cuda",
                    "cuda_available": True,
                    "cuda_device_name": "Fake GPU",
                },
            ):
                run_benchmark(str(manifest), str(out), backend="mobile_sam")
            payload = json.loads((out / "runtime_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("mobile_sam", payload["backend_resolved"])
            self.assertEqual("cuda", payload["resolved_device"])

    def test_prompt_perturbations_within_bounds(self):
        variants = generate_bbox_prompt_variants([0, 0, 5, 5], 20, 20, jitter_ratio=0.2)
        self.assertTrue(variants)
        for variant in variants:
            x1, y1, x2, y2 = variant["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 19)
            self.assertLessEqual(y2, 19)


if __name__ == "__main__":
    unittest.main()
