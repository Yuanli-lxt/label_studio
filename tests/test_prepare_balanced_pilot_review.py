import json
import shutil
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.prepare_balanced_pilot_review import (
    prepare_balanced_pilot_review,
    validate_balanced_pilot_artifacts,
)


def queue_row(idx, *, image=None, current=None, learned=None, sample_id=None, annotation_id=None):
    current_score = float(current if current is not None else idx / 100.0)
    learned_score = float(learned if learned is not None else (100 - idx) / 100.0)
    return {
        "task_id": f"task-{idx}",
        "sample_id": sample_id if sample_id is not None else f"sample-{idx}",
        "image_id": f"image-{idx}",
        "annotation_id": annotation_id if annotation_id is not None else f"ann-{idx}",
        "prediction_id": f"pred-{idx}",
        "dataset": "unit",
        "category_name": "Object",
        "image": image or f"/data/local-files/?d=images/unit_{idx}.png",
        "gt_reference": f"/data/local-files/?d=gt/unit_{idx}.png",
        "mobilesam_preview": f"/data/local-files/?d=preview/unit_{idx}.png",
        "priority_score": current_score,
        "rank": idx,
        "prompt_bbox": [1, 2, 20, 30],
        "source_metadata": {
            "review": {"needs_review": idx % 2 == 0},
        },
        "mask_quality": {"mask_area_ratio": 0.1},
        "prediction_features": {"pred_area_ratio": 0.1},
        "shadow_scores": {
            "learned_boundary_shape_only_score": learned_score,
            "shadow_only": True,
        },
        "shadow_score_metadata": {
            "artifact_version": "unit-v1",
            "artifact_validation_status": "valid",
            "shadow_only": True,
            "affects_default_ranking": False,
        },
        "evaluation_only": {"delta": {"major_correction": idx % 3 == 0}},
        "boundary_metadata": {"thin_structure_score": 999},
        "gt_mask_path": "/private/gt.png",
        "labels": ["major"],
    }


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BalancedPilotReviewTests(unittest.TestCase):
    def test_builds_60_balanced_tasks_and_safe_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = []
            for idx in range(1, 81):
                rows.append(queue_row(idx, current=idx / 100, learned=(100 - idx) / 100))
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=15,
                pilot_id="unit_pilot",
            )

            self.assertEqual(60, result["rows"])
            self.assertEqual(
                {
                    "high_learned_low_current": 15,
                    "high_current_low_learned": 15,
                    "top_learned": 15,
                    "control_current_top": 15,
                },
                result["group_counts"],
            )
            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            self.assertEqual(60, len(tasks))
            self.assertEqual("/data/local-files/?d=images/unit_1.png", tasks[0]["data"]["image"])
            self.assertEqual("/data/local-files/?d=gt/unit_1.png", tasks[0]["data"]["gt_reference"])
            self.assertEqual("/data/local-files/?d=preview/unit_1.png", tasks[0]["data"]["mobilesam_preview"])
            self.assertEqual("/data/local-files/?d=gt/unit_1.png", tasks[0]["data"]["gt_mask_preview"])
            self.assertEqual("/data/local-files/?d=preview/unit_1.png", tasks[0]["data"]["mask_preview"])
            self.assertEqual([1, 2, 20, 30], tasks[0]["data"]["bbox"])
            self.assertEqual("unit", tasks[0]["data"]["dataset"])
            self.assertEqual("Object", tasks[0]["data"]["category_name"])
            self.assertEqual("high_learned_low_current", tasks[0]["data"]["review_group"])
            self.assertEqual("unit_pilot", tasks[0]["meta"]["shadow_review_pilot_id"])
            self.assertTrue(tasks[0]["meta"]["shadow_only"])
            self.assertFalse(tasks[0]["meta"]["affects_default_ranking"])

            validation = validate_balanced_pilot_artifacts(
                result["outputs"]["review_assignment"],
                result["outputs"]["label_studio_tasks"],
                per_group=15,
            )
            self.assertEqual([], validation["forbidden_field_hits"])

    def test_dedupes_across_groups_by_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = [queue_row(idx, current=idx / 100, learned=(100 - idx) / 100) for idx in range(1, 90)]
            rows[30]["sample_id"] = rows[0]["sample_id"]
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(str(queue), str(root / "pilot"), per_group=10)

            assignment_rows = [
                json.loads(line)
                for line in Path(result["outputs"]["review_assignment"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len({row["sample_id"] for row in assignment_rows}), len(assignment_rows))

    def test_fails_when_group_has_too_few_importable_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = [queue_row(idx, current=idx / 100, learned=(100 - idx) / 100) for idx in range(1, 20)]
            for row in rows:
                row["image"] = ""
            write_jsonl(queue, rows)

            with self.assertRaisesRegex(ValueError, "fewer than 15 importable rows"):
                prepare_balanced_pilot_review(str(queue), str(root / "pilot"), per_group=15)

    def test_cli_entrypoint_writes_expected_files(self):
        from image_segmentation.benchmark.prepare_balanced_pilot_review import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = [queue_row(idx, current=idx / 100, learned=(100 - idx) / 100) for idx in range(1, 81)]
            write_jsonl(queue, rows)

            code = main([
                "--review-queue", str(queue),
                "--output-dir", str(root / "pilot"),
                "--per-group", "15",
                "--pilot-id", "unit_pilot",
            ])

            self.assertEqual(0, code)
            self.assertTrue((root / "pilot" / "label_studio_tasks.json").exists())
            self.assertTrue((root / "pilot" / "review_assignment.jsonl").exists())
            self.assertTrue((root / "pilot" / "balanced_pilot_summary.json").exists())

    def test_generated_tasks_load_with_existing_segmentation_importer(self):
        from scripts.import_image_segmentation_review_tasks_to_label_studio import (
            load_segmentation_review_tasks,
            validate_tasks_for_summary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = [queue_row(idx, current=idx / 100, learned=(100 - idx) / 100) for idx in range(1, 81)]
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(str(queue), str(root / "pilot"), per_group=15)
            tasks = load_segmentation_review_tasks(Path(result["outputs"]["label_studio_tasks"]))
            valid, invalid = validate_tasks_for_summary(tasks)

            self.assertEqual(60, len(tasks))
            self.assertEqual(60, len(valid))
            self.assertEqual([], invalid)
            self.assertIn("review_group", valid[0]["meta"])

    def test_round2_outputs_independent_validation_templates_and_overlap_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = []
            for idx in range(1, 101):
                rows.append(queue_row(idx, current=idx / 100, learned=(100 - idx) / 100))
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "round2"),
                per_group=20,
                pilot_id="segmentation_balanced_pilot_round2_2026_06_07",
                validation_round="round2_independent",
                repeat_review_overlap=2,
                source_window_id="production_shadow_broader_observation/window_006",
                previous_assignment=str(queue),
            )

            self.assertEqual(80, result["rows"])
            self.assertEqual("round2_independent", result["validation_round"])
            self.assertEqual(2, result["repeat_review_overlap_per_group"])
            self.assertEqual("production_shadow_broader_observation/window_006", result["source_window_id"])
            self.assertEqual(
                {
                    "high_learned_low_current": 20,
                    "high_current_low_learned": 20,
                    "top_learned": 20,
                    "control_current_top": 20,
                },
                result["group_counts"],
            )
            self.assertEqual(
                [
                    "review_assignment",
                    "review_assignment_template_csv",
                    "label_studio_tasks",
                    "review_guidelines",
                    "summary_template",
                    "closure_report_template",
                    "artifact_validation",
                    "summary",
                ],
                list(result["outputs"].keys()),
            )

            assignment_rows = [
                json.loads(line)
                for line in Path(result["outputs"]["review_assignment"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("repeat_review_overlap", assignment_rows[0])
            self.assertIn("reviewer_secondary", assignment_rows[0])
            self.assertIn("reviewed_at_secondary", assignment_rows[0])
            self.assertIn("review_outcome_secondary", assignment_rows[0])
            self.assertEqual(8, sum(bool(row["repeat_review_overlap"]) for row in assignment_rows))
            self.assertIsNone(assignment_rows[0]["human_review_outcome"])
            self.assertIsNone(assignment_rows[0]["reviewer"])
            self.assertIsNone(assignment_rows[0]["reviewed_at"])

            guidelines = Path(result["outputs"]["review_guidelines"]).read_text(encoding="utf-8")
            for label in ("No fix", "Minor fix", "Major fix", "Redo", "Skip"):
                self.assertIn(label, guidelines)
            self.assertIn("production scoring inputs", guidelines)

            closure = Path(result["outputs"]["closure_report_template"]).read_text(encoding="utf-8")
            self.assertIn("Independent Validation Closure Report", closure)
            self.assertIn("Cannot claim production efficiency lift", closure)

            template = json.loads(Path(result["outputs"]["summary_template"]).read_text(encoding="utf-8"))
            self.assertEqual(80, template["planned_rows"])
            self.assertEqual(["no_fix", "minor_fix", "major_fix", "redo", "skip"], template["review_outcomes"])
            self.assertFalse(template["learned_boundary_shape_promoted"])

    def test_validates_local_file_refs_duplicates_safe_fields_and_shadow_only_flags(self):
        from image_segmentation.benchmark.prepare_balanced_pilot_review import validate_balanced_pilot_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "review_queue.shadow_scored.jsonl"
            local = root / "local-files"
            (local / "images").mkdir(parents=True)
            (local / "gt").mkdir(parents=True)
            (local / "preview").mkdir(parents=True)
            for idx in range(1, 101):
                (local / "images" / f"unit_{idx}.png").write_bytes(b"image")
                (local / "gt" / f"unit_{idx}.png").write_bytes(b"gt")
                (local / "preview" / f"unit_{idx}.png").write_bytes(b"preview")
            rows = [
                queue_row(
                    idx,
                    image=f"/data/local-files/?d=images/unit_{idx}.png",
                    current=idx / 100,
                    learned=(100 - idx) / 100,
                )
                for idx in range(1, 101)
            ]
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=20,
                pilot_id="unit_round2",
                local_files_root=str(local),
            )
            validation = validate_balanced_pilot_artifacts(
                assignment_path=result["outputs"]["review_assignment"],
                tasks_path=result["outputs"]["label_studio_tasks"],
                per_group=20,
                local_files_root=str(local),
            )

            self.assertTrue(validation["valid"])
            self.assertEqual([], validation["errors"])
            self.assertEqual(80, validation["task_count"])
            self.assertEqual(0, validation["missing_file_refs"])
            self.assertEqual(0, validation["duplicate_samples"])
            self.assertTrue(validation["shadow_only_all_true"])
            self.assertTrue(validation["affects_default_ranking_all_false"])
            self.assertTrue(validation["label_studio_segmentation_contract"])
            self.assertEqual(
                ["priority_score", "rank", "priority_bucket"],
                validation["default_ranking_fields_not_written"],
            )

    def test_localizes_image_refs_for_label_studio_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            for idx in range(1, 101):
                (source_dir / f"unit_{idx}.jpg").write_bytes(b"image")
            queue = root / "review_queue.shadow_scored.jsonl"
            rows = [
                queue_row(
                    idx,
                    image=str(source_dir / f"unit_{idx}.jpg"),
                    current=idx / 100,
                    learned=(100 - idx) / 100,
                )
                for idx in range(1, 101)
            ]
            for row in rows:
                row.pop("gt_reference", None)
                row.pop("mobilesam_preview", None)
            write_jsonl(queue, rows)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=20,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            self.assertTrue(tasks[0]["data"]["image"].startswith("/data/local-files/?d=unit_round2/"))
            self.assertEqual(tasks[0]["data"]["image"], tasks[0]["data"]["gt_reference"])
            self.assertEqual(tasks[0]["data"]["image"], tasks[0]["data"]["mobilesam_preview"])
            localized = root / "local-files" / tasks[0]["data"]["image"].split("?d=", 1)[1]
            self.assertTrue(localized.exists())

            validation = json.loads(Path(result["outputs"]["artifact_validation"]).read_text(encoding="utf-8"))
            self.assertTrue(validation["valid"])
            self.assertGreater(validation["preview_fallback_counts"]["gt_reference"], 0)

    def test_attaches_mobilesam_prediction_as_editable_preannotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            from PIL import Image

            rows = []
            manifests = []
            predictions = []
            for idx in range(1, 13):
                image_path = source_dir / f"unit_{idx}.png"
                Image.new("RGB", (4, 4), (idx, idx, idx)).save(image_path)
                rows.append(
                    queue_row(
                        idx,
                        image=str(image_path),
                        current=idx / 100,
                        learned=(100 - idx) / 100,
                        sample_id=f"LVIS_{idx}_{idx}",
                    )
                )
                rows[-1].pop("gt_reference", None)
                rows[-1].pop("mobilesam_preview", None)
                manifests.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "width": 4,
                        "height": 4,
                        "gt_segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                        "gt_iscrowd": 0,
                    }
                )
                predictions.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "model_version": "demo-rule-v61",
                        "result": {
                            "id": f"LVIS_{idx}_{idx}_mask",
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": 4,
                            "original_height": 4,
                            "value": {"format": "rle", "rle": [0, 4, 12], "brushlabels": ["Object"]},
                        },
                    }
                )
            queue = root / "review_queue.shadow_scored.jsonl"
            manifest_path = root / "manifest.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(queue, rows)
            write_jsonl(manifest_path, manifests)
            write_jsonl(prediction_path, predictions)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=1,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
                manifest_paths=[str(manifest_path)],
                prediction_paths=[str(prediction_path)],
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            self.assertTrue(tasks[0]["predictions"])
            prediction = tasks[0]["predictions"][0]
            self.assertEqual("demo-rule-v61", prediction["model_version"])
            self.assertEqual("mask_label", prediction["result"][0]["from_name"])
            self.assertEqual("image", prediction["result"][0]["to_name"])
            self.assertEqual("brushlabels", prediction["result"][0]["type"])
            self.assertIn("rle", prediction["result"][0]["value"])

    def test_can_override_prediction_display_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            from PIL import Image

            rows = []
            predictions = []
            for idx in range(1, 13):
                image_path = source_dir / f"unit_{idx}.png"
                Image.new("RGB", (4, 4), (idx, idx, idx)).save(image_path)
                rows.append(
                    queue_row(
                        idx,
                        image=str(image_path),
                        current=idx / 100,
                        learned=(100 - idx) / 100,
                        sample_id=f"LVIS_{idx}_{idx}",
                    )
                )
                rows[-1].pop("gt_reference", None)
                rows[-1].pop("mobilesam_preview", None)
                predictions.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "model_version": "demo-rule-v61",
                        "result": {
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": 4,
                            "original_height": 4,
                            "value": {"format": "rle", "rle": [0, 4, 12], "brushlabels": ["Object"]},
                        },
                    }
                )
            queue = root / "review_queue.shadow_scored.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(queue, rows)
            write_jsonl(prediction_path, predictions)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=1,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
                prediction_paths=[str(prediction_path)],
                prediction_display_model_version="mobilesam-seg-v0001",
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            self.assertEqual("mobilesam-seg-v0001", tasks[0]["predictions"][0]["model_version"])

    def test_can_render_previews_as_pixel_masks_instead_of_overlays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            from PIL import Image

            rows = []
            manifests = []
            predictions = []
            for idx in range(1, 13):
                image_path = source_dir / f"unit_{idx}.png"
                Image.new("RGB", (4, 4), (80, 80, 80)).save(image_path)
                rows.append(
                    queue_row(
                        idx,
                        image=str(image_path),
                        current=idx / 100,
                        learned=(100 - idx) / 100,
                        sample_id=f"LVIS_{idx}_{idx}",
                    )
                )
                rows[-1].pop("gt_reference", None)
                rows[-1].pop("mobilesam_preview", None)
                manifests.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "width": 4,
                        "height": 4,
                        "gt_segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]],
                        "gt_iscrowd": 0,
                    }
                )
                predictions.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "model_version": "demo-rule-v61",
                        "result": {
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": 4,
                            "original_height": 4,
                            "value": {"format": "rle", "rle": [0, 4, 12], "brushlabels": ["Object"]},
                        },
                    }
                )
            queue = root / "review_queue.shadow_scored.jsonl"
            manifest_path = root / "manifest.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(queue, rows)
            write_jsonl(manifest_path, manifests)
            write_jsonl(prediction_path, predictions)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=1,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
                manifest_paths=[str(manifest_path)],
                prediction_paths=[str(prediction_path)],
                preview_style="mask",
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            gt_ref = tasks[0]["data"]["gt_mask_preview"].split("?d=", 1)[1]
            pred_ref = tasks[0]["data"]["mask_preview"].split("?d=", 1)[1]
            gt_image = Image.open(root / "local-files" / gt_ref).convert("RGB")
            pred_image = Image.open(root / "local-files" / pred_ref).convert("RGB")
            self.assertIn((0, 0, 0), set(gt_image.getdata()))
            self.assertIn((34, 210, 160), set(gt_image.getdata()))
            self.assertIn((0, 0, 0), set(pred_image.getdata()))
            self.assertIn((255, 214, 30), set(pred_image.getdata()))

    def test_overlay_previews_draw_visible_mask_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            from PIL import Image

            rows = []
            manifests = []
            predictions = []
            for idx in range(1, 13):
                image_path = source_dir / f"unit_{idx}.png"
                Image.new("RGB", (6, 6), (90, 90, 90)).save(image_path)
                rows.append(
                    queue_row(
                        idx,
                        image=str(image_path),
                        current=idx / 100,
                        learned=(100 - idx) / 100,
                        sample_id=f"LVIS_{idx}_{idx}",
                    )
                )
                rows[-1].pop("gt_reference", None)
                rows[-1].pop("mobilesam_preview", None)
                manifests.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "width": 6,
                        "height": 6,
                        "gt_segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                        "gt_iscrowd": 0,
                    }
                )
                predictions.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "model_version": "demo-rule-v61",
                        "result": {
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": 6,
                            "original_height": 6,
                            "value": {"format": "rle", "rle": [7, 3, 3, 3, 3, 3, 17], "brushlabels": ["Object"]},
                        },
                    }
                )
            queue = root / "review_queue.shadow_scored.jsonl"
            manifest_path = root / "manifest.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(queue, rows)
            write_jsonl(manifest_path, manifests)
            write_jsonl(prediction_path, predictions)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=1,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
                manifest_paths=[str(manifest_path)],
                prediction_paths=[str(prediction_path)],
                preview_style="overlay",
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            gt_ref = tasks[0]["data"]["gt_mask_preview"].split("?d=", 1)[1]
            pred_ref = tasks[0]["data"]["mask_preview"].split("?d=", 1)[1]
            gt_image = Image.open(root / "local-files" / gt_ref).convert("RGB")
            pred_image = Image.open(root / "local-files" / pred_ref).convert("RGB")
            self.assertIn((34, 210, 160), set(gt_image.getdata()))
            self.assertIn((255, 214, 30), set(pred_image.getdata()))

    def test_can_render_gt_as_mask_and_mobilesam_as_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            from PIL import Image

            rows = []
            manifests = []
            predictions = []
            for idx in range(1, 13):
                image_path = source_dir / f"unit_{idx}.png"
                Image.new("RGB", (6, 6), (90, 90, 90)).save(image_path)
                rows.append(
                    queue_row(
                        idx,
                        image=str(image_path),
                        current=idx / 100,
                        learned=(100 - idx) / 100,
                        sample_id=f"LVIS_{idx}_{idx}",
                    )
                )
                rows[-1].pop("gt_reference", None)
                rows[-1].pop("mobilesam_preview", None)
                manifests.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "width": 6,
                        "height": 6,
                        "gt_segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                        "gt_iscrowd": 0,
                    }
                )
                predictions.append(
                    {
                        "dataset": "LVIS",
                        "sample_id": f"LVIS_{idx}_{idx}",
                        "model_version": "demo-rule-v61",
                        "result": {
                            "from_name": "mask_label",
                            "to_name": "image",
                            "type": "brushlabels",
                            "original_width": 6,
                            "original_height": 6,
                            "value": {"format": "rle", "rle": [7, 3, 3, 3, 3, 3, 17], "brushlabels": ["Object"]},
                        },
                    }
                )
            queue = root / "review_queue.shadow_scored.jsonl"
            manifest_path = root / "manifest.jsonl"
            prediction_path = root / "predictions.jsonl"
            write_jsonl(queue, rows)
            write_jsonl(manifest_path, manifests)
            write_jsonl(prediction_path, predictions)

            result = prepare_balanced_pilot_review(
                str(queue),
                str(root / "pilot"),
                per_group=1,
                pilot_id="unit_round2",
                local_files_root=str(root / "local-files"),
                local_files_subdir="unit_round2",
                manifest_paths=[str(manifest_path)],
                prediction_paths=[str(prediction_path)],
                gt_preview_style="mask",
                prediction_preview_style="overlay",
            )

            tasks = json.loads(Path(result["outputs"]["label_studio_tasks"]).read_text(encoding="utf-8"))
            gt_ref = tasks[0]["data"]["gt_mask_preview"].split("?d=", 1)[1]
            pred_ref = tasks[0]["data"]["mask_preview"].split("?d=", 1)[1]
            gt_image = Image.open(root / "local-files" / gt_ref).convert("RGB")
            pred_image = Image.open(root / "local-files" / pred_ref).convert("RGB")
            self.assertEqual({(0, 0, 0), (34, 210, 160)}, set(gt_image.getdata()))
            self.assertIn((90, 90, 90), set(pred_image.getdata()))
            self.assertIn((255, 214, 30), set(pred_image.getdata()))


if __name__ == "__main__":
    unittest.main()
