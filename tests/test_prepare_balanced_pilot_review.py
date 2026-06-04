import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.prepare_balanced_pilot_review import (
    FORBIDDEN_REVIEW_FIELDS,
    prepare_balanced_pilot_review,
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
            self.assertEqual([1, 2, 20, 30], tasks[0]["data"]["bbox"])
            self.assertEqual("unit_pilot", tasks[0]["meta"]["shadow_review_pilot_id"])
            self.assertTrue(tasks[0]["meta"]["shadow_only"])
            self.assertFalse(tasks[0]["meta"]["affects_default_ranking"])

            combined = "\n".join(
                Path(path).read_text(encoding="utf-8")
                for path in result["outputs"].values()
                if str(path).endswith((".json", ".jsonl", ".csv", ".md"))
            )
            for forbidden in FORBIDDEN_REVIEW_FIELDS:
                self.assertNotIn(forbidden, combined)

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


if __name__ == "__main__":
    unittest.main()
