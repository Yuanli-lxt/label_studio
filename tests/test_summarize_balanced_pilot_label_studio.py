import json
import tempfile
import unittest
from pathlib import Path

from image_segmentation.benchmark.summarize_balanced_pilot_label_studio import (
    GROUPS,
    summarize_balanced_pilot_reviews,
)


def assignment_row(task_id, group, outcome=None):
    return {
        "task_id": task_id,
        "sample_id": task_id,
        "group": group,
        "human_review_outcome": outcome,
    }


def label_studio_task(task_id, group, choice):
    return {
        "id": task_id,
        "data": {"image": f"/data/local-files/?d={task_id}.jpg"},
        "meta": {
            "sample_id": task_id,
            "review_group": group,
            "shadow_review_pilot_id": "unit_pilot",
        },
        "annotations": [
            {
                "id": f"{task_id}-annotation",
                "completed_by": {"email": "reviewer@example.com"},
                "created_at": "2026-06-05T00:00:00Z",
                "result": [
                    {
                        "from_name": "review_outcome",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": [choice]},
                    }
                ],
            }
        ],
    }


def legacy_coco_label_studio_task(label_studio_id, task_id, group, choice):
    task = label_studio_task(label_studio_id, group, choice)
    task["data"] = {
        "image": f"/data/local-files/?d=first_round/{task_id}.jpg",
        "review_group": group,
        "dataset": "COCO",
    }
    task["meta"] = {
        "sample_id": None,
        "prediction_id": f"{task_id}_mask",
        "review_group": group,
        "shadow_review_pilot_id": "segmentation_balanced_pilot_2026_06_04",
    }
    task["annotations"][0]["result"][0]["from_name"] = "correction_effort"
    return task


class SummarizeBalancedPilotLabelStudioTests(unittest.TestCase):
    def test_extracts_label_studio_review_outcomes_and_counts_by_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignment = root / "review_assignment.jsonl"
            export = root / "label_studio_export.json"
            out = root / "summary"

            rows = [
                assignment_row("a1", "high_learned_low_current"),
                assignment_row("a2", "high_learned_low_current"),
                assignment_row("b1", "high_current_low_learned"),
                assignment_row("c1", "top_learned"),
                assignment_row("d1", "control_current_top"),
            ]
            assignment.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            export.write_text(
                json.dumps(
                    [
                        label_studio_task("a1", "high_learned_low_current", "Major fix"),
                        label_studio_task("a2", "high_learned_low_current", "Redo"),
                        label_studio_task("b1", "high_current_low_learned", "No fix"),
                        label_studio_task("c1", "top_learned", "Minor fix"),
                        label_studio_task("d1", "control_current_top", "Skip"),
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_balanced_pilot_reviews(
                assignment_path=str(assignment),
                output_dir=str(out),
                label_studio_export_path=str(export),
                pilot_id="unit_pilot",
            )

            self.assertEqual(GROUPS, list(summary["groups"]))
            self.assertEqual(5, summary["reviewed_total"])
            self.assertEqual(2, summary["groups"]["high_learned_low_current"]["total"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["major_fix"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["redo"])
            self.assertEqual(1.0, summary["groups"]["high_learned_low_current"]["major_or_redo_rate"])
            self.assertEqual(1, summary["groups"]["high_current_low_learned"]["no_fix"])
            self.assertEqual(1, summary["groups"]["top_learned"]["minor_fix"])
            self.assertEqual(1, summary["groups"]["control_current_top"]["skip"])
            self.assertTrue(summary["shadow_only"])
            self.assertFalse(summary["affects_default_ranking"])
            self.assertTrue((out / "balanced_pilot_review_summary.json").exists())
            self.assertTrue((out / "balanced_pilot_review_summary.md").exists())

    def test_uses_assignment_outcomes_when_no_label_studio_export_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignment = root / "review_assignment.jsonl"
            out = root / "summary"
            rows = [
                assignment_row("a1", "high_learned_low_current", "major_correction_needed"),
                assignment_row("a2", "high_learned_low_current", "ok"),
                assignment_row("b1", "high_current_low_learned", "redo"),
            ]
            assignment.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = summarize_balanced_pilot_reviews(
                assignment_path=str(assignment),
                output_dir=str(out),
                pilot_id="unit_pilot",
            )

            self.assertEqual(2, summary["groups"]["high_learned_low_current"]["total"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["major_fix"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["no_fix"])
            self.assertEqual(1, summary["groups"]["high_current_low_learned"]["redo"])
            self.assertEqual(0, summary["groups"]["top_learned"]["total"])

    def test_matches_round2_label_studio_ids_when_sample_id_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignment = root / "review_assignment.jsonl"
            export = root / "label_studio_export.json"
            out = root / "summary"
            rows = [
                {
                    "task_id": "LVIS_130613_34386",
                    "sample_id": None,
                    "group": "high_learned_low_current",
                    "human_review_outcome": None,
                }
            ]
            assignment.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            task = label_studio_task(
                "segmentation_balanced_pilot_round2_2026_06_07:high_learned_low_current:task_id:LVIS_130613_34386",
                "high_learned_low_current",
                "Major fix",
            )
            task["meta"]["sample_id"] = None
            export.write_text(json.dumps([task]), encoding="utf-8")

            summary = summarize_balanced_pilot_reviews(
                assignment_path=str(assignment),
                output_dir=str(out),
                label_studio_export_path=str(export),
                pilot_id="unit_round2",
            )

            self.assertEqual(1, summary["reviewed_total"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["major_fix"])

    def test_matches_legacy_coco_project_by_prediction_id_and_correction_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignment = root / "review_assignment.jsonl"
            export = root / "label_studio_export.json"
            out = root / "summary"
            rows = [
                {
                    "task_id": "COCO_29675_1069835",
                    "sample_id": None,
                    "group": "high_learned_low_current",
                    "human_review_outcome": None,
                }
            ]
            assignment.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            export.write_text(
                json.dumps(
                    [
                        legacy_coco_label_studio_task(
                            14,
                            "COCO_29675_1069835",
                            "high_learned_low_current",
                            "Minor fix",
                        )
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_balanced_pilot_reviews(
                assignment_path=str(assignment),
                output_dir=str(out),
                label_studio_export_path=str(export),
                pilot_id="unit_coco",
            )

            self.assertEqual(1, summary["reviewed_total"])
            self.assertEqual(1, summary["groups"]["high_learned_low_current"]["minor_fix"])


if __name__ == "__main__":
    unittest.main()
