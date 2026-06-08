import json
from pathlib import Path

from image_segmentation.benchmark.summarize_gt_correction_effort import (
    correction_effort_from_iou,
    size_bucket_from_area,
    summarize_gt_correction_effort,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(dataset: str, iou: float, human_area_px: int, sample_id: str) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "delta": {
            "model_human_iou": iou,
            "model_human_dice": iou,
            "human_area_px": human_area_px,
        },
        "record_status": "ok",
    }


def test_correction_effort_thresholds_are_iou_derived():
    assert correction_effort_from_iou(0.95) == "No fix"
    assert correction_effort_from_iou(0.90) == "No fix"
    assert correction_effort_from_iou(0.80) == "Minor fix"
    assert correction_effort_from_iou(0.75) == "Minor fix"
    assert correction_effort_from_iou(0.60) == "Major fix"
    assert correction_effort_from_iou(0.50) == "Major fix"
    assert correction_effort_from_iou(0.49) == "Redo"


def test_size_bucket_uses_gt_area_px():
    assert size_bucket_from_area(32 * 32 - 1) == "small"
    assert size_bucket_from_area(32 * 32) == "medium"
    assert size_bucket_from_area(96 * 96 - 1) == "medium"
    assert size_bucket_from_area(96 * 96) == "large"


def test_summarizes_dataset_and_size_bucket_correction_effort(tmp_path):
    coco = tmp_path / "coco" / "correction_delta_dataset.jsonl"
    lvis = tmp_path / "lvis" / "correction_delta_dataset.jsonl"
    _write_jsonl(
        coco,
        [
            _row("COCO", 0.95, 500, "c1"),
            _row("COCO", 0.80, 2_000, "c2"),
            _row("COCO", 0.60, 10_000, "c3"),
        ],
    )
    _write_jsonl(
        lvis,
        [
            _row("LVIS", 0.40, 100, "l1"),
            _row("LVIS", 0.70, 800, "l2"),
        ],
    )

    summary = summarize_gt_correction_effort(
        runs=[f"COCO={coco.parent}", f"LVIS={lvis.parent}"],
        output_dir=str(tmp_path / "out"),
    )

    assert summary["datasets"]["COCO"]["total"] == 3
    assert summary["datasets"]["COCO"]["major_or_redo"] == 1
    assert summary["datasets"]["COCO"]["major_or_redo_rate"] == 1 / 3
    assert summary["datasets"]["LVIS"]["major_or_redo"] == 2
    assert summary["datasets"]["LVIS"]["size_buckets"]["small"]["redo"] == 1
    assert summary["datasets"]["LVIS"]["size_buckets"]["small"]["major_or_redo_rate"] == 1.0
    assert summary["total"]["total"] == 5
    markdown_path = tmp_path / "out" / "gt_derived_correction_effort_summary.md"
    assert (tmp_path / "out" / "gt_derived_correction_effort_summary.json").exists()
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Key Findings" in markdown
    assert "highest overall Major/Redo rate" in markdown
