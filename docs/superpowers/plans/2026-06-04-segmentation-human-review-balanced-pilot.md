# Segmentation Human Review Balanced Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 60-task balanced Label Studio human review pilot for segmentation shadow scoring, with four groups of 15 samples and strict shadow-only safety.

**Architecture:** Add a focused benchmark CLI that reads a shadow-scored review queue, builds balanced groups, writes an offline review assignment, and writes a Label Studio import JSON. The CLI will use only production-safe metadata in task `meta`, keep `image` and optional prompt `bbox` in task `data`, and fail if it cannot produce 15 valid rows per group. Existing Label Studio bootstrap/import and `summarize_shadow_human_feedback` remain the downstream workflow.

**Tech Stack:** Python stdlib (`argparse`, `json`, `csv`, `pathlib`), existing benchmark JSONL conventions, existing Label Studio import helper, `pytest`/`unittest`.

---

## File Structure

- Create `image_segmentation/benchmark/prepare_balanced_pilot_review.py`
  - Pure functions for group ranking, dedupe, safety filtering, Label Studio task conversion, assignment CSV/JSONL writing, and CLI entrypoint.
- Create `tests/test_prepare_balanced_pilot_review.py`
  - Unit tests for balanced selection, safe fields, Label Studio task shape, dedupe, and failure cases.
- Modify `README.zh.md`
  - Add a short runbook section for the 60-task balanced pilot and shadow-only constraints.
- Modify `README.md`
  - Add matching concise English runbook notes.

Do not modify default Layer 5 weights, `services/trainer/segmentation_review_queue.py`, or shadow scoring logic.

## Task 1: Balanced Pilot Selection And Safe Outputs

**Files:**
- Create: `tests/test_prepare_balanced_pilot_review.py`
- Create: `image_segmentation/benchmark/prepare_balanced_pilot_review.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_balanced_pilot_review.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_prepare_balanced_pilot_review.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'image_segmentation.benchmark.prepare_balanced_pilot_review'`.

- [ ] **Step 3: Implement the balanced pilot CLI**

Create `image_segmentation/benchmark/prepare_balanced_pilot_review.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GROUPS = [
    "high_learned_low_current",
    "high_current_low_learned",
    "top_learned",
    "control_current_top",
]

FORBIDDEN_REVIEW_FIELDS = {
    "evaluation_only",
    "boundary_metadata",
    "gt_mask_path",
    "gt_mask",
    "gt_path",
    "iou",
    "dice",
    "delta",
    "severity",
    "major_correction",
    "labels",
}


def prepare_balanced_pilot_review(
    review_queue: str,
    output_dir: str,
    per_group: int = 15,
    pilot_id: str = "segmentation_balanced_pilot_2026_06_04",
) -> dict:
    rows = [_safe_candidate(row) for row in _read_jsonl(Path(review_queue))]
    rows = [row for row in rows if _is_importable(row)]
    groups = _candidate_groups(rows)
    assignment = _balanced_assignment(groups, int(per_group))
    _assert_group_counts(assignment, int(per_group))
    tasks = [_label_studio_task(row, pilot_id) for row in assignment]
    _assert_safe_payload(assignment)
    _assert_safe_payload(tasks)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "review_assignment.jsonl", assignment)
    _write_csv(output / "review_assignment_template.csv", assignment)
    _write_json(output / "label_studio_tasks.json", tasks)
    (output / "review_guidelines.md").write_text(_guidelines(), encoding="utf-8")
    summary = {
        "input_review_queue": review_queue,
        "output_dir": str(output),
        "pilot_id": pilot_id,
        "per_group": int(per_group),
        "rows": len(assignment),
        "group_counts": {group: sum(row.get("group") == group for row in assignment) for group in GROUPS},
        "safe_fields_only": True,
        "offline_analysis_only": True,
        "shadow_only": True,
        "affects_default_ranking": False,
        "outputs": {
            "review_assignment": str(output / "review_assignment.jsonl"),
            "review_assignment_template_csv": str(output / "review_assignment_template.csv"),
            "label_studio_tasks": str(output / "label_studio_tasks.json"),
            "review_guidelines": str(output / "review_guidelines.md"),
            "summary": str(output / "balanced_pilot_summary.json"),
        },
    }
    _write_json(output / "balanced_pilot_summary.json", summary)
    return summary


def _safe_candidate(row: dict) -> dict:
    metadata = row.get("shadow_score_metadata") if isinstance(row.get("shadow_score_metadata"), dict) else {}
    return {
        "task_id": row.get("task_id"),
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "prediction_id": row.get("prediction_id"),
        "dataset": row.get("dataset"),
        "category_name": row.get("category_name"),
        "image": row.get("image"),
        "bbox": _prompt_bbox(row),
        "current_priority_score": _num(row.get("priority_score")),
        "learned_shadow_score": _shadow_value(row, "learned_boundary_shape_only_score"),
        "rank_current": row.get("rank"),
        "rank_learned": None,
        "rank_delta": None,
        "prediction_time_safe_features": _safe_prediction_features(row),
        "artifact_version": metadata.get("artifact_version"),
        "shadow_metadata_status": metadata.get("artifact_validation_status"),
        "shadow_only": metadata.get("shadow_only") is True,
        "affects_default_ranking": metadata.get("affects_default_ranking") is True,
        "human_review_outcome": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": None,
    }


def _prompt_bbox(row: dict) -> list[float] | None:
    for key in ("prompt_bbox", "prompt_box", "bbox"):
        value = row.get(key)
        if isinstance(value, list) and len(value) == 4:
            return value
    mask_quality = row.get("mask_quality") if isinstance(row.get("mask_quality"), dict) else {}
    value = mask_quality.get("prompt_bbox") or mask_quality.get("prompt_box")
    if isinstance(value, list) and len(value) == 4:
        return value
    source = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
    quality = source.get("mask_quality") if isinstance(source.get("mask_quality"), dict) else {}
    value = quality.get("prompt_bbox") or quality.get("prompt_box")
    return value if isinstance(value, list) and len(value) == 4 else None


def _safe_prediction_features(row: dict) -> dict:
    features = row.get("prediction_features") if isinstance(row.get("prediction_features"), dict) else {}
    if not features:
        quality = row.get("mask_quality") if isinstance(row.get("mask_quality"), dict) else {}
        features = quality.get("prediction_time_boundary_shape") if isinstance(quality.get("prediction_time_boundary_shape"), dict) else {}
    return {str(key): value for key, value in features.items() if str(key).startswith("pred_")}


def _candidate_groups(rows: list[dict]) -> dict[str, list[dict]]:
    current_rank = _rank_map(rows, "current_priority_score")
    learned_rank = _rank_map(rows, "learned_shadow_score")
    enriched = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["rank_current"] = current_rank[idx] + 1
        out["rank_learned"] = learned_rank[idx] + 1
        out["rank_delta"] = out["rank_current"] - out["rank_learned"]
        enriched.append(out)
    return {
        "high_learned_low_current": sorted(
            [row for row in enriched if _num(row.get("rank_delta")) > 0],
            key=lambda row: (-_num(row.get("rank_delta")), row.get("rank_learned") or 10**9),
        ),
        "high_current_low_learned": sorted(
            [row for row in enriched if _num(row.get("rank_delta")) < 0],
            key=lambda row: (_num(row.get("rank_delta")), row.get("rank_current") or 10**9),
        ),
        "top_learned": sorted(enriched, key=lambda row: (-_num(row.get("learned_shadow_score")), row.get("rank_learned") or 10**9)),
        "control_current_top": sorted(enriched, key=lambda row: (-_num(row.get("current_priority_score")), row.get("rank_current") or 10**9)),
    }


def _balanced_assignment(groups: dict[str, list[dict]], per_group: int) -> list[dict]:
    assignment = []
    seen = set()
    for group in GROUPS:
        for row in groups.get(group, []):
            key = _dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["group"] = group
            assignment.append(out)
            if sum(item.get("group") == group for item in assignment) >= per_group:
                break
    return assignment


def _assert_group_counts(rows: list[dict], per_group: int) -> None:
    counts = {group: sum(row.get("group") == group for row in rows) for group in GROUPS}
    short = {group: count for group, count in counts.items() if count < per_group}
    if short:
        raise ValueError(f"fewer than {per_group} importable rows for groups: {short}")


def _label_studio_task(row: dict, pilot_id: str) -> dict:
    data = {"image": str(row.get("image")).strip()}
    if row.get("bbox") is not None:
        data["bbox"] = row["bbox"]
    meta = {
        "review_group": row.get("group"),
        "shadow_review_pilot_id": pilot_id,
        "sample_id": row.get("sample_id"),
        "image_id": row.get("image_id"),
        "annotation_id": row.get("annotation_id"),
        "prediction_id": row.get("prediction_id"),
        "dataset": row.get("dataset"),
        "category_name": row.get("category_name"),
        "current_priority_score": row.get("current_priority_score"),
        "learned_shadow_score": row.get("learned_shadow_score"),
        "rank_current": row.get("rank_current"),
        "rank_learned": row.get("rank_learned"),
        "rank_delta": row.get("rank_delta"),
        "artifact_version": row.get("artifact_version"),
        "shadow_metadata_status": row.get("shadow_metadata_status"),
        "shadow_only": True,
        "affects_default_ranking": False,
    }
    return {"id": f"{pilot_id}:{row.get('group')}:{_dedupe_key(row)}", "data": data, "meta": meta}


def _assert_safe_payload(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False)
    for forbidden in FORBIDDEN_REVIEW_FIELDS:
        if forbidden in text:
            raise ValueError(f"forbidden field leaked into balanced pilot payload: {forbidden}")


def _is_importable(row: dict) -> bool:
    return isinstance(row.get("image"), str) and bool(row["image"].strip()) and row.get("learned_shadow_score") is not None


def _rank_map(rows: list[dict], field: str) -> list[int]:
    order = sorted(range(len(rows)), key=lambda idx: (-_num(rows[idx].get(field)), idx))
    ranks = [0] * len(rows)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _dedupe_key(row: dict) -> str:
    if row.get("sample_id") is not None:
        return f"sample_id:{row.get('sample_id')}"
    if row.get("image_id") is not None and row.get("annotation_id") is not None:
        return f"image_annotation:{row.get('image_id')}:{row.get('annotation_id')}"
    if row.get("task_id") is not None:
        return f"task_id:{row.get('task_id')}"
    return f"image:{row.get('image')}"


def _shadow_value(row: dict, field: str) -> float | None:
    shadow = row.get("shadow_scores") if isinstance(row.get("shadow_scores"), dict) else {}
    value = shadow.get(field)
    if value is None:
        return None
    return _num(value)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "task_id", "sample_id", "image_id", "annotation_id", "group",
        "current_priority_score", "learned_shadow_score", "rank_current",
        "rank_learned", "rank_delta", "artifact_version",
        "shadow_metadata_status", "human_review_outcome", "reviewer",
        "reviewed_at", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _guidelines() -> str:
    return "\n".join(
        [
            "# Segmentation Balanced Pilot Review Guidelines",
            "",
            "Valid outcomes: ok, minor_correction_needed, major_correction_needed, unclear.",
            "This feedback is offline analysis only.",
            "Do not use these outcomes to change default Layer 5 weights or sorting.",
        ]
    ) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a 4-group balanced segmentation human review pilot.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=15)
    parser.add_argument("--pilot-id", default="segmentation_balanced_pilot_2026_06_04")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_balanced_pilot_review(
        args.review_queue,
        args.output_dir,
        per_group=args.per_group,
        pilot_id=args.pilot_id,
    )
    print(f"[OK] balanced pilot rows={result['rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_prepare_balanced_pilot_review.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add image_segmentation/benchmark/prepare_balanced_pilot_review.py tests/test_prepare_balanced_pilot_review.py
git commit -m "Add segmentation balanced pilot review prep"
```

## Task 2: End-To-End Compatibility With Existing Label Studio Import

**Files:**
- Modify: `tests/test_prepare_balanced_pilot_review.py`
- Existing integration target: `scripts/import_image_segmentation_review_tasks_to_label_studio.py`

- [ ] **Step 1: Add a failing compatibility test**

Append this test method to `BalancedPilotReviewTests` in `tests/test_prepare_balanced_pilot_review.py`:

```python
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
```

- [ ] **Step 2: Run the compatibility test**

Run:

```bash
.venv/bin/python -m pytest tests/test_prepare_balanced_pilot_review.py::BalancedPilotReviewTests::test_generated_tasks_load_with_existing_segmentation_importer -q
```

Expected before Task 1 implementation is complete: FAIL. Expected after Task 1: PASS.

- [ ] **Step 3: Verify generated task shape in implementation**

Open `image_segmentation/benchmark/prepare_balanced_pilot_review.py` and verify `_label_studio_task()` writes exactly this top-level shape for each task:

```python
{
    "id": "...",
    "data": {"image": "...", "bbox": [1, 2, 20, 30]},
    "meta": {"review_group": "...", "shadow_only": True, "affects_default_ranking": False},
}
```

If the shape differs, replace `_label_studio_task()` with the version from Task 1 Step 3 before continuing.

- [ ] **Step 4: Run related tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_prepare_balanced_pilot_review.py tests/test_import_image_segmentation_review_tasks_to_label_studio.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add image_segmentation/benchmark/prepare_balanced_pilot_review.py tests/test_prepare_balanced_pilot_review.py
git commit -m "Verify balanced pilot tasks import into Label Studio"
```

## Task 3: Pilot Runbook Documentation

**Files:**
- Modify: `README.zh.md`
- Modify: `README.md`

- [ ] **Step 1: Add Chinese runbook section**

Add this section to `README.zh.md` after the shadow observation or Label Studio segmentation review section:

```markdown
## Segmentation Balanced Human Review Pilot

第一轮人工 review 验证使用 4 组平衡抽样，每组 15 个样本，总计 60 个 Label Studio 分割任务：

- `high_learned_low_current`
- `high_current_low_learned`
- `top_learned`
- `control_current_top`

该 pilot 只验证 learned boundary/shape shadow signal 是否能补充当前默认 Layer 5 排序。默认权重、默认 `review_queue.jsonl` 排序和 production scoring 都保持不变。

从 shadow-scored queue 准备 pilot：

```bash
.venv/bin/python -m image_segmentation.benchmark.prepare_balanced_pilot_review \
  --review-queue <shadow_scored_review_queue.jsonl> \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04 \
  --per-group 15 \
  --pilot-id segmentation_balanced_pilot_2026_06_04
```

导入 Label Studio：

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'

scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json \
  --dry-run
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json
```

人工 review 后，将 `human_review_outcome` 填回 `review_assignment.jsonl` 或 assignment CSV，再运行离线汇总：

```bash
.venv/bin/python -m image_segmentation.benchmark.summarize_shadow_human_feedback \
  --review-packet demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/review_assignment.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/human_feedback_summary
```

安全约束：人工结果只用于离线分析，不得接入 production scoring，不得修改默认 Layer 5 weights 或默认排序。
```

- [ ] **Step 2: Add English runbook section**

Add this section to `README.md` near the image segmentation benchmark or Label Studio segmentation section:

```markdown
## Segmentation Balanced Human Review Pilot

The first real-human validation pilot uses four balanced groups with 15 samples each, for 60 Label Studio segmentation tasks:

- `high_learned_low_current`
- `high_current_low_learned`
- `top_learned`
- `control_current_top`

This pilot validates whether learned boundary/shape shadow scoring adds useful review signal. It does not change default Layer 5 weights, default `review_queue.jsonl` sorting, or production scoring.

Prepare the pilot from a shadow-scored queue:

```bash
.venv/bin/python -m image_segmentation.benchmark.prepare_balanced_pilot_review \
  --review-queue <shadow_scored_review_queue.jsonl> \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04 \
  --per-group 15 \
  --pilot-id segmentation_balanced_pilot_2026_06_04
```

Import into Label Studio:

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'

scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json \
  --dry-run
scripts/import_image_segmentation_review_tasks_to_label_studio.py \
  --tasks-path demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_tasks.json
```

After review, fill `human_review_outcome` in `review_assignment.jsonl` or the assignment CSV, then summarize offline feedback:

```bash
.venv/bin/python -m image_segmentation.benchmark.summarize_shadow_human_feedback \
  --review-packet demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/review_assignment.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/human_feedback_summary
```

Human feedback remains offline-only and must not feed production scoring or default sorting.
```

- [ ] **Step 3: Run documentation grep checks**

Run:

```bash
rg -n "Segmentation Balanced Human Review Pilot|prepare_balanced_pilot_review|shadow-only|default Layer 5" README.md README.zh.md
```

Expected: both READMEs contain the new runbook and safety language.

- [ ] **Step 4: Commit**

```bash
git add README.md README.zh.md
git commit -m "Document segmentation balanced human review pilot"
```

## Task 4: Full Verification And Dry-Run Command

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_prepare_balanced_pilot_review.py tests/test_import_image_segmentation_review_tasks_to_label_studio.py tests/test_shadow_window_schema.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run a local fixture CLI smoke test**

Run:

```bash
tmpdir="$(mktemp -d)"
.venv/bin/python - <<'PY' "$tmpdir/review_queue.shadow_scored.jsonl"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
rows = []
for idx in range(1, 81):
    rows.append({
        "task_id": f"task-{idx}",
        "sample_id": f"sample-{idx}",
        "image_id": f"image-{idx}",
        "annotation_id": f"ann-{idx}",
        "prediction_id": f"pred-{idx}",
        "dataset": "smoke",
        "category_name": "Object",
        "image": f"/data/local-files/?d=images/unit_{idx}.png",
        "priority_score": idx / 100,
        "rank": idx,
        "prompt_bbox": [1, 2, 20, 30],
        "shadow_scores": {"learned_boundary_shape_only_score": (100 - idx) / 100, "shadow_only": True},
        "shadow_score_metadata": {
            "artifact_version": "smoke",
            "artifact_validation_status": "valid",
            "shadow_only": True,
            "affects_default_ranking": False,
        },
    })
path.write_text("\\n".join(json.dumps(row) for row in rows) + "\\n", encoding="utf-8")
PY
.venv/bin/python -m image_segmentation.benchmark.prepare_balanced_pilot_review \
  --review-queue "$tmpdir/review_queue.shadow_scored.jsonl" \
  --output-dir "$tmpdir/pilot" \
  --per-group 15 \
  --pilot-id smoke_pilot
python3 - <<'PY' "$tmpdir/pilot/balanced_pilot_summary.json"
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
assert summary["rows"] == 60
assert all(count == 15 for count in summary["group_counts"].values())
print("smoke_ok")
PY
```

Expected: command prints `[OK] balanced pilot rows=60 ...` and `smoke_ok`.

- [ ] **Step 3: Check worktree only contains intended changes**

Run:

```bash
git status --short
```

Expected: no uncommitted changes from this plan, except pre-existing unrelated files if they were already dirty before execution.

- [ ] **Step 4: Final handoff**

Report:

- Plan tasks completed.
- Test commands and results.
- Generated CLI command for the real pilot.
- Reminder that `summarize_shadow_human_feedback.py` and `tests/test_benchmark_ablate_review_weights.py` had pre-existing local modifications and were not touched unless the user explicitly asked.
