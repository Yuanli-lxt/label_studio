# Segmentation Human Review Balanced Pilot Design

Date: 2026-06-04

## Goal

Run a first real-human validation loop for image segmentation review prioritization while keeping boundary/shape and learned boundary-shape scoring strictly shadow-only.

The pilot will review 60 samples in Label Studio: 15 samples from each of four groups. The result should answer whether learned boundary-shape shadow scores find useful correction opportunities that the current Layer 5 priority misses, without changing production/default ranking.

## Scope

In scope:

- Build a balanced human review packet with 15 rows per group:
  - `high_learned_low_current`
  - `high_current_low_learned`
  - `top_learned`
  - `control_current_top`
- Convert the selected safe rows into Label Studio image segmentation review tasks.
- Import the tasks into the existing image segmentation review project.
- Have reviewers inspect model pre-labels and make Brush mask corrections when needed.
- Collect a small structured outcome for each reviewed row:
  - `ok`
  - `minor_correction_needed`
  - `major_correction_needed`
  - `unclear`
- Summarize human feedback offline by group and score bucket.
- Preserve boundary/shape and learned boundary-shape as shadow-only signals.

Out of scope:

- Changing default Layer 5 weights.
- Changing default `review_queue.jsonl` sorting.
- Promoting learned boundary-shape into production scoring.
- Training a new segmentation model from this pilot.
- Using human outcomes as production scoring inputs.
- Large-scale reviewer assignment, QA adjudication, or multi-reviewer agreement beyond optional fields already supported by the feedback summary.

## Existing Context

The repository already has the pieces needed for most of the workflow:

- `image_segmentation.benchmark.live_shadow_rollout` can generate shadow-scored queues and a `human_review_task_packet`.
- `image_segmentation.benchmark.summarize_shadow_human_feedback` summarizes offline review outcomes.
- `scripts/bootstrap_label_studio_image_segmentation_review.py` creates or reuses the Label Studio segmentation review project.
- `scripts/import_image_segmentation_review_tasks_to_label_studio.py` imports segmentation review tasks with dedupe.
- The current comparison reports recommend keeping default Layer 5 weights unchanged while continuing boundary/shape experiments.

The missing piece is a narrow bridge from the safe shadow human-review packet to importable Label Studio segmentation tasks, plus a documented pilot runbook.

## Sampling Design

The pilot uses a fixed balanced sample count:

```text
high_learned_low_current: 15
high_current_low_learned: 15
top_learned: 15
control_current_top: 15
total: 60
```

Rows should come from the latest production-like shadow-only observation output, preferably `human_review_task_packet/` produced by `live_shadow_rollout`.

Selection should be deterministic:

- Preserve the packet order within each group.
- Take the first 15 valid importable rows from each group.
- Deduplicate by the strongest available stable id in this order: `sample_id`, `image_id + annotation_id`, `task_id`, `image`.
- If a group has fewer than 15 importable rows, fail with a clear message instead of silently changing the target balance.

The selected packet must retain group and shadow metadata for later offline analysis, but only safe fields should be imported into Label Studio task `meta`.

## Label Studio Task Contract

Each selected review task should include:

- `data.image`: the image URL or local-files path that Label Studio can load.
- `data.bbox`: a pixel prompt box when available, so the ML backend can generate the same style of pre-label.
- `meta.review_group`: one of the four pilot groups.
- `meta.shadow_review_pilot_id`: stable pilot id, for example `segmentation_balanced_pilot_2026_06_04`.
- `meta.sample_id`, `meta.image_id`, `meta.annotation_id`, `meta.prediction_id` when safe and available.
- `meta.current_priority_score`, `meta.learned_shadow_score`, `meta.rank_current`, `meta.rank_learned`, `meta.rank_delta`.
- `meta.shadow_only: true`.
- `meta.affects_default_ranking: false`.

The import task must not include ground truth masks, IoU, Dice, correction deltas, severity labels, GT paths, or evaluation-only fields.

## Reviewer Workflow

Reviewers work in the existing Label Studio segmentation project:

1. Open the imported pilot tasks.
2. Inspect the model pre-label mask.
3. Correct the mask with Brush tools when needed.
4. Record the outcome as one of:
   - `ok`
   - `minor_correction_needed`
   - `major_correction_needed`
   - `unclear`
5. Optionally add reviewer name and notes.

If the existing Label Studio config cannot capture the outcome directly, outcomes may be entered into the exported assignment template or review packet after annotation. The analysis step treats this feedback as offline-only.

## Analysis

After review, run the offline feedback summary over the completed review packet or assignment file. The summary should report:

- total reviewed count
- reviewed count by group
- outcome counts
- major correction rate by group
- major-or-minor correction rate by group
- learned lift vs `control_current_top`
- missed risk discovery rate for `high_learned_low_current`
- over-prioritization rate for `high_learned_low_current`
- unclear rate
- optional inter-reviewer agreement if duplicate reviews exist

Interpretation:

- `useful_signal`: learned groups have at least 1.25x major-or-minor correction rate versus control and `high_learned_low_current` beats control.
- `mixed`: learned signal finds some useful examples but is not clearly better than control.
- `not_useful`: `high_learned_low_current` mostly produces `ok` outcomes.
- `insufficient_feedback`: fewer than 20 reviewed rows.

These labels are advisory only and must not trigger default promotion.

## Shadow-Only Safety

The pilot must preserve these invariants:

- Default Layer 5 weights remain unchanged.
- Default review queue order remains unchanged.
- Boundary/shape and learned boundary-shape scores remain shadow-only.
- Human feedback is used only for offline analysis.
- Production scoring does not read `human_review_outcome`.
- Production scoring does not read `evaluation_only`, GT masks, GT paths, IoU, Dice, correction deltas, severity labels, or boundary metadata.
- Any generated task packet includes `shadow_only=true` and `affects_default_ranking=false`.

## Failure Handling

The pilot should fail early if:

- The shadow packet does not contain all four groups.
- Any group has fewer than 15 importable rows.
- Required image paths cannot be resolved to Label Studio-compatible task data.
- Safe-field validation finds evaluation-only or GT-derived fields in import payloads.
- Label Studio API credentials are missing.

Partial Label Studio imports should remain recoverable through existing import dedupe.

## Verification

Implementation should include focused tests for:

- balanced group selection
- deterministic dedupe
- safe-field filtering
- Label Studio task shape
- failure when a group has fewer than 15 valid rows
- compatibility with `summarize_shadow_human_feedback`

Manual verification should include:

- bootstrap or reuse the Label Studio segmentation review project
- dry-run import the selected 60 tasks
- import the selected 60 tasks
- confirm tasks display images and receive ML pre-labels
- complete a small subset of review outcomes
- run offline human feedback summary

## Deliverables

- A selected balanced pilot task packet.
- A Label Studio importable JSON task file.
- A short runbook for running the pilot.
- Offline feedback summary JSON/Markdown after review.
- A final pilot decision note that keeps default scoring unchanged and states whether the learned boundary-shape signal should continue shadow-only observation, expand to a larger pilot, or pause.
