# Segmentation Boundary/Shape Shadow Scoring Rollout

This runbook covers the production-adjacent shadow-only rollout for `learned_boundary_shape_only_score`.

## Feature Flags

Default state:

```bash
SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=false
SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=false
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_DIR=
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY=
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION=
SEGMENTATION_BOUNDARY_SHAPE_SHADOW_VERSION=boundary_shape_shadow_v1
SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN=true
```

Enabling `SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=true` only appends `shadow_scores` and `shadow_score_metadata` after default Layer 5 scoring and ordering. Learned inference additionally requires `SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=true` and an explicit artifact dir, registry version, or registry `CURRENT`.

Production shadow rollout example:

```bash
SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=true
SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=true
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY=/artifacts/segmentation/boundary_shape
SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION=learned_boundary_shape_only_v1
SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN=true
```

Startup/runtime logs include the shadow flag state, learned flag state, resolved artifact path, registry, configured version, artifact load and validation status, fail-open mode, shadow-only mode, and `affects_default_ranking=false`.

## Artifact Registry

Recommended layout:

```text
artifacts/segmentation/boundary_shape/
  learned_boundary_shape_only_v1/
    learned_boundary_shape_only_model.pkl
    learned_current_plus_boundary_shape_model.pkl
    feature_schema.json
    model_metadata.json
    validation_report.json
    validation_report.md
  CURRENT
```

Resolution order:

1. `SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_DIR`
2. `SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY` plus `SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION`
3. `SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY` plus `CURRENT`

Artifact metadata should include artifact version, model type, training datasets, training sample count, positive count, feature names, excluded leaky fields, sklearn version, random seed, created time, training command, validation report path when available, `shadow_only: true`, and `affects_default_ranking: false`.

## Validation

Validate artifact load and schema:

```bash
python -m image_segmentation.benchmark.learn_boundary_shape_fusion \
  --validate-artifact <artifact_dir> \
  --review-queue <validation_review_queue.jsonl>
```

Replay without changing the source queue:

```bash
.venv/bin/python -m image_segmentation.benchmark.apply_shadow_scores \
  --review-queue <input_review_queue.jsonl> \
  --output <output_review_queue.shadow_scored.jsonl> \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow
```

Confirm `default_field_equality_check=true` and `ordering_equality_check=true` in `shadow_score_replay_summary.json`.

Run monitoring:

```bash
.venv/bin/python -m image_segmentation.benchmark.compare_shadow_scores \
  --review-queue <review_queue.shadow_scored.jsonl> \
  --output-dir <output_dir>/shadow_monitoring \
  --window-id <production-window-id> \
  --baseline <previous_shadow_monitoring.json> \
  --production-mode \
  --no-labels
```

Export the safe human review packet:

```bash
.venv/bin/python -m image_segmentation.benchmark.export_shadow_review_packet \
  --review-queue <review_queue.shadow_scored.jsonl> \
  --output-dir <output_dir>/shadow_review_packet \
  --top-k 50 \
  --production-mode
```

Run multiple production-shadow windows with automatic previous-window baselines:

```bash
.venv/bin/python -m image_segmentation.benchmark.live_shadow_rollout \
  --input-queues <queue_window_001.jsonl> <queue_window_002.jsonl> <queue_window_003.jsonl> \
  --output-root demo_data/model_state/image_segmentation/benchmark/production_shadow_multi_window_observation \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow \
  --production-mode \
  --no-labels \
  --max-windows 5
```

Directory-based windows are also supported:

```bash
.venv/bin/python -m image_segmentation.benchmark.live_shadow_rollout \
  --input-dir <production_queue_dir> \
  --window-glob "review_queue_*.jsonl" \
  --output-root demo_data/model_state/image_segmentation/benchmark/production_shadow_expanded_observation \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow \
  --production-mode \
  --no-labels \
  --max-windows 5
```

Each window writes its own shadow-scored queue, monitoring output, safe review packet, and rollout report. The source queue is never overwritten; default ordering and default queue fields are verified for every window.

For broader observation, use at least 10 production-like windows and write to the broader output root:

```bash
.venv/bin/python -m image_segmentation.benchmark.live_shadow_rollout \
  --input-dir <production_queue_dir> \
  --window-glob "review_queue_*.jsonl" \
  --output-root demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation \
  --artifact-dir <artifact_dir> \
  --enable-learned-boundary-shape-shadow \
  --production-mode \
  --no-labels \
  --max-windows 10 \
  --decision-report-path demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_rollout_decision_report.md
```

## Monitoring

Track learned score coverage, null rate, artifact load failures, inference errors, score distribution drift, top-k overlap/Jaccard with current priority, high disagreement volume, correction rate by learned score bucket when offline labels are available, and queue generation latency if measurable.

Production-mode monitoring does not read labels or `evaluation_only` fields. Offline benchmark mode may compute label metrics, and those metrics are marked `evaluation_only`.

For expanded and broader multi-window observation, `multi_window_shadow_summary.json`, `expanded_shadow_multi_window_summary.json`, `broader_shadow_multi_window_summary.json`, and their `.md` companions aggregate coverage stability, distribution stability, top-k agreement stability, latency, safety checks, and per-window alert status. The first window may use a supplied `--baseline`; subsequent windows use the previous window's `shadow_monitoring.json` as the drift baseline.

Distribution drift records learned score p05/p25/median/p75/p95, median delta, p95 shift in baseline standard deviations, top100 Jaccard relative change, score distribution PSI, null-rate drift, coverage drift, latency p95 drift, and a simplified quantile drift score versus the first window. Latency fields remain `null` when instrumentation is unavailable; do not invent latency values.

Latency schema is stable in `shadow_monitoring.json` and the multi-window summaries:

```text
performance.schema_version
performance.queue_generation_latency_ms
performance.queue_generation_latency_unavailable_reason
performance.shadow_scoring_total_time_ms
performance.monitoring_latency_ms
performance.review_packet_export_latency_ms
performance.artifact_load_latency_ms.{count,p50,p95,p99,mean,max}
performance.per_item_learned_inference_latency_ms.{count,p50,p95,p99,mean,max}
performance.per_item_shadow_scoring_latency_ms.{count,p50,p95,p99,mean,max}
performance.artifact_loaded_once
performance.batch_size
performance.rows_processed
performance.timing_source
performance.timing_available
```

Queue generation latency is `null` unless the production queue generator supplies it. Shadow scoring, monitoring, review packet export, and per-item learned inference timings use `time.perf_counter`. Artifact load latency stays `null` when the artifact loader cannot separate load time from per-item inference.

Drift outliers are listed when p95 drift exceeds `1.0` baseline standard deviations. Treat those windows as investigation targets: check whether the input queue changed dataset, annotation mix, prediction-feature availability, or rank construction before interpreting the learned score as unstable. A drift outlier does not change default ranking, but it blocks any default promotion and may pause further shadow expansion until the cause is understood.

Root-cause analysis CLIs are offline-only:

```bash
.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_drift \
  --multi-window-root demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation \
  --baseline-window window_001 \
  --windows window_004 window_007 \
  --output-dir demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation/drift_root_cause

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_missing_features \
  --multi-window-root demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation \
  --windows window_001 window_004 \
  --output-dir demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation/missing_feature_analysis

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_topk_jaccard \
  --multi-window-root demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation \
  --baseline-window window_001 \
  --windows window_002 window_004 window_005 window_007 \
  --output-dir demo_data/model_state/image_segmentation/benchmark/production_shadow_broader_observation/topk_jaccard_analysis
```

The drift report compares learned/current/boundary score distributions, rank deltas, queue mix, safe feature shifts, and safe outlier samples. The missing-feature report distinguishes old queue schema or neutral fallback from extractor failures. The top-k report includes same-sample overlap and normalized shared-sample Jaccard; cross-window top-k alerts are weak evidence when sample sets differ substantially.

Initial alert thresholds:

```text
learned_score_null_rate > 0.05
artifact_validation_status != valid for any active queue batch
inference_error_rate > 0.01
missing_safe_feature_count p95 > 0
shadow_scoring_latency_p95 > configured threshold
queue_generation_latency_regression > 10%
score_distribution_p95 shifts > 3 std from baseline
top100_jaccard changes > 50% from baseline window
any affects_default_ranking != false
any shadow_only != true
production monitoring labels read != false
default field equality != true
ordering equality != true
```

If no metrics system is available, schedule the JSON/MD monitoring report and have the scheduler read `shadow_monitoring.json.alerts`.

Aggregate alert rules mark the multi-window run unhealthy if any window breaches null rate, artifact validity, inference error rate, missing safe feature p95, latency regression, p95 distribution shift, top100 Jaccard relative change, production label read, default field equality, ordering equality, `shadow_only`, or `affects_default_ranking`. Hard safety alerts include default field changes, ordering changes, production labels/evaluation-only reads, `affects_default_ranking != false`, `shadow_only != true`, unsafe artifact schema, or leakage guard failure. Hard safety alerts block expansion and recommend pause or rollback.

## Human Disagreement Review

Multi-window observation writes:

```text
human_disagreement_review_summary/
  combined_high_learned_low_current.jsonl
  combined_high_current_low_learned.jsonl
  combined_top_learned.jsonl
  combined_top_current.jsonl
  disagreement_review_summary.md
```

Combined packets dedupe by safe sample/image/annotation/task id, preserve the list of windows where each sample appeared, and include empty placeholders for future human review status, outcome, and notes. Production packets include only safe fields: ids when safe, current priority score, learned shadow score, current/learned rank, rank delta, prediction-time safe features, artifact version, and shadow metadata status. They do not include GT masks, GT paths, IoU, Dice, boundary metrics, correction deltas, labels, or `boundary_metadata`.

Broader observation also writes a real human review task packet:

```text
human_review_task_packet/
  review_tasks_high_learned_low_current.jsonl
  review_tasks_high_current_low_learned.jsonl
  review_tasks_top_learned.jsonl
  review_tasks_control_current_top.jsonl
  human_review_task_summary.md
```

Each task row includes only safe fields plus `group`, `window_ids`, and empty review placeholders: `human_review_status`, `human_review_outcome`, `human_review_notes`, `reviewer`, and `reviewed_at`.

For reviewer assignment:

```bash
.venv/bin/python -m image_segmentation.benchmark.build_shadow_human_review_assignment \
  --task-packet-dir <broader_root>/human_review_task_packet \
  --output-dir <broader_root>/human_review_assignment
```

This writes `review_assignment.jsonl`, `review_assignment_template.csv`, and `review_guidelines.md`.

Optional offline feedback summary:

```bash
.venv/bin/python -m image_segmentation.benchmark.summarize_shadow_human_feedback \
  --review-packet <human_review_results.jsonl> \
  --output-dir <output_dir>/human_feedback_summary
```

Accepted feedback rows may include `sample_id`, `group`, `human_review_outcome`, `reviewer`, `reviewed_at`, and `notes`. Valid outcomes are `major_correction_needed`, `minor_correction_needed`, `ok`, and `unclear`. The summary reports reviewed counts by group, major correction rate by group, major+minor correction rate by group, group hit rates, learned lift versus `control_current_top` when enough reviewed samples exist, unclear rate, examples, and inter-reviewer agreement when possible. This is offline analysis only and must not feed production scoring, learned feature matrices, default sorting, or Layer 5 weights.

Feedback import accepts JSONL or CSV. With no reviewed rows it writes `total_reviewed=0` and `recommendation=insufficient_feedback`.

## Schema-Aware Analysis

Before treating cross-window drift or top-k Jaccard changes as learned-score instability, classify windows by production-safe schema metadata:

```bash
.venv/bin/python -m image_segmentation.benchmark.classify_shadow_windows \
  --multi-window-root <broader_root> \
  --output-dir <broader_root>/schema_aware_analysis
```

Then run schema-aware drift, top-k, and missing-feature analysis:

```bash
.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_drift \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/schema_aware_analysis/shadow_window_classification.json \
  --output-dir <broader_root>/schema_aware_drift

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_topk_jaccard \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/schema_aware_analysis/shadow_window_classification.json \
  --output-dir <broader_root>/schema_aware_topk_jaccard

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_missing_features \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/schema_aware_analysis/shadow_window_classification.json \
  --output-dir <broader_root>/schema_aware_missing_features
```

The classification report labels each window with `queue_schema_version`, `feature_completeness_bucket`, `queue_type`, safe feature presence, learned score coverage, neutral fallback rate, artifact status, sample-set fingerprint, and safe metadata fingerprint. Schema-aware drift and top-k analysis only promote comparable full-context pair results to stability evidence. Cross-schema pairs, mixed feature completeness, and zero same-sample overlap are reported as `compatibility_warning`, not rollback or promotion evidence. Old schema fallback missing raw prediction-time boundary/shape features is also a `compatibility_warning`; full prediction-feature windows with missing safe features remain `stability_alert`.

Generate the schema-aware decision report:

```bash
.venv/bin/python -m image_segmentation.benchmark.write_shadow_schema_aware_decision_report \
  --classification <broader_root>/schema_aware_analysis/shadow_window_classification.json \
  --drift <broader_root>/schema_aware_drift/schema_aware_drift.json \
  --topk <broader_root>/schema_aware_topk_jaccard/schema_aware_topk_jaccard.json \
  --missing-features <broader_root>/schema_aware_missing_features/schema_aware_missing_features.json \
  --output demo_data/model_state/image_segmentation/benchmark/production_shadow_schema_aware_decision_report.md
```

Alert severity is ordered as `hard_safety_alert`, `stability_alert`, `compatibility_warning`, and `info`. Compatibility warnings may support cautious same-scope shadow observation, but small expansion and default promotion stay held until comparable-window stability and human feedback are available.

## Controlled Comparable-Window Validation

After schema-aware analysis, select a controlled full-feature comparable subset before making any expansion decision:

```bash
.venv/bin/python -m image_segmentation.benchmark.select_comparable_shadow_windows \
  --classification <broader_root>/schema_aware_analysis/shadow_window_classification.json \
  --multi-window-root <broader_root> \
  --output-dir <broader_root>/controlled_comparable_windows \
  --require-schema full_prediction_features \
  --require-feature-completeness full \
  --require-queue-type benchmark_replay
```

This writes `comparable_window_selection.json`, `selected_windows.txt`, `rejected_windows.jsonl`, and `selected_shadow_window_classification.json`. Old schema fallback, unknown schema, incomplete boundary/shape features, hard safety failures, ordering/default-field failures, label reads, artifact failures, high null rate, or low learned coverage are rejected with an alert severity.

Run controlled analysis against the filtered classification:

```bash
.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_drift \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/controlled_comparable_windows/selected_shadow_window_classification.json \
  --output-dir <broader_root>/controlled_schema_aware_drift

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_topk_jaccard \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/controlled_comparable_windows/selected_shadow_window_classification.json \
  --output-dir <broader_root>/controlled_schema_aware_topk

.venv/bin/python -m image_segmentation.benchmark.analyze_shadow_missing_features \
  --multi-window-root <broader_root> \
  --schema-aware \
  --classification <broader_root>/controlled_comparable_windows/selected_shadow_window_classification.json \
  --output-dir <broader_root>/controlled_schema_aware_missing_features
```

Prepare the offline human review assignment from selected comparable windows:

```bash
.venv/bin/python -m image_segmentation.benchmark.prepare_controlled_human_review_assignment \
  --multi-window-root <broader_root> \
  --selected-windows-file <broader_root>/controlled_comparable_windows/selected_windows.txt \
  --output-dir <broader_root>/controlled_human_review_assignment
```

This writes `review_assignment.jsonl`, `review_assignment_template.csv`, and `review_guidelines.md`. The rows contain only safe ids, current and learned scores, ranks, rank delta, prediction-time safe features, artifact metadata, window ids, and empty human review placeholders. Feedback remains offline-only:

```bash
.venv/bin/python -m image_segmentation.benchmark.summarize_shadow_human_feedback \
  --review-packet <broader_root>/controlled_human_review_assignment/review_assignment.jsonl \
  --output-dir <broader_root>/controlled_human_feedback_summary
```

Finally write the controlled validation report:

```bash
.venv/bin/python -m image_segmentation.benchmark.write_controlled_shadow_validation_report \
  --selection <broader_root>/controlled_comparable_windows/comparable_window_selection.json \
  --drift <broader_root>/controlled_schema_aware_drift/schema_aware_drift.json \
  --topk <broader_root>/controlled_schema_aware_topk/schema_aware_topk_jaccard.json \
  --missing-features <broader_root>/controlled_schema_aware_missing_features/schema_aware_missing_features.json \
  --human-review-assignment-dir <broader_root>/controlled_human_review_assignment \
  --feedback-summary <broader_root>/controlled_human_feedback_summary/human_feedback_summary.json \
  --output demo_data/model_state/image_segmentation/benchmark/controlled_comparable_shadow_validation_report.md
```

If full-feature same-schema windows still show controlled drift stability alerts, keep small expansion held. If controlled full-feature windows are stable but feedback is absent, same shadow traffic can continue cautiously while broader expansion and default promotion remain blocked. Same-sample replay or deterministic resampling is the next stronger validation when selected production windows have weak overlap.

## Rollout Decision

The expanded runner writes `production_shadow_expanded_rollout_decision_report.md`; broader observation writes `production_shadow_broader_rollout_decision_report.md`. Broader observation may recommend expanding shadow traffic only when at least ten broader shadow-only windows have no hard safety alert, null rate stays below 5%, inference error rate stays below 1%, artifact valid rate is effectively 100%, default fields and ordering remain equal, production labels are not read, drift is not abnormal or outlier windows have a concrete sample-mix explanation, top-k disagreement is explainable, latency has no obvious regression, and rollback remains verified. Without real human feedback, keep default promotion on hold even if shadow traffic can continue.

After non-hard alerts, write the root-cause and feedback decision report:

```bash
.venv/bin/python -m image_segmentation.benchmark.write_shadow_drift_feedback_decision_report \
  --multi-window-root <broader_root> \
  --drift-dir <broader_root>/drift_root_cause \
  --missing-dir <broader_root>/missing_feature_analysis \
  --topk-dir <broader_root>/topk_jaccard_analysis \
  --assignment-dir <broader_root>/human_review_assignment \
  --feedback-dir <broader_root>/human_feedback_summary \
  --output-path demo_data/model_state/image_segmentation/benchmark/production_shadow_drift_feedback_decision_report.md
```

Non-hard drift, missing-feature, and top-k alerts block further expansion until explained, but do not require rollback when default fields/order are unchanged, production labels are not read, artifact validation is clean, and shadow scores remain isolated from ranking.

Default promotion remains out of scope. `learned_boundary_shape_only_score` is not ready for default sorting or Layer 5 weight promotion without a future task and label-backed or human-feedback-backed evidence.

## Default Sorting Verification

For replay, confirm:

- `default_field_equality_check=true`
- `ordering_equality_check=true`
- `priority_score`, `rank`, `priority_bucket`, `review_weight_weights`, `score_components`, and `review_reasons` match the input queue

For production queue generation, compare a flag-off queue and a flag-on queue after removing only `shadow_scores` and `shadow_score_metadata`.

## Rollback

Use the smallest effective rollback:

1. Set `SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW=false`.
2. Set `SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW=false`.
3. Clear artifact env vars to force `not_configured`.
4. Switch `SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION`.
5. Modify registry `CURRENT`.
6. Redeploy the previous config.

Rollback triggers include artifact load failure spikes, inference errors, unexpected coverage drops, sharp score distribution drift, queue generation latency regression, any evidence that shadow scores affect default ranking, schema mismatch, or leakage guard failure.

## Recommendation Boundary

`learned_boundary_shape_only_score` may run behind the shadow-only feature flag after validation and dry-run replay. It is not ready for default sorting, Layer 5 weight promotion, or gated fusion promotion. `boundary_shape_calibrated_score` and gated fusion remain shadow/offline until real shadow feedback proves stable lift without leakage or operational regressions.

Promotion criteria require stable live coverage, low null/error rates, no latency regression, no leakage findings, stable distributions across windows, human-reviewed disagreement usefulness, and label-backed lift on representative production traffic.
