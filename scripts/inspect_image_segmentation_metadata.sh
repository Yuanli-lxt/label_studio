#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"

metadata="$(curl -sS "$TRAINER_URL/models/image-segmentation/current")"

python3 - <<'PY' "$metadata"
import json
import sys

metadata = json.loads(sys.argv[1])
corrections = metadata.get("correction_deltas") or {}

print("Model version:", metadata.get("model_version"))
print("Trained at:", metadata.get("trained_at"))
print("Task type:", metadata.get("task_type"))
print("\nDataset")
print("  total:", metadata.get("dataset", {}).get("total_size"))
print("  labels:", metadata.get("dataset", {}).get("quality", {}).get("label_distribution"))
print("  snapshot_path:", metadata.get("dataset", {}).get("snapshot_path"))

print("\nCorrection deltas")
print("  total records:", corrections.get("records_total", 0))
print("  ok records:", corrections.get("records_ok", 0))
print("  skipped records:", corrections.get("records_skipped", 0))
print("  major corrections:", corrections.get("major_corrections", 0))
print("  severity counts:", corrections.get("severity_counts") or {})
print("  mean model-human IoU:", corrections.get("mean_model_human_iou"))
print("  mean correction area ratio:", corrections.get("mean_correction_area_ratio"))
print("  output path:", corrections.get("output_path"))
PY
