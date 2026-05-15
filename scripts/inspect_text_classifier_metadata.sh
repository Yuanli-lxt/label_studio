#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"

metadata="$(curl -sS "${TRAINER_URL}/models/text-classification/current")"

python3 - <<'PY' "$metadata"
import json
import sys

metadata = json.loads(sys.argv[1])

quality = metadata.get("dataset", {}).get("quality", {})
split = metadata.get("dataset", {}).get("split", {})
metrics = metadata.get("metrics", {})

print("Model version:", metadata.get("model_version"))
print("Trained at:", metadata.get("trained_at"))
print("Labels:", metadata.get("labels"))
print("\nDataset quality")
print("  total:", quality.get("total_samples"), "unique:", quality.get("unique_text_samples"))
print("  distribution:", quality.get("label_distribution"))
print("  validation passed:", quality.get("validation", {}).get("passed"))
print("  validation errors:", quality.get("validation", {}).get("errors"))
dataset = metadata.get("dataset", {})
print("\nDataset sources")
print("  source:", dataset.get("source"))
print("  source_path:", dataset.get("source_path"))
print("  snapshot_path:", dataset.get("snapshot_path"))
print("\nSplit")
print("  strategy:", split.get("strategy"))
print("  train distribution:", split.get("train_label_distribution"))
print("  eval distribution:", split.get("eval_label_distribution"))
print("\nMetrics")
print("  accuracy:", metrics.get("accuracy"), "macro_f1:", metrics.get("macro_f1"))
print("  per-label keys:", list((metrics.get("per_label") or {}).keys()))
print("  confidence summary:", (metrics.get("confidence") or {}).get("evaluation"))
print("  confusion matrix:", metrics.get("confusion_matrix"))
PY
