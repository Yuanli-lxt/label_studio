#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${1:-${IMAGE_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/image_classification_labeled_export.json}}"

PAYLOAD="$(cat <<JSON
{
  "task_type": "image_classification",
  "dataset_path": "${DATASET_PATH}",
  "notes": "phase-4 real image classifier training"
}
JSON
)"

echo "POST ${TRAINER_URL}/train/image-classification"
curl -sS -X POST "${TRAINER_URL}/train/image-classification" \
  -H 'Content-Type: application/json' \
  -d "${PAYLOAD}"
echo

echo "Current image-classifier metadata"
curl -sS "${TRAINER_URL}/models/image-classification/current"
echo
