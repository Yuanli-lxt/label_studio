#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${1:-${TEXT_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/text_classification_labeled_export.json}}"

PAYLOAD="$(cat <<JSON
{
  "task_type": "text_classification",
  "dataset_path": "${DATASET_PATH}",
  "notes": "phase-4 real text classifier training"
}
JSON
)"

echo "POST ${TRAINER_URL}/train/text-classification"
curl -sS -X POST "${TRAINER_URL}/train/text-classification" \
  -H 'Content-Type: application/json' \
  -d "${PAYLOAD}"
echo

echo "Current text-classifier metadata"
curl -sS "${TRAINER_URL}/models/text-classification/current"
echo
