#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
ENDPOINT="${1:-/webhook/label-studio}"

PAYLOAD='{
  "action": "ANNOTATION_UPDATED",
  "project": {"id": 1},
  "task_number": 2,
  "useful_annotation_number": 2,
  "annotation": {
    "id": 101,
    "task": 1,
    "result": [
      {
        "from_name": "image_label",
        "to_name": "image",
        "type": "choices",
        "value": {"choices": ["Product"]}
      }
    ]
  }
}'

echo "POST ${TRAINER_URL}${ENDPOINT}"
curl -sS -X POST "${TRAINER_URL}${ENDPOINT}" \
  -H 'Content-Type: application/json' \
  -d "${PAYLOAD}"
echo

echo "Current trainer model state:"
curl -sS "${TRAINER_URL}/model-state"
echo
