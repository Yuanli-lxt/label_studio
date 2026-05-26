#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${IMAGE_SEG_DATASET_PATH:-}"

if [[ -n "$DATASET_PATH" ]]; then
  payload="{\"task_type\":\"image_segmentation\",\"dataset_path\":\"$DATASET_PATH\"}"
else
  payload='{"task_type":"image_segmentation","samples":[{"image":"/data/local-files/?d=images/demo_blue.png","label":"Object","rle":[0,1,2,3],"original_width":320,"original_height":240,"source":"segmentation_smoke_sample"}]}'
fi

curl -sS -X POST "$TRAINER_URL/retrain/image-segmentation" \
  -H 'Content-Type: application/json' \
  -d "$payload"
echo
