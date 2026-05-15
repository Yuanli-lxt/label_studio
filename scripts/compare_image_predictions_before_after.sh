#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://localhost:9090}"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${1:-${IMAGE_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/image_classification_labeled_export.json}}"

PRED_PAYLOAD='{
  "tasks": [
    {
      "id": "img-before-after-blue",
      "data": {
        "image": "/data/local-files/?d=images/demo_blue.png",
        "caption": "blue sample"
      }
    },
    {
      "id": "img-before-after-green",
      "data": {
        "image": "/data/local-files/?d=images/demo_green.png",
        "caption": "green sample"
      }
    },
    {
      "id": "img-before-after-gray",
      "data": {
        "image": "/data/local-files/?d=images/demo_gray.png",
        "caption": "gray sample"
      }
    }
  ],
  "label_config": "<View><Image name=\"image\" value=\"$image\"/><Choices name=\"image_label\" toName=\"image\"><Choice value=\"Product\"/><Choice value=\"Other\"/></Choices></View>"
}'

before="$(curl -sS -X POST "${ML_URL}/predict" -H 'Content-Type: application/json' -d "${PRED_PAYLOAD}")"

TRAIN_PAYLOAD="$(cat <<JSON
{
  "task_type": "image_classification",
  "dataset_path": "${DATASET_PATH}",
  "notes": "compare-image-before-after"
}
JSON
)"

train_response="$(curl -sS -X POST "${TRAINER_URL}/train/image-classification" -H 'Content-Type: application/json' -d "${TRAIN_PAYLOAD}")"
after="$(curl -sS -X POST "${ML_URL}/predict" -H 'Content-Type: application/json' -d "${PRED_PAYLOAD}")"
metadata="$(curl -sS "${TRAINER_URL}/models/image-classification/current")"

python3 - <<'PY' "$before" "$train_response" "$after" "$metadata"
import json
import sys

before = json.loads(sys.argv[1])
train = json.loads(sys.argv[2])
after = json.loads(sys.argv[3])
metadata = json.loads(sys.argv[4])

print("Train response:", json.dumps(train, indent=2))

def rows(payload):
    out = []
    for item in payload.get("results", []):
        result = (item.get("result") or [{}])[0]
        label = (result.get("value", {}).get("choices") or [None])[0]
        conf = item.get("confidence", {}) if isinstance(item.get("confidence"), dict) else {}
        out.append(
            {
                "model": item.get("model_version"),
                "label": label,
                "score": item.get("score"),
                "source": item.get("prediction_source"),
                "uncertain": conf.get("uncertain"),
                "top_probs": conf.get("top_probabilities", [])[:2],
            }
        )
    return out

print("\nBefore top model_version:", before.get("model_version"))
for idx, row in enumerate(rows(before), start=1):
    print(f"  task{idx}: {row}")

print("\nAfter top model_version:", after.get("model_version"))
for idx, row in enumerate(rows(after), start=1):
    print(f"  task{idx}: {row}")

print("\nMetadata summary:")
print("  model_version:", metadata.get("model_version"))
print("  accuracy:", metadata.get("metrics", {}).get("accuracy"), "macro_f1:", metadata.get("metrics", {}).get("macro_f1"))
print("  label_distribution:", metadata.get("dataset", {}).get("quality", {}).get("label_distribution"))
PY
