#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://localhost:9090}"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
DATASET_PATH="${1:-${TEXT_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/text_classification_labeled_export.json}}"

PRED_PAYLOAD='{
  "tasks": [
    {
      "id": "txt-phase4-pos",
      "data": {
        "text": "The service is helpful and excellent for daily work."
      }
    },
    {
      "id": "txt-phase4-neg",
      "data": {
        "text": "This experience is awful and I dislike this product."
      }
    }
  ],
  "label_config": "<View><Text name=\"text\" value=\"$text\"/><Choices name=\"text_label\" toName=\"text\"><Choice value=\"Positive\"/><Choice value=\"Negative\"/></Choices></View>"
}'

before="$(curl -sS -X POST "${ML_URL}/predict" -H 'Content-Type: application/json' -d "${PRED_PAYLOAD}")"

TRAIN_PAYLOAD="$(cat <<JSON
{
  "task_type": "text_classification",
  "dataset_path": "${DATASET_PATH}",
  "notes": "quality-tuning-compare-before-after"
}
JSON
)"

train_response="$(curl -sS -X POST "${TRAINER_URL}/train/text-classification" -H 'Content-Type: application/json' -d "${TRAIN_PAYLOAD}")"
after="$(curl -sS -X POST "${ML_URL}/predict" -H 'Content-Type: application/json' -d "${PRED_PAYLOAD}")"
metadata="$(curl -sS "${TRAINER_URL}/models/text-classification/current")"

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
        choice = item.get("result", [{}])[0].get("value", {}).get("choices", [None])[0]
        conf = item.get("confidence", {})
        out.append({
            "model": item.get("model_version"),
            "label": choice,
            "score": item.get("score"),
            "source": item.get("prediction_source"),
            "bucket": conf.get("confidence_bucket"),
            "uncertain": conf.get("uncertain"),
            "top_probs": conf.get("top_probabilities", [])[:2],
        })
    return out

print("\nBefore top model_version:", before.get("model_version"))
for idx, row in enumerate(rows(before), start=1):
    print(f"  task{idx}: {row}")

print("\nAfter top model_version:", after.get("model_version"))
for idx, row in enumerate(rows(after), start=1):
    print(f"  task{idx}: {row}")

metrics = metadata.get("metrics", {})
quality = metadata.get("dataset", {}).get("quality", {})
print("\nMetadata summary:")
print("  model_version:", metadata.get("model_version"))
print("  accuracy:", metrics.get("accuracy"), "macro_f1:", metrics.get("macro_f1"))
print("  label_distribution:", quality.get("label_distribution"))
print("  confusion_labels:", metrics.get("confusion_matrix", {}).get("labels"))
print("  confusion_matrix:", metrics.get("confusion_matrix", {}).get("matrix"))
PY
