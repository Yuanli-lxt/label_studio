#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
THRESHOLD="${1:-${IMAGE_REVIEW_CONFIDENCE_THRESHOLD:-0.7}}"
OUT_PATH="${2:-demo_data/tasks/image_classification_review_tasks.json}"
METADATA_PATH="${IMAGE_REVIEW_METADATA_PATH:-}"

if [ -n "${METADATA_PATH}" ]; then
  if [ ! -f "${METADATA_PATH}" ]; then
    echo "metadata file not found: ${METADATA_PATH}" >&2
    exit 2
  fi
  metadata_json="$(cat "${METADATA_PATH}")"
else
metadata_json="$(python3 - <<'PY' "${TRAINER_URL}"
import sys
import urllib.request
base = sys.argv[1].rstrip('/')
url = base + '/models/image-classification/current'
print(urllib.request.urlopen(url, timeout=10).read().decode('utf-8'))
PY
)"
fi

python3 - <<'PY' "$metadata_json" "$THRESHOLD" "$OUT_PATH"
import json
import os
import sys
from pathlib import Path

metadata = json.loads(sys.argv[1])
threshold = float(sys.argv[2])
out_path = Path(sys.argv[3])

metrics = metadata.get("metrics", {})
eval_predictions = metrics.get("eval_predictions") or []
misclassified = metrics.get("eval_misclassified") or []

review_by_filename = {}

def add_row(row, reason):
    if not isinstance(row, dict):
        return
    filename = row.get("filename")
    if not filename:
        image_path = row.get("image_path")
        if isinstance(image_path, str) and image_path:
            filename = os.path.basename(image_path)
    image = row.get("image")
    if not filename or not isinstance(image, str) or not image:
        return

    confidence = row.get("confidence")
    existing = review_by_filename.get(filename)
    item = {
        "filename": filename,
        "image": image,
        "true_label": row.get("true_label"),
        "predicted_label": row.get("predicted_label"),
        "confidence": confidence,
        "reason": reason,
    }
    if existing is None:
        review_by_filename[filename] = item
        return

    # Keep strongest review reason and lower confidence row when duplicated.
    reason_rank = {"misclassified": 2, "low_confidence": 1}
    if reason_rank.get(reason, 0) > reason_rank.get(existing.get("reason"), 0):
        review_by_filename[filename] = item
        return

    try:
        old_conf = float(existing.get("confidence"))
        new_conf = float(confidence)
        if new_conf < old_conf:
            review_by_filename[filename] = item
    except Exception:
        pass

for row in misclassified:
    add_row(row, "misclassified")

for row in eval_predictions:
    try:
        conf = float(row.get("confidence"))
    except Exception:
        continue
    if conf < threshold:
        add_row(row, "low_confidence")

review_rows = sorted(review_by_filename.values(), key=lambda x: (x["reason"], x["filename"]))

review_tasks = []
for idx, row in enumerate(review_rows, start=1):
    task = {
        "id": f"image-review-{idx:03d}",
        "data": {
            "image": row["image"],
            "caption": (
                f"Review {row['filename']} | reason={row['reason']} | "
                f"true={row.get('true_label')} pred={row.get('predicted_label')} "
                f"confidence={row.get('confidence')}"
            ),
        },
        "meta": {
            "review_reason": row["reason"],
            "filename": row["filename"],
            "true_label": row.get("true_label"),
            "predicted_label": row.get("predicted_label"),
            "confidence": row.get("confidence"),
            "source": "image_classifier_metadata",
            "model_version": metadata.get("model_version"),
        },
    }
    review_tasks.append(task)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(review_tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"model_version={metadata.get('model_version')}")
print(f"threshold={threshold}")
print(f"eval_predictions={len(eval_predictions)}")
print(f"eval_misclassified={len(misclassified)}")
print(f"review_tasks={len(review_tasks)}")
print(f"output={out_path}")
PY
