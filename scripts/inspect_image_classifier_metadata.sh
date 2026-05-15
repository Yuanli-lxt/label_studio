#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"

metadata="$(python3 - <<'PY' "${TRAINER_URL}"
import sys
import urllib.request
base = sys.argv[1].rstrip("/")
url = base + "/models/image-classification/current"
print(urllib.request.urlopen(url, timeout=10).read().decode("utf-8"))
PY
)"

python3 - <<'PY' "$metadata"
import json
import sys

metadata = json.loads(sys.argv[1])
quality = metadata.get("dataset", {}).get("quality", {})
split = metadata.get("dataset", {}).get("split", {})
fixed_eval = metadata.get("dataset", {}).get("fixed_eval", {})
candidate_info = metadata.get("dataset", {}).get("candidates", {})
metrics = metadata.get("metrics", {})
cm2 = metrics.get("confusion_matrix_product_other", {})
misclassified = metrics.get("eval_misclassified", [])

print("Model version:", metadata.get("model_version"))
print("Trained at:", metadata.get("trained_at"))
print("Labels:", metadata.get("labels"))

print("\nDataset")
print("  source_path:", metadata.get("dataset", {}).get("source_path"))
print("  total:", quality.get("total_samples"), "unique_images:", quality.get("unique_image_samples"))
print("  distribution:", quality.get("label_distribution"))
print("  validation passed:", quality.get("validation", {}).get("passed"))
print("  validation errors:", quality.get("validation", {}).get("errors"))

print("\nSplit")
print("  strategy:", split.get("strategy"))
print("  train distribution:", split.get("train_label_distribution"))
print("  eval distribution:", split.get("eval_label_distribution"))
print("  fixed eval enabled:", fixed_eval.get("enabled"))
print("  fixed eval id:", fixed_eval.get("eval_set_id"))
print("  fixed eval filenames:", fixed_eval.get("eval_filenames"))
print("  manifest path:", fixed_eval.get("manifest_path"))
print("  manifest checked:", fixed_eval.get("manifest_checked"))
print("  candidates path:", candidate_info.get("path"))
print("  candidates stats:", candidate_info.get("stats"))
print("  candidates used after eval filter:", candidate_info.get("used_after_eval_filter"))
print("  candidate eval leakage skipped:", candidate_info.get("eval_leakage_skipped"))

print("\nMetrics")
print("  accuracy:", metrics.get("accuracy"), "macro_f1:", metrics.get("macro_f1"))
print("  confidence summary:", (metrics.get("confidence") or {}).get("evaluation"))
print("  confusion matrix:", metrics.get("confusion_matrix"))
print("  confusion matrix 2x2:")
print("    Product->Product:", cm2.get("Product->Product"))
print("    Product->Other:", cm2.get("Product->Other"))
print("    Other->Product:", cm2.get("Other->Product"))
print("    Other->Other:", cm2.get("Other->Other"))

print("\nMisclassified eval samples")
print("  count:", len(misclassified))
for row in misclassified:
    print(
        "  -",
        row.get("filename"),
        "| true=", row.get("true_label"),
        "| pred=", row.get("predicted_label"),
        "| conf=", row.get("confidence"),
    )

print("\nArtifacts")
print("  classifier:", metadata.get("artifacts", {}).get("classifier_path"))
print("  versioned_classifier:", metadata.get("artifacts", {}).get("versioned_classifier_path"))
PY
