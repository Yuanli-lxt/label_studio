#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://localhost:9090}"
PROBE_FILE="${1:-demo_data/tasks/text_classification_regression_probes.json}"

if [ ! -f "${PROBE_FILE}" ]; then
  echo "Probe file not found: ${PROBE_FILE}" >&2
  exit 2
fi

probes_json="$(cat "${PROBE_FILE}")"

python3 - <<'PY' "$ML_URL" "$probes_json"
import json
import sys
import urllib.request

ml_url = sys.argv[1].rstrip("/")
probes = json.loads(sys.argv[2])

if not isinstance(probes, list) or not probes:
    print("Probe file is empty or invalid", file=sys.stderr)
    raise SystemExit(2)

label_config = (
    '<View><Text name="text" value="$text"/>'
    '<Choices name="text_label" toName="text">'
    '<Choice value="Positive"/><Choice value="Negative"/>'
    '</Choices></View>'
)

payload = {
    "tasks": [{"id": probe["id"], "data": {"text": probe["text"]}} for probe in probes],
    "label_config": label_config,
}

req = urllib.request.Request(
    f"{ml_url}/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    response = urllib.request.urlopen(req, timeout=10)
except Exception as exc:
    print(f"Prediction request failed: {exc}", file=sys.stderr)
    raise SystemExit(2)

pred_payload = json.loads(response.read().decode("utf-8"))
preds_by_id = {}
for pred in pred_payload.get("results", []):
    result = (pred.get("result") or [{}])[0]
    pred_id = str(result.get("id", ""))
    task_id = pred_id.replace("_txt_cls", "") if pred_id else None
    if task_id:
        preds_by_id[task_id] = pred

print("Text Classification Regression Probes")
print(f"Model version: {pred_payload.get('model_version')}")
print(f"Probe count: {len(probes)}")
print()

failures = []
for probe in probes:
    probe_id = probe["id"]
    pred = preds_by_id.get(probe_id)
    if pred is None:
        failures.append((probe_id, "missing_prediction"))
        print(f"[FAIL] {probe_id}: missing prediction")
        continue

    result = (pred.get("result") or [{}])[0]
    predicted_label = (result.get("value", {}).get("choices") or [None])[0]
    score = pred.get("score")
    conf = pred.get("confidence", {}) if isinstance(pred.get("confidence"), dict) else {}
    uncertain = conf.get("uncertain")

    expected_label = probe["expected_label"]
    min_score = probe.get("min_score")
    expected_uncertain = probe.get("expected_uncertain", None)

    reasons = []
    if predicted_label != expected_label:
        reasons.append(f"expected_label={expected_label}, got={predicted_label}")
    if min_score is not None and (score is None or float(score) < float(min_score)):
        reasons.append(f"score<{min_score} (got={score})")
    if expected_uncertain is not None and uncertain != expected_uncertain:
        reasons.append(f"expected_uncertain={expected_uncertain}, got={uncertain}")

    status = "PASS" if not reasons else "FAIL"
    print(
        f"[{status}] {probe_id} | {probe['category']} | "
        f"pred={predicted_label} score={score} uncertain={uncertain} | "
        f"expected={expected_label}"
    )
    print(f"       text: {probe['text']}")
    if reasons:
        print(f"       reasons: {'; '.join(reasons)}")
        failures.append((probe_id, reasons))

print()
if failures:
    print(f"Regression probe failures: {len(failures)}")
    raise SystemExit(1)

print("All regression probes passed.")
PY
