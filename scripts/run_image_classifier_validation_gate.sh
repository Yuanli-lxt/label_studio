#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env}"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
ML_URL="${ML_URL:-http://localhost:9090}"
GATE_MODE="${IMAGE_GATE_MODE:-auto}"  # auto|host|container

DEMO_DATASET_HOST="${IMAGE_CLASSIFIER_DEMO_DATASET_PATH_HOST:-demo_data/tasks/image_classification_labeled_export.json}"
DEMO_DATASET_CONTAINER="${IMAGE_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/image_classification_labeled_export.json}"
PROBE_FILE_HOST="${IMAGE_CLASSIFIER_PROBE_FILE:-demo_data/tasks/image_classification_regression_probes.json}"
PROBE_FILE_CONTAINER="${IMAGE_CLASSIFIER_PROBE_FILE_CONTAINER:-/demo-tasks/image_classification_regression_probes.json}"
ARTIFACT_DIR="${IMAGE_CLASSIFIER_ARTIFACT_DIR:-demo_data/model_state/image_classifier}"
EVAL_MANIFEST_HOST="${IMAGE_CLASSIFIER_EVAL_MANIFEST_PATH_HOST:-demo_data/tasks/image_classification_eval_manifest.json}"
EXPECTED_EVAL_SET_ID="${IMAGE_CLASSIFIER_EXPECTED_EVAL_SET_ID:-image-cls-eval-v1}"

COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

log() {
  echo "[image-gate] $*"
}

fail() {
  echo "[image-gate][FAIL] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -f "${ROOT_DIR}/${path}" ] || fail "required file not found: ${path}"
}

run_script_in_trainer() {
  local script_path="$1"
  local env_prefix="$2"
  local arg_string="$3"
  local script_name
  script_name="$(basename "${script_path}")"

  local cmd="cat > /tmp/${script_name} && chmod +x /tmp/${script_name} && ${env_prefix} /tmp/${script_name} ${arg_string}"
  "${COMPOSE_CMD[@]}" exec -T trainer sh -lc "${cmd}" < "${script_path}"
}

retrain_in_container() {
  "${COMPOSE_CMD[@]}" exec -T trainer python - <<'PY' "${DEMO_DATASET_CONTAINER}"
import json
import sys
import urllib.request

dataset_path = sys.argv[1]
payload = {
    "task_type": "image_classification",
    "dataset_path": dataset_path,
    "notes": "image-validation-gate-container-mode",
}
req = urllib.request.Request(
    "http://127.0.0.1:9091/train/image-classification",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
if not resp.get("accepted"):
    raise SystemExit(f"train request not accepted: {resp}")
if resp.get("mode") != "real-image-classifier":
    raise SystemExit(f"unexpected training mode: {resp.get('mode')} train_error={resp.get('train_error')}")
if resp.get("task_type") != "image_classification":
    raise SystemExit(f"unexpected task_type: {resp.get('task_type')}")
print(json.dumps(resp, indent=2))
PY
}

retrain_from_host() {
  python3 - <<'PY' "${TRAINER_URL}" "${DEMO_DATASET_CONTAINER}"
import json
import sys
import urllib.request

trainer_url = sys.argv[1].rstrip("/")
dataset_path = sys.argv[2]
payload = {
    "task_type": "image_classification",
    "dataset_path": dataset_path,
    "notes": "image-validation-gate-host-mode",
}
req = urllib.request.Request(
    trainer_url + "/train/image-classification",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
if not resp.get("accepted"):
    raise SystemExit(f"train request not accepted: {resp}")
if resp.get("mode") != "real-image-classifier":
    raise SystemExit(f"unexpected training mode: {resp.get('mode')} train_error={resp.get('train_error')}")
if resp.get("task_type") != "image_classification":
    raise SystemExit(f"unexpected task_type: {resp.get('task_type')}")
print(json.dumps(resp, indent=2))
PY
}

metadata_from_host() {
  python3 - <<'PY' "$TRAINER_URL"
import sys
import urllib.request
url = sys.argv[1].rstrip('/') + '/models/image-classification/current'
print(urllib.request.urlopen(url, timeout=10).read().decode())
PY
}

metadata_from_container() {
  "${COMPOSE_CMD[@]}" exec -T trainer python - <<'PY'
import urllib.request
print(urllib.request.urlopen('http://127.0.0.1:9091/models/image-classification/current', timeout=10).read().decode())
PY
}

print_summary() {
  local metadata_json="$1"
  python3 - <<'PY' "$metadata_json"
import json
import sys
m = json.loads(sys.argv[1])
q = m.get('dataset', {}).get('quality', {})
print('[image-gate] Summary')
print('  model_version:', m.get('model_version'))
print('  source_path:', m.get('dataset', {}).get('source_path'))
print('  total_samples:', q.get('total_samples'))
print('  label_distribution:', q.get('label_distribution'))
print('  accuracy:', m.get('metrics', {}).get('accuracy'))
print('  macro_f1:', m.get('metrics', {}).get('macro_f1'))
PY
}

check_artifacts() {
  local base="${ROOT_DIR}/${ARTIFACT_DIR}"
  local required=(classifier.joblib metadata.json last_training_dataset.jsonl)
  for f in "${required[@]}"; do
    [ -s "${base}/${f}" ] || fail "missing or empty artifact: ${ARTIFACT_DIR}/${f}"
  done
  log "artifact check passed (${ARTIFACT_DIR})"
}

check_dataset_integrity() {
  local metadata_json="$1"
  python3 - <<'PY' "$metadata_json" "${ROOT_DIR}/${EVAL_MANIFEST_HOST}" "${EXPECTED_EVAL_SET_ID}"
import json
import sys
from pathlib import Path

metadata = json.loads(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_eval_set_id = sys.argv[3]

if not manifest_path.exists():
    raise SystemExit(f"missing eval manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_eval_set_id = manifest.get("eval_set_id")
if manifest_eval_set_id != expected_eval_set_id:
    raise SystemExit(
        f"eval manifest id mismatch: expected {expected_eval_set_id}, got {manifest_eval_set_id}"
    )

manifest_images = manifest.get("images")
if not isinstance(manifest_images, list) or not manifest_images:
    raise SystemExit("eval manifest images is empty or invalid")

manifest_labels = {}
for row in manifest_images:
    if not isinstance(row, dict):
        raise SystemExit("eval manifest row must be object")
    filename = row.get("filename")
    label = row.get("label")
    if not isinstance(filename, str) or not filename:
        raise SystemExit("eval manifest row missing filename")
    if label not in {"Product", "Other"}:
        raise SystemExit(f"eval manifest label invalid for {filename}: {label}")
    manifest_labels[filename] = label

fixed_eval = metadata.get("dataset", {}).get("fixed_eval", {})
eval_set_id = fixed_eval.get("eval_set_id")
if eval_set_id != expected_eval_set_id:
    raise SystemExit(f"metadata eval_set_id mismatch: expected {expected_eval_set_id}, got {eval_set_id}")

train_files = set(fixed_eval.get("train_filenames") or [])
eval_files = set(fixed_eval.get("eval_filenames") or [])
manifest_files = set(manifest_labels.keys())

overlap = sorted(train_files.intersection(manifest_files))
if overlap:
    raise SystemExit(f"eval leakage detected: train filenames overlap eval manifest: {overlap}")

if eval_files != manifest_files:
    missing = sorted(manifest_files - eval_files)
    extra = sorted(eval_files - manifest_files)
    raise SystemExit(f"eval filename mismatch vs manifest: missing={missing} extra={extra}")

eval_predictions = metadata.get("metrics", {}).get("eval_predictions") or []
pred_true_labels = {}
for row in eval_predictions:
    if not isinstance(row, dict):
        continue
    filename = row.get("filename")
    true_label = row.get("true_label")
    if isinstance(filename, str) and filename:
        pred_true_labels[filename] = true_label

label_mismatch = []
for filename, expected_label in sorted(manifest_labels.items()):
    actual_label = pred_true_labels.get(filename)
    if actual_label != expected_label:
        label_mismatch.append(
            {"filename": filename, "expected_label": expected_label, "actual_label": actual_label}
        )
if label_mismatch:
    raise SystemExit(f"eval label mismatch vs manifest: {label_mismatch}")

print("[image-gate] dataset integrity check passed")
PY
}

should_use_container_mode() {
  if [ "${GATE_MODE}" = "container" ]; then
    return 0
  fi
  if [ "${GATE_MODE}" = "host" ]; then
    return 1
  fi

  # auto mode: prefer host if reachable, otherwise fall back to container
  if python3 - <<'PY' "$TRAINER_URL" >/dev/null 2>&1
import sys
import urllib.request
url = sys.argv[1].rstrip('/') + '/health'
urllib.request.urlopen(url, timeout=2)
PY
  then
    return 1
  fi
  return 0
}

main() {
  cd "${ROOT_DIR}"

  require_file "scripts/trigger_image_classifier_retrain.sh"
  require_file "scripts/inspect_image_classifier_metadata.sh"
  require_file "scripts/run_image_classifier_regression_probes.sh"
  require_file "${PROBE_FILE_HOST}"
  require_file "${EVAL_MANIFEST_HOST}"

  log "starting image-classifier validation gate"

  local metadata_json
  if should_use_container_mode; then
    log "mode=container (using docker compose exec for local service calls)"

    log "step 1/4 retrain"
    retrain_in_container

    log "step 2/4 inspect metadata"
    metadata_json="$(metadata_from_container)" || fail "unable to fetch image metadata from trainer"
    print_summary "${metadata_json}"

    log "step 3/4 run regression probes"
    run_script_in_trainer "${ROOT_DIR}/scripts/run_image_classifier_regression_probes.sh" \
      "ML_URL=http://ml-backend:9090" \
      "${PROBE_FILE_CONTAINER}"
  else
    log "mode=host"

    log "step 1/4 retrain"
    retrain_from_host

    log "step 2/4 inspect metadata"
    metadata_json="$(metadata_from_host)" || fail "unable to fetch image metadata from trainer"
    print_summary "${metadata_json}"
    "${ROOT_DIR}/scripts/inspect_image_classifier_metadata.sh"

    log "step 3/4 run regression probes"
    "${ROOT_DIR}/scripts/run_image_classifier_regression_probes.sh" "${PROBE_FILE_HOST}"
  fi

  log "step 4/4 check persisted artifacts"
  check_artifacts
  check_dataset_integrity "${metadata_json}"

  echo
  echo "[image-gate][PASS] image-classifier validation gate completed"
}

main "$@"
