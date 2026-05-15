#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env}"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
ML_URL="${ML_URL:-http://localhost:9090}"
GATE_MODE="${TEXT_GATE_MODE:-auto}"  # auto|host|container

DEMO_DATASET_HOST="${TEXT_CLASSIFIER_DEMO_DATASET_PATH_HOST:-demo_data/tasks/text_classification_labeled_export.json}"
DEMO_DATASET_CONTAINER="${TEXT_CLASSIFIER_DEMO_DATASET_PATH:-/demo-tasks/text_classification_labeled_export.json}"
PROBE_FILE_HOST="${TEXT_CLASSIFIER_PROBE_FILE:-demo_data/tasks/text_classification_regression_probes.json}"
PROBE_FILE_CONTAINER="${TEXT_CLASSIFIER_PROBE_FILE_CONTAINER:-/demo-tasks/text_classification_regression_probes.json}"
ARTIFACT_DIR="${TEXT_CLASSIFIER_ARTIFACT_DIR:-demo_data/model_state/text_classifier}"

COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

log() {
  echo "[gate] $*"
}

fail() {
  echo "[gate][FAIL] $*" >&2
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
    "task_type": "text_classification",
    "dataset_path": dataset_path,
    "notes": "validation-gate-container-mode",
}
req = urllib.request.Request(
    "http://127.0.0.1:9091/train/text-classification",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
if not resp.get("accepted"):
    raise SystemExit(f"train request not accepted: {resp}")
if resp.get("mode") != "real-text-classifier":
    raise SystemExit(f"unexpected training mode: {resp.get('mode')} train_error={resp.get('train_error')}")
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
    "task_type": "text_classification",
    "dataset_path": dataset_path,
    "notes": "validation-gate-host-mode",
}
req = urllib.request.Request(
    trainer_url + "/train/text-classification",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
if not resp.get("accepted"):
    raise SystemExit(f"train request not accepted: {resp}")
if resp.get("mode") != "real-text-classifier":
    raise SystemExit(f"unexpected training mode: {resp.get('mode')} train_error={resp.get('train_error')}")
print(json.dumps(resp, indent=2))
PY
}

metadata_from_host() {
  python3 - <<'PY' "$TRAINER_URL"
import json
import sys
import urllib.request
url = sys.argv[1].rstrip('/') + '/models/text-classification/current'
print(urllib.request.urlopen(url, timeout=10).read().decode())
PY
}

metadata_from_container() {
  "${COMPOSE_CMD[@]}" exec -T trainer python - <<'PY'
import urllib.request
print(urllib.request.urlopen('http://127.0.0.1:9091/models/text-classification/current', timeout=10).read().decode())
PY
}

print_summary() {
  local metadata_json="$1"
  python3 - <<'PY' "$metadata_json"
import json
import sys
m = json.loads(sys.argv[1])
q = m.get('dataset', {}).get('quality', {})
print('[gate] Summary')
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
  local required=(classifier.joblib vectorizer.joblib metadata.json)
  for f in "${required[@]}"; do
    [ -s "${base}/${f}" ] || fail "missing or empty artifact: ${ARTIFACT_DIR}/${f}"
  done
  log "artifact check passed (${ARTIFACT_DIR})"
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

  require_file "scripts/trigger_text_classifier_retrain.sh"
  require_file "scripts/inspect_text_classifier_metadata.sh"
  require_file "scripts/run_text_classifier_regression_probes.sh"
  require_file "${PROBE_FILE_HOST}"

  log "starting text-classifier validation gate"

  local metadata_json
  if should_use_container_mode; then
    log "mode=container (using docker compose exec for local service calls)"

    log "step 1/4 retrain"
    retrain_in_container

    log "step 2/4 inspect metadata"
    metadata_json="$(metadata_from_container)" || fail "unable to fetch metadata from trainer"
    print_summary "${metadata_json}"

    log "step 3/4 run regression probes"
    run_script_in_trainer "${ROOT_DIR}/scripts/run_text_classifier_regression_probes.sh" \
      "ML_URL=http://ml-backend:9090" \
      "${PROBE_FILE_CONTAINER}"
  else
    log "mode=host"

    log "step 1/4 retrain"
    retrain_from_host

    log "step 2/4 inspect metadata"
    metadata_json="$(metadata_from_host)" || fail "unable to fetch metadata from trainer"
    print_summary "${metadata_json}"
    "${ROOT_DIR}/scripts/inspect_text_classifier_metadata.sh"

    log "step 3/4 run regression probes"
    "${ROOT_DIR}/scripts/run_text_classifier_regression_probes.sh" "${PROBE_FILE_HOST}"
  fi

  log "step 4/4 check persisted artifacts"
  check_artifacts

  echo
  echo "[gate][PASS] text-classifier validation gate completed"
}

main "$@"
