#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env}"
LABEL_STUDIO_URL="${LABEL_STUDIO_URL:-http://localhost:18080}"
IMAGE_REVIEW_TASKS_PATH="${IMAGE_REVIEW_TASKS_PATH:-demo_data/tasks/image_classification_review_tasks.json}"

SKIP_EXPORT=0
DRY_RUN_IMPORT=0
NO_ML_BACKEND=0
NO_WEBHOOK=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-export) SKIP_EXPORT=1 ;;
    --dry-run-import) DRY_RUN_IMPORT=1 ;;
    --no-ml-backend) NO_ML_BACKEND=1 ;;
    --no-webhook) NO_WEBHOOK=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/run_label_studio_human_review_flow_check.sh [options]

Options:
  --skip-export      Do not regenerate image review tasks before import.
  --dry-run-import  Validate and dedupe import plan, but do not import tasks.
  --no-ml-backend   Skip ML backend setup during bootstrap.
  --no-webhook      Skip webhook setup during bootstrap.
EOF
      exit 0
      ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

COMPOSE_CMD=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

log() { echo "[INFO] $*"; }
ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

check_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

check_service() {
  local service="$1"
  local raw status health
  if ! raw="$(${COMPOSE_CMD[@]} ps --format json "$service" 2>/dev/null)" || [ -z "$raw" ]; then
    fail "compose service not found or not running: ${service}. Try: docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} up -d --build"
  fi
  status="$(RAW_JSON="$raw" python3 -c 'import json, os
rows=[]
for line in os.environ.get("RAW_JSON", "").splitlines():
    line=line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
print((rows[0].get("State") or rows[0].get("Status") or "unknown") if rows else "unknown")')"
  health="$(RAW_JSON="$raw" python3 -c 'import json, os
rows=[]
for line in os.environ.get("RAW_JSON", "").splitlines():
    line=line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
print((rows[0].get("Health") or "") if rows else "")')"
  if [ "$status" != "running" ]; then
    fail "compose service ${service} is not running (state=${status}). Try: docker compose --env-file ${ENV_FILE} -f ${COMPOSE_FILE} ps"
  fi
  if [ -n "$health" ] && [ "$health" != "healthy" ]; then
    warn "compose service ${service} is running but health=${health}"
  else
    ok "compose service ${service} is running${health:+ and ${health}}"
  fi
}

main() {
  cd "${ROOT_DIR}"
  check_command docker
  check_command python3

  [ -f "${COMPOSE_FILE}" ] || fail "compose file not found: ${COMPOSE_FILE}"
  [ -f "scripts/bootstrap_label_studio_image_review.py" ] || fail "missing bootstrap script"
  [ -f "scripts/import_image_review_tasks_to_label_studio.py" ] || fail "missing import script"

  log "checking docker compose services"
  check_service label-studio
  check_service trainer
  check_service ml-backend

  if [ -z "${LABEL_STUDIO_API_TOKEN:-}" ]; then
    fail "LABEL_STUDIO_API_TOKEN is required. In Label Studio, open Account & Settings -> Access Token, then run: export LABEL_STUDIO_API_TOKEN='<your-token>'"
  fi
  log "Label Studio URL: ${LABEL_STUDIO_URL}"
  ok "LABEL_STUDIO_API_TOKEN is set (value hidden)"

  log "checking Label Studio API accessibility"
  python3 - <<'PY'
from scripts.lib.label_studio_client import LabelStudioClient, LabelStudioSettings
client = LabelStudioClient(LabelStudioSettings.from_env(require_token=True))
client.health_check()
print('[OK] Label Studio API reachable')
PY

  bootstrap_args=()
  if [ "${NO_ML_BACKEND}" = "1" ]; then bootstrap_args+=(--skip-ml-backend); fi
  if [ "${NO_WEBHOOK}" = "1" ]; then bootstrap_args+=(--skip-webhook); fi

  log "bootstrapping Label Studio image review project"
  python3 scripts/bootstrap_label_studio_image_review.py "${bootstrap_args[@]}"

  if [ "${SKIP_EXPORT}" = "1" ]; then
    warn "skipping review task export"
  else
    log "exporting image review tasks from current image classifier metadata"
    scripts/export_image_review_tasks_from_metadata.sh
  fi

  import_args=(--tasks-path "${IMAGE_REVIEW_TASKS_PATH}")
  if [ "${DRY_RUN_IMPORT}" = "1" ]; then import_args+=(--dry-run); fi
  log "importing review tasks into Label Studio"
  python3 scripts/import_image_review_tasks_to_label_studio.py "${import_args[@]}"

  cat <<EOF

[OK] Automated part of human review flow completed.

[INFO] Manual checkpoint. Do not skip this by pretending annotations happened:
1. Open the project URL printed above, or ${LABEL_STUDIO_URL}/projects.
2. Open the "Image Classification Human Review" project.
3. Review imported images, correct/confirm Product vs Other, then Submit/Update annotations.
4. Label Studio webhook should call trainer: ${LABEL_STUDIO_TRAINER_WEBHOOK_URL:-http://trainer:9091/webhook/label-studio}
5. Trainer should fetch the full task through Label Studio API and append candidates to ${IMAGE_TRAINING_CANDIDATES_PATH:-demo_data/tasks/image_classification_training_candidates.jsonl}.

[INFO] Follow-up verification commands:
tail -n 20 ${IMAGE_TRAINING_CANDIDATES_PATH:-demo_data/tasks/image_classification_training_candidates.jsonl}
scripts/trigger_image_classifier_retrain.sh
scripts/inspect_image_classifier_metadata.sh
scripts/run_image_classifier_validation_gate.sh
EOF
}

main "$@"
