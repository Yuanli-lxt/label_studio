#!/usr/bin/env bash
set -euo pipefail

TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
curl -sS "$TRAINER_URL/models/image-segmentation/current"
echo
