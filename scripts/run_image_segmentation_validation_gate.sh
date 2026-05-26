#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINER_URL="${TRAINER_URL:-http://localhost:9091}"
ML_BACKEND_URL="${ML_BACKEND_URL:-http://localhost:9090}"

echo "[INFO] trigger image segmentation retrain"
TRAINER_URL="$TRAINER_URL" "$ROOT/scripts/trigger_image_segmentation_retrain.sh" >/tmp/image_seg_retrain.json
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/image_seg_retrain.json").read_text(encoding="utf-8"))
assert data.get("accepted") is True, data
assert data.get("task_type") == "image_segmentation", data
assert data.get("mode") == "placeholder-image-segmentation", data
PY

echo "[INFO] inspect trainer metadata"
curl -sS "$TRAINER_URL/models/image-segmentation/current" >/tmp/image_seg_metadata.json
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/image_seg_metadata.json").read_text(encoding="utf-8"))
assert data.get("task_type") == "image_segmentation", data
assert data.get("model_version", "").startswith("image-seg-v"), data
assert data.get("dataset", {}).get("total_size", 0) >= 1, data
PY

echo "[INFO] check ML backend prediction"
python3 - <<PY
import json
import urllib.request

config = open("label_configs/image_segmentation.xml", encoding="utf-8").read()
payload = {
    "label_config": config,
    "tasks": [{"id": "gate-seg", "data": {"image": "/data/local-files/?d=images/demo_blue.png"}}],
}
req = urllib.request.Request(
    "${ML_BACKEND_URL}/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=5) as resp:
    data = json.loads(resp.read().decode())
assert data["mode"] == "image_segmentation", data
result = data["results"][0]["result"][0]
assert result["type"] == "brushlabels", result
assert result["value"]["format"] == "rle", result
assert result["value"]["brushlabels"] == ["Object"], result
PY

echo "[OK] image segmentation validation gate passed"
