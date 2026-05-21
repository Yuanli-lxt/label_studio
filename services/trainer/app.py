import json
import os
import threading
import base64
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.request

import joblib
import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - handled with explicit runtime checks
    Image = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SERVICE_NAME = os.getenv("SERVICE_NAME", "trainer")
PORT = int(os.getenv("PORT", "9091"))
MODEL_STATE_PATH = os.getenv("MODEL_STATE_PATH", "/model-state/current_model.json")
TRAINING_EVENT_LOG_PATH = os.getenv("TRAINING_EVENT_LOG_PATH", "/model-state/training_events.jsonl")
TEXT_MODEL_ARTIFACTS_DIR = os.getenv(
    "TEXT_MODEL_ARTIFACTS_DIR",
    os.getenv("MODEL_ARTIFACTS_DIR", "/model-state/text_classifier"),
)
IMAGE_MODEL_ARTIFACTS_DIR = os.getenv("IMAGE_MODEL_ARTIFACTS_DIR", "/model-state/image_classifier")
IMAGE_SEG_MODEL_ARTIFACTS_DIR = os.getenv(
    "IMAGE_SEG_MODEL_ARTIFACTS_DIR",
    "/model-state/image_segmentation",
)
TRAINING_DATASET_EVENTS_PATH = os.getenv(
    "TRAINING_DATASET_EVENTS_PATH",
    "/model-state/text_classifier/training_dataset_events.jsonl",
)
IMAGE_LOCAL_FILES_ROOT = os.getenv("IMAGE_LOCAL_FILES_ROOT", "/demo-local-files")

TEXT_CLS_MIN_TOTAL_SAMPLES = int(os.getenv("TEXT_CLS_MIN_TOTAL_SAMPLES", "8"))
TEXT_CLS_MIN_PER_CLASS_SAMPLES = int(os.getenv("TEXT_CLS_MIN_PER_CLASS_SAMPLES", "3"))
TEXT_CLS_MAX_IMBALANCE_RATIO = float(os.getenv("TEXT_CLS_MAX_IMBALANCE_RATIO", "3.0"))
TEXT_CLS_EVAL_SPLIT_RATIO = float(os.getenv("TEXT_CLS_EVAL_SPLIT_RATIO", "0.25"))
TEXT_CLS_MIN_EVAL_SAMPLES = int(os.getenv("TEXT_CLS_MIN_EVAL_SAMPLES", "2"))
TEXT_CLS_UNCERTAIN_THRESHOLD = float(os.getenv("TEXT_CLS_UNCERTAIN_THRESHOLD", "0.60"))

IMAGE_CLS_MIN_TOTAL_SAMPLES = int(os.getenv("IMAGE_CLS_MIN_TOTAL_SAMPLES", "6"))
IMAGE_CLS_MIN_PER_CLASS_SAMPLES = int(os.getenv("IMAGE_CLS_MIN_PER_CLASS_SAMPLES", "2"))
IMAGE_CLS_MAX_IMBALANCE_RATIO = float(os.getenv("IMAGE_CLS_MAX_IMBALANCE_RATIO", "3.0"))
IMAGE_CLS_EVAL_SPLIT_RATIO = float(os.getenv("IMAGE_CLS_EVAL_SPLIT_RATIO", "0.25"))
IMAGE_CLS_MIN_EVAL_SAMPLES = int(os.getenv("IMAGE_CLS_MIN_EVAL_SAMPLES", "2"))
IMAGE_CLS_UNCERTAIN_THRESHOLD = float(os.getenv("IMAGE_CLS_UNCERTAIN_THRESHOLD", "0.60"))
IMAGE_CLS_EVAL_MANIFEST_PATH = os.getenv(
    "IMAGE_CLS_EVAL_MANIFEST_PATH",
    "/demo-tasks/image_classification_eval_manifest.json",
)
IMAGE_TRAINING_CANDIDATES_PATH = os.getenv(
    "IMAGE_TRAINING_CANDIDATES_PATH",
    "/demo-tasks/image_classification_training_candidates.jsonl",
)
IMAGE_SEG_TRAINING_CANDIDATES_PATH = os.getenv(
    "IMAGE_SEG_TRAINING_CANDIDATES_PATH",
    "/demo-tasks/image_segmentation_training_candidates.jsonl",
)
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080")
LABEL_STUDIO_API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN", "")
LABEL_STUDIO_TIMEOUT_SECONDS = float(os.getenv("LABEL_STUDIO_TIMEOUT_SECONDS", "5.0"))
_LABEL_STUDIO_ACCESS_TOKEN_CACHE = {}

_TEXT_CLASSIFIER_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "classifier.joblib")
_TEXT_VECTORIZER_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "vectorizer.joblib")
_TEXT_METADATA_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "metadata.json")
_TEXT_LAST_DATASET_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "last_training_dataset.jsonl")

_IMAGE_CLASSIFIER_PATH = os.path.join(IMAGE_MODEL_ARTIFACTS_DIR, "classifier.joblib")
_IMAGE_METADATA_PATH = os.path.join(IMAGE_MODEL_ARTIFACTS_DIR, "metadata.json")
_IMAGE_LAST_DATASET_PATH = os.path.join(IMAGE_MODEL_ARTIFACTS_DIR, "last_training_dataset.jsonl")
_IMAGE_SEG_ARTIFACT_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "segmenter.joblib")
_IMAGE_SEG_METADATA_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "metadata.json")
_IMAGE_SEG_LAST_DATASET_PATH = os.path.join(
    IMAGE_SEG_MODEL_ARTIFACTS_DIR,
    "last_training_dataset.jsonl",
)

# Backward-compatible aliases retained for existing tests/scripts.
MODEL_ARTIFACTS_DIR = TEXT_MODEL_ARTIFACTS_DIR
_CLASSIFIER_PATH = _TEXT_CLASSIFIER_PATH
_VECTORIZER_PATH = _TEXT_VECTORIZER_PATH
_METADATA_PATH = _TEXT_METADATA_PATH
_LAST_DATASET_PATH = _TEXT_LAST_DATASET_PATH

_STATE_LOCK = threading.Lock()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_state():
    return {
        "model_version": "demo-rule-v1",
        "training_run": 0,
        "updated_at": "1970-01-01T00:00:00Z",
        "source": "trainer-default",
        "text_classifier": {
            "active": False,
            "model_version": None,
            "trained_at": None,
            "metadata_path": _TEXT_METADATA_PATH,
        },
        "image_classifier": {
            "active": False,
            "model_version": None,
            "trained_at": None,
            "metadata_path": _IMAGE_METADATA_PATH,
        },
        "image_segmentation": {
            "active": False,
            "model_version": None,
            "trained_at": None,
            "artifact_path": _IMAGE_SEG_ARTIFACT_PATH,
            "metadata_path": _IMAGE_SEG_METADATA_PATH,
        },
        "behavior": {
            "image_positive_tokens": ["blue", "product", "object", "demo_blue"],
            "image_detection": {
                "x": 20.0,
                "y": 22.0,
                "width": 45.0,
                "height": 45.0,
                "score": 0.74,
            },
            "text_positive_tokens": ["love", "great", "good", "helpful", "excellent"],
            "ner_keywords": {
                "openai": "ORG",
                "alice": "PERSON",
                "paris": "LOC",
                "berlin": "LOC",
            },
            "score_boost": 0.0,
        },
    }


def _deepcopy_json(obj):
    return json.loads(json.dumps(obj))


def _round4(value):
    return round(float(value), 4)


def _round_nested(obj):
    if isinstance(obj, dict):
        return {key: _round_nested(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(item) for item in obj]
    if isinstance(obj, float):
        return _round4(obj)
    return obj


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_state():
    default_state = _default_state()
    if not os.path.exists(MODEL_STATE_PATH):
        _save_state(default_state)
        return default_state

    try:
        with open(MODEL_STATE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        _save_state(default_state)
        return default_state

    if not isinstance(loaded, dict):
        _save_state(default_state)
        return default_state

    state = _deepcopy_json(default_state)
    state.update(loaded)
    if isinstance(loaded.get("behavior"), dict):
        state["behavior"].update(loaded["behavior"])
    if isinstance(loaded.get("text_classifier"), dict):
        state["text_classifier"].update(loaded["text_classifier"])
    if isinstance(loaded.get("image_classifier"), dict):
        state["image_classifier"].update(loaded["image_classifier"])
    if isinstance(loaded.get("image_segmentation"), dict):
        state["image_segmentation"].update(loaded["image_segmentation"])
    return state


def _save_state(state):
    _ensure_parent(MODEL_STATE_PATH)
    with open(MODEL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _append_event_log(path, entry):
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length) if length else b"{}"
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def _send_json(handler, status_code, payload):
    content = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _collect_label_counts(result_items):
    counts = {}
    for item in result_items:
        value = item.get("value", {}) if isinstance(item, dict) else {}
        if not isinstance(value, dict):
            continue
        for key in ("choices", "rectanglelabels", "labels"):
            labels = value.get(key, [])
            if not isinstance(labels, list):
                continue
            for label in labels:
                name = str(label)
                counts[name] = counts.get(name, 0) + 1
    return counts


def _summarize_training_event(payload, trigger):
    project = payload.get("project")
    project_id = project.get("id") if isinstance(project, dict) else project

    annotation = payload.get("annotation")
    annotation_id = annotation.get("id") if isinstance(annotation, dict) else None
    annotation_result = annotation.get("result", []) if isinstance(annotation, dict) else []

    summary = {
        "trigger": trigger,
        "action": payload.get("action", "MANUAL_TRIGGER"),
        "project_id": project_id,
        "annotation_id": annotation_id,
        "task_number": payload.get("task_number"),
        "useful_annotation_number": payload.get("useful_annotation_number"),
        "label_counts": _collect_label_counts(annotation_result if isinstance(annotation_result, list) else []),
    }
    return summary


def _behavior_for_run(run_number):
    if run_number % 2 == 1:
        return {
            "image_positive_tokens": ["blue", "product", "object", "demo_blue", "green"],
            "image_detection": {
                "x": 30.0,
                "y": 28.0,
                "width": 38.0,
                "height": 40.0,
                "score": 0.82,
            },
            "text_positive_tokens": ["love", "great", "good", "helpful", "excellent", "traveled"],
            "ner_keywords": {
                "openai": "ORG",
                "alice": "PERSON",
                "paris": "LOC",
                "berlin": "LOC",
                "work": "ORG",
            },
            "score_boost": 0.08,
        }

    return {
        "image_positive_tokens": ["blue", "product", "object", "demo_blue"],
        "image_detection": {
            "x": 18.0,
            "y": 18.0,
            "width": 50.0,
            "height": 48.0,
            "score": 0.68,
        },
        "text_positive_tokens": ["love", "great", "good", "helpful", "excellent"],
        "ner_keywords": {
            "openai": "ORG",
            "alice": "PERSON",
            "paris": "LOC",
            "berlin": "LOC",
            "demo": "ORG",
        },
        "score_boost": -0.03,
    }


def _normalize_label(value):
    if value is None:
        return None
    label = " ".join(str(value).strip().split())
    return label if label else None


def _normalize_text(value):
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text if text else None


def _normalize_dataset_split(value):
    if value is None:
        return None
    split = str(value).strip().lower()
    if split in {"train", "eval"}:
        return split
    if split in {"validation", "val", "test"}:
        return "eval"
    return None


def _extract_choice_label(result_items):
    if not isinstance(result_items, list):
        return None
    for item in result_items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            label = _normalize_label(choices[0])
            if label:
                return label
    return None


def _parse_webhook_text_sample(payload):
    annotation = payload.get("annotation") if isinstance(payload, dict) else None
    annotation = annotation if isinstance(annotation, dict) else {}

    label = _extract_choice_label(annotation.get("result", []))

    task_obj = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task_obj, dict):
        task_obj = annotation.get("task") if isinstance(annotation.get("task"), dict) else None

    text = None
    task_id = None
    if isinstance(task_obj, dict):
        task_id = task_obj.get("id")
        data = task_obj.get("data")
        if isinstance(data, dict):
            text = _normalize_text(data.get("text"))

    if text and label:
        return {
            "text": text,
            "label": label,
            "task_id": task_id,
            "annotation_id": annotation.get("id"),
        }
    return None


def _normalize_project_id(value):
    if isinstance(value, dict):
        return _normalize_project_id(value.get("id"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_webhook_task_id(payload):
    if not isinstance(payload, dict):
        return None
    candidates = [payload.get("task_id")]
    task = payload.get("task")
    if isinstance(task, dict):
        candidates.append(task.get("id"))
    annotation = payload.get("annotation")
    if isinstance(annotation, dict):
        annotation_task = annotation.get("task")
        if isinstance(annotation_task, dict):
            candidates.append(annotation_task.get("id"))
        else:
            candidates.append(annotation_task)
    for candidate in candidates:
        try:
            if candidate is None:
                continue
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _extract_webhook_project_id(payload):
    if not isinstance(payload, dict):
        return None
    candidates = [payload.get("project"), payload.get("project_id")]
    task = payload.get("task")
    if isinstance(task, dict):
        candidates.append(task.get("project"))
        data = task.get("data")
        if isinstance(data, dict):
            candidates.append(data.get("project"))
    for candidate in candidates:
        project_id = _normalize_project_id(candidate)
        if project_id is not None:
            return project_id
    return None


def _label_studio_task_endpoint(task_id, project_id=None):
    base = LABEL_STUDIO_URL.rstrip("/")
    endpoint = f"{base}/api/tasks/{int(task_id)}/"
    if project_id is not None:
        endpoint = f"{endpoint}?project={int(project_id)}"
    return endpoint


def _jwt_payload(token):
    parts = str(token or "").strip().split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
        return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None


def _jwt_token_type(token):
    data = _jwt_payload(token)
    token_type = data.get("token_type") if isinstance(data, dict) else None
    return token_type if token_type in {"access", "refresh"} else None


def _refresh_label_studio_jwt_access_token(refresh_token):
    cache_key = (LABEL_STUDIO_URL.rstrip("/"), refresh_token)
    cached = _LABEL_STUDIO_ACCESS_TOKEN_CACHE.get(cache_key)
    if cached:
        access_token, expires_at = cached
        if expires_at and time.time() < (expires_at - 60):
            return access_token

    endpoint = f"{LABEL_STUDIO_URL.rstrip('/')}/api/token/refresh/"
    payload = json.dumps({"refresh": refresh_token}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=LABEL_STUDIO_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""
        raise ValueError(
            f"Label Studio JWT refresh failed: HTTP {exc.code}, endpoint={endpoint}, body={error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Label Studio JWT refresh failed: {exc.reason}, endpoint={endpoint}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label Studio JWT refresh returned invalid JSON, endpoint={endpoint}") from exc
    access_token = data.get("access") if isinstance(data, dict) else None
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError(f"Label Studio JWT refresh response missing access token, endpoint={endpoint}")
    access_token = access_token.strip()
    access_payload = _jwt_payload(access_token)
    expires_at = access_payload.get("exp") if isinstance(access_payload, dict) else None
    if isinstance(expires_at, (int, float)):
        _LABEL_STUDIO_ACCESS_TOKEN_CACHE[cache_key] = (access_token, float(expires_at))
    return access_token


def _label_studio_auth_header_value(token):
    token = str(token or "").strip()
    token_type = _jwt_token_type(token)
    if token_type == "access":
        return f"Bearer {token}"
    if token_type == "refresh":
        return f"Bearer {_refresh_label_studio_jwt_access_token(token)}"
    return f"Token {token}"


def _fetch_label_studio_task(task_id, project_id=None):
    if task_id is None:
        raise ValueError("missing task_id for Label Studio task fetch")
    if not LABEL_STUDIO_URL.strip():
        raise ValueError("LABEL_STUDIO_URL is empty")
    if not LABEL_STUDIO_API_TOKEN.strip():
        raise ValueError("LABEL_STUDIO_API_TOKEN is empty")

    endpoint = _label_studio_task_endpoint(task_id, project_id)
    req = urllib.request.Request(endpoint, method="GET")
    req.add_header("Authorization", _label_studio_auth_header_value(LABEL_STUDIO_API_TOKEN))
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=LABEL_STUDIO_TIMEOUT_SECONDS) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise ValueError(
            f"Label Studio API task fetch failed: HTTP {exc.code} for task_id={task_id}, "
            f"project_id={project_id}, endpoint={endpoint}, body={body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ValueError(
            f"Label Studio API task fetch failed: {exc.reason} for task_id={task_id}, "
            f"project_id={project_id}, endpoint={endpoint}"
        ) from exc

    try:
        task = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Label Studio API task fetch returned invalid JSON for task_id={task_id}, endpoint={endpoint}"
        ) from exc

    if not isinstance(task, dict):
        raise ValueError(
            f"Label Studio API task fetch returned non-object payload for task_id={task_id}, endpoint={endpoint}"
        )
    return task


def _normalize_image_classification_label(value):
    label = _normalize_label(value)
    if not label:
        return None
    low = label.lower()
    if low == "product":
        return "Product"
    if low == "other":
        return "Other"
    return None


def _normalize_image_segmentation_label(value):
    label = _normalize_label(value)
    if not label:
        return None
    if label.lower() == "object":
        return "Object"
    return None


def _positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _extract_brush_mask_result(result_items):
    if not isinstance(result_items, list):
        return None
    for item in result_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "brushlabels":
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        if str(value.get("format") or "").strip().lower() != "rle":
            continue

        rle = value.get("rle")
        if not isinstance(rle, list) or not rle:
            continue

        labels = value.get("brushlabels")
        if not isinstance(labels, list):
            continue

        label = None
        for raw_label in labels:
            label = _normalize_image_segmentation_label(raw_label)
            if label:
                break
        if label is None:
            continue

        original_width = _positive_int(item.get("original_width"))
        original_height = _positive_int(item.get("original_height"))
        if original_width is None or original_height is None:
            continue

        return {
            "label": label,
            "rle": rle,
            "original_width": original_width,
            "original_height": original_height,
        }
    return None


def _parse_image_classification_sample_from_full_task(task, fallback_project_id=None):
    if not isinstance(task, dict):
        raise ValueError("Label Studio task payload is not an object")

    task_id = task.get("id")
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        raise ValueError("Label Studio task payload missing valid task id")

    project_id = _normalize_project_id(task.get("project"))
    if project_id is None:
        project_id = _normalize_project_id(fallback_project_id)

    data = task.get("data")
    data = data if isinstance(data, dict) else {}
    image = data.get("image")
    if image is None:
        raise ValueError(f"Label Studio task {task_id} has no `data.image` field")

    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"Label Studio task {task_id} has no annotations")

    selected = None
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled"):
            continue
        raw_label = _extract_choice_label(annotation.get("result", []))
        label = _normalize_image_classification_label(raw_label)
        if label is None:
            continue
        updated_at = _normalize_text(annotation.get("updated_at") or annotation.get("created_at"))
        selected = {
            "annotation_id": annotation.get("id"),
            "label": label,
            "updated_at": updated_at,
        }

    if selected is None:
        raise ValueError(
            f"Label Studio task {task_id} has no non-cancelled image classification annotation "
            "with Product/Other label"
        )

    return {
        "task_id": task_id,
        "project_id": project_id,
        "image": image,
        "label": selected["label"],
        "source": "label_studio_webhook_task_fetch",
        "annotation_id": selected["annotation_id"],
        "updated_at": selected["updated_at"] or _utc_now_iso(),
    }


def _parse_image_segmentation_sample_from_full_task(task, fallback_project_id=None):
    if not isinstance(task, dict):
        raise ValueError("Label Studio task payload is not an object")

    task_id = task.get("id")
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        raise ValueError("Label Studio task payload missing valid task id")

    project_id = _normalize_project_id(task.get("project"))
    if project_id is None:
        project_id = _normalize_project_id(fallback_project_id)

    data = task.get("data")
    data = data if isinstance(data, dict) else {}
    image = data.get("image")
    if image is None:
        raise ValueError(f"Label Studio task {task_id} has no `data.image` field")

    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"Label Studio task {task_id} has no annotations")

    selected = None
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled"):
            continue
        mask = _extract_brush_mask_result(annotation.get("result", []))
        if mask is None:
            continue
        updated_at = _normalize_text(annotation.get("updated_at") or annotation.get("created_at"))
        selected = {
            "annotation_id": annotation.get("id"),
            "updated_at": updated_at,
            **mask,
        }

    if selected is None:
        raise ValueError(
            f"Label Studio task {task_id} has no non-cancelled image segmentation annotation "
            "with Object BrushLabels RLE mask"
        )

    return {
        "task_id": task_id,
        "project_id": project_id,
        "image": image,
        "label": selected["label"],
        "rle": selected["rle"],
        "original_width": selected["original_width"],
        "original_height": selected["original_height"],
        "source": "label_studio_webhook_task_fetch",
        "annotation_id": selected["annotation_id"],
        "updated_at": selected["updated_at"] or _utc_now_iso(),
    }


def _append_image_training_candidate(candidate):
    required_fields = ["task_id", "project_id", "image", "label", "source", "annotation_id", "updated_at"]
    for field in required_fields:
        if candidate.get(field) is None:
            raise ValueError(f"image training candidate missing required field: {field}")

    _ensure_parent(IMAGE_TRAINING_CANDIDATES_PATH)
    with open(IMAGE_TRAINING_CANDIDATES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def _append_image_segmentation_training_candidate(candidate):
    required_fields = [
        "task_id",
        "project_id",
        "image",
        "label",
        "rle",
        "original_width",
        "original_height",
        "source",
        "annotation_id",
        "updated_at",
    ]
    for field in required_fields:
        if candidate.get(field) is None:
            raise ValueError(f"image segmentation training candidate missing required field: {field}")

    _ensure_parent(IMAGE_SEG_TRAINING_CANDIDATES_PATH)
    with open(IMAGE_SEG_TRAINING_CANDIDATES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def _read_image_training_candidate_samples():
    if not os.path.exists(IMAGE_TRAINING_CANDIDATES_PATH):
        return [], {"total": 0, "used": 0, "skipped": 0, "errors": []}

    samples = []
    errors = []
    total = 0
    used = 0
    skipped = 0
    try:
        with open(IMAGE_TRAINING_CANDIDATES_PATH, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                content = line.strip()
                if not content:
                    continue
                total += 1
                try:
                    row = json.loads(content)
                except json.JSONDecodeError:
                    errors.append(f"line {line_no}: invalid JSON")
                    skipped += 1
                    continue
                if not isinstance(row, dict):
                    errors.append(f"line {line_no}: candidate row is not an object")
                    skipped += 1
                    continue

                image_ref = row.get("image")
                image_path = _resolve_image_file_path(image_ref)
                if not image_path:
                    errors.append(f"line {line_no}: cannot resolve image path for {image_ref}")
                    skipped += 1
                    continue

                label = _normalize_image_classification_label(row.get("label"))
                if label is None:
                    errors.append(
                        f"line {line_no}: unsupported label {row.get('label')}, expected Product/Other"
                    )
                    skipped += 1
                    continue

                sample = {
                    "image": image_ref,
                    "image_path": image_path,
                    "label": label,
                    "caption": _normalize_text(row.get("caption")),
                    "dataset_split": "train",
                    "eval_set_id": None,
                    "task_id": row.get("task_id"),
                    "annotation_id": row.get("annotation_id"),
                    "project_id": row.get("project_id"),
                    "updated_at": row.get("updated_at"),
                    "source": row.get("source") or "training_candidate",
                }
                samples.append(sample)
                used += 1
    except OSError as exc:
        errors.append(f"failed to read candidates file: {exc}")

    return samples, {"total": total, "used": used, "skipped": skipped, "errors": errors}


def _read_image_segmentation_training_candidate_samples():
    if not os.path.exists(IMAGE_SEG_TRAINING_CANDIDATES_PATH):
        return [], {"total": 0, "used": 0, "skipped": 0, "errors": []}

    samples = []
    errors = []
    total = 0
    used = 0
    skipped = 0
    try:
        with open(IMAGE_SEG_TRAINING_CANDIDATES_PATH, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                content = line.strip()
                if not content:
                    continue
                total += 1
                try:
                    row = json.loads(content)
                except json.JSONDecodeError:
                    errors.append(f"line {line_no}: invalid JSON")
                    skipped += 1
                    continue
                if not isinstance(row, dict):
                    errors.append(f"line {line_no}: candidate row is not an object")
                    skipped += 1
                    continue

                image_ref = row.get("image")
                image_path = _resolve_image_file_path(image_ref)
                if not image_path:
                    errors.append(f"line {line_no}: cannot resolve image path for {image_ref}")
                    skipped += 1
                    continue

                label = _normalize_image_segmentation_label(row.get("label"))
                if label is None:
                    errors.append(f"line {line_no}: unsupported label {row.get('label')}, expected Object")
                    skipped += 1
                    continue

                rle = row.get("rle")
                if not isinstance(rle, list) or not rle:
                    errors.append(f"line {line_no}: missing non-empty RLE mask")
                    skipped += 1
                    continue

                original_width = _positive_int(row.get("original_width"))
                original_height = _positive_int(row.get("original_height"))
                if original_width is None or original_height is None:
                    errors.append(f"line {line_no}: missing positive original dimensions")
                    skipped += 1
                    continue

                sample = {
                    "image": image_ref,
                    "image_path": image_path,
                    "label": label,
                    "rle": rle,
                    "original_width": original_width,
                    "original_height": original_height,
                    "dataset_split": "train",
                    "task_id": row.get("task_id"),
                    "annotation_id": row.get("annotation_id"),
                    "project_id": row.get("project_id"),
                    "updated_at": row.get("updated_at"),
                    "source": row.get("source") or "training_candidate",
                }
                samples.append(sample)
                used += 1
    except OSError as exc:
        errors.append(f"failed to read candidates file: {exc}")

    return samples, {"total": total, "used": used, "skipped": skipped, "errors": errors}


def _maybe_record_image_candidate_from_webhook(payload):
    task_id = _extract_webhook_task_id(payload)
    project_id = _extract_webhook_project_id(payload)
    if task_id is None:
        return None, "image webhook candidate ingest skipped: task_id not found in payload"

    full_task = _fetch_label_studio_task(task_id, project_id)
    candidate = _parse_image_classification_sample_from_full_task(full_task, project_id)
    _append_image_training_candidate(candidate)
    return candidate, None


def _maybe_record_image_segmentation_candidate_from_webhook(payload):
    task_id = _extract_webhook_task_id(payload)
    project_id = _extract_webhook_project_id(payload)
    if task_id is None:
        return None, "image segmentation webhook candidate ingest skipped: task_id not found in payload"

    full_task = _fetch_label_studio_task(task_id, project_id)
    candidate = _parse_image_segmentation_sample_from_full_task(full_task, project_id)
    _append_image_segmentation_training_candidate(candidate)
    return candidate, None


def _parse_export_task(task):
    if not isinstance(task, dict):
        return []

    data = task.get("data")
    text = _normalize_text(data.get("text")) if isinstance(data, dict) else None
    if not text:
        return []

    samples = []
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return samples

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled"):
            continue
        label = _extract_choice_label(annotation.get("result", []))
        if not label:
            continue
        samples.append(
            {
                "text": text,
                "label": label,
                "task_id": task.get("id"),
                "annotation_id": annotation.get("id"),
            }
        )
    return samples


def _read_samples_from_export(path):
    data = _read_json(path)
    if not isinstance(data, list):
        return []

    samples = []
    for task in data:
        samples.extend(_parse_export_task(task))
    return samples


def _normalize_manual_samples(samples):
    normalized = []
    if not isinstance(samples, list):
        return normalized

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        text = _normalize_text(sample.get("text"))
        label = _normalize_label(sample.get("label"))
        if text and label:
            normalized.append(
                {
                    "text": text,
                    "label": label,
                    "task_id": sample.get("task_id"),
                    "annotation_id": sample.get("annotation_id"),
                }
            )
    return normalized


def _dedupe_samples(samples):
    seen = set()
    output = []
    for sample in samples:
        dedupe_key = (sample.get("text", "").lower(), sample.get("label", "").lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        output.append(sample)
    return output


def _load_dataset_event_samples():
    if not os.path.exists(TRAINING_DATASET_EVENTS_PATH):
        return []

    samples = []
    try:
        with open(TRAINING_DATASET_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _normalize_text(entry.get("text"))
                label = _normalize_label(entry.get("label"))
                if text and label:
                    samples.append(
                        {
                            "text": text,
                            "label": label,
                            "task_id": entry.get("task_id"),
                            "annotation_id": entry.get("annotation_id"),
                        }
                    )
    except OSError:
        return []

    return samples


def _maybe_record_webhook_sample(payload):
    sample = _parse_webhook_text_sample(payload)
    if not sample:
        return None

    record = {
        "recorded_at": _utc_now_iso(),
        "text": sample["text"],
        "label": sample["label"],
        "task_id": sample.get("task_id"),
        "annotation_id": sample.get("annotation_id"),
    }
    _append_event_log(TRAINING_DATASET_EVENTS_PATH, record)
    return record


def _resolve_dataset_path(payload):
    if not isinstance(payload, dict):
        return None

    for key in ("label_studio_export_path", "dataset_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else None
    if nested:
        for key in ("label_studio_export_path", "dataset_path"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _safe_join_under_root(root, relative_path):
    root_abs = os.path.abspath(root)
    candidate = os.path.abspath(os.path.join(root_abs, relative_path.lstrip("/")))
    if candidate == root_abs or candidate.startswith(root_abs + os.sep):
        return candidate
    return None


def _resolve_image_file_path(image_value, local_root=IMAGE_LOCAL_FILES_ROOT):
    if image_value is None:
        return None

    raw = str(image_value).strip()
    if not raw:
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        return None

    parsed = urlparse(raw)
    if raw.startswith("/data/local-files/"):
        rel = parse_qs(parsed.query).get("d", [None])[0]
        if rel:
            candidate = _safe_join_under_root(local_root, rel)
            if candidate and os.path.exists(candidate):
                return candidate

    if raw.startswith("/label-studio/files/"):
        rel = raw[len("/label-studio/files/") :]
        candidate = _safe_join_under_root(local_root, rel)
        if candidate and os.path.exists(candidate):
            return candidate

    if raw.startswith("/"):
        if os.path.exists(raw):
            return raw

    candidate = _safe_join_under_root(local_root, raw)
    if candidate and os.path.exists(candidate):
        return candidate

    if "/images/" in raw:
        rel = raw.split("/images/", 1)[1]
        candidate = _safe_join_under_root(local_root, os.path.join("images", rel))
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def _normalize_manual_image_samples(samples):
    normalized = []
    if not isinstance(samples, list):
        return normalized

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_ref = sample.get("image") if sample.get("image") is not None else sample.get("image_path")
        image_path = _resolve_image_file_path(image_ref)
        label = _normalize_label(sample.get("label"))
        dataset_split = _normalize_dataset_split(sample.get("dataset_split") or sample.get("split"))
        eval_set_id = _normalize_text(sample.get("eval_set_id") or sample.get("eval_id"))
        if not image_path or not label:
            continue
        normalized.append(
            {
                "image": image_ref,
                "image_path": image_path,
                "label": label,
                "caption": _normalize_text(sample.get("caption")),
                "dataset_split": dataset_split,
                "eval_set_id": eval_set_id,
                "task_id": sample.get("task_id"),
                "annotation_id": sample.get("annotation_id"),
                "source": sample.get("source") or "manual",
            }
        )
    return normalized


def _parse_image_export_task(task):
    if not isinstance(task, dict):
        return []

    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    image_ref = data.get("image")
    image_path = _resolve_image_file_path(image_ref)
    if not image_path:
        return []

    caption = _normalize_text(data.get("caption"))
    dataset_split = _normalize_dataset_split(
        meta.get("dataset_split") or meta.get("split") or data.get("dataset_split") or data.get("split")
    )
    eval_set_id = _normalize_text(
        meta.get("eval_set_id") or meta.get("eval_id") or data.get("eval_set_id") or data.get("eval_id")
    )
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return []

    samples = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled"):
            continue
        label = _extract_choice_label(annotation.get("result", []))
        if not label:
            continue
        samples.append(
            {
                "image": image_ref,
                "image_path": image_path,
                "label": label,
                "caption": caption,
                "dataset_split": dataset_split,
                "eval_set_id": eval_set_id,
                "task_id": task.get("id"),
                "annotation_id": annotation.get("id"),
                "source": "label_studio_export",
            }
        )
    return samples


def _read_image_samples_from_export(path):
    data = _read_json(path)
    if not isinstance(data, list):
        return []

    samples = []
    for task in data:
        samples.extend(_parse_image_export_task(task))
    return samples


def _dedupe_image_samples(samples):
    deduped = {}
    order = []
    for sample in samples:
        dedupe_key = (sample.get("image_path", ""), sample.get("dataset_split") or "train")
        if dedupe_key not in deduped:
            order.append(dedupe_key)
        # Keep latest seen sample so webhook candidates can override base export labels.
        deduped[dedupe_key] = sample
    return [deduped[key] for key in order]


def _split_image_dataset(samples):
    labels = [sample["label"] for sample in samples]

    explicit_split_present = any(
        sample.get("dataset_split") in {"train", "eval"} for sample in samples
    )
    if not explicit_split_present:
        return _split_dataset_indices(
            labels,
            IMAGE_CLS_EVAL_SPLIT_RATIO,
            IMAGE_CLS_MIN_EVAL_SAMPLES,
        )

    train_idx = []
    eval_idx = []
    missing_idx = []
    eval_set_ids = set()
    for idx, sample in enumerate(samples):
        split = sample.get("dataset_split")
        if split == "train":
            train_idx.append(idx)
        elif split == "eval":
            eval_idx.append(idx)
        else:
            missing_idx.append(idx)
        eval_set_id = sample.get("eval_set_id")
        if eval_set_id:
            eval_set_ids.add(str(eval_set_id))

    if missing_idx:
        raise ValueError(
            "image dataset contains explicit split markers but some samples are missing "
            "`dataset_split` (train/eval)"
        )
    if not train_idx:
        raise ValueError("image dataset fixed split has no train samples")
    if not eval_idx:
        raise ValueError("image dataset fixed split has no eval samples")

    train_labels = [labels[i] for i in train_idx]
    eval_labels = [labels[i] for i in eval_idx]

    split_info = {
        "strategy": "fixed_eval_set",
        "source_field": "task.meta.dataset_split",
        "eval_set_id": (
            sorted(eval_set_ids)[0]
            if len(eval_set_ids) == 1
            else sorted(eval_set_ids)
            if eval_set_ids
            else None
        ),
        "train_size": len(train_idx),
        "eval_size": len(eval_idx),
        "train_label_distribution": {
            label: count for label, count in sorted(Counter(train_labels).items())
        },
        "eval_label_distribution": {
            label: count for label, count in sorted(Counter(eval_labels).items())
        },
    }
    return train_idx, eval_idx, split_info


def _build_image_classification_dataset(payload):
    request_payload = payload if isinstance(payload, dict) else {}
    dataset_path = _resolve_dataset_path(request_payload)

    base_samples = []
    samples = []
    manual_samples = _normalize_manual_image_samples(request_payload.get("samples"))
    base_samples.extend(manual_samples)

    nested_payload = (
        request_payload.get("payload")
        if isinstance(request_payload.get("payload"), dict)
        else None
    )
    if nested_payload:
        nested_manual = _normalize_manual_image_samples(nested_payload.get("samples"))
        base_samples.extend(nested_manual)

    if dataset_path:
        export_samples = _read_image_samples_from_export(dataset_path)
        base_samples.extend(export_samples)

    candidate_samples, candidate_stats = _read_image_training_candidate_samples()
    eval_manifest = None
    eval_filenames = set()
    if os.path.exists(IMAGE_CLS_EVAL_MANIFEST_PATH):
        eval_manifest = _load_image_eval_manifest(IMAGE_CLS_EVAL_MANIFEST_PATH)
        eval_filenames = set(eval_manifest["images"].keys())

    filtered_candidate_samples = []
    eval_leakage_skipped = 0
    for sample in candidate_samples:
        image_path = sample.get("image_path")
        file_name = os.path.basename(image_path) if image_path else None
        if file_name and file_name in eval_filenames:
            eval_leakage_skipped += 1
            continue
        filtered_candidate_samples.append(sample)

    samples.extend(base_samples)
    samples.extend(filtered_candidate_samples)

    samples = _dedupe_image_samples(samples)
    context = {
        "dataset_path": dataset_path,
        "local_files_root": IMAGE_LOCAL_FILES_ROOT,
        "candidates_path": IMAGE_TRAINING_CANDIDATES_PATH,
        "candidate_stats": candidate_stats,
        "candidate_used_after_eval_filter": len(filtered_candidate_samples),
        "candidate_eval_leakage_skipped": eval_leakage_skipped,
        "eval_manifest": {
            "path": IMAGE_CLS_EVAL_MANIFEST_PATH,
            "present": eval_manifest is not None,
            "eval_set_id": (eval_manifest or {}).get("eval_set_id"),
            "filename_count": len(eval_filenames),
        },
    }
    return samples, context


def _normalize_manual_image_segmentation_samples(samples):
    normalized = []
    if not isinstance(samples, list):
        return normalized

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        image_ref = sample.get("image")
        image_path = _resolve_image_file_path(image_ref)
        if not image_path and sample.get("image_path") is not None:
            image_ref = sample.get("image_path")
            image_path = _resolve_image_file_path(image_ref)
        label = _normalize_image_segmentation_label(sample.get("label"))
        rle = sample.get("rle")
        original_width = _positive_int(sample.get("original_width"))
        original_height = _positive_int(sample.get("original_height"))
        dataset_split = _normalize_dataset_split(sample.get("dataset_split") or sample.get("split"))
        if not image_path or label is None or not isinstance(rle, list) or not rle:
            continue
        if original_width is None or original_height is None:
            continue

        normalized.append(
            {
                "image": image_ref,
                "image_path": image_path,
                "label": label,
                "rle": rle,
                "original_width": original_width,
                "original_height": original_height,
                "dataset_split": dataset_split,
                "task_id": sample.get("task_id"),
                "annotation_id": sample.get("annotation_id"),
                "project_id": sample.get("project_id"),
                "updated_at": sample.get("updated_at"),
                "source": sample.get("source") or "manual",
            }
        )
    return normalized


def _parse_image_segmentation_export_task(task):
    if not isinstance(task, dict):
        return []

    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    image_ref = data.get("image") or meta.get("image")
    image_path = _resolve_image_file_path(image_ref)
    if not image_path:
        return []
    dataset_split = _normalize_dataset_split(
        meta.get("dataset_split") or meta.get("split") or data.get("dataset_split") or data.get("split")
    )

    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return []

    samples = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("was_cancelled"):
            continue
        mask = _extract_brush_mask_result(annotation.get("result", []))
        if mask is None:
            continue
        samples.append(
            {
                "image": image_ref,
                "image_path": image_path,
                "task_id": task.get("id"),
                "project_id": _normalize_project_id(task.get("project")),
                "annotation_id": annotation.get("id"),
                "updated_at": _normalize_text(annotation.get("updated_at") or annotation.get("created_at")),
                "dataset_split": dataset_split,
                "source": "label_studio_export",
                **mask,
            }
        )
    return samples


def _read_image_segmentation_samples_from_export(path):
    data = _read_json(path)
    if not isinstance(data, list):
        return []

    samples = []
    for task in data:
        samples.extend(_parse_image_segmentation_export_task(task))
    return samples


def _dedupe_image_segmentation_samples(samples):
    deduped = {}
    order = []
    for sample in samples:
        dedupe_key = (
            sample.get("image_path", ""),
            sample.get("label", ""),
            sample.get("dataset_split") or "train",
        )
        if dedupe_key not in deduped:
            order.append(dedupe_key)
        deduped[dedupe_key] = sample
    return [deduped[key] for key in order]


def _build_image_segmentation_dataset(payload):
    request_payload = payload if isinstance(payload, dict) else {}
    dataset_path = _resolve_dataset_path(request_payload)

    samples = []
    samples.extend(_normalize_manual_image_segmentation_samples(request_payload.get("samples")))

    nested_payload = (
        request_payload.get("payload")
        if isinstance(request_payload.get("payload"), dict)
        else None
    )
    if nested_payload:
        samples.extend(_normalize_manual_image_segmentation_samples(nested_payload.get("samples")))

    if dataset_path:
        samples.extend(_read_image_segmentation_samples_from_export(dataset_path))

    candidate_samples, candidate_stats = _read_image_segmentation_training_candidate_samples()
    samples.extend(candidate_samples)
    samples = _dedupe_image_segmentation_samples(samples)

    context = {
        "dataset_path": dataset_path,
        "local_files_root": IMAGE_LOCAL_FILES_ROOT,
        "candidates_path": IMAGE_SEG_TRAINING_CANDIDATES_PATH,
        "candidate_stats": candidate_stats,
    }
    return samples, context


def _build_text_classification_dataset(payload, webhook_sample):
    request_payload = payload if isinstance(payload, dict) else {}
    dataset_path = _resolve_dataset_path(request_payload)

    samples = []
    manual_samples = _normalize_manual_samples(request_payload.get("samples"))
    samples.extend(manual_samples)

    nested_payload = (
        request_payload.get("payload")
        if isinstance(request_payload.get("payload"), dict)
        else None
    )
    if nested_payload:
        nested_manual = _normalize_manual_samples(nested_payload.get("samples"))
        samples.extend(nested_manual)

    if dataset_path:
        export_samples = _read_samples_from_export(dataset_path)
        samples.extend(export_samples)

    if webhook_sample:
        samples.append(
            {
                "text": webhook_sample["text"],
                "label": webhook_sample["label"],
                "task_id": webhook_sample.get("task_id"),
                "annotation_id": webhook_sample.get("annotation_id"),
            }
        )

    event_samples = _load_dataset_event_samples()
    samples.extend(event_samples)
    samples = _dedupe_samples(samples)

    context = {
        "dataset_path": dataset_path,
    }

    return samples, context


def _dataset_quality_report(samples):
    labels = [sample["label"] for sample in samples]
    label_counts = Counter(labels)
    total = len(samples)
    unique_texts = len({sample["text"].lower() for sample in samples})

    distribution = {label: label_counts[label] for label in sorted(label_counts.keys())}
    class_count = len(distribution)
    min_count = min(distribution.values()) if distribution else 0
    max_count = max(distribution.values()) if distribution else 0
    imbalance_ratio = _round4(max_count / min_count) if min_count else None

    token_lengths = [len(sample["text"].split()) for sample in samples]
    avg_tokens = _round4(sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0

    errors = []
    if total < TEXT_CLS_MIN_TOTAL_SAMPLES:
        errors.append(
            f"need at least {TEXT_CLS_MIN_TOTAL_SAMPLES} samples for text classifier training, got {total}"
        )

    if class_count < 2:
        errors.append("need at least 2 distinct labels for text classifier training")

    underrepresented = {
        label: count
        for label, count in distribution.items()
        if count < TEXT_CLS_MIN_PER_CLASS_SAMPLES
    }
    if underrepresented:
        details = ", ".join(f"{label}={count}" for label, count in underrepresented.items())
        errors.append(
            f"each class needs at least {TEXT_CLS_MIN_PER_CLASS_SAMPLES} samples; below minimum: {details}"
        )

    if imbalance_ratio is not None and imbalance_ratio > TEXT_CLS_MAX_IMBALANCE_RATIO:
        errors.append(
            "label distribution too imbalanced: "
            f"max/min ratio {imbalance_ratio} exceeds allowed {TEXT_CLS_MAX_IMBALANCE_RATIO}"
        )

    return {
        "total_samples": total,
        "unique_text_samples": unique_texts,
        "class_count": class_count,
        "label_distribution": distribution,
        "min_per_class": min_count,
        "max_per_class": max_count,
        "imbalance_ratio": imbalance_ratio,
        "text_length": {
            "avg_tokens": avg_tokens,
            "median_tokens": median_tokens,
            "min_tokens": min(token_lengths) if token_lengths else 0,
            "max_tokens": max(token_lengths) if token_lengths else 0,
        },
        "validation": {
            "passed": len(errors) == 0,
            "errors": errors,
            "rules": {
                "min_total_samples": TEXT_CLS_MIN_TOTAL_SAMPLES,
                "min_per_class_samples": TEXT_CLS_MIN_PER_CLASS_SAMPLES,
                "max_imbalance_ratio": TEXT_CLS_MAX_IMBALANCE_RATIO,
            },
        },
    }


def _image_dataset_quality_report(samples):
    labels = [sample["label"] for sample in samples]
    label_counts = Counter(labels)
    total = len(samples)
    unique_images = len({sample["image_path"] for sample in samples})

    distribution = {label: label_counts[label] for label in sorted(label_counts.keys())}
    class_count = len(distribution)
    min_count = min(distribution.values()) if distribution else 0
    max_count = max(distribution.values()) if distribution else 0
    imbalance_ratio = _round4(max_count / min_count) if min_count else None

    errors = []
    if total < IMAGE_CLS_MIN_TOTAL_SAMPLES:
        errors.append(
            f"need at least {IMAGE_CLS_MIN_TOTAL_SAMPLES} samples for image classifier training, got {total}"
        )
    if class_count < 2:
        errors.append("need at least 2 distinct labels for image classifier training")

    underrepresented = {
        label: count
        for label, count in distribution.items()
        if count < IMAGE_CLS_MIN_PER_CLASS_SAMPLES
    }
    if underrepresented:
        details = ", ".join(f"{label}={count}" for label, count in underrepresented.items())
        errors.append(
            f"each class needs at least {IMAGE_CLS_MIN_PER_CLASS_SAMPLES} samples; below minimum: {details}"
        )
    if imbalance_ratio is not None and imbalance_ratio > IMAGE_CLS_MAX_IMBALANCE_RATIO:
        errors.append(
            "label distribution too imbalanced: "
            f"max/min ratio {imbalance_ratio} exceeds allowed {IMAGE_CLS_MAX_IMBALANCE_RATIO}"
        )

    return {
        "total_samples": total,
        "unique_image_samples": unique_images,
        "class_count": class_count,
        "label_distribution": distribution,
        "min_per_class": min_count,
        "max_per_class": max_count,
        "imbalance_ratio": imbalance_ratio,
        "validation": {
            "passed": len(errors) == 0,
            "errors": errors,
            "rules": {
                "min_total_samples": IMAGE_CLS_MIN_TOTAL_SAMPLES,
                "min_per_class_samples": IMAGE_CLS_MIN_PER_CLASS_SAMPLES,
                "max_imbalance_ratio": IMAGE_CLS_MAX_IMBALANCE_RATIO,
            },
        },
    }


def _image_segmentation_dataset_quality_report(samples):
    labels = [sample["label"] for sample in samples]
    label_counts = Counter(labels)
    total = len(samples)
    unique_images = len({sample["image_path"] for sample in samples})
    distribution = {label: label_counts[label] for label in sorted(label_counts.keys())}
    dimensions = [
        (sample.get("original_width"), sample.get("original_height"))
        for sample in samples
        if sample.get("original_width") and sample.get("original_height")
    ]
    unique_dimensions = sorted({f"{width}x{height}" for width, height in dimensions})

    errors = []
    if total < 1:
        errors.append("need at least 1 sample for image segmentation training")
    unsupported = sorted(label for label in distribution if label != "Object")
    if unsupported:
        errors.append(f"unsupported segmentation labels: {unsupported}; expected Object")
    invalid_rle = [
        idx + 1
        for idx, sample in enumerate(samples)
        if not isinstance(sample.get("rle"), list) or not sample.get("rle")
    ]
    if invalid_rle:
        errors.append(f"segmentation samples missing non-empty rle: rows {invalid_rle}")
    invalid_dimensions = [
        idx + 1
        for idx, sample in enumerate(samples)
        if _positive_int(sample.get("original_width")) is None
        or _positive_int(sample.get("original_height")) is None
    ]
    if invalid_dimensions:
        errors.append(
            "segmentation samples missing positive original dimensions: "
            f"rows {invalid_dimensions}"
        )

    return {
        "total_samples": total,
        "unique_image_samples": unique_images,
        "class_count": len(distribution),
        "label_distribution": distribution,
        "mask_dimensions": {
            "unique": unique_dimensions,
            "count": len(unique_dimensions),
            "min_width": min((width for width, _ in dimensions), default=0),
            "max_width": max((width for width, _ in dimensions), default=0),
            "min_height": min((height for _, height in dimensions), default=0),
            "max_height": max((height for _, height in dimensions), default=0),
        },
        "validation": {
            "passed": len(errors) == 0,
            "errors": errors,
            "rules": {
                "min_total_samples": 1,
                "allowed_labels": ["Object"],
            },
        },
    }


def _split_dataset_indices(labels, eval_split_ratio, min_eval_samples):
    class_count = len(set(labels))
    total = len(labels)
    if total < 3 or class_count < 2:
        return list(range(total)), [], {"strategy": "train_only", "reason": "insufficient_data"}

    tentative = int(round(total * eval_split_ratio))
    test_size = max(min_eval_samples, tentative, class_count)

    min_train_size = class_count
    max_test_size = total - min_train_size
    if test_size > max_test_size:
        test_size = max_test_size

    if test_size <= 0 or (total - test_size) <= 0:
        return list(range(total)), [], {"strategy": "train_only", "reason": "invalid_split_size"}

    label_counts = Counter(labels)
    can_stratify = (
        min(label_counts.values()) >= 2
        and test_size >= class_count
        and (total - test_size) >= class_count
    )

    indices = list(range(total))
    try:
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=labels if can_stratify else None,
        )
        strategy = "stratified" if can_stratify else "random"
    except ValueError:
        train_idx, eval_idx = train_test_split(indices, test_size=test_size, random_state=42)
        strategy = "random_fallback"

    train_labels = [labels[i] for i in train_idx]
    eval_labels = [labels[i] for i in eval_idx]
    split_info = {
        "strategy": strategy,
        "eval_split_ratio": eval_split_ratio,
        "eval_size_target": test_size,
        "train_size": len(train_idx),
        "eval_size": len(eval_idx),
        "train_label_distribution": {
            label: count for label, count in sorted(Counter(train_labels).items())
        },
        "eval_label_distribution": {
            label: count for label, count in sorted(Counter(eval_labels).items())
        },
    }
    return train_idx, eval_idx, split_info


def _split_dataset(samples, label_counts):
    texts = [sample["text"] for sample in samples]
    labels = [sample["label"] for sample in samples]
    _ = label_counts
    train_idx, eval_idx, split_info = _split_dataset_indices(
        labels,
        TEXT_CLS_EVAL_SPLIT_RATIO,
        TEXT_CLS_MIN_EVAL_SAMPLES,
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    eval_texts = [texts[i] for i in eval_idx]
    eval_labels = [labels[i] for i in eval_idx]

    return train_texts, train_labels, eval_texts, eval_labels, split_info


def _confidence_summary(max_probabilities, threshold):
    if not max_probabilities:
        return None

    values = sorted(float(value) for value in max_probabilities)
    middle = len(values) // 2
    if len(values) % 2 == 0:
        median = (values[middle - 1] + values[middle]) / 2.0
    else:
        median = values[middle]

    uncertain_count = sum(1 for value in values if value < threshold)

    return {
        "mean": _round4(sum(values) / len(values)),
        "median": _round4(median),
        "min": _round4(values[0]),
        "max": _round4(values[-1]),
        "uncertain_threshold": threshold,
        "uncertain_count": uncertain_count,
        "uncertain_rate": _round4(uncertain_count / len(values)),
    }


def _confusion_matrix_2x2_product_other(y_true, y_pred):
    summary = {
        "Product->Product": 0,
        "Product->Other": 0,
        "Other->Product": 0,
        "Other->Other": 0,
        "total": len(y_true),
    }
    for true_label, pred_label in zip(y_true, y_pred):
        key = f"{true_label}->{pred_label}"
        if key in summary:
            summary[key] += 1
    return summary


def _build_image_eval_predictions(samples, predicted_labels, confidences):
    rows = []
    for sample, predicted_label, confidence in zip(samples, predicted_labels, confidences):
        true_label = sample.get("label")
        image_path = sample.get("image_path")
        row = {
            "image": sample.get("image"),
            "image_path": image_path,
            "filename": os.path.basename(image_path) if image_path else None,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": _round4(confidence),
            "correct": bool(predicted_label == true_label),
        }
        rows.append(row)
    return rows


def _load_image_eval_manifest(path):
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid image eval manifest: expected object at {path}")

    items = data.get("images")
    if not isinstance(items, list) or not items:
        raise ValueError(f"invalid image eval manifest: `images` must be a non-empty list at {path}")

    eval_set_id = _normalize_text(data.get("eval_set_id"))
    entries = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"invalid image eval manifest: each image entry must be an object at {path}")
        raw_name = item.get("filename")
        if raw_name is None:
            raw_name = item.get("image")
        if raw_name is None:
            raise ValueError(f"invalid image eval manifest: missing filename in entry at {path}")
        file_name = os.path.basename(str(raw_name).strip())
        if not file_name:
            raise ValueError(f"invalid image eval manifest: empty filename in entry at {path}")
        label = _normalize_label(item.get("label"))
        if not label:
            raise ValueError(f"invalid image eval manifest: missing label for {file_name} at {path}")
        if file_name in entries:
            raise ValueError(f"invalid image eval manifest: duplicate filename {file_name} at {path}")
        entries[file_name] = label

    return {
        "path": path,
        "eval_set_id": eval_set_id,
        "images": entries,
    }


def _validate_image_eval_manifest(samples, train_idx, eval_idx, split_info):
    manifest_info = {
        "path": IMAGE_CLS_EVAL_MANIFEST_PATH,
        "present": os.path.exists(IMAGE_CLS_EVAL_MANIFEST_PATH),
        "checked": False,
        "eval_set_id": split_info.get("eval_set_id"),
        "expected_filenames": [],
        "actual_filenames": [],
    }

    if split_info.get("strategy") != "fixed_eval_set":
        return manifest_info

    eval_samples = [samples[idx] for idx in eval_idx]
    actual = {}
    for sample in eval_samples:
        image_path = sample.get("image_path")
        if not image_path:
            continue
        filename = os.path.basename(image_path)
        actual[filename] = sample.get("label")
    manifest_info["actual_filenames"] = sorted(actual.keys())

    if not manifest_info["present"]:
        return manifest_info

    manifest = _load_image_eval_manifest(IMAGE_CLS_EVAL_MANIFEST_PATH)
    manifest_info["checked"] = True
    manifest_info["expected_filenames"] = sorted(manifest["images"].keys())

    manifest_eval_set = manifest.get("eval_set_id")
    if manifest_eval_set:
        manifest_info["eval_set_id"] = manifest_eval_set
        split_eval_set = split_info.get("eval_set_id")
        if split_eval_set and split_eval_set != manifest_eval_set:
            raise ValueError(
                "image fixed eval set id mismatch: "
                f"dataset={split_eval_set}, manifest={manifest_eval_set}"
            )

    expected_files = set(manifest["images"].keys())
    actual_files = set(actual.keys())
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    if missing or extra:
        raise ValueError(
            "image fixed eval set mismatch with manifest: "
            f"missing={missing}, extra={extra}"
        )

    label_mismatch = []
    for filename in sorted(expected_files):
        expected_label = manifest["images"][filename]
        actual_label = actual.get(filename)
        if actual_label != expected_label:
            label_mismatch.append(
                {
                    "filename": filename,
                    "expected_label": expected_label,
                    "actual_label": actual_label,
                }
            )
    if label_mismatch:
        raise ValueError(f"image fixed eval labels mismatch with manifest: {label_mismatch}")

    train_files = set()
    for idx in train_idx:
        sample = samples[idx]
        image_path = sample.get("image_path")
        if image_path:
            train_files.add(os.path.basename(image_path))
    overlap = sorted(train_files.intersection(expected_files))
    if overlap:
        raise ValueError(f"image fixed eval leakage detected in train split: {overlap}")

    return manifest_info


def _top_terms_by_class(classifier, vectorizer, top_n=12):
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_
    classes = [str(label) for label in classifier.classes_]

    top_terms = {}
    if coef.shape[0] == 1 and len(classes) == 2:
        weights = coef[0]
        pos_idx = weights.argsort()[-top_n:][::-1]
        neg_idx = weights.argsort()[:top_n]

        top_terms[classes[1]] = [
            {"term": feature_names[i], "weight": _round4(weights[i])} for i in pos_idx
        ]
        top_terms[classes[0]] = [
            {"term": feature_names[i], "weight": _round4(weights[i])} for i in neg_idx
        ]
        return top_terms

    for row_idx, label in enumerate(classes):
        weights = coef[row_idx]
        top_idx = weights.argsort()[-top_n:][::-1]
        top_terms[label] = [
            {"term": feature_names[i], "weight": _round4(weights[i])} for i in top_idx
        ]
    return top_terms


def _predict_with_probabilities(classifier, vectorizer, texts):
    matrix = vectorizer.transform(texts)
    probabilities = classifier.predict_proba(matrix)
    labels = [str(item) for item in classifier.classes_]

    predictions = []
    max_probs = []
    for row in probabilities:
        entries = [{"label": labels[idx], "probability": _round4(row[idx])} for idx in range(len(labels))]
        entries.sort(key=lambda item: item["probability"], reverse=True)
        predictions.append(entries[0]["label"])
        max_probs.append(entries[0]["probability"])

    return predictions, max_probs, probabilities, labels


def _train_real_text_classifier(samples, training_run, dataset_context=None):
    context = dataset_context if isinstance(dataset_context, dict) else {}
    quality = _dataset_quality_report(samples)
    if not quality["validation"]["passed"]:
        raise ValueError("; ".join(quality["validation"]["errors"]))

    labels = [sample["label"] for sample in samples]
    label_set = sorted(set(labels))

    train_texts, train_labels, eval_texts, eval_labels, split_info = _split_dataset(
        samples,
        Counter(labels),
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
    )
    train_matrix = vectorizer.fit_transform(train_texts)

    classifier = LogisticRegression(
        max_iter=1500,
        random_state=42,
        class_weight="balanced",
        solver="liblinear",
        C=3.0,
    )
    classifier.fit(train_matrix, train_labels)

    train_pred, train_max_probs, _, _ = _predict_with_probabilities(classifier, vectorizer, train_texts)

    if eval_texts:
        eval_pred, eval_max_probs, _, class_order = _predict_with_probabilities(classifier, vectorizer, eval_texts)
        y_true = eval_labels
        y_pred = eval_pred
        confidence_summary = _confidence_summary(eval_max_probs, TEXT_CLS_UNCERTAIN_THRESHOLD)
        evaluation_scope = "eval"
    else:
        _, eval_max_probs, _, class_order = _predict_with_probabilities(classifier, vectorizer, train_texts)
        y_true = train_labels
        y_pred = train_pred
        confidence_summary = _confidence_summary(eval_max_probs, TEXT_CLS_UNCERTAIN_THRESHOLD)
        evaluation_scope = "train"

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_label = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_order)

    model_version = f"text-cls-v{training_run:04d}"
    run_id = f"train-{training_run:04d}"
    trained_at = _utc_now_iso()

    os.makedirs(TEXT_MODEL_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(classifier, _TEXT_CLASSIFIER_PATH)
    joblib.dump(vectorizer, _TEXT_VECTORIZER_PATH)

    versioned_classifier = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, f"classifier_{model_version}.joblib")
    versioned_vectorizer = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, f"vectorizer_{model_version}.joblib")
    joblib.dump(classifier, versioned_classifier)
    joblib.dump(vectorizer, versioned_vectorizer)

    with open(_TEXT_LAST_DATASET_PATH, "w", encoding="utf-8") as dataset_file:
        for sample in samples:
            dataset_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "model_version": model_version,
        "training_run": training_run,
        "training_run_id": run_id,
        "trained_at": trained_at,
        "task_type": "text_classification",
        "labels": label_set,
        "dataset": {
            "train_size": len(train_texts),
            "eval_size": len(eval_texts),
            "total_size": len(samples),
            "source": "label_studio_annotations",
            "source_path": context.get("dataset_path"),
            "snapshot_path": _TEXT_LAST_DATASET_PATH,
            "quality": quality,
            "split": split_info,
        },
        "metrics": {
            "evaluation_scope": evaluation_scope,
            "accuracy": _round4(accuracy),
            "macro_f1": _round4(macro_f1),
            "per_label": _round_nested(per_label),
            "confusion_matrix": {
                "labels": class_order,
                "matrix": matrix.tolist(),
            },
            "confidence": {
                "evaluation": confidence_summary,
                "train": _confidence_summary(train_max_probs, TEXT_CLS_UNCERTAIN_THRESHOLD),
            },
        },
        "model_details": {
            "vectorizer": {
                "lowercase": True,
                "strip_accents": "unicode",
                "ngram_range": [1, 2],
                "min_df": 1,
                "sublinear_tf": True,
                "stop_words": "english",
                "vocabulary_size": int(train_matrix.shape[1]),
            },
            "classifier": {
                "name": "LogisticRegression",
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 1500,
                "C": 3.0,
            },
            "top_terms_by_class": _top_terms_by_class(classifier, vectorizer),
        },
        "artifacts": {
            "classifier_path": _TEXT_CLASSIFIER_PATH,
            "vectorizer_path": _TEXT_VECTORIZER_PATH,
            "versioned_classifier_path": versioned_classifier,
            "versioned_vectorizer_path": versioned_vectorizer,
        },
        "framework": {
            "library": "scikit-learn",
            "estimator": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
        },
    }

    with open(_TEXT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _extract_image_feature_vector(image_path):
    if Image is None:
        raise ValueError("Pillow is required for image classifier training")
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    luminance = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])

    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
        hist_features.extend(hist.tolist())

    aspect_ratio = (float(width) / float(height)) if height else 1.0
    features = [
        float(width) / 1000.0,
        float(height) / 1000.0,
        aspect_ratio,
        float(mean_rgb[0]),
        float(mean_rgb[1]),
        float(mean_rgb[2]),
        float(std_rgb[0]),
        float(std_rgb[1]),
        float(std_rgb[2]),
        float(luminance.mean()),
        float(luminance.std()),
    ]
    features.extend(float(item) for item in hist_features)
    return np.asarray(features, dtype=np.float32)


def _image_matrix_and_labels(samples):
    features = []
    labels = []
    image_paths = []

    for sample in samples:
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image sample missing image_path")
        if not os.path.exists(image_path):
            raise ValueError(f"image file not found: {image_path}")
        try:
            feature_vector = _extract_image_feature_vector(image_path)
        except Exception as exc:
            raise ValueError(f"failed to read image features for {image_path}: {exc}") from exc
        features.append(feature_vector)
        labels.append(sample["label"])
        image_paths.append(image_path)

    return np.vstack(features), labels, image_paths


def _predict_probabilities_from_matrix(classifier, matrix):
    probabilities = classifier.predict_proba(matrix)
    labels = [str(item) for item in classifier.classes_]

    predictions = []
    max_probs = []
    for row in probabilities:
        entries = [{"label": labels[idx], "probability": _round4(row[idx])} for idx in range(len(labels))]
        entries.sort(key=lambda item: item["probability"], reverse=True)
        predictions.append(entries[0]["label"])
        max_probs.append(entries[0]["probability"])

    return predictions, max_probs, labels


def _train_real_image_classifier(samples, training_run, dataset_context=None):
    context = dataset_context if isinstance(dataset_context, dict) else {}
    quality = _image_dataset_quality_report(samples)
    if not quality["validation"]["passed"]:
        raise ValueError("; ".join(quality["validation"]["errors"]))

    labels = [sample["label"] for sample in samples]
    label_set = sorted(set(labels))
    train_idx, eval_idx, split_info = _split_image_dataset(samples)

    train_samples = [samples[idx] for idx in train_idx]
    eval_samples = [samples[idx] for idx in eval_idx]
    eval_manifest_info = _validate_image_eval_manifest(samples, train_idx, eval_idx, split_info)
    train_matrix, train_labels, _ = _image_matrix_and_labels(train_samples)

    eval_classifier = KNeighborsClassifier(
        n_neighbors=1,
        weights="distance",
        metric="euclidean",
    )
    eval_classifier.fit(train_matrix, train_labels)

    train_pred, train_max_probs, _ = _predict_probabilities_from_matrix(eval_classifier, train_matrix)

    if eval_samples:
        eval_matrix, eval_labels, _ = _image_matrix_and_labels(eval_samples)
        eval_pred, eval_max_probs, class_order = _predict_probabilities_from_matrix(eval_classifier, eval_matrix)
        y_true = eval_labels
        y_pred = eval_pred
        eval_confidences = eval_max_probs
        eval_prediction_rows = _build_image_eval_predictions(eval_samples, y_pred, eval_confidences)
        confidence_summary = _confidence_summary(eval_max_probs, IMAGE_CLS_UNCERTAIN_THRESHOLD)
        evaluation_scope = "eval"
    else:
        _, eval_max_probs, class_order = _predict_probabilities_from_matrix(eval_classifier, train_matrix)
        y_true = train_labels
        y_pred = train_pred
        eval_confidences = eval_max_probs
        eval_prediction_rows = _build_image_eval_predictions(train_samples, y_pred, eval_confidences)
        confidence_summary = _confidence_summary(eval_max_probs, IMAGE_CLS_UNCERTAIN_THRESHOLD)
        evaluation_scope = "train"

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_label = classification_report(
        y_true,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_order)

    model_version = f"image-cls-v{training_run:04d}"
    run_id = f"train-{training_run:04d}"
    trained_at = _utc_now_iso()
    train_filenames = sorted(
        {
            os.path.basename(sample.get("image_path"))
            for sample in train_samples
            if sample.get("image_path")
        }
    )
    eval_filenames = sorted(
        {
            os.path.basename(sample.get("image_path"))
            for sample in eval_samples
            if sample.get("image_path")
        }
    )

    serving_classifier = KNeighborsClassifier(
        n_neighbors=1,
        weights="distance",
        metric="euclidean",
    )
    # Keep eval samples independent for stable cross-version metrics.
    serving_classifier.fit(train_matrix, train_labels)

    os.makedirs(IMAGE_MODEL_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(serving_classifier, _IMAGE_CLASSIFIER_PATH)
    versioned_classifier = os.path.join(
        IMAGE_MODEL_ARTIFACTS_DIR,
        f"classifier_{model_version}.joblib",
    )
    joblib.dump(serving_classifier, versioned_classifier)

    with open(_IMAGE_LAST_DATASET_PATH, "w", encoding="utf-8") as dataset_file:
        for sample in samples:
            dataset_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "model_version": model_version,
        "training_run": training_run,
        "training_run_id": run_id,
        "trained_at": trained_at,
        "task_type": "image_classification",
        "labels": label_set,
        "dataset": {
            "train_size": len(train_samples),
            "eval_size": len(eval_samples),
            "total_size": len(samples),
            "source": "label_studio_annotations",
            "source_path": context.get("dataset_path"),
            "local_files_root": context.get("local_files_root"),
            "snapshot_path": _IMAGE_LAST_DATASET_PATH,
            "quality": quality,
            "split": split_info,
            "candidates": {
                "path": context.get("candidates_path"),
                "stats": context.get("candidate_stats"),
                "used_after_eval_filter": context.get("candidate_used_after_eval_filter"),
                "eval_leakage_skipped": context.get("candidate_eval_leakage_skipped"),
            },
            "fixed_eval": {
                "enabled": split_info.get("strategy") == "fixed_eval_set",
                "eval_set_id": split_info.get("eval_set_id"),
                "source_field": split_info.get("source_field"),
                "manifest_path": IMAGE_CLS_EVAL_MANIFEST_PATH,
                "manifest_present": eval_manifest_info.get("present"),
                "manifest_checked": eval_manifest_info.get("checked"),
                "train_filenames": train_filenames,
                "eval_filenames": eval_filenames,
                "manifest_eval_filenames": eval_manifest_info.get("expected_filenames"),
            },
        },
        "metrics": {
            "evaluation_scope": evaluation_scope,
            "accuracy": _round4(accuracy),
            "macro_f1": _round4(macro_f1),
            "per_label": _round_nested(per_label),
            "confusion_matrix": {
                "labels": class_order,
                "matrix": matrix.tolist(),
            },
            "confusion_matrix_product_other": _confusion_matrix_2x2_product_other(y_true, y_pred),
            "eval_predictions": eval_prediction_rows,
            "eval_misclassified": [row for row in eval_prediction_rows if not row.get("correct")],
            "confidence": {
                "evaluation": confidence_summary,
                "train": _confidence_summary(train_max_probs, IMAGE_CLS_UNCERTAIN_THRESHOLD),
            },
        },
        "model_details": {
            "feature_extractor": {
                "name": "color_statistics_histogram",
                "histogram_bins_per_channel": 8,
                "features": [
                    "width",
                    "height",
                    "aspect_ratio",
                    "mean_rgb",
                    "std_rgb",
                    "luminance_mean_std",
                    "rgb_histogram_density",
                ],
            },
            "classifier": {
                "name": "KNeighborsClassifier",
                "n_neighbors": 1,
                "weights": "distance",
                "metric": "euclidean",
                "serving_fit": "train_split_only",
            },
        },
        "artifacts": {
            "classifier_path": _IMAGE_CLASSIFIER_PATH,
            "versioned_classifier_path": versioned_classifier,
        },
        "framework": {
            "library": "scikit-learn",
            "estimator": "KNeighborsClassifier",
            "feature_extractor": "numpy+Pillow color statistics",
        },
    }

    with open(_IMAGE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _train_placeholder_image_segmentation(samples, training_run, dataset_context=None):
    context = dataset_context if isinstance(dataset_context, dict) else {}
    quality = _image_segmentation_dataset_quality_report(samples)
    if not quality["validation"]["passed"]:
        raise ValueError("; ".join(quality["validation"]["errors"]))

    model_version = f"image-seg-v{training_run:04d}"
    run_id = f"train-{training_run:04d}"
    trained_at = _utc_now_iso()

    os.makedirs(IMAGE_SEG_MODEL_ARTIFACTS_DIR, exist_ok=True)
    with open(_IMAGE_SEG_LAST_DATASET_PATH, "w", encoding="utf-8") as dataset_file:
        for sample in samples:
            dataset_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    artifact = {
        "model_version": model_version,
        "task_type": "image_segmentation",
        "labels": ["Object"],
        "strategy": "placeholder_center_mask",
        "rle_format": "label_studio_brush",
        "trained_at": trained_at,
    }
    with open(_IMAGE_SEG_ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    metadata = {
        "model_version": model_version,
        "training_run": training_run,
        "training_run_id": run_id,
        "trained_at": trained_at,
        "task_type": "image_segmentation",
        "labels": ["Object"],
        "dataset": {
            "train_size": len(samples),
            "eval_size": 0,
            "total_size": len(samples),
            "source": "label_studio_brush_masks",
            "source_path": context.get("dataset_path"),
            "local_files_root": context.get("local_files_root"),
            "snapshot_path": _IMAGE_SEG_LAST_DATASET_PATH,
            "quality": quality,
            "candidates": {
                "path": context.get("candidates_path"),
                "stats": context.get("candidate_stats"),
            },
        },
        "metrics": {
            "evaluation_scope": "not_applicable_placeholder",
            "mask_count": len(samples),
        },
        "model_details": {
            "name": "placeholder_center_mask",
            "rle_format": "label_studio_brush",
            "replaceable_with": [
                "SAM",
                "YOLO-seg",
                "Detectron2",
                "custom",
            ],
        },
        "artifacts": {
            "artifact_path": _IMAGE_SEG_ARTIFACT_PATH,
            "metadata_path": _IMAGE_SEG_METADATA_PATH,
            "placeholder_model_path": _IMAGE_SEG_ARTIFACT_PATH,
        },
        "framework": {
            "library": "none",
            "estimator": "placeholder",
        },
    }

    with open(_IMAGE_SEG_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _run_deterministic_fallback(payload, trigger, train_error=None):
    current = _load_state()
    previous_run = int(current.get("training_run", 0))
    training_run = previous_run + 1
    model_version = f"demo-rule-v{training_run + 1}"

    event_summary = _summarize_training_event(payload, trigger)
    train_record = {
        "run_id": f"train-{training_run:04d}",
        "training_run": training_run,
        "model_version": model_version,
        "received_at": _utc_now_iso(),
        "mode": "deterministic-fallback",
        "event": event_summary,
    }

    next_state = {
        "model_version": model_version,
        "training_run": training_run,
        "updated_at": train_record["received_at"],
        "source": trigger,
        "behavior": _behavior_for_run(training_run),
        "text_classifier": current.get("text_classifier", _default_state()["text_classifier"]),
        "image_classifier": current.get("image_classifier", _default_state()["image_classifier"]),
        "image_segmentation": current.get("image_segmentation", _default_state()["image_segmentation"]),
        "last_training_event": train_record,
    }

    _save_state(next_state)
    _append_event_log(TRAINING_EVENT_LOG_PATH, train_record)

    return {
        "next_state": next_state,
        "train_record": train_record,
        "mode": "deterministic-fallback",
        "metadata": None,
        "train_error": train_error,
    }


def _run_text_training(payload, trigger):
    with _STATE_LOCK:
        request_payload = payload if isinstance(payload, dict) else {}
        webhook_sample = _maybe_record_webhook_sample(request_payload)
        dataset_samples, dataset_context = _build_text_classification_dataset(
            request_payload,
            webhook_sample,
        )

        current = _load_state()
        previous_run = int(current.get("training_run", 0))
        training_run = previous_run + 1

        try:
            metadata = _train_real_text_classifier(dataset_samples, training_run, dataset_context)
            training_mode = "real-text-classifier"
            model_version = metadata["model_version"]
        except ValueError as exc:
            return _run_deterministic_fallback(request_payload, trigger, train_error=str(exc))

        event_summary = _summarize_training_event(request_payload, trigger)
        train_record = {
            "run_id": metadata["training_run_id"],
            "training_run": training_run,
            "model_version": model_version,
            "received_at": metadata["trained_at"],
            "mode": training_mode,
            "dataset_size": metadata["dataset"]["total_size"],
            "dataset_path": metadata["dataset"].get("source_path"),
            "metrics": {
                "accuracy": metadata["metrics"]["accuracy"],
                "macro_f1": metadata["metrics"]["macro_f1"],
                "evaluation_scope": metadata["metrics"]["evaluation_scope"],
            },
            "event": event_summary,
        }

        next_state = {
            "model_version": model_version,
            "training_run": training_run,
            "updated_at": metadata["trained_at"],
            "source": trigger,
            "behavior": _behavior_for_run(training_run),
            "text_classifier": {
                "active": True,
                "model_version": model_version,
                "trained_at": metadata["trained_at"],
                "metadata_path": _TEXT_METADATA_PATH,
                "training_run_id": metadata["training_run_id"],
                "metrics": {
                    "accuracy": metadata["metrics"]["accuracy"],
                    "macro_f1": metadata["metrics"]["macro_f1"],
                },
                "dataset_summary": {
                    "total_size": metadata["dataset"]["total_size"],
                    "label_distribution": metadata["dataset"]["quality"]["label_distribution"],
                },
            },
            "image_classifier": current.get("image_classifier", _default_state()["image_classifier"]),
            "image_segmentation": current.get("image_segmentation", _default_state()["image_segmentation"]),
            "last_training_event": train_record,
        }

        _save_state(next_state)
        _append_event_log(TRAINING_EVENT_LOG_PATH, train_record)

        return {
            "next_state": next_state,
            "train_record": train_record,
            "mode": training_mode,
            "metadata": metadata,
            "train_error": None,
        }


def _run_image_training(payload, trigger):
    with _STATE_LOCK:
        request_payload = payload if isinstance(payload, dict) else {}
        warnings = []
        webhook_candidate = None
        if trigger == "label-studio-webhook":
            try:
                webhook_candidate, warning = _maybe_record_image_candidate_from_webhook(request_payload)
                if warning:
                    warnings.append(warning)
            except ValueError as exc:
                warnings.append(f"image webhook candidate ingest failed: {exc}")

        dataset_samples, dataset_context = _build_image_classification_dataset(request_payload)
        current = _load_state()
        previous_run = int(current.get("training_run", 0))
        training_run = previous_run + 1

        try:
            metadata = _train_real_image_classifier(dataset_samples, training_run, dataset_context)
        except ValueError as exc:
            return {
                "ok": False,
                "status_code": 422,
                "mode": "real-image-classifier",
                "task_type": "image_classification",
                "train_error": str(exc),
                "next_state": current,
                "train_record": None,
                "metadata": None,
                "warnings": warnings,
            }

        event_summary = _summarize_training_event(request_payload, trigger)
        train_record = {
            "run_id": metadata["training_run_id"],
            "training_run": training_run,
            "model_version": metadata["model_version"],
            "received_at": metadata["trained_at"],
            "mode": "real-image-classifier",
            "task_type": "image_classification",
            "dataset_size": metadata["dataset"]["total_size"],
            "dataset_path": metadata["dataset"].get("source_path"),
            "metrics": {
                "accuracy": metadata["metrics"]["accuracy"],
                "macro_f1": metadata["metrics"]["macro_f1"],
                "evaluation_scope": metadata["metrics"]["evaluation_scope"],
            },
            "warnings": warnings,
            "webhook_candidate": webhook_candidate,
            "event": event_summary,
        }

        next_state = {
            "model_version": metadata["model_version"],
            "training_run": training_run,
            "updated_at": metadata["trained_at"],
            "source": trigger,
            "behavior": _behavior_for_run(training_run),
            "text_classifier": current.get("text_classifier", _default_state()["text_classifier"]),
            "image_classifier": {
                "active": True,
                "model_version": metadata["model_version"],
                "trained_at": metadata["trained_at"],
                "metadata_path": _IMAGE_METADATA_PATH,
                "training_run_id": metadata["training_run_id"],
                "metrics": {
                    "accuracy": metadata["metrics"]["accuracy"],
                    "macro_f1": metadata["metrics"]["macro_f1"],
                },
                "dataset_summary": {
                    "total_size": metadata["dataset"]["total_size"],
                    "label_distribution": metadata["dataset"]["quality"]["label_distribution"],
                },
            },
            "image_segmentation": current.get("image_segmentation", _default_state()["image_segmentation"]),
            "last_training_event": train_record,
        }

        _save_state(next_state)
        _append_event_log(TRAINING_EVENT_LOG_PATH, train_record)
        return {
            "ok": True,
            "status_code": 202,
            "mode": "real-image-classifier",
            "task_type": "image_classification",
            "train_error": None,
            "next_state": next_state,
            "train_record": train_record,
            "metadata": metadata,
            "warnings": warnings,
        }


def _run_image_segmentation_training(payload, trigger):
    with _STATE_LOCK:
        request_payload = payload if isinstance(payload, dict) else {}
        warnings = []
        webhook_candidate = None
        if trigger == "label-studio-webhook":
            try:
                webhook_candidate, warning = _maybe_record_image_segmentation_candidate_from_webhook(
                    request_payload
                )
                if warning:
                    warnings.append(warning)
            except ValueError as exc:
                warnings.append(f"image segmentation webhook candidate ingest failed: {exc}")

        dataset_samples, dataset_context = _build_image_segmentation_dataset(request_payload)
        current = _load_state()
        previous_run = int(current.get("training_run", 0))
        training_run = previous_run + 1

        try:
            metadata = _train_placeholder_image_segmentation(
                dataset_samples,
                training_run,
                dataset_context,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "status_code": 422,
                "mode": "placeholder-image-segmentation",
                "task_type": "image_segmentation",
                "train_error": str(exc),
                "next_state": current,
                "train_record": None,
                "metadata": None,
                "warnings": warnings,
            }

        event_summary = _summarize_training_event(request_payload, trigger)
        train_record = {
            "run_id": metadata["training_run_id"],
            "training_run": training_run,
            "model_version": metadata["model_version"],
            "received_at": metadata["trained_at"],
            "mode": "placeholder-image-segmentation",
            "task_type": "image_segmentation",
            "dataset_size": metadata["dataset"]["total_size"],
            "dataset_path": metadata["dataset"].get("source_path"),
            "metrics": metadata["metrics"],
            "warnings": warnings,
            "webhook_candidate": webhook_candidate,
            "event": event_summary,
        }

        next_state = {
            "model_version": metadata["model_version"],
            "training_run": training_run,
            "updated_at": metadata["trained_at"],
            "source": trigger,
            "behavior": _behavior_for_run(training_run),
            "text_classifier": current.get("text_classifier", _default_state()["text_classifier"]),
            "image_classifier": current.get("image_classifier", _default_state()["image_classifier"]),
            "image_segmentation": {
                "active": True,
                "model_version": metadata["model_version"],
                "trained_at": metadata["trained_at"],
                "artifact_path": _IMAGE_SEG_ARTIFACT_PATH,
                "metadata_path": _IMAGE_SEG_METADATA_PATH,
                "training_run_id": metadata["training_run_id"],
                "dataset_summary": {
                    "total_size": metadata["dataset"]["total_size"],
                    "label_distribution": metadata["dataset"]["quality"]["label_distribution"],
                },
            },
            "last_training_event": train_record,
        }

        _save_state(next_state)
        _append_event_log(TRAINING_EVENT_LOG_PATH, train_record)
        return {
            "ok": True,
            "status_code": 202,
            "mode": "placeholder-image-segmentation",
            "task_type": "image_segmentation",
            "train_error": None,
            "next_state": next_state,
            "train_record": train_record,
            "metadata": metadata,
            "warnings": warnings,
        }


def _normalize_task_type(value):
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"text_classification", "image_classification", "image_segmentation"}:
        return normalized
    return None


def _resolve_task_type(payload, route_task_type=None):
    route_value = _normalize_task_type(route_task_type)
    if route_value:
        return route_value

    request_payload = payload if isinstance(payload, dict) else {}
    candidates = [
        request_payload.get("task_type"),
        request_payload.get("task"),
    ]
    nested = request_payload.get("payload") if isinstance(request_payload.get("payload"), dict) else {}
    candidates.extend(
        [
            nested.get("task_type"),
            nested.get("task"),
        ]
    )
    for candidate in candidates:
        normalized = _normalize_task_type(candidate)
        if normalized:
            return normalized

    task_obj = request_payload.get("task")
    if isinstance(task_obj, dict):
        data = task_obj.get("data")
        if isinstance(data, dict):
            if data.get("image") is not None:
                return "image_classification"
            if data.get("text") is not None:
                return "text_classification"

    annotation = request_payload.get("annotation")
    if isinstance(annotation, dict):
        ann_task = annotation.get("task")
        if isinstance(ann_task, dict):
            data = ann_task.get("data")
            if isinstance(data, dict):
                if data.get("image") is not None:
                    return "image_classification"
                if data.get("text") is not None:
                    return "text_classification"
    return "text_classification"


def _run_training(payload, trigger, route_task_type=None):
    task_type = _resolve_task_type(payload, route_task_type)
    if task_type == "image_segmentation":
        return _run_image_segmentation_training(payload, trigger)
    if task_type == "image_classification":
        return _run_image_training(payload, trigger)

    outcome = _run_text_training(payload, trigger)
    outcome["ok"] = True
    outcome["status_code"] = 202
    outcome["task_type"] = "text_classification"
    return outcome


def _train_route_task_type(route):
    if "image-segmentation" in route:
        return "image_segmentation"
    if "image-classification" in route:
        return "image_classification"
    if "text-classification" in route:
        return "text_classification"
    return None


def _read_text_classifier_metadata():
    data = _read_json(_TEXT_METADATA_PATH)
    if isinstance(data, dict):
        return data
    return None


def _read_image_classifier_metadata():
    data = _read_json(_IMAGE_METADATA_PATH)
    if isinstance(data, dict):
        return data
    return None


class TrainerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.rstrip("/") or "/"
        if route in ("/", "/health"):
            state = _load_state()
            text_metadata = _read_text_classifier_metadata()
            image_metadata = _read_image_classifier_metadata()
            _send_json(
                self,
                200,
                {
                    "status": "UP",
                    "service": SERVICE_NAME,
                    "phase": "phase-4-text-and-image-classifiers",
                    "model_version": state.get("model_version", "demo-rule-v1"),
                    "training_run": int(state.get("training_run", 0)),
                    "text_classifier": {
                        "active": bool(text_metadata),
                        "metadata": text_metadata,
                        "metadata_path": _TEXT_METADATA_PATH,
                    },
                    "image_classifier": {
                        "active": bool(image_metadata),
                        "metadata": image_metadata,
                        "metadata_path": _IMAGE_METADATA_PATH,
                    },
                },
            )
            return

        if route == "/model-state":
            _send_json(self, 200, _load_state())
            return

        if route in ("/models/text-classification", "/models/text-classification/current"):
            metadata = _read_text_classifier_metadata()
            if not metadata:
                _send_json(self, 404, {"detail": "no trained text classification model"})
                return
            _send_json(self, 200, metadata)
            return

        if route in ("/models/image-classification", "/models/image-classification/current"):
            metadata = _read_image_classifier_metadata()
            if not metadata:
                _send_json(self, 404, {"detail": "no trained image classification model"})
                return
            _send_json(self, 200, metadata)
            return

        _send_json(self, 404, {"detail": "not found"})

    def do_POST(self):
        route = self.path.rstrip("/") or "/"
        payload = _read_json_body(self)

        if route in (
            "/train",
            "/retrain",
            "/train/text-classification",
            "/retrain/text-classification",
            "/train/image-classification",
            "/retrain/image-classification",
            "/train/image-segmentation",
            "/retrain/image-segmentation",
        ):
            route_task_type = _train_route_task_type(route)
            outcome = _run_training(payload, trigger="manual-train", route_task_type=route_task_type)
            next_state = outcome["next_state"]
            train_record = outcome["train_record"]
            _send_json(
                self,
                int(outcome.get("status_code", 202)),
                {
                    "accepted": bool(outcome.get("ok", True)),
                    "message": "training event processed"
                    if outcome.get("ok", True)
                    else "training request rejected",
                    "task_type": outcome.get("task_type"),
                    "mode": outcome["mode"],
                    "model_version": next_state["model_version"],
                    "training_run": next_state["training_run"],
                    "run_id": train_record["run_id"] if train_record else None,
                    "metrics": (outcome["metadata"] or {}).get("metrics"),
                    "train_error": outcome.get("train_error"),
                    "warnings": outcome.get("warnings", []),
                },
            )
            return

        if route in ("/webhook", "/webhook/label-studio"):
            outcome = _run_training(payload, trigger="label-studio-webhook")
            next_state = outcome["next_state"]
            train_record = outcome["train_record"]
            _send_json(
                self,
                int(outcome.get("status_code", 202)),
                {
                    "accepted": bool(outcome.get("ok", True)),
                    "message": "webhook event processed"
                    if outcome.get("ok", True)
                    else "webhook training request rejected",
                    "action": payload.get("action"),
                    "task_type": outcome.get("task_type"),
                    "mode": outcome["mode"],
                    "model_version": next_state["model_version"],
                    "training_run": next_state["training_run"],
                    "run_id": train_record["run_id"] if train_record else None,
                    "metrics": (outcome["metadata"] or {}).get("metrics"),
                    "train_error": outcome.get("train_error"),
                    "warnings": outcome.get("warnings", []),
                },
            )
            return

        _send_json(self, 404, {"detail": "not found"})

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    _load_state()
    HTTPServer(("0.0.0.0", PORT), TrainerHandler).serve_forever()
