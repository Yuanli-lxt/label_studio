import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - handled by fallback path
    Image = None

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from segmentation_quality import evaluate_segmentation_quality
from segmentation_uncertainty import evaluate_prompt_stability, generate_bbox_prompt_variants

SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-backend")
PORT = int(os.getenv("PORT", "9090"))
MODEL_STATE_PATH = os.getenv("MODEL_STATE_PATH", "/model-state/current_model.json")
TEXT_MODEL_ARTIFACTS_DIR = os.getenv(
    "TEXT_MODEL_ARTIFACTS_DIR",
    os.getenv("MODEL_ARTIFACTS_DIR", "/model-state/text_classifier"),
)
IMAGE_MODEL_ARTIFACTS_DIR = os.getenv("IMAGE_MODEL_ARTIFACTS_DIR", "/model-state/image_classifier")
IMAGE_SEG_MODEL_ARTIFACTS_DIR = os.getenv(
    "IMAGE_SEG_MODEL_ARTIFACTS_DIR",
    "/model-state/image_segmentation",
)
IMAGE_LOCAL_FILES_ROOT = os.getenv("IMAGE_LOCAL_FILES_ROOT", "/demo-local-files")
TRAINER_URL = os.getenv("TRAINER_URL", "http://trainer:9091/train")
TRAINER_WEBHOOK_URL = os.getenv("TRAINER_WEBHOOK_URL", "http://trainer:9091/webhook/label-studio")
TRAINER_TIMEOUT_SECONDS = float(os.getenv("TRAINER_TIMEOUT_SECONDS", "2.0"))

TEXT_CLS_UNCERTAIN_THRESHOLD = float(os.getenv("TEXT_CLS_UNCERTAIN_THRESHOLD", "0.60"))
TEXT_CLS_UNCERTAIN_MARGIN = float(os.getenv("TEXT_CLS_UNCERTAIN_MARGIN", "0.10"))
IMAGE_CLS_UNCERTAIN_THRESHOLD = float(os.getenv("IMAGE_CLS_UNCERTAIN_THRESHOLD", "0.60"))

DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240

_TEXT_CLASSIFIER_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "classifier.joblib")
_TEXT_VECTORIZER_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "vectorizer.joblib")
_TEXT_METADATA_PATH = os.path.join(TEXT_MODEL_ARTIFACTS_DIR, "metadata.json")
_IMAGE_CLASSIFIER_PATH = os.path.join(IMAGE_MODEL_ARTIFACTS_DIR, "classifier.joblib")
_IMAGE_METADATA_PATH = os.path.join(IMAGE_MODEL_ARTIFACTS_DIR, "metadata.json")
_IMAGE_SEG_ARTIFACT_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "placeholder_model.json")
_IMAGE_SEG_METADATA_PATH = os.path.join(IMAGE_SEG_MODEL_ARTIFACTS_DIR, "metadata.json")
IMAGE_SEG_BACKEND = os.getenv("IMAGE_SEG_BACKEND", "placeholder").strip().lower()
IMAGE_SEG_MODEL_TYPE = os.getenv("IMAGE_SEG_MODEL_TYPE", "vit_t").strip()
IMAGE_SEG_CHECKPOINT = os.getenv("IMAGE_SEG_CHECKPOINT", "").strip()
IMAGE_SEG_MODEL_ID = os.getenv("IMAGE_SEG_MODEL_ID", "facebook/sam2-hiera-large").strip()
IMAGE_SEG_DEVICE = os.getenv("IMAGE_SEG_DEVICE", "cpu").strip()
IMAGE_SEG_PROMPT_STABILITY_ENABLED = os.getenv("IMAGE_SEG_PROMPT_STABILITY_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
IMAGE_SEG_PROMPT_STABILITY_VARIANTS = int(os.getenv("IMAGE_SEG_PROMPT_STABILITY_VARIANTS", "7"))
IMAGE_SEG_PROMPT_STABILITY_JITTER = float(os.getenv("IMAGE_SEG_PROMPT_STABILITY_JITTER", "0.05"))

_TEXT_MODEL_LOCK = threading.Lock()
_TEXT_MODEL_CACHE = {
    "fingerprint": None,
    "bundle": None,
}
_IMAGE_MODEL_LOCK = threading.Lock()
_IMAGE_MODEL_CACHE = {
    "fingerprint": None,
    "bundle": None,
}
_IMAGE_SEG_MODEL_LOCK = threading.Lock()
_IMAGE_SEG_MODEL_CACHE = {
    "fingerprint": None,
    "bundle": None,
}


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_state():
    return {
        "model_version": "demo-rule-v1",
        "training_run": 0,
        "updated_at": "1970-01-01T00:00:00Z",
        "source": "ml-backend-default",
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


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_model_state():
    default_state = _default_state()
    if not os.path.exists(MODEL_STATE_PATH):
        _ensure_parent(MODEL_STATE_PATH)
        with open(MODEL_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(default_state, f, indent=2)
        return default_state

    try:
        with open(MODEL_STATE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_state

    if not isinstance(loaded, dict):
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


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _round4(value):
    return round(float(value), 4)


def _artifact_fingerprint(paths):
    if not all(os.path.exists(path) for path in paths):
        return None

    try:
        return tuple((path, os.path.getmtime(path)) for path in paths)
    except OSError:
        return None


def _load_text_classifier_bundle():
    fingerprint = _artifact_fingerprint((_TEXT_CLASSIFIER_PATH, _TEXT_VECTORIZER_PATH, _TEXT_METADATA_PATH))
    if fingerprint is None:
        return None

    with _TEXT_MODEL_LOCK:
        if _TEXT_MODEL_CACHE["fingerprint"] == fingerprint and _TEXT_MODEL_CACHE["bundle"]:
            return _TEXT_MODEL_CACHE["bundle"]

        metadata = _load_json(_TEXT_METADATA_PATH)
        if not isinstance(metadata, dict):
            _TEXT_MODEL_CACHE["fingerprint"] = None
            _TEXT_MODEL_CACHE["bundle"] = None
            return None

        try:
            classifier = joblib.load(_TEXT_CLASSIFIER_PATH)
            vectorizer = joblib.load(_TEXT_VECTORIZER_PATH)
        except Exception:
            _TEXT_MODEL_CACHE["fingerprint"] = None
            _TEXT_MODEL_CACHE["bundle"] = None
            return None

        bundle = {
            "classifier": classifier,
            "vectorizer": vectorizer,
            "metadata": metadata,
        }
        _TEXT_MODEL_CACHE["fingerprint"] = fingerprint
        _TEXT_MODEL_CACHE["bundle"] = bundle
        return bundle


def _load_image_classifier_bundle():
    fingerprint = _artifact_fingerprint((_IMAGE_CLASSIFIER_PATH, _IMAGE_METADATA_PATH))
    if fingerprint is None:
        return None

    with _IMAGE_MODEL_LOCK:
        if _IMAGE_MODEL_CACHE["fingerprint"] == fingerprint and _IMAGE_MODEL_CACHE["bundle"]:
            return _IMAGE_MODEL_CACHE["bundle"]

        metadata = _load_json(_IMAGE_METADATA_PATH)
        if not isinstance(metadata, dict):
            _IMAGE_MODEL_CACHE["fingerprint"] = None
            _IMAGE_MODEL_CACHE["bundle"] = None
            return None

        try:
            classifier = joblib.load(_IMAGE_CLASSIFIER_PATH)
        except Exception:
            _IMAGE_MODEL_CACHE["fingerprint"] = None
            _IMAGE_MODEL_CACHE["bundle"] = None
            return None

        bundle = {
            "classifier": classifier,
            "metadata": metadata,
        }
        _IMAGE_MODEL_CACHE["fingerprint"] = fingerprint
        _IMAGE_MODEL_CACHE["bundle"] = bundle
        return bundle


def _load_image_segmentation_metadata():
    metadata = _load_json(_IMAGE_SEG_METADATA_PATH)
    return metadata if isinstance(metadata, dict) else None


def _local_tag(tag):
    return tag.split("}", 1)[-1]


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


def _extract_control(root, control_tag, default_name, default_to_name):
    child_tag = "Choice" if control_tag == "Choices" else "Label"
    for node in root.iter():
        if _local_tag(node.tag) != control_tag:
            continue
        labels = []
        for child in node:
            if _local_tag(child.tag) != child_tag:
                continue
            value = child.attrib.get("value")
            if value:
                labels.append(value)
        return {
            "name": node.attrib.get("name", default_name),
            "to_name": node.attrib.get("toName", default_to_name),
            "labels": labels,
        }
    return None


def _parse_label_config(label_config):
    parsed = {
        "image_name": "image",
        "text_name": "text",
        "choices": None,
        "rectanglelabels": None,
        "brushlabels": None,
        "labels": None,
    }

    if not label_config:
        return parsed

    try:
        root = ET.fromstring(label_config)
    except ET.ParseError:
        return parsed

    for node in root.iter():
        node_type = _local_tag(node.tag)
        if node_type == "Image" and parsed["image_name"] == "image":
            parsed["image_name"] = node.attrib.get("name", "image")
        if node_type == "Text" and parsed["text_name"] == "text":
            parsed["text_name"] = node.attrib.get("name", "text")

    parsed["choices"] = _extract_control(
        root,
        "Choices",
        default_name="choices",
        default_to_name=parsed["image_name"],
    )
    parsed["rectanglelabels"] = _extract_control(
        root,
        "RectangleLabels",
        default_name="bbox_label",
        default_to_name=parsed["image_name"],
    )
    parsed["brushlabels"] = _extract_control(
        root,
        "BrushLabels",
        default_name="mask_label",
        default_to_name=parsed["image_name"],
    )
    parsed["labels"] = _extract_control(
        root,
        "Labels",
        default_name="ner_label",
        default_to_name=parsed["text_name"],
    )

    return parsed


def _infer_mode(parsed, tasks):
    image_name = parsed["image_name"]
    text_name = parsed["text_name"]

    rect = parsed["rectanglelabels"]
    brush = parsed["brushlabels"]
    choices = parsed["choices"]
    labels = parsed["labels"]

    if brush and brush.get("to_name") == image_name:
        return "image_segmentation"
    if rect and rect.get("to_name") == image_name:
        return "image_detection"
    if labels and labels.get("to_name") == text_name:
        return "text_ner"
    if choices and choices.get("to_name") == image_name:
        return "image_classification"
    if choices and choices.get("to_name") == text_name:
        return "text_classification"

    if tasks:
        data = tasks[0].get("data", {})
        if "image" in data:
            return "image_classification"
        if "text" in data:
            return "text_classification"

    return "unknown"


def _pick_positive_label(labels):
    if not labels:
        return "Positive"

    preferred = ["positive", "product", "object", "yes", "present"]
    for label in labels:
        low = label.lower()
        if any(token in low for token in preferred):
            return label

    return labels[0]


def _pick_negative_label(labels):
    if not labels:
        return "Negative"

    preferred = ["negative", "other", "none", "no", "absent"]
    for label in labels:
        low = label.lower()
        if any(token in low for token in preferred):
            return label

    if len(labels) >= 2:
        return labels[1]

    return labels[0]


def _clamp_score(value):
    if value < 0.01:
        return 0.01
    if value > 0.99:
        return 0.99
    return round(value, 3)


def _score(base_score, state):
    boost = float(state.get("behavior", {}).get("score_boost", 0.0))
    return _clamp_score(base_score + boost)


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


def _extract_image_feature_vector(image_path):
    if Image is None:
        raise ValueError("Pillow is required for trained image classifier inference")
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


def _predict_with_trained_image_model(task, choices, bundle):
    classifier = bundle["classifier"]
    metadata = bundle["metadata"]

    image_value = task.get("data", {}).get("image")
    image_path = _resolve_image_file_path(image_value)
    if not image_path:
        return None

    try:
        features = _extract_image_feature_vector(image_path).reshape(1, -1)
    except Exception:
        return None

    probabilities = classifier.predict_proba(features)[0]
    label_names = [str(item) for item in classifier.classes_]
    probability_items = [
        {"label": label_names[idx], "probability": _round4(probabilities[idx])}
        for idx in range(len(label_names))
    ]
    probability_items.sort(key=lambda item: item["probability"], reverse=True)

    predicted_raw = probability_items[0]["label"]
    labels = choices.get("labels") or metadata.get("labels") or [predicted_raw]
    label = _normalize_predicted_label(predicted_raw, labels)
    top_probability = float(probability_items[0]["probability"])

    confidence_details = {
        "prediction_source": "trained-image-classifier",
        "confidence": _clamp_score(top_probability),
        "confidence_bucket": _confidence_bucket(top_probability),
        "uncertain": top_probability < IMAGE_CLS_UNCERTAIN_THRESHOLD,
        "uncertainty_reason": {
            "threshold": IMAGE_CLS_UNCERTAIN_THRESHOLD,
            "low_confidence": top_probability < IMAGE_CLS_UNCERTAIN_THRESHOLD,
        },
        "top_probabilities": probability_items,
        "image_path": image_path,
    }
    return label, _clamp_score(top_probability), metadata, confidence_details


def _image_classification(task, parsed, state):
    choices = parsed["choices"] or {
        "name": "image_label",
        "to_name": parsed["image_name"],
        "labels": ["Product", "Other"],
    }

    bundle = _load_image_classifier_bundle()
    if bundle:
        trained_prediction = _predict_with_trained_image_model(task, choices, bundle)
    else:
        trained_prediction = None

    if trained_prediction:
        label, score_value, metadata, confidence_details = trained_prediction
        model_version = metadata.get("model_version") or state["model_version"]
        prediction_source = "trained-image-classifier"
    else:
        image_value = str(task.get("data", {}).get("image", "")).lower()
        caption_value = str(task.get("data", {}).get("caption", "")).lower()
        combined = f"{image_value} {caption_value}"

        tokens = state.get("behavior", {}).get("image_positive_tokens", [])
        if not isinstance(tokens, list):
            tokens = []
        tokens = [str(token).lower() for token in tokens]

        likely_positive = any(token in combined for token in tokens)
        labels = choices.get("labels") or ["Product", "Other"]
        label = _pick_positive_label(labels) if likely_positive else _pick_negative_label(labels)
        score_value = _score(0.87, state)
        model_version = state["model_version"]
        prediction_source = "demo-rule-fallback"
        confidence_details = {
            "prediction_source": prediction_source,
            "confidence": score_value,
            "confidence_bucket": _confidence_bucket(score_value),
            "uncertain": False,
            "uncertainty_reason": {
                "threshold": IMAGE_CLS_UNCERTAIN_THRESHOLD,
                "low_confidence": False,
            },
            "top_probabilities": [],
            "image_path": _resolve_image_file_path(task.get("data", {}).get("image")),
        }

    result = {
        "id": f"{task.get('id', 'task')}_cls",
        "from_name": choices["name"],
        "to_name": choices["to_name"],
        "type": "choices",
        "value": {"choices": [label]},
    }

    return {
        "model_version": model_version,
        "score": score_value,
        "result": [result],
        "prediction_source": prediction_source,
        "confidence": confidence_details,
    }


def _image_detection(task, parsed, state):
    rect = parsed["rectanglelabels"] or {
        "name": "bbox_label",
        "to_name": parsed["image_name"],
        "labels": ["Object"],
    }

    detection = state.get("behavior", {}).get("image_detection", {})
    if not isinstance(detection, dict):
        detection = {}

    label = (rect.get("labels") or ["Object"])[0]
    result = {
        "id": f"{task.get('id', 'task')}_bbox",
        "from_name": rect["name"],
        "to_name": rect["to_name"],
        "type": "rectanglelabels",
        "original_width": DEFAULT_IMAGE_WIDTH,
        "original_height": DEFAULT_IMAGE_HEIGHT,
        "image_rotation": 0,
        "value": {
            "rotation": 0,
            "x": float(detection.get("x", 20.0)),
            "y": float(detection.get("y", 22.0)),
            "width": float(detection.get("width", 45.0)),
            "height": float(detection.get("height", 45.0)),
            "rectanglelabels": [label],
        },
    }

    base_score = float(detection.get("score", 0.74))
    return {
        "model_version": state["model_version"],
        "score": _score(base_score, state),
        "result": [result],
    }


def _image_dimensions(task):
    image_path = _resolve_image_file_path(task.get("data", {}).get("image"))
    if image_path and Image is not None:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > 0 and height > 0:
                    return int(width), int(height)
        except Exception:
            pass
    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def _fallback_mask_rle(width, height):
    mask_left = max(0, width // 4)
    mask_right = min(width, width - mask_left)
    mask_top = max(0, height // 4)
    mask_bottom = min(height, height - mask_top)

    runs = []
    current = 0
    length = 0
    for y in range(height):
        for x in range(width):
            value = 1 if mask_left <= x < mask_right and mask_top <= y < mask_bottom else 0
            if value == current:
                length += 1
            else:
                runs.append(length)
                current = value
                length = 1
    runs.append(length)
    return runs


def _binary_mask_to_minimal_rle(mask):
    runs = []
    current = 0
    length = 0
    for value in _iter_binary_mask_values(mask):
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1
    runs.append(length)
    return runs


def _iter_binary_mask_values(mask):
    if hasattr(mask, "ndim") and hasattr(mask, "reshape"):
        arr = mask
        if arr.ndim == 3:
            arr = arr[0]
        try:
            arr = (arr > 0).astype(np.uint8)
        except Exception:
            pass
        for value in arr.reshape(-1):
            yield 1 if int(value) > 0 else 0
        return

    if isinstance(mask, (list, tuple)):
        for item in mask:
            if isinstance(item, (list, tuple)) or hasattr(item, "reshape"):
                for nested in _iter_binary_mask_values(item):
                    yield nested
            else:
                yield 1 if int(item) > 0 else 0
        return

    yield 1 if int(mask) > 0 else 0


def _encode_segmentation_mask_rle(mask):
    try:
        from label_studio_converter.brush import mask2rle

        arr = np.asarray(mask) if hasattr(np, "asarray") else mask
        if hasattr(arr, "ndim") and arr.ndim == 3:
            arr = arr[0]
        if hasattr(arr, "astype"):
            arr = (arr > 0).astype(np.uint8)
        return mask2rle(arr), "label-studio-converter"
    except Exception:
        return _binary_mask_to_minimal_rle(mask), "fallback-minimal"


def _placeholder_segmentation_rle(width, height):
    mask = _placeholder_segmentation_mask(width, height)
    if mask is None:
        return _fallback_mask_rle(width, height), "fallback-minimal", None
    rle, encoder = _encode_segmentation_mask_rle(mask)
    return rle, encoder, mask


def _placeholder_segmentation_mask(width, height):
    try:
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : max(height // 4 + 1, (height * 3) // 4), width // 4 : max(width // 4 + 1, (width * 3) // 4)] = 1
        return mask
    except Exception:
        return None


def _segmentation_mask_bbox(mask):
    try:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[0]
        foreground = arr > 0
        if not foreground.any():
            return None
        ys, xs = np.where(foreground)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    except Exception:
        return None


def _normalize_image_segmentation_backend(value):
    normalized = str(value or "placeholder").strip().lower().replace("-", "_")
    aliases = {
        "mobile_sam": "mobilesam",
        "mobile": "mobilesam",
        "segment_anything": "sam",
        "segmentanything": "sam",
        "sam_v1": "sam",
    }
    return aliases.get(normalized, normalized)


def _selected_image_segmentation_backend(metadata):
    env_backend = _normalize_image_segmentation_backend(IMAGE_SEG_BACKEND)
    if env_backend != "placeholder":
        return env_backend
    if isinstance(metadata, dict):
        metadata_backend = metadata.get("backend")
        if not metadata_backend and isinstance(metadata.get("model"), dict):
            metadata_backend = metadata["model"].get("backend")
        if metadata_backend:
            return _normalize_image_segmentation_backend(metadata_backend)
    return "placeholder"


def _image_segmentation_device():
    if IMAGE_SEG_DEVICE and IMAGE_SEG_DEVICE.lower() != "auto":
        return IMAGE_SEG_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_image_rgb_array(image_path):
    if Image is None:
        raise ValueError("Pillow is required for SAM image segmentation inference")
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if hasattr(np, "asarray"):
            return np.asarray(rgb)
        return list(rgb.getdata())


def _center_box_prompt(width, height):
    left = max(0.0, float(width) * 0.18)
    top = max(0.0, float(height) * 0.18)
    right = min(float(width - 1), float(width) * 0.82)
    bottom = min(float(height - 1), float(height) * 0.82)
    return _box_prompt_array(left, top, right, bottom)


def _box_prompt_array(left, top, right, bottom):
    if hasattr(np, "asarray") and hasattr(np, "float32"):
        return np.asarray([left, top, right, bottom], dtype=np.float32)
    return [left, top, right, bottom]


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _box_unit_is_percent(value):
    unit = None
    if isinstance(value, dict):
        unit = value.get("unit") or value.get("units") or value.get("coordinate_system")
    return str(unit or "").strip().lower() in {"percent", "percentage", "pct", "%"}


def _box_unit_is_normalized(value):
    if not isinstance(value, dict):
        return False
    return value.get("normalized") is True or str(value.get("coordinate_system", "")).strip().lower() in {
        "normalized",
        "normalized_xyxy",
    }


def _box_values_from_mapping(value):
    if all(key in value for key in ("x_min", "y_min", "x_max", "y_max")):
        return (
            _as_float(value.get("x_min")),
            _as_float(value.get("y_min")),
            _as_float(value.get("x_max")),
            _as_float(value.get("y_max")),
        )
    if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
        return (
            _as_float(value.get("xmin")),
            _as_float(value.get("ymin")),
            _as_float(value.get("xmax")),
            _as_float(value.get("ymax")),
        )
    if all(key in value for key in ("left", "top", "right", "bottom")):
        return (
            _as_float(value.get("left")),
            _as_float(value.get("top")),
            _as_float(value.get("right")),
            _as_float(value.get("bottom")),
        )
    if all(key in value for key in ("x", "y", "width", "height")):
        left = _as_float(value.get("x"))
        top = _as_float(value.get("y"))
        box_width = _as_float(value.get("width"))
        box_height = _as_float(value.get("height"))
        if None in (left, top, box_width, box_height):
            return None
        return left, top, left + box_width, top + box_height
    return None


def _normalize_segmentation_box_prompt(value, width, height):
    if value is None:
        return None

    percent = _box_unit_is_percent(value)
    normalized = _box_unit_is_normalized(value)
    if isinstance(value, dict):
        raw_values = value.get("box") or value.get("bbox")
        if raw_values is not None:
            nested = _normalize_segmentation_box_prompt(raw_values, width, height)
            if nested and (percent or normalized):
                left, top, right, bottom = nested["box"]
                divisor = 100.0 if percent else 1.0
                coordinate_system = "percent_xyxy" if percent else "normalized_xyxy"
                return _clip_segmentation_box(
                    left * float(width) / divisor,
                    top * float(height) / divisor,
                    right * float(width) / divisor,
                    bottom * float(height) / divisor,
                    width,
                    height,
                    coordinate_system,
                )
            return nested
        raw_values = _box_values_from_mapping(value)
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        raw_values = tuple(_as_float(item) for item in value)
    else:
        raw_values = None

    if not raw_values or any(item is None for item in raw_values):
        return None

    left, top, right, bottom = raw_values
    coordinate_system = "pixel_xyxy"
    if percent or normalized:
        divisor = 100.0 if percent else 1.0
        left = left * float(width) / divisor
        right = right * float(width) / divisor
        top = top * float(height) / divisor
        bottom = bottom * float(height) / divisor
        coordinate_system = "percent_xyxy" if percent else "normalized_xyxy"
    return _clip_segmentation_box(left, top, right, bottom, width, height, coordinate_system)


def _clip_segmentation_box(left, top, right, bottom, width, height, coordinate_system):
    left = max(0.0, min(float(width - 1), float(left)))
    top = max(0.0, min(float(height - 1), float(top)))
    right = max(0.0, min(float(width - 1), float(right)))
    bottom = max(0.0, min(float(height - 1), float(bottom)))
    if right <= left or bottom <= top:
        return None
    return {
        "box": [left, top, right, bottom],
        "coordinate_system": coordinate_system,
    }


def _segmentation_box_prompt_for_task(task, width, height):
    data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
    meta = task.get("meta", {}) if isinstance(task.get("meta"), dict) else {}
    for source_name, container in (("data.bbox", data), ("data.box", data), ("meta.bbox", meta), ("meta.box", meta)):
        key = source_name.split(".")[1]
        prompt = _normalize_segmentation_box_prompt(container.get(key), width, height)
        if prompt:
            prompt["source"] = source_name
            return prompt
    prompt = _choose_best_candidate_box(data.get("candidates"), width, height)
    if prompt:
        return prompt
    prompt = _choose_best_candidate_box(meta.get("candidates"), width, height)
    if prompt:
        return prompt
    prompt = _extract_rectanglelabels_prompt_box(task, width, height)
    if prompt:
        return prompt
    return {
        "box": [float(item) for item in _center_box_prompt(width, height)],
        "coordinate_system": "pixel_xyxy",
        "source": "center_fallback",
    }


def _candidate_score(candidate):
    if not isinstance(candidate, dict):
        return None
    for key in ("score", "confidence", "probability", "prob"):
        score = _as_float(candidate.get(key))
        if score is not None:
            return score
    return None


def _extract_bbox_from_candidate(candidate, width, height):
    if not isinstance(candidate, dict):
        return None
    for key in ("bbox", "box"):
        prompt = _normalize_segmentation_box_prompt(candidate.get(key), width, height)
        if prompt:
            prompt["source"] = f"webhook_candidate.{key}"
            return prompt
    value = candidate.get("value")
    if isinstance(value, dict):
        for key in ("bbox", "box"):
            prompt = _normalize_segmentation_box_prompt(value.get(key), width, height)
            if prompt:
                prompt["source"] = f"webhook_candidate.{key}"
                return prompt
    return None


def _choose_best_candidate_box(candidates, width, height):
    if not isinstance(candidates, list):
        return None
    valid = []
    for index, candidate in enumerate(candidates):
        prompt = _extract_bbox_from_candidate(candidate, width, height)
        if prompt:
            valid.append((index, _candidate_score(candidate), prompt))
    if not valid:
        return None
    scored = [item for item in valid if item[1] is not None]
    if scored:
        return max(scored, key=lambda item: (item[1], -item[0]))[2]
    return valid[0][2]


def _iter_label_studio_results(task, key):
    rows = task.get(key)
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        results = row.get("result")
        if not isinstance(results, list):
            continue
        for result in results:
            if isinstance(result, dict):
                yield result


def _extract_bbox_from_rectanglelabels(result, width, height, source):
    if result.get("type") != "rectanglelabels":
        return None
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    raw_values = _box_values_from_mapping(value)
    if not raw_values or any(item is None for item in raw_values):
        return None
    left, top, right, bottom = raw_values
    prompt = _clip_segmentation_box(
        left * float(width) / 100.0,
        top * float(height) / 100.0,
        right * float(width) / 100.0,
        bottom * float(height) / 100.0,
        width,
        height,
        "percent_xyxy",
    )
    if prompt:
        prompt["source"] = source
    return prompt


def _extract_rectanglelabels_prompt_box(task, width, height):
    for key, source in (("predictions", "prediction.rectanglelabels"), ("annotations", "annotation.rectanglelabels")):
        for result in _iter_label_studio_results(task, key):
            prompt = _extract_bbox_from_rectanglelabels(result, width, height, source)
            if prompt:
                return prompt
    return None


def _load_sam2_image_predictor():
    if not IMAGE_SEG_MODEL_ID:
        raise ValueError("IMAGE_SEG_MODEL_ID is required for IMAGE_SEG_BACKEND=sam2")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _image_segmentation_device()
    try:
        predictor = SAM2ImagePredictor.from_pretrained(IMAGE_SEG_MODEL_ID, device=device)
    except TypeError:
        predictor = SAM2ImagePredictor.from_pretrained(IMAGE_SEG_MODEL_ID)
        model = getattr(predictor, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(device=device)
    return predictor


def _load_segment_anything_predictor(backend):
    if not IMAGE_SEG_CHECKPOINT:
        raise ValueError("IMAGE_SEG_CHECKPOINT is required for IMAGE_SEG_BACKEND=mobilesam or sam")
    if backend == "mobilesam":
        from mobile_sam import SamPredictor, sam_model_registry
    else:
        from segment_anything import SamPredictor, sam_model_registry

    if IMAGE_SEG_MODEL_TYPE not in sam_model_registry:
        available = ", ".join(sorted(str(item) for item in sam_model_registry.keys()))
        raise ValueError(f"unknown IMAGE_SEG_MODEL_TYPE={IMAGE_SEG_MODEL_TYPE}; available={available}")
    model = sam_model_registry[IMAGE_SEG_MODEL_TYPE](checkpoint=IMAGE_SEG_CHECKPOINT)
    if hasattr(model, "to"):
        model.to(device=_image_segmentation_device())
    return SamPredictor(model)


def _load_image_segmentation_predictor(backend):
    fingerprint = (
        backend,
        IMAGE_SEG_MODEL_TYPE,
        IMAGE_SEG_CHECKPOINT,
        IMAGE_SEG_MODEL_ID,
        IMAGE_SEG_DEVICE,
    )
    with _IMAGE_SEG_MODEL_LOCK:
        if _IMAGE_SEG_MODEL_CACHE["fingerprint"] == fingerprint and _IMAGE_SEG_MODEL_CACHE["bundle"]:
            return _IMAGE_SEG_MODEL_CACHE["bundle"]

        if backend == "sam2":
            predictor = _load_sam2_image_predictor()
        elif backend in {"mobilesam", "sam"}:
            predictor = _load_segment_anything_predictor(backend)
        else:
            raise ValueError(f"unsupported IMAGE_SEG_BACKEND={backend}")

        bundle = {"backend": backend, "predictor": predictor}
        _IMAGE_SEG_MODEL_CACHE["fingerprint"] = fingerprint
        _IMAGE_SEG_MODEL_CACHE["bundle"] = bundle
    return bundle


def _predict_sam_mask_for_prompt(predictor, prompt_box):
    masks, scores, _ = predictor.predict(box=_box_prompt_array(*prompt_box), multimask_output=False)
    selected_mask = _select_first_mask(masks)
    if selected_mask is None:
        raise ValueError("SAM-compatible backend returned no masks")
    return selected_mask, scores


def _predict_with_sam_image_backend(task, backend, width, height):
    image_path = _resolve_image_file_path(task.get("data", {}).get("image"))
    if not image_path:
        raise ValueError("local image path is required for SAM image segmentation inference")

    bundle = _load_image_segmentation_predictor(backend)
    predictor = bundle["predictor"]
    predictor.set_image(_load_image_rgb_array(image_path))
    prompt = _segmentation_box_prompt_for_task(task, width, height)
    selected_mask, scores = _predict_sam_mask_for_prompt(predictor, prompt["box"])
    rle, rle_encoder = _encode_segmentation_mask_rle(selected_mask)
    mask_bbox = _segmentation_mask_bbox(selected_mask)
    score_value = _first_segmentation_score(scores)
    uncertainty = _evaluate_sam_prompt_stability(predictor, selected_mask, prompt, width, height)

    return {
        "rle": rle,
        "rle_encoder": rle_encoder,
        "score": score_value,
        "prediction_source": f"{backend}-image-segmentation",
        "backend": backend,
        "image_path": image_path,
        "prompt": prompt,
        "mask": selected_mask,
        "mask_bbox": mask_bbox,
        "uncertainty": uncertainty,
    }


def _evaluate_sam_prompt_stability(predictor, original_mask, prompt, width, height):
    if not IMAGE_SEG_PROMPT_STABILITY_ENABLED:
        return None
    try:
        variants = generate_bbox_prompt_variants(
            prompt.get("box"),
            width,
            height,
            jitter_ratio=IMAGE_SEG_PROMPT_STABILITY_JITTER,
        )
        limit = max(1, int(IMAGE_SEG_PROMPT_STABILITY_VARIANTS))
        variants = variants[:limit]
        if not variants:
            return {
                "method": "prompt_stability",
                "enabled": True,
                "error": "valid prompt bbox is required",
                "stable": False,
                "stability_bucket": "unknown",
                "reason": ["prompt_stability_error"],
            }

        masks = [original_mask]
        for variant in variants[1:]:
            mask, _ = _predict_sam_mask_for_prompt(predictor, variant["bbox"])
            masks.append(mask)
        return evaluate_prompt_stability(
            masks,
            prompt_variants=variants,
            image_width=width,
            image_height=height,
        )["uncertainty"]
    except Exception as exc:
        return {
            "method": "prompt_stability",
            "enabled": True,
            "error": str(exc),
            "stable": False,
            "stability_bucket": "unknown",
            "reason": ["prompt_stability_error"],
        }


def _select_first_mask(masks):
    if masks is None:
        return None
    if hasattr(masks, "size") and masks.size == 0:
        return None
    if hasattr(masks, "ndim"):
        return masks[0] if masks.ndim >= 3 else masks
    if isinstance(masks, (list, tuple)):
        if not masks:
            return None
        return masks[0]
    return masks


def _first_segmentation_score(scores):
    try:
        if hasattr(scores, "reshape"):
            score_arr = scores.reshape(-1)
            if getattr(score_arr, "size", 0):
                return _clamp_score(float(score_arr[0]))
        if isinstance(scores, (list, tuple)) and scores:
            return _clamp_score(float(scores[0]))
    except Exception:
        pass
    return 0.8


def _image_segmentation(task, parsed, state):
    brush = parsed["brushlabels"] or {
        "name": "mask_label",
        "to_name": parsed["image_name"],
        "labels": ["Object"],
    }
    label = (brush.get("labels") or ["Object"])[0]
    width, height = _image_dimensions(task)
    metadata = _load_image_segmentation_metadata() or {}
    selected_backend = _selected_image_segmentation_backend(metadata)
    backend_error = None

    if selected_backend != "placeholder":
        try:
            backend_prediction = _predict_with_sam_image_backend(task, selected_backend, width, height)
        except Exception as exc:
            backend_error = str(exc)
            backend_prediction = None
    else:
        backend_prediction = None

    if backend_prediction:
        rle = backend_prediction["rle"]
        rle_encoder = backend_prediction["rle_encoder"]
        score_value = backend_prediction["score"]
        prediction_source = backend_prediction["prediction_source"]
        selected_mask = backend_prediction.get("mask")
        mask_bbox = backend_prediction.get("mask_bbox")
    else:
        rle, rle_encoder, selected_mask = _placeholder_segmentation_rle(width, height)
        mask_bbox = _segmentation_mask_bbox(selected_mask)
        score_value = 0.65
        prediction_source = "placeholder-image-segmentation"

    model_version = (
        metadata.get("model_version")
        or state.get("image_segmentation", {}).get("model_version")
        or state["model_version"]
    )

    result = {
        "id": f"{task.get('id', 'task')}_mask",
        "from_name": brush["name"],
        "to_name": brush["to_name"],
        "type": "brushlabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "format": "rle",
            "rle": rle,
            "brushlabels": [label],
        },
    }

    confidence = {
        "prediction_source": prediction_source,
        "confidence": score_value,
        "confidence_bucket": _confidence_bucket(score_value),
        "uncertain": False,
        "rle_encoder": rle_encoder,
    }
    if backend_prediction:
        confidence.update(
            {
                "backend": backend_prediction["backend"],
                "image_path": backend_prediction["image_path"],
                "prompt": backend_prediction["prompt"]["source"],
                "prompt_box": backend_prediction["prompt"]["box"],
                "prompt_coordinate_system": backend_prediction["prompt"]["coordinate_system"],
            }
        )
    elif selected_backend != "placeholder":
        confidence.update(
            {
                "requested_backend": selected_backend,
                "backend_error": backend_error or "backend did not return a mask",
                "fallback": "placeholder",
            }
        )

    result["meta"] = {
        key: confidence[key]
        for key in (
            "prediction_source",
            "rle_encoder",
            "backend",
            "requested_backend",
            "backend_error",
            "prompt",
            "prompt_box",
            "prompt_coordinate_system",
            "mask_bbox",
            "fallback",
        )
        if key in confidence
    }
    if mask_bbox is not None:
        result["meta"]["mask_bbox"] = mask_bbox
    if backend_prediction and backend_prediction.get("uncertainty"):
        result["meta"]["uncertainty"] = backend_prediction["uncertainty"]
    quality_metadata = evaluate_segmentation_quality(
        selected_mask,
        width,
        height,
        prompt_bbox=confidence.get("prompt_box"),
        mask_bbox=mask_bbox,
        rle_length=len(rle) if isinstance(rle, list) else None,
        backend_metadata=confidence,
    )
    result["meta"].update(quality_metadata)

    return {
        "model_version": model_version,
        "score": score_value,
        "result": [result],
        "prediction_source": prediction_source,
        "confidence": confidence,
    }


def _normalize_predicted_label(predicted, supported_labels):
    if not supported_labels:
        return predicted

    for label in supported_labels:
        if label == predicted:
            return label

    low_pred = str(predicted).lower()
    for label in supported_labels:
        if str(label).lower() == low_pred:
            return label

    return supported_labels[0]


def _confidence_bucket(score):
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _predict_with_trained_text_model(text, choices, bundle):
    classifier = bundle["classifier"]
    vectorizer = bundle["vectorizer"]
    metadata = bundle["metadata"]

    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]

    label_names = [str(item) for item in classifier.classes_]
    probability_items = [
        {"label": label_names[idx], "probability": _round4(probabilities[idx])}
        for idx in range(len(label_names))
    ]
    probability_items.sort(key=lambda item: item["probability"], reverse=True)

    predicted_raw = probability_items[0]["label"]
    labels = choices.get("labels") or metadata.get("labels") or [predicted_raw]
    label = _normalize_predicted_label(predicted_raw, labels)

    top_probability = float(probability_items[0]["probability"])
    second_probability = float(probability_items[1]["probability"]) if len(probability_items) > 1 else 0.0
    margin = top_probability - second_probability

    uncertain = bool(
        top_probability < TEXT_CLS_UNCERTAIN_THRESHOLD or margin < TEXT_CLS_UNCERTAIN_MARGIN
    )

    details = {
        "prediction_source": "trained-text-classifier",
        "confidence": _clamp_score(top_probability),
        "confidence_bucket": _confidence_bucket(top_probability),
        "uncertain": uncertain,
        "uncertainty_reason": {
            "threshold": TEXT_CLS_UNCERTAIN_THRESHOLD,
            "margin_threshold": TEXT_CLS_UNCERTAIN_MARGIN,
            "observed_margin": _round4(margin),
            "low_confidence": top_probability < TEXT_CLS_UNCERTAIN_THRESHOLD,
            "small_margin": margin < TEXT_CLS_UNCERTAIN_MARGIN,
        },
        "top_probabilities": probability_items,
    }

    return label, _clamp_score(top_probability), metadata, details


def _text_classification(task, parsed, state):
    choices = parsed["choices"] or {
        "name": "text_label",
        "to_name": parsed["text_name"],
        "labels": ["Positive", "Negative"],
    }

    text = str(task.get("data", {}).get("text", ""))

    bundle = _load_text_classifier_bundle()
    if bundle:
        label, score_value, metadata, confidence_details = _predict_with_trained_text_model(text, choices, bundle)
        model_version = metadata.get("model_version") or state["model_version"]
        prediction_source = "trained-text-classifier"
    else:
        low = text.lower()
        positive_tokens = state.get("behavior", {}).get("text_positive_tokens", [])
        if not isinstance(positive_tokens, list):
            positive_tokens = []
        positive_tokens = [str(token).lower() for token in positive_tokens]

        positive = any(token in low for token in positive_tokens)

        labels = choices.get("labels") or ["Positive", "Negative"]
        label = _pick_positive_label(labels) if positive else _pick_negative_label(labels)
        score_value = _score(0.83, state)
        model_version = state["model_version"]
        prediction_source = "demo-rule-fallback"
        confidence_details = {
            "prediction_source": prediction_source,
            "confidence": score_value,
            "confidence_bucket": _confidence_bucket(score_value),
            "uncertain": False,
            "uncertainty_reason": {
                "threshold": TEXT_CLS_UNCERTAIN_THRESHOLD,
                "margin_threshold": TEXT_CLS_UNCERTAIN_MARGIN,
                "observed_margin": None,
                "low_confidence": False,
                "small_margin": False,
            },
            "top_probabilities": [],
        }

    result = {
        "id": f"{task.get('id', 'task')}_txt_cls",
        "from_name": choices["name"],
        "to_name": choices["to_name"],
        "type": "choices",
        "value": {"choices": [label]},
    }

    return {
        "model_version": model_version,
        "score": score_value,
        "result": [result],
        "prediction_source": prediction_source,
        "confidence": confidence_details,
    }


def _find_spans(text, state):
    keyword_map = state.get("behavior", {}).get("ner_keywords", {})
    if not isinstance(keyword_map, dict) or not keyword_map:
        keyword_map = {
            "openai": "ORG",
            "alice": "PERSON",
            "paris": "LOC",
            "berlin": "LOC",
        }

    found = []
    for keyword, label in keyword_map.items():
        pattern = r"\b" + re.escape(str(keyword)) + r"\b"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append((match.start(), match.end(), str(label)))

    found.sort(key=lambda x: x[0])
    return found


def _choose_ner_label(candidates, desired):
    for candidate in candidates:
        if candidate.lower() == desired.lower():
            return candidate
    return candidates[0] if candidates else desired


def _text_ner(task, parsed, state):
    labels_cfg = parsed["labels"] or {
        "name": "ner_label",
        "to_name": parsed["text_name"],
        "labels": ["ORG", "PERSON", "LOC"],
    }

    text = str(task.get("data", {}).get("text", ""))
    spans = _find_spans(text, state)
    supported = labels_cfg.get("labels") or ["ORG", "PERSON", "LOC"]

    results = []
    for idx, (start, end, desired_label) in enumerate(spans, start=1):
        label = _choose_ner_label(supported, desired_label)
        results.append(
            {
                "id": f"{task.get('id', 'task')}_ner_{idx}",
                "from_name": labels_cfg["name"],
                "to_name": labels_cfg["to_name"],
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [label],
                },
            }
        )

    if not results and text:
        first_word = text.split()[0]
        end = len(first_word)
        fallback_label = supported[0] if supported else "ORG"
        results.append(
            {
                "id": f"{task.get('id', 'task')}_ner_fallback",
                "from_name": labels_cfg["name"],
                "to_name": labels_cfg["to_name"],
                "type": "labels",
                "value": {
                    "start": 0,
                    "end": end,
                    "text": text[:end],
                    "labels": [fallback_label],
                },
            }
        )

    return {
        "model_version": state["model_version"],
        "score": _score(0.79, state),
        "result": results,
    }


def _predict(payload):
    tasks = payload.get("tasks") or []
    project = payload.get("project")
    project_label_config = project.get("label_config") if isinstance(project, dict) else None
    label_config = payload.get("label_config") or project_label_config

    parsed = _parse_label_config(label_config or "")
    mode = _infer_mode(parsed, tasks)
    state = _load_model_state()

    predictions = []
    for task in tasks:
        if mode == "image_segmentation":
            predictions.append(_image_segmentation(task, parsed, state))
        elif mode == "image_detection":
            predictions.append(_image_detection(task, parsed, state))
        elif mode == "text_ner":
            predictions.append(_text_ner(task, parsed, state))
        elif mode == "image_classification":
            predictions.append(_image_classification(task, parsed, state))
        elif mode == "text_classification":
            predictions.append(_text_classification(task, parsed, state))
        else:
            predictions.append({"model_version": state["model_version"], "score": 0.1, "result": []})

    if mode in ("text_classification", "image_classification", "image_segmentation") and predictions:
        top_version = predictions[0].get("model_version", state["model_version"])
    else:
        top_version = state["model_version"]

    return {
        "model_version": top_version,
        "training_run": int(state.get("training_run", 0)),
        "mode": mode,
        "results": predictions,
    }


def _post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TRAINER_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8") if response.length != 0 else ""
        if not raw:
            return {"status_code": response.status}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        parsed["status_code"] = response.status
        return parsed


def _trigger_trainer(payload, webhook=False):
    url = TRAINER_WEBHOOK_URL if webhook else TRAINER_URL
    wrapped = {
        "triggered_at": _utc_now_iso(),
        "source": "ml-backend",
        "payload": payload,
    }

    if isinstance(payload, dict):
        wrapped.update(payload)

    try:
        return {
            "accepted": True,
            "trainer_url": url,
            "trainer_response": _post_json(url, wrapped),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "accepted": False,
            "trainer_url": url,
            "status_code": exc.code,
            "error": body,
        }
    except urllib.error.URLError as exc:
        return {
            "accepted": False,
            "trainer_url": url,
            "error": str(exc),
        }


class BackendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.rstrip("/") or "/"
        if route in ("/", "/health"):
            state = _load_model_state()
            text_bundle = _load_text_classifier_bundle()
            image_bundle = _load_image_classifier_bundle()
            text_metadata = text_bundle["metadata"] if text_bundle else _load_json(_TEXT_METADATA_PATH)
            image_metadata = image_bundle["metadata"] if image_bundle else _load_json(_IMAGE_METADATA_PATH)
            image_segmentation_metadata = _load_image_segmentation_metadata()
            _send_json(
                self,
                200,
                {
                    "status": "UP",
                    "service": SERVICE_NAME,
                    "phase": "phase-4-text-and-image-classifiers",
                    "model_version": state.get("model_version", "demo-rule-v1"),
                    "training_run": int(state.get("training_run", 0)),
                    "model_state_path": MODEL_STATE_PATH,
                    "text_classifier": {
                        "active": bool(text_metadata),
                        "metadata": text_metadata,
                        "metadata_path": _TEXT_METADATA_PATH,
                        "uncertainty": {
                            "confidence_threshold": TEXT_CLS_UNCERTAIN_THRESHOLD,
                            "margin_threshold": TEXT_CLS_UNCERTAIN_MARGIN,
                        },
                    },
                    "image_classifier": {
                        "active": bool(image_metadata),
                        "metadata": image_metadata,
                        "metadata_path": _IMAGE_METADATA_PATH,
                        "uncertainty": {
                            "confidence_threshold": IMAGE_CLS_UNCERTAIN_THRESHOLD,
                        },
                    },
                    "image_segmentation": {
                        "active": bool(image_segmentation_metadata),
                        "metadata": image_segmentation_metadata,
                        "metadata_path": _IMAGE_SEG_METADATA_PATH,
                        "artifact_path": _IMAGE_SEG_ARTIFACT_PATH,
                    },
                },
            )
            return

        if route == "/model-state":
            _send_json(self, 200, _load_model_state())
            return

        if route in ("/models/text-classification", "/models/text-classification/current"):
            metadata = _load_json(_TEXT_METADATA_PATH)
            if not isinstance(metadata, dict):
                _send_json(self, 404, {"detail": "no trained text classification model"})
                return
            _send_json(self, 200, metadata)
            return

        if route in ("/models/image-classification", "/models/image-classification/current"):
            metadata = _load_json(_IMAGE_METADATA_PATH)
            if not isinstance(metadata, dict):
                _send_json(self, 404, {"detail": "no trained image classification model"})
                return
            _send_json(self, 200, metadata)
            return

        if route in ("/models/image-segmentation", "/models/image-segmentation/current"):
            metadata = _load_image_segmentation_metadata()
            if not isinstance(metadata, dict):
                _send_json(self, 404, {"detail": "no trained image segmentation model"})
                return
            _send_json(self, 200, metadata)
            return

        _send_json(self, 404, {"detail": "not found"})

    def do_POST(self):
        route = self.path.rstrip("/") or "/"
        payload = _read_json_body(self)

        if route == "/predict":
            _send_json(self, 200, _predict(payload))
            return

        if route == "/setup":
            state = _load_model_state()
            _send_json(
                self,
                200,
                {
                    "model_version": state.get("model_version"),
                    "service": SERVICE_NAME,
                    "ready": True,
                },
            )
            return

        if route == "/train":
            result = _trigger_trainer(payload, webhook=False)
            status_code = 202 if result.get("accepted") else 502
            _send_json(self, status_code, result)
            return

        if route in ("/webhook", "/webhook/label-studio"):
            result = _trigger_trainer(payload, webhook=True)
            status_code = 202 if result.get("accepted") else 502
            _send_json(self, status_code, result)
            return

        _send_json(self, 404, {"detail": "not found"})

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), BackendHandler).serve_forever()
