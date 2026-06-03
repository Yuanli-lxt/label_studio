from __future__ import annotations

import os
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.prediction_feature_scoring import (
    PREDICTION_FEATURE_NAMES,
    calibrated_boundary_shape_score,
    prediction_features,
    rank_boundary_shape_scores,
)


SHADOW_SCORE_VERSION = "boundary_shape_shadow_v1"
LEARNED_ARTIFACT_ENV = "IMAGE_SEG_LEARNED_FUSION_ARTIFACT_DIR"
ENABLE_SHADOW_ENV = "IMAGE_SEG_ENABLE_SHADOW_SCORING"
ENABLE_LEARNED_ENV = "IMAGE_SEG_ENABLE_LEARNED_SHADOW_SCORES"
ENABLE_GATED_ENV = "IMAGE_SEG_ENABLE_GATED_SHADOW_SCORES"
SEGMENTATION_ENABLE_SHADOW_ENV = "SEGMENTATION_ENABLE_BOUNDARY_SHAPE_SHADOW"
SEGMENTATION_ENABLE_LEARNED_ENV = "SEGMENTATION_ENABLE_LEARNED_BOUNDARY_SHAPE_SHADOW"
SEGMENTATION_ARTIFACT_DIR_ENV = "SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_DIR"
SEGMENTATION_ARTIFACT_REGISTRY_ENV = "SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_REGISTRY"
SEGMENTATION_ARTIFACT_VERSION_ENV = "SEGMENTATION_BOUNDARY_SHAPE_ARTIFACT_VERSION"
SEGMENTATION_SHADOW_VERSION_ENV = "SEGMENTATION_BOUNDARY_SHAPE_SHADOW_VERSION"
SEGMENTATION_FAIL_OPEN_ENV = "SEGMENTATION_BOUNDARY_SHAPE_FAIL_OPEN"

LEARNED_SCORE_KEYS = {
    "learned_boundary_shape_only": "learned_boundary_shape_only_score",
    "learned_current_plus_boundary_shape": "learned_current_plus_boundary_shape_score",
}
LOGGER = logging.getLogger(__name__)


def shadow_scores_for_item(
    item: dict,
    boundary_shape_rank_score: float | None = None,
    learned_artifact_dir: str | None = None,
    enable_learned_shadow_scores: bool | None = None,
    enable_gated_shadow_scores: bool | None = None,
) -> dict:
    """Return prediction-time-only experimental scores for a queue item."""
    calibrated = _component_or_calibrated(item)
    rank_score = _clip(_num(boundary_shape_rank_score, default=0.5))
    current = _clip(_num(item.get("priority_score")))
    learned_enabled = _learned_enabled(enable_learned_shadow_scores)
    gated_enabled = _flag_enabled_any([ENABLE_GATED_ENV], False) if enable_gated_shadow_scores is None else bool(enable_gated_shadow_scores)
    learned = _timed_learned_shadow_result(item, learned_artifact_dir, learned_enabled)["scores"]
    return {
        "boundary_shape_calibrated_score": calibrated,
        "boundary_shape_rank_score": rank_score,
        "learned_boundary_shape_only_score": learned.get("learned_boundary_shape_only_score"),
        "learned_current_plus_boundary_shape_score": learned.get("learned_current_plus_boundary_shape_score"),
        "gated_boundary_shape_score": gated_boundary_shape_score(item, rank_score) if gated_enabled else None,
        "gated_current_boundary_score": gated_current_boundary_score(item, current, calibrated) if gated_enabled else None,
        "score_version": _shadow_score_version(),
        "enabled": True,
        "shadow_only": True,
    }


def shadow_score_metadata(
    item: dict,
    learned_artifact_dir: str | None = None,
    enable_learned_shadow_scores: bool | None = None,
    enable_gated_shadow_scores: bool | None = None,
) -> dict:
    features = prediction_features(item)
    missing = [name for name in PREDICTION_FEATURE_NAMES if name not in features or features.get(name) is None]
    learned_enabled = _learned_enabled(enable_learned_shadow_scores)
    learned = _timed_learned_shadow_result(item, learned_artifact_dir, learned_enabled)
    metadata = {
        "score_version": _shadow_score_version(),
        "artifact_version": learned.get("artifact_version"),
        "artifact_path": learned.get("artifact_path"),
        "artifact_loaded": bool(learned.get("artifact_loaded")),
        "artifact_validation_status": learned.get("artifact_validation_status"),
        "shadow_only": True,
        "affects_default_ranking": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safe_feature_count": len(PREDICTION_FEATURE_NAMES) + 2,
        "missing_safe_feature_count": len(missing),
        "missing_safe_features": missing,
        "learned_feature_count": learned.get("learned_feature_count"),
        "inference_error": learned.get("inference_error"),
        "error_type": learned.get("error_type"),
        "error_message": learned.get("error_message"),
        "enable_learned_shadow_scores": learned_enabled,
        "enable_gated_shadow_scores": _flag_enabled_any([ENABLE_GATED_ENV], False) if enable_gated_shadow_scores is None else bool(enable_gated_shadow_scores),
        "learned_inference_latency_ms": learned.get("learned_inference_latency_ms"),
        "shadow_scoring_latency_ms": learned.get("shadow_scoring_latency_ms"),
        "artifact_load_latency_ms": learned.get("artifact_load_latency_ms"),
        "timing_source": "time.perf_counter",
        "timing_available": learned.get("timing_available"),
    }
    return metadata


def attach_shadow_scores(
    items: list[dict],
    learned_artifact_dir: str | None = None,
    enable_shadow_scoring: bool | None = None,
    enable_learned_shadow_scores: bool | None = None,
    enable_gated_shadow_scores: bool | None = None,
) -> list[dict]:
    enabled = _shadow_enabled(enable_shadow_scoring)
    if not enabled:
        return [dict(item) for item in items]
    LOGGER.info("segmentation boundary/shape shadow scoring config: %s", shadow_scoring_runtime_config(
        learned_artifact_dir=learned_artifact_dir,
        enable_shadow_scoring=enable_shadow_scoring,
        enable_learned_shadow_scores=enable_learned_shadow_scores,
        enable_gated_shadow_scores=enable_gated_shadow_scores,
    ))
    ranks = rank_boundary_shape_scores(items)
    out = []
    for idx, item in enumerate(items):
        row = dict(item)
        scores, metadata = _shadow_scores_and_metadata(
            row,
            boundary_shape_rank_score=ranks[idx] if idx < len(ranks) else 0.5,
            learned_artifact_dir=learned_artifact_dir,
            enable_learned_shadow_scores=enable_learned_shadow_scores,
            enable_gated_shadow_scores=enable_gated_shadow_scores,
        )
        row["shadow_scores"] = scores
        row["shadow_score_metadata"] = metadata
        out.append(row)
    return out


def _shadow_scores_and_metadata(
    item: dict,
    boundary_shape_rank_score: float | None = None,
    learned_artifact_dir: str | None = None,
    enable_learned_shadow_scores: bool | None = None,
    enable_gated_shadow_scores: bool | None = None,
) -> tuple[dict, dict]:
    calibrated = _component_or_calibrated(item)
    rank_score = _clip(_num(boundary_shape_rank_score, default=0.5))
    current = _clip(_num(item.get("priority_score")))
    learned_enabled = _learned_enabled(enable_learned_shadow_scores)
    gated_enabled = _flag_enabled_any([ENABLE_GATED_ENV], False) if enable_gated_shadow_scores is None else bool(enable_gated_shadow_scores)
    learned = _timed_learned_shadow_result(item, learned_artifact_dir, learned_enabled)
    features = prediction_features(item)
    missing = [name for name in PREDICTION_FEATURE_NAMES if name not in features or features.get(name) is None]
    scores = {
        "boundary_shape_calibrated_score": calibrated,
        "boundary_shape_rank_score": rank_score,
        "learned_boundary_shape_only_score": learned["scores"].get("learned_boundary_shape_only_score"),
        "learned_current_plus_boundary_shape_score": learned["scores"].get("learned_current_plus_boundary_shape_score"),
        "gated_boundary_shape_score": gated_boundary_shape_score(item, rank_score) if gated_enabled else None,
        "gated_current_boundary_score": gated_current_boundary_score(item, current, calibrated) if gated_enabled else None,
        "score_version": _shadow_score_version(),
        "enabled": True,
        "shadow_only": True,
    }
    metadata = {
        "score_version": _shadow_score_version(),
        "artifact_version": learned.get("artifact_version"),
        "artifact_path": learned.get("artifact_path"),
        "artifact_loaded": bool(learned.get("artifact_loaded")),
        "artifact_validation_status": learned.get("artifact_validation_status"),
        "shadow_only": True,
        "affects_default_ranking": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "safe_feature_count": len(PREDICTION_FEATURE_NAMES) + 2,
        "missing_safe_feature_count": len(missing),
        "missing_safe_features": missing,
        "learned_feature_count": learned.get("learned_feature_count"),
        "inference_error": learned.get("inference_error"),
        "error_type": learned.get("error_type"),
        "error_message": learned.get("error_message"),
        "enable_learned_shadow_scores": learned_enabled,
        "enable_gated_shadow_scores": gated_enabled,
        "learned_inference_latency_ms": learned.get("learned_inference_latency_ms"),
        "shadow_scoring_latency_ms": learned.get("shadow_scoring_latency_ms"),
        "artifact_load_latency_ms": learned.get("artifact_load_latency_ms"),
        "timing_source": "time.perf_counter",
        "timing_available": learned.get("timing_available"),
    }
    return scores, metadata


def shadow_scoring_runtime_config(
    learned_artifact_dir: str | None = None,
    enable_shadow_scoring: bool | None = None,
    enable_learned_shadow_scores: bool | None = None,
    enable_gated_shadow_scores: bool | None = None,
) -> dict:
    """Return startup/runtime config for logs and rollout reports."""
    artifact_path = _resolve_artifact_dir(learned_artifact_dir)
    learned_enabled = _learned_enabled(enable_learned_shadow_scores)
    status = "disabled"
    artifact_version = None
    artifact_validation_status = "disabled"
    if learned_enabled:
        if not artifact_path:
            status = "not_configured"
            artifact_validation_status = "not_configured"
        elif not artifact_path.exists():
            status = "missing"
            artifact_validation_status = "missing"
        else:
            status = "configured"
            artifact_validation_status = "configured"
            try:
                import json

                metadata = json.loads((artifact_path / "model_metadata.json").read_text(encoding="utf-8"))
                artifact_version = metadata.get("artifact_version")
                validation_path = artifact_path / "validation_report.json"
                if validation_path.exists():
                    validation = json.loads(validation_path.read_text(encoding="utf-8"))
                    artifact_validation_status = "valid" if validation.get("passed") is True else "invalid"
            except Exception:
                status = "configured_unreadable_metadata"
                artifact_validation_status = "load_failed"
    return {
        "enable_boundary_shape_shadow": _shadow_enabled(enable_shadow_scoring),
        "enable_learned_boundary_shape_shadow": learned_enabled,
        "enable_gated_shadow_scores": _flag_enabled_any([ENABLE_GATED_ENV], False)
        if enable_gated_shadow_scores is None
        else bool(enable_gated_shadow_scores),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "artifact_registry": os.getenv(SEGMENTATION_ARTIFACT_REGISTRY_ENV),
        "artifact_configured_version": os.getenv(SEGMENTATION_ARTIFACT_VERSION_ENV),
        "artifact_version": artifact_version,
        "artifact_load_status": status,
        "artifact_validation_status": artifact_validation_status,
        "fail_open": _fail_open(),
        "shadow_only": True,
        "affects_default_ranking": False,
        "score_version": _shadow_score_version(),
    }


def validate_artifact_for_shadow(artifact_dir: str | None = None) -> dict:
    """Explicit validation helper: schema/load failures raise instead of failing open."""
    path = _resolve_artifact_dir(artifact_dir)
    if not path:
        raise ValueError("learned boundary/shape artifact is not configured")
    result = _learned_shadow_result({}, str(path), True, explicit_validation=True)
    if result.get("artifact_validation_status") != "valid":
        raise ValueError(f"learned artifact validation failed: {result.get('artifact_validation_status')}")
    return result


def gated_boundary_shape_score(item: dict, rank_score: float | None = None) -> float:
    current = _clip(_num(item.get("priority_score")))
    calibrated = _component_or_calibrated(item)
    rank = _clip(_num(rank_score, default=0.5))
    boundary = 0.65 * calibrated + 0.35 * rank
    gate = boundary_shape_gate(item, calibrated)
    return _clip((1.0 - gate) * current + gate * boundary)


def gated_current_boundary_score(item: dict, current: float | None = None, calibrated: float | None = None) -> float:
    current_score = _clip(_num(item.get("priority_score") if current is None else current))
    boundary = _clip(_num(_component_or_calibrated(item) if calibrated is None else calibrated))
    gate = boundary_shape_gate(item, boundary)
    conservative_gate = min(0.45, gate)
    return _clip((1.0 - conservative_gate) * current_score + conservative_gate * boundary)


def boundary_shape_gate(item: dict, calibrated: float | None = None) -> float:
    components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
    boundary = _clip(_num(_component_or_calibrated(item) if calibrated is None else calibrated))
    risk = _clip(_num(components.get("correction_risk_score")))
    uncertainty = _clip(_num(components.get("uncertainty_score")))
    geometry = _clip(_num(components.get("geometry_complexity_score")))
    logit = 4.0 * boundary + 1.25 * risk + 0.75 * uncertainty + 0.50 * geometry - 2.75
    return _clip(1.0 / (1.0 + pow(2.718281828459045, -logit)))


def _component_or_calibrated(item: dict) -> float:
    components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
    value = components.get("boundary_shape_calibrated_score")
    if isinstance(value, (int, float)):
        return _clip(float(value))
    shadow = item.get("shadow_scores") if isinstance(item.get("shadow_scores"), dict) else {}
    value = shadow.get("boundary_shape_calibrated_score")
    if isinstance(value, (int, float)):
        return _clip(float(value))
    return _clip(calibrated_boundary_shape_score(item))


def _empty_learned_score_fields() -> dict[str, float | None]:
    return {
        "learned_boundary_shape_only_score": None,
        "learned_current_plus_boundary_shape_score": None,
    }


def _learned_shadow_result(
    item: dict,
    artifact_dir: str | None,
    enabled: bool,
    explicit_validation: bool = False,
) -> dict:
    result = {
        "scores": _empty_learned_score_fields(),
        "artifact_version": None,
        "artifact_path": None,
        "artifact_loaded": False,
        "artifact_validation_status": "disabled" if not enabled else "not_configured",
        "learned_feature_count": None,
        "inference_error": None,
        "error_type": None,
        "error_message": None,
    }
    if not enabled:
        return result
    root = _resolve_artifact_dir(artifact_dir)
    if not root:
        return result
    result["artifact_path"] = str(root)
    if not root.exists():
        result["artifact_validation_status"] = "missing"
        result["error_type"] = "FileNotFoundError"
        result["error_message"] = str(root)
        if explicit_validation or not _fail_open():
            raise FileNotFoundError(str(root))
        return result
    try:
        import json

        schema_path = root / "feature_schema.json"
        metadata_path = root / "model_metadata.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["artifact_version"] = metadata.get("artifact_version") or schema.get("artifact_version")
        feature_names = _learned_boundary_feature_names(schema)
        result["learned_feature_count"] = len(feature_names) if feature_names else None
        from image_segmentation.benchmark.prediction_feature_scoring import assert_no_leaky_feature_names

        assert_no_leaky_feature_names(feature_names)
        if metadata.get("affects_default_ranking") is True:
            raise ValueError("learned artifact must not affect default ranking")
        if metadata.get("experimental_shadow_only") is not True and metadata.get("shadow_only") is not True:
            raise ValueError("learned artifact metadata must be shadow-only")
        from image_segmentation.benchmark.learn_boundary_shape_fusion import predict_with_artifacts

        raw_scores = predict_with_artifacts(item, str(root))
        result["scores"] = {
            LEARNED_SCORE_KEYS[name]: value
            for name, value in raw_scores.items()
            if name in LEARNED_SCORE_KEYS
        }
        result["scores"] = {**_empty_learned_score_fields(), **result["scores"]}
        result["artifact_loaded"] = True
        result["artifact_validation_status"] = "valid"
        return result
    except ValueError as exc:
        text = str(exc).lower()
        if "schema" in text or "leaky" in text:
            status = "schema_mismatch"
        elif result.get("artifact_version"):
            status = "inference_failed"
        else:
            status = "load_failed"
        return _handle_artifact_error(result, status, exc, explicit_validation)
    except FileNotFoundError as exc:
        return _handle_artifact_error(result, "load_failed", exc, explicit_validation)
    except Exception as exc:
        status = "inference_failed" if result.get("artifact_version") else "load_failed"
        result["inference_error"] = str(exc) if status == "inference_failed" else None
        return _handle_artifact_error(result, status, exc, explicit_validation)


def _timed_learned_shadow_result(
    item: dict,
    artifact_dir: str | None,
    enabled: bool,
    explicit_validation: bool = False,
) -> dict:
    started = time.perf_counter()
    result = _learned_shadow_result(item, artifact_dir, enabled, explicit_validation=explicit_validation)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result["learned_inference_latency_ms"] = elapsed_ms if enabled else None
    result["shadow_scoring_latency_ms"] = elapsed_ms if enabled else None
    result["artifact_load_latency_ms"] = None
    result["timing_available"] = bool(enabled)
    return result


def _handle_artifact_error(result: dict, status: str, exc: Exception, explicit_validation: bool) -> dict:
    result["artifact_validation_status"] = status
    result["error_type"] = type(exc).__name__
    result["error_message"] = str(exc)
    if status == "inference_failed":
        result["inference_error"] = str(exc)
    if explicit_validation or not _fail_open():
        raise exc
    return result


def _learned_boundary_feature_names(schema: dict) -> list[str]:
    experiments = schema.get("experiments") if isinstance(schema.get("experiments"), dict) else {}
    names = ((experiments.get("learned_boundary_shape_only") or {}).get("feature_names") or [])
    return list(names)


def _resolve_artifact_dir(artifact_dir: str | None) -> Path | None:
    explicit = artifact_dir or os.getenv(SEGMENTATION_ARTIFACT_DIR_ENV) or os.getenv(LEARNED_ARTIFACT_ENV)
    if explicit:
        return Path(explicit)
    registry = os.getenv(SEGMENTATION_ARTIFACT_REGISTRY_ENV)
    if not registry:
        return None
    root = Path(registry)
    version = os.getenv(SEGMENTATION_ARTIFACT_VERSION_ENV)
    if not version:
        current = root / "CURRENT"
        if current.exists():
            version = current.read_text(encoding="utf-8").strip()
    return root / version if version else None


def _shadow_score_version() -> str:
    return os.getenv(SEGMENTATION_SHADOW_VERSION_ENV) or SHADOW_SCORE_VERSION


def _shadow_enabled(value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    return _flag_enabled_any([SEGMENTATION_ENABLE_SHADOW_ENV, ENABLE_SHADOW_ENV], False)


def _learned_enabled(value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    return _flag_enabled_any([SEGMENTATION_ENABLE_LEARNED_ENV, ENABLE_LEARNED_ENV], False)


def _fail_open() -> bool:
    return _flag_enabled_any([SEGMENTATION_FAIL_OPEN_ENV], True)


def _flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _flag_enabled_any(names: list[str], default: bool) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if number == number else float(default)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
