from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.datasets.coco import segmentation_to_mask
from image_segmentation.benchmark.evaluation import build_evaluation_report, render_markdown_report
from image_segmentation.benchmark.metrics import boundary_metrics, mask_bbox, prediction_time_boundary_shape_features
from image_segmentation.benchmark.schema import clip_xyxy


ROOT = Path(__file__).resolve().parents[2]
ML_APP_PATH = ROOT / "services" / "ml-backend" / "app.py"
ML_BACKEND_DIR = ROOT / "services" / "ml-backend"
TRAINER_DIR = ROOT / "services" / "trainer"
if str(ML_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND_DIR))
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_quality import evaluate_segmentation_quality  # noqa: E402
from segmentation_uncertainty import evaluate_prompt_stability, generate_bbox_prompt_variants  # noqa: E402
from segmentation_correction_risk import predict_correction_risk  # noqa: E402
from segmentation_corrections import compute_mask_delta_metrics  # noqa: E402
from segmentation_review_queue import build_segmentation_review_queue  # noqa: E402


LABEL_CONFIG = """<View><Image name="image" value="$image"/><BrushLabels name="mask_label" toName="image"><Label value="Object"/></BrushLabels></View>"""
BACKEND_CHOICES = ("bbox_rect", "mobile_sam", "placeholder", "sam", "sam2")
SAM_BACKENDS = {"mobile_sam", "sam", "sam2"}


def run_benchmark(
    manifest_path: str,
    output_dir: str,
    backend: str = "placeholder",
    enable_prompt_stability: bool = False,
    risk_model_dir: str | None = None,
    max_samples: int | None = None,
    resume: bool = False,
) -> dict:
    started = time.monotonic()
    started_at = _now_iso()
    backend_info = resolve_backend(backend)
    rows = _read_jsonl(manifest_path)
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume_predictions, resume_warnings = _load_resume_predictions(output / "predictions.jsonl") if resume else ({}, [])
    resumed_existing_predictions = sum(1 for row in rows if str(row.get("sample_id")) in resume_predictions)
    missing_rows = [row for row in rows if str(row.get("sample_id")) not in resume_predictions]
    app = (
        _load_backend_app(backend_info["backend_resolved"], enable_prompt_stability)
        if backend_info["uses_ml_app"] and missing_rows
        else None
    )
    device_info = _runtime_device_info()
    if (
        backend_info["backend_resolved"] in SAM_BACKENDS
        and str(device_info.get("requested_device") or "").lower() == "cuda"
        and device_info.get("cuda_available") is False
    ):
        raise RuntimeError("IMAGE_SEG_DEVICE=cuda was requested, but torch.cuda.is_available() is false")

    prediction_rows: list[dict] = []
    correction_rows: list[dict] = []
    review_candidates: list[dict] = []
    sample_seconds: list[float] = []
    per_image_seconds: dict[str, list[float]] = {}
    predictions_path = output / "predictions.jsonl"
    corrections_path = output / "correction_delta_dataset.jsonl"
    for index, sample in enumerate(rows, start=1):
        sample_started = time.monotonic()
        sample = _normalize_sample_paths(sample)
        _validate_prompt_box(sample.get("gt_bbox_xyxy"), int(sample["width"]), int(sample["height"]))
        existing_prediction_row = resume_predictions.get(str(sample.get("sample_id")))
        if existing_prediction_row:
            pred_row, correction_row = _rows_from_existing_prediction(
                existing_prediction_row,
                sample,
                backend_info,
                risk_model_dir,
            )
            prediction_rows.append(pred_row)
            correction_rows.append(correction_row)
            review_candidates.append(correction_row)
            _checkpoint_partial_outputs(predictions_path, corrections_path, prediction_rows, correction_rows)
            elapsed_sample = time.monotonic() - sample_started
            sample_seconds.append(elapsed_sample)
            per_image_seconds.setdefault(str(sample.get("image_id")), []).append(elapsed_sample)
            _print_progress(index, len(rows), sample, started, sample_seconds, resumed=True)
            continue
        prediction = _predict_sample(app, sample, backend_info, enable_prompt_stability=enable_prompt_stability)
        pred_row, correction_row = _rows_from_prediction(prediction, sample, backend_info, risk_model_dir)
        prediction_rows.append(pred_row)
        correction_rows.append(correction_row)
        review_candidates.append(correction_row)
        _checkpoint_partial_outputs(predictions_path, corrections_path, prediction_rows, correction_rows)
        elapsed_sample = time.monotonic() - sample_started
        sample_seconds.append(elapsed_sample)
        per_image_seconds.setdefault(str(sample.get("image_id")), []).append(elapsed_sample)
        _print_progress(index, len(rows), sample, started, sample_seconds, resumed=False)

    prediction_rows = _dedupe_rows(prediction_rows, "sample_id")
    correction_rows = _dedupe_rows(correction_rows, "sample_id")
    review_candidates = _dedupe_rows(review_candidates, "sample_id")

    queue_path = output / "review_queue.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(corrections_path, correction_rows)
    queue_result = build_segmentation_review_queue(review_candidates, str(queue_path))
    queue_items = queue_result["items"]
    report = build_evaluation_report(queue_items, correction_rows)
    runtime = _runtime_metadata(
        backend_info,
        device_info,
        rows,
        prediction_rows,
        started_at,
        started,
        sample_seconds,
        per_image_seconds,
        resume,
        resumed_existing_predictions,
    )
    report["backend"] = {
        "backend_requested": backend_info["backend_requested"],
        "backend_resolved": backend_info["backend_resolved"],
    }
    report["runtime"] = runtime
    if resume_warnings:
        report.setdefault("metric_warnings", []).extend(resume_warnings)
        report.setdefault("overall_quality", {}).setdefault("metric_warnings", []).extend(resume_warnings)
    _write_json(output / "evaluation_report.json", report)
    _write_json(output / "runtime_metadata.json", runtime)
    (output / "evaluation_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return {
        "output_dir": str(output),
        "n_samples": len(rows),
        "predictions_path": str(predictions_path),
        "correction_delta_dataset_path": str(corrections_path),
        "review_queue_path": str(queue_path),
        "evaluation_report_path": str(output / "evaluation_report.json"),
        "runtime_metadata_path": str(output / "runtime_metadata.json"),
        "warnings": resume_warnings,
    }


def _rows_from_prediction(
    prediction: dict,
    sample: dict,
    backend_info: dict,
    risk_model_dir: str | None = None,
) -> tuple[dict, dict]:
    _validate_backend_prediction(prediction, backend_info, sample)
    result = prediction["result"][0]
    return _build_rows(prediction, result, sample, backend_info, risk_model_dir)


def _rows_from_existing_prediction(
    prediction_row: dict,
    sample: dict,
    backend_info: dict,
    risk_model_dir: str | None = None,
) -> tuple[dict, dict]:
    prediction = prediction_row.get("prediction") if isinstance(prediction_row.get("prediction"), dict) else {}
    result = prediction_row.get("result") if isinstance(prediction_row.get("result"), dict) else None
    if result is None:
        result = (prediction.get("result") or [{}])[0]
    prediction = dict(prediction)
    prediction.setdefault("result", [result])
    prediction.setdefault("prediction_source", prediction_row.get("prediction_source"))
    prediction.setdefault("model_version", prediction_row.get("model_version"))
    _validate_backend_prediction(prediction, backend_info, sample)
    return _build_rows(prediction, result, sample, backend_info, risk_model_dir)


def _build_rows(
    prediction: dict,
    result: dict,
    sample: dict,
    backend_info: dict,
    risk_model_dir: str | None = None,
) -> tuple[dict, dict]:
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        width = int(sample["width"])
        height = int(sample["height"])
        model_mask = _decode_prediction_mask(result)
        if model_mask.shape != (height, width):
            model_mask = _resize_mask_to_shape(model_mask, width, height)
            model_bbox = mask_bbox(model_mask)
            _rewrite_result_mask(result, model_mask, width, height, model_bbox)
        _refresh_quality_metadata(result, model_mask, sample, backend_info)
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        gt_mask = _gt_mask_for_sample(sample, width, height)
        if model_mask.shape != gt_mask.shape:
            raise ValueError(
                f"prediction and GT mask shape mismatch for {sample.get('sample_id')}: "
                f"prediction={model_mask.shape}, gt={gt_mask.shape}"
            )
        model_bbox = meta.get("mask_bbox") or mask_bbox(model_mask)
        gt_bbox = mask_bbox(gt_mask) or sample["gt_bbox_xyxy"]
        _validate_bbox_in_bounds("prompt_box", meta.get("prompt_box"), width, height)
        if model_bbox is not None:
            _validate_bbox_in_bounds("model_bbox", model_bbox, width, height)
        _validate_bbox_in_bounds("human_bbox", gt_bbox, width, height)
        prediction_features = meta.get("prediction_features") if isinstance(meta.get("prediction_features"), dict) else {}
        delta = compute_mask_delta_metrics(
            model_mask,
            gt_mask,
            width,
            height,
            model_bbox=model_bbox,
            human_bbox=gt_bbox,
        )
        delta["boundary"] = boundary_metrics(model_mask, gt_mask, tolerance_px=int(sample.get("boundary_tolerance_px") or 2))
        pred_row = {
            "benchmark_id": sample.get("benchmark_id"),
            "dataset": sample.get("dataset"),
            "sample_id": sample.get("sample_id"),
            "image_id": sample.get("image_id"),
            "annotation_id": sample.get("annotation_id"),
            "image_path": sample.get("image_path"),
            "category_id": sample.get("category_id"),
            "category_name": sample.get("category_name"),
            "category_frequency": sample.get("category_frequency"),
            "backend_requested": backend_info["backend_requested"],
            "backend_resolved": backend_info["backend_resolved"],
            "prediction_source": prediction.get("prediction_source"),
            "model_version": prediction.get("model_version"),
            "model_config": meta.get("model_config"),
            "prediction": prediction,
            "result": result,
            "mask_quality": meta.get("mask_quality"),
            "prediction_features": prediction_features,
            "review": meta.get("review"),
            "uncertainty": meta.get("uncertainty"),
        }
        correction_row = {
            "source": "public_gt_simulated_human_annotation",
            "benchmark_id": sample.get("benchmark_id"),
            "dataset": sample.get("dataset"),
            "sample_id": sample.get("sample_id"),
            "task_id": sample.get("sample_id"),
            "category_name": sample.get("category_name"),
            "category_frequency": sample.get("category_frequency"),
            "image": sample.get("image_path"),
            "image_id": sample.get("image_id"),
            "annotation_id": sample.get("annotation_id"),
            "prediction_id": result.get("id"),
            "model_version": prediction.get("model_version"),
            "label": sample.get("category_name") or "Object",
            "image_width": width,
            "image_height": height,
            "prompt_bbox": sample.get("gt_bbox_xyxy"),
            "model_mask_bbox": model_bbox,
            "human_mask_bbox": gt_bbox,
            "mask_quality": meta.get("mask_quality"),
            "prediction_features": prediction_features,
            "review": meta.get("review"),
            "uncertainty": meta.get("uncertainty"),
            "difficulty_tags": sample.get("difficulty_tags") or [],
            "boundary_metadata": sample.get("boundary_metadata"),
            "delta": delta,
            "record_status": "ok",
            "skip_reason": None,
        }
        if risk_model_dir:
            try:
                correction_row.update(predict_correction_risk(correction_row, risk_model_dir))
            except Exception as exc:
                correction_row["correction_risk_error"] = str(exc)
        return pred_row, correction_row


def resolve_backend(backend: str) -> dict:
    requested = (backend or "").strip().lower()
    if requested not in BACKEND_CHOICES:
        raise ValueError(
            f"unknown benchmark backend '{backend}'. Choose one of: {', '.join(BACKEND_CHOICES)}. "
            "Use --backend placeholder explicitly only for plumbing smoke tests."
        )
    return {
        "backend_requested": requested,
        "backend_resolved": requested,
        "uses_ml_app": requested in SAM_BACKENDS or requested == "placeholder",
    }


def _predict_sample(
    app: Any,
    sample: dict,
    backend_info: dict,
    enable_prompt_stability: bool = False,
) -> dict:
    if backend_info["backend_resolved"] == "bbox_rect":
        return _predict_bbox_rect(sample, backend_info, enable_prompt_stability=enable_prompt_stability)
    if app is None:
        raise ValueError(f"backend {backend_info['backend_resolved']} requires an ML backend app")
    response = app._predict(
        {
            "label_config": LABEL_CONFIG,
            "tasks": [
                {
                    "id": sample.get("sample_id"),
                    "data": {
                        "image": sample.get("resolved_image_path") or sample.get("image_path"),
                        "bbox": sample.get("gt_bbox_xyxy"),
                    },
                }
            ],
        }
    )
    prediction = response["results"][0]
    result = prediction["result"][0]
    meta = result.setdefault("meta", {})
    meta.setdefault("prompt", "benchmark.gt_bbox")
    meta.setdefault("prompt_box", sample.get("gt_bbox_xyxy"))
    meta.setdefault("prompt_coordinate_system", "pixel_xyxy")
    meta["backend_requested"] = backend_info["backend_requested"]
    meta["backend_resolved"] = backend_info["backend_resolved"]
    return prediction


def _load_backend_app(backend: str, enable_prompt_stability: bool) -> Any:
    local_state = ROOT / "demo_data" / "model_state"
    os.environ.setdefault("MODEL_STATE_PATH", str(local_state / "current_model.json"))
    os.environ.setdefault("TEXT_MODEL_ARTIFACTS_DIR", str(local_state / "text_classifier"))
    os.environ.setdefault("IMAGE_MODEL_ARTIFACTS_DIR", str(local_state / "image_classifier"))
    os.environ.setdefault("IMAGE_SEG_MODEL_ARTIFACTS_DIR", str(local_state / "image_segmentation"))
    os.environ.setdefault("IMAGE_LOCAL_FILES_ROOT", str(ROOT / "demo_data" / "local-files"))
    os.environ["IMAGE_SEG_BACKEND"] = backend
    os.environ["IMAGE_SEG_PROMPT_STABILITY_ENABLED"] = "true" if enable_prompt_stability else "false"
    module_name = f"benchmark_ml_backend_app_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, ML_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _predict_bbox_rect(sample: dict, backend_info: dict, enable_prompt_stability: bool = False) -> dict:
    width = int(sample["width"])
    height = int(sample["height"])
    bbox = clip_xyxy(sample["gt_bbox_xyxy"], width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = _integer_box(bbox, width, height)
    mask[y1:y2, x1:x2] = 1
    mask_box = mask_bbox(mask)
    rle, encoder = _encode_mask(mask)
    result = {
        "id": f"{sample.get('sample_id', 'sample')}_mask",
        "from_name": "mask_label",
        "to_name": "image",
        "type": "brushlabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {"format": "rle", "rle": rle, "brushlabels": [sample.get("category_name") or "Object"]},
        "meta": {
            "prediction_source": "bbox-rect-benchmark-baseline",
            "backend": "bbox_rect",
            "backend_requested": backend_info["backend_requested"],
            "backend_resolved": backend_info["backend_resolved"],
            "rle_encoder": encoder,
            "prompt": "benchmark.gt_bbox",
            "prompt_box": bbox,
            "prompt_coordinate_system": "pixel_xyxy",
            "mask_bbox": mask_box,
        },
    }
    _refresh_quality_metadata(result, mask, sample, backend_info)
    if enable_prompt_stability:
        result["meta"]["uncertainty"] = _bbox_rect_uncertainty(sample, mask)
    return {
        "model_version": "benchmark-bbox-rect-v1",
        "score": 1.0,
        "result": [result],
        "prediction_source": "bbox-rect-benchmark-baseline",
        "confidence": {
            "prediction_source": "bbox-rect-benchmark-baseline",
            "confidence": 1.0,
            "confidence_bucket": "high",
            "backend": "bbox_rect",
            "backend_requested": backend_info["backend_requested"],
            "backend_resolved": backend_info["backend_resolved"],
            "prompt": "benchmark.gt_bbox",
            "prompt_box": bbox,
            "prompt_coordinate_system": "pixel_xyxy",
            "model_config": {"backend": "bbox_rect"},
        },
    }


def _bbox_rect_uncertainty(sample: dict, original_mask: np.ndarray) -> dict:
    width = int(sample["width"])
    height = int(sample["height"])
    variants = generate_bbox_prompt_variants(sample["gt_bbox_xyxy"], width, height)
    masks = [original_mask]
    for variant in variants[1:]:
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = _integer_box(variant["bbox"], width, height)
        mask[y1:y2, x1:x2] = 1
        masks.append(mask)
    uncertainty = evaluate_prompt_stability(masks, prompt_variants=variants, image_width=width, image_height=height)[
        "uncertainty"
    ]
    if "disagreement_area" not in uncertainty and uncertainty.get("disagreement_area_ratio") is not None:
        uncertainty["disagreement_area"] = int(round(float(uncertainty["disagreement_area_ratio"]) * width * height))
    uncertainty["uncertainty_reason"] = uncertainty.get("reason") or []
    return uncertainty


def _validate_backend_prediction(prediction: dict, backend_info: dict, sample: dict) -> None:
    source = str(prediction.get("prediction_source") or "")
    confidence = prediction.get("confidence") if isinstance(prediction.get("confidence"), dict) else {}
    result = (prediction.get("result") or [{}])[0]
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if backend_info["backend_resolved"] in SAM_BACKENDS:
        if source == "placeholder-image-segmentation" or confidence.get("fallback") or meta.get("fallback"):
            error = confidence.get("backend_error") or meta.get("backend_error") or "backend fell back to placeholder"
            raise RuntimeError(
                f"{backend_info['backend_resolved']} benchmark backend failed for {sample.get('sample_id')}: {error}"
            )
    meta["backend_requested"] = backend_info["backend_requested"]
    meta["backend_resolved"] = backend_info["backend_resolved"]
    if backend_info["backend_resolved"] in SAM_BACKENDS:
        meta.setdefault("model_config", _model_config_for_backend(backend_info["backend_resolved"]))


def _model_config_for_backend(backend: str) -> dict:
    return {
        "backend": backend,
        "model_type": os.getenv("IMAGE_SEG_MODEL_TYPE", ""),
        "checkpoint": os.getenv("IMAGE_SEG_CHECKPOINT", ""),
        "model_id": os.getenv("IMAGE_SEG_MODEL_ID", ""),
        "device": os.getenv("IMAGE_SEG_DEVICE", ""),
    }


def _refresh_quality_metadata(result: dict, mask: np.ndarray, sample: dict, backend_info: dict) -> None:
    meta = result.setdefault("meta", {})
    width = int(sample["width"])
    height = int(sample["height"])
    prompt_bbox = meta.get("prompt_box") or sample.get("gt_bbox_xyxy")
    if prompt_bbox is None and meta.get("prompt") == "benchmark.gt_bbox":
        raise ValueError(f"missing benchmark prompt bbox for {sample.get('sample_id')}")
    prompt_bbox = clip_xyxy(prompt_bbox, width, height)
    model_bbox = mask_bbox(mask)
    rle = (result.get("value") or {}).get("rle")
    quality = evaluate_segmentation_quality(
        mask,
        width,
        height,
        prompt_bbox=prompt_bbox,
        mask_bbox=model_bbox,
        rle_length=len(rle) if isinstance(rle, list) else None,
        backend_metadata=meta,
    )
    prediction_features = prediction_time_boundary_shape_features(mask, width=width, height=height)
    meta.update(
        {
            "backend_requested": backend_info["backend_requested"],
            "backend_resolved": backend_info["backend_resolved"],
            "prompt": "benchmark.gt_bbox",
            "prompt_box": prompt_bbox,
            "prompt_coordinate_system": "pixel_xyxy",
            "mask_bbox": model_bbox,
            "prediction_features": prediction_features,
        }
    )
    uncertainty = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else None
    if uncertainty is not None:
        uncertainty.setdefault("uncertainty_reason", uncertainty.get("reason") or [])
        if "disagreement_area" not in uncertainty and uncertainty.get("disagreement_area_ratio") is not None:
            uncertainty["disagreement_area"] = int(
                round(float(uncertainty["disagreement_area_ratio"]) * width * height)
            )
    meta.update(quality)
    meta.setdefault("mask_quality", {}).setdefault("prediction_time_boundary_shape", prediction_features)
    result["original_width"] = width
    result["original_height"] = height


def _decode_prediction_mask(result: dict) -> np.ndarray:
    width = int(result.get("original_width") or 0)
    height = int(result.get("original_height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("prediction result must include positive original_width/original_height")
    value = result.get("value") if isinstance(result.get("value"), dict) else {}
    rle = value.get("rle")
    encoder = (result.get("meta") or {}).get("rle_encoder")
    if encoder == "fallback-minimal":
        return _decode_minimal_rle(rle, width, height)
    try:
        from label_studio_converter.brush import decode_rle

        decoded = decode_rle(rle)
        rgba = decoded.reshape((height, width, 4))
        return ((rgba[:, :, 3] > 0) | (rgba[:, :, :3].max(axis=2) > 0)).astype(np.uint8)
    except Exception:
        return _decode_minimal_rle(rle, width, height)


def _decode_minimal_rle(rle: Any, width: int, height: int) -> np.ndarray:
    if not isinstance(rle, list):
        raise ValueError("prediction RLE must be a list")
    values: list[int] = []
    current = 0
    for count in rle:
        values.extend([current] * int(count))
        current = 1 - current
    total = width * height
    if len(values) < total:
        values.extend([0] * (total - len(values)))
    return np.asarray(values[:total], dtype=np.uint8).reshape((height, width))


def _rewrite_result_mask(result: dict, mask: np.ndarray, width: int, height: int, model_bbox: list[float] | None) -> None:
    rle, encoder = _encode_mask(mask)
    result["original_width"] = width
    result["original_height"] = height
    result.setdefault("value", {})["rle"] = rle
    result.setdefault("value", {})["format"] = "rle"
    result.setdefault("meta", {})["rle_encoder"] = encoder
    result["meta"]["mask_bbox"] = model_bbox


def _encode_mask(mask: np.ndarray) -> tuple[list[int], str]:
    try:
        from label_studio_converter.brush import mask2rle

        return mask2rle((np.asarray(mask) > 0).astype(np.uint8)), "label-studio-converter"
    except Exception:
        return _minimal_rle(mask), "fallback-minimal"


def _minimal_rle(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    length = 0
    for value in (np.asarray(mask) > 0).astype(np.uint8).reshape(-1):
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1
    runs.append(length)
    return runs


def _resize_mask_to_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image

        image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
        return (np.asarray(resized) > 0).astype(np.uint8)
    except Exception as exc:
        raise ValueError(f"could not resize prediction mask to {width}x{height}: {exc}") from exc


def _gt_mask_for_sample(sample: dict, width: int, height: int) -> np.ndarray:
    mask_path = sample.get("mask_path") or ((sample.get("metadata") or {}).get("mask_path") if isinstance(sample.get("metadata"), dict) else None)
    if mask_path:
        path = Path(str(mask_path))
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"benchmark GT mask file not found: {mask_path}")
        from PIL import Image

        with Image.open(resolved) as mask_image:
            arr = np.asarray(mask_image.convert("L"))
        if arr.shape != (height, width):
            raise ValueError(
                f"GT mask shape mismatch for {sample.get('sample_id')}: manifest={(height, width)}, mask={arr.shape}"
            )
        return (arr > 0).astype(np.uint8)
    return segmentation_to_mask(sample.get("gt_segmentation"), width, height, int(sample.get("gt_iscrowd") or 0))


def _integer_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return left, top, right, bottom


def _normalize_sample_paths(sample: dict) -> dict:
    row = dict(sample)
    path = Path(str(row.get("image_path") or ""))
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"benchmark image file not found: {row.get('image_path')}")
    _validate_image_size(resolved, int(row["width"]), int(row["height"]))
    row["resolved_image_path"] = str(resolved)
    return row


def _validate_image_size(path: Path, width: int, height: int) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            actual = (int(image.width), int(image.height))
    except Exception as exc:
        raise ValueError(f"could not read benchmark image size for {path}: {exc}") from exc
    if actual != (width, height):
        raise ValueError(f"image size mismatch for {path}: manifest={width}x{height}, file={actual[0]}x{actual[1]}")


def _validate_prompt_box(bbox: Any, width: int, height: int) -> None:
    _validate_bbox_in_bounds("prompt_box", bbox, width, height)


def _validate_bbox_in_bounds(name: str, bbox: Any, width: int, height: int, tolerance: float = 1e-3) -> None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"{name} must be a 4-item bbox")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} has non-positive area: {bbox}")
    if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
        raise ValueError(f"{name} out of image bounds {width}x{height}: {bbox}")


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_resume_predictions(path: Path) -> tuple[dict[str, dict], list[str]]:
    if not path.exists():
        return {}, []
    rows: dict[str, dict] = {}
    warnings: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f"resume_predictions_json_decode_warning_line_{line_number}")
                    continue
                sample_id = row.get("sample_id")
                if not sample_id:
                    warnings.append(f"resume_predictions_missing_sample_id_line_{line_number}")
                    continue
                rows.setdefault(str(sample_id), row)
    except OSError as exc:
        warnings.append(f"resume_predictions_read_warning: {exc}")
    return rows, warnings


def _dedupe_rows(rows: list[dict], key: str) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        out.append(row)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _checkpoint_partial_outputs(
    predictions_path: Path,
    corrections_path: Path,
    prediction_rows: list[dict],
    correction_rows: list[dict],
) -> None:
    _write_jsonl(predictions_path, _dedupe_rows(prediction_rows, "sample_id"))
    _write_jsonl(corrections_path, _dedupe_rows(correction_rows, "sample_id"))


def _print_progress(
    index: int,
    total: int,
    sample: dict,
    started: float,
    sample_seconds: list[float],
    resumed: bool,
) -> None:
    action = "resumed" if resumed else "predicted"
    elapsed = time.monotonic() - started
    avg = statistics.mean(sample_seconds) if sample_seconds else 0.0
    remaining = max(0, total - index)
    eta = remaining * avg if avg else None
    eta_text = f"{eta:.1f}" if eta is not None else "n/a"
    cuda = _cuda_memory_mb()
    cuda_text = ""
    if cuda:
        cuda_text = f" cuda_allocated_mb={cuda.get('allocated_mb'):.1f} cuda_reserved_mb={cuda.get('reserved_mb'):.1f}"
    print(
        f"[{action}] {index}/{total} sample_id={sample.get('sample_id')} image={sample.get('image_path')} "
        f"elapsed_s={elapsed:.1f} avg_s_per_sample={avg:.3f} eta_s={eta_text}{cuda_text}",
        flush=True,
    )


def _runtime_metadata(
    backend_info: dict,
    device_info: dict,
    requested_rows: list[dict],
    completed_rows: list[dict],
    started_at: str,
    started: float,
    sample_seconds: list[float],
    per_image_seconds: dict[str, list[float]],
    resume_enabled: bool,
    resumed_existing_predictions: int,
) -> dict:
    elapsed = time.monotonic() - started
    image_totals = [sum(values) for values in per_image_seconds.values()]
    peak = _cuda_peak_memory_mb()
    return {
        "backend_requested": backend_info["backend_requested"],
        "backend_resolved": backend_info["backend_resolved"],
        "requested_device": device_info.get("requested_device"),
        "resolved_device": device_info.get("resolved_device"),
        "cuda_available": device_info.get("cuda_available"),
        "cuda_device_name": device_info.get("cuda_device_name"),
        "n_samples_requested": len(requested_rows),
        "n_samples_completed": len(completed_rows),
        "n_unique_images": len({str(row.get("image_id")) for row in completed_rows}),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "elapsed_seconds": elapsed,
        "seconds_per_sample_mean": statistics.mean(sample_seconds) if sample_seconds else None,
        "seconds_per_sample_median": statistics.median(sample_seconds) if sample_seconds else None,
        "seconds_per_image_mean": statistics.mean(image_totals) if image_totals else None,
        "resume_enabled": bool(resume_enabled),
        "resumed_existing_predictions": int(resumed_existing_predictions),
        "peak_cuda_memory_allocated_mb": peak.get("allocated_mb") if peak else None,
        "peak_cuda_memory_reserved_mb": peak.get("reserved_mb") if peak else None,
    }


def _runtime_device_info() -> dict:
    requested = os.getenv("IMAGE_SEG_DEVICE", "cpu").strip() or "cpu"
    info = {
        "requested_device": requested,
        "resolved_device": requested,
        "cuda_available": None,
        "cuda_device_name": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if requested.lower() == "cuda":
            if not info["cuda_available"]:
                info["resolved_device"] = None
                return info
            index = torch.cuda.current_device()
            info["resolved_device"] = "cuda"
            info["cuda_device_name"] = torch.cuda.get_device_name(index)
    except Exception:
        if requested.lower() == "cuda":
            info["resolved_device"] = None
    return info


def _cuda_memory_mb() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        return {
            "allocated_mb": float(torch.cuda.memory_allocated(index) / (1024 * 1024)),
            "reserved_mb": float(torch.cuda.memory_reserved(index) / (1024 * 1024)),
        }
    except Exception:
        return None


def _cuda_peak_memory_mb() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        return {
            "allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024 * 1024)),
            "reserved_mb": float(torch.cuda.max_memory_reserved(index) / (1024 * 1024)),
        }
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Benchmark v0.1 on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", required=True, choices=BACKEND_CHOICES)
    parser.add_argument("--enable-prompt-stability", default="false")
    parser.add_argument("--risk-model-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", default="false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    enabled = str(args.enable_prompt_stability).strip().lower() in {"1", "true", "yes", "on"}
    resume = str(args.resume).strip().lower() in {"1", "true", "yes", "on"}
    try:
        summary = run_benchmark(
            args.manifest,
            args.output_dir,
            backend=args.backend,
            enable_prompt_stability=enabled,
            risk_model_dir=args.risk_model_dir,
            max_samples=args.max_samples,
            resume=resume,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] benchmark complete: {summary['n_samples']} samples")
    print(f"output_dir={summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
