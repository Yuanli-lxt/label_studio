from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from image_segmentation.benchmark.datasets.coco import segmentation_to_mask
from image_segmentation.benchmark.evaluation import build_evaluation_report, render_markdown_report
from image_segmentation.benchmark.metrics import mask_bbox


ROOT = Path(__file__).resolve().parents[2]
ML_APP_PATH = ROOT / "services" / "ml-backend" / "app.py"
TRAINER_DIR = ROOT / "services" / "trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from segmentation_correction_risk import predict_correction_risk  # noqa: E402
from segmentation_corrections import compute_mask_delta_metrics  # noqa: E402
from segmentation_review_queue import build_segmentation_review_queue  # noqa: E402


LABEL_CONFIG = """<View><Image name="image" value="$image"/><BrushLabels name="mask_label" toName="image"><Label value="Object"/></BrushLabels></View>"""


def run_benchmark(
    manifest_path: str,
    output_dir: str,
    backend: str = "placeholder",
    enable_prompt_stability: bool = False,
    risk_model_dir: str | None = None,
) -> dict:
    rows = _read_jsonl(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    app = _load_backend_app(backend, enable_prompt_stability)

    prediction_rows: list[dict] = []
    correction_rows: list[dict] = []
    review_candidates: list[dict] = []
    for sample in rows:
        prediction = _predict_sample(app, sample)
        result = prediction["result"][0]
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        width = int(sample["width"])
        height = int(sample["height"])
        model_mask = _decode_prediction_mask(result, width, height)
        gt_mask = segmentation_to_mask(sample.get("gt_segmentation"), width, height, int(sample.get("gt_iscrowd") or 0))
        model_bbox = meta.get("mask_bbox") or mask_bbox(model_mask)
        gt_bbox = mask_bbox(gt_mask) or sample["gt_bbox_xyxy"]
        delta = compute_mask_delta_metrics(
            model_mask,
            gt_mask,
            width,
            height,
            model_bbox=model_bbox,
            human_bbox=gt_bbox,
        )
        pred_row = {
            "benchmark_id": sample.get("benchmark_id"),
            "dataset": sample.get("dataset"),
            "sample_id": sample.get("sample_id"),
            "image_id": sample.get("image_id"),
            "annotation_id": sample.get("annotation_id"),
            "image_path": sample.get("image_path"),
            "category_id": sample.get("category_id"),
            "category_name": sample.get("category_name"),
            "prediction": prediction,
            "result": result,
            "mask_quality": meta.get("mask_quality"),
            "review": meta.get("review"),
            "uncertainty": meta.get("uncertainty"),
        }
        correction_row = {
            "source": "public_gt_simulated_human_annotation",
            "benchmark_id": sample.get("benchmark_id"),
            "dataset": sample.get("dataset"),
            "sample_id": sample.get("sample_id"),
            "task_id": sample.get("sample_id"),
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
            "review": meta.get("review"),
            "uncertainty": meta.get("uncertainty"),
            "difficulty_tags": sample.get("difficulty_tags") or [],
            "delta": delta,
            "record_status": "ok",
            "skip_reason": None,
        }
        if risk_model_dir:
            try:
                correction_row.update(predict_correction_risk(correction_row, risk_model_dir))
            except Exception as exc:
                correction_row["correction_risk_error"] = str(exc)
        prediction_rows.append(pred_row)
        correction_rows.append(correction_row)
        review_candidates.append(correction_row)

    predictions_path = output / "predictions.jsonl"
    corrections_path = output / "correction_delta_dataset.jsonl"
    queue_path = output / "review_queue.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    _write_jsonl(corrections_path, correction_rows)
    queue_result = build_segmentation_review_queue(review_candidates, str(queue_path))
    queue_items = queue_result["items"]
    report = build_evaluation_report(queue_items, correction_rows)
    _write_json(output / "evaluation_report.json", report)
    (output / "evaluation_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return {
        "output_dir": str(output),
        "n_samples": len(rows),
        "predictions_path": str(predictions_path),
        "correction_delta_dataset_path": str(corrections_path),
        "review_queue_path": str(queue_path),
        "evaluation_report_path": str(output / "evaluation_report.json"),
    }


def _predict_sample(app: Any, sample: dict) -> dict:
    response = app._predict(
        {
            "label_config": LABEL_CONFIG,
            "tasks": [
                {
                    "id": sample.get("sample_id"),
                    "data": {
                        "image": sample.get("image_path"),
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
    mask_quality = meta.get("mask_quality")
    if isinstance(mask_quality, dict):
        mask_quality.setdefault("prompt_bbox", sample.get("gt_bbox_xyxy"))
    return prediction


def _load_backend_app(backend: str, enable_prompt_stability: bool) -> Any:
    local_state = ROOT / "demo_data" / "model_state"
    os.environ.setdefault("MODEL_STATE_PATH", str(local_state / "current_model.json"))
    os.environ.setdefault("TEXT_MODEL_ARTIFACTS_DIR", str(local_state / "text_classifier"))
    os.environ.setdefault("IMAGE_MODEL_ARTIFACTS_DIR", str(local_state / "image_classifier"))
    os.environ.setdefault("IMAGE_SEG_MODEL_ARTIFACTS_DIR", str(local_state / "image_segmentation"))
    os.environ.setdefault("IMAGE_LOCAL_FILES_ROOT", str(ROOT / "demo_data" / "local-files"))
    os.environ["IMAGE_SEG_BACKEND"] = _normalize_backend(backend)
    os.environ["IMAGE_SEG_PROMPT_STABILITY_ENABLED"] = "true" if enable_prompt_stability else "false"
    module_name = f"benchmark_ml_backend_app_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, ML_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _normalize_backend(value: str) -> str:
    raw = (value or "placeholder").strip().lower()
    if raw in {"mobile_sam_or_existing_backend", "existing", "auto"}:
        return os.getenv("IMAGE_SEG_BACKEND", "placeholder")
    return raw


def _decode_prediction_mask(result: dict, width: int, height: int) -> np.ndarray:
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


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Benchmark v0.1 on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", default="placeholder")
    parser.add_argument("--enable-prompt-stability", default="false")
    parser.add_argument("--risk-model-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    enabled = str(args.enable_prompt_stability).strip().lower() in {"1", "true", "yes", "on"}
    try:
        summary = run_benchmark(
            args.manifest,
            args.output_dir,
            backend=args.backend,
            enable_prompt_stability=enabled,
            risk_model_dir=args.risk_model_dir,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[OK] benchmark complete: {summary['n_samples']} samples")
    print(f"output_dir={summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
