from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.run_benchmark import BACKEND_CHOICES, ROOT


ML_APP_PATH = ROOT / "services" / "ml-backend" / "app.py"
DEFAULT_MOBILE_SAM_CHECKPOINT = ROOT / "models" / "mobilesam" / "mobile_sam.pt"


def preflight_backend(
    backend: str,
    checkpoint: str | None = None,
    device: str | None = None,
    construct_predictor: bool = True,
) -> dict:
    requested = (backend or "").strip().lower()
    result: dict[str, Any] = {
        "backend": requested,
        "status": "failed",
        "checkpoint": checkpoint or os.getenv("IMAGE_SEG_CHECKPOINT") or None,
        "checkpoint_path": checkpoint or os.getenv("IMAGE_SEG_CHECKPOINT") or None,
        "requested_device": device or os.getenv("IMAGE_SEG_DEVICE", "cpu"),
        "device": device or os.getenv("IMAGE_SEG_DEVICE", "cpu"),
        "resolved_device": None,
        "cuda_available": None,
        "cuda_device_name": None,
        "cuda_memory": None,
        "model_class": None,
        "predictor_class": None,
        "model_version": None,
        "model_on_cuda": None,
        "error": None,
    }
    if requested not in BACKEND_CHOICES:
        result["error"] = f"unknown backend '{backend}'. Choose one of: {', '.join(BACKEND_CHOICES)}"
        return result
    if requested not in {"mobile_sam", "sam", "sam2"}:
        result["status"] = "ok"
        result["model_class"] = requested
        result["resolved_device"] = "cpu"
        return result

    normalized_backend = "mobilesam" if requested == "mobile_sam" else requested
    checkpoint_path = Path(
        checkpoint
        or os.getenv("IMAGE_SEG_CHECKPOINT")
        or (str(DEFAULT_MOBILE_SAM_CHECKPOINT) if requested == "mobile_sam" else "")
    )
    result["checkpoint_path"] = str(checkpoint_path) if str(checkpoint_path) else None
    result["checkpoint"] = result["checkpoint_path"]
    if requested in {"mobile_sam", "sam"} and (not str(checkpoint_path) or not checkpoint_path.exists()):
        result["error"] = f"IMAGE_SEG_CHECKPOINT does not exist: {checkpoint_path}"
        return result
    if device:
        os.environ["IMAGE_SEG_DEVICE"] = device
    if checkpoint_path:
        os.environ["IMAGE_SEG_CHECKPOINT"] = str(checkpoint_path)
    os.environ["IMAGE_SEG_BACKEND"] = normalized_backend

    dependency_error = _dependency_error(normalized_backend)
    if dependency_error:
        result["error"] = dependency_error
        return result
    device_info = _device_info(result["requested_device"])
    result.update(device_info)
    device_error = device_info.get("error")
    if device_error:
        result["error"] = device_error
        return result
    if not construct_predictor:
        result["status"] = "ok"
        return result
    try:
        app = _load_backend_app()
        bundle = app._load_image_segmentation_predictor(normalized_backend)
        predictor = bundle.get("predictor") if isinstance(bundle, dict) else None
        model = getattr(predictor, "model", None)
        result["predictor_class"] = type(predictor).__name__ if predictor is not None else None
        result["model_class"] = type(model).__name__ if model is not None else result["predictor_class"]
        result["model_version"] = getattr(predictor, "model_version", None) or getattr(predictor, "name", None)
        result["model_on_cuda"] = _model_on_cuda(model)
        if str(result.get("requested_device") or "").lower() == "cuda" and result["model_on_cuda"] is False:
            result["error"] = "IMAGE_SEG_DEVICE=cuda was requested, but model parameters are not on cuda"
            return result
        result["status"] = "ok"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _dependency_error(backend: str) -> str | None:
    if backend == "mobilesam":
        try:
            import mobile_sam  # noqa: F401
        except Exception as exc:
            return f"mobile_sam dependency is not importable: {exc}"
    if backend == "sam":
        try:
            import segment_anything  # noqa: F401
        except Exception as exc:
            return f"segment_anything dependency is not importable: {exc}"
    if backend == "sam2":
        try:
            import sam2  # noqa: F401
        except Exception as exc:
            return f"sam2 dependency is not importable: {exc}"
    return None


def _device_info(device: str | None) -> dict:
    requested = str(device or "cpu").lower()
    info = {
        "requested_device": device or "cpu",
        "resolved_device": requested,
        "cuda_available": None,
        "cuda_device_name": None,
        "cuda_memory": None,
        "error": None,
    }
    if requested != "cuda":
        return info
    try:
        import torch

        available = bool(torch.cuda.is_available())
        info["cuda_available"] = available
        if not available:
            info["error"] = "IMAGE_SEG_DEVICE=cuda was requested, but torch.cuda.is_available() is false"
            return info
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info["resolved_device"] = "cuda"
        info["cuda_device_name"] = torch.cuda.get_device_name(index)
        info["cuda_memory"] = {
            "total_memory": int(props.total_memory),
            "allocated": int(torch.cuda.memory_allocated(index)),
            "reserved": int(torch.cuda.memory_reserved(index)),
        }
    except Exception as exc:
        info["error"] = f"could not validate cuda device: {exc}"
    return info


def _model_on_cuda(model: Any) -> bool | None:
    if model is None:
        return None
    try:
        parameter = next(model.parameters())
        return str(parameter.device).startswith("cuda")
    except StopIteration:
        return None
    except Exception:
        device = getattr(model, "device", None)
        return str(device).startswith("cuda") if device is not None else None


def _load_backend_app() -> Any:
    module_name = f"benchmark_backend_preflight_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, ML_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight a benchmark segmentation backend.")
    parser.add_argument("--backend", required=True, choices=BACKEND_CHOICES)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--skip-construct-predictor", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = preflight_backend(
        args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        construct_predictor=not args.skip_construct_predictor,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"backend={result['backend']}")
        print(f"checkpoint={result.get('checkpoint')}")
        print(f"requested_device={result.get('requested_device')}")
        print(f"resolved_device={result.get('resolved_device')}")
        print(f"cuda_available={result.get('cuda_available')}")
        print(f"cuda_device_name={result.get('cuda_device_name')}")
        print(f"model_class={result.get('model_class')}")
        print(f"predictor_class={result.get('predictor_class')}")
        print(f"model_version={result.get('model_version')}")
        print(f"model_on_cuda={result.get('model_on_cuda')}")
        print(f"status={result['status']}")
        if result.get("error"):
            print(f"error={result['error']}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
