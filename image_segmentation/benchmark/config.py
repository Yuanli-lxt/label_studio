from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("benchmark config must be a mapping")
    return data


def enabled_coco_config(config: dict) -> dict | None:
    datasets = config.get("datasets") if isinstance(config.get("datasets"), dict) else {}
    coco = datasets.get("coco") if isinstance(datasets.get("coco"), dict) else {}
    return coco if coco.get("enabled") is True else None

