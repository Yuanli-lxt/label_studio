from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def make_eval_splits(
    delta_dataset: str,
    output_dir: str,
    strategy: str = "stratified",
    train_ratio: float = 0.67,
    random_seed: int = 42,
) -> dict:
    records = _read_jsonl(delta_dataset)
    warnings: list[str] = []
    ratio = max(0.05, min(0.95, float(train_ratio)))
    rows = [(idx, row, _is_major(row)) for idx, row in enumerate(records)]
    positives = [item for item in rows if item[2]]
    negatives = [item for item in rows if not item[2]]
    rng = random.Random(int(random_seed))
    rng.shuffle(positives)
    rng.shuffle(negatives)

    if strategy != "stratified":
        warnings.append(f"unsupported_strategy_{strategy}; using random")
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, round(len(shuffled) * ratio))) if len(shuffled) > 1 else len(shuffled)
        train = shuffled[:n_train]
        eval_rows = shuffled[n_train:]
    else:
        if 0 < len(positives) < 2:
            warnings.append("too_few_positive_samples_for_stable_stratified_split")
        train_pos_count = _split_count(len(positives), ratio)
        train_neg_count = _split_count(len(negatives), ratio)
        train = positives[:train_pos_count] + negatives[:train_neg_count]
        eval_rows = positives[train_pos_count:] + negatives[train_neg_count:]
        rng.shuffle(train)
        rng.shuffle(eval_rows)

    if positives and not any(item[2] for item in train):
        warnings.append("train split has no positive samples")
    if positives and not any(item[2] for item in eval_rows):
        warnings.append("eval split has no positive samples; review ranking metrics are not informative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_records = [item[1] for item in train]
    eval_records = [item[1] for item in eval_rows]
    train_ids = [_record_id(item[1], item[0]) for item in train]
    eval_ids = [_record_id(item[1], item[0]) for item in eval_rows]

    _write_json(output / "train_ids.json", train_ids)
    _write_json(output / "eval_ids.json", eval_ids)
    _write_jsonl(output / "train_delta_dataset.jsonl", train_records)
    _write_jsonl(output / "eval_delta_dataset.jsonl", eval_records)
    metadata = {
        "n_total": len(records),
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        "total_positive_rate": _positive_rate(records),
        "train_positive_rate": _positive_rate(train_records),
        "eval_positive_rate": _positive_rate(eval_records),
        "strategy": strategy,
        "random_seed": int(random_seed),
        "train_ratio": ratio,
        "warnings": warnings,
    }
    _write_json(output / "split_metadata.json", metadata)
    return metadata


def _split_count(n: int, ratio: float) -> int:
    if n <= 1:
        return n
    return max(1, min(n - 1, int(round(n * ratio))))


def _record_id(record: dict, fallback: int) -> str:
    for key in ("sample_id", "task_id", "prediction_id", "annotation_id", "image_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"row_{fallback}"


def _is_major(record: dict) -> bool:
    delta = record.get("delta") if isinstance(record.get("delta"), dict) else {}
    return delta.get("major_correction") is True or str(delta.get("correction_severity") or "").lower() == "major"


def _positive_rate(records: list[dict]) -> float | None:
    return float(sum(1 for row in records if _is_major(row)) / len(records)) if records else None


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create train/eval split manifests for correction delta datasets.")
    parser.add_argument("--delta-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", default="stratified")
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = make_eval_splits(
        args.delta_dataset,
        args.output_dir,
        strategy=args.strategy,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
    )
    print(f"[OK] wrote split: train={metadata['n_train']} eval={metadata['n_eval']}")
    for warning in metadata.get("warnings") or []:
        print(f"[WARN] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
