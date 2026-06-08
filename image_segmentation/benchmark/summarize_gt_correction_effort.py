from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


EFFORT_ORDER = ["No fix", "Minor fix", "Major fix", "Redo"]
SIZE_BUCKETS = ["small", "medium", "large"]


def correction_effort_from_iou(iou: float | int | None) -> str | None:
    if iou is None:
        return None
    value = float(iou)
    if value >= 0.90:
        return "No fix"
    if value >= 0.75:
        return "Minor fix"
    if value >= 0.50:
        return "Major fix"
    return "Redo"


def size_bucket_from_area(area_px: float | int | None) -> str | None:
    if area_px is None:
        return None
    area = float(area_px)
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def summarize_gt_correction_effort(runs: list[str], output_dir: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_inputs = [_parse_run_spec(spec) for spec in runs]
    skipped = 0
    for run_name, path in run_inputs:
        for row in _read_jsonl(path):
            normalized = _normalize_row(row, run_name)
            if normalized is None:
                skipped += 1
                continue
            rows.append(normalized)

    datasets: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        datasets[dataset] = _summarize_rows(dataset_rows)
        datasets[dataset]["size_buckets"] = {
            bucket: _summarize_rows([row for row in dataset_rows if row["size_bucket"] == bucket])
            for bucket in SIZE_BUCKETS
        }

    summary = {
        "method": {
            "correction_effort": {
                "No fix": "IoU >= 0.90",
                "Minor fix": "0.75 <= IoU < 0.90",
                "Major fix": "0.50 <= IoU < 0.75",
                "Redo": "IoU < 0.50",
            },
            "size_buckets": {
                "small": "GT/human area < 32^2 px",
                "medium": "32^2 <= GT/human area < 96^2 px",
                "large": "GT/human area >= 96^2 px",
            },
            "leakage_note": "GT-derived correction effort is an offline evaluation label and must not be used as a prediction-time risk feature.",
        },
        "runs": [{"name": name, "path": str(path)} for name, path in run_inputs],
        "total": _summarize_rows(rows),
        "datasets": datasets,
        "skipped_rows": skipped,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gt_derived_correction_effort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "gt_derived_correction_effort_summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GT-Derived Correction Effort Summary",
        "",
        "Correction effort is derived from model-vs-GT IoU and is used only as an offline evaluation label.",
        "",
        "## Method",
        "",
        "- No fix: IoU >= 0.90",
        "- Minor fix: 0.75 <= IoU < 0.90",
        "- Major fix: 0.50 <= IoU < 0.75",
        "- Redo: IoU < 0.50",
        "- Size buckets use GT/human mask area: small < 32^2 px, medium < 96^2 px, large >= 96^2 px.",
        "",
        "## Key Findings",
        "",
        *_key_finding_lines(summary),
        "",
        "## Dataset Summary",
        "",
        f"Overall rows: {summary['total']['total']}; overall Major/Redo rate: {_fmt_pct(summary['total']['major_or_redo_rate'])}.",
        "",
        "| dataset | rows | No fix | Minor fix | Major fix | Redo | Major/Redo | Major/Redo rate | mean IoU | median IoU | mean Dice | small rows | small Major/Redo rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, stats in summary["datasets"].items():
        small = stats["size_buckets"]["small"]
        lines.append(
            f"| {dataset} | {stats['total']} | {stats['no_fix']} | {stats['minor_fix']} | "
            f"{stats['major_fix']} | {stats['redo']} | {stats['major_or_redo']} | "
            f"{_fmt_pct(stats['major_or_redo_rate'])} | {_fmt_num(stats['mean_iou'])} | "
            f"{_fmt_num(stats['median_iou'])} | {_fmt_num(stats['mean_dice'])} | "
            f"{small['total']} | {_fmt_pct(small['major_or_redo_rate'])} |"
        )

    lines.extend(["", "## Size-Stratified Summary"])
    for dataset, stats in summary["datasets"].items():
        lines.extend(
            [
                "",
                f"### {dataset}",
                "",
                "| size bucket | rows | No fix | Minor fix | Major fix | Redo | Major/Redo | Major/Redo rate | mean IoU | median IoU |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for bucket in SIZE_BUCKETS:
            bucket_stats = stats["size_buckets"][bucket]
            lines.append(
                f"| {bucket} | {bucket_stats['total']} | {bucket_stats['no_fix']} | "
                f"{bucket_stats['minor_fix']} | {bucket_stats['major_fix']} | {bucket_stats['redo']} | "
                f"{bucket_stats['major_or_redo']} | {_fmt_pct(bucket_stats['major_or_redo_rate'])} | "
                f"{_fmt_num(bucket_stats['mean_iou'])} | {_fmt_num(bucket_stats['median_iou'])} |"
            )

    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    for run in summary["runs"]:
        lines.append(f"- {run['name']}: `{run['path']}`")
    if summary.get("skipped_rows"):
        lines.extend(["", f"Skipped rows without usable IoU/GT area: {summary['skipped_rows']}"])
    return "\n".join(lines) + "\n"


def _key_finding_lines(summary: dict[str, Any]) -> list[str]:
    datasets = summary.get("datasets") or {}
    if not datasets:
        return ["- No usable rows were found."]
    highest = max(
        datasets.items(),
        key=lambda item: item[1].get("major_or_redo_rate") if item[1].get("major_or_redo_rate") is not None else -1.0,
    )
    lines = [
        f"- {highest[0]} has the highest overall Major/Redo rate: {_fmt_pct(highest[1].get('major_or_redo_rate'))} "
        f"({highest[1].get('major_or_redo')}/{highest[1].get('total')})."
    ]
    small_rows = [
        (name, stats["size_buckets"]["small"])
        for name, stats in datasets.items()
        if stats["size_buckets"]["small"].get("total", 0) >= 10
    ]
    if small_rows:
        highest_small = max(
            small_rows,
            key=lambda item: item[1].get("major_or_redo_rate") if item[1].get("major_or_redo_rate") is not None else -1.0,
        )
        lines.append(
            f"- {highest_small[0]} has the highest reliable small-object Major/Redo rate among datasets with at least 10 small objects: "
            f"{_fmt_pct(highest_small[1].get('major_or_redo_rate'))} ({highest_small[1].get('major_or_redo')}/{highest_small[1].get('total')})."
        )
    gap_rows = []
    for name, stats in datasets.items():
        small = stats["size_buckets"]["small"]
        large = stats["size_buckets"]["large"]
        if small.get("major_or_redo_rate") is None or large.get("major_or_redo_rate") is None:
            continue
        if small.get("total", 0) < 10 or large.get("total", 0) < 10:
            continue
        gap_rows.append((name, small["major_or_redo_rate"] - large["major_or_redo_rate"], small, large))
    if gap_rows:
        name, gap, small, large = max(gap_rows, key=lambda item: item[1])
        lines.append(
            f"- {name} shows the strongest reliable small-vs-large gap: small {_fmt_pct(small.get('major_or_redo_rate'))} "
            f"vs large {_fmt_pct(large.get('major_or_redo_rate'))} (gap {_fmt_pct(gap)})."
        )
    lines.append("- This report uses GT only to define offline labels; these labels must stay out of prediction-time risk features.")
    return lines


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
    else:
        path = Path(spec)
        name = path.name
    if path.is_dir():
        path = path / "correction_delta_dataset.jsonl"
    return name, path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize_row(row: dict[str, Any], run_name: str) -> dict[str, Any] | None:
    if row.get("record_status") not in (None, "ok"):
        return None
    delta = row.get("delta") or (row.get("evaluation_only") or {}).get("delta") or {}
    iou = delta.get("model_human_iou", delta.get("iou"))
    effort = correction_effort_from_iou(iou)
    human_area_px = delta.get("human_area_px")
    size_bucket = size_bucket_from_area(human_area_px)
    if effort is None or size_bucket is None:
        return None
    return {
        "dataset": str(row.get("dataset") or run_name),
        "run": run_name,
        "sample_id": row.get("sample_id") or row.get("task_id"),
        "iou": float(iou),
        "dice": _to_float_or_none(delta.get("model_human_dice", delta.get("dice"))),
        "human_area_px": float(human_area_px),
        "size_bucket": size_bucket,
        "effort": effort,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = {effort: 0 for effort in EFFORT_ORDER}
    for row in rows:
        counts[row["effort"]] += 1
    major_or_redo = counts["Major fix"] + counts["Redo"]
    ious = [row["iou"] for row in rows]
    dice_values = [row["dice"] for row in rows if row["dice"] is not None]
    return {
        "total": total,
        "no_fix": counts["No fix"],
        "minor_fix": counts["Minor fix"],
        "major_fix": counts["Major fix"],
        "redo": counts["Redo"],
        "major_or_redo": major_or_redo,
        "major_or_redo_rate": _safe_rate(major_or_redo, total),
        "mean_iou": mean(ious) if ious else None,
        "median_iou": median(ious) if ious else None,
        "mean_dice": mean(dice_values) if dice_values else None,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Run spec as name=path or path. Path may be a run dir or correction_delta_dataset.jsonl.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_gt_correction_effort(runs=args.run, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
