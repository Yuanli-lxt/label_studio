from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.evaluation import (
    average_precision,
    lift_at_fraction,
    precision_at_fraction,
    recall_at_fraction,
)


STRATEGIES = ["random", "mask_quality_only", "uncertainty_only", "correction_risk_only", "full_priority"]


BOOTSTRAP_METRICS = [
    "precision_at_20_percent",
    "recall_at_20_percent",
    "lift_at_20_percent_over_random",
    "average_precision_for_major_correction",
]


def compare_review_strategies(
    review_queue: str,
    output_dir: str,
    random_seed: int = 42,
    evaluation_type: str | None = None,
    train_size: int | None = None,
    eval_size: int | None = None,
    eval_positive_rate: float | None = None,
    bootstrap_iters: int = 0,
) -> dict:
    items = _read_jsonl(review_queue)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inferred = _infer_eval_context(review_queue, output_dir)
    evaluation_type = evaluation_type or inferred.get("evaluation_type")
    train_size = train_size if train_size is not None else inferred.get("train_size")
    eval_size = eval_size if eval_size is not None else inferred.get("eval_size")
    eval_positive_rate = (
        eval_positive_rate if eval_positive_rate is not None else inferred.get("eval_positive_rate")
    )
    base_rate = _safe_div(sum(1 for item in items if _is_major(item)), len(items))
    results = {}
    for strategy in STRATEGIES:
        ranked, available, reason = _rank_strategy(items, strategy, random_seed)
        if not available:
            results[strategy] = {"available": False, "reason": reason}
            continue
        row = {
            "available": True,
            "precision_at_10_percent": precision_at_fraction(ranked, 0.10),
            "precision_at_20_percent": precision_at_fraction(ranked, 0.20),
            "recall_at_10_percent": recall_at_fraction(ranked, 0.10),
            "recall_at_20_percent": recall_at_fraction(ranked, 0.20),
            "lift_at_10_percent_over_random": lift_at_fraction(ranked, 0.10),
            "lift_at_20_percent_over_random": lift_at_fraction(ranked, 0.20),
            "average_precision_for_major_correction": average_precision(ranked),
            "top_20_percent_major_correction_count": sum(1 for item in _top_fraction(ranked, 0.20) if _is_major(item)),
        }
        if bootstrap_iters:
            row["bootstrap_ci"] = _bootstrap_ci(ranked, strategy, int(bootstrap_iters), int(random_seed))
        results[strategy] = row
    warnings = []
    positives = sum(1 for item in items if _is_major(item))
    if evaluation_type == "held_out" and positives < 5:
        warnings.append("held-out eval has very few positive samples; metrics may have high variance.")
    report = {
        "evaluation_type": evaluation_type,
        "train_split_size": train_size,
        "eval_split_size": eval_size if eval_size is not None else len(items),
        "eval_positive_rate": eval_positive_rate if eval_positive_rate is not None else base_rate,
        "major_correction_base_rate": base_rate,
        "random_seed": random_seed,
        "bootstrap_iters": int(bootstrap_iters),
        "warnings": warnings,
        "strategies": results,
        "summary": _summary(results, base_rate),
    }
    report["recommendation"] = _strategy_recommendation(report)
    _write_json(output / "strategy_comparison.json", report)
    (output / "strategy_comparison.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _rank_strategy(items: list[dict], strategy: str, random_seed: int) -> tuple[list[dict], bool, str | None]:
    rows = list(items)
    if strategy == "random":
        rng = random.Random(int(random_seed))
        rng.shuffle(rows)
        return rows, True, None
    key_name = {
        "mask_quality_only": "rule_review_score",
        "uncertainty_only": "uncertainty_score",
        "correction_risk_only": "correction_risk_score",
        "full_priority": None,
    }[strategy]
    if strategy == "full_priority":
        if not any(isinstance(item.get("priority_score"), (int, float)) for item in rows):
            return rows, False, "missing_priority_score"
        return sorted(rows, key=lambda item: -_num(item.get("priority_score"))), True, None
    if not any(isinstance((item.get("score_components") or {}).get(key_name), (int, float)) for item in rows):
        return rows, False, f"missing_{key_name}"
    return sorted(rows, key=lambda item: -_num((item.get("score_components") or {}).get(key_name))), True, None


def _bootstrap_ci(items: list[dict], strategy: str, bootstrap_iters: int, random_seed: int) -> dict:
    labels = [_is_major(item) for item in items]
    if len(set(labels)) < 2:
        return {name: {"value": None, "ci95_low": None, "ci95_high": None, "warning": "degenerate_labels"} for name in BOOTSTRAP_METRICS}
    rng = random.Random(random_seed + sum(ord(ch) for ch in strategy))
    values = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(max(0, bootstrap_iters)):
        sample = [items[rng.randrange(len(items))] for _ in items]
        ranked, available, _ = _rank_strategy(sample, strategy, random_seed)
        if not available:
            continue
        metrics = {
            "precision_at_20_percent": precision_at_fraction(ranked, 0.20),
            "recall_at_20_percent": recall_at_fraction(ranked, 0.20),
            "lift_at_20_percent_over_random": lift_at_fraction(ranked, 0.20),
            "average_precision_for_major_correction": average_precision(ranked),
        }
        for name, value in metrics.items():
            if value is not None:
                values[name].append(float(value))
    current = {
        "precision_at_20_percent": precision_at_fraction(items, 0.20),
        "recall_at_20_percent": recall_at_fraction(items, 0.20),
        "lift_at_20_percent_over_random": lift_at_fraction(items, 0.20),
        "average_precision_for_major_correction": average_precision(items),
    }
    out = {}
    for name in BOOTSTRAP_METRICS:
        samples = sorted(values[name])
        warning = None
        if not samples:
            warning = "bootstrap_metric_degenerate"
        out[name] = {
            "value": current[name],
            "ci95_low": _percentile(samples, 2.5) if samples else None,
            "ci95_high": _percentile(samples, 97.5) if samples else None,
        }
        if warning:
            out[name]["warning"] = warning
    return out


def _summary(results: dict, base_rate: float | None) -> dict:
    available = {name: row for name, row in results.items() if row.get("available")}
    best = None
    if available:
        best = max(
            available,
            key=lambda name: _num(available[name].get("average_precision_for_major_correction")),
        )
    full = available.get("full_priority")
    return {
        "best_strategy_by_ap": best,
        "full_priority_exceeds_random": _beats(full, available.get("random")),
        "full_priority_exceeds_mask_quality_only": _beats(full, available.get("mask_quality_only")),
        "full_priority_exceeds_uncertainty_only": _beats(full, available.get("uncertainty_only")),
        "full_priority_exceeds_correction_risk_only": _beats(full, available.get("correction_risk_only")),
        "meets_initial_effectiveness_standard": bool(
            full
            and _num(full.get("lift_at_20_percent_over_random")) > 1.5
            and base_rate is not None
            and _num(full.get("average_precision_for_major_correction")) > base_rate
            and _num(full.get("precision_at_20_percent")) > base_rate
        ),
        "correction_risk_only_meets_initial_effectiveness_standard": bool(
            available.get("correction_risk_only")
            and _num(available["correction_risk_only"].get("lift_at_20_percent_over_random")) > 1.5
            and base_rate is not None
            and _num(available["correction_risk_only"].get("average_precision_for_major_correction")) > base_rate
            and _num(available["correction_risk_only"].get("precision_at_20_percent")) > base_rate
        ),
    }


def _beats(left: dict | None, right: dict | None) -> bool | None:
    if not left or not right:
        return None
    return _num(left.get("average_precision_for_major_correction")) > _num(
        right.get("average_precision_for_major_correction")
    )


def _render_markdown(report: dict) -> str:
    lines = [
        "# Review Strategy Comparison",
        "",
        f"- evaluation type: {report.get('evaluation_type') or 'unspecified'}",
        f"- train split size: {report.get('train_split_size')}",
        f"- eval split size: {report.get('eval_split_size')}",
        f"- eval positive rate: {_fmt(report.get('eval_positive_rate'))}",
        f"- major correction base rate: {_fmt(report.get('major_correction_base_rate'))}",
        f"- best strategy by AP: {report.get('summary', {}).get('best_strategy_by_ap')}",
        f"- full priority meets initial standard: {report.get('summary', {}).get('meets_initial_effectiveness_standard')}",
        "",
        "## Strategies",
    ]
    for name, row in report.get("strategies", {}).items():
        if not row.get("available"):
            lines.append(f"- {name}: unavailable ({row.get('reason')})")
            continue
        lines.append(
            f"- {name}: precision@20={_fmt(row.get('precision_at_20_percent'))}, "
            f"recall@20={_fmt(row.get('recall_at_20_percent'))}, "
            f"lift@20={_fmt(row.get('lift_at_20_percent_over_random'))}, "
            f"AP={_fmt(row.get('average_precision_for_major_correction'))}"
        )
    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            "## Full Priority Checks",
            f"- beats random: {summary.get('full_priority_exceeds_random')}",
            f"- beats mask_quality_only: {summary.get('full_priority_exceeds_mask_quality_only')}",
            f"- beats uncertainty_only: {summary.get('full_priority_exceeds_uncertainty_only')}",
            f"- beats correction_risk_only: {summary.get('full_priority_exceeds_correction_risk_only')}",
            f"- correction_risk_only is best: {summary.get('best_strategy_by_ap') == 'correction_risk_only'}",
            "",
            "Initial effective standard: lift@20% > 1.5, AP > base rate, and precision@20% > random expected precision.",
            "",
            "## Bootstrap CI",
            *_bootstrap_lines((report.get("strategies") or {}).get("full_priority") or {}),
            "",
            "## Recommendation",
            f"- {report.get('recommendation')}",
        ]
    )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _strategy_recommendation(report: dict) -> str:
    strategies = report.get("strategies") or {}
    summary = report.get("summary") or {}
    positive_count = int(round(_num(report.get("major_correction_base_rate")) * _num(report.get("eval_split_size"))))
    full = strategies.get("full_priority") if isinstance(strategies.get("full_priority"), dict) else {}
    risk = strategies.get("correction_risk_only") if isinstance(strategies.get("correction_risk_only"), dict) else {}
    uncertainty = strategies.get("uncertainty_only") if isinstance(strategies.get("uncertainty_only"), dict) else {}
    random_row = strategies.get("random") if isinstance(strategies.get("random"), dict) else {}
    full_ap = _num(full.get("average_precision_for_major_correction"))
    risk_ap = _num(risk.get("average_precision_for_major_correction"))
    uncertainty_ap = _num(uncertainty.get("average_precision_for_major_correction"))
    random_ap = _num(random_row.get("average_precision_for_major_correction"))
    notes = []
    if risk.get("available") and risk_ap > full_ap * 1.10:
        notes.append("consider increasing correction_risk weight, but validate on another split before changing defaults")
    elif uncertainty.get("available") and uncertainty_ap > full_ap * 1.10:
        notes.append("consider increasing uncertainty weight or improving risk model")
    elif summary.get("best_strategy_by_ap") == "full_priority" and summary.get("meets_initial_effectiveness_standard"):
        notes.append("current Layer 5 fusion is promising; validate on harder datasets next")
    elif max(full_ap, risk_ap, uncertainty_ap, random_ap) <= random_ap * 1.10 + 1e-12:
        notes.append("increase sample size, improve features, or add harder datasets before tuning weights")
    else:
        notes.append("do not tune Layer 5 defaults yet; validate on another split")
    if positive_count < 30:
        notes.append("positive count is still low; metrics may be high variance")
    lift_ci = ((full.get("bootstrap_ci") or {}).get("lift_at_20_percent_over_random") or {})
    if lift_ci.get("ci95_low") is not None and float(lift_ci["ci95_low"]) <= 1.0:
        notes.append("lift confidence interval includes no improvement over random")
    return "; ".join(_dedupe(notes))


def _bootstrap_lines(row: dict) -> list[str]:
    ci = row.get("bootstrap_ci") or {}
    if not ci:
        return ["- unavailable"]
    labels = {
        "precision_at_20_percent": "precision@20% CI",
        "recall_at_20_percent": "recall@20% CI",
        "lift_at_20_percent_over_random": "lift@20% CI",
        "average_precision_for_major_correction": "AP CI",
    }
    return [
        f"- {label}: {_fmt((ci.get(name) or {}).get('ci95_low'))} - {_fmt((ci.get(name) or {}).get('ci95_high'))}"
        for name, label in labels.items()
    ]


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _infer_eval_context(review_queue: str, output_dir: str) -> dict:
    text = f"{review_queue} {output_dir}".lower()
    context: dict[str, Any] = {}
    if "heldout" in text or "held_out" in text or "held-out" in text:
        context["evaluation_type"] = "held_out"
    if "oof" in text or "out_of_fold" in text:
        context["evaluation_type"] = "out_of_fold"
    queue_path = Path(review_queue)
    for directory in [queue_path.parent, Path(output_dir), Path(output_dir).parent]:
        metadata_path = directory / "split_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            context.setdefault("evaluation_type", "held_out")
            context["train_size"] = metadata.get("n_train")
            context["eval_size"] = metadata.get("n_eval")
            context["eval_positive_rate"] = metadata.get("eval_positive_rate")
            break
    return context


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _top_fraction(items: list[dict], fraction: float) -> list[dict]:
    import math

    return items[: max(1, int(math.ceil(len(items) * fraction)))] if items else []


def _is_major(item: dict) -> bool:
    return (((item.get("evaluation_only") or {}).get("delta") or {}).get("major_correction")) is True


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare review queue ranking strategies.")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--evaluation-type")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--eval-size", type=int)
    parser.add_argument("--eval-positive-rate", type=float)
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compare_review_strategies(
        args.review_queue,
        args.output_dir,
        random_seed=args.random_seed,
        evaluation_type=args.evaluation_type,
        train_size=args.train_size,
        eval_size=args.eval_size,
        eval_positive_rate=args.eval_positive_rate,
        bootstrap_iters=args.bootstrap_iters,
    )
    print(f"[OK] wrote strategy comparison to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
