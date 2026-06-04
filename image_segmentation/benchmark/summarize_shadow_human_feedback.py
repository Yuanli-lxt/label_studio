from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from image_segmentation.benchmark.shadow_root_cause_analysis import read_feedback_rows


OUTCOMES_MAJOR = {"major_correction_needed"}
OUTCOMES_NEEDING_CORRECTION = {"major_correction_needed", "minor", "minor_correction_needed"}
OUTCOMES_UNCLEAR = {"unclear"}
GROUPS = ["high_learned_low_current", "high_current_low_learned", "top_learned", "control_current_top"]


def summarize_shadow_human_feedback(review_packet: str, output_dir: str) -> dict:
    rows = read_feedback_rows(Path(review_packet))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reviewed = [row for row in rows if str(row.get("human_review_outcome") or "").strip()]
    group_metrics = _group_metrics(rows)
    summary = {
        "input_review_packet": review_packet,
        "output_dir": str(output),
        "offline_analysis_only": True,
        "feedback_available": bool(reviewed),
        "not_available_reason": None if reviewed else "no human_review_outcome values were provided; template/placeholder summary only",
        "rows": len(rows),
        "total_reviewed": len(reviewed),
        "reviewed_by_group": {group: group_metrics[group]["total_reviewed"] for group in GROUPS},
        "outcome_counts": dict(sorted(Counter(str(row.get("human_review_outcome") or "missing") for row in rows).items())),
        "major_correction_rate_by_group": {
            group: group_metrics[group]["major_correction_rate"] for group in GROUPS
        },
        "major_or_minor_correction_rate_by_group": {
            group: group_metrics[group]["major_or_minor_correction_rate"] for group in GROUPS
        },
        "correction_rate_by_learned_score_bucket": _bucket_rates(rows, "learned_shadow_score"),
        "correction_rate_by_current_score_bucket": _bucket_rates(rows, "current_priority_score"),
        "high_learned_low_current_hit_rate": _hit_rate(
            row for row in rows if _group(row) == "high_learned_low_current"
        ),
        "high_current_low_learned_hit_rate": _hit_rate(
            row for row in rows if _group(row) == "high_current_low_learned"
        ),
        "top_learned_hit_rate": _hit_rate(row for row in rows if _group(row) == "top_learned"),
        "control_current_top_hit_rate": _hit_rate(row for row in rows if _group(row) == "control_current_top"),
        "learned_lift_vs_control": _learned_lift_vs_control(group_metrics),
        "learned_missed_risk_discovery_rate": _missed_risk_discovery_rate(group_metrics),
        "over_prioritization_rate": _over_prioritization_rate(rows),
        "learned_found_missed_risks": _examples(
            rows,
            lambda row: _is_hit(row)
            and _group(row) in {"high_learned_low_current", "top_learned"},
        ),
        "learned_over_prioritized": _examples(
            rows,
            lambda row: _reviewed(row)
            and not _is_hit(row)
            and _group(row) in {"high_learned_low_current", "top_learned"},
        ),
        "inter_reviewer_agreement": _inter_reviewer_agreement(rows),
        "unclear_rate": _unclear_rate(rows),
        "recommendation": _recommendation(reviewed, group_metrics),
    }
    _write_json(output / "human_feedback_summary.json", summary)
    (output / "human_feedback_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    _write_jsonl(output / "human_feedback_examples.jsonl", summary["learned_found_missed_risks"] + summary["learned_over_prioritized"])
    return summary


def render_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "# Shadow Human Feedback Summary",
            "",
            f"- input_review_packet: {summary.get('input_review_packet')}",
            f"- rows: {summary.get('rows')}",
            f"- total_reviewed: {summary.get('total_reviewed')}",
            f"- feedback_available: {summary.get('feedback_available')}",
            f"- not_available_reason: {summary.get('not_available_reason')}",
            f"- offline_analysis_only: {summary.get('offline_analysis_only')}",
            f"- outcome_counts: {summary.get('outcome_counts')}",
            f"- reviewed_by_group: {summary.get('reviewed_by_group')}",
            "",
            "## Rates",
            f"- major_correction_rate_by_group: {summary.get('major_correction_rate_by_group')}",
            f"- major_or_minor_correction_rate_by_group: {summary.get('major_or_minor_correction_rate_by_group')}",
            f"- correction_rate_by_learned_score_bucket: {summary.get('correction_rate_by_learned_score_bucket')}",
            f"- correction_rate_by_current_score_bucket: {summary.get('correction_rate_by_current_score_bucket')}",
            f"- high_learned_low_current_hit_rate: {summary.get('high_learned_low_current_hit_rate')}",
            f"- high_current_low_learned_hit_rate: {summary.get('high_current_low_learned_hit_rate')}",
            f"- top_learned_hit_rate: {summary.get('top_learned_hit_rate')}",
            f"- control_current_top_hit_rate: {summary.get('control_current_top_hit_rate')}",
            f"- learned_lift_vs_control: {summary.get('learned_lift_vs_control')}",
            f"- learned_missed_risk_discovery_rate: {summary.get('learned_missed_risk_discovery_rate')}",
            f"- over_prioritization_rate: {summary.get('over_prioritization_rate')}",
            f"- inter_reviewer_agreement: {summary.get('inter_reviewer_agreement')}",
            f"- unclear_rate: {summary.get('unclear_rate')}",
            "",
            "## Recommendation",
            f"- {summary.get('recommendation')}",
        ]
    ) + "\n"


def _bucket_rates(rows: list[dict], score_field: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        score = _num(row.get(score_field))
        if score < 0.2:
            bucket = "0.0-0.2"
        elif score < 0.4:
            bucket = "0.2-0.4"
        elif score < 0.6:
            bucket = "0.4-0.6"
        elif score < 0.8:
            bucket = "0.6-0.8"
        else:
            bucket = "0.8-1.0"
        buckets[bucket].append(row)
    return [
        {
            "bucket": bucket,
            "count": len(bucket_rows),
            "correction_rate": _hit_rate(bucket_rows).get("hit_rate"),
        }
        for bucket, bucket_rows in sorted(buckets.items())
    ]


def _hit_rate(rows_iter) -> dict:
    rows = [row for row in rows_iter if _reviewed(row)]
    hits = sum(_is_hit(row) for row in rows)
    return {"count": len(rows), "hits": hits, "hit_rate": float(hits / len(rows)) if rows else None}


def _group_metrics(rows: list[dict]) -> dict:
    metrics = {}
    for group in GROUPS:
        group_rows = [row for row in rows if _group(row) == group]
        reviewed = [row for row in group_rows if _reviewed(row)]
        major = sum(_is_major(row) for row in reviewed)
        major_or_minor = sum(_is_hit(row) for row in reviewed)
        metrics[group] = {
            "total_reviewed": len(reviewed),
            "major_count": major,
            "major_or_minor_count": major_or_minor,
            "major_correction_rate": float(major / len(reviewed)) if reviewed else None,
            "major_or_minor_correction_rate": float(major_or_minor / len(reviewed)) if reviewed else None,
        }
    return metrics


def _learned_lift_vs_control(group_metrics: dict) -> dict:
    learned_groups = ["high_learned_low_current", "top_learned"]
    learned_reviewed = sum(group_metrics[group]["total_reviewed"] for group in learned_groups)
    learned_hits = sum(group_metrics[group]["major_or_minor_count"] for group in learned_groups)
    control = group_metrics["control_current_top"]
    learned_rate = float(learned_hits / learned_reviewed) if learned_reviewed else None
    control_rate = control["major_or_minor_correction_rate"]
    return {
        "available": learned_rate is not None and control_rate not in {None, 0},
        "learned_reviewed": learned_reviewed,
        "control_reviewed": control["total_reviewed"],
        "learned_major_or_minor_rate": learned_rate,
        "control_major_or_minor_rate": control_rate,
        "lift": float(learned_rate / control_rate) if learned_rate is not None and control_rate else None,
    }


def _missed_risk_discovery_rate(group_metrics: dict) -> dict:
    learned = group_metrics["high_learned_low_current"]["major_or_minor_correction_rate"]
    control = group_metrics["control_current_top"]["major_or_minor_correction_rate"]
    return {
        "available": learned is not None and control is not None,
        "high_learned_low_current_major_or_minor_rate": learned,
        "control_current_top_major_or_minor_rate": control,
        "delta": float(learned - control) if learned is not None and control is not None else None,
    }


def _over_prioritization_rate(rows: list[dict]) -> dict:
    group_rows = [
        row for row in rows
        if _group(row) == "high_learned_low_current" and _reviewed(row)
    ]
    ok = sum(str(row.get("human_review_outcome") or "").strip().lower() == "ok" for row in group_rows)
    return {"count": len(group_rows), "ok": ok, "rate": float(ok / len(group_rows)) if group_rows else None}


def _recommendation(reviewed: list[dict], group_metrics: dict) -> str:
    if len(reviewed) < 20:
        return "insufficient_feedback"
    lift = _learned_lift_vs_control(group_metrics).get("lift")
    missed = _missed_risk_discovery_rate(group_metrics).get("delta")
    over = group_metrics["high_learned_low_current"]["major_or_minor_correction_rate"]
    if lift is not None and lift >= 1.25 and missed is not None and missed > 0:
        return "useful_signal"
    if over is not None and over <= 0.1:
        return "not_useful"
    return "mixed"


def _examples(rows: list[dict], predicate) -> list[dict]:
    out = []
    for row in rows:
        if predicate(row):
            out.append(
                {
                    "sample_id": row.get("sample_id"),
                    "image_id": row.get("image_id"),
                    "annotation_id": row.get("annotation_id"),
                    "task_id": row.get("task_id"),
                    "learned_shadow_score": row.get("learned_shadow_score"),
                    "current_priority_score": row.get("current_priority_score"),
                    "human_review_outcome": row.get("human_review_outcome"),
                }
            )
        if len(out) >= 20:
            break
    return out


def _inter_reviewer_agreement(rows: list[dict]) -> dict:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        reviewer = row.get("reviewer")
        outcome = row.get("human_review_outcome")
        if not reviewer or not outcome:
            continue
        by_sample[_dedupe_key(row)].append(row)
    comparable = [sample_rows for sample_rows in by_sample.values() if len({row.get("reviewer") for row in sample_rows}) > 1]
    if not comparable:
        return {"available": False, "sample_count": 0, "exact_agreement_rate": None}
    agreed = 0
    for sample_rows in comparable:
        outcomes = {str(row.get("human_review_outcome") or "").strip().lower() for row in sample_rows}
        if len(outcomes) == 1:
            agreed += 1
    return {
        "available": True,
        "sample_count": len(comparable),
        "exact_agreement_rate": float(agreed / len(comparable)) if comparable else None,
    }


def _dedupe_key(row: dict) -> str:
    for key in ("sample_id", "image_id", "annotation_id", "task_id"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    return json.dumps(row, sort_keys=True)


def _is_hit(row: dict) -> bool:
    return str(row.get("human_review_outcome") or "").strip().lower() in OUTCOMES_NEEDING_CORRECTION


def _reviewed(row: dict) -> bool:
    return bool(str(row.get("human_review_outcome") or "").strip())


def _is_major(row: dict) -> bool:
    return str(row.get("human_review_outcome") or "").strip().lower() in OUTCOMES_MAJOR


def _group(row: dict) -> str | None:
    value = row.get("group")
    if value:
        return str(value)
    learned = _num(row.get("learned_shadow_score"))
    current = _num(row.get("current_priority_score"))
    if learned >= 0.7 and current <= 0.3:
        return "high_learned_low_current"
    if current >= 0.7 and learned <= 0.3:
        return "high_current_low_learned"
    return None


def _unclear_rate(rows: list[dict]) -> dict:
    reviewed = [row for row in rows if str(row.get("human_review_outcome") or "").strip()]
    unclear = sum(str(row.get("human_review_outcome") or "").strip().lower() in OUTCOMES_UNCLEAR for row in reviewed)
    return {"count": len(reviewed), "unclear": unclear, "rate": float(unclear / len(reviewed)) if reviewed else None}


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize offline human feedback for shadow disagreement packets.")
    parser.add_argument("--review-packet", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize_shadow_human_feedback(args.review_packet, args.output_dir)
    print(f"[OK] shadow human feedback summary rows={result['rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
