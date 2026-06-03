from __future__ import annotations

import argparse

from image_segmentation.benchmark.shadow_root_cause_analysis import write_drift_feedback_decision_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write drift and feedback decision report for shadow rollout hold/resume decisions.")
    parser.add_argument("--multi-window-root", required=True)
    parser.add_argument("--drift-dir", required=True)
    parser.add_argument("--missing-dir", required=True)
    parser.add_argument("--topk-dir", required=True)
    parser.add_argument("--assignment-dir", required=True)
    parser.add_argument("--feedback-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_drift_feedback_decision_report(
        args.multi_window_root,
        args.drift_dir,
        args.missing_dir,
        args.topk_dir,
        args.assignment_dir,
        args.feedback_dir,
        args.output_path,
    )
    print(f"[OK] decision={result['decision']} output={args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
