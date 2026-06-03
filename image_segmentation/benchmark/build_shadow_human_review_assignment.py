from __future__ import annotations

import argparse

from image_segmentation.benchmark.shadow_root_cause_analysis import build_human_review_assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reviewer-friendly assignment files from safe shadow review task packets.")
    parser.add_argument("--task-packet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_human_review_assignment(args.task_packet_dir, args.output_dir)
    print(f"[OK] human review assignment rows={result['rows']} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
