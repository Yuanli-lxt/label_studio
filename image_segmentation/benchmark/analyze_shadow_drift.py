from __future__ import annotations

import argparse

from image_segmentation.benchmark.shadow_root_cause_analysis import analyze_schema_aware_shadow_drift, analyze_shadow_drift


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze shadow rollout score drift root causes.")
    parser.add_argument("--multi-window-root", required=True)
    parser.add_argument("--baseline-window")
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema-aware", action="store_true")
    parser.add_argument("--classification")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.schema_aware:
        if not args.classification:
            raise SystemExit("--classification is required with --schema-aware")
        result = analyze_schema_aware_shadow_drift(args.multi_window_root, args.classification, args.output_dir)
        print(f"[OK] schema-aware drift comparable_pairs={len(result['comparable_pairs'])} output_dir={args.output_dir}")
        return 0
    if not args.baseline_window or not args.windows:
        raise SystemExit("--baseline-window and --windows are required unless --schema-aware is set")
    result = analyze_shadow_drift(args.multi_window_root, args.baseline_window, args.windows, args.output_dir)
    print(f"[OK] drift root-cause windows={len(result['windows'])} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
