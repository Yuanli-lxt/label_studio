from __future__ import annotations

import argparse

from image_segmentation.benchmark.shadow_window_schema import classify_shadow_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify shadow rollout windows by production-safe schema and queue type.")
    parser.add_argument("--multi-window-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = classify_shadow_windows(args.multi_window_root, args.output_dir)
    print(f"[OK] classified shadow windows={len(result['windows'])} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
