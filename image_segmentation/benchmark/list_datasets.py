from __future__ import annotations

import argparse
import json

from image_segmentation.benchmark.dataset_registry import DATASET_REGISTRY, registry_as_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List supported benchmark datasets.")
    parser.add_argument("--json", action="store_true", help="Emit registry JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json:
        print(json.dumps(registry_as_dict(), indent=2, ensure_ascii=False))
        return 0
    print("dataset\tdisplay_name\tloader\tdownload_mode\tscripted_download\tmanual_import")
    for entry in DATASET_REGISTRY.values():
        print(
            "\t".join(
                [
                    entry.dataset_name,
                    entry.display_name,
                    entry.loader_type,
                    entry.download_mode,
                    str(entry.supports_scripted_download).lower(),
                    str(entry.supports_manual_import).lower(),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
