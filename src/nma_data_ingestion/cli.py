"""Command line entrypoints for dataset ingestion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from nma_data_ingestion.neuromatch import (
    DEFAULT_DATA_DIR,
    download_neuromatch_dataset,
    validate_neuromatch_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download NMA project datasets.")
    parser.add_argument(
        "--dataset",
        choices=("neuromatch",),
        default="neuromatch",
        help="Dataset to download. Only the Neuromatch preprocessed parquet is supported by default.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the raw dataset file should be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if the target file already exists.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate required columns after download or when reusing an existing file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_path = download_neuromatch_dataset(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        validate=args.validate,
    )
    print(f"Dataset ready: {dataset_path}")

    if args.validate:
        summary = validate_neuromatch_dataset(dataset_path)
        for key, value in asdict(summary).items():
            print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
