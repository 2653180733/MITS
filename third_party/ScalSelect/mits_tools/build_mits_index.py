"""Build a lightweight JSONL index for MITS samples."""

from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl")

from mits_pipeline.mits_io import build_index_record, dataset_subdirs, iter_mits_items, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded MITS JSONL index.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"MITS dataset root containing images/ and vqas/ (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_INDEX_PATH,
        help=f"Output JSONL index path (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum samples to read. Use 0 with --allow-full-scan for all.")
    parser.add_argument("--allow-full-scan", action="store_true", help="Allow an intentional full streaming scan.")
    parser.add_argument("--shard", action="append", dest="shards", help="Optional shard name filter. Can be repeated.")
    parser.add_argument("--category", action="append", dest="categories", help="Optional category filter. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _image_root, vqa_root = dataset_subdirs(args.dataset_root)
    limit = None if args.limit == 0 else args.limit

    def records():
        for numeric_id, (json_path, item) in enumerate(
            iter_mits_items(
                vqa_root=vqa_root,
                limit=limit,
                allow_full_scan=args.allow_full_scan,
                shards=args.shards,
                categories=args.categories,
            )
        ):
            yield build_index_record(json_path=json_path, item=item, numeric_id=numeric_id)

    count = write_jsonl(records(), args.output)
    print(f"Wrote {count} index records to {args.output}")


if __name__ == "__main__":
    main()
