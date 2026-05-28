"""Build non-targeted ScalSelect train sets at larger ratios for fair targeted ablations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import canonical_image_key, collect_identity_keys, normalize_key, read_json_or_jsonl  # noqa: E402
from mits_pipeline.mits_io import dataset_subdirs, load_json, mits_item_to_sharegpt, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-targeted ScalSelect train set with eval exclusions.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--selected-index", required=True, help="Existing ScalSelect index, e.g. mits_selected_20_48g_fast.jsonl.")
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-sharegpt", required=True)
    parser.add_argument("--exclude-eval", action="append", default=[], help="Eval JSONL/index to exclude. Can repeat.")
    parser.add_argument("--max-pairs-per-sample", type=int, default=32)
    parser.add_argument("--qa-filter", default="balanced", choices=["none", "quality", "balanced"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _identity_sets(paths: Sequence[str], image_root: str) -> tuple[set[str], set[str], int]:
    original_ids: set[str] = set()
    image_keys: set[str] = set()
    count = 0
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for record in read_json_or_jsonl(path):
            identities = collect_identity_keys(record, image_root=image_root)
            original_ids.update(identities["original_ids"])
            image_keys.update(identities["image_keys"])
            count += 1
    return original_ids, image_keys, count


def _is_excluded(record: Mapping[str, Any], original_ids: set[str], image_keys: set[str], image_root: str) -> bool:
    if normalize_key(record.get("original_id")) in original_ids:
        return True
    image_key = canonical_image_key(record.get("image"), image_root=image_root)
    return bool(image_key and image_key in image_keys)


def _write_sharegpt(records: Sequence[Mapping[str, Any]], path: str, image_root: str, max_pairs: int, qa_filter: str) -> int:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            try:
                item = load_json(str(record["json_path"]))
            except FileNotFoundError as exc:
                print(f"Warning: skip sample {record.get('id', index)} because source JSON is missing: {exc}")
                continue
            sample = mits_item_to_sharegpt(
                item=item,
                image_root=image_root,
                sample_id=record.get("id", index),
                json_path=str(record.get("json_path") or ""),
                max_pairs_per_sample=max_pairs,
                qa_filter=qa_filter,
            )
            if not sample.get("messages"):
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    image_root, _vqa_root = dataset_subdirs(args.dataset_root)
    exclude_original_ids, exclude_image_keys, excluded_read = _identity_sets(args.exclude_eval, image_root=image_root)

    records = [
        dict(record)
        for record in read_jsonl(args.selected_index)
        if not _is_excluded(record, exclude_original_ids, exclude_image_keys, image_root=image_root)
    ]
    summary = {
        "selected_index": args.selected_index,
        "selected_records_after_exclusion": len(records),
        "excluded_eval_records_read": excluded_read,
        "excluded_original_ids": len(exclude_original_ids),
        "excluded_image_keys": len(exclude_image_keys),
        "output_index": args.output_index,
        "output_sharegpt": args.output_sharegpt,
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        write_jsonl(records, args.output_index)
        summary["sharegpt_records"] = _write_sharegpt(
            records,
            path=args.output_sharegpt,
            image_root=image_root,
            max_pairs=args.max_pairs_per_sample,
            qa_filter=args.qa_filter,
        )
    else:
        summary["sharegpt_records"] = 0

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
