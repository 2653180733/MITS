"""Convert indexed MITS samples to ShareGPT JSON format."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Iterator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.environ.get("MITS_INDEX_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl"))
DEFAULT_SHAREGPT_PATH = os.environ.get("MITS_SHAREGPT_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_sharegpt.json"))
DEFAULT_LOWLIGHT_OUTPUT_ROOT = os.environ.get(
    "LOWLIGHT_OUTPUT_ROOT",
    os.path.join(DEFAULT_WORK_DIR, "lowlight_aug", "images"),
)

from mits_pipeline.lowlight import (
    DEFAULT_LOWLIGHT_THRESHOLD,
    is_low_light_by_luminance,
    is_low_light_by_meta,
    load_rgb_image,
    save_lowlight_enhanced_image,
)
from mits_pipeline.mits_io import (
    dataset_subdirs,
    extract_rare_tags,
    join_path,
    load_json,
    mits_item_to_sharegpt,
    read_jsonl,
    write_sharegpt_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MITS index records to ShareGPT JSON.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"MITS dataset root containing images/ and vqas/ (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Input index JSONL from build_mits_index.py (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_SHAREGPT_PATH,
        help=f"Output ShareGPT JSON array (default: {DEFAULT_SHAREGPT_PATH}).",
    )
    parser.add_argument(
        "--max-pairs-per-sample",
        type=int,
        default=None,
        help="Maximum QA pairs per image. Default is 32; 0 keeps all pairs that pass the QA filter.",
    )
    parser.add_argument(
        "--qa-filter",
        default="balanced",
        choices=["none", "quality", "balanced"],
        help="QA filtering mode: none keeps original order, quality filters/dedupes, balanced also applies task quotas.",
    )
    parser.add_argument(
        "--max-pairs-per-task",
        type=int,
        default=8,
        help="Soft cap per inferred task when --qa-filter balanced is used. 0 disables the task cap.",
    )
    parser.add_argument("--min-question-chars", type=int, default=8)
    parser.add_argument("--min-answer-chars", type=int, default=3)
    parser.add_argument("--max-question-chars", type=int, default=512)
    parser.add_argument("--max-answer-chars", type=int, default=1024)
    parser.add_argument(
        "--legacy-no-filter",
        action="store_true",
        help="Use no QA quality filter and no default 32-pair cap unless explicitly set.",
    )
    parser.add_argument(
        "--lowlight-augment",
        action="store_true",
        help="Append enhanced dual-view samples for low-light images.",
    )
    parser.add_argument(
        "--lowlight-output-root",
        default=DEFAULT_LOWLIGHT_OUTPUT_ROOT,
        help=f"Directory to save enhanced low-light images (default: {DEFAULT_LOWLIGHT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--lowlight-threshold", type=float, default=DEFAULT_LOWLIGHT_THRESHOLD)
    parser.add_argument("--lowlight-gamma", type=float, default=0.6)
    parser.add_argument("--lowlight-contrast", type=float, default=1.25)
    parser.add_argument("--lowlight-sharpness", type=float, default=1.1)
    parser.add_argument(
        "--lowlight-max-aug-ratio",
        type=float,
        default=0.25,
        help="Maximum fraction of final output occupied by low-light augmented samples.",
    )
    parser.add_argument(
        "--lowlight-force-all",
        action="store_true",
        help="Augment every detected low-light sample, ignoring --lowlight-max-aug-ratio.",
    )
    return parser.parse_args()


def should_augment_lowlight(
    item: dict,
    record: dict,
    image_root: str,
    threshold: float,
) -> tuple[bool, dict]:
    image_rel = item.get("image", "")
    source_image = join_path(image_root, image_rel) if image_rel else ""
    rare_tags = record.get("rare_tags") or extract_rare_tags(item, json_path=record.get("json_path"))
    lowlight_by_meta = is_low_light_by_meta(
        item=item,
        json_path=record.get("json_path"),
        rare_tags=rare_tags,
    )

    mean_luminance = None
    lowlight_by_luminance = False
    if source_image and os.path.exists(source_image):
        image = load_rgb_image(source_image)
        lowlight_by_luminance, mean_luminance = is_low_light_by_luminance(
            image,
            threshold=threshold,
        )

    return lowlight_by_meta or lowlight_by_luminance, {
        "source_image": source_image,
        "image_rel": image_rel,
        "mean_luminance": mean_luminance,
        "lowlight_by_meta": lowlight_by_meta,
        "lowlight_by_luminance": lowlight_by_luminance,
    }


def make_augmented_sample(
    sample: dict,
    image_info: dict,
    saved_info: dict,
) -> dict:
    augmented = copy.deepcopy(sample)
    augmented["id"] = f"{sample.get('id')}_lowlight_aug"
    augmented["images"] = [saved_info["image_path"]]

    meta = dict(augmented.get("meta", {}))
    meta.update(
        {
            "augmentation": "lowlight_enhanced",
            "source_sample_id": sample.get("id"),
            "source_image": image_info.get("source_image"),
            "enhanced_image": saved_info["image_path"],
            "mean_luminance": saved_info.get("mean_luminance"),
            "lowlight_by_meta": image_info.get("lowlight_by_meta", False),
            "lowlight_by_luminance": saved_info.get("lowlight_by_luminance", False),
        }
    )
    augmented["meta"] = meta
    return augmented


def main() -> None:
    args = parse_args()
    image_root, _vqa_root = dataset_subdirs(args.dataset_root)
    qa_filter = "none" if args.legacy_no_filter else args.qa_filter
    if args.max_pairs_per_sample is None:
        max_pairs = None if args.legacy_no_filter else 32
    elif args.max_pairs_per_sample == 0:
        max_pairs = None
    else:
        max_pairs = args.max_pairs_per_sample
    if args.lowlight_max_aug_ratio < 0:
        raise ValueError("--lowlight-max-aug-ratio must be non-negative.")
    if args.lowlight_max_aug_ratio >= 1 and not args.lowlight_force_all:
        raise ValueError("--lowlight-max-aug-ratio must be less than 1 unless --lowlight-force-all is used.")

    records = list(read_jsonl(args.index))
    if args.lowlight_force_all:
        max_aug_samples = None
    else:
        max_aug_samples = int(
            (args.lowlight_max_aug_ratio * len(records))
            / max(1e-12, 1.0 - args.lowlight_max_aug_ratio)
        )
    base_count = 0
    aug_count = 0
    skipped_aug_count = 0

    def samples() -> Iterator[dict]:
        nonlocal base_count, aug_count, skipped_aug_count
        for record in records:
            item = load_json(record["json_path"])
            sample = mits_item_to_sharegpt(
                item=item,
                image_root=image_root,
                sample_id=int(record["id"]),
                json_path=record["json_path"],
                max_pairs_per_sample=max_pairs,
                qa_filter=qa_filter,
                max_pairs_per_task=args.max_pairs_per_task,
                min_question_chars=args.min_question_chars,
                min_answer_chars=args.min_answer_chars,
                max_question_chars=args.max_question_chars,
                max_answer_chars=args.max_answer_chars,
            )
            base_count += 1
            yield sample

            if not args.lowlight_augment:
                continue

            is_lowlight, image_info = should_augment_lowlight(
                item=item,
                record=record,
                image_root=image_root,
                threshold=args.lowlight_threshold,
            )
            if not is_lowlight:
                continue
            if not image_info.get("source_image") or not os.path.exists(image_info["source_image"]):
                skipped_aug_count += 1
                continue
            if max_aug_samples is not None:
                if aug_count >= max_aug_samples:
                    skipped_aug_count += 1
                    continue

            saved_info = save_lowlight_enhanced_image(
                source_path=image_info["source_image"],
                image_root=image_root,
                output_root=args.lowlight_output_root,
                image_rel=image_info["image_rel"],
                threshold=args.lowlight_threshold,
                gamma=args.lowlight_gamma,
                contrast=args.lowlight_contrast,
                sharpness=args.lowlight_sharpness,
            )
            aug_count += 1
            yield make_augmented_sample(
                sample=sample,
                image_info=image_info,
                saved_info=saved_info,
            )

    count = write_sharegpt_json(samples(), args.output)
    print(f"Wrote {count} ShareGPT samples to {args.output}")
    if args.lowlight_augment:
        print(
            "Low-light augmentation: "
            f"base={base_count}, augmented={aug_count}, skipped={skipped_aug_count}, "
            f"output_root={args.lowlight_output_root}"
        )


if __name__ == "__main__":
    main()
