"""Diagnose coverage gaps between MITS images, VQA JSON files, and index rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.mits_io import (
    dataset_subdirs,
    is_mits_input_dir,
    scene_from_image_path,
    scene_from_mits_path,
    shard_from_image_path,
    shard_from_mits_path,
)


DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.environ.get("MITS_INDEX_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dataset/images with vqas/*/*/integratedinput*.json and an "
            "optional generated mits_index.jsonl."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"MITS dataset root containing images/ and vqas/ (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Optional generated index path to count and inspect (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument("--examples", type=int, default=10, help="Number of example paths to print per mismatch type.")
    return parser.parse_args()


def normalize_rel(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def rel_to_image_root(path: Path, image_root: Path) -> str:
    return normalize_rel(str(path.relative_to(image_root)))


def iter_image_files(image_root: Path) -> Iterator[Path]:
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def iter_all_json_files(vqa_root: Path) -> Iterator[Path]:
    for path in vqa_root.rglob("*.json"):
        if path.is_file():
            yield path


def iter_mits_input_json_files(vqa_root: Path) -> Iterator[Path]:
    for path in vqa_root.rglob("*.json"):
        if not path.is_file():
            continue
        if len(path.parts) >= 2 and is_mits_input_dir(path.parts[-2]):
            yield path


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            item = json.load(handle)
        return item if isinstance(item, dict) else None
    except Exception:
        return None


def image_ref_variants(image_ref: str) -> Tuple[str, ...]:
    normalized = normalize_rel(image_ref)
    variants = [normalized]
    if normalized.startswith("images/"):
        variants.append(normalized[len("images/") :])
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def resolve_image_ref(image_ref: str, image_root: Path, image_rel_paths: set[str]) -> Optional[str]:
    if not image_ref:
        return None

    ref_path = Path(image_ref)
    if ref_path.is_absolute():
        try:
            return rel_to_image_root(ref_path.resolve(), image_root.resolve())
        except ValueError:
            return None

    for variant in image_ref_variants(image_ref):
        if variant in image_rel_paths:
            return variant
    return normalize_rel(image_ref)


def count_index(index_path: Path) -> Tuple[int, Counter[str], Counter[str]]:
    scenes: Counter[str] = Counter()
    shards: Counter[str] = Counter()
    count = 0
    if not index_path.exists():
        return count, scenes, shards

    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenes[str(record.get("scene") or "unknown")] += 1
            shards[str(record.get("shard") or "unknown")] += 1
    return count, scenes, shards


def print_counter(title: str, counter: Counter[str], limit: int = 20) -> None:
    print(f"\n{title}")
    if not counter:
        print("  (none)")
        return
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")
    if len(counter) > limit:
        print(f"  ... {len(counter) - limit} more")


def print_examples(title: str, values: Iterable[str], limit: int) -> None:
    print(f"\n{title}")
    for idx, value in enumerate(sorted(values)):
        if idx >= limit:
            break
        print(f"  {value}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_root_str, vqa_root_str = dataset_subdirs(str(dataset_root))
    image_root = Path(image_root_str)
    vqa_root = Path(vqa_root_str)
    index_path = Path(args.index)

    image_rel_paths = {rel_to_image_root(path, image_root) for path in iter_image_files(image_root)}
    image_scene_counts = Counter(scene_from_image_path(path) for path in image_rel_paths)
    image_shard_counts = Counter(shard_from_image_path(path) for path in image_rel_paths)

    all_json_paths = list(iter_all_json_files(vqa_root))
    mits_input_json_paths = list(iter_mits_input_json_files(vqa_root))
    non_mits_input_json_paths = sorted(set(all_json_paths) - set(mits_input_json_paths))

    json_scene_counts: Counter[str] = Counter()
    json_shard_counts: Counter[str] = Counter()
    referenced_images: Counter[str] = Counter()
    empty_image_field = 0
    malformed_json = 0

    for json_path in mits_input_json_paths:
        item = load_json(json_path)
        if item is None:
            malformed_json += 1
            continue

        json_scene_counts[scene_from_mits_path(str(json_path), item=item)] += 1
        json_shard_counts[shard_from_mits_path(str(json_path), item=item)] += 1

        image_ref = str(item.get("image") or "")
        if not image_ref:
            empty_image_field += 1
            continue

        resolved = resolve_image_ref(image_ref, image_root, image_rel_paths)
        if resolved:
            referenced_images[resolved] += 1

    referenced_image_set = set(referenced_images)
    unreferenced_images = image_rel_paths - referenced_image_set
    missing_referenced_images = referenced_image_set - image_rel_paths
    duplicate_refs = {path for path, count in referenced_images.items() if count > 1}

    index_count, index_scene_counts, index_shard_counts = count_index(index_path)

    print("=" * 72)
    print("MITS index coverage diagnosis")
    print("=" * 72)
    print(f"Dataset root: {dataset_root}")
    print(f"Image root:   {image_root}")
    print(f"VQA root:     {vqa_root}")
    print(f"Index path:   {index_path}")
    print()
    print(f"Image files under images/:                         {len(image_rel_paths)}")
    print(f"All JSON files under vqas/:                        {len(all_json_paths)}")
    print(f"JSON files under */integratedinput*/*.json:        {len(mits_input_json_paths)}")
    print(f"JSON files outside integratedinput* dirs:          {len(non_mits_input_json_paths)}")
    print(f"Generated index rows:                              {index_count}")
    print()
    print(f"Integrated JSON with empty image field:            {empty_image_field}")
    print(f"Integrated JSON that failed to parse as object:    {malformed_json}")
    print(f"Distinct image refs used by integrated JSON:       {len(referenced_image_set)}")
    print(f"Image refs missing on disk:                        {len(missing_referenced_images)}")
    print(f"Images on disk not referenced by integrated JSON:  {len(unreferenced_images)}")
    print(f"Image refs used by multiple integrated JSON files: {len(duplicate_refs)}")

    print_counter("Images by scene", image_scene_counts)
    print_counter("Integrated JSON by scene", json_scene_counts)
    if index_count:
        print_counter("Index rows by scene", index_scene_counts)

    print_counter("Images by shard", image_shard_counts)
    print_counter("Integrated JSON by shard", json_shard_counts)
    if index_count:
        print_counter("Index rows by shard", index_shard_counts)

    print_examples("Example images not referenced by integrated JSON", unreferenced_images, args.examples)
    print_examples("Example referenced images missing on disk", missing_referenced_images, args.examples)
    print_examples("Example JSON files outside integratedinput* dirs", (str(path) for path in non_mits_input_json_paths), args.examples)
    print_examples("Example duplicated image refs", duplicate_refs, args.examples)


if __name__ == "__main__":
    main()
