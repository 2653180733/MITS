"""Build MITS validation/test QA holdouts without training-set leakage."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.environ.get("MITS_INDEX_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl"))
DEFAULT_TRAIN_DATASET = os.environ.get(
    "TRAIN_DATASET",
    os.path.join(DEFAULT_WORK_DIR, "mits_selected_15_train32_sharegpt.jsonl"),
)
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_WORK_DIR, "eval")

from mits_pipeline.eval_utils import (  # noqa: E402
    canonical_image_key,
    collect_identity_keys,
    infer_answer_type,
    normalize_key,
    read_json_or_jsonl,
    resolve_image_path,
)
from mits_pipeline.mits_io import (  # noqa: E402
    DEFAULT_TASK_ORDER,
    dataset_subdirs,
    infer_task_from_text,
    load_json,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-free MITS eval QA files.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Full MITS index JSONL.")
    parser.add_argument(
        "--train-dataset",
        default=DEFAULT_TRAIN_DATASET,
        help="Selected training ShareGPT JSON/JSONL or selected index JSONL used for exclusion.",
    )
    parser.add_argument(
        "--train-index",
        action="append",
        default=[],
        help="Optional selected training index JSONL used for exclusion. Can be repeated.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-images", type=int, default=1000)
    parser.add_argument("--test-images", type=int, default=5000)
    parser.add_argument("--max-qas-per-image", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qa-filter",
        default="quality",
        choices=["none", "quality", "balanced"],
        help="Quality filter applied before picking per-task eval QA.",
    )
    parser.add_argument(
        "--require-train-dataset",
        action="store_true",
        help="Fail if --train-dataset is missing. By default a missing train file only warns.",
    )
    return parser.parse_args()


def _candidate_train_paths(train_path: str) -> List[str]:
    paths = [train_path] if train_path else []
    work_dir = os.path.dirname(train_path) if train_path else DEFAULT_WORK_DIR
    paths.extend(
        [
            os.path.join(work_dir, "mits_selected_15.jsonl"),
            os.path.join(work_dir, "mits_selected_15_train32_sharegpt.jsonl"),
            os.path.join(work_dir, "mits_selected_15_train32_sharegpt.json"),
        ]
    )
    seen = set()
    unique = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _collect_exclusion_sets(train_paths: Sequence[str], image_root: str, required: bool) -> Tuple[set[str], set[str], int]:
    original_ids: set[str] = set()
    image_keys: set[str] = set()
    count = 0

    existing_paths = [path for path in train_paths if path and os.path.exists(path)]
    if not existing_paths:
        message = f"Training dataset/index for exclusion not found: {list(train_paths)}"
        if required:
            raise FileNotFoundError(message)
        print(f"Warning: {message}; holdout will only rely on index split order.")
        return original_ids, image_keys, count

    for train_path in existing_paths:
        for record in read_json_or_jsonl(train_path):
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


def _select_eval_pairs(item: Mapping[str, Any], max_qas: int, qa_filter: str) -> List[Tuple[str, str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    seen = set()
    for field, question, answer in _iter_item_pairs(item):
        question = " ".join(str(question).replace("<image>", " ").split())
        answer = " ".join(str(answer).split())
        if not question or not answer:
            continue
        if qa_filter != "none" and len(question) < 8:
            continue
        if qa_filter != "none" and len(question) > 512:
            continue
        if qa_filter != "none" and len(answer) > 1536:
            continue
        key = (field, question.lower(), answer.lower())
        if qa_filter != "none" and key in seen:
            continue
        seen.add(key)
        pairs.append((field, question, answer))

    candidates: List[Tuple[str, str, str, str]] = [
        (field, question, answer, infer_task_from_text(field, question))
        for field, question, answer in pairs
    ]

    selected: List[Tuple[str, str, str, str]] = []
    used = set()
    for task in DEFAULT_TASK_ORDER:
        for index, candidate in enumerate(candidates):
            if index in used or candidate[3] != task:
                continue
            selected.append(candidate)
            used.add(index)
            break
        if len(selected) >= max_qas:
            return selected

    for index, candidate in enumerate(candidates):
        if len(selected) >= max_qas:
            break
        if index in used:
            continue
        selected.append(candidate)
        used.add(index)
    return selected


def _iter_item_pairs(item: Mapping[str, Any]) -> Iterable[Tuple[str, str, str]]:
    from mits_pipeline.mits_io import iter_message_pairs

    yield from iter_message_pairs(dict(item))


def _record_to_eval_qas(record: Mapping[str, Any], image_root: str, max_qas: int, qa_filter: str) -> List[Dict[str, Any]]:
    item = load_json(str(record["json_path"]))
    image_rel = str(record.get("image") or item.get("image") or "")
    image_abs = resolve_image_path(image_rel, image_root=image_root)
    selected = _select_eval_pairs(item, max_qas=max_qas, qa_filter=qa_filter)
    qas: List[Dict[str, Any]] = []

    for qa_idx, (field, question, answer, task) in enumerate(selected):
        sample_id = record.get("id")
        qas.append(
            {
                "id": f"{sample_id}:{qa_idx}",
                "sample_id": sample_id,
                "original_id": record.get("original_id") or item.get("id"),
                "image": image_abs,
                "image_rel": image_rel,
                "scene": record.get("scene"),
                "rare_tags": record.get("rare_tags", []),
                "task": task,
                "field": field,
                "question": question,
                "answer": answer,
                "answer_type": infer_answer_type(task, question, answer),
                "source_json": record.get("json_path"),
            }
        )
    return qas


def _write_split(
    split_name: str,
    records: Sequence[Mapping[str, Any]],
    output_dir: str,
    image_root: str,
    max_qas: int,
    qa_filter: str,
) -> Tuple[str, str, int]:
    index_path = os.path.join(output_dir, f"mits_{split_name}_index.jsonl")
    qas_path = os.path.join(output_dir, f"mits_{split_name}_qas.jsonl")
    write_jsonl((dict(record) for record in records), index_path)

    qa_count = 0

    def qas() -> Iterable[Dict[str, Any]]:
        nonlocal qa_count
        for record in records:
            try:
                sample_qas = _record_to_eval_qas(
                    record=record,
                    image_root=image_root,
                    max_qas=max_qas,
                    qa_filter=qa_filter,
                )
            except FileNotFoundError as exc:
                print(f"Warning: skip sample {record.get('id')} because source JSON is missing: {exc}")
                continue
            for qa in sample_qas:
                qa_count += 1
                yield qa

    write_jsonl(qas(), qas_path)
    return index_path, qas_path, qa_count


def _scene_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(record.get("scene") or "unknown") for record in records))


def main() -> None:
    args = parse_args()
    image_root, _vqa_root = dataset_subdirs(args.dataset_root)
    os.makedirs(args.output_dir, exist_ok=True)

    train_paths = _candidate_train_paths(args.train_dataset) + list(args.train_index or [])
    train_original_ids, train_image_keys, train_count = _collect_exclusion_sets(
        train_paths,
        image_root=image_root,
        required=args.require_train_dataset,
    )

    candidates = [
        record
        for record in read_jsonl(args.index)
        if not _is_excluded(record, train_original_ids, train_image_keys, image_root=image_root)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    target = args.val_images + args.test_images
    if len(candidates) < target:
        print(f"Warning: requested {target} images but only {len(candidates)} clean candidates are available.")
    selected = candidates[:target]
    val_records = selected[: args.val_images]
    test_records = selected[args.val_images : args.val_images + args.test_images]

    val_index, val_qas, val_qa_count = _write_split(
        "val",
        val_records,
        args.output_dir,
        image_root=image_root,
        max_qas=args.max_qas_per_image,
        qa_filter=args.qa_filter,
    )
    test_index, test_qas, test_qa_count = _write_split(
        "test",
        test_records,
        args.output_dir,
        image_root=image_root,
        max_qas=args.max_qas_per_image,
        qa_filter=args.qa_filter,
    )

    summary = {
        "index": args.index,
        "train_dataset": args.train_dataset,
        "train_records_read": train_count,
        "excluded_original_ids": len(train_original_ids),
        "excluded_image_keys": len(train_image_keys),
        "clean_candidates": len(candidates),
        "seed": args.seed,
        "val_images": len(val_records),
        "test_images": len(test_records),
        "val_qas": val_qa_count,
        "test_qas": test_qa_count,
        "val_scene_counts": _scene_counts(val_records),
        "test_scene_counts": _scene_counts(test_records),
        "outputs": {
            "val_index": val_index,
            "test_index": test_index,
            "val_qas": val_qas,
            "test_qas": test_qas,
        },
    }
    summary_path = os.path.join(args.output_dir, "mits_eval_set_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
