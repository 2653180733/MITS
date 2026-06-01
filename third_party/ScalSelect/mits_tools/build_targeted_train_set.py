"""Build targeted MITS ShareGPT train sets for hard traffic QA scenes/tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")
DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.environ.get("MITS_INDEX_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl"))
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_DIR

from mits_pipeline.eval_utils import canonical_image_key, collect_identity_keys, normalize_key, read_json_or_jsonl  # noqa: E402
from mits_pipeline.mits_io import (  # noqa: E402
    dataset_subdirs,
    extract_task_tags,
    filter_quality_pairs,
    infer_task_from_text,
    iter_message_pairs,
    load_json,
    mits_item_to_sharegpt,
    read_jsonl,
    write_jsonl,
)


TARGET_SCENES = ["accident", "construction", "firesmoke"]
TARGET_TASKS = ["counting", "localization"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted MITS train ShareGPT JSONL files.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ratio", type=int, action="append", default=None, help="Target percent. Can repeat.")
    parser.add_argument("--base-ratio", type=int, default=15, help="Reference selected percent used to infer target size.")
    parser.add_argument(
        "--base-selected",
        default=os.path.join(DEFAULT_WORK_DIR, "mits_selected_15.jsonl"),
        help="Existing 15 percent selected index JSONL.",
    )
    parser.add_argument("--exclude-eval", action="append", default=[], help="Eval JSONL/index to exclude. Can repeat.")
    parser.add_argument("--target-scene", action="append", default=[], help="Override/add target scene. Can repeat.")
    parser.add_argument("--target-task", action="append", default=[], help="Override/add target task. Can repeat.")
    parser.add_argument("--target-multiplier", type=float, default=4.0)
    parser.add_argument("--max-pairs-per-sample", type=int, default=32)
    parser.add_argument("--qa-filter", default="balanced", choices=["none", "quality", "balanced"])
    parser.add_argument(
        "--task-aware",
        action="store_true",
        help="Use task quotas when converting each selected image to ShareGPT messages.",
    )
    parser.add_argument(
        "--task-quota",
        action="append",
        default=[],
        help="Task quota in task=N form for --task-aware. Can repeat.",
    )
    parser.add_argument(
        "--output-suffix",
        default="targeted",
        help="Output filename suffix after mits_selected_{ratio}_ (default: targeted).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_task_quotas(values: Sequence[str]) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--task-quota must be in task=N form, got: {value}")
        task, raw_count = value.split("=", 1)
        task = task.strip().lower()
        if not task:
            raise ValueError(f"--task-quota has an empty task name: {value}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"--task-quota count must be an integer, got: {value}") from exc
        if count < 0:
            raise ValueError(f"--task-quota count must be non-negative, got: {value}")
        quotas[task] = count
    return quotas


def _identity_sets(paths: Sequence[str], image_root: str) -> Tuple[set[str], set[str], int]:
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


def _base_selected_records(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    return [dict(record) for record in read_json_or_jsonl(path)]


def _record_tasks(record: Mapping[str, Any]) -> List[str]:
    tasks = record.get("task_tags")
    if isinstance(tasks, list):
        return [str(task).lower() for task in tasks]
    try:
        item = load_json(str(record["json_path"]))
    except (KeyError, FileNotFoundError):
        return []
    return [task.lower() for task in extract_task_tags(item)]


def _score_record(record: Mapping[str, Any], target_scenes: set[str], target_tasks: set[str], target_multiplier: float) -> float:
    scene = str(record.get("scene") or "").lower()
    tasks = set(_record_tasks(record))
    score = 1.0
    if scene in target_scenes:
        score *= target_multiplier
    if tasks.intersection(target_tasks):
        score *= target_multiplier
    if scene in target_scenes and tasks.intersection(target_tasks):
        score *= target_multiplier
    return score


def _weighted_sample(records: Sequence[Dict[str, Any]], size: int, rng: random.Random) -> List[Dict[str, Any]]:
    keyed = []
    for record in records:
        weight = max(0.0001, float(record.get("_target_weight", 1.0)))
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, record))
    keyed.sort(reverse=True, key=lambda item: item[0])
    return [record for _key, record in keyed[:size]]


def _target_size(base_count: int, base_ratio: int, ratio: int, total_available: int) -> int:
    if base_count > 0 and base_ratio > 0:
        return min(total_available, max(base_count, round(base_count * ratio / base_ratio)))
    return min(total_available, max(1, round(total_available * ratio / 100)))


def _task_aware_pairs(
    item: Mapping[str, Any],
    max_pairs_per_sample: int,
    qa_filter: str,
    task_quotas: Mapping[str, int],
) -> List[Tuple[str, str, str]]:
    pairs = filter_quality_pairs(
        pairs=list(iter_message_pairs(dict(item))),
        max_pairs_per_sample=None,
        qa_filter="quality" if qa_filter == "balanced" else qa_filter,
        min_answer_chars=1,
    )
    if not task_quotas:
        return pairs[:max_pairs_per_sample]

    grouped: Dict[str, List[Tuple[str, str, str]]] = {}
    for field, question, answer in pairs:
        task = infer_task_from_text(field, question)
        grouped.setdefault(task, []).append((field, question, answer))

    selected: List[Tuple[str, str, str]] = []
    selected_keys = set()
    for task, quota in task_quotas.items():
        if len(selected) >= max_pairs_per_sample:
            break
        for pair in grouped.get(task, [])[:quota]:
            if len(selected) >= max_pairs_per_sample:
                break
            key = (pair[0], pair[1], pair[2])
            if key in selected_keys:
                continue
            selected.append(pair)
            selected_keys.add(key)

    task_order = list(task_quotas.keys()) + sorted(task for task in grouped if task not in task_quotas)
    while len(selected) < max_pairs_per_sample:
        progressed = False
        for task in task_order:
            for pair in grouped.get(task, []):
                key = (pair[0], pair[1], pair[2])
                if key in selected_keys:
                    continue
                selected.append(pair)
                selected_keys.add(key)
                progressed = True
                break
            if len(selected) >= max_pairs_per_sample:
                break
        if not progressed:
            break
    return selected


def _sharegpt_from_pairs(
    item: Mapping[str, Any],
    record: Mapping[str, Any],
    image_root: str,
    sample_id: Any,
    pairs: Sequence[Tuple[str, str, str]],
    qa_filter: str,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    first_user = True
    for _field, question, answer in pairs:
        if first_user and "<image>" not in question:
            question = "<image>\n" + question
        first_user = False
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    image_rel = item.get("image", "")
    return {
        "id": sample_id,
        "messages": messages,
        "images": [os.path.normpath(os.path.join(image_root, image_rel))] if image_rel else [],
        "meta": {
            "original_id": item.get("id"),
            "json_path": record.get("json_path"),
            "image": image_rel,
            "scene": record.get("scene"),
            "shard": record.get("shard"),
            "task_tags": sorted({infer_task_from_text(field, question) for field, question, _answer in pairs}),
            "rare_tags": record.get("rare_tags", []),
            "positive_labels": record.get("positive_labels", []),
            "qa_filter": qa_filter,
            "qa_pairs_kept": len(pairs),
            "task_aware": True,
        },
    }


def _write_sharegpt_jsonl(
    records: Sequence[Mapping[str, Any]],
    output_path: str,
    image_root: str,
    max_pairs_per_sample: int,
    qa_filter: str,
    task_aware: bool,
    task_quotas: Mapping[str, int],
) -> int:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            try:
                item = load_json(str(record["json_path"]))
            except FileNotFoundError as exc:
                print(f"Warning: skip sample {record.get('id')} because source JSON is missing: {exc}")
                continue
            if task_aware:
                pairs = _task_aware_pairs(
                    item=item,
                    max_pairs_per_sample=max_pairs_per_sample,
                    qa_filter=qa_filter,
                    task_quotas=task_quotas,
                )
                sample = _sharegpt_from_pairs(
                    item=item,
                    record=record,
                    image_root=image_root,
                    sample_id=record.get("id", index),
                    pairs=pairs,
                    qa_filter=qa_filter,
                )
            else:
                sample = mits_item_to_sharegpt(
                    item=item,
                    image_root=image_root,
                    sample_id=record.get("id", index),
                    json_path=str(record.get("json_path") or ""),
                    max_pairs_per_sample=max_pairs_per_sample,
                    qa_filter=qa_filter,
                )
            if not sample.get("messages"):
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def _scene_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(record.get("scene") or "unknown") for record in records))


def _target_counts(records: Sequence[Mapping[str, Any]], target_scenes: set[str], target_tasks: set[str]) -> Dict[str, int]:
    scene_hits = 0
    task_hits = 0
    both_hits = 0
    for record in records:
        scene_hit = str(record.get("scene") or "").lower() in target_scenes
        task_hit = bool(set(_record_tasks(record)).intersection(target_tasks))
        scene_hits += int(scene_hit)
        task_hits += int(task_hit)
        both_hits += int(scene_hit and task_hit)
    return {"target_scene": scene_hits, "target_task": task_hits, "target_scene_and_task": both_hits}


def main() -> None:
    args = parse_args()
    image_root, _vqa_root = dataset_subdirs(args.dataset_root)
    rng = random.Random(args.seed)
    task_quotas = _parse_task_quotas(args.task_quota)

    target_scenes = set(args.target_scene or TARGET_SCENES)
    target_tasks = set(args.target_task or TARGET_TASKS)
    exclude_original_ids, exclude_image_keys, excluded_count = _identity_sets(args.exclude_eval, image_root=image_root)

    base_selected = _base_selected_records(args.base_selected)
    selected_ids = {normalize_key(record.get("original_id")) for record in base_selected if record.get("original_id")}
    selected_images = {canonical_image_key(record.get("image"), image_root=image_root) for record in base_selected}
    selected_images.discard("")

    candidates: List[Dict[str, Any]] = []
    for record in read_jsonl(args.index):
        if _is_excluded(record, exclude_original_ids, exclude_image_keys, image_root=image_root):
            continue
        item = dict(record)
        item["_target_weight"] = _score_record(
            item,
            target_scenes=target_scenes,
            target_tasks=target_tasks,
            target_multiplier=args.target_multiplier,
        )
        candidates.append(item)

    existing_records = [
        record
        for record in candidates
        if normalize_key(record.get("original_id")) in selected_ids
        or canonical_image_key(record.get("image"), image_root=image_root) in selected_images
    ]
    supplemental_pool = [
        record
        for record in candidates
        if normalize_key(record.get("original_id")) not in selected_ids
        and canonical_image_key(record.get("image"), image_root=image_root) not in selected_images
    ]

    ratios = args.ratio or [20, 30]
    summaries = []
    for ratio in sorted(set(ratios)):
        size = _target_size(len(base_selected), args.base_ratio, ratio, len(candidates))
        need = max(0, size - len(existing_records))
        supplemental = _weighted_sample(supplemental_pool, need, rng)
        selected = list(existing_records) + supplemental
        rng.shuffle(selected)

        index_path = os.path.join(args.output_dir, f"mits_selected_{ratio}_{args.output_suffix}.jsonl")
        sharegpt_path = os.path.join(args.output_dir, f"mits_selected_{ratio}_{args.output_suffix}_train32_sharegpt.jsonl")
        if not args.dry_run:
            write_jsonl(({key: value for key, value in record.items() if not key.startswith("_")} for record in selected), index_path)
            sharegpt_count = _write_sharegpt_jsonl(
                selected,
                output_path=sharegpt_path,
                image_root=image_root,
                max_pairs_per_sample=args.max_pairs_per_sample,
                qa_filter=args.qa_filter,
                task_aware=args.task_aware,
                task_quotas=task_quotas,
            )
        else:
            sharegpt_count = 0

        summaries.append(
            {
                "ratio": ratio,
                "target_size": size,
                "selected_records": len(selected),
                "base_records_kept": len(existing_records),
                "supplemental_records": len(supplemental),
                "sharegpt_records": sharegpt_count,
                "scene_counts": _scene_counts(selected),
                "target_counts": _target_counts(selected, target_scenes=target_scenes, target_tasks=target_tasks),
                "outputs": {"index": index_path, "sharegpt": sharegpt_path},
            }
        )

    summary = {
        "index": args.index,
        "base_selected": args.base_selected,
        "base_selected_records": len(base_selected),
        "base_ratio": args.base_ratio,
        "candidate_records": len(candidates),
        "excluded_eval_records_read": excluded_count,
        "excluded_original_ids": len(exclude_original_ids),
        "excluded_image_keys": len(exclude_image_keys),
        "target_scenes": sorted(target_scenes),
        "target_tasks": sorted(target_tasks),
        "target_multiplier": args.target_multiplier,
        "task_aware": args.task_aware,
        "task_quotas": task_quotas,
        "output_suffix": args.output_suffix,
        "dry_run": args.dry_run,
        "runs": summaries,
    }
    summary_path = os.path.join(args.output_dir, "mits_targeted_train_summary.json")
    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
