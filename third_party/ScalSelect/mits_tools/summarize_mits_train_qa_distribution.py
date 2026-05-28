"""Summarize scene/task QA distribution in MITS index or ShareGPT train files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import read_json_or_jsonl  # noqa: E402
from mits_pipeline.mits_io import (  # noqa: E402
    extract_task_tags,
    infer_task_from_text,
    iter_message_pairs,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MITS training data by scene and QA task.")
    parser.add_argument("--input", action="append", required=True, help="Input index/sharegpt JSON or JSONL. Can repeat.")
    parser.add_argument("--label", action="append", default=[], help="Optional label per --input.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--compare-base-label",
        default=None,
        help="Optional label used as the base row for delta columns in comparison CSV.",
    )
    return parser.parse_args()


def _messages_to_pairs(messages: Sequence[Mapping[str, Any]]) -> Iterator[Tuple[str, str, str]]:
    index = 0
    while index + 1 < len(messages):
        first = messages[index]
        second = messages[index + 1]
        first_role = str(first.get("role") or first.get("from") or "").lower()
        second_role = str(second.get("role") or second.get("from") or "").lower()
        if first_role in {"user", "human"} and second_role in {"assistant", "gpt"}:
            question = str(first.get("content") or first.get("value") or "")
            answer = str(second.get("content") or second.get("value") or "")
            if question and answer:
                yield "sharegpt", question, answer
            index += 2
            continue
        index += 1


def _record_scene(record: Mapping[str, Any]) -> str:
    meta = record.get("meta")
    if isinstance(meta, Mapping) and meta.get("scene"):
        return str(meta.get("scene"))
    return str(record.get("scene") or "unknown")


def _record_source_id(record: Mapping[str, Any]) -> str:
    meta = record.get("meta")
    if isinstance(meta, Mapping):
        for key in ("original_id", "json_path", "image"):
            if meta.get(key):
                return str(meta.get(key))
    for key in ("original_id", "json_path", "image", "id"):
        if record.get(key):
            return str(record.get(key))
    return ""


def _record_pairs(record: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    messages = record.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return list(_messages_to_pairs(messages))

    json_path = record.get("json_path")
    if json_path:
        try:
            item = load_json(str(json_path))
        except FileNotFoundError:
            return []
        return list(iter_message_pairs(item))
    return []


def _record_task_tags(record: Mapping[str, Any], pairs: Sequence[Tuple[str, str, str]]) -> List[str]:
    meta = record.get("meta")
    tags = None
    if isinstance(meta, Mapping):
        tags = meta.get("task_tags")
    if tags is None:
        tags = record.get("task_tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        return sorted({str(tag).lower() for tag in tags})
    if record.get("json_path"):
        try:
            return extract_task_tags(load_json(str(record["json_path"])))
        except FileNotFoundError:
            pass
    return sorted({infer_task_from_text(field, question) for field, question, _answer in pairs})


def _summarize(path: str, label: str, limit: Optional[int]) -> Dict[str, Any]:
    records = 0
    qa_pairs = 0
    scenes: Counter[str] = Counter()
    sample_task_tags: Counter[str] = Counter()
    qa_tasks: Counter[str] = Counter()
    qa_by_scene_task: Counter[Tuple[str, str]] = Counter()
    samples_with_task: Counter[str] = Counter()
    missing_pairs = 0

    for index, record in enumerate(read_json_or_jsonl(path)):
        if limit is not None and index >= limit:
            break
        records += 1
        scene = _record_scene(record)
        scenes[scene] += 1
        pairs = _record_pairs(record)
        if not pairs:
            missing_pairs += 1
        task_tags = _record_task_tags(record, pairs)
        for task in task_tags:
            sample_task_tags[task] += 1
        source_id = _record_source_id(record)
        seen_tasks_for_sample = set()
        for field, question, _answer in pairs:
            task = infer_task_from_text(field, question)
            qa_pairs += 1
            qa_tasks[task] += 1
            qa_by_scene_task[(scene, task)] += 1
            seen_tasks_for_sample.add(task)
        for task in seen_tasks_for_sample:
            samples_with_task[task] += 1
        if not source_id:
            continue

    return {
        "label": label,
        "path": path,
        "records": records,
        "qa_pairs": qa_pairs,
        "missing_pair_records": missing_pairs,
        "scene_counts": dict(scenes),
        "sample_task_tag_counts": dict(sample_task_tags),
        "qa_task_counts": dict(qa_tasks),
        "samples_with_task_counts": dict(samples_with_task),
        "qa_by_scene_task": {f"{scene}|{task}": count for (scene, task), count in qa_by_scene_task.items()},
    }


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _write_summary_json(summaries: Sequence[Mapping[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"inputs": list(summaries)}, handle, ensure_ascii=False, indent=2)


def _write_task_csv(summaries: Sequence[Mapping[str, Any]], path: str) -> None:
    tasks = sorted({task for summary in summaries for task in summary["qa_task_counts"]})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "task", "qa_count", "qa_percent", "samples_with_task"])
        writer.writeheader()
        for summary in summaries:
            total = int(summary["qa_pairs"])
            samples_with_task = summary["samples_with_task_counts"]
            for task in tasks:
                count = int(summary["qa_task_counts"].get(task, 0))
                writer.writerow(
                    {
                        "label": summary["label"],
                        "task": task,
                        "qa_count": count,
                        "qa_percent": f"{_percent(count, total):.4f}",
                        "samples_with_task": samples_with_task.get(task, 0),
                    }
                )


def _write_scene_csv(summaries: Sequence[Mapping[str, Any]], path: str) -> None:
    scenes = sorted({scene for summary in summaries for scene in summary["scene_counts"]})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "scene", "sample_count", "sample_percent"])
        writer.writeheader()
        for summary in summaries:
            total = int(summary["records"])
            for scene in scenes:
                count = int(summary["scene_counts"].get(scene, 0))
                writer.writerow(
                    {
                        "label": summary["label"],
                        "scene": scene,
                        "sample_count": count,
                        "sample_percent": f"{_percent(count, total):.4f}",
                    }
                )


def _write_markdown(summaries: Sequence[Mapping[str, Any]], path: str) -> None:
    lines = ["# MITS Train QA Distribution", ""]
    lines.append("| Label | Samples | QA pairs | Missing QA records |")
    lines.append("|---|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['records']} | {summary['qa_pairs']} | {summary['missing_pair_records']} |"
        )
    lines.append("")

    tasks = sorted({task for summary in summaries for task in summary["qa_task_counts"]})
    lines.append("## QA Task Distribution")
    lines.append("")
    lines.append("| Label | " + " | ".join(tasks) + " |")
    lines.append("|---" + "|---:" * len(tasks) + "|")
    for summary in summaries:
        total = int(summary["qa_pairs"])
        cells = [
            f"{summary['qa_task_counts'].get(task, 0)} ({_percent(int(summary['qa_task_counts'].get(task, 0)), total):.2f}%)"
            for task in tasks
        ]
        lines.append(f"| {summary['label']} | " + " | ".join(cells) + " |")
    lines.append("")

    scenes = sorted({scene for summary in summaries for scene in summary["scene_counts"]})
    lines.append("## Scene Distribution")
    lines.append("")
    lines.append("| Label | " + " | ".join(scenes) + " |")
    lines.append("|---" + "|---:" * len(scenes) + "|")
    for summary in summaries:
        total = int(summary["records"])
        cells = [
            f"{summary['scene_counts'].get(scene, 0)} ({_percent(int(summary['scene_counts'].get(scene, 0)), total):.2f}%)"
            for scene in scenes
        ]
        lines.append(f"| {summary['label']} | " + " | ".join(cells) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = list(args.label)
    while len(labels) < len(args.input):
        labels.append(os.path.splitext(os.path.basename(args.input[len(labels)]))[0])

    summaries = [_summarize(path, label, args.limit) for path, label in zip(args.input, labels)]
    _write_summary_json(summaries, os.path.join(args.output_dir, "train_qa_distribution_summary.json"))
    _write_task_csv(summaries, os.path.join(args.output_dir, "train_qa_task_distribution.csv"))
    _write_scene_csv(summaries, os.path.join(args.output_dir, "train_scene_distribution.csv"))
    _write_markdown(summaries, os.path.join(args.output_dir, "train_qa_distribution.md"))
    print(json.dumps({"outputs": args.output_dir, "inputs": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
