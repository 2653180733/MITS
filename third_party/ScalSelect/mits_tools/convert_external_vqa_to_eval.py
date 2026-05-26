"""Convert external VQA JSONL into the unified MITS eval schema."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import (  # noqa: E402
    canonical_image_key,
    infer_answer_type,
    read_json_or_jsonl,
    resolve_image_path,
    write_jsonl,
)
from mits_pipeline.mits_io import infer_task_from_text  # noqa: E402


DEFAULT_IMAGE_ROOT = os.environ.get("DATASET_ROOT", "/root/autodl-tmp/data/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external VQA/ShareGPT JSONL into eval schema.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL/JSON with image/question/answer or ShareGPT images/messages.",
    )
    parser.add_argument("--output", required=True, help="Output eval JSONL.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, help="Root for relative image paths.")
    parser.add_argument("--scene", default="external")
    parser.add_argument("--task", default=None, help="Optional fixed task label if not inferable.")
    parser.add_argument("--source-name", default=None, help="Optional source dataset label.")
    return parser.parse_args()


def _message_role(message: Mapping[str, Any], fallback: str) -> str:
    role = str(message.get("role", message.get("from", fallback))).lower()
    if role in {"human", "user", "question"} or "question" in role:
        return "user"
    if role in {"gpt", "assistant", "answer"} or "answer" in role:
        return "assistant"
    return fallback


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("value") is not None:
                    parts.append(str(item["value"]))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _iter_message_pairs(record: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
    messages = record.get("messages", record.get("conversations"))
    if not isinstance(messages, list):
        return

    i = 0
    while i + 1 < len(messages):
        first = messages[i]
        second = messages[i + 1]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            i += 1
            continue
        first_role = _message_role(first, "user")
        second_role = _message_role(second, "assistant")
        question = _message_text(first.get("content", first.get("value"))).replace("<image>", " ").strip()
        answer = _message_text(second.get("content", second.get("value"))).strip()
        if first_role == "user" and second_role == "assistant" and question and answer:
            yield question, answer
            i += 2
        else:
            i += 1


def _first_image(record: Mapping[str, Any]) -> str:
    image = record.get("image", record.get("image_path"))
    if image:
        return str(image)
    images = record.get("images")
    if isinstance(images, str):
        return images
    if isinstance(images, list) and images:
        return str(images[0])
    return ""


def _resolve_eval_image(image: str, image_root: str) -> str:
    if not image:
        return ""
    if os.path.isabs(image):
        if os.path.exists(image):
            return image
        if image_root:
            rebased_key = canonical_image_key(image)
            if rebased_key and rebased_key != image.replace("\\", "/").lower().rstrip("/"):
                return resolve_image_path(rebased_key, image_root=image_root)
        return image
    return resolve_image_path(image, image_root=image_root)


def _base_meta(record: Mapping[str, Any], index: int, scene: str, source_name: Optional[str]) -> Dict[str, Any]:
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    return {
        "sample_id": record.get("sample_id", record.get("id", index)),
        "original_id": record.get("original_id", meta.get("original_id", record.get("id", index))),
        "scene": record.get("scene", meta.get("scene", scene)),
        "rare_tags": record.get("rare_tags", meta.get("rare_tags", [])),
        "source_json": record.get("source_json", meta.get("json_path", source_name)),
    }


def _make_eval_record(
    record: Mapping[str, Any],
    index: int,
    qa_idx: int,
    image_root: str,
    scene: str,
    task_override: Optional[str],
    source_name: Optional[str],
    question: str,
    answer: str,
    field: Optional[str] = None,
) -> Dict[str, Any]:
    image = _first_image(record)
    base_id = record.get("id", index)
    task = (task_override or record.get("task") or infer_task_from_text(str(field or record.get("field") or ""), question)).lower()
    if task not in {"background", "recognition", "counting", "localization", "reasoning"}:
        task = infer_task_from_text(str(field or record.get("field") or ""), question)
    meta = _base_meta(record, index=index, scene=scene, source_name=source_name)

    return {
        "id": f"{base_id}:{qa_idx}",
        "sample_id": meta["sample_id"],
        "original_id": meta["original_id"],
        "image": _resolve_eval_image(image, image_root=image_root),
        "image_rel": image,
        "scene": meta["scene"],
        "rare_tags": meta["rare_tags"],
        "task": task,
        "field": field or record.get("field", task),
        "question": question,
        "answer": answer,
        "answer_type": record.get("answer_type") or infer_answer_type(task, question, answer),
        "source_json": meta["source_json"],
    }


def _normalize_record(record: Dict[str, Any], index: int, image_root: str, scene: str, task_override: Optional[str], source_name: Optional[str]) -> List[Dict[str, Any]]:
    message_pairs = list(_iter_message_pairs(record))
    if message_pairs:
        return [
            _make_eval_record(
                record=record,
                index=index,
                qa_idx=qa_idx,
                image_root=image_root,
                scene=scene,
                task_override=task_override,
                source_name=source_name,
                question=question,
                answer=answer,
            )
            for qa_idx, (question, answer) in enumerate(message_pairs)
        ]

    image = str(record.get("image") or record.get("image_path") or "")
    question = str(record.get("question") or record.get("query") or "")
    answer = str(record.get("answer") or record.get("label") or "")
    if not question or not answer:
        return []
    simple_record = dict(record)
    simple_record["image"] = image
    eval_record = _make_eval_record(
        record=simple_record,
        index=index,
        qa_idx=0,
        image_root=image_root,
        scene=scene,
        task_override=task_override,
        source_name=source_name,
        question=question,
        answer=answer,
        field=str(record.get("field") or ""),
    )
    eval_record["id"] = str(record.get("id", index))
    return [eval_record]


def main() -> None:
    args = parse_args()
    items: List[Dict[str, Any]] = []
    for index, record in enumerate(read_json_or_jsonl(args.input)):
        items.extend(
            _normalize_record(
                record=dict(record),
                index=index,
                image_root=args.image_root,
                scene=args.scene,
                task_override=args.task,
                source_name=args.source_name or os.path.basename(args.input),
            )
        )
    write_jsonl(items, args.output)
    print(f"Wrote {len(items)} eval records to {args.output}")


if __name__ == "__main__":
    main()
