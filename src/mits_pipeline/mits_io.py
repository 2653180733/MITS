"""MITS dataset indexing and ShareGPT conversion utilities."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_QA_FIELD_ORDER = [
    "basecaption",
    "optimizedcaption",
    "foregroundqa",
    "backgroundqa",
    "reasoningqa",
    "eventreasoningqa",
    "llmqa",
]

DEFAULT_TASK_ORDER = [
    "background",
    "recognition",
    "counting",
    "localization",
    "reasoning",
]

RARE_EVENT_KEYS = [
    "accident",
    "firesmoke",
    "spill",
    "construction",
    "weather",
    "jam",
]

UNKNOWN_SCENE = "unknown"
UNKNOWN_SHARD = "unknown"

LOW_INFORMATION_ANSWERS = {
    "yes",
    "no",
    "unknown",
    "n/a",
    "na",
    "none",
    "not sure",
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
}

LOW_INFORMATION_PHRASES = [
    "cannot be determined from the given information",
    "not enough information to determine",
    "unable to determine",
]


def join_path(*parts: str) -> str:
    """Join paths with Windows-compatible semantics."""
    return os.path.normpath(os.path.join(*parts))


def path_parts(path: str) -> List[str]:
    """Split a path into normalized non-empty parts across Windows/Linux paths."""
    normalized = str(path).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def is_mits_input_dir(name: str) -> bool:
    """Return whether a VQA directory stores integrated training inputs."""
    lowered = str(name).lower()
    return lowered == "integratedinput" or lowered.startswith("integratedinput_")


def scene_from_image_path(image_path: str) -> str:
    """Infer the primary traffic scene from a relative or absolute MITS image path."""
    parts = path_parts(image_path)
    if not parts:
        return UNKNOWN_SCENE

    for index, part in enumerate(parts):
        if "_train_" in part and index + 1 < len(parts):
            return parts[index + 1]

    if len(parts) >= 3 and parts[-2] == "images":
        return parts[-3]

    if "images" in parts:
        image_root_index = len(parts) - 1 - parts[::-1].index("images")
        if image_root_index + 2 < len(parts):
            return parts[image_root_index + 2]

    if len(parts) >= 2:
        return parts[-2]
    return UNKNOWN_SCENE


def shard_from_image_path(image_path: str) -> str:
    """Infer the source shard from a relative or absolute MITS image path."""
    parts = path_parts(image_path)
    if not parts:
        return UNKNOWN_SHARD

    for part in parts:
        if "_train_" in part:
            return part

    if len(parts) >= 4 and parts[-2] == "images":
        return parts[-4]

    if "images" in parts:
        image_root_index = len(parts) - 1 - parts[::-1].index("images")
        if image_root_index + 1 < len(parts):
            return parts[image_root_index + 1]

    return parts[0]


def dataset_subdirs(dataset_root: str) -> Tuple[str, str]:
    """Return image and VQA roots from a MITS dataset root."""
    return join_path(dataset_root, "images"), join_path(dataset_root, "vqas")


def scene_from_mits_path(json_path: Optional[str], item: Optional[Dict[str, Any]] = None) -> str:
    """Infer the traffic scene directory for a MITS sample.

    Expected VQA layouts:
    ``vqas/<shard>/<scene>/integratedinput/<sample>.json`` and
    ``vqas/<shard>/<scene>/integratedinput_<subscene>/<sample>.json``.
    """
    if json_path:
        parts = path_parts(json_path)
        if len(parts) >= 3 and is_mits_input_dir(parts[-2]):
            return parts[-3]

    image = str((item or {}).get("image", ""))
    if image:
        scene = scene_from_image_path(image)
        if scene != UNKNOWN_SCENE:
            return scene

    original_id = str((item or {}).get("id", ""))
    if original_id:
        parts = path_parts(original_id)
        if len(parts) >= 2:
            return parts[1]

    return UNKNOWN_SCENE


def shard_from_mits_path(json_path: Optional[str], item: Optional[Dict[str, Any]] = None) -> str:
    """Infer the source shard directory for a MITS sample."""
    if json_path:
        parts = path_parts(json_path)
        if len(parts) >= 4 and is_mits_input_dir(parts[-2]):
            return parts[-4]

    image = str((item or {}).get("image", ""))
    if image:
        shard = shard_from_image_path(image)
        if shard != UNKNOWN_SHARD:
            return shard

    original_id = str((item or {}).get("id", ""))
    if original_id:
        parts = path_parts(original_id)
        if parts:
            return parts[0]

    return UNKNOWN_SHARD


def require_safe_scan(limit: Optional[int], allow_full_scan: bool) -> None:
    """Prevent accidental full traversal of the large MITS dataset."""
    if limit is None and not allow_full_scan:
        raise ValueError(
            "Refusing to scan without a limit. Pass --limit N for a bounded "
            "scan or --allow-full-scan for an intentional full streaming pass."
        )


def iter_mits_json_paths(
    vqa_root: str,
    limit: Optional[int] = 100,
    allow_full_scan: bool = False,
    shards: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> Iterator[str]:
    """Yield MITS JSON sample paths without loading images.

    The default limit is 100 to keep exploratory runs bounded.
    """
    require_safe_scan(limit, allow_full_scan)

    shard_filter = set(shards) if shards else None
    category_filter = set(categories) if categories else None
    emitted = 0

    for shard in sorted(os.listdir(vqa_root)):
        if shard_filter and shard not in shard_filter:
            continue

        shard_dir = join_path(vqa_root, shard)
        if not os.path.isdir(shard_dir):
            continue

        for category in sorted(os.listdir(shard_dir)):
            if category_filter and category not in category_filter:
                continue

            category_dir = join_path(shard_dir, category)
            if not os.path.isdir(category_dir):
                continue

            for input_name in sorted(os.listdir(category_dir)):
                if not is_mits_input_dir(input_name):
                    continue

                input_dir = join_path(category_dir, input_name)
                if not os.path.isdir(input_dir):
                    continue

                for filename in sorted(os.listdir(input_dir)):
                    if not filename.endswith(".json"):
                        continue

                    yield join_path(input_dir, filename)
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def iter_mits_items(
    vqa_root: str,
    limit: Optional[int] = 100,
    allow_full_scan: bool = False,
    shards: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(json_path, item)`` pairs one sample at a time."""
    for path in iter_mits_json_paths(
        vqa_root=vqa_root,
        limit=limit,
        allow_full_scan=allow_full_scan,
        shards=shards,
        categories=categories,
    ):
        yield path, load_json(path)


def is_turn_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) for item in value)


def is_excluded_qa_field(field: str) -> bool:
    """Return whether a field is metadata rather than training QA."""
    lowered = field.lower()
    if lowered in {"baselabel", "categorylabel"}:
        return True
    if lowered.endswith("_task_type") or "tasktype" in lowered:
        return True
    return False


def ordered_qa_fields(item: Dict[str, Any]) -> List[str]:
    """Return known QA fields first, followed by other turn-like fields."""
    fields = [
        field
        for field in DEFAULT_QA_FIELD_ORDER
        if not is_excluded_qa_field(field) and is_turn_list(item.get(field))
    ]
    known = set(fields)

    for field, value in item.items():
        if field in known:
            continue
        if is_excluded_qa_field(field):
            continue
        if is_turn_list(value) and _has_text_turn(value):
            fields.append(field)

    return fields


def _has_text_turn(turns: Sequence[Dict[str, Any]]) -> bool:
    return any("value" in turn or "Question" in turn or "Answer" in turn for turn in turns)


def _turn_value(turn: Dict[str, Any], key: str = "value") -> str:
    value = turn.get(key, "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _role_from_turn(turn: Dict[str, Any], fallback: str) -> str:
    source = str(turn.get("from", turn.get("role", ""))).lower()
    if "question" in source or source in {"human", "user"}:
        return "user"
    if "answer" in source or source in {"assistant", "gpt"}:
        return "assistant"
    if "qwen" in source or "gpt" in source or "deepseek" in source:
        return "assistant"
    return fallback


def iter_message_pairs(item: Dict[str, Any]) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(field, question, answer)`` pairs from a MITS item."""
    for field in ordered_qa_fields(item):
        turns = item.get(field, [])
        if not isinstance(turns, list):
            continue

        # Handle records such as {"Question": "...", "Answer": "..."}.
        for turn in turns:
            if isinstance(turn, dict) and "Question" in turn and "Answer" in turn:
                yield field, _turn_value(turn, "Question"), _turn_value(turn, "Answer")

        # Handle alternating MITS turns such as question/answer dictionaries.
        i = 0
        while i + 1 < len(turns):
            first = turns[i]
            second = turns[i + 1]
            if not isinstance(first, dict) or not isinstance(second, dict):
                i += 1
                continue

            first_role = _role_from_turn(first, "user")
            second_role = _role_from_turn(second, "assistant")
            first_text = _turn_value(first).strip()
            second_text = _turn_value(second).strip()

            if first_role == "user" and second_role == "assistant" and first_text and second_text:
                yield field, first_text, second_text
                i += 2
            else:
                i += 1


def infer_task_from_text(field: str, question: str) -> str:
    text = f"{field} {question}".lower()
    if any(key in text for key in ["how many", "number of", "count", "quantity"]):
        return "counting"
    if any(key in text for key in ["where", "location", "locate", "position", "bounding box"]):
        return "localization"
    if any(key in text for key in ["why", "reason", "infer", "likely", "because", "cause"]):
        return "reasoning"
    if any(key in text for key in ["background", "road type", "weather", "time type", "traffic flow", "lighting condition"]):
        return "background"
    if "caption" in text:
        return "background"
    return "recognition"


def extract_task_tags(item: Dict[str, Any]) -> List[str]:
    tags = set()
    for field, question, _answer in iter_message_pairs(item):
        tags.add(infer_task_from_text(field, question))
    return sorted(tags)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("<image>", " ")).strip()


def _normalize_for_dedupe(text: str) -> str:
    normalized = _normalize_text(text).lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\b(can you|could you|please|tell me|in the image|in this image|in the picture)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_low_information_answer(answer: str) -> bool:
    normalized = _normalize_for_dedupe(answer)
    if normalized in LOW_INFORMATION_ANSWERS:
        return True
    return any(phrase in normalized for phrase in LOW_INFORMATION_PHRASES)


def _passes_quality_rules(
    question: str,
    answer: str,
    min_question_chars: int = 8,
    min_answer_chars: int = 3,
    max_question_chars: int = 512,
    max_answer_chars: int = 1024,
) -> bool:
    question = _normalize_text(question)
    answer = _normalize_text(answer)
    if not question or not answer:
        return False
    if len(question) < min_question_chars or len(answer) < min_answer_chars:
        return False
    if len(question) > max_question_chars or len(answer) > max_answer_chars:
        return False
    if _normalize_for_dedupe(question) == _normalize_for_dedupe(answer):
        return False
    if _is_low_information_answer(answer):
        return False
    return True


def filter_quality_pairs(
    pairs: Sequence[Tuple[str, str, str]],
    max_pairs_per_sample: Optional[int] = None,
    qa_filter: str = "balanced",
    max_pairs_per_task: int = 8,
    min_question_chars: int = 8,
    min_answer_chars: int = 3,
    max_question_chars: int = 512,
    max_answer_chars: int = 1024,
) -> List[Tuple[str, str, str]]:
    """Filter and optionally balance QA pairs for one image sample."""
    qa_filter = qa_filter.lower()
    if qa_filter not in {"none", "quality", "balanced"}:
        raise ValueError("qa_filter must be one of: none, quality, balanced.")

    if qa_filter == "none":
        kept = list(pairs)
        return kept if max_pairs_per_sample is None else kept[:max_pairs_per_sample]

    candidates: List[Tuple[str, str, str, str]] = []
    seen_pairs = set()
    seen_same_answer_questions = set()
    for field, question, answer in pairs:
        question = _normalize_text(question)
        answer = _normalize_text(answer)
        if not _passes_quality_rules(
            question=question,
            answer=answer,
            min_question_chars=min_question_chars,
            min_answer_chars=min_answer_chars,
            max_question_chars=max_question_chars,
            max_answer_chars=max_answer_chars,
        ):
            continue

        pair_key = (_normalize_for_dedupe(question), _normalize_for_dedupe(answer))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        task = infer_task_from_text(field, question)
        same_answer_question_key = (task, pair_key[0], pair_key[1])
        if same_answer_question_key in seen_same_answer_questions:
            continue
        seen_same_answer_questions.add(same_answer_question_key)
        candidates.append((field, question, answer, task))

    if qa_filter == "quality" or max_pairs_per_sample is None:
        trimmed = candidates if max_pairs_per_sample is None else candidates[:max_pairs_per_sample]
        return [(field, question, answer) for field, question, answer, _task in trimmed]

    selected: List[Tuple[str, str, str, str]] = []
    selected_ids = set()
    task_order = list(DEFAULT_TASK_ORDER)
    task_order.extend(sorted({candidate[3] for candidate in candidates if candidate[3] not in task_order}))
    task_cap = max_pairs_per_task if max_pairs_per_task > 0 else max_pairs_per_sample
    grouped_indices = {
        task: [index for index, candidate in enumerate(candidates) if candidate[3] == task]
        for task in task_order
    }
    group_offsets = {task: 0 for task in task_order}
    task_counts = {task: 0 for task in task_order}

    while len(selected) < max_pairs_per_sample:
        progressed = False
        for task in task_order:
            if len(selected) >= max_pairs_per_sample:
                break
            if task_counts[task] >= task_cap:
                continue
            indices = grouped_indices[task]
            while group_offsets[task] < len(indices) and indices[group_offsets[task]] in selected_ids:
                group_offsets[task] += 1
            if group_offsets[task] >= len(indices):
                continue
            index = indices[group_offsets[task]]
            selected.append(candidates[index])
            selected_ids.add(index)
            group_offsets[task] += 1
            task_counts[task] += 1
            progressed = True
        if not progressed:
            break

    for index, candidate in enumerate(candidates):
        if len(selected) >= max_pairs_per_sample:
            break
        if index in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(index)

    return [(field, question, answer) for field, question, answer, _task in selected]


def _walk_label_values(value: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                yield from _walk_label_values(child)
            else:
                yield str(key), child
    elif isinstance(value, list):
        for child in value:
            yield from _walk_label_values(child)


def positive_label_keys(item: Dict[str, Any]) -> List[str]:
    keys = set()
    for key, value in _walk_label_values(item.get("categorylabel", [])):
        if isinstance(value, (int, float)) and value > 0:
            keys.add(key)
    return sorted(keys)


def extract_rare_tags(item: Dict[str, Any], json_path: Optional[str] = None) -> List[str]:
    positives = set(positive_label_keys(item))
    rare = {key for key in RARE_EVENT_KEYS if key in positives}

    baselabel = item.get("baselabel", [])
    for _key, value in _walk_label_values(baselabel):
        if isinstance(value, str):
            lowered = value.lower()
            if "night" in lowered or "low" in lowered:
                rare.add("low_light")
            if lowered not in {"normal", "daytime", "non-highway roads"} and "weather" in lowered:
                rare.add("weather")

    if json_path:
        lowered_path = json_path.lower()
        for key in RARE_EVENT_KEYS:
            if f"{os.sep}{key}{os.sep}" in lowered_path:
                rare.add(key)

    return sorted(rare)


def mits_item_to_sharegpt(
    item: Dict[str, Any],
    image_root: str,
    sample_id: Optional[int] = None,
    json_path: Optional[str] = None,
    max_pairs_per_sample: Optional[int] = None,
    qa_filter: str = "balanced",
    max_pairs_per_task: int = 8,
    min_question_chars: int = 8,
    min_answer_chars: int = 3,
    max_question_chars: int = 512,
    max_answer_chars: int = 1024,
) -> Dict[str, Any]:
    """Convert one MITS item into ShareGPT format."""
    messages: List[Dict[str, str]] = []
    first_user = True
    scene = scene_from_mits_path(json_path, item=item)
    shard = shard_from_mits_path(json_path, item=item)
    all_pairs = list(iter_message_pairs(item))
    selected_pairs = filter_quality_pairs(
        pairs=all_pairs,
        max_pairs_per_sample=max_pairs_per_sample,
        qa_filter=qa_filter,
        max_pairs_per_task=max_pairs_per_task,
        min_question_chars=min_question_chars,
        min_answer_chars=min_answer_chars,
        max_question_chars=max_question_chars,
        max_answer_chars=max_answer_chars,
    )

    for _field, question, answer in selected_pairs:
        if first_user and "<image>" not in question:
            question = "<image>\n" + question
        first_user = False

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    image_rel = item.get("image", "")
    return {
        "id": sample_id if sample_id is not None else item.get("id"),
        "messages": messages,
        "images": [join_path(image_root, image_rel)] if image_rel else [],
        "meta": {
            "original_id": item.get("id"),
            "json_path": json_path,
            "image": image_rel,
            "scene": scene,
            "shard": shard,
            "task_tags": extract_task_tags(item),
            "rare_tags": extract_rare_tags(item, json_path=json_path),
            "positive_labels": positive_label_keys(item),
            "qa_filter": qa_filter,
            "qa_pairs_total": len(all_pairs),
            "qa_pairs_kept": len(selected_pairs),
        },
    }


def build_index_record(json_path: str, item: Dict[str, Any], numeric_id: int) -> Dict[str, Any]:
    return {
        "id": numeric_id,
        "original_id": item.get("id"),
        "json_path": json_path,
        "image": item.get("image"),
        "scene": scene_from_mits_path(json_path, item=item),
        "shard": shard_from_mits_path(json_path, item=item),
        "task_tags": extract_task_tags(item),
        "rare_tags": extract_rare_tags(item, json_path=json_path),
        "positive_labels": positive_label_keys(item),
    }


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> int:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(input_path: str) -> Iterator[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_sharegpt_json(samples: Iterable[Dict[str, Any]], output_path: str) -> int:
    """Write a ShareGPT JSON array without holding the whole output in memory."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("[\n")
        for sample in samples:
            if not sample.get("messages"):
                continue
            if count:
                handle.write(",\n")
            handle.write(json.dumps(sample, ensure_ascii=False, indent=2))
            count += 1
        handle.write("\n]\n")
    return count
