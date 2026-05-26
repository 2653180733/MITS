"""Utilities for MITS model evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from mits_pipeline.mits_io import DEFAULT_TASK_ORDER, infer_task_from_text, join_path


EVAL_TASK_ORDER = list(DEFAULT_TASK_ORDER)
AUTO_SCORE_TASKS = {"recognition", "counting", "localization"}
JUDGE_TASKS = {"background", "reasoning"}

_NUMBER_WORDS = {
    "zero": 0,
    "none": 0,
    "no": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_NO_OBJECT_PATTERNS = [
    r"\bthere (?:is|are) no\b",
    r"\bno [a-z0-9_\- ]+ (?:in|on|visible|present)\b",
    r"\bnot visible\b",
    r"\bnone (?:visible|present)\b",
    r"\bnot present\b",
    r"\bno object\b",
    r"\bno target\b",
    r"\b(?:does|do|did) not contain\b",
    r"\bdoesn't contain\b",
]


def read_json_or_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield records from a JSONL file, a JSON array, or one JSON object."""
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item
        return

    with open(path, "r", encoding="utf-8-sig") as handle:
        prefix = _first_non_whitespace_char(handle)
        handle.seek(0)
        if prefix == "[":
            yield from _iter_json_array(handle)
            return
        if prefix == "{":
            text = handle.read().strip()
            if "\n" not in text:
                item = json.loads(text)
                if isinstance(item, dict):
                    yield item
                return
            handle.seek(0)
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item


def _first_non_whitespace_char(handle) -> str:
    while True:
        char = handle.read(1)
        if not char:
            return ""
        if not char.isspace():
            return char


def _iter_json_array(handle) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False

    while True:
        if not finished:
            chunk = handle.read(65536)
            if chunk:
                buffer += chunk
            else:
                finished = True

        index = 0
        if not started:
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index >= len(buffer):
                if finished:
                    return
                buffer = ""
                continue
            if buffer[index] != "[":
                raise ValueError("Expected a JSON array")
            started = True
            index += 1

        while True:
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if index >= len(buffer):
                break
            if buffer[index] == ",":
                index += 1
                continue
            if buffer[index] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                yield item
            index = end

        buffer = buffer[index:]
        if finished and not buffer.strip():
            return


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str) -> int:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("<image>", " ")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def normalize_answer_text(text: Any) -> str:
    value = normalize_text(text)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    text = re.sub(r"/+", "/", text)
    return text.rstrip("/").lower()


def canonical_image_key(path: Any, image_root: Optional[str] = None) -> str:
    """Return a stable key for relative or absolute MITS image paths."""
    text = normalize_key(path)
    if not text:
        return ""

    if image_root:
        root = normalize_key(image_root)
        if text.startswith(root + "/"):
            text = text[len(root) + 1 :]

    marker = "/data/dataset/images/"
    if marker in text:
        text = text.split(marker, 1)[1]

    parts = [part for part in text.split("/") if part]
    for index, part in enumerate(parts):
        if re.match(r"^v\d", part):
            return "/".join(parts[index:])
    return text


def collect_identity_keys(record: Mapping[str, Any], image_root: Optional[str] = None) -> Dict[str, List[str]]:
    original_ids: List[str] = []
    image_keys: List[str] = []

    for key in ("original_id",):
        value = record.get(key)
        if value is not None and value != "":
            original_ids.append(normalize_key(value))

    meta = record.get("meta")
    if isinstance(meta, Mapping):
        for key in ("original_id",):
            value = meta.get(key)
            if value is not None and value != "":
                original_ids.append(normalize_key(value))
        for key in ("image", "image_rel"):
            value = meta.get(key)
            image_key = canonical_image_key(value, image_root=image_root)
            if image_key:
                image_keys.append(image_key)

    for key in ("image", "image_rel"):
        image_key = canonical_image_key(record.get(key), image_root=image_root)
        if image_key:
            image_keys.append(image_key)

    images = record.get("images")
    if isinstance(images, str):
        images = [images]
    if isinstance(images, Sequence):
        for image in images:
            image_key = canonical_image_key(image, image_root=image_root)
            if image_key:
                image_keys.append(image_key)

    return {
        "original_ids": sorted(set(original_ids)),
        "image_keys": sorted(set(image_keys)),
    }


def infer_answer_type(task: str, question: str, answer: str) -> str:
    task = (task or infer_task_from_text("", question)).lower()
    if task == "counting":
        return "count"
    if task == "localization":
        return "bbox"
    if extract_yes_no(answer) is not None:
        return "yesno"
    if parse_bboxes(answer):
        return "bbox"
    return "text"


def extract_yes_no(text: Any) -> Optional[str]:
    value = normalize_text(text)
    if not value:
        return None
    if re.match(r"^(yes|yeah|yep|true)\b", value):
        return "yes"
    if re.match(r"^(no|nope|false)\b", value):
        return "no"
    if any(re.search(pattern, value) for pattern in _NO_OBJECT_PATTERNS):
        return "no"
    if re.search(r"\bthere (?:is|are) (?:a|an|some|one|two|three|four|five|\d+)\b", value):
        return "yes"
    return None


def is_no_object_answer(text: Any) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    if extract_yes_no(value) == "no":
        return True
    return any(re.search(pattern, value) for pattern in _NO_OBJECT_PATTERNS)


def extract_count(text: Any) -> Optional[int]:
    value = normalize_text(text)
    if not value:
        return None
    if is_no_object_answer(value):
        return 0
    match = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?", value)
    if match:
        return int(round(float(match.group(0))))
    for word, number in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", value):
            return number
    return None


def parse_bboxes(text: Any) -> List[List[float]]:
    """Parse common bbox formats into [x1, y1, x2, y2] boxes."""
    if text is None:
        return []
    value = str(text)
    boxes: List[List[float]] = []

    # Qwen-style pairs: (x1,y1),(x2,y2)
    pair_pattern = re.compile(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    )
    for match in pair_pattern.finditer(value):
        boxes.append(_sanitize_box([float(match.group(i)) for i in range(1, 5)]))

    # JSON-like bracket groups: [x1, y1, x2, y2]
    bracket_pattern = re.compile(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
    )
    for match in bracket_pattern.finditer(value):
        box = _sanitize_box([float(match.group(i)) for i in range(1, 5)])
        if box not in boxes:
            boxes.append(box)

    return boxes


def _sanitize_box(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    return [left, top, right, bottom]


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _sanitize_box(box_a)
    bx1, by1, bx2, by2 = _sanitize_box(box_b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_bbox_score(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> Tuple[float, Dict[str, Any]]:
    """Match boxes greedily by IoU and return mean IoU over ground-truth boxes."""
    if not gt_boxes:
        return (1.0 if not pred_boxes else 0.0), {
            "gt_boxes": 0,
            "pred_boxes": len(pred_boxes),
            "matched": 0,
            "ious": [],
            "iou50": 1.0 if not pred_boxes else 0.0,
        }

    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred in enumerate(pred_boxes):
        for gt_index, gt in enumerate(gt_boxes):
            candidates.append((bbox_iou(pred, gt), pred_index, gt_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_pred = set()
    used_gt = set()
    matched_ious: List[float] = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matched_ious.append(iou)
        if len(used_gt) == len(gt_boxes):
            break

    score = sum(matched_ious) / max(1, len(gt_boxes))
    iou50 = sum(1 for iou in matched_ious if iou >= 0.5) / max(1, len(gt_boxes))
    return score, {
        "gt_boxes": len(gt_boxes),
        "pred_boxes": len(pred_boxes),
        "matched": len(matched_ious),
        "ious": matched_ious,
        "iou50": iou50,
    }


def score_prediction(record: Mapping[str, Any], judge_score: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    task = str(record.get("task") or "").lower() or infer_task_from_text(
        str(record.get("field") or ""),
        str(record.get("question") or ""),
    )
    answer = str(record.get("answer") or "")
    prediction = str(record.get("prediction") or "")
    answer_type = str(record.get("answer_type") or infer_answer_type(task, str(record.get("question") or ""), answer))

    result: Dict[str, Any] = {
        "score": None,
        "score_source": "none",
        "needs_judge": False,
        "score_detail": {},
    }

    if judge_score is not None:
        result["score"] = _clamp_score(judge_score.get("score"))
        result["score_source"] = judge_score.get("source", "judge")
        result["needs_judge"] = False
        result["score_detail"] = {key: value for key, value in judge_score.items() if key not in {"id", "score"}}
        return result

    if task in JUDGE_TASKS:
        result["needs_judge"] = True
        return result

    if task == "counting" or answer_type == "count":
        gt_count = extract_count(answer)
        pred_count = extract_count(prediction)
        result["score_detail"] = {"gt_count": gt_count, "pred_count": pred_count}
        if gt_count is None or pred_count is None:
            return result
        result["score"] = max(0.0, 1.0 - abs(pred_count - gt_count) / max(gt_count, 1))
        result["score_source"] = "auto_count"
        return result

    if task == "localization" or answer_type == "bbox":
        gt_boxes = parse_bboxes(answer)
        pred_boxes = parse_bboxes(prediction)
        if not gt_boxes and is_no_object_answer(answer):
            result["score"] = 1.0 if not pred_boxes and is_no_object_answer(prediction) else 0.0
            result["score_source"] = "auto_no_object_localization"
            result["score_detail"] = {
                "gt_boxes": 0,
                "pred_boxes": len(pred_boxes),
                "gt_no_object": True,
                "pred_no_object": is_no_object_answer(prediction),
            }
            return result
        if not gt_boxes:
            result["score_detail"] = {"error": "no_ground_truth_bbox"}
            return result
        score, detail = greedy_bbox_score(pred_boxes, gt_boxes)
        result["score"] = score
        result["score_source"] = "auto_bbox_iou"
        result["score_detail"] = detail
        return result

    gt_yes_no = extract_yes_no(answer)
    pred_yes_no = extract_yes_no(prediction)
    if gt_yes_no is not None:
        result["score"] = 1.0 if pred_yes_no == gt_yes_no else 0.0
        result["score_source"] = "auto_yesno"
        result["score_detail"] = {"gt_yes_no": gt_yes_no, "pred_yes_no": pred_yes_no}
        return result

    gt_text = normalize_answer_text(answer)
    pred_text = normalize_answer_text(prediction)
    result["score"] = 1.0 if pred_text == gt_text and gt_text else 0.0
    result["score_source"] = "auto_exact_match"
    result["score_detail"] = {"gt_text": gt_text, "pred_text": pred_text}
    return result


def _clamp_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return max(0.0, min(1.0, score))


def load_judge_scores(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    scores: Dict[str, Dict[str, Any]] = {}
    for item in read_json_or_jsonl(path):
        sample_id = item.get("id")
        if sample_id is None:
            sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        scores[str(sample_id)] = {
            "id": item.get("id", sample_id),
            "score": item.get("score", item.get("judge_score")),
            "source": item.get("source", item.get("judge_source", "judge")),
            "raw": dict(item),
        }
    return scores


def summarize_scores(records: Sequence[Mapping[str, Any]], external: bool = False) -> Dict[str, Any]:
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        task = str(record.get("task") or "unknown").lower()
        by_task[task].append(record)

    task_summary: Dict[str, Dict[str, Any]] = {}
    for task in EVAL_TASK_ORDER:
        task_records = by_task.get(task, [])
        scored = [
            float(record["score"])
            for record in task_records
            if record.get("score") is not None
        ]
        task_summary[task] = {
            "n": len(task_records),
            "scored_n": len(scored),
            "score": sum(scored) / len(scored) if scored else None,
            "complete": len(task_records) > 0 and len(scored) == len(task_records),
        }

    auto_tasks = [task for task in ("recognition", "counting", "localization") if task_summary[task]["scored_n"] > 0]
    automated_avg = (
        sum(float(task_summary[task]["score"]) for task in auto_tasks) / len(auto_tasks)
        if auto_tasks
        else None
    )

    required_complete = all(task_summary[task]["score"] is not None for task in EVAL_TASK_ORDER)
    mits_avg = (
        sum(float(task_summary[task]["score"]) for task in EVAL_TASK_ORDER) / len(EVAL_TASK_ORDER)
        if required_complete and not external
        else None
    )

    all_scored = [float(record["score"]) for record in records if record.get("score") is not None]
    external_avg = sum(all_scored) / len(all_scored) if all_scored else None

    return {
        "evaluation_type": "external" if external else "mits",
        "total": len(records),
        "scored_total": len(all_scored),
        "needs_judge_total": sum(1 for record in records if record.get("needs_judge")),
        "tasks": task_summary,
        "automated_avg": automated_avg,
        "external_avg": external_avg if external else None,
        "mits_avg": mits_avg,
        "mits_avg_complete": mits_avg is not None,
    }


def write_summary_csv(summary: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "n", "scored_n", "score", "complete"],
        )
        writer.writeheader()
        for task in EVAL_TASK_ORDER:
            row = dict(summary.get("tasks", {}).get(task, {}))
            row["task"] = task
            writer.writerow(row)


def resolve_image_path(image: str, image_root: Optional[str] = None) -> str:
    if not image:
        return ""
    if os.path.isabs(image):
        return image
    if image_root:
        return join_path(image_root, image)
    return image
