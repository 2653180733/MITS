"""Task-aware and rarity-aware subset selection for MITS."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ScoreMap = Mapping[int, float]


def load_importance_scores(path: str) -> Dict[int, float]:
    """Load ScalSelect CUR scores from JSONL.

    Expected keys include ``sample_id`` and ``importance``. The loader is
    permissive to make it usable with lightly modified score files.
    """
    scores: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            raw_id = item.get("sample_id", item.get("id"))
            raw_score = item.get("importance", item.get("score", item.get("cur_score", 0.0)))
            try:
                sample_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            scores[sample_id] = float(raw_score)
    return scores


def _inverse_frequency_bonus(tags: Sequence[str], counts: Counter, total: int) -> float:
    if not tags or total <= 0:
        return 0.0
    return sum(math.log1p(total / max(1, counts[tag])) for tag in tags) / len(tags)


def enrich_records_with_scores(
    records: Sequence[Dict[str, Any]],
    cur_scores: Optional[ScoreMap] = None,
    lambda_task: float = 0.2,
    lambda_rare: float = 0.1,
) -> List[Dict[str, Any]]:
    """Attach final selection scores to index records."""
    cur_scores = cur_scores or {}
    task_counts: Counter = Counter()
    rare_counts: Counter = Counter()

    for record in records:
        task_counts.update(record.get("task_tags", []))
        rare_counts.update(record.get("rare_tags", []))

    total = len(records)
    enriched: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        sample_id = int(record.get("id", position))
        cur_score = float(cur_scores.get(sample_id, 0.0))
        task_bonus = _inverse_frequency_bonus(record.get("task_tags", []), task_counts, total)
        rarity_bonus = _inverse_frequency_bonus(record.get("rare_tags", []), rare_counts, total)
        final_score = cur_score + lambda_task * task_bonus + lambda_rare * rarity_bonus

        item = dict(record)
        item["cur_score"] = cur_score
        item["task_bonus"] = task_bonus
        item["rarity_bonus"] = rarity_bonus
        item["final_score"] = final_score
        enriched.append(item)

    return enriched


def resolve_num_selected(
    total: int,
    ratio: Optional[float] = None,
    num_selected: Optional[int] = None,
) -> int:
    if num_selected is not None:
        return max(1, min(total, num_selected))
    if ratio is None:
        raise ValueError("Either ratio or num_selected must be provided.")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    if ratio > 1:
        ratio = ratio / 100.0
    return max(1, min(total, int(round(total * ratio))))


def select_records(
    records: Sequence[Dict[str, Any]],
    ratio: Optional[float] = None,
    num_selected: Optional[int] = None,
    min_task_fraction: float = 0.05,
) -> List[Dict[str, Any]]:
    """Select records by final score while reserving a small per-task quota."""
    target = resolve_num_selected(len(records), ratio=ratio, num_selected=num_selected)
    ranked = sorted(records, key=lambda item: item.get("final_score", 0.0), reverse=True)

    all_tasks = sorted({tag for record in records for tag in record.get("task_tags", [])})
    quota = max(1, int(target * min_task_fraction)) if all_tasks else 0

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    task_fill = Counter()

    for task in all_tasks:
        for record in ranked:
            sample_id = record.get("id")
            if sample_id in selected_ids:
                continue
            if task not in record.get("task_tags", []):
                continue
            selected.append(record)
            selected_ids.add(sample_id)
            task_fill.update(record.get("task_tags", []))
            if task_fill[task] >= quota or len(selected) >= target:
                break
        if len(selected) >= target:
            break

    for record in ranked:
        if len(selected) >= target:
            break
        sample_id = record.get("id")
        if sample_id in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(sample_id)

    return selected[:target]


def _group_value(record: Mapping[str, Any], group_field: str, fallback: str) -> str:
    value = record.get(group_field)
    if value is None or value == "":
        return fallback
    return str(value)


def select_records_by_group(
    records: Sequence[Dict[str, Any]],
    group_field: str = "scene",
    ratio: Optional[float] = None,
    num_selected: Optional[int] = None,
    min_task_fraction: float = 0.05,
    min_per_group: int = 0,
    max_per_group: Optional[int] = None,
    fallback_group: str = "unknown",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select records independently inside each group.

    This preserves low-frequency traffic scenes by assigning each scene its own
    ratio-based quota before merging the selected records.
    """
    if not group_field:
        return select_records(
            records=records,
            ratio=ratio,
            num_selected=num_selected,
            min_task_fraction=min_task_fraction,
        ), []

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_group_value(record, group_field, fallback_group)].append(record)

    selected: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for group_name in sorted(groups):
        group_records = groups[group_name]
        group_num_selected = None
        group_ratio = ratio

        if num_selected is not None:
            group_num_selected = resolve_num_selected(
                total=len(group_records),
                ratio=None,
                num_selected=num_selected,
            )

        target = resolve_num_selected(
            total=len(group_records),
            ratio=group_ratio,
            num_selected=group_num_selected,
        )
        if min_per_group > 0:
            target = max(target, min_per_group)
        if max_per_group is not None:
            target = min(target, max_per_group)
        target = max(1, min(len(group_records), target))

        group_selected = select_records(
            records=group_records,
            num_selected=target,
            min_task_fraction=min_task_fraction,
        )
        selected.extend(group_selected)
        summaries.append(
            {
                "group": group_name,
                "total": len(group_records),
                "selected": len(group_selected),
                "ratio": len(group_selected) / len(group_records) if group_records else 0.0,
            }
        )

    return selected, summaries


def write_selected_index(records: Iterable[Dict[str, Any]], output_path: str) -> int:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
